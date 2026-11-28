"""Non-overlapping two-layer CNN planar SVG reconstruction with full RGBA transparency support."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import heapq
import math
import os
import re

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import label

from .boundary_corridor import SourceCorridor
from .color import srgb_to_oklab
from .planar_partition import build_planar_partition
from .semantic_parts import (
    PART_BG,
    PART_CLOTH,
    PART_EYES_MOUTH,
    PART_FACE_SKIN,
    PART_HAIR,
    PART_HEADWEAR,
    PART_NAMES,
    PART_NECK_HAND,
    PART_OBJECTS,
    render_part_map_rgb,
    segment_semantic_parts,
)
from .shared_regions import trace_regions
from .structural_curves import fit_compact_outline, refit_sketch_chain, smooth_outline_junctions
from .svg_layers import boundary_then_fill


def _prepare_alpha_source(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract an alpha-safe RGB canvas and optional 1x alpha channel `(H, W) uint8`.

    When an input image has transparent regions (`alpha < 250`):
      - Bleeds nearest visible pixel colors outward across the first ~6px outside the
        silhouette so SR / MeanShift / CNN filters never pull black or white edge fringes
        into semi-transparent anti-aliased strokes.
      - Smoothly transitions the far exterior (> 6px) to a uniform contrasting backdrop
        so line detectors see clean outer contours without phantom background artifacts.
      - Returns `alpha_1x` so final planar labeling masks transparent pixels back to `-1`
        (zero background `<path>` emitted in SVG, full alpha preserved in PNG preview).
    """
    has_alpha = ("A" in img.getbands()) or ("transparency" in img.info)
    if not has_alpha:
        return np.asarray(img.convert("RGB")), None

    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[:, :, 3]
    if np.all(alpha >= 250):
        return rgba[:, :, :3].copy(), None

    rgb_raw = rgba[:, :, :3]
    visible = alpha >= 128
    if not visible.any():
        return np.full_like(rgb_raw, 255), alpha

    # Bleed nearest visible RGB outward across the transparent exterior (eliminates black/white fringe)
    dist_out, (iy, ix) = distance_transform_edt(~visible, return_indices=True)
    bled = rgb_raw[iy, ix].astype(np.float32)

    # Pick a clean uniform backdrop for far exterior (> 6px) contrasting with silhouette border
    border_ring = (~visible) & (dist_out <= 2.5)
    border_lum = float(bled[border_ring].mean()) if border_ring.any() else 128.0
    backdrop_val = 248.0 if border_lum < 210.0 else 32.0

    # Smooth transition from edge-bled color (0..4px) to clean uniform backdrop (6px+)
    blend = np.clip((dist_out - 3.0) / 4.0, 0.0, 1.0)[:, :, None]
    comp = np.where(
        visible[:, :, None],
        rgb_raw.astype(np.float32),
        bled * (1.0 - blend) + backdrop_val * blend,
    )
    return np.clip(np.rint(comp), 0, 255).astype(np.uint8), alpha


def compact_fragments(labels: np.ndarray, colors: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Absorb sub-pixel micro-specks (`< scale^2`) into best adjacent visible neighbor while preserving `-1` transparency."""
    if len(colors) == 0:
        return labels, colors
    has_trans = bool((labels < 0).any())
    vis = labels >= 0
    if not vis.any():
        return labels, colors

    area = np.bincount(labels[vis].ravel(), minlength=len(colors))
    boxes = find_objects(np.where(vis, labels + 1, 0))
    span = np.array([max(s.stop - s.start for s in b) if b is not None else 0 for b in boxes])

    k1, k2 = [], []
    for a, b in [(labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])]:
        different = (a != b) & (a >= 0) & (b >= 0)
        if not different.any():
            continue
        va = a[different].astype(np.uint64)
        vb = b[different].astype(np.uint64)
        k1.append((va << 32) | vb)
        k2.append((vb << 32) | va)
    if not k1:
        return labels, colors

    uniq_k, contacts = np.unique(np.concatenate(k1 + k2), return_counts=True)
    src = (uniq_k >> 32).astype(np.int32)
    dst = (uniq_k & 0xFFFFFFFF).astype(np.int32)
    valid = (area[src] < scale ** 2) & (span[src] < 4 * scale) & (area[dst] >= scale ** 2)
    src, dst, contacts = src[valid], dst[valid], contacts[valid]
    if len(src) == 0:
        return labels, colors

    distance = np.linalg.norm(colors[src].astype(float) - colors[dst], axis=1)
    order = np.lexsort((-contacts, distance, src))
    mapping = np.arange(len(colors), dtype=np.int32)
    seen = set()
    for a_id, b_id in zip(src[order].tolist(), dst[order].tolist()):
        if a_id not in seen:
            mapping[a_id] = b_id
            seen.add(a_id)

    palette, ids = np.unique(colors, axis=0, return_inverse=True)
    if has_trans:
        field = np.full(labels.shape, -1, dtype=np.int32)
        field[vis] = ids[mapping[labels[vis]]]
        regions = label(np.where(vis, field + 1, 0), connectivity=1, background=0).astype(np.int32) - 1
        n_reg = int(regions.max()) + 1
        values = np.zeros((max(0, n_reg), 3), np.uint8)
        if n_reg > 0:
            r_vis = regions >= 0
            values[regions[r_vis]] = palette[field[r_vis]]
        return regions, values

    field = ids[mapping[labels]]
    regions = label(field + 1, connectivity=1).astype(np.int32) - 1
    values = np.empty((regions.max() + 1, 3), np.uint8)
    values[regions.ravel()] = palette[field.ravel()]
    return regions, values


def merge_perceptual_regions(
    regions: np.ndarray,
    colors: np.ndarray,
    line_mask: np.ndarray,
    *,
    fmaps=None,
    delicate_hi: np.ndarray | None = None,
    scale: int = 2,
    core_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Line-preserved Area~Color-Count quota recoloring + same-color boundary line erasure.

    4-stage topology-safe algorithm:
      0. Keep 100% of original native lines intact upfront as structural fences (zero broken strands).
      1. Segment into 8 neural semantic subjects (`weights/anime_part_seg.onnx`) and assign each
         subject an area-scaled color quota `K_max(part, area)` (small dense hair bun -> tight color
         count; large flowing hair -> generous color count; eyes/face/headwear -> high detail).
      2. Under the line framework, iteratively pick the adjacent same-subject fill pair with lowest
         `dE * sqrt(min_area_1x)` ("相邻 + 相近色彩 + 面积最小") and overwrite the smaller-area
         region with the larger-area region's EXACT color ("取 2/3 色彩填入 1/3 区域", zero color
         muddying) until every subject satisfies its Area~Color quota (or perceptual ceiling).
      3. Pixel-exact same-color boundary line erasure: any thin line pixel whose opposite-side fills
         now share the EXACT same recolored color & subject (and is not an intentional high-contrast
         drawn ink stroke or T-junction of a kept boundary) dissolves seamlessly into that shared
         color, merging same-color blocks into clean single polygons.
    """
    n_in = len(colors)
    if n_in <= 1:
        return regions, colors, {"before": n_in, "after": n_in, "colors_before": n_in, "colors_after": n_in}

    has_trans = bool((regions < 0).any())
    vis = regions >= 0
    if not vis.any():
        return regions, colors, {"before": n_in, "after": n_in, "colors_before": n_in, "colors_after": n_in}

    hs, ws = regions.shape
    scale2 = float(max(1, scale * scale))
    r_areas = np.bincount(regions[vis].ravel(), minlength=n_in).astype(np.float64)
    r_area_1x = (r_areas / scale2).astype(np.float64)
    r_lm = (
        np.bincount(regions[vis].ravel(), weights=line_mask[vis].astype(np.float32).ravel(), minlength=n_in)
        / np.maximum(r_areas, 1.0)
    )
    is_line = r_lm >= 0.35

    part_ids, part_stats = segment_semantic_parts(regions, colors, line_mask, fmaps, scale=scale)
    part_debug_out = os.environ.get("LUMINA_PART_MAP_OUT", "").strip()
    if part_debug_out:
        try:
            Image.fromarray(render_part_map_rgb(regions, part_ids)).save(part_debug_out)
        except Exception:
            pass

    total_vis_1x = float(r_area_1x.sum()) or 1.0
    vis_linear_1x = max(1.0, math.sqrt(total_vis_1x))

    # Local micro-fragment density (~1.1% of visible linear scale) separates congested dense
    # hair buns (sakuragirl) from cleanly spaced hair locks (hutao11) resolution-independently.
    dens_k = max(5, int(round(max(7.0, 0.011 * vis_linear_1x) * scale))) | 1
    is_tiny_frag = r_area_1x <= max(10.0, 0.000018 * total_vis_1x)
    tiny_px = np.zeros((hs, ws), dtype=np.float32)
    tiny_px[vis] = is_tiny_frag[regions[vis]].astype(np.float32)
    frag_dens_px = cv2.boxFilter(tiny_px, -1, (dens_k, dens_k), normalize=True)
    r_frag_dens = (
        np.bincount(regions[vis].ravel(), weights=frag_dens_px[vis].ravel(), minlength=n_in)
        / np.maximum(r_areas, 1.0)
    ).astype(np.float32)

    # Line CNN stroke probability & Gradient CNN grad_core_95 maps (full hs x ws + per-region)
    stroke_hi = np.zeros((hs, ws), dtype=np.float32)
    r_grad_core = np.zeros(n_in, dtype=np.float32)
    if fmaps is not None:
        s1x = None
        vf = getattr(fmaps, "line_vector_fields", None)
        if vf is not None and vf.ndim == 3 and vf.shape[0] >= 1:
            s1x = vf[0].astype(np.float32)
        lm = getattr(fmaps, "line_net_maps", None)
        if lm is not None and lm.ndim == 3 and lm.shape[0] >= 2:
            lm_ink = np.maximum(lm[0], lm[1]).astype(np.float32)
            if s1x is None:
                s1x = lm_ink
            else:
                if lm_ink.shape != s1x.shape:
                    lm_ink = cv2.resize(lm_ink, (s1x.shape[1], s1x.shape[0]), interpolation=cv2.INTER_LINEAR)
                s1x = np.maximum(s1x, lm_ink)
        if s1x is not None:
            stroke_hi = s1x if s1x.shape == (hs, ws) else cv2.resize(s1x, (ws, hs), interpolation=cv2.INTER_LINEAR)

        gc_1x = getattr(fmaps, "grad_core_95", None)
        if gc_1x is not None:
            gc_hi = (
                gc_1x.astype(np.float32)
                if gc_1x.shape == (hs, ws)
                else cv2.resize(gc_1x.astype(np.float32), (ws, hs), interpolation=cv2.INTER_LINEAR)
            )
            r_grad_core = (
                np.bincount(regions[vis].ravel(), weights=gc_hi[vis].ravel(), minlength=n_in)
                / np.maximum(r_areas, 1.0)
            ).astype(np.float32)

    labs = srgb_to_oklab(colors).astype(np.float32)
    colors_before = int(len(np.unique(colors, axis=0)))

    # ------------------------------------------------------------------
    # Stage 1: Resolution-Independent Area-Ratio Linear Quotas K_max[p] ∝ area_ratio[p]
    # ------------------------------------------------------------------
    # (K_full_canvas, min_colors, direct_dE_ceil, cross_line_dE_ceil)
    # Strictly proportional to subject area fraction `area_p / total_vis_area`:
    #   - sakuragirl hair bun (1.0% of canvas) -> K_max = 6 colors (true compact bun!)
    #   - hutao11 flowing hair (16.6% of canvas) -> K_max = 83 colors (preserves every lock!)
    #   - 1.png chibi hair (30.3% of canvas) -> K_max = 152 colors
    PART_BUDGET = {
        PART_EYES_MOUTH: (6000.0,  6, 0.020, 0.000),  # never cross lines; 1/3 near-tint -> 2/3
        PART_FACE_SKIN:  (2500.0,  6, 0.022, 0.000),  # never cross lines; protect cheek blush
        PART_HEADWEAR:   (4500.0, 10, 0.024, 0.000),  # never cross lines; protect peony/pearls/hat
        PART_HAIR:       ( 500.0,  5, 0.023, 0.095),  # 1.0% bun -> 5-6 col; 16.6% hutao -> 83 col!
        PART_NECK_HAND:  (1500.0,  8, 0.022, 0.014),  # scarf knits / gloves / obi
        PART_OBJECTS:    (1500.0,  8, 0.022, 0.014),
        PART_CLOTH:      ( 800.0, 10, 0.024, 0.014),  # coat / kimono / folds
        PART_BG:         ( 680.0, 10, 0.028, 0.016),  # linear with background area fraction
    }

    part_area_ratio: dict[int, float] = {}
    part_k_max: dict[int, int] = {}
    for pid, (k_full, k_min, _, _) in PART_BUDGET.items():
        m_p = (~is_line) & (part_ids == pid)
        a_p = float(r_area_1x[m_p].sum()) if m_p.any() else 0.0
        ratio_p = a_p / total_vis_1x
        part_area_ratio[pid] = ratio_p
        part_k_max[pid] = max(k_min, int(round(k_full * ratio_p))) if a_p > 0.0 else 0

    # Compact hair mass flag (<= 4.5% of total canvas, e.g. sakuragirl's 1.0% hair bun):
    # receives full convergence to its tight area-ratio color quota & full interior clutter erasure!
    # Large flowing hair (e.g. hutao11's 16.6% hair) keeps spaced hair-strand boundary lock!
    is_compact_hair = 0.0 < part_area_ratio[PART_HAIR] <= 0.045

    # ------------------------------------------------------------------
    # Stage 2: Build Fill-Adjacency Graphs (Phase A: Direct, Phase B: Across-Thin-Line Voronoi)
    # ------------------------------------------------------------------
    # 2A. Direct 4-connected fill-to-fill neighbors within same semantic subject
    k_direct = []
    for a, b in ((regions[:-1], regions[1:]), (regions[:, :-1], regions[:, 1:])):
        diff = (a != b) & (a >= 0) & (b >= 0)
        if not diff.any():
            continue
        va = a[diff]
        vb = b[diff]
        ok = (~is_line[va]) & (~is_line[vb]) & (part_ids[va] == part_ids[vb])
        if not ok.any():
            continue
        va64 = va[ok].astype(np.int64)
        vb64 = vb[ok].astype(np.int64)
        lo = np.minimum(va64, vb64)
        hi = np.maximum(va64, vb64)
        k_direct.append((lo << 32) | hi)

    if k_direct:
        u_dir_all = np.unique(np.concatenate(k_direct))
        dir_u = (u_dir_all >> 32).astype(np.int32)
        dir_v = (u_dir_all & 0xFFFFFFFF).astype(np.int32)
    else:
        dir_u = np.zeros(0, dtype=np.int32)
        dir_v = np.zeros(0, dtype=np.int32)

    # 2B. Across-thin-line fill-to-fill neighbors via EDT Voronoi projection on line pixels:
    # bridges even webbed hair-bun skeletons without touching any line pixel yet!
    # Thin-line half-width scales proportionally with visible linear scale (`0.28%` of sqrt(vis_area)).
    thin_line_half_1x = max(2.2, 0.0028 * vis_linear_1x)
    thin_line_half_px = float(max(3.5, thin_line_half_1x * float(max(1, scale))))
    fill_px = vis & (~line_mask)
    vor_reg = None
    dist_to_fill = None
    if fill_px.any() and (vis & line_mask).any():
        dist_to_fill, (iy, ix) = distance_transform_edt(~fill_px, return_indices=True)
        vor_reg = regions[iy, ix]
        near_line = vis & line_mask & (dist_to_fill <= thin_line_half_px)
        k_cross = []
        for (va, vb, m_ab) in (
            (vor_reg[:-1], vor_reg[1:], near_line[:-1] | near_line[1:]),
            (vor_reg[:, :-1], vor_reg[:, 1:], near_line[:, :-1] | near_line[:, 1:]),
        ):
            diff = m_ab & (va != vb) & (va >= 0) & (vb >= 0)
            if not diff.any():
                continue
            a_ids = va[diff]
            b_ids = vb[diff]
            ok = (~is_line[a_ids]) & (~is_line[b_ids]) & (part_ids[a_ids] == part_ids[b_ids])
            if not ok.any():
                continue
            a64 = a_ids[ok].astype(np.int64)
            b64 = b_ids[ok].astype(np.int64)
            lo = np.minimum(a64, b64)
            hi = np.maximum(a64, b64)
            k_cross.append((lo << 32) | hi)
        if k_cross:
            u_cr_all = np.unique(np.concatenate(k_cross))
            if k_direct:
                # Exclude pairs already directly touching
                u_cr_all = np.setdiff1d(u_cr_all, u_dir_all, assume_unique=True)
            cr_u = (u_cr_all >> 32).astype(np.int32)
            cr_v = (u_cr_all & 0xFFFFFFFF).astype(np.int32)
        else:
            cr_u = np.zeros(0, dtype=np.int32)
            cr_v = np.zeros(0, dtype=np.int32)
    else:
        cr_u = np.zeros(0, dtype=np.int32)
        cr_v = np.zeros(0, dtype=np.int32)

    # ------------------------------------------------------------------
    # Stage 2C: Group-First, Max-Original-Area Winner Recoloring
    # ("先遍历确定必须合并的色彩组，再取组内最大原始面积主色一口吞掉其余小色，最后统一改色")
    # ------------------------------------------------------------------
    # Why this completely eliminates snowballing ("连锁吞色"):
    #   Suppose a group has 3 colors with original areas 1(20), 2(25), 3(40).
    #   Greedy pairwise merging pools 1+2 -> 2(45), which then falsely swallows 3(40)!
    #   Instead, during grouping we NEVER pool distinct colors' areas: each group `G` tracks
    #   `grp_col_orig_area[G][orig_color]` (the true original area of each distinct original color
    #   inside `G`). When {1, 2} groups with {3}, true original areas are 20, 25, 40 -> `3` ALWAYS
    #   wins as the group's true anchor `win_rep`, every member's original OKLab is verified against
    #   `labs[win_rep]`, and ONLY AFTER all grouping finishes does `colors[win_rep]` overwrite `1,2,3`!
    rgb_keys = (
        colors[:, 0].astype(np.int32) * 65536
        + colors[:, 1].astype(np.int32) * 256
        + colors[:, 2].astype(np.int32)
    )
    part_color_counts: dict[int, dict[int, int]] = {p: {} for p in PART_BUDGET}
    for i in range(n_in):
        if is_line[i]:
            continue
        p = int(part_ids[i])
        ck = int(rgb_keys[i])
        d = part_color_counts[p]
        d[ck] = d.get(ck, 0) + 1
    part_unique_n = {p: len(d) for p, d in part_color_counts.items()}

    parent = np.arange(n_in, dtype=np.int32)
    grp_members: list[list[int]] = [[i] for i in range(n_in)]
    # Per-group map: orig_rgb_key -> (sum of original 1x area for that exact color, sample region_id)
    grp_col_area: list[dict[int, tuple[float, int]]] = [
        {int(rgb_keys[i]): (float(r_area_1x[i]), i)} for i in range(n_in)
    ]
    grp_win_rep = np.arange(n_in, dtype=np.int32)  # region_id of true max-original-area color in group (final color!)
    grp_win_area = r_area_1x.copy()                # true original area of winning color (determines winner: 4 & 6!)
    grp_total_area = r_area_1x.copy()              # SUM of all member areas in group (determines merge sort rank: 1&2 vs 3!)
    grp_ver = np.zeros(n_in, dtype=np.int32)       # lazy heap invalidation version
    # Chroma-weighted OKLab vectors `[L, 2.2*a, 2.2*b]` prevent hue-crossing bleeds
    # (e.g. warm/lilac neck shadow turning cyan-blue kimono, or peach cheek turning pink)!
    labs_w = labs.copy()
    labs_w[:, 1] *= 2.2
    labs_w[:, 2] *= 2.2
    grp_orig_min = labs_w.copy()
    grp_orig_max = labs_w.copy()
    grp_grad = r_grad_core.copy()
    grp_fdens = r_frag_dens.copy()

    def _find(x: int) -> int:
        x = int(x)
        while int(parent[x]) != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def _drift_ok(root: int, target_lab_w: np.ndarray, max_drift: float) -> bool:
        """Verify EVERY original member region in `root` is within `max_drift` chroma-weighted OKLab
        distance of `target_lab_w` (eliminates transitive chaining AND cross-hue neck/scarf bleeds!)."""
        d_corner = np.maximum(np.abs(grp_orig_max[root] - target_lab_w), np.abs(grp_orig_min[root] - target_lab_w))
        if float(np.linalg.norm(d_corner)) <= max_drift:
            return True
        m_ids = grp_members[root]
        if len(m_ids) == 1:
            return float(np.linalg.norm(labs_w[m_ids[0]] - target_lab_w)) <= max_drift
        diffs = labs_w[m_ids] - target_lab_w[None, :]
        return float(np.sqrt((diffs * diffs).sum(axis=1)).max()) <= max_drift

    def _pick_combined_winner(ra: int, rb: int) -> tuple[int, float]:
        """Find the single original color with the LARGEST true original area in `G_a U G_b`
        (distinct colors never pool areas to gang up on a larger original color!)."""
        da = grp_col_area[ra]
        db = grp_col_area[rb]
        best_rep = int(grp_win_rep[ra])
        best_a = float(grp_win_area[ra])
        if float(grp_win_area[rb]) > best_a:
            best_rep = int(grp_win_rep[rb])
            best_a = float(grp_win_area[rb])
        # Only need to check keys shared by BOTH groups (sum of same original color)
        if len(da) <= len(db):
            small_d, big_d = da, db
        else:
            small_d, big_d = db, da
        for ck, (aa, rep_i) in small_d.items():
            other = big_d.get(ck)
            if other is not None:
                comb = aa + other[0]
                if comb > best_a:
                    best_a = comb
                    best_rep = rep_i
        return best_rep, best_a

    def _edge_limit(ra: int, rb: int, *, cross_line: bool) -> float:
        p = int(part_ids[ra])
        _, _, lim_dir, lim_cr = PART_BUDGET[p]
        lim = lim_cr if cross_line else lim_dir
        if lim <= 0.0:
            return 0.0
        in_grad = max(float(grp_grad[ra]), float(grp_grad[rb])) >= 0.55
        if in_grad:
            lim = min(lim, 0.018 if not cross_line else 0.014)
        min_a = min(float(grp_win_area[ra]), float(grp_win_area[rb]))
        if p == PART_HAIR:
            fd = max(float(grp_fdens[ra]), float(grp_fdens[rb]))
            if is_compact_hair:
                # Small-area-ratio hair bun (e.g. sakuragirl 1.0%): consolidate aggressively toward
                # its tight area-ratio color quota while keeping headwear/peony locked out!
                lim = 0.145 if cross_line else 0.120
            elif cross_line:
                if fd >= 0.35 and min_a <= 24.0:
                    lim = 0.030
                else:
                    lim = 0.012
            else:
                if (not in_grad) and fd >= 0.35 and min_a <= 24.0:
                    lim = 0.032
                else:
                    lim = 0.022
        elif (not cross_line) and (not in_grad) and p in (PART_BG, PART_CLOTH) and min_a <= 18.0:
            lim *= 1.10
        return lim

    def _merge_two_groups(ra: int, rb: int, win_rep: int, win_area: float) -> None:
        """Merge group `rb` into `ra` (or vice versa) with true max-original-area winner `win_rep`,
        updating live subject unique-color counts cleanly without mutating pixel colors yet."""
        p = int(part_ids[ra])
        old_ck_a = int(rgb_keys[int(grp_win_rep[ra])])
        old_ck_b = int(rgb_keys[int(grp_win_rep[rb])])
        new_ck = int(rgb_keys[win_rep])
        cnt_a = len(grp_members[ra])
        cnt_b = len(grp_members[rb])
        dc = part_color_counts[p]

        # Decrement old winner color counts, increment new winner color count
        for old_ck, cnt in ((old_ck_a, cnt_a), (old_ck_b, cnt_b)):
            if old_ck != new_ck:
                rem = dc.get(old_ck, 0) - cnt
                if rem <= 0:
                    dc.pop(old_ck, None)
                    part_unique_n[p] -= 1
                else:
                    dc[old_ck] = rem
                dc[new_ck] = dc.get(new_ck, 0) + cnt

        # Keep the larger member-list root as DSU parent for fast dict/list union
        if len(grp_members[ra]) < len(grp_members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        grp_members[ra].extend(grp_members[rb])
        grp_members[rb] = []
        da = grp_col_area[ra]
        for ck, (ab, rep_b) in grp_col_area[rb].items():
            prev = da.get(ck)
            da[ck] = (prev[0] + ab, prev[1]) if prev is not None else (ab, rep_b)
        grp_col_area[rb] = {}
        grp_win_rep[ra] = win_rep
        grp_win_area[ra] = win_area
        grp_total_area[ra] = float(grp_total_area[ra]) + float(grp_total_area[rb])
        grp_ver[ra] += 1
        grp_orig_min[ra] = np.minimum(grp_orig_min[ra], grp_orig_min[rb])
        grp_orig_max[ra] = np.maximum(grp_orig_max[ra], grp_orig_max[rb])
        grp_grad[ra] = max(float(grp_grad[ra]), float(grp_grad[rb]))
        grp_fdens[ra] = max(float(grp_fdens[ra]), float(grp_fdens[rb]))

    def _run_group_pass(edge_u: np.ndarray, edge_v: np.ndarray, *, cross_line: bool) -> int:
        """Progressive-threshold equal-weight (area + color) grouping with sum-area rank &
        max-original-area color winner:
          - Progressive tiers (0.35 -> 0.60 -> 0.82 -> 1.00 of subject `lim`): true near-color
            pairs (1&2, 3&4, 5&6) ALWAYS merge first before looser cross-pairs are considered!
          - Within each tier, dynamic heap ranks pairs with 1:1 equal weight on normalized
            summed group area (`grp_total_area`, updated dynamically as groups merge!) and
            normalized color distance (`dE_w / tier_lim`).
          - Final group color is ALWAYS the single max-original-area color (`grp_win_rep`: 4 & 6!).
        """
        if len(edge_u) == 0:
            return 0
        # Build neighbor map on current roots once for fast dynamic heap re-pushing
        adj: dict[int, list[int]] = {}
        seen_e = set()
        for uu, vv in zip(edge_u.tolist(), edge_v.tolist()):
            ra, rb = _find(uu), _find(vv)
            if ra == rb:
                continue
            if ra > rb:
                ra, rb = rb, ra
            ek = (ra << 32) | rb
            if ek in seen_e:
                continue
            seen_e.add(ek)
            adj.setdefault(ra, []).append(rb)
            adj.setdefault(rb, []).append(ra)

        log_area_scale = math.log1p(max(float(np.percentile(r_area_1x[~is_line], 95)) if (~is_line).any() else 100.0, 10.0))
        merges = 0

        for tier_frac in (0.35, 0.60, 0.82, 1.00):
            heap: list[tuple[float, int, int, int, int]] = []

            def _push_edge(ra: int, rb: int) -> None:
                lim = _edge_limit(ra, rb, cross_line=cross_line)
                if lim <= 0.0:
                    return
                tier_lim = lim * tier_frac
                wa, wb = int(grp_win_rep[ra]), int(grp_win_rep[rb])
                de_w = float(np.linalg.norm(labs_w[wa] - labs_w[wb]))
                if de_w > tier_lim:
                    return
                a_lo = min(float(grp_total_area[ra]), float(grp_total_area[rb]))
                a_hi = max(float(grp_total_area[ra]), float(grp_total_area[rb]))
                # 1:1 equal weight on normalized summed group area & normalized color distance
                norm_a = min(1.0, math.log1p(a_lo) / log_area_scale)
                norm_d = de_w / max(tier_lim, 1e-6)
                cost = 0.50 * norm_a + 0.50 * norm_d + 0.05 * min(1.0, math.log1p(a_hi) / log_area_scale)
                heapq.heappush(heap, (cost, ra, rb, int(grp_ver[ra]), int(grp_ver[rb])))

            # Seed heap with all current root-to-root edges meeting this tier's near-color threshold
            pushed = set()
            for r0, nbs in list(adj.items()):
                ra = _find(r0)
                for nb in nbs:
                    rb = _find(nb)
                    if ra == rb:
                        continue
                    lo, hi = (ra, rb) if ra < rb else (rb, ra)
                    ek = (lo << 32) | hi
                    if ek in pushed:
                        continue
                    pushed.add(ek)
                    _push_edge(lo, hi)

            while heap:
                _cost, a0, b0, va, vb = heapq.heappop(heap)
                ra, rb = _find(a0), _find(b0)
                if ra == rb or int(grp_ver[ra]) != va or int(grp_ver[rb]) != vb:
                    continue
                p = int(part_ids[ra])
                lim = _edge_limit(ra, rb, cross_line=cross_line)
                if lim <= 0.0:
                    continue
                tier_lim = lim * tier_frac
                wa, wb = int(grp_win_rep[ra]), int(grp_win_rep[rb])
                de_cur = float(np.linalg.norm(labs_w[wa] - labs_w[wb]))
                if de_cur > tier_lim:
                    continue
                if part_unique_n[p] <= part_k_max[p] and (cross_line or de_cur > 0.011):
                    continue

                win_rep, win_area = _pick_combined_winner(ra, rb)
                win_lab_w = labs_w[win_rep]
                if not (_drift_ok(ra, win_lab_w, lim * 1.08) and _drift_ok(rb, win_lab_w, lim * 1.08)):
                    continue

                # Remember which root gets absorbed so we can re-attach its neighbor list
                keep_r, drop_r = (ra, rb) if len(grp_members[ra]) >= len(grp_members[rb]) else (rb, ra)
                _merge_two_groups(ra, rb, win_rep, win_area)
                merges += 1

                # Re-attach neighbor list of absorbed group & re-evaluate live edges with updated `grp_total_area`!
                drop_nbs = adj.pop(drop_r, ())
                keep_nbs = adj.setdefault(keep_r, [])
                seen_nb = set()
                for nb in drop_nbs:
                    rnb = _find(nb)
                    if rnb != keep_r and rnb not in seen_nb:
                        seen_nb.add(rnb)
                        keep_nbs.append(rnb)
                # Re-push keep_r's current neighbors with updated total_area & grp_ver
                for nb in keep_nbs:
                    rnb = _find(nb)
                    if rnb != keep_r:
                        _push_edge(keep_r, rnb)
        return merges

    merges_direct = _run_group_pass(dir_u, dir_v, cross_line=False)
    merges_cross = _run_group_pass(cr_u, cr_v, cross_line=True)

    # Pass 3: Compact Hair-Mass Progressive Equal-Weight Convergence (scoped per spatial connected
    # component!). Mirrors the exact 6-color -> 2-color progressive algorithm:
    #   1. Progressive near-color tiers (0.045 -> 0.085 -> 0.14 -> 0.22): pairs closest colors
    #      starting from smallest `grp_total_area` (e.g. 1&2, then 3&4, then 5&6)!
    #   2. After each merge, `grp_total_area` updates to `A_a + A_b` (re-ranking remaining groups),
    #      while final color is ALWAYS the single max-original-area color (`grp_win_rep`, e.g. 4 & 6!).
    merges_quota = 0
    if is_compact_hair and part_unique_n[PART_HAIR] > part_k_max[PART_HAIR]:
        hair_px_bin = np.zeros((hs, ws), dtype=np.uint8)
        hair_px_bin[vis] = (part_ids[regions[vis]] == PART_HAIR).astype(np.uint8)
        hair_cc = label(cv2.dilate(hair_px_bin, np.ones((3, 3), np.uint8)), connectivity=1)
        hair_roots = [
            i for i in range(n_in)
            if (not is_line[i]) and int(part_ids[i]) == PART_HAIR and _find(i) == i
        ]
        r_cc_sum = np.bincount(regions[vis].ravel(), weights=hair_cc[vis].astype(np.float64).ravel(), minlength=n_in)
        r_hair_cc = np.rint(r_cc_sum / np.maximum(r_areas, 1.0)).astype(np.int32)

        cc_groups: dict[int, list[int]] = {}
        for r in hair_roots:
            cc_groups.setdefault(int(r_hair_cc[r]), []).append(r)

        tot_hair_a = float(sum(float(grp_total_area[r]) for r in hair_roots)) or 1.0
        for _cid_cc, cc_roots in cc_groups.items():
            cc_a = float(sum(float(grp_total_area[r]) for r in cc_roots))
            cc_k = max(3, int(round(part_k_max[PART_HAIR] * (cc_a / tot_hair_a))))
            active = list(cc_roots)
            if len(active) <= cc_k:
                continue

            # Progressive near-color tiers: merge smallest-total-area near-color pairs first!
            for tier_d in (0.045, 0.085, 0.140, 0.220):
                while len(active) > cc_k:
                    # Sort active groups ascending by CURRENT summed `grp_total_area`
                    active.sort(key=lambda r: float(grp_total_area[r]))
                    merged_one = False
                    # Scan from smallest total_area to find first group with a `<= tier_d` near-color partner
                    for idx_a, ra in enumerate(active):
                        wa = int(grp_win_rep[ra])
                        best_idx_b = -1
                        best_score = 1e9
                        la_tot = math.log1p(float(grp_total_area[ra]))
                        max_tot_log = math.log1p(float(grp_total_area[active[-1]])) + 1e-6
                        for idx_b in range(idx_a + 1, len(active)):
                            rb = active[idx_b]
                            wb = int(grp_win_rep[rb])
                            d_w = float(np.linalg.norm(labs_w[wa] - labs_w[wb]))
                            if d_w > tier_d:
                                continue
                            # Equal weight on normalized partner area & normalized color distance
                            sc = 0.50 * (math.log1p(float(grp_total_area[rb])) / max_tot_log) + 0.50 * (d_w / tier_d)
                            if sc < best_score:
                                best_score = sc
                                best_idx_b = idx_b
                        if best_idx_b >= 0:
                            rb = active[best_idx_b]
                            win_rep, win_area = _pick_combined_winner(ra, rb)
                            keep_r = ra if len(grp_members[ra]) >= len(grp_members[rb]) else rb
                            drop_idx = best_idx_b if keep_r == ra else idx_a
                            _merge_two_groups(ra, rb, win_rep, win_area)
                            active.pop(drop_idx)
                            merges_quota += 1
                            merged_one = True
                            break
                    if not merged_one:
                        break

    # ------------------------------------------------------------------
    # Stage 2D: Length-Proportional Line Color Unification ("线条色彩归一按长度比例计算")
    # ------------------------------------------------------------------
    # Compute exact 1x stroke length `r_len_1x[i]` for every line region via boundary-pixel perimeter/2.
    # Applies the EXACT SAME Group-First Max-Original-Length Winner rule to adjacent line segments
    # within the same semantic subject, with quota K_line_max[p] ∝ line_length_ratio[p]!
    b_cnt = np.zeros(n_in, dtype=np.float64)
    for p1, p2 in ((regions[:-1], regions[1:]), (regions[:, :-1], regions[:, 1:])):
        diff = p1 != p2
        if not diff.any():
            continue
        m1 = diff & (p1 >= 0)
        m2 = diff & (p2 >= 0)
        if m1.any():
            b_cnt += np.bincount(p1[m1].ravel(), minlength=n_in)
        if m2.any():
            b_cnt += np.bincount(p2[m2].ravel(), minlength=n_in)
    r_len_1x = np.maximum(b_cnt / (2.0 * float(max(1, scale))), 1.0)

    LINE_K_FULL = {
        PART_EYES_MOUTH: (1800.0, 4, 0.022),
        PART_FACE_SKIN:  ( 900.0, 4, 0.026),
        PART_HEADWEAR:   (1600.0, 6, 0.030),
        PART_HAIR:       ( 220.0, 3, 0.065 if is_compact_hair else 0.036),
        PART_NECK_HAND:  ( 600.0, 4, 0.030),
        PART_OBJECTS:    ( 600.0, 4, 0.030),
        PART_CLOTH:      ( 360.0, 6, 0.032),
        PART_BG:         ( 340.0, 6, 0.036),
    }
    total_line_len = float(r_len_1x[is_line].sum()) or 1.0
    line_k_max: dict[int, int] = {}
    line_color_counts: dict[int, dict[int, int]] = {p: {} for p in PART_BUDGET}
    for i in range(n_in):
        if not is_line[i]:
            continue
        p = int(part_ids[i])
        ck = int(rgb_keys[i])
        d = line_color_counts[p]
        d[ck] = d.get(ck, 0) + 1
    line_unique_n = {p: len(d) for p, d in line_color_counts.items()}
    for pid, (kl_full, kl_min, _) in LINE_K_FULL.items():
        ml = is_line & (part_ids == pid)
        lp = float(r_len_1x[ml].sum()) if ml.any() else 0.0
        line_k_max[pid] = max(kl_min, int(round(kl_full * (lp / total_line_len)))) if lp > 0.0 else 0

    # Collect adjacent line-to-line pairs:
    # Allow directly touching continuous line segments to unify ACROSS semantic-mask edges
    # (e.g. a curling hair wisp extending past the 512x512 hair mask into bg or over cheek!)
    # whenever their original stroke colors are close — eliminating mid-strand color breaks!
    k_line_pairs = []
    for a, b in ((regions[:-1], regions[1:]), (regions[:, :-1], regions[:, 1:])):
        diff = (a != b) & (a >= 0) & (b >= 0)
        if not diff.any():
            continue
        va, vb = a[diff], b[diff]
        same_or_wisp = (part_ids[va] == part_ids[vb]) | (
            (part_ids[va] == PART_HAIR) | (part_ids[vb] == PART_HAIR)
        )
        ok = is_line[va] & is_line[vb] & same_or_wisp
        if not ok.any():
            continue
        va64, vb64 = va[ok].astype(np.int64), vb[ok].astype(np.int64)
        k_line_pairs.append((np.minimum(va64, vb64) << 32) | np.maximum(va64, vb64))

    line_merges = 0
    l_parent = np.arange(n_in, dtype=np.int32)
    l_members: list[list[int]] = [[i] for i in range(n_in)]
    l_col_len: list[dict[int, tuple[float, int]]] = [
        {int(rgb_keys[i]): (float(r_len_1x[i]), i)} for i in range(n_in)
    ]
    l_win_rep = np.arange(n_in, dtype=np.int32)
    l_win_len = r_len_1x.copy()
    l_total_len = r_len_1x.copy()
    l_ver = np.zeros(n_in, dtype=np.int32)
    l_orig_min = labs_w.copy()
    l_orig_max = labs_w.copy()

    def _lfind(x: int) -> int:
        x = int(x)
        while int(l_parent[x]) != x:
            l_parent[x] = l_parent[l_parent[x]]
            x = int(l_parent[x])
        return x

    def _ldrift_ok(root: int, target_lw: np.ndarray, max_d: float) -> bool:
        d_c = np.maximum(np.abs(l_orig_max[root] - target_lw), np.abs(l_orig_min[root] - target_lw))
        if float(np.linalg.norm(d_c)) <= max_d:
            return True
        m_ids = l_members[root]
        diffs = labs_w[m_ids] - target_lw[None, :]
        return float(np.sqrt((diffs * diffs).sum(axis=1)).max()) <= max_d

    if k_line_pairs:
        ul_all = np.unique(np.concatenate(k_line_pairs))
        lu_arr = (ul_all >> 32).astype(np.int32)
        lv_arr = (ul_all & 0xFFFFFFFF).astype(np.int32)
        l_adj: dict[int, list[int]] = {}
        for uu, vv in zip(lu_arr.tolist(), lv_arr.tolist()):
            l_adj.setdefault(uu, []).append(vv)
            l_adj.setdefault(vv, []).append(uu)
        log_len_scale = math.log1p(max(float(np.percentile(r_len_1x[is_line], 95)) if is_line.any() else 50.0, 6.0))

        for l_tier in (0.40, 0.72, 1.00):
            l_heap: list[tuple[float, int, int, int, int]] = []

            def _lpush(ra: int, rb: int) -> None:
                pa, pb = int(part_ids[ra]), int(part_ids[rb])
                base_lim = LINE_K_FULL[pa][2] if pa == pb else min(LINE_K_FULL[pa][2], LINE_K_FULL[pb][2], 0.044)
                t_lim = base_lim * l_tier
                wa, wb = int(l_win_rep[ra]), int(l_win_rep[rb])
                de_w = float(np.linalg.norm(labs_w[wa] - labs_w[wb]))
                if de_w > t_lim:
                    return
                l_lo = min(float(l_total_len[ra]), float(l_total_len[rb]))
                l_hi = max(float(l_total_len[ra]), float(l_total_len[rb]))
                norm_l = min(1.0, math.log1p(l_lo) / log_len_scale)
                norm_d = de_w / max(t_lim, 1e-6)
                heapq.heappush(l_heap, (0.50 * norm_l + 0.50 * norm_d + 0.05 * min(1.0, math.log1p(l_hi) / log_len_scale), ra, rb, int(l_ver[ra]), int(l_ver[rb])))

            seen_lp = set()
            for r0, nbs in list(l_adj.items()):
                ra = _lfind(r0)
                for nb in nbs:
                    rb = _lfind(nb)
                    if ra == rb:
                        continue
                    lo, hi = (ra, rb) if ra < rb else (rb, ra)
                    ek = (lo << 32) | hi
                    if ek in seen_lp:
                        continue
                    seen_lp.add(ek)
                    _lpush(lo, hi)

            while l_heap:
                _c, a0, b0, va, vb = heapq.heappop(l_heap)
                ra, rb = _lfind(a0), _lfind(b0)
                if ra == rb or int(l_ver[ra]) != va or int(l_ver[rb]) != vb:
                    continue
                pa, pb = int(part_ids[ra]), int(part_ids[rb])
                lim_l = LINE_K_FULL[pa][2] if pa == pb else min(LINE_K_FULL[pa][2], LINE_K_FULL[pb][2], 0.044)
                t_lim = lim_l * l_tier
                wa, wb = int(l_win_rep[ra]), int(l_win_rep[rb])
                de_cur = float(np.linalg.norm(labs_w[wa] - labs_w[wb]))
                if de_cur > t_lim:
                    continue
                if pa == pb and line_unique_n[pa] <= line_k_max[pa] and de_cur > 0.014:
                    continue
                # Max-original-length winner (zero snowballing!)
                da, db = l_col_len[ra], l_col_len[rb]
                w_rep, w_len = (wa, float(l_win_len[ra])) if float(l_win_len[ra]) >= float(l_win_len[rb]) else (wb, float(l_win_len[rb]))
                sd, bd = (da, db) if len(da) <= len(db) else (db, da)
                for ck, (la_v, rep_i) in sd.items():
                    oth = bd.get(ck)
                    if oth is not None and (la_v + oth[0]) > w_len:
                        w_len = la_v + oth[0]
                        w_rep = rep_i
                w_lw = labs_w[w_rep]
                if not (_ldrift_ok(ra, w_lw, lim_l * 1.10) and _ldrift_ok(rb, w_lw, lim_l * 1.10)):
                    continue
                new_ck = int(rgb_keys[w_rep])
                for p_side, old_ck, cnt in ((pa, int(rgb_keys[wa]), len(l_members[ra])), (pb, int(rgb_keys[wb]), len(l_members[rb]))):
                    if old_ck != new_ck:
                        dc = line_color_counts[p_side]
                        rem = dc.get(old_ck, 0) - cnt
                        if rem <= 0:
                            dc.pop(old_ck, None)
                            line_unique_n[p_side] -= 1
                        else:
                            dc[old_ck] = rem
                        dc[new_ck] = dc.get(new_ck, 0) + cnt
                keep_r, drop_r = (ra, rb) if len(l_members[ra]) >= len(l_members[rb]) else (rb, ra)
                l_parent[drop_r] = keep_r
                l_members[keep_r].extend(l_members[drop_r])
                l_members[drop_r] = []
                da = l_col_len[keep_r]
                for ck, (lb_v, rep_b) in l_col_len[drop_r].items():
                    prev = da.get(ck)
                    da[ck] = (prev[0] + lb_v, prev[1]) if prev is not None else (lb_v, rep_b)
                l_col_len[drop_r] = {}
                l_win_rep[keep_r] = w_rep
                l_win_len[keep_r] = w_len
                l_total_len[keep_r] = float(l_total_len[keep_r]) + float(l_total_len[drop_r])
                l_ver[keep_r] += 1
                l_orig_min[keep_r] = np.minimum(l_orig_min[keep_r], l_orig_min[drop_r])
                l_orig_max[keep_r] = np.maximum(l_orig_max[keep_r], l_orig_max[drop_r])
                line_merges += 1
                drop_nbs = l_adj.pop(drop_r, ())
                keep_nbs = l_adj.setdefault(keep_r, [])
                for nb in drop_nbs:
                    rnb = _lfind(nb)
                    if rnb != keep_r:
                        keep_nbs.append(rnb)
                for nb in keep_nbs:
                    rnb = _lfind(nb)
                    if rnb != keep_r:
                        _lpush(keep_r, rnb)

    # Also converge disconnected same-subject line colors by length-weighted dominant anchors
    # strictly within chroma-weighted tolerance so lines across each subject unify cleanly!
    for pid, (_, _, lim_l) in LINE_K_FULL.items():
        if line_unique_n[pid] <= line_k_max[pid]:
            continue
        p_lroots = [i for i in range(n_in) if is_line[i] and int(part_ids[i]) == pid and _lfind(i) == i]
        if len(p_lroots) <= line_k_max[pid]:
            continue
        p_lroots.sort(key=lambda r: -float(l_win_len[r]))
        anchors: list[int] = []
        a_labs_w: list[np.ndarray] = []
        for r in p_lroots:
            wr = int(l_win_rep[r])
            lw = labs_w[wr]
            if a_labs_w:
                arr = np.asarray(a_labs_w, dtype=np.float32)
                ds = np.linalg.norm(arr - lw, axis=1)
                bi = int(np.argmin(ds))
                if float(ds[bi]) <= lim_l and (len(anchors) >= line_k_max[pid] or float(ds[bi]) <= lim_l * 0.65):
                    l_win_rep[r] = int(l_win_rep[anchors[bi]])
                    line_merges += 1
                    continue
            anchors.append(r)
            a_labs_w.append(lw)

    # ONE-SHOT FINAL COMMIT: fills use max-original-area winner; lines use max-original-length winner!
    final_rgb = colors.copy()
    for i in range(n_in):
        if is_line[i]:
            lr = _lfind(i)
            parent[i] = lr
            final_rgb[i] = colors[int(l_win_rep[lr])]
        else:
            r = _find(i)
            parent[i] = r
            final_rgb[i] = colors[int(grp_win_rep[r])]

    # Stage 2E: Strip-Shaped Hair Strand Detection + Proportional Mean-Width Monochrome Averaging
    # ("按最粗发丝宽度占整图比例判断条状区域，平均宽度小于该比例的发丝整条均色融合为单色线条")
    # Calibrated directly from reference curling hair wisps (thickest strand 6.0px / 1254.4px = 0.48%):
    #   - Strip mean width:    W_mean(C) = 2 * Area_1x(C) / Perim_1x(C) <= 0.0048 * sqrt(vis_area_1x)
    #   - Strip peak width:    W_max(C)  <= 0.0064 * sqrt(vis_area_1x)
    #   - Strip elongation:    E(C)      = Perim_1x(C)^2 / (16 * Area_1x(C)) >= 2.2 (true strip vs blob!)
    # Every qualifying connected hair-strand strip (including dark ink borders + gray/purple cores &
    # anti-aliased edges) is averaged by area-weighted RGB into ONE clean monochrome strand polygon!
    final_lab = srgb_to_oklab(final_rgb).astype(np.float32)
    final_lab_w = final_lab.copy()
    final_lab_w[:, 1] *= 2.2
    final_lab_w[:, 2] *= 2.2

    # Compute exact inscribed max width (at 1x) and mean width (2*A/P) of every region
    reg_edge = np.zeros((hs, ws), dtype=bool)
    reg_edge[:-1, :] |= regions[:-1, :] != regions[1:, :]
    reg_edge[1:, :] |= regions[:-1, :] != regions[1:, :]
    reg_edge[:, :-1] |= regions[:, :-1] != regions[:, 1:]
    reg_edge[:, 1:] |= regions[:, :-1] != regions[:, 1:]
    if has_trans:
        reg_edge |= ~vis
    dist_in = distance_transform_edt(~reg_edge)
    r_max_half_px = np.zeros(n_in, dtype=np.float32)
    np.maximum.at(r_max_half_px, regions[vis], dist_in[vis])
    r_max_width_1x = 2.0 * (r_max_half_px + 0.5) / float(max(1, scale))
    r_perim_1x = np.maximum(b_cnt / float(max(1, scale)), 1.0)
    r_mean_width_1x = (2.0 * r_area_1x) / r_perim_1x

    # Resolution-independent proportional strip width limits (0.48% mean / 0.80% X-junction peak)
    max_strip_w_1x = max(1.8, 0.0048 * vis_linear_1x)
    max_strip_peak_1x = max(2.8, 0.0080 * vis_linear_1x)
    max_strip_area_1x = max(80.0, 0.008 * total_vis_1x)

    # Dominant hair median color (chroma-weighted OKLab) distinguishes genuine hair tones from
    # adjacent crimson peony petals, bright white window holes, or pale peach neck skin!
    hair_fill_mask_2e = (~is_line) & (part_ids == PART_HAIR)
    if hair_fill_mask_2e.any():
        hair_lab_med = np.median(labs[hair_fill_mask_2e], axis=0).astype(np.float32)
        hair_lab_w_med = np.median(labs_w[hair_fill_mask_2e], axis=0).astype(np.float32)
    else:
        hair_lab_med = None
        hair_lab_w_med = None

    # Pure geometry + color-coherence thin-strip candidates across hair, background wisps, and neck
    # (NO fragile dependency on 512x512 PART_HAIR mask reaching 2px curling wisps emerging past the
    # peony or over white window glass!). Protected subjects (eyes/mouth, face skin, peony/pearl
    # headwear, clothes) and wide shapes (e.g. window bars > 0.48% width) are automatically excluded.
    l_ceil_2e = min(0.70, float(hair_lab_med[0]) + 0.46) if hair_lab_med is not None else 0.66
    hair_ab_med = hair_lab_med[1:3] if hair_lab_med is not None else np.zeros(2, dtype=np.float32)
    r_chroma = np.linalg.norm(final_lab[:, 1:3], axis=1)
    r_de_ab_hair = np.linalg.norm(final_lab[:, 1:3] - hair_ab_med[None, :], axis=1)

    cand_strip = np.zeros(n_in, dtype=bool)
    for i in range(n_in):
        p_i = int(part_ids[i])
        if p_i not in (PART_HAIR, PART_BG, PART_NECK_HAND):
            continue
        if float(r_mean_width_1x[i]) > max_strip_w_1x or float(r_max_width_1x[i]) > max_strip_w_1x * 1.35:
            continue
        # Only dark/hair-toned strips (reject white/pale window glass, peach skin, & vivid petals)
        if float(final_lab[i, 0]) > l_ceil_2e:
            continue
        if float(r_chroma[i]) > 0.075 and float(r_de_ab_hair[i]) > 0.12:
            continue
        # On PART_BG / PART_NECK_HAND, require at least one strip-like cue (is_line OR elongated
        # narrow fill) so compact dots/pebbles in background aren't grouped
        if p_i != PART_HAIR and (not is_line[i]):
            elong_i = (float(r_perim_1x[i]) ** 2) / max(16.0 * float(r_area_1x[i]), 1e-6)
            if elong_i < 1.8 and float(r_max_width_1x[i]) > max_strip_w_1x * 0.75:
                continue
        cand_strip[i] = True

    strand_cores_fused = 0
    if cand_strip.any():
        strip_px = np.zeros((hs, ws), dtype=bool)
        strip_px[vis] = cand_strip[regions[vis]]
        cc_map = label(strip_px, connectivity=1).astype(np.int32)
        n_cc = int(cc_map.max())
        if n_cc > 0:
            # Exact per-component area, outer perimeter, exterior non-hair contact & inscribed peak width
            cc_valid = cc_map > 0
            cc_areas = np.bincount(cc_map[cc_valid].ravel(), minlength=n_cc + 1).astype(np.float64)
            cc_area_1x = cc_areas / scale2

            cc_perim = np.zeros(n_cc + 1, dtype=np.float64)
            cc_ext_nonhair = np.zeros(n_cc + 1, dtype=np.float64)
            is_hair_px = np.zeros((hs, ws), dtype=bool)
            is_hair_px[vis] = part_ids[regions[vis]] == PART_HAIR
            for (c1, c2, h1, h2) in (
                (cc_map[:-1], cc_map[1:], is_hair_px[:-1], is_hair_px[1:]),
                (cc_map[:, :-1], cc_map[:, 1:], is_hair_px[:, :-1], is_hair_px[:, 1:]),
            ):
                diff = c1 != c2
                if not diff.any():
                    continue
                m1 = diff & (c1 > 0)
                m2 = diff & (c2 > 0)
                if m1.any():
                    v1 = c1[m1].ravel()
                    cc_perim += np.bincount(v1, minlength=n_cc + 1)
                    ext1 = (~h2[m1]).ravel()
                    if ext1.any():
                        cc_ext_nonhair += np.bincount(v1[ext1], minlength=n_cc + 1)
                if m2.any():
                    v2 = c2[m2].ravel()
                    cc_perim += np.bincount(v2, minlength=n_cc + 1)
                    ext2 = (~h1[m2]).ravel()
                    if ext2.any():
                        cc_ext_nonhair += np.bincount(v2[ext2], minlength=n_cc + 1)
            # Also count image border pixels of strip components
            for edge_c in (cc_map[0, :], cc_map[-1, :], cc_map[:, 0], cc_map[:, -1]):
                em = edge_c > 0
                if em.any():
                    cc_perim += np.bincount(edge_c[em], minlength=n_cc + 1)

            cc_perim_1x = np.maximum(cc_perim / float(max(1, scale)), 1.0)
            cc_mean_w_1x = (2.0 * cc_area_1x) / cc_perim_1x
            cc_elong = (cc_perim_1x * cc_perim_1x) / np.maximum(16.0 * cc_area_1x, 1e-6)
            cc_ext_frac = cc_ext_nonhair / np.maximum(cc_perim, 1.0)

            dt_strip = distance_transform_edt(strip_px)
            cc_max_half = np.zeros(n_cc + 1, dtype=np.float32)
            np.maximum.at(cc_max_half, cc_map[cc_valid], dt_strip[cc_valid])
            cc_max_w_1x = 2.0 * (cc_max_half + 0.5) / float(max(1, scale))

            # Map each candidate region to its strip component (1-to-1 since regions are connected)
            cand_regs = np.flatnonzero(cand_strip)
            reg_cc = np.zeros(n_in, dtype=np.int32)
            reg_cc[regions[cc_valid]] = cc_map[cc_valid]
            cc_members: dict[int, list[int]] = {}
            for ri in cand_regs.tolist():
                ci = int(reg_cc[ri])
                if ci > 0:
                    cc_members.setdefault(ci, []).append(ri)

            fused_reg_ids: list[int] = []
            for ci, mems in cc_members.items():
                if len(mems) < 2:
                    continue
                if float(cc_area_1x[ci]) > max_strip_area_1x:
                    continue
                # Strip geometry & proportional mean-width check:
                # True strip (elongation >= 2.2) AND mean width <= 0.48% AND peak width <= 0.64%
                if float(cc_mean_w_1x[ci]) > max_strip_w_1x or float(cc_max_w_1x[ci]) > max_strip_peak_1x:
                    continue
                if float(cc_elong[ci]) < 2.2:
                    continue

                m_arr = np.asarray(mems, dtype=np.int32)
                m_labs = final_lab[m_arr]
                ab_span = float(np.linalg.norm(m_labs[:, 1:3].max(axis=0) - m_labs[:, 1:3].min(axis=0)))
                if ab_span > 0.095:
                    continue
                l_span = float(m_labs[:, 0].max() - m_labs[:, 0].min())
                l_min_c = float(m_labs[:, 0].min())
                c_max_c = float(r_chroma[m_arr].max())

                is_free_wisp = float(cc_ext_frac[ci]) >= 0.25
                if is_free_wisp and l_min_c <= 0.42 and c_max_c <= 0.075:
                    # Free-standing dark/charcoal curling hair wisp against contrasting bg/skin:
                    # average dark ink + gray core/edge into ONE silky monochrome hair strand!
                    if l_span > 0.62:
                        continue
                else:
                    # Interior closed hair filament or colored thin line: require closer lightness
                    has_l = bool(is_line[m_arr].any())
                    has_f = bool((~is_line[m_arr]).any())
                    if (not (has_l and has_f) and not is_free_wisp) or l_span > 0.18:
                        continue

                # Area-weighted mean RGB (均色处理) across the entire hair strand component!
                m_areas = r_areas[m_arr]
                mean_rgb = np.clip(
                    np.rint((final_rgb[m_arr].astype(np.float64) * m_areas[:, None]).sum(axis=0) / max(float(m_areas.sum()), 1.0)),
                    0,
                    255,
                ).astype(np.uint8)
                rep_r = int(m_arr[int(np.argmax(m_areas))])
                keep_part = int(part_ids[rep_r])
                for ri in mems:
                    final_rgb[ri] = mean_rgb
                    parent[ri] = rep_r
                    is_line[ri] = True
                    part_ids[ri] = keep_part
                fused_reg_ids.extend(mems)
                strand_cores_fused += len(mems)

            if fused_reg_ids:
                fr_mask = np.zeros(n_in, dtype=bool)
                fr_mask[fused_reg_ids] = True
                line_mask[vis] |= fr_mask[regions[vis]]

    # Stage 2F: Pure-AA 1px Boundary Fringe Cleanup ("边界去毛边")
    # Eliminates ragged 1px linear anti-aliasing transition slivers along crisp high-contrast edges
    # (e.g. horizontal window frames against bright sky where a 1px mid-gray AA fringe sits strictly
    # on the linear blend segment between dark bar `A` and white sky `B`).
    # Real drawn outlines are extrema (`L < min(L_A, L_B)` -> `t < 0` or `perp > 0.032`) or high-ink
    # (`stroke_hi >= 0.55`), so genuine anime lines & hair strands are NEVER affected!
    aa_fringes_cleaned = 0
    r_stroke_mean = (
        np.bincount(regions[vis].ravel(), weights=stroke_hi[vis].ravel(), minlength=n_in)
        / np.maximum(r_areas, 1.0)
    ).astype(np.float32)
    aa_lab = srgb_to_oklab(final_rgb).astype(np.float32)
    # Build region-to-fill border contact counts for candidate ultra-thin AA slivers
    max_aa_w_1x = max(1.9, 0.0018 * vis_linear_1x)
    max_aa_mean_w_1x = max(1.55, 0.0014 * vis_linear_1x)
    aa_cand = (
        (r_max_width_1x <= max_aa_w_1x)
        & (r_mean_width_1x <= max_aa_mean_w_1x)
        & (r_stroke_mean < 0.80)
        & (part_ids != PART_EYES_MOUTH)
        & (part_ids != PART_FACE_SKIN)
        & (part_ids != PART_HEADWEAR)
        & (part_ids != PART_HAIR)
    )
    if aa_cand.any():
        aa_pairs = []
        for a, b in ((regions[:-1], regions[1:]), (regions[:, :-1], regions[:, 1:])):
            diff = (a != b) & (a >= 0) & (b >= 0)
            if not diff.any():
                continue
            va, vb = a[diff], b[diff]
            ca = aa_cand[va] & (~is_line[vb]) & (~aa_cand[vb])
            cb = aa_cand[vb] & (~is_line[va]) & (~aa_cand[va])
            if ca.any():
                aa_pairs.append((va[ca].astype(np.int64) << 32) | vb[ca].astype(np.int64))
            if cb.any():
                aa_pairs.append((vb[cb].astype(np.int64) << 32) | va[cb].astype(np.int64))
        if aa_pairs:
            u_aa, c_aa = np.unique(np.concatenate(aa_pairs), return_counts=True)
            aa_s = (u_aa >> 32).astype(np.int32)
            aa_d = (u_aa & 0xFFFFFFFF).astype(np.int32)
            top1_nb = np.full(n_in, -1, dtype=np.int32)
            top1_c = np.zeros(n_in, dtype=np.int64)
            top2_nb = np.full(n_in, -1, dtype=np.int32)
            top2_c = np.zeros(n_in, dtype=np.int64)
            for si, di, ci in zip(aa_s.tolist(), aa_d.tolist(), c_aa.tolist()):
                if ci > top1_c[si]:
                    top2_nb[si] = top1_nb[si]
                    top2_c[si] = top1_c[si]
                    top1_nb[si] = di
                    top1_c[si] = ci
                elif ci > top2_c[si]:
                    top2_nb[si] = di
                    top2_c[si] = ci

            tot_b_aa = np.maximum(b_cnt, 1.0)
            cleaned_ids: list[int] = []
            for si in np.flatnonzero(aa_cand).tolist():
                a_nb = int(top1_nb[si])
                b_nb = int(top2_nb[si])
                if a_nb < 0 or b_nb < 0:
                    continue
                # Must sit cleanly between two fill sides covering >= 55% of its perimeter
                if (float(top1_c[si] + top2_c[si]) / float(tot_b_aa[si])) < 0.55:
                    continue
                if float(top2_c[si]) < 0.18 * float(tot_b_aa[si]):
                    continue
                la = aa_lab[a_nb]
                lb = aa_lab[b_nb]
                v = lb - la
                d_ab = float(np.linalg.norm(v))
                if d_ab < 0.12:
                    continue
                ls = aa_lab[si]
                t = float(np.dot(ls - la, v)) / (d_ab * d_ab)
                if not (0.10 <= t <= 0.90):
                    continue
                perp = float(np.linalg.norm(ls - (la + t * v)))
                if perp > 0.032:
                    continue
                # Pure linear AA transition sliver! Snap cleanly to the closer side (`A` or `B`)
                tgt = a_nb if t < 0.50 else b_nb
                final_rgb[si] = final_rgb[tgt]
                parent[si] = parent[tgt]
                is_line[si] = False
                part_ids[si] = part_ids[tgt]
                cleaned_ids.append(si)
                aa_fringes_cleaned += 1

            if cleaned_ids:
                cl_mask = np.zeros(n_in, dtype=bool)
                cl_mask[cleaned_ids] = True
                line_mask[vis] &= ~cl_mask[regions[vis]]

    # Gentle imperceptible-duplicate palette snap (never merges distinct hair locks or eyes!)
    roots_uniq, root_inv = np.unique(parent, return_inverse=True)
    n_roots = len(roots_uniq)
    root_areas = np.bincount(root_inv, weights=r_areas, minlength=n_roots)
    root_is_line = is_line[roots_uniq]
    root_part = part_ids[roots_uniq]
    root_rgb = final_rgb[roots_uniq].copy()
    root_lab = srgb_to_oklab(root_rgb).astype(np.float32)

    pal_thr = {
        PART_EYES_MOUTH: 0.006,
        PART_FACE_SKIN: 0.008,
        PART_HEADWEAR: 0.010,
        PART_NECK_HAND: 0.011,
        PART_OBJECTS: 0.011,
        PART_CLOTH: 0.012,
        PART_BG: 0.013,
        PART_HAIR: 0.011,
    }
    for grp_line in (False, True):
        idxs = np.flatnonzero(root_is_line == grp_line)
        if len(idxs) <= 1:
            continue
        idxs = idxs[np.argsort(-root_areas[idxs])]
        rep_labs: list[np.ndarray] = []
        rep_rgbs: list[np.ndarray] = []
        for ri in idxs.tolist():
            lv = root_lab[ri]
            t_p = 0.014 if grp_line else pal_thr.get(int(root_part[ri]), 0.011)
            if rep_labs:
                arr_l = np.asarray(rep_labs, dtype=np.float32)
                dists = np.linalg.norm(arr_l - lv, axis=1)
                best = int(np.argmin(dists))
                if float(dists[best]) <= t_p:
                    root_rgb[ri] = rep_rgbs[best]
                    continue
            rep_labs.append(lv)
            rep_rgbs.append(root_rgb[ri].copy())

    unified_rgb = root_rgb[root_inv]
    palette, col_ids = np.unique(unified_rgb, axis=0, return_inverse=True)
    pal_labs = srgb_to_oklab(palette).astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 3: Pixel-Exact Voronoi Medial-Axis Same-Color Line Erasure ("对同色色块的边界线抹除")
    # ------------------------------------------------------------------
    # Project recolored fill keys `(color_id * 16 + part_id + 1)` onto every line pixel via the
    # exact EDT Voronoi nearest-fill map (`vor_reg`). Checking a tight 3x3 window on `vor_fkey`
    # inspects ONLY the immediate opposite fills across a line (zero overshoot across 2px slivers
    # in sakuragirl's bun, and automatic T-junction lock where 3 fills or 2 colors meet!).
    fkey_reg = (col_ids.astype(np.int32) * 16 + part_ids.astype(np.int32) + 1)
    fill_vis = vis & (~is_line[np.maximum(regions, 0)])
    line_vis = vis & (~fill_vis)
    erased_line_px = 0
    erased_hair_px = 0
    field_key = np.zeros((hs, ws), dtype=np.int32)
    if fill_vis.any():
        field_key[fill_vis] = fkey_reg[regions[fill_vis]] * 2  # even = fill
    if line_vis.any():
        field_key[line_vis] = fkey_reg[regions[line_vis]] * 2 + 1  # odd = line

    if line_vis.any() and fill_vis.any() and vor_reg is not None and dist_to_fill is not None:
        vor_fkey = fkey_reg[np.maximum(vor_reg, 0)].astype(np.float64)
        k3 = np.ones((3, 3), dtype=np.uint8)
        # On a line ribbon, a pixel's 3x3 Voronoi neighborhood has min == max iff BOTH immediate
        # opposite sides (within thin-line radius <= 3.2*scale) share the exact same (color, part)!
        # Note: to bridge line thickness cleanly, propagate across the full ribbon via a 2-step
        # check: pixels touching two different Voronoi keys (the medial axis of a kept boundary)
        # form a protected core that dilates back across the ribbon width so kept lines NEVER split!
        vor_max = np.rint(cv2.dilate(vor_fkey, k3)).astype(np.int32)
        vor_min = np.rint(cv2.erode(vor_fkey, k3)).astype(np.int32)
        diff_axis = line_vis & (vor_min != vor_max)
        # Protect full width of every kept different-color/different-subject boundary line + T-junctions.
        # Inside congested dense hair buns (frag_dens >= 0.28), use a tight 1px guard around diff_axis
        # so 2px clutter lines between now-same-color dark slivers actually dissolve without being
        # blocked by a 5px shadow from a highlight several pixels away!
        prot_r = max(3, int(round(min(3.5, thin_line_half_1x * 0.80) * float(max(1, scale)))))
        prot_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * prot_r + 1, 2 * prot_r + 1))
        kept_line_zone_wide = cv2.dilate(diff_axis.astype(np.uint8), prot_disk) > 0
        kept_line_zone_tight = cv2.dilate(diff_axis.astype(np.uint8), k3) > 0

        vor_part_img = (np.rint(vor_fkey).astype(np.int32) - 1) % 16
        is_bun_px = (vor_part_img == PART_HAIR) & (is_compact_hair | (frag_dens_px >= 0.28))
        kept_line_zone = np.where(is_bun_px, kept_line_zone_tight, kept_line_zone_wide)

        same_side = line_vis & (~kept_line_zone) & (dist_to_fill <= thin_line_half_px)
        if has_trans:
            near_trans = cv2.dilate((~vis).astype(np.uint8), prot_disk) > 0
            same_side &= ~near_trans

        if same_side.any():
            ss_y, ss_x = np.where(same_side)
            shared_key_plus1 = np.rint(vor_fkey[ss_y, ss_x]).astype(np.int32)
            shared_fkey = shared_key_plus1 - 1
            shared_col_id = shared_fkey // 16
            shared_part = shared_fkey % 16

            line_col_id = col_ids[regions[ss_y, ss_x]]
            de_lf_px = np.linalg.norm(pal_labs[line_col_id] - pal_labs[shared_col_id], axis=1)
            line_L_px = pal_labs[line_col_id, 0]
            shared_L_px = pal_labs[shared_col_id, 0]
            st_px = stroke_hi[ss_y, ss_x]
            fd_px = frag_dens_px[ss_y, ss_x]

            # Dominant hair median color (chroma-weighted OKLab) so hair-bun erasure ONLY fires on
            # genuine hair tones — NEVER erases dark curling wisps over pale peach neck skin AND
            # NEVER flattens dark crimson inner peony petal lines!
            if hair_lab_w_med is not None:
                pal_labs_w = pal_labs.copy()
                pal_labs_w[:, 1] *= 2.2
                pal_labs_w[:, 2] *= 2.2
                de_hair_med_w = np.linalg.norm(pal_labs_w[shared_col_id] - hair_lab_w_med[None, :], axis=1)
                is_true_hair_fill = de_hair_med_w <= 0.22
            else:
                is_true_hair_fill = np.ones(len(ss_y), dtype=bool)

            # Per-subject erasure contrast guard:
            # - eyes/face/headwear: only ultra-low-contrast non-ink micro-seams (<= 0.018)
            # - compact hair bun (sakuragirl 1.0%) or congested hair tangle: erase redundant
            #   interior hair clutter lines ONLY when surrounding fill is genuine hair tone AND
            #   the line is NOT a crisp dark wisp drawn over lighter neck/skin (`line_L >= shared_L - 0.12`)!
            # - spaced large hair (Hu Tao 16.6%): NEVER break intentional drawn ink strands!
            # - cloth/bg/neck/objects: only ultra-low-contrast non-ink noise seams (<= 0.022)
            ok_prot = ((shared_part == PART_EYES_MOUTH) | (shared_part == PART_FACE_SKIN) | (shared_part == PART_HEADWEAR)) & (de_lf_px <= 0.018) & (st_px < 0.35)
            not_dark_wisp_on_light = line_L_px >= (shared_L_px - 0.12)
            ok_hair_bun = (
                (shared_part == PART_HAIR)
                & (bool(is_compact_hair) | (fd_px >= 0.28))
                & is_true_hair_fill
                & not_dark_wisp_on_light
                & (de_lf_px <= (0.18 if is_compact_hair else 0.135))
            )
            ok_hair_spaced = (shared_part == PART_HAIR) & (not is_compact_hair) & (fd_px < 0.28) & (de_lf_px <= 0.024) & (st_px < 0.35)
            ok_other = (
                (shared_part == PART_CLOTH)
                | (shared_part == PART_BG)
                | (shared_part == PART_NECK_HAND)
                | (shared_part == PART_OBJECTS)
            ) & (de_lf_px <= 0.022) & (st_px < 0.35)

            do_erase = np.asarray(ok_prot | ok_hair_bun | ok_hair_spaced | ok_other, dtype=bool)
            if do_erase.any():
                ey, ex = ss_y[do_erase], ss_x[do_erase]
                erased_line_px = int(len(ey))
                erased_hair_px = int((shared_part[do_erase] == PART_HAIR).sum())
                line_mask[ey, ex] = False
                # Re-assign erased line pixels directly to the shared fill key (even = fill)
                # so connected-component relabeling fuses both sides + erased line into 1 polygon!
                field_key[ey, ex] = shared_key_plus1[do_erase] * 2

    # ------------------------------------------------------------------
    # Stage 4: Order-1 Connected-Component Relabeling
    # ------------------------------------------------------------------
    if has_trans:
        new_regions = label(np.where(vis, field_key, 0), connectivity=1, background=0).astype(np.int32) - 1
        n_out = int(new_regions.max()) + 1
        new_colors = np.zeros((max(0, n_out), 3), np.uint8)
        new_parts = np.zeros(max(0, n_out), dtype=np.int8)
        new_is_line = np.zeros(max(0, n_out), dtype=bool)
        if n_out > 0:
            rv = new_regions >= 0
            fk_rv = field_key[rv]
            cid_rv = ((fk_rv // 2) - 1) // 16
            new_colors[new_regions[rv]] = palette[cid_rv]
            new_parts[new_regions[rv]] = ((fk_rv // 2) - 1) % 16
            new_is_line[new_regions[rv]] = (fk_rv & 1) == 1
    else:
        new_regions = label(field_key, connectivity=1).astype(np.int32) - 1
        n_out = int(new_regions.max()) + 1
        new_colors = np.empty((n_out, 3), np.uint8)
        new_parts = np.empty(n_out, dtype=np.int8)
        new_is_line = np.empty(n_out, dtype=bool)
        cid_all = ((field_key // 2) - 1) // 16
        new_colors[new_regions.ravel()] = palette[cid_all.ravel()]
        new_parts[new_regions.ravel()] = (((field_key // 2) - 1) % 16).ravel()
        new_is_line[new_regions.ravel()] = ((field_key & 1) == 1).ravel()

    # ------------------------------------------------------------------
    # Stage 5: Order-2 Post-Merge Cleanup ("二阶并色": recomputes areas & adjacency on Order-1
    # merged polygons to cleanly absorb tiny newly-isolated specks—fixes green-circle bun dots!)
    # ------------------------------------------------------------------
    order2_absorbed = 0
    if n_out > 1:
        vis2 = new_regions >= 0
        a2 = np.bincount(new_regions[vis2].ravel(), minlength=n_out).astype(np.float64)
        a2_1x = a2 / scale2
        labs2 = srgb_to_oklab(new_colors).astype(np.float32)
        labs2_w = labs2.copy()
        labs2_w[:, 1] *= 2.2
        labs2_w[:, 2] *= 2.2

        # Adjacency & border contact counts on the Order-1 fused polygons
        kp2 = []
        for p1, p2 in ((new_regions[:-1], new_regions[1:]), (new_regions[:, :-1], new_regions[:, 1:])):
            diff = (p1 != p2) & (p1 >= 0) & (p2 >= 0)
            if not diff.any():
                continue
            va = p1[diff].astype(np.int64)
            vb = p2[diff].astype(np.int64)
            kp2.append((va << 32) | vb)
            kp2.append((vb << 32) | va)
        if kp2:
            uk2, cw2 = np.unique(np.concatenate(kp2), return_counts=True)
            s2 = (uk2 >> 32).astype(np.int32)
            d2 = (uk2 & 0xFFFFFFFF).astype(np.int32)
            tot_b2 = np.bincount(s2, weights=cw2, minlength=n_out).astype(np.float64)

            best_nb = np.full(n_out, -1, dtype=np.int32)
            best_cw = np.zeros(n_out, dtype=np.int64)
            for si, di, ci in zip(s2.tolist(), d2.tolist(), cw2.tolist()):
                # Only absorb into a strictly much-larger same-subject neighbor (never into headwear/eyes!)
                if new_parts[si] != new_parts[di]:
                    continue
                if new_is_line[di]:
                    continue
                if ci > best_cw[si]:
                    best_cw[si] = ci
                    best_nb[si] = di

            remap2 = np.arange(n_out, dtype=np.int32)
            o2_bun_area_cap = max(12.0, 0.000020 * total_vis_1x)
            o2_hair_area_cap = max(6.0, 0.000012 * total_vis_1x)
            o2_other_area_cap = max(4.0, 0.000008 * total_vis_1x)
            for si in range(n_out):
                di = int(best_nb[si])
                if di < 0:
                    continue
                p_si = int(new_parts[si])
                # Never absorb eyes/mouth, face skin, or pearl/peony headwear!
                if p_si in (PART_EYES_MOUTH, PART_FACE_SKIN, PART_HEADWEAR):
                    continue
                nb_frac = float(best_cw[si]) / max(float(tot_b2[si]), 1.0)
                area_ratio_sd = float(a2_1x[di]) / max(float(a2_1x[si]), 0.1)
                de_w = float(np.linalg.norm(labs2_w[si] - labs2_w[di]))
                # Inside compact hair bun: swallow any remaining tiny micro-speck/residual hair
                # island (proportional area ratio, >= 58% surrounded by a >= 4x larger hair mass)
                if p_si == PART_HAIR and is_compact_hair:
                    if a2_1x[si] <= o2_bun_area_cap and nb_frac >= 0.58 and area_ratio_sd >= 4.0 and de_w <= 0.19:
                        remap2[si] = di
                        order2_absorbed += 1
                elif p_si == PART_HAIR:
                    if (not new_is_line[si]) and a2_1x[si] <= o2_hair_area_cap and nb_frac >= 0.72 and area_ratio_sd >= 6.0 and de_w <= 0.045:
                        remap2[si] = di
                        order2_absorbed += 1
                else:
                    if (not new_is_line[si]) and a2_1x[si] <= o2_other_area_cap and nb_frac >= 0.75 and area_ratio_sd >= 8.0 and de_w <= 0.032:
                        remap2[si] = di
                        order2_absorbed += 1

            if order2_absorbed > 0:
                # Resolve chains in remap2
                for si in range(n_out):
                    t = si
                    while remap2[t] != t:
                        t = remap2[t]
                    remap2[si] = t
                # Update line_mask for any absorbed hair micro-line speck
                abs_mask = remap2 != np.arange(n_out, dtype=np.int32)
                if (abs_mask & new_is_line).any():
                    line_mask[vis2] &= ~(abs_mask & new_is_line)[new_regions[vis2]]
                col2 = new_colors[remap2]
                pal2, cid2 = np.unique(col2, axis=0, return_inverse=True)
                line2 = new_is_line & (~abs_mask)
                key2 = cid2.astype(np.int32) * 32 + new_parts[remap2].astype(np.int32) * 2 + line2.astype(np.int32) + 1
                if has_trans:
                    f2 = np.zeros_like(new_regions)
                    f2[vis2] = key2[new_regions[vis2]]
                    new_regions = label(f2, connectivity=1, background=0).astype(np.int32) - 1
                    n_out = int(new_regions.max()) + 1
                    new_colors = np.zeros((max(0, n_out), 3), np.uint8)
                    if n_out > 0:
                        rv2 = new_regions >= 0
                        new_colors[new_regions[rv2]] = pal2[(f2[rv2] - 1) // 32]
                else:
                    f2 = key2[new_regions]
                    new_regions = label(f2, connectivity=1).astype(np.int32) - 1
                    n_out = int(new_regions.max()) + 1
                    new_colors = np.empty((n_out, 3), np.uint8)
                    new_colors[new_regions.ravel()] = pal2[((f2 - 1) // 32).ravel()]

    colors_after = int(len(np.unique(new_colors, axis=0))) if len(new_colors) > 0 else 0
    quota_report = {PART_NAMES[p]: f"{part_unique_n[p]}/{part_k_max[p]}" for p in PART_BUDGET if part_unique_n[p] > 0}
    return new_regions, new_colors, {
        "before": n_in,
        "after": int(len(new_colors)),
        "merges_direct": merges_direct,
        "merges_cross": merges_cross,
        "merges_quota": merges_quota,
        "line_merges": line_merges,
        "strand_cores_fused": strand_cores_fused,
        "aa_fringes_cleaned": aa_fringes_cleaned,
        "order2_absorbed": order2_absorbed,
        "erased_line_px": erased_line_px,
        "erased_hair_px": erased_hair_px,
        "colors_before": colors_before,
        "colors_after": colors_after,
        "quotas": quota_report,
        "parts": part_stats,
    }


def trace_binary_ink(img: Image.Image, rgb: np.ndarray, outline_layer: bool):
    """Trace exact pure black/white pixel artwork without deleting pixels or filling holes."""
    if "A" in img.getbands() or "transparency" in img.info:
        alpha = np.asarray(img.convert("RGBA"))[..., 3]
        if not np.all((alpha == 0) | (alpha == 255)):
            return None
        visible = alpha == 255
    else:
        visible = np.ones(rgb.shape[:2], bool)
    black = np.all(rgb == 0, axis=2) & visible
    white = np.all(rgb == 255, axis=2) & visible
    if np.any(visible & ~(black | white)):
        return None
    if not visible.any():
        h, w = rgb.shape[:2]
        empty = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"/>'
        return empty, img.convert("RGBA")
    regions = label(black.astype(np.uint8) + 2 * white, connectivity=1, background=0).astype(np.int32) - 1
    palette = np.empty(regions.max() + 1, dtype=bool)
    palette[regions[visible]] = white[visible]
    svg, _ = trace_regions(regions, np.where(palette, "#ffffff", "#000000"), exact_grid=True)
    if outline_layer:
        start = svg.index(">") + 1
        svg = svg[:start] + '<g id="region-boundaries"/><g id="region-colours">' + svg[start:-6] + "</g></svg>"
    return svg, img.convert("RGB" if visible.all() else "RGBA")


def _vectorize_clean_planar(
    source: np.ndarray,
    fmaps,
    orig_shape: tuple[int, int],
    *,
    alpha_1x: np.ndarray | None = None,
    smooth_lines: bool = True,
    refit_outline: bool = True,
    locate_boundaries: bool = True,
    outline_layer: bool = True,
    color_merge: bool = True,
    core_mode: bool = False,
    progress=None,
):
    """Run lossless planar reconstruction directly at SR resolution with alpha transparency preservation."""
    step = progress or (lambda *_: None)
    h_orig, w_orig = orig_shape
    h, w = source.shape[:2]
    if h == h_orig and w == w_orig and (h * w) <= 4_000_000:
        sr_in = cv2.resize(source, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4 if core_mode else cv2.INTER_CUBIC)
    else:
        sr_in = source
    hs, ws = sr_in.shape[:2]

    visible_hi = None
    if alpha_1x is not None:
        alpha_hi = alpha_1x if alpha_1x.shape == (hs, ws) else cv2.resize(alpha_1x, (ws, hs), interpolation=cv2.INTER_LINEAR)
        visible_hi = alpha_hi >= 128

    step(0.45 if core_mode else 0.25, "核心线稿骨架提取 + 平涂色区分割" if core_mode else "Stage 2->3：无损线条骨架提取 + 保护区解锁 + 渐变色阶轮廓划分")
    regions, colors, line_mask, p_stats = build_planar_partition(
        sr_in, fmaps, (h_orig, w_orig), core_mode=core_mode, visible_mask=visible_hi
    )
    print("[planar_partition]", p_stats, flush=True)

    sr_int = max(1, int(round(hs / max(1, h_orig))))
    regions, colors = compact_fragments(regions, colors, scale=sr_int)
    if color_merge:
        step(0.62, "主体语义分割 + 比例色彩合并、细发丝均色与边界去毛边")
        regions, colors, m_stats = merge_perceptual_regions(
            regions,
            colors,
            line_mask,
            fmaps=fmaps,
            scale=sr_int,
            core_mode=core_mode,
        )
        print("[perceptual_merge]", m_stats, flush=True)
    else:
        step(0.62, "已关闭色彩合并：保留全量原始分区色彩")
        print("[perceptual_merge] skipped (color_merge=False)", flush=True)

    step(0.80 if core_mode else 0.76, "核心阶段：SVG 共享矢量边界拟合" if core_mode else "Stage 4：SVG共享矢量边界拟合")
    fitting = {}
    corridor = None
    metadata = {
        "data-planar-engine": "core-ai-v1" if core_mode else "v1",
        "data-raw-compartments": str(p_stats.get("n_regions_raw", len(colors))),
        "data-color-merge": "1" if color_merge else "0",
    }
    if refit_outline:
        fitting = dict(
            structural_fitter=partial(fit_compact_outline, smooth_lines=smooth_lines),
            secondary_fitter=partial(refit_sketch_chain, smooth_lines=smooth_lines),
            secondary_network=partial(smooth_outline_junctions, smooth_lines=smooth_lines),
        )
        if locate_boundaries:
            corridor = SourceCorridor(sr_in, colors)
            if core_mode:
                fitting.update(chain_locator=corridor, source_scale=1)
            else:
                gc95_hi = cv2.resize(fmaps.grad_core_95.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0
                vf0 = fmaps.line_vector_fields[0]
                vf0_hi = vf0 if vf0.shape == (hs, ws) else cv2.resize(vf0, (ws, hs), interpolation=cv2.INTER_LINEAR)

                def _grad_guarded_corridor(pts, right, left, sc):
                    if min(right, left) >= 0 and len(pts) >= 4:
                        ys_c = np.clip(np.round(pts[:, 1]).astype(int), 0, hs - 1)
                        xs_c = np.clip(np.round(pts[:, 0]).astype(int), 0, ws - 1)
                        if float(np.mean(gc95_hi[ys_c, xs_c])) > 0.75 and float(np.max(vf0_hi[ys_c, xs_c])) < 0.24:
                            return pts
                    return corridor(pts, right, left, sc)

                fitting.update(chain_locator=_grad_guarded_corridor, source_scale=1)

    chain_scale_fn = None
    if (not core_mode) and smooth_lines and hasattr(fmaps, "delicate_detail"):
        del_hi = fmaps.delicate_detail if fmaps.delicate_detail.shape == (hs, ws) else cv2.resize(fmaps.delicate_detail, (ws, hs), interpolation=cv2.INTER_LINEAR)
        wash_hi = fmaps.mergeable_wash if fmaps.mergeable_wash.shape == (hs, ws) else cv2.resize(fmaps.mergeable_wash, (ws, hs), interpolation=cv2.INTER_LINEAR)

        def _chain_scale(pts, base_scale, s_scale):
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, hs - 1)
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, ws - 1)
            if float(np.mean(del_hi[ys, xs])) > 0.35:
                return base_scale * 0.85
            if float(np.mean(wash_hi[ys, xs])) > 0.48:
                return base_scale * 1.40
            return base_scale

        chain_scale_fn = _chain_scale

    tol = 0.36 if smooth_lines else 0.28
    smoothing = 0.95 if smooth_lines else 0.75
    struct_sc = 1.35 if smooth_lines else 1.15
    svg, stats = trace_regions(
        regions,
        ["#%02x%02x%02x" % tuple(c) for c in colors],
        tol=tol,
        seam=0,
        geometry_guard=True,
        smoothing=smoothing,
        structural_scale=struct_sc,
        compact_commands=True,
        chain_scale_fn=chain_scale_fn,
        **fitting,
    )
    svg = re.sub(r'width="\d+" height="\d+"', f'width="{w_orig}" height="{h_orig}"', svg, count=1)

    if outline_layer and len(colors) > 0:
        step(0.96, "生成教师SVG结构（gap-filler轮廓 + 线稿轮廓约束单色填充层）")
        vis = regions >= 0
        r_areas = np.bincount(regions[vis].ravel(), minlength=len(colors))
        r_lm = np.bincount(
            regions[vis].ravel(),
            weights=line_mask[vis].astype(np.float32).ravel(),
            minlength=len(colors),
        ) / np.maximum(r_areas, 1)
        line_region_ids = set(np.flatnonzero(r_lm >= 0.55).tolist())
        metadata["data-line-art-paths"] = str(len(line_region_ids))
        metadata["data-color-fill-paths"] = str(len(colors) - len(line_region_ids))
        metadata["data-segments-before-refit"] = str(stats.get("segments_before_secondary", 0))
        metadata["data-segments-after-refit"] = str(stats.get("curve_segments", 0))
        if corridor is not None:
            metadata["data-source-chains-considered"] = str(corridor.considered)
            metadata["data-source-chains-moved"] = str(corridor.moved)
        svg = boundary_then_fill(svg, hidden_strokes=(), metadata=metadata, line_region_ids=line_region_ids)

    if visible_hi is not None:
        prev_rgba = np.zeros((hs, ws, 4), np.uint8)
        vis = regions >= 0
        if len(colors) > 0 and vis.any():
            prev_rgba[vis, :3] = colors[regions[vis]]
            prev_rgba[vis, 3] = 255
        preview = Image.fromarray(prev_rgba, mode="RGBA").resize((w_orig, h_orig), Image.Resampling.LANCZOS)
    else:
        preview = Image.fromarray(colors[regions]).resize((w_orig, h_orig), Image.Resampling.LANCZOS)

    step(1.0, "完成")
    return svg, preview


def vectorize_print(
    img: Image.Image,
    bands: int = 112,
    recover_strands: bool = True,
    progress=None,
    outline_layer: bool = True,
    gradient_mode: str = "clean",
    refit_outline: bool = True,
    redraw_source: bool = False,
    smooth_lines: bool = True,
    feature_routing: bool = True,
    color_merge: bool = True,
    sr_scale: float | None = None,
    *,
    locate_boundaries: bool = True,
):
    """Main CNN two-layer planar vectorization entry point with full transparency preservation."""
    step = progress or (lambda *_: None)

    source, alpha_1x = _prepare_alpha_source(img)
    h, w = source.shape[:2]
    h_orig, w_orig = h, w

    exact = trace_binary_ink(img, source, outline_layer)
    if exact is not None:
        step(1.0, "完成：保留纯黑白像素边界")
        return exact

    from .feature_routing import infer_feature_maps, render_vector_heatmap_png

    step(0.004, "整图倍率放大 + 去杂色平滑锐化 + CNN 热力线条识别")
    fmaps = infer_feature_maps(source, sr_scale=sr_scale, core_mode=False)
    if fmaps is None or getattr(fmaps, "line_vector_fields", None) is None:
        raise RuntimeError("缺少核心 CNN ONNX 权重（weights/line_vector_net.onnx）")

    print("[feature_routing]", fmaps.summary, flush=True)
    hm_out = os.environ.get("LUMINA_HEATMAP_OUT", "").strip()
    if hm_out:
        try:
            render_vector_heatmap_png(source, hm_out, vec_fields=fmaps.line_vector_fields)
        except Exception:
            pass

    if getattr(fmaps, "sr_hi_rgb", None) is not None:
        source = fmaps.sr_hi_rgb
        h, w = source.shape[:2]
        fmaps = replace(
            fmaps,
            ink_main_geom=cv2.resize(fmaps.ink_main_geom, (w, h), interpolation=cv2.INTER_LINEAR),
            fine_strand=cv2.resize(fmaps.fine_strand, (w, h), interpolation=cv2.INTER_LINEAR),
            delicate_detail=cv2.resize(fmaps.delicate_detail, (w, h), interpolation=cv2.INTER_LINEAR),
            mergeable_wash=cv2.resize(fmaps.mergeable_wash, (w, h), interpolation=cv2.INTER_LINEAR),
            surface_probs=np.stack(
                [cv2.resize(fmaps.surface_probs[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)],
                axis=0,
            ),
            line_net_maps=(
                np.stack(
                    [cv2.resize(fmaps.line_net_maps[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)],
                    axis=0,
                )
                if fmaps.line_net_maps is not None
                else None
            ),
            elem_grad_maps=(
                np.stack(
                    [cv2.resize(fmaps.elem_grad_maps[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(8)],
                    axis=0,
                )
                if fmaps.elem_grad_maps is not None
                else None
            ),
            grad_core_95=(
                cv2.resize(fmaps.grad_core_95.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if getattr(fmaps, "grad_core_95", None) is not None
                else None
            ),
            line_buffer_5=(
                cv2.resize(fmaps.line_buffer_5.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if getattr(fmaps, "line_buffer_5", None) is not None
                else None
            ),
            sr_clean_1x=source,
        )
    elif fmaps.sr_clean_1x is not None:
        source = fmaps.sr_clean_1x

    return _vectorize_clean_planar(
        source,
        fmaps,
        (h_orig, w_orig),
        alpha_1x=alpha_1x,
        smooth_lines=smooth_lines,
        refit_outline=refit_outline,
        locate_boundaries=locate_boundaries,
        outline_layer=outline_layer,
        color_merge=color_merge,
        core_mode=False,
        progress=progress,
    )
