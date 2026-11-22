"""Vector-Field & Bezier-Wall Constrained Coloring Layer Engine (`#region-colours`).

Builds a watertight, anti-aliasing-free color-fill partition strictly at the native 2x SR
vector coordinate grid bounded by `LineVectorNet` (`stroke_prob`, `color_bound`) AND the
actual traced `#line-art-layer` Bezier centerlines, eliminating cross-line color bleeds
and dark ink blotches inside compartments.
"""
from __future__ import annotations

from functools import partial
import re

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label

from .color import srgb_to_oklab
from .outline_stages import snap_gray
from .region_merge import merge_transition_patches
from .shared_regions import trace_regions
from .structural_curves import fit_compact_outline, refit_sketch_chain, smooth_outline_junctions


def _propagate_interior_inside_compartments(
    rgb: np.ndarray,
    wall_mask: np.ndarray,
    core_mask: np.ndarray,
    comp_lbl: np.ndarray,
    passes: int = 3,
) -> np.ndarray:
    """Propagate clean `core_mask` interior colors outward into AA-rim pixels strictly within
    each 4-connected compartment ID (never crossing `wall_mask`)."""
    h, w = rgb.shape[:2]
    comp = ~wall_mask
    dt_wall = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)

    resolved = comp & core_mask
    target_rim = comp & (~resolved)
    if not resolved.any() or not target_rim.any():
        return rgb

    out = rgb.copy()
    score_map = np.where(resolved, dt_wall, -1.0).astype(np.float32)
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    for _ in range(passes):
        if not target_rim.any():
            break
        best_score = np.full((h, w), -1.0, dtype=np.float32)
        best_rgb = out.copy()
        for dy, dx in offsets:
            y0d, y1d = max(0, dy), h + min(0, dy)
            x0d, x1d = max(0, dx), w + min(0, dx)
            y0s, y1s = max(0, -dy), h + min(0, -dy)
            x0s, x1s = max(0, -dx), w + min(0, -dx)

            same_c = comp_lbl[y0d:y1d, x0d:x1d] == comp_lbl[y0s:y1s, x0s:x1s]
            cand = resolved[y0s:y1s, x0s:x1s] & same_c & target_rim[y0d:y1d, x0d:x1d]
            if not cand.any():
                continue
            sc = score_map[y0s:y1s, x0s:x1s] - (0.05 if (dy != 0 and dx != 0) else 0.0)
            better = cand & (sc > best_score[y0d:y1d, x0d:x1d])
            if better.any():
                best_score[y0d:y1d, x0d:x1d][better] = sc[better]
                best_rgb[y0d:y1d, x0d:x1d][better] = out[y0s:y1s, x0s:x1s][better]

        filled = target_rim & (best_score >= 0.0)
        if not filled.any():
            break
        out[filled] = best_rgb[filled]
        score_map[filled] = best_score[filled]
        resolved |= filled
        target_rim &= ~filled

    return out


def _majority_denoise_in_compartments(
    lbl_map: np.ndarray,
    comp_lbl: np.ndarray,
    wall_mask: np.ndarray,
    delic_mask: np.ndarray,
    passes: int = 2,
) -> np.ndarray:
    """Remove 1-2px salt-and-pepper quantization confetti strictly WITHIN each enclosed
    compartment (`comp_lbl`) without ever voting across `wall_mask` or blurring eyes."""
    h, w = lbl_map.shape
    out = lbl_map.copy()
    valid = (~wall_mask) & (~delic_mask)
    if not valid.any():
        return out
    offs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(passes):
        # Collect 8 same-compartment neighbor labels + weights (4-nbrs=2, diag=1)
        cand_lbls = []
        cand_wts = []
        for dy, dx in offs:
            y0d, y1d = max(0, dy), h + min(0, dy)
            x0d, x1d = max(0, dx), w + min(0, dx)
            y0s, y1s = max(0, -dy), h + min(0, -dy)
            x0s, x1s = max(0, -dx), w + min(0, -dx)
            arr_l = out.copy()
            wt = np.zeros((h, w), dtype=np.int16)
            same_c = comp_lbl[y0d:y1d, x0d:x1d] == comp_lbl[y0s:y1s, x0s:x1s]
            arr_l[y0d:y1d, x0d:x1d][same_c] = out[y0s:y1s, x0s:x1s][same_c]
            wt[y0d:y1d, x0d:x1d][same_c] = 2 if (dy == 0 or dx == 0) else 1
            cand_lbls.append(arr_l)
            cand_wts.append(wt)
        stack_l = np.stack(cand_lbls, axis=0)  # [8, H, W]
        stack_w = np.stack(cand_wts, axis=0)   # [8, H, W]
        # Self support weight = 2
        self_sup = np.sum((stack_l == out[None]) * stack_w, axis=0) + 2
        # Find mode neighbor label when self support is weak (<= 5 out of 14 -> isolated confetti!)
        weak = valid & (self_sup <= 5)
        if not weak.any():
            break
        sl_w = stack_l[:, weak]  # [8, M]
        sw_w = stack_w[:, weak]  # [8, M]
        best_l = out[weak].copy()
        best_s = np.zeros(sl_w.shape[1], dtype=np.int16)
        for k_i in range(8):
            cand_k = sl_w[k_i]
            sc_k = np.sum((sl_w == cand_k[None]) * sw_w, axis=0)
            upd = sc_k > best_s
            best_s[upd] = sc_k[upd]
            best_l[upd] = cand_k[upd]
        out[weak] = np.where(best_s >= 5, best_l, out[weak])
    return out


def _quantize_from_core_oklab(
    clean_rgb: np.ndarray,
    core_mask: np.ndarray,
    delic_mask: np.ndarray,
    k_bands: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a crisp color palette strictly from clean un-contaminated `core_mask` interior pixels
    and return `(quant_rgb, label_map)`."""
    h, w = clean_rgb.shape[:2]
    pool_mask = core_mask if core_mask.any() else np.ones((h, w), bool)
    pts_core = clean_rgb[pool_mask]
    del_pts = clean_rgb[pool_mask & delic_mask]

    n_main = min(45000, len(pts_core))
    step_m = max(1, len(pts_core) // max(1, n_main))
    samp_list = [pts_core[::step_m]]
    if len(del_pts) > 0:
        step_d = max(1, len(del_pts) // 10000)
        samp_list.append(del_pts[::step_d])
        samp_list.append(del_pts[::step_d])

    # Boost chromatic/midtone gradient regions (cheek blush, hair sheen) so uniform bg doesn't starve palette slots
    core_lab = srgb_to_oklab(pts_core[::step_m]).astype(np.float32)
    chroma = np.hypot(core_lab[:, 1], core_lab[:, 2])
    mid_chrom = pts_core[::step_m][(chroma > 0.025) & (core_lab[:, 0] > 0.40) & (core_lab[:, 0] < 0.90)]
    if len(mid_chrom) > 0:
        step_c = max(1, len(mid_chrom) // 12000)
        samp_list.append(mid_chrom[::step_c])

    samp_rgb = np.concatenate(samp_list, axis=0)
    samp_lab = srgb_to_oklab(samp_rgb).astype(np.float32)

    k = min(int(k_bands), len(samp_lab))
    if k < 2:
        return clean_rgb.copy(), np.zeros((h, w), dtype=np.int32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 0.0012)
    _, labels_s, centers_lab = cv2.kmeans(samp_lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    lbl_flat = labels_s.ravel()
    centers_rgb = np.zeros((k, 3), dtype=np.uint8)
    valid_k = np.zeros(k, dtype=bool)
    for ki in range(k):
        mk = lbl_flat == ki
        if mk.any():
            centers_rgb[ki] = np.median(samp_rgb[mk], axis=0).astype(np.uint8)
            valid_k[ki] = True
    centers_lab = centers_lab[valid_k]
    centers_rgb = centers_rgb[valid_k]

    # Chunked nearest-neighbor Oklab projection across full 2x image
    flat_lab = srgb_to_oklab(clean_rgb.reshape(-1, 3)).astype(np.float32)
    lbl_out = np.empty(h * w, dtype=np.int32)
    chunk_sz = 120000
    for s0 in range(0, h * w, chunk_sz):
        c_lab = flat_lab[s0:s0 + chunk_sz]
        d2 = np.sum((c_lab[:, None, :] - centers_lab[None, :, :]) ** 2, axis=2)
        lbl_out[s0:s0 + chunk_sz] = np.argmin(d2, axis=1)
    lbl_map = lbl_out.reshape(h, w)
    return centers_rgb, lbl_map


def build_vector_constrained_fill_svg(
    source_1x: np.ndarray,
    fmaps,
    centerlines_sr: list[np.ndarray],
    bands: int = 84,
    smooth_lines: bool = True,
    refit_outline: bool = True,
) -> tuple[str, np.ndarray]:
    """Construct the line-art-constrained `#region-colours` SVG strictly inside `LineVectorNet`
    & traced Bezier centerline compartments at native `sr_scale` resolution."""
    h1, w1 = source_1x.shape[:2]
    sr_rgb = fmaps.sr_hi_rgb
    vf = fmaps.line_vector_fields
    sr_scale = float(getattr(fmaps, "sr_scale", 2.0))
    hs, ws = sr_rgb.shape[:2]

    # 1. Exact Sub-Pixel Vector Wall Mask at SR resolution (Bezier strokes + LineVectorNet probs + L2 seam wall)
    wall_hi = (vf[0] >= 0.40) | (vf[1] >= 0.46)
    line_draw = np.zeros((hs, ws), dtype=np.uint8)
    for cl in centerlines_sr:
        if len(cl) >= 2:
            pts_i = np.rint(cl).astype(np.int32)
            cv2.polylines(line_draw, [pts_i], False, 1, thickness=2, lineType=cv2.LINE_8)
    wall_hi |= (line_draw > 0)

    # Upsample auxiliary CNN protection & surface priors cleanly to SR grid
    delic_hi = cv2.resize(fmaps.delicate_detail, (ws, hs), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "delicate_detail", None) is not None else np.zeros((hs, ws), np.float32)
    wash_hi = cv2.resize(fmaps.mergeable_wash, (ws, hs), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "mergeable_wash", None) is not None else np.zeros((hs, ws), np.float32)
    cel_hi = cv2.resize(fmaps.surface_probs[0], (ws, hs), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "surface_probs", None) is not None else np.zeros((hs, ws), np.float32)
    iris_hi = cv2.resize(fmaps.elem_grad_maps[3], (ws, hs), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "elem_grad_maps", None) is not None else np.zeros((hs, ws), np.float32)
    ink_hi = cv2.resize(fmaps.ink_main_geom, (ws, hs), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "ink_main_geom", None) is not None else np.zeros((hs, ws), np.float32)
    strand_hi = cv2.resize(fmaps.fine_strand, (ws, hs), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "fine_strand", None) is not None else np.zeros((hs, ws), np.float32)
    g95_hi = (cv2.resize(fmaps.grad_core_95.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0) if getattr(fmaps, "grad_core_95", None) is not None else np.zeros((hs, ws), bool)
    surf_hi = np.stack([cv2.resize(fmaps.surface_probs[c], (ws, hs), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=0) if getattr(fmaps, "surface_probs", None) is not None else np.stack([cel_hi, wash_hi, np.zeros_like(cel_hi), np.zeros_like(cel_hi)], axis=0)
    from types import SimpleNamespace
    fmaps_hi = SimpleNamespace(
        grad_core_95=g95_hi,
        delicate_detail=delic_hi,
        mergeable_wash=wash_hi,
        surface_probs=surf_hi,
        ink_main_geom=ink_hi,
        fine_strand=strand_hi,
        summary=getattr(fmaps, "summary", None) or {},
    )

    L_sr = srgb_to_oklab(sr_rgb)[..., 0].astype(np.float32)
    glint_ex = (L_sr > 0.88) & ((iris_hi > 0.28) | (delic_hi > 0.30))
    if glint_ex.any():
        wall_hi &= ~glint_ex

    # 2. Strong edge-preserving interior smoothing strictly inside clean `core_mask` (zero wall/AA contamination!)
    sm_sr = cv2.bilateralFilter(sr_rgb, 5, 16, 5)
    sm_wash = cv2.bilateralFilter(sm_sr, 7, 26, 7)
    sm_sr = np.where((wash_hi > 0.35)[..., None], sm_wash, sm_sr)
    # Keep tiny crisp eye glints un-blurred
    sm_sr = np.where((delic_hi > 0.42)[..., None], sr_rgb, sm_sr)

    comp = ~wall_hi
    dt_wall = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
    _, comp_lbl = cv2.connectedComponents(comp.astype(np.uint8), connectivity=4)

    # Core interior pixels strictly free of line AA halos
    core_mask = comp & ((dt_wall >= 2.4) | ((delic_hi > 0.36) & (dt_wall >= 1.0)))
    clean_comp = _propagate_interior_inside_compartments(sm_sr, wall_hi, core_mask, comp_lbl, passes=4)
    from .compartment_coloring import sanitize_colors_inside_compartments
    clean_comp = sanitize_colors_inside_compartments(clean_comp, fmaps=fmaps_hi, wall_mask=wall_hi)

    # 3. Quantize using ONLY pure interior core colors + majority denoise strictly inside compartments!
    k_target = min(max(int(bands), 64), 112)
    centers_rgb, lbl_map = _quantize_from_core_oklab(clean_comp, core_mask, delic_hi > 0.35, k_bands=k_target)
    if centers_rgb.ndim == 2:
        lbl_map = _majority_denoise_in_compartments(lbl_map, comp_lbl, wall_hi, delic_hi > 0.35, passes=2)
        quant_sr = centers_rgb[lbl_map]
    else:
        quant_sr = clean_comp.copy()

    # Absorb tiny confetti/AA specks inside compartments strictly WITHOUT crossing wall_hi
    # Note `area_scale = sr_scale ** 2` aligns pixel area ladders to 2x SR resolution!
    quant_sr = merge_transition_patches(
        quant_sr,
        sr_rgb,
        passes=4,
        protect_map=delic_hi,
        wash_map=wash_hi,
        cel_map=cel_hi,
        wall_mask=wall_hi,
        area_scale=sr_scale ** 2,
    )

    # 4. Watertight Voronoi Tuck-Under across wall_hi without shooting bright spikes across dark ink
    valid_seed = ~wall_hi
    n_nl, l_nl, st_nl, _ = cv2.connectedComponentsWithStats(valid_seed.astype(np.uint8), connectivity=4)
    q_L = srgb_to_oklab(quant_sr)[..., 0]
    small_ids = np.flatnonzero(st_nl[:, cv2.CC_STAT_AREA] < 24)
    for sid in small_ids:
        if sid == 0:
            continue
        x0s, y0s, bws, bhs = [int(st_nl[sid, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        sm_s = l_nl[y0s:y0s + bhs, x0s:x0s + bws] == sid
        if delic_hi[y0s:y0s + bhs, x0s:x0s + bws][sm_s].any():
            continue
        if float(np.max(L_sr[y0s:y0s + bhs, x0s:x0s + bws][sm_s])) < 0.38 and float(np.mean(q_L[y0s:y0s + bhs, x0s:x0s + bws][sm_s])) < 0.45:
            valid_seed[y0s:y0s + bhs, x0s:x0s + bws][sm_s] = False
    if not valid_seed.any():
        valid_seed = ~wall_hi

    fill_sr = quant_sr.copy()
    if (~valid_seed).any():
        dist_v, (iy_w, ix_w) = distance_transform_edt(~valid_seed, return_indices=True)
        voronoi_rgb = quant_sr[iy_w, ix_w]
        L_vor = q_L[iy_w, ix_w]
        fill_sr[~valid_seed] = voronoi_rgb[~valid_seed]
        # Protect bright hair strands, snowflakes & cat white rims inside wall_hi from dark Voronoi overwrite
        is_bright_in_wall = (~valid_seed) & (L_sr >= 0.54) & (L_vor < L_sr - 0.18)
        if is_bright_in_wall.any():
            fill_sr[is_bright_in_wall] = quant_sr[is_bright_in_wall]
        is_dark_wall = (~valid_seed) & (L_sr < 0.18) & (dist_v > 1.25) & (L_vor > 0.28)
        if is_dark_wall.any():
            dark_seeds = valid_seed & (q_L < 0.22)
            if dark_seeds.any():
                d_dk, (iy_d, ix_d) = distance_transform_edt(~dark_seeds, return_indices=True)
                use_dk = is_dark_wall & (d_dk <= 24.0)
                fill_sr[use_dk] = quant_sr[iy_d[use_dk], ix_d[use_dk]]
                rem_dk = is_dark_wall & (~use_dk)
                if rem_dk.any():
                    fill_sr[rem_dk] = (8, 8, 12)
            else:
                fill_sr[is_dark_wall] = (8, 8, 12)

    from .gradient_topology import enforce_global_gradient_topology
    fill_sr = enforce_global_gradient_topology(fill_sr, sr_rgb, fmaps=fmaps_hi, wall_mask=wall_hi)

    # 5. Vectorize fill_sr directly at sr_scale coordinate grid (aligning seamlessly under transform="scale(sr_scale)")
    from .print_partition import compact_fragments
    fill_regs = sk_label(fill_sr.astype(np.int32) @ np.array([65536, 256, 1]) + 1, connectivity=1) - 1
    fill_cols = np.empty((fill_regs.max() + 1, 3), np.uint8)
    fill_cols[fill_regs.ravel()] = fill_sr.reshape(-1, 3)
    fill_cols = snap_gray(fill_cols, False)
    fill_regs, fill_cols = compact_fragments(fill_regs, fill_cols, 1)

    fitting = {}
    if refit_outline:
        fitting = dict(
            structural_fitter=partial(fit_compact_outline, smooth_lines=smooth_lines),
            secondary_fitter=partial(refit_sketch_chain, smooth_lines=smooth_lines),
            secondary_network=partial(smooth_outline_junctions, smooth_lines=smooth_lines),
        )

    tol = 0.25 * sr_scale
    smoothing = 0.55 * sr_scale
    svg_fill, _ = trace_regions(
        fill_regs,
        ["#%02x%02x%02x" % tuple(c) for c in fill_cols],
        tol=tol,
        seam=0,
        geometry_guard=True,
        smoothing=smoothing,
        structural_scale=sr_scale,
        compact_commands=True,
        **fitting,
    )
    svg_fill = re.sub(r'width="\d+" height="\d+"', f'width="{w1}" height="{h1}"', svg_fill, count=1)
    preview_1x = cv2.resize(fill_sr, (w1, h1), interpolation=cv2.INTER_AREA)
    return svg_fill, preview_1x
