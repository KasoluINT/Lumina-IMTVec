"""Post-CNN Heatmap-Driven Two-Layer SVG Reconstruction Engine.

Replaces single-grid planar contour slicing with a decoupled, heatmap-driven Two-Layer Vector
Architecture:
  1. `#line-art-layer`: Extracted from `DedicatedLineNet` (L0 contour, L1 internal colored line,
     L2 free hair wisp, L3 AA-halo exclusion) + 1D valley/step-edge trough evidence, stroke-color
     unified (eliminating mottled multi-shade steps), and sub-pixel smoothed at `scale` resolution.
  2. `#region-colours`: 100% watertight, line-constrained color-fill layer beneath `#line-art-layer`
     (via Voronoi tuck-under so hiding `#line-art-layer` reveals zero holes), cleaned of pinched
     accidental white slits, chroma-bled tips, and `A -> B -> A` gradient pockets, with level-set
     smoothed gradient boundaries.
"""
from __future__ import annotations

from collections import deque
from functools import partial
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import label as sk_label
from skimage.morphology import skeletonize

from .boundary_corridor import SourceCorridor
from .color import srgb_to_oklab
from .gradient_topology import enforce_global_gradient_topology
from .outline_stages import snap_gray
from .shared_regions import trace_regions
from .structural_curves import fit_compact_outline, refit_sketch_chain, smooth_outline_junctions
from .svg_layers import boundary_then_fill


def _trace_skel_tangent(skel: np.ndarray, sy: int, sx: int, max_steps: int = 8) -> tuple[float, float] | None:
    """Walk backward along a 1-px skeleton branch from endpoint (sy, sx) to estimate outward unit tangent."""
    h, w = skel.shape
    curr = (sy, sx)
    prev = None
    path = [curr]
    for _ in range(max_steps):
        cy, cx = curr
        nbrs = []
        for dy in (-1, 0, 1):
            ny = cy + dy
            if ny < 0 or ny >= h:
                continue
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nx = cx + dx
                if 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev:
                    nbrs.append((ny, nx))
        if len(nbrs) != 1:
            break
        prev = curr
        curr = nbrs[0]
        path.append(curr)
    if len(path) < 3:
        return None
    ty, tx = path[-1]
    vy, vx = float(sy - ty), float(sx - tx)
    norm = float(np.hypot(vy, vx))
    if norm < 1e-4:
        return None
    return (vy / norm, vx / norm)


def _bfs_near_skel(skel: np.ndarray, sy: int, sx: int, max_depth: int = 22) -> set[tuple[int, int]]:
    """Return skeleton pixels reachable within `max_depth` steps along `skel` to avoid corner short-circuits."""
    h, w = skel.shape
    visited = {(sy, sx): 0}
    q = deque([(sy, sx, 0)])
    while q:
        cy, cx, d = q.popleft()
        if d >= max_depth:
            continue
        nd = d + 1
        for dy in (-1, 0, 1):
            ny = cy + dy
            if ny < 0 or ny >= h:
                continue
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nx = cx + dx
                if 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                    visited[(ny, nx)] = nd
                    q.append((ny, nx, nd))
    return set(visited.keys())


def _stitch_line_mask_gaps(
    line_mask: np.ndarray,
    cnn_prob: np.ndarray,
    L_raw: np.ndarray,
    exclude_mask: np.ndarray,
    max_gap: int = 20,
) -> np.ndarray:
    """Tangent-aligned geodesic stitching of broken hair/contour strokes in `line_mask`
    verified against CNN ridge probability and source dark-trough evidence."""
    if not line_mask.any():
        return line_mask
    h, w = line_mask.shape
    out = line_mask.copy()
    skel = skeletonize(out)
    s_u8 = skel.astype(np.uint8)
    nbr = cv2.filter2D(s_u8, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - s_u8

    eps_mask = skel & (nbr == 1) & (~exclude_mask)
    ep_ys, ep_xs = np.where(eps_mask)
    if len(ep_ys) == 0:
        return out

    # Precompute tangents at endpoints cleanly
    tangents: dict[tuple[int, int], tuple[float, float]] = {}
    for ey, ex in zip(ep_ys.tolist(), ep_xs.tolist()):
        t = _trace_skel_tangent(skel, ey, ex, max_steps=8)
        if t is not None:
            tangents[(ey, ex)] = t

    stitched_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (ey, ex), (uy, ux) in tangents.items():
        y0, y1 = max(0, ey - max_gap), min(h, ey + max_gap + 1)
        x0, x1 = max(0, ex - max_gap), min(w, ex + max_gap + 1)
        sub_sk = skel[y0:y1, x0:x1]
        sy_arr, sx_arr = np.where(sub_sk)
        if len(sy_arr) == 0:
            continue
        near_same = _bfs_near_skel(skel, ey, ex, max_depth=max_gap + 4)

        best_cand = None
        best_score = 1e9
        for r_y, r_x in zip(sy_arr.tolist(), sx_arr.tolist()):
            qy, qx = y0 + r_y, x0 + r_x
            if (qy, qx) in near_same or exclude_mask[qy, qx]:
                continue
            dy, dx = float(qy - ey), float(qx - ex)
            dist = float(np.hypot(dy, dx))
            if dist < 1.5 or dist > float(max_gap):
                continue
            cos_a = (dy * uy + dx * ux) / dist
            if cos_a < 0.80:
                continue
            is_ep_q = (qy, qx) in tangents
            if is_ep_q:
                uqy, uqx = tangents[(qy, qx)]
                # Smooth continuation: outward tangents of opposing endpoints point roughly toward each other
                if -(uy * uqy + ux * uqx) < 0.35:
                    continue
                sc = dist * (1.0 - 0.32 * cos_a)
            else:
                if cos_a < 0.86:
                    continue
                sc = dist * (1.22 - 0.30 * cos_a)
            if sc < best_score:
                best_score = sc
                best_cand = (qy, qx, dist, cos_a)

        if best_cand is None:
            continue
        qy, qx, dist, _ = best_cand
        pair_key = tuple(sorted(((ey, ex), (qy, qx))))
        if pair_key in stitched_pairs:
            continue

        # Trace ridge-hugging path strictly inside ±1px corridor perpendicular to (p -> q)
        dy, dx = float(qy - ey), float(qx - ex)
        ny_perp, nx_perp = -dx / dist, dy / dist
        n_steps = max(4, int(np.ceil(dist * 1.35)))
        path_pts: list[tuple[int, int]] = []
        side_contrasts: list[float] = []
        valid_path = True
        for k in range(1, n_steps):
            t_f = k / float(n_steps)
            by = ey + t_f * dy
            bx = ex + t_f * dx
            best_pt = None
            best_obj = -1e9
            for off in (0.0, -1.0, 1.0):
                py = int(np.clip(round(by + off * ny_perp), 0, h - 1))
                px = int(np.clip(round(bx + off * nx_perp), 0, w - 1))
                if exclude_mask[py, px] or L_raw[py, px] > 0.80:
                    continue
                obj = float(cnn_prob[py, px]) - 2.2 * float(L_raw[py, px]) - 0.08 * abs(off)
                if obj > best_obj:
                    best_obj = obj
                    best_pt = (py, px)
            if best_pt is None:
                valid_path = False
                break
            path_pts.append(best_pt)
            # Check perpendicular side contrast (trough depth against bright flank)
            sy_p, sx_p = best_pt
            p1_y = int(np.clip(round(sy_p + 2.0 * ny_perp), 0, h - 1))
            p1_x = int(np.clip(round(sx_p + 2.0 * nx_perp), 0, w - 1))
            p2_y = int(np.clip(round(sy_p - 2.0 * ny_perp), 0, h - 1))
            p2_x = int(np.clip(round(sx_p - 2.0 * nx_perp), 0, w - 1))
            side_contrasts.append(float(max(L_raw[p1_y, p1_x], L_raw[p2_y, p2_x]) - L_raw[sy_p, sx_p]))

        if not valid_path or not path_pts:
            continue
        p_ys = np.array([p[0] for p in path_pts], dtype=np.intp)
        p_xs = np.array([p[1] for p in path_pts], dtype=np.intp)
        mean_cnn = float(np.mean(cnn_prob[p_ys, p_xs]))
        min_cnn = float(np.min(cnn_prob[p_ys, p_xs]))
        mean_sc = float(np.mean(side_contrasts))
        ok = (
            (mean_cnn >= 0.22 and min_cnn >= 0.09)
            or (dist <= 9.0 and mean_cnn >= 0.14 and min_cnn >= 0.05 and mean_sc >= 0.03)
        )
        if not ok:
            continue

        # Connect path with 4/8 continuity along the ridge
        full_pts = [(ey, ex)] + path_pts + [(qy, qx)]
        for (ay, ax), (by_p, bx_p) in zip(full_pts[:-1], full_pts[1:]):
            out[ay, ax] = True
            out[by_p, bx_p] = True
            if abs(ay - by_p) == 1 and abs(ax - bx_p) == 1:
                if L_raw[ay, bx_p] <= L_raw[by_p, ax]:
                    out[ay, bx_p] = True
                else:
                    out[by_p, ax] = True
        stitched_pairs.add(pair_key)

    return out


def repair_line_heatmap(
    heat_raw: np.ndarray,
    L_raw: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    max_gap: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair small colinear breaks along continuous CNN line heatmaps before thresholding.

    Finds strong ridge anchors (`heat_raw >= 0.44`), traces outward tangents across
    weak-purple valleys (`>= 0.14`), aborts whenever an orthogonal high-confidence contour
    (`heat_raw >= 0.45`) crosses transversely across the gap, and boosts validated ridges.

    Returns `(repaired_heat, bridged_mask)`.
    """
    h, w = heat_raw.shape[:2]
    repaired = heat_raw.astype(np.float32, copy=True)
    bridged = np.zeros((h, w), bool)
    excl = exclude_mask if exclude_mask is not None else np.zeros((h, w), bool)

    strong = (repaired >= 0.44) & (~excl) & (L_raw < 0.75)
    if not strong.any():
        return repaired, bridged

    skel = skeletonize(strong)
    s_u8 = skel.astype(np.uint8)
    nbr = cv2.filter2D(s_u8, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - s_u8
    eps_mask = skel & (nbr == 1) & (~excl)
    ep_ys, ep_xs = np.where(eps_mask)
    if len(ep_ys) == 0:
        return repaired, bridged

    tangents: dict[tuple[int, int], tuple[float, float]] = {}
    for ey, ex in zip(ep_ys.tolist(), ep_xs.tolist()):
        t = _trace_skel_tangent(skel, ey, ex, max_steps=8)
        if t is not None:
            tangents[(ey, ex)] = t

    repaired_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (ey, ex), (uy, ux) in tangents.items():
        y0, y1 = max(0, ey - max_gap), min(h, ey + max_gap + 1)
        x0, x1 = max(0, ex - max_gap), min(w, ex + max_gap + 1)
        sub_sk = skel[y0:y1, x0:x1]
        sy_arr, sx_arr = np.where(sub_sk)
        if len(sy_arr) == 0:
            continue
        near_same = _bfs_near_skel(skel, ey, ex, max_depth=max_gap + 4)

        best_cand = None
        best_score = 1e9
        for r_y, r_x in zip(sy_arr.tolist(), sx_arr.tolist()):
            qy, qx = y0 + r_y, x0 + r_x
            if (qy, qx) in near_same or excl[qy, qx]:
                continue
            dy, dx = float(qy - ey), float(qx - ex)
            dist = float(np.hypot(dy, dx))
            if dist < 2.0 or dist > float(max_gap):
                continue
            cos_a = (dy * uy + dx * ux) / dist
            min_cos = 0.82 if dist <= 9.0 else 0.87
            if cos_a < min_cos:
                continue
            if (qy, qx) in tangents:
                uqy, uqx = tangents[(qy, qx)]
                if -(uy * uqy + ux * uqx) < 0.45:
                    continue
                sc = dist * (1.0 - 0.34 * cos_a)
            else:
                if cos_a < 0.90:
                    continue
                sc = dist * (1.20 - 0.30 * cos_a)
            if sc < best_score:
                best_score = sc
                best_cand = (qy, qx, dist)

        if best_cand is None:
            continue
        qy, qx, dist = best_cand
        pair_key = tuple(sorted(((ey, ex), (qy, qx))))
        if pair_key in repaired_pairs:
            continue

        dy, dx = float(qy - ey), float(qx - ex)
        ud_y, ud_x = dy / dist, dx / dist
        ny_perp, nx_perp = -ud_x, ud_y
        n_steps = max(5, int(np.ceil(dist * 1.35)))
        path_pts: list[tuple[int, int]] = []
        side_contrasts: list[float] = []
        valid_path = True
        crossed_orthogonal = False

        for k in range(1, n_steps):
            t_f = k / float(n_steps)
            by, bx = ey + t_f * dy, ex + t_f * dx
            best_pt = None
            best_obj = -1e9
            for off in (0.0, -1.0, 1.0):
                py = int(np.clip(round(by + off * ny_perp), 0, h - 1))
                px = int(np.clip(round(bx + off * nx_perp), 0, w - 1))
                if excl[py, px] or L_raw[py, px] > 0.75:
                    continue
                obj = float(repaired[py, px]) - 1.5 * float(L_raw[py, px]) - 0.06 * abs(off)
                if obj > best_obj:
                    best_obj = obj
                    best_pt = (py, px)
            if best_pt is None:
                valid_path = False
                break

            sy_p, sx_p = best_pt
            # Orthogonal crossing guard: abort if an intersecting strong contour (>= 0.45) cuts transversely
            if 0.14 <= t_f <= 0.86:
                p1_y = int(np.clip(round(sy_p + 2.5 * ny_perp), 0, h - 1))
                p1_x = int(np.clip(round(sx_p + 2.5 * nx_perp), 0, w - 1))
                p2_y = int(np.clip(round(sy_p - 2.5 * ny_perp), 0, h - 1))
                p2_x = int(np.clip(round(sx_p - 2.5 * nx_perp), 0, w - 1))
                if (
                    repaired[sy_p, sx_p] >= 0.40
                    and repaired[p1_y, p1_x] >= 0.45
                    and repaired[p2_y, p2_x] >= 0.45
                ):
                    crossed_orthogonal = True
                    break
                if skel[sy_p, sx_p] and (sy_p, sx_p) not in near_same:
                    # Check if a strong contour skeleton at this interior point runs perpendicular
                    local_t = _trace_skel_tangent(skel, sy_p, sx_p, max_steps=5)
                    if local_t is not None and abs(local_t[0] * ud_y + local_t[1] * ud_x) < 0.55:
                        crossed_orthogonal = True
                        break

            path_pts.append(best_pt)
            c1_y = int(np.clip(round(sy_p + 2.0 * ny_perp), 0, h - 1))
            c1_x = int(np.clip(round(sx_p + 2.0 * nx_perp), 0, w - 1))
            c2_y = int(np.clip(round(sy_p - 2.0 * ny_perp), 0, h - 1))
            c2_x = int(np.clip(round(sx_p - 2.0 * nx_perp), 0, w - 1))
            side_contrasts.append(float(max(L_raw[c1_y, c1_x], L_raw[c2_y, c2_x]) - L_raw[sy_p, sx_p]))

        if not valid_path or crossed_orthogonal or not path_pts:
            continue

        p_ys = np.array([p[0] for p in path_pts], dtype=np.intp)
        p_xs = np.array([p[1] for p in path_pts], dtype=np.intp)
        mean_h = float(np.mean(repaired[p_ys, p_xs]))
        min_h = float(np.min(repaired[p_ys, p_xs]))
        sc_arr = np.asarray(side_contrasts, dtype=np.float32)
        mean_sc = float(np.mean(sc_arr))
        valley_frac = float(np.mean(sc_arr >= 0.015))
        ok = (
            (mean_h >= 0.16 and min_h >= 0.08 and (valley_frac >= 0.58 or mean_h >= 0.25))
            or (dist <= 9.0 and mean_h >= 0.12 and min_h >= 0.06 and mean_sc >= 0.025 and valley_frac >= 0.65)
        )
        if not ok:
            continue

        boost_val = float(np.clip(0.5 * (heat_raw[ey, ex] + heat_raw[qy, qx]), 0.54, 0.88))
        full_pts = [(ey, ex)] + path_pts + [(qy, qx)]
        for (ay, ax), (by_p, bx_p) in zip(full_pts[:-1], full_pts[1:]):
            for py, px in ((ay, ax), (by_p, bx_p)):
                if repaired[py, px] < boost_val:
                    repaired[py, px] = boost_val
                bridged[py, px] = True
            if abs(ay - by_p) == 1 and abs(ax - bx_p) == 1:
                cy, cx = (ay, bx_p) if L_raw[ay, bx_p] <= L_raw[by_p, ax] else (by_p, ax)
                if repaired[cy, cx] < boost_val:
                    repaired[cy, cx] = boost_val
                bridged[cy, cx] = True
        repaired_pairs.add(pair_key)

    return repaired, bridged


def extract_cnn_line_art_mask(
    clean_base: np.ndarray,
    source: np.ndarray,
    fmaps=None,
    line_evidence: np.ndarray | None = None,
    heat_repaired: np.ndarray | None = None,
    heat_bridged: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine verified dark/colored thin outlines in `clean_base` with CNN heatmap detections
    (internal valleys, sub-pixel outer hair/cloth edges, and free hair wisps).

    Returns:
      line_mask : [H, W] bool mask of foreground line-art strokes
      line_rgb  : [H, W, 3] uint8 unified clean ink color per stroke segment
    """
    h, w = source.shape[:2]
    lab_raw = srgb_to_oklab(source).astype(np.float32)
    L_raw = lab_raw[..., 0]
    sm = cv2.bilateralFilter(source, 5, 18, 5)
    L_sm = srgb_to_oklab(sm)[..., 0].astype(np.float32)

    lab_cb = srgb_to_oklab(clean_base).astype(np.float32)
    L_cb = lab_cb[..., 0]

    # Build unified CNN line probability and eye/glint protection mask
    cnn_prob = np.zeros((h, w), np.float32)
    l0_prob = np.zeros((h, w), np.float32)
    l1_prob = np.zeros((h, w), np.float32)
    l2_prob = np.zeros((h, w), np.float32)
    fine_prob = np.zeros((h, w), np.float32)
    iris_prob = np.zeros((h, w), np.float32)
    delicate_prob = np.zeros((h, w), np.float32)
    if fmaps is not None:
        delicate_prob = fmaps.delicate_detail
        if getattr(fmaps, "line_net_maps", None) is not None:
            lm = fmaps.line_net_maps
            # New 4-ch semantic line_net_maps: [0: dark_ink, 1: colored_line_ridge, 2: color_block_bound, 3: free_wisp]
            l0_prob, l1_prob, l2_prob = lm[0], lm[1], lm[3]
            # Anchor line-art extraction on crisp ink/colored strokes & wisps (NEVER draw L2 step edges as black ink!)
            l1_clean = np.where((l0_prob > 0.14) | (l1_prob > 0.50), l1_prob * 0.92, l1_prob * 0.48)
            l2_clean = np.where((l0_prob > 0.14) | (l2_prob > 0.56), l2_prob * 0.88, l2_prob * 0.44)
            fine_prob = np.maximum(l1_clean, l2_clean)
            cnn_prob = np.maximum(l0_prob, np.maximum(fine_prob, fmaps.ink_main_geom * 0.90))
        else:
            fine_prob = fmaps.fine_strand
            cnn_prob = np.maximum(fmaps.ink_main_geom, fine_prob * 0.80)
        if getattr(fmaps, "elem_grad_maps", None) is not None:
            iris_prob = fmaps.elem_grad_maps[3]
    if line_evidence is not None:
        cnn_prob = np.maximum(cnn_prob, line_evidence * 0.78)

    # Only mask compact pure-white specular glints inside eye irises (never mask skin gaps between hair!)
    bright_iris = (L_raw > 0.88) & ((iris_prob > 0.28) | (delicate_prob > 0.30))
    n_g, l_g, st_g, _ = cv2.connectedComponentsWithStats(bright_iris.astype(np.uint8), connectivity=4)
    glint_rim = np.zeros((h, w), bool)
    for gi in range(1, n_g):
        if int(st_g[gi, cv2.CC_STAT_AREA]) <= 95:
            glint_rim |= (l_g == gi)
    if glint_rim.any():
        glint_rim = cv2.dilate(glint_rim.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    # 0. Continuous heatmap-domain directional gap repair BEFORE any hard thresholding
    if heat_repaired is None or heat_bridged is None:
        stroke_raw = np.maximum(l0_prob, fine_prob)
        wash_p = fmaps.mergeable_wash if (fmaps is not None and getattr(fmaps, "mergeable_wash", None) is not None) else np.zeros((h, w), np.float32)
        wash_ex = (wash_p >= 0.65) & (stroke_raw < 0.50)
        heat_repaired, heat_bridged = repair_line_heatmap(
            stroke_raw,
            L_raw,
            exclude_mask=(glint_rim & (L_raw > 0.45)) | wash_ex,
            max_gap=18,
        )
    if heat_bridged.any():
        l0_prob = np.where(heat_bridged, np.maximum(l0_prob, heat_repaired), l0_prob)
        cnn_prob = np.maximum(cnn_prob, heat_repaired)

    # 1. Verified heavy ink outlines from `clean_base` (fallback for >=4px calligraphic ink walls)
    packed_cb = (
        (clean_base[..., 0].astype(np.int32) << 16)
        | (clean_base[..., 1].astype(np.int32) << 8)
        | clean_base[..., 2].astype(np.int32)
    )
    regs_cb = sk_label(packed_cb + 1, connectivity=1) - 1
    n_cb = int(regs_cb.max()) + 1
    areas_cb = np.bincount(regs_cb.ravel(), minlength=n_cb)
    boxes_cb = find_objects(regs_cb + 1)
    reg_rgb_cb = np.zeros((n_cb, 3), np.uint8)
    reg_rgb_cb[regs_cb.ravel()] = clean_base.reshape(-1, 3)
    reg_L_cb = srgb_to_oklab(reg_rgb_cb)[:, 0]
    reg_cnn_cb = np.bincount(regs_cb.ravel(), weights=cnn_prob.ravel(), minlength=n_cb) / np.maximum(areas_cb, 1)
    reg_del_cb = np.bincount(regs_cb.ravel(), weights=delicate_prob.ravel(), minlength=n_cb) / np.maximum(areas_cb, 1)

    is_base_line = np.zeros(n_cb, bool)
    for ri, bx in enumerate(boxes_cb):
        if bx is None or areas_cb[ri] == 0:
            continue
        span = max(bx[0].stop - bx[0].start, bx[1].stop - bx[1].start)
        ar = int(areas_cb[ri])
        if reg_L_cb[ri] < 0.34 and (ar <= 3.6 * span or (reg_del_cb[ri] > 0.28 and ar <= 6.5 * span)):
            is_base_line[ri] = True
        elif reg_L_cb[ri] < 0.54 and span >= 5 and ar <= 2.6 * span and reg_cnn_cb[ri] > 0.40:
            is_base_line[ri] = True
    line_mask_base = is_base_line[regs_cb]

    # 2. Pure Heatmap-Direct Ridge Extraction: convert repaired L0+L1+L3 ridges without pixel-lightness vetoes!
    loc_min3 = cv2.erode(L_raw, np.ones((3, 3), np.uint8))
    loc_max5 = cv2.dilate(L_raw, np.ones((5, 5), np.uint8))
    wash_prob = fmaps.mergeable_wash if (fmaps is not None and getattr(fmaps, "mergeable_wash", None) is not None) else np.zeros((h, w), np.float32)
    glint_excl = glint_rim & (L_raw > 0.45)
    wash_excl = (wash_prob >= 0.65) & (heat_repaired < 0.50)
    hm_excl = glint_excl | wash_excl

    # 1D 4-direction Non-Maximum Suppression (strict 1-1.5px ridge center) + 1px topological skeleton
    hs = cv2.GaussianBlur(heat_repaired, (3, 3), 0.75)
    p_h = np.pad(hs, 1, mode="edge")
    nms_h = (hs > p_h[1:-1, :-2]) & (hs >= p_h[1:-1, 2:]) | (hs >= p_h[1:-1, :-2]) & (hs > p_h[1:-1, 2:])
    nms_v = (hs > p_h[:-2, 1:-1]) & (hs >= p_h[2:, 1:-1]) | (hs >= p_h[:-2, 1:-1]) & (hs > p_h[2:, 1:-1])
    nms_d1 = (hs > p_h[:-2, :-2]) & (hs >= p_h[2:, 2:]) | (hs >= p_h[:-2, :-2]) & (hs > p_h[2:, 2:])
    nms_d2 = (hs > p_h[:-2, 2:]) & (hs >= p_h[2:, :-2]) | (hs >= p_h[:-2, 2:]) & (hs > p_h[2:, :-2])
    ridge_peak = (heat_repaired >= 0.48) & (nms_h | nms_v | nms_d1 | nms_d2) & (~hm_excl)
    skel_pure = skeletonize((heat_repaired >= 0.46) & (~hm_excl))
    k3_ext = np.ones((3, 3), np.uint8)
    hm_cand = skel_pure | ridge_peak | (heat_bridged & (~hm_excl))

    # Drop only tiny isolated dust specks while keeping delicate details & blush ticks intact
    n_ct, l_ct, st_ct, _ = cv2.connectedComponentsWithStats(hm_cand.astype(np.uint8), connectivity=8)
    hm_clean = np.zeros((h, w), bool)
    for i in range(1, n_ct):
        ar = int(st_ct[i, cv2.CC_STAT_AREA])
        bw = int(st_ct[i, cv2.CC_STAT_WIDTH])
        bh = int(st_ct[i, cv2.CC_STAT_HEIGHT])
        if ar >= 5 or max(bw, bh) >= 5:
            hm_clean |= (l_ct == i)
        elif ar >= 3:
            x0_c, y0_c = int(st_ct[i, cv2.CC_STAT_LEFT]), int(st_ct[i, cv2.CC_STAT_TOP])
            m_i = l_ct[y0_c:y0_c + bh, x0_c:x0_c + bw] == i
            if delicate_prob[y0_c:y0_c + bh, x0_c:x0_c + bw][m_i].max() > 0.25:
                hm_clean[y0_c:y0_c + bh, x0_c:x0_c + bw] |= m_i

    line_mask = line_mask_base | hm_clean

    # 2b. Subtle tip extension + pocket hole filling along heatmap ridges
    if line_mask.any():
        soft_tip = (
            (~hm_excl)
            & (L_raw < 0.86)
            & (heat_repaired > 0.24)
            & (L_raw <= loc_min3 + 0.04)
            & (loc_max5 - L_raw >= 0.04)
        )
        for _step in range(2):
            grow = (cv2.dilate(line_mask.astype(np.uint8), k3_ext) > 0) & soft_tip & (~line_mask)
            if not grow.any():
                break
            line_mask |= grow

        stitch_excl = hm_excl | (L_raw > 0.88)
        line_mask = _stitch_line_mask_gaps(
            line_mask,
            heat_repaired,
            L_raw,
            exclude_mask=stitch_excl,
            max_gap=18,
        )

        # Fill tiny <= 26px narrow pockets/eyelets trapped inside double-tracked strokes
        n_hl, l_hl, st_hl, _ = cv2.connectedComponentsWithStats((~line_mask).astype(np.uint8), connectivity=4)
        for hi in range(1, n_hl):
            ar_h = int(st_hl[hi, cv2.CC_STAT_AREA])
            hx, hy, hbw, hbh = [int(st_hl[hi, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            if ar_h > 26 or (ar_h > 14 and min(hbw, hbh) > 6):
                continue
            if hx == 0 or hy == 0 or hx + hbw >= w or hy + hbh >= h:
                continue
            hm = l_hl[hy:hy + hbh, hx:hx + hbw] == hi
            if glint_rim[hy:hy + hbh, hx:hx + hbw][hm].any():
                continue
            if float(np.mean(L_raw[hy:hy + hbh, hx:hx + hbw][hm])) < 0.72:
                line_mask[hy:hy + hbh, hx:hx + hbw][hm] = True

    # 3. Assign & quantize clean un-contaminated stroke colors smoothly along continuous strokes
    line_rgb = np.full_like(clean_base, 255)
    if line_mask.any():
        # Separate bright-on-dark trim lines from normal dark-on-light lines so they never mix
        is_bright_trim = (L_raw > 0.72) & ((L_raw - loc_min3) > 0.14)
        is_aa_rim = line_mask & (~is_bright_trim) & (L_raw > loc_min3 + 0.08)
        pigment_src = source.copy()
        best_L = L_raw.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            y0d, y1d = max(0, dy), h + min(0, dy)
            x0d, x1d = max(0, dx), w + min(0, dx)
            y0s, y1s = max(0, -dy), h + min(0, -dy)
            x0s, x1s = max(0, -dx), w + min(0, -dx)
            sub_m = is_aa_rim[y0d:y1d, x0d:x1d] & line_mask[y0s:y1s, x0s:x1s] & (~is_bright_trim[y0s:y1s, x0s:x1s])
            cand_L = L_raw[y0s:y1s, x0s:x1s]
            de_hue = np.hypot(
                lab_raw[y0d:y1d, x0d:x1d, 1] - lab_raw[y0s:y1s, x0s:x1s, 1],
                lab_raw[y0d:y1d, x0d:x1d, 2] - lab_raw[y0s:y1s, x0s:x1s, 2],
            )
            upd = sub_m & (cand_L < best_L[y0d:y1d, x0d:x1d]) & (de_hue < 0.09)
            if upd.any():
                best_L[y0d:y1d, x0d:x1d][upd] = cand_L[upd]
                pigment_src[y0d:y1d, x0d:x1d][upd] = source[y0s:y1s, x0s:x1s][upd]

        # Smooth normal dark lines separately from bright-on-dark trim lines
        dark_m = line_mask & (~is_bright_trim)
        core_wt = (dark_m & (L_raw <= loc_min3 + 0.06)).astype(np.float32) + 0.25 * dark_m.astype(np.float32)
        num_c = cv2.GaussianBlur(pigment_src.astype(np.float32) * core_wt[..., None], (0, 0), 1.6)
        den_c = cv2.GaussianBlur(core_wt, (0, 0), 1.6)[..., None]
        smooth_col = np.where(
            (~is_bright_trim[..., None]) & (den_c > 1e-4),
            np.rint(np.clip(num_c / np.maximum(den_c, 1e-4), 0, 255)),
            pigment_src,
        ).astype(np.uint8)

        pts_rgb = smooth_col[line_mask]
        # Quantize stroke colors to <= 14 coherent Oklab ink tones so long strokes stay single SVG paths
        pts_lab = srgb_to_oklab(pts_rgb).astype(np.float32)
        k_ink = min(14, len(pts_lab))
        if k_ink >= 2:
            step_s = max(1, len(pts_lab) // 8000)
            samp = pts_lab[::step_s]
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.002)
            _, _, centers_lab = cv2.kmeans(samp, k_ink, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            # Map every line pixel in small chunks to keep peak memory tiny
            q_rgb = np.empty_like(pts_rgb)
            # Represent each cluster center by the median RGB of assigned sample points in that cluster
            d_samp = np.sum((samp[:, None, :] - centers_lab[None, :, :]) ** 2, axis=2)
            s_lbl = np.argmin(d_samp, axis=1)
            samp_rgb = pts_rgb[::step_s]
            centers_rgb = np.zeros((k_ink, 3), np.uint8)
            for ki in range(k_ink):
                m_k = s_lbl == ki
                if m_k.any():
                    centers_rgb[ki] = np.median(samp_rgb[m_k], axis=0).astype(np.uint8)
            for s0 in range(0, len(pts_lab), 50000):
                chunk = pts_lab[s0:s0 + 50000]
                idx_c = np.argmin(np.sum((chunk[:, None, :] - centers_lab[None, :, :]) ** 2, axis=2), axis=1)
                q_rgb[s0:s0 + 50000] = centers_rgb[idx_c]
            line_rgb[line_mask] = q_rgb
        else:
            line_rgb[line_mask] = pts_rgb

    return line_mask, line_rgb


def build_two_layer_svg(
    clean_base: np.ndarray,
    source: np.ndarray,
    palette: np.ndarray,
    fmaps=None,
    line_evidence: np.ndarray | None = None,
    smooth_lines: bool = True,
    refit_outline: bool = True,
    locate_boundaries: bool = True,
    wash_comb: np.ndarray | None = None,
    stage: int = 0,
    reason: str = "",
    removed: int = 0,
    outline_layer: bool = True,
    progress=None,
    heat_repaired: np.ndarray | None = None,
    heat_bridged: np.ndarray | None = None,
) -> tuple[str, np.ndarray]:
    """Assemble decoupled Two-Layer SVG (`#region-colours` closed underneath + `#line-art-layer`
    smooth vector ink on top) and return `(svg, composite_preview_rgb)`."""
    step = progress or (lambda *_: None)
    h, w = source.shape[:2]
    scale = 4 if w * h <= 2000000 else 2

    step(0.62, "基于 CNN 矢量场与热力图提取独立矢量线稿与约束填色层")
    has_vec = (
        fmaps is not None
        and getattr(fmaps, "line_vector_fields", None) is not None
        and getattr(fmaps, "sr_hi_rgb", None) is not None
    )
    vec_paths: list[str] = []
    if has_vec:
        from .vector_tracer import trace_vector_fields_to_svg_paths
        vec_paths, _ = trace_vector_fields_to_svg_paths(
            fmaps.line_vector_fields,
            fmaps.sr_hi_rgb,
            sr_scale=float(getattr(fmaps, "sr_scale", 2.0)),
            tapered=False,
        )
        stroke_1x = cv2.resize(fmaps.line_vector_fields[0], (w, h), interpolation=cv2.INTER_AREA)
        delic_1x = (fmaps.delicate_detail > 0.32) if getattr(fmaps, "delicate_detail", None) is not None else np.zeros((h, w), bool)
        cb_l_raw = srgb_to_oklab(clean_base)[..., 0]
        # Tuck-under cleanly under actual stroke centerlines while protecting delicate eye pupils & hair glints
        base_core = stroke_1x >= 0.42
        dark_rim = (cv2.dilate((stroke_1x >= 0.38).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (cb_l_raw < 0.45) & (~delic_1x)
        line_mask = base_core | dark_rim
        line_rgb = source.copy()
    else:
        line_mask, line_rgb = extract_cnn_line_art_mask(
            clean_base,
            source,
            fmaps=fmaps,
            line_evidence=line_evidence,
            heat_repaired=heat_repaired,
            heat_bridged=heat_bridged,
        )

    # If the image has almost no detected line strokes (e.g. painterly lineless wash), fall back cleanly
    if not line_mask.any():
        fill_1x = clean_base.copy()
    else:
        # LAYER 1 (`#region-colours`): Voronoi Tuck-Under beneath `line_mask` so hiding lines leaves ZERO white holes!
        # Absorb tiny dark/midtone line-trapped cusps (< 85px, L < 0.68 outside facial eyes/glints) so shadow
        # cusps under crossed ink tips never expand into circular blobs in `fit_structural`!
        fill_1x = clean_base.copy()
        non_line = ~line_mask
        n_nl, l_nl, st_nl, _ = cv2.connectedComponentsWithStats(non_line.astype(np.uint8), connectivity=4)
        valid_seed = non_line.copy()
        cb_l = srgb_to_oklab(clean_base)[..., 0]
        del_m = (fmaps.delicate_detail > 0.35) if (fmaps is not None and getattr(fmaps, "delicate_detail", None) is not None) else np.zeros((h, w), bool)
        small_dark_ids = np.flatnonzero(st_nl[:, cv2.CC_STAT_AREA] < 85)
        for sid in small_dark_ids:
            if sid == 0:
                continue
            x0_s, y0_s, bw_s, bh_s = [int(st_nl[sid, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            sm_s = l_nl[y0_s:y0_s + bh_s, x0_s:x0_s + bw_s] == sid
            if del_m[y0_s:y0_s + bh_s, x0_s:x0_s + bw_s][sm_s].any():
                continue
            if float(np.mean(cb_l[y0_s:y0_s + bh_s, x0_s:x0_s + bw_s][sm_s])) < 0.68:
                valid_seed[y0_s:y0_s + bh_s, x0_s:x0_s + bw_s][sm_s] = False
        if not valid_seed.any():
            valid_seed = non_line
        _, (iy_f, ix_f) = distance_transform_edt(~valid_seed, return_indices=True)
        fill_1x[~valid_seed] = clean_base[iy_f[~valid_seed], ix_f[~valid_seed]]

    step(0.70, "整理线稿约束填色区域并拟合平滑共享边界")
    fill_work = cv2.resize(fill_1x, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    fill_regs = sk_label(fill_work.astype(np.int32) @ np.array([65536, 256, 1]) + 1, connectivity=1) - 1
    fill_cols = np.empty((fill_regs.max() + 1, 3), np.uint8)
    fill_cols[fill_regs.ravel()] = fill_work.reshape(-1, 3)
    fill_cols = snap_gray(fill_cols, False)

    # Compact tiny stair-step fragments inside `#region-colours` matching v10 smoothness
    from .print_partition import compact_fragments
    fill_regs, fill_cols = compact_fragments(fill_regs, fill_cols, scale)

    fitting = {}
    corridor = None
    metadata = None
    if refit_outline:
        fitting = dict(
            structural_fitter=partial(fit_compact_outline, smooth_lines=smooth_lines),
            secondary_fitter=partial(refit_sketch_chain, smooth_lines=smooth_lines),
            secondary_network=partial(smooth_outline_junctions, smooth_lines=smooth_lines),
        )
        if locate_boundaries:
            corridor = SourceCorridor(source, fill_cols)
            fitting.update(chain_locator=corridor, source_scale=scale)
        metadata = {
            "data-stage": str(stage),
            "data-stop-reason": reason,
            "data-weak-islands-removed": str(removed),
        }

    tol = (0.32 if smooth_lines else 0.25) * scale
    smoothing = (0.85 if smooth_lines else 0.65) * scale
    chain_scale_fn = None
    if fmaps is not None and smooth_lines and wash_comb is not None:
        cbound_map = fmaps.line_net_maps[2] if getattr(fmaps, "line_net_maps", None) is not None else None
        def chain_scale_fn(pts, base_scale, s_scale):
            ys = np.clip(np.round(pts[:, 1] / s_scale).astype(int), 0, h - 1)
            xs = np.clip(np.round(pts[:, 0] / s_scale).astype(int), 0, w - 1)
            if float(np.mean(fmaps.delicate_detail[ys, xs])) > 0.35:
                return base_scale * 0.85
            if cbound_map is not None and float(np.mean(cbound_map[ys, xs])) > 0.36:
                return base_scale * 0.90
            if float(np.mean(wash_comb[ys, xs])) > 0.48:
                return base_scale * 1.45
            if float(np.mean(fmaps.ink_main_geom[ys, xs])) > 0.38:
                return base_scale * 1.15
            return base_scale

    svg_fill, stats = trace_regions(
        fill_regs,
        ["#%02x%02x%02x" % tuple(c) for c in fill_cols],
        tol=tol,
        seam=0,
        geometry_guard=True,
        smoothing=smoothing,
        structural_scale=scale,
        compact_commands=True,
        chain_scale_fn=chain_scale_fn,
        **fitting,
    )
    svg_fill = re.sub(r'width="\d+" height="\d+"', f'width="{w}" height="{h}"', svg_fill, count=1)

    if not outline_layer or not line_mask.any():
        if outline_layer:
            svg_fill = boundary_then_fill(svg_fill, hidden_strokes=(), metadata=metadata)
        return svg_fill, fill_1x

    if metadata is not None:
        metadata["data-segments-before-refit"] = str(stats["segments_before_secondary"])
        metadata["data-segments-after-refit"] = str(stats["curve_segments"])
        if corridor is not None:
            metadata["data-source-chains-considered"] = str(corridor.considered)
            metadata["data-source-chains-moved"] = str(corridor.moved)
            metadata["data-source-tonal-chains"] = str(corridor.tonal_chains)
        metadata["data-color-fill-paths"] = str(len(fill_cols))

    svg_layered = boundary_then_fill(svg_fill, hidden_strokes=(), metadata=metadata)

    # LAYER 2 (`#line-art-layer`): Direct LineVectorNet Cubic Bezier path layer when available!
    if has_vec and vec_paths:
        step(0.86, "输出 LineVectorNet 矢量场直出线稿层 (#line-art-layer)")
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ns = "{http://www.w3.org/2000/svg}"
        root_out = ET.fromstring(svg_layered)
        g_lines = ET.Element(
            ns + "g",
            {"id": "line-art-layer", "data-role": "cnn-vector-field-layer", "transform": f"scale({scale})"},
        )
        for p_idx, p_str in enumerate(vec_paths):
            el = ET.fromstring(p_str)
            el.tag = ns + "path"
            el.set("id", f"vec-line-{p_idx}")
            g_lines.append(el)
        root_out.append(g_lines)
        outlines_g = next((g for g in root_out if g.get("id") == "region-boundaries"), None)
        if outlines_g is not None:
            outlines_g.set("data-line-art-paths", str(len(g_lines)))
        return ET.tostring(root_out, encoding="unicode"), fill_1x

    # Fallback LAYER 2 (`#line-art-layer`): Exact-width 0-bloat line mask + sub-pixel diagonal elbow bridges
    step(0.86, "重构高精度独立线稿矢量路径 (#line-art-layer)")
    L_raw = srgb_to_oklab(source)[..., 0].astype(np.float32)
    line_hi_mask = cv2.resize(line_mask.astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) > 0
    pad = max(1, scale // 2)
    d1 = line_mask[:-1, :-1] & line_mask[1:, 1:] & (~line_mask[:-1, 1:]) & (~line_mask[1:, :-1])
    for y, x in zip(*np.where(d1)):
        cy, cx = (y + 1) * scale, (x + 1) * scale
        if L_raw[y + 1, x] <= L_raw[y, x + 1]:
            line_hi_mask[cy:cy + pad, cx - pad:cx] = True
        else:
            line_hi_mask[cy - pad:cy, cx:cx + pad] = True
    d2 = line_mask[:-1, 1:] & line_mask[1:, :-1] & (~line_mask[:-1, :-1]) & (~line_mask[1:, 1:])
    for y, x in zip(*np.where(d2)):
        cy, cx = (y + 1) * scale, (x + 1) * scale
        if L_raw[y, x] <= L_raw[y + 1, x + 1]:
            line_hi_mask[cy - pad:cy, cx - pad:cx] = True
        else:
            line_hi_mask[cy:cy + pad, cx:cx + pad] = True

    # Use sentinel RGB (1, 254, 3) outside line_hi_mask so even #ffffff highlights can never collide with background
    line_rgb_hi = cv2.resize(line_rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    valid_rgb = cv2.resize(line_mask.astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) > 0
    need_snap = line_hi_mask & (~valid_rgb)
    if need_snap.any() and valid_rgb.any():
        _, (iy_r, ix_r) = distance_transform_edt(~valid_rgb, return_indices=True)
        line_rgb_hi[need_snap] = line_rgb_hi[iy_r[need_snap], ix_r[need_snap]]

    line_work = np.empty((h * scale, w * scale, 3), np.uint8)
    line_work[:] = (1, 254, 3)
    # Remove tiny isolated dust specks in line_hi_mask (< 6 source-pixel equivalent area outside delicate facial zones)
    if fmaps is not None and getattr(fmaps, "delicate_detail", None) is not None:
        del_hi = cv2.resize(fmaps.delicate_detail, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) > 0.25
    else:
        del_hi = np.zeros((h * scale, w * scale), bool)
    n_lm, l_lm, st_lm, _ = cv2.connectedComponentsWithStats(line_hi_mask.astype(np.uint8), connectivity=8)
    min_ar_hi = 5 * scale * scale
    for li in range(1, n_lm):
        ar_li = int(st_lm[li, cv2.CC_STAT_AREA])
        if ar_li < min_ar_hi:
            x0_l, y0_l, bw_l, bh_l = [int(st_lm[li, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            sm_l = l_lm[y0_l:y0_l + bh_l, x0_l:x0_l + bw_l] == li
            if not del_hi[y0_l:y0_l + bh_l, x0_l:x0_l + bw_l][sm_l].any() and max(bw_l, bh_l) < 4 * scale:
                line_hi_mask[y0_l:y0_l + bh_l, x0_l:x0_l + bw_l][sm_l] = False
    line_work[line_hi_mask] = line_rgb_hi[line_hi_mask]

    line_regs = sk_label(line_work.astype(np.int32) @ np.array([65536, 256, 1]) + 1, connectivity=1) - 1
    line_cols = np.empty((line_regs.max() + 1, 3), np.uint8)
    line_cols[line_regs.ravel()] = line_work.reshape(-1, 3)
    # Absorb tiny color-split shards (< 8 source-px equiv) inside multi-tone ink strokes so curves stay continuous
    is_bg = (line_cols[:, 0] == 1) & (line_cols[:, 1] == 254) & (line_cols[:, 2] == 3)
    l_areas = np.bincount(line_regs.ravel(), minlength=len(line_cols))
    small_fg = (~is_bg) & (l_areas > 0) & (l_areas < 8 * scale * scale)
    if small_fg.any():
        fg_pairs = []
        for a_r, b_r in ((line_regs[:-1], line_regs[1:]), (line_regs[:, :-1], line_regs[:, 1:])):
            diff = (a_r != b_r) & (~is_bg[a_r]) & (~is_bg[b_r])
            if diff.any():
                fg_pairs.append(np.column_stack([a_r[diff], b_r[diff]]))
        if fg_pairs:
            fp = np.concatenate(fg_pairs)
            fp_bi = np.vstack([fp, fp[:, ::-1]])
            u_fp, cnt_fp = np.unique(fp_bi, axis=0, return_counts=True)
            a_u, b_u = u_fp[:, 0], u_fp[:, 1]
            ok_m = small_fg[a_u] & (l_areas[b_u] > l_areas[a_u])
            if ok_m.any():
                ord_m = np.lexsort((-cnt_fp[ok_m], a_u[ok_m]))
                au2, bu2 = a_u[ok_m][ord_m], b_u[ok_m][ord_m]
                fst = np.r_[True, au2[1:] != au2[:-1]]
                l_map = np.arange(len(line_cols))
                l_map[au2[fst]] = bu2[fst]
                for _ in range(8):
                    l_map = l_map[l_map]
                line_work = line_cols[l_map[line_regs]]
                line_regs = sk_label(line_work.astype(np.int32) @ np.array([65536, 256, 1]) + 1, connectivity=1) - 1
                line_cols = np.empty((line_regs.max() + 1, 3), np.uint8)
                line_cols[line_regs.ravel()] = line_work.reshape(-1, 3)
    line_regs, line_cols = compact_fragments(line_regs, line_cols, 1)

    line_fitting = dict(structural_fitter=partial(fit_compact_outline, smooth_lines=smooth_lines))
    if refit_outline:
        line_fitting.update(
            secondary_fitter=partial(refit_sketch_chain, smooth_lines=smooth_lines),
            secondary_network=partial(smooth_outline_junctions, smooth_lines=smooth_lines),
        )

    svg_lines_raw, _ = trace_regions(
        line_regs,
        ["#%02x%02x%02x" % tuple(c) for c in line_cols],
        tol=(0.32 if smooth_lines else 0.25) * scale,
        seam=0,
        geometry_guard=True,
        smoothing=(0.85 if smooth_lines else 0.65) * scale,
        structural_scale=scale,
        compact_commands=True,
        **line_fitting,
    )

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ns = "{http://www.w3.org/2000/svg}"
    root_out = ET.fromstring(svg_layered)
    g_lines = ET.Element(ns + "g", {"id": "line-art-layer", "data-role": "cnn-line-art-layer"})
    root_lin = ET.fromstring(svg_lines_raw)
    for p in list(root_lin):
        if p.get("fill", "").lower() == "#01fe03":
            continue
        p.set("id", "line-" + p.get("id", "0"))
        g_lines.append(p)
    root_out.append(g_lines)

    outlines_g = next((g for g in root_out if g.get("id") == "region-boundaries"), None)
    if outlines_g is not None:
        outlines_g.set("data-line-art-paths", str(len(g_lines)))

    composite_hi = np.where(line_hi_mask[..., None], line_rgb_hi, cv2.resize(fill_1x, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST))
    preview_1x = cv2.resize(composite_hi, (w, h), interpolation=cv2.INTER_AREA)
    return ET.tostring(root_out, encoding="unicode"), preview_1x


def build_two_layer_vector_svg(
    source: np.ndarray,
    fmaps,
    *,
    bands: int = 84,
    smooth_lines: bool = True,
    refit_outline: bool = True,
    outline_layer: bool = True,
    progress=None,
) -> tuple[str, np.ndarray]:
    """End-to-end Neural Vector-Field Two-Layer SVG Builder:
      1. Traces smooth Cubic Bezier `#line-art-layer` directly from `LineVectorNet` [9, Hs, Ws].
      2. Builds `#region-colours` directly inside the exact vector-bounded compartments at
         native SR resolution (bypassing legacy pixel-pinning, ink blotches, and bleed).
    """
    from .feature_routing import render_vector_heatmap_png
    from .vector_coloring import build_vector_constrained_fill_svg
    from .vector_tracer import trace_vector_fields_to_svg_paths

    step = progress or (lambda *_: None)
    sr_scale = float(getattr(fmaps, "sr_scale", 2.0))

    hm_out = os.environ.get("LUMINA_HEATMAP_OUT", "").strip()
    if hm_out and getattr(fmaps, "line_vector_fields", None) is not None:
        try:
            render_vector_heatmap_png(source, hm_out, vec_fields=fmaps.line_vector_fields)
        except Exception:
            pass

    step(0.35, "从 LineVectorNet 矢量场直出 Cubic Bezier 线稿路径")
    vf_trace = fmaps.line_vector_fields.copy()
    sr_rgb = fmaps.sr_hi_rgb
    hs_t, ws_t = sr_rgb.shape[:2]
    L_sr_t = srgb_to_oklab(sr_rgb)[..., 0].astype(np.float32)
    delic_t = cv2.resize(fmaps.delicate_detail, (ws_t, hs_t), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "delicate_detail", None) is not None else np.zeros((hs_t, ws_t), np.float32)
    wash_t = cv2.resize(fmaps.mergeable_wash, (ws_t, hs_t), interpolation=cv2.INTER_LINEAR) if getattr(fmaps, "mergeable_wash", None) is not None else np.zeros((hs_t, ws_t), np.float32)
    glint_ex_t = (L_sr_t > 0.88) & (delic_t > 0.30)
    wash_ex_t = (wash_t >= 0.68) & (vf_trace[0] < 0.48)
    hr_t, _ = repair_line_heatmap(
        np.clip(vf_trace[0], 0.0, 1.0),
        L_sr_t,
        exclude_mask=(glint_ex_t | wash_ex_t),
        max_gap=18,
    )
    vf_trace[0] = np.maximum(vf_trace[0], hr_t)
    vec_paths, _, centerlines_sr = trace_vector_fields_to_svg_paths(
        vf_trace,
        sr_rgb,
        sr_scale=sr_scale,
        tapered=False,
        return_centerlines=True,
    )

    step(0.60, "在矢量线稿封闭分区内构建无溢色、无残屑填色层 (#region-colours)")
    svg_fill, preview_1x = build_vector_constrained_fill_svg(
        source,
        fmaps,
        centerlines_sr,
        bands=bands,
        smooth_lines=smooth_lines,
        refit_outline=refit_outline,
    )

    step(0.88, "组装纯双层 SVG（着色填充层 + 矢量线稿层）")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ns = "{http://www.w3.org/2000/svg}"
    root_out = ET.fromstring(svg_fill)
    g_fill = ET.Element(ns + "g", {"id": "region-colours", "data-role": "line-constrained-color-layer"})
    for idx_f, p_el in enumerate(list(root_out)):
        if p_el.tag != ns + "path":
            continue
        col_f = p_el.get("fill")
        if col_f and col_f != "none":
            # Same-colour hairline seam seal directly on fill paths (eliminates redundant 3rd #region-boundaries layer)
            p_el.set("stroke", col_f)
            p_el.set("stroke-width", "0.65")
            p_el.set("stroke-linejoin", "round")
            p_el.set("vector-effect", "non-scaling-stroke")
        p_el.set("id", p_el.get("id") or f"fill-{idx_f}")
        root_out.remove(p_el)
        g_fill.append(p_el)
    root_out.append(g_fill)

    if not outline_layer or not vec_paths:
        return ET.tostring(root_out, encoding="unicode"), preview_1x

    g_lines = ET.Element(
        ns + "g",
        {"id": "line-art-layer", "data-role": "cnn-vector-field-layer", "transform": f"scale({sr_scale:.4f})"},
    )
    for p_idx, p_str in enumerate(vec_paths):
        el = ET.fromstring(p_str)
        el.tag = ns + "path"
        el.set("id", f"vec-line-{p_idx}")
        g_lines.append(el)
    root_out.append(g_lines)

    # Composite traced Bezier/Arc strokes accurately onto preview_1x (parse d with path_polys so A/C/L mix stays exact)
    from .svgraster import path_polys
    h1, w1 = source.shape[:2]
    hs_p, ws_p = fmaps.sr_hi_rgb.shape[:2]
    comp_sr = cv2.resize(preview_1x, (ws_p, hs_p), interpolation=cv2.INTER_LINEAR)
    for p_str in vec_paths:
        m_d = re.search(r'\bd="([^"]+)"', p_str)
        if not m_d:
            continue
        m_f = re.search(r'\bfill="#([0-9a-fA-F]{6})"', p_str)
        m_s = re.search(r'\bstroke="#([0-9a-fA-F]{6})"', p_str)
        polys = path_polys(m_d.group(1), n_bez=8)
        pts_list = [np.rint(p * sr_scale).astype(np.int32) for p in polys if len(p) >= 2]
        if not pts_list:
            continue
        if m_f:
            hx = m_f.group(1)
            col_rgb = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
            poly_fill = [p for p in pts_list if len(p) >= 3]
            if poly_fill:
                cv2.fillPoly(comp_sr, poly_fill, col_rgb, lineType=cv2.LINE_AA)
        elif m_s:
            hx = m_s.group(1)
            col_rgb = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
            m_w = re.search(r'stroke-width="([\d.]+)"', p_str)
            thk = max(1, int(round(float(m_w.group(1)) * sr_scale))) if m_w else max(1, int(round(sr_scale)))
            cv2.polylines(comp_sr, pts_list, False, col_rgb, thickness=thk, lineType=cv2.LINE_AA)
    preview_1x = cv2.resize(comp_sr, (w1, h1), interpolation=cv2.INTER_AREA)

    return ET.tostring(root_out, encoding="unicode"), preview_1x
