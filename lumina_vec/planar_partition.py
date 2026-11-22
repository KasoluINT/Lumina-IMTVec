"""Lossless Line-CNN + Gradient-Band Planar Partition for Two-Layer Teacher SVGs.

Implements the strict Stage 2 -> Stage 2.5 (5% buffer dual-edge & line closure) -> Stage 3
(gradient color banding -> smooth gradient-step contours + single-arc cheek blush) -> Stage 4
(disjoint planar single-color fill) decomposition in high-resolution super-sampled space.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import label as sk_label
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from .color import srgb_to_oklab
from .outline_stages import snap_gray
from .two_layer_engine import repair_line_heatmap


def _union_find_merge(
    labels_map: np.ndarray,
    lab_colors: np.ndarray,
    mask_valid: np.ndarray,
    dE_always: float,
    area_sm: int,
    dE_sm: float,
    area_tiny: int,
    dE_tiny: float,
    min_tiny_abs: int = 0,
    no_merge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Merge 4-adjacent regions inside `mask_valid` by OKLab proximity and size."""
    areas = np.bincount(labels_map.ravel())
    pairs_list = []
    for dh, dw in ((0, 1), (1, 0)):
        a = labels_map[:-1 if dh else None, :-1 if dw else None]
        b = labels_map[1 if dh else 0:, 1 if dw else 0:]
        diff = (a != b) & (a > 0) & (b > 0)
        if diff.any():
            pa, pb = a[diff], b[diff]
            lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
            pairs_list.append((lo.astype(np.int64) << 32) | hi.astype(np.int64))
    if not pairs_list:
        return labels_map

    all_edges = np.unique(np.concatenate(pairs_list))
    ea = (all_edges >> 32).astype(np.int32)
    eb = (all_edges & 0xFFFFFFFF).astype(np.int32)
    dE = np.linalg.norm(lab_colors[ea] - lab_colors[eb], axis=1)

    parent = np.arange(len(areas), dtype=np.int32)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    area_cur = areas.copy()
    for idx in np.argsort(dE):
        u, v, de_val = int(ea[idx]), int(eb[idx]), float(dE[idx])
        if no_merge_mask is not None and (no_merge_mask[u] != no_merge_mask[v]):
            continue
        ru, rv = _find(u), _find(v)
        if ru == rv:
            continue
        min_a = min(int(area_cur[ru]), int(area_cur[rv]))
        should_merge = (
            (de_val < dE_always)
            or (min_a < min_tiny_abs)
            or (min_a < area_sm and de_val < dE_sm)
            or (min_a < area_tiny and de_val < dE_tiny)
        )
        if should_merge:
            if area_cur[ru] < area_cur[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            area_cur[ru] += area_cur[rv]

    for i in range(1, len(parent)):
        parent[i] = _find(i)
    return np.where(mask_valid, parent[labels_map], 0)


def _bridge_diagonal_4conn(mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """Turn 8-connected diagonal 1-px steps into watertight 4-connected walls."""
    out = mask.copy()
    d1 = out[:-1, :-1] & out[1:, 1:] & (~out[1:, :-1]) & (~out[:-1, 1:])
    ys1, xs1 = np.where(d1)
    if len(ys1):
        pick_a = guide[ys1 + 1, xs1] >= guide[ys1, xs1 + 1]
        out[ys1[pick_a] + 1, xs1[pick_a]] = True
        out[ys1[~pick_a], xs1[~pick_a] + 1] = True

    d2 = out[1:, :-1] & out[:-1, 1:] & (~out[:-1, :-1]) & (~out[1:, 1:])
    ys2, xs2 = np.where(d2)
    if len(ys2):
        pick_a = guide[ys2, xs2] >= guide[ys2 + 1, xs2 + 1]
        out[ys2[pick_a], xs2[pick_a]] = True
        out[ys2[~pick_a] + 1, xs2[~pick_a] + 1] = True
    return out


def _seal_line_junctions(
    line_mask: np.ndarray,
    repaired_heat: np.ndarray,
    is_trough: np.ndarray,
    max_step: int = 8,
) -> np.ndarray:
    """Close small 1..max_step px open T-junction/loop gaps so first-layer contours close cleanly."""
    if max_step < 2:
        return line_mask
    h, w = line_mask.shape
    out = line_mask.copy()
    skel = skeletonize(out)
    s_u8 = skel.astype(np.uint8)
    nbr = cv2.filter2D(s_u8, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - s_u8
    ep_ys, ep_xs = np.where(skel & (nbr == 1))
    if len(ep_ys) == 0:
        return out

    for ey, ex in zip(ep_ys.tolist(), ep_xs.tolist()):
        # Trace 4..6 steps backward along skeleton to get outward tangent
        curr = (ey, ex)
        prev = None
        trail = [curr]
        for _ in range(6):
            cy, cx = curr
            found = None
            for dy in (-1, 0, 1):
                ny = cy + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nx = cx + dx
                    if 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev:
                        found = (ny, nx)
                        break
                if found is not None:
                    break
            if found is None:
                break
            prev = curr
            curr = found
            trail.append(curr)
        if len(trail) < 3:
            continue
        ty, tx = trail[-1]
        vy, vx = float(ey - ty), float(ex - tx)
        norm = float(np.hypot(vy, vx))
        if norm < 1e-4:
            continue
        vy, vx = vy / norm, vx / norm
        trail_set = set(trail)
        hit = None
        pts_along = []
        # Step forward up to max_step along tangent checking for stroke evidence or very short <=3px snap
        for s in range(1, max_step + 1):
            py = int(round(ey + vy * s))
            px = int(round(ex + vx * s))
            if py < 0 or py >= h or px < 0 or px >= w:
                break
            if (py, px) in trail_set:
                continue
            if out[py, px]:
                if s >= 2:
                    hit = (py, px)
                break
            # Check small 1px lateral offset if approaching a line
            found_lat = False
            for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ly, lx = py + oy, px + ox
                if 0 <= ly < h and 0 <= lx < w and out[ly, lx] and (ly, lx) not in trail_set:
                    if s >= 2:
                        hit = (ly, lx)
                        found_lat = True
                    break
            if found_lat:
                break
            pts_along.append((py, px))

        if hit is not None and pts_along:
            # Verify evidence along gap (short <= 4px always closes if >=0.08 heat or trough; longer needs >=0.12)
            heats = [float(repaired_heat[p[0], p[1]]) for p in pts_along]
            troughs = [bool(is_trough[p[0], p[1]]) for p in pts_along]
            req = 0.08 if len(pts_along) <= 4 else 0.14
            if np.mean(heats) >= req or any(troughs):
                for py, px in pts_along:
                    out[py, px] = True
    return out


def _extract_closed_line_mask(
    vf0: np.ndarray,
    lab_hi: np.ndarray,
    safe_grad_hi: np.ndarray,
    l_sc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2A: Convert LineVectorNet ch0 heatmap into lossless 4-connected closed line strips."""
    Hs, Ws = vf0.shape[:2]
    L_hi = lab_hi[:, :, 0]
    repaired_heat, _ = repair_line_heatmap(
        vf0.astype(np.float32),
        np.minimum(L_hi, 0.72),
        exclude_mask=safe_grad_hi,
        max_gap=int(round(10 * l_sc)),
    )
    blur_h = cv2.GaussianBlur(repaired_heat, (0, 0), 0.9)
    is_heat_ridge = (repaired_heat >= 0.24) & (repaired_heat >= blur_h - 0.02)

    is_trough = np.zeros((Hs, Ws), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        Lp = np.roll(np.roll(L_hi, dy, axis=0), dx, axis=1)
        Ln = np.roll(np.roll(L_hi, -dy, axis=0), -dx, axis=1)
        is_trough |= ((np.minimum(Lp, Ln) - L_hi) > 0.022)

    line_mask = (
        (repaired_heat >= 0.42)
        | (is_heat_ridge & (repaired_heat >= 0.25))
        | ((repaired_heat >= 0.18) & is_trough & (~safe_grad_hi))
    )
    l_cc = sk_label(line_mask, connectivity=2)
    l_areas = np.bincount(l_cc.ravel())
    keep_l = l_areas >= int(round(3 * l_sc))
    keep_l[0] = False
    line_mask = keep_l[l_cc]

    line_mask = _seal_line_junctions(line_mask, repaired_heat, is_trough, max_step=int(round(3.5 * l_sc)))
    line_mask = _bridge_diagonal_4conn(line_mask, repaired_heat)
    return line_mask, repaired_heat


def _partition_line_ribbons(
    line_mask: np.ndarray,
    repaired_heat: np.ndarray,
    lab_hi: np.ndarray,
    l_sc: float,
) -> tuple[np.ndarray, int]:
    """Partition `line_mask` along its skeleton centerlines and merge sub-segments cleanly."""
    Hs, Ws = line_mask.shape
    if not line_mask.any():
        return np.zeros((Hs, Ws), np.int32), 0

    skel_line = skeletonize(line_mask)
    lm_cc = sk_label(line_mask, connectivity=2)
    has_sk = np.zeros(lm_cc.max() + 1, bool)
    has_sk[lm_cc[skel_line]] = True
    if (~has_sk[1:]).any():
        skel_line |= (~has_sk[lm_cc]) & line_mask

    lm_f = line_mask.astype(np.float32)
    lm_w = cv2.GaussianBlur(lm_f, (0, 0), 3.0 * l_sc)
    lab_line_smooth = np.zeros_like(lab_hi)
    for ch in range(3):
        lab_line_smooth[:, :, ch] = cv2.GaussianBlur(lab_hi[:, :, ch] * lm_f, (0, 0), 3.0 * l_sc) / np.maximum(lm_w, 1e-4)

    sk_pts = lab_line_smooth[skel_line]
    k_line = min(12, max(1, len(sk_pts) // 20))
    if k_line >= 2:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.005)
        s_idx = np.linspace(0, len(sk_pts) - 1, min(30000, len(sk_pts)), dtype=np.int32)
        _, _, line_centers = cv2.kmeans(sk_pts[s_idx], k_line, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        sk_q = np.argmin(np.linalg.norm(sk_pts[:, None, :] - line_centers[None, :, :], axis=2), axis=1).astype(np.int32) + 1
    else:
        sk_q = np.ones(len(sk_pts), np.int32)

    sk_grid = np.zeros((Hs, Ws), np.int32)
    sk_grid[skel_line] = sk_q
    sk_cc = sk_label(sk_grid, connectivity=2)
    _, (ny, nx) = distance_transform_edt(~skel_line, return_indices=True)
    line_comp = np.zeros((Hs, Ws), np.int32)
    line_comp[line_mask] = sk_cc[ny[line_mask], nx[line_mask]]
    line_comp = sk_label(line_comp, connectivity=1)
    line_comp[~line_mask] = 0

    lc_max = int(line_comp.max()) + 1
    lc_w = np.maximum(repaired_heat, 0.1) * line_mask
    lc_wsum = np.bincount(line_comp.ravel(), weights=lc_w.ravel(), minlength=lc_max)
    lc_lab = np.stack([
        np.bincount(line_comp.ravel(), weights=(lab_hi[:, :, ch] * lc_w).ravel(), minlength=lc_max) / np.maximum(lc_wsum, 1e-6)
        for ch in range(3)
    ], axis=1).astype(np.float32)

    line_comp = _union_find_merge(
        line_comp, lc_lab, line_mask,
        dE_always=0.042,
        area_sm=int(40 * l_sc * l_sc), dE_sm=0.11,
        area_tiny=int(16 * l_sc * l_sc), dE_tiny=0.22,
        min_tiny_abs=int(8 * l_sc * l_sc),
    )
    _, flat = np.unique(line_comp[line_mask], return_inverse=True)
    out = np.zeros((Hs, Ws), np.int32)
    out[line_mask] = flat + 1
    return out, int(flat.max()) + 1


def _process_line_buffer_5(
    fill_labels: np.ndarray,
    lab_hi: np.ndarray,
    grad_E: np.ndarray,
    gc95_hi: np.ndarray,
    lb5_hi: np.ndarray,
    line_mask: np.ndarray,
    l_sc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2.5: Post-Line-CNN buffer dual-edge color check, solid/grad judgment, and merge.

    For each component B of `line_buffer_5`:
      - Samples abutting `grad_core_95` interior edge and outer non-line/cel edge;
      - If flat/distinct solid rim (low internal dispersion or strong cel step vs gc95), keeps/locks
        it as a flat solid compartment (excluded from Stage 3 gradient banding);
      - If monotonic gradient continuation, merges B directly into its enclosed `grad_core_95`
        compartment so Stage 3 bands cleanly all the way to and closes flush against `line_mask`.

    Returns `(updated_fill_labels, unified_grad_mask_hi)`.
    """
    fill_labels = fill_labels.copy()
    unified_grad = gc95_hi.copy()
    lb_active = lb5_hi & (~line_mask)
    if not lb_active.any():
        return fill_labels, unified_grad

    lb_cc = sk_label(lb_active, connectivity=1)
    n_lb = int(lb_cc.max())
    if n_lb == 0:
        return fill_labels, unified_grad

    f_areas_lb = np.bincount(fill_labels.ravel())

    objs = find_objects(lb_cc)
    kern5 = np.ones((5, 5), np.uint8)
    kern3 = np.ones((3, 3), np.uint8)
    next_id = int(fill_labels.max()) + 1

    # Distance away from ink lines so outer-edge sampling ignores 1px ink AA bleed
    dt_from_line = distance_transform_edt(~line_mask)

    for bid in range(1, n_lb + 1):
        sl = objs[bid - 1]
        if sl is None:
            continue
        # Pad slice slightly to sample touching inner gc95 & outer cel edges
        y0 = max(0, sl[0].start - 4)
        y1 = min(fill_labels.shape[0], sl[0].stop + 4)
        x0 = max(0, sl[1].start - 4)
        x1 = min(fill_labels.shape[1], sl[1].stop + 4)
        psl = (slice(y0, y1), slice(x0, x1))

        m_b = (lb_cc[psl] == bid)
        if int(m_b.sum()) < 4:
            continue

        dil = cv2.dilate(m_b.astype(np.uint8), kern5) > 0
        in_ring = dil & gc95_hi[psl]
        out_ring = dil & (~m_b) & (~gc95_hi[psl]) & (~line_mask[psl]) & (dt_from_line[psl] >= 1.5)
        if not in_ring.any():
            continue

        lab_sub = lab_hi[psl]
        pts_b_clean = lab_sub[m_b & (dt_from_line[psl] >= 1.2)]
        if len(pts_b_clean) < 3:
            pts_b_clean = lab_sub[m_b]

        c_in = np.median(lab_sub[in_ring], axis=0)
        c_b = np.median(pts_b_clean, axis=0)
        std_b = float(np.linalg.norm(pts_b_clean.std(axis=0)))
        dE_in = float(np.linalg.norm(c_b - c_in))

        is_solid_rim = False
        if out_ring.any():
            c_out = np.median(lab_sub[out_ring], axis=0)
            # Check if a crisp cel wall separates B from gc95_hi while B aligns closely with outer cel fill
            edge_grad_in = float(np.percentile(grad_E[psl][m_b], 90))
            dE_out = float(np.linalg.norm(c_b - c_out))
            if (std_b < 0.020 and dE_in >= 0.028) or (edge_grad_in >= 0.036 and dE_out < dE_in * 0.55 and std_b < 0.024):
                is_solid_rim = True

        if is_solid_rim:
            # Lock flat solid buffer compartment cleanly or attach to matching outer solid id
            fill_sub = fill_labels[psl]
            if out_ring.any():
                out_ids = fill_sub[out_ring]
                out_ids = out_ids[out_ids > 0]
                if len(out_ids):
                    maj_out = int(np.bincount(out_ids).argmax())
                    if np.linalg.norm(c_b - np.median(lab_sub[fill_sub == maj_out], axis=0)) < 0.024:
                        fill_sub[m_b] = maj_out
                        continue
            fill_sub[m_b] = next_id
            next_id += 1
        else:
            # Merge 5% buffer directly into its abutting grad_core_95 compartment without crossing distinct Stage 2B cel walls!
            fill_sub = fill_labels[psl]
            in_ids = fill_sub[in_ring]
            in_ids = in_ids[in_ids > 0]
            if len(in_ids):
                maj_in = int(np.bincount(in_ids).argmax())
                m_merge = m_b & (fill_sub == maj_in)
                for rid in np.unique(fill_sub[m_b]):
                    if rid <= 0 or rid == maj_in:
                        continue
                    r_mask = m_b & (fill_sub == rid)
                    if int(f_areas_lb[rid]) < int(16 * l_sc * l_sc) and float(np.linalg.norm(np.median(lab_sub[r_mask], axis=0) - c_in)) < 0.025:
                        m_merge |= r_mask
                if m_merge.any():
                    fill_sub[m_merge] = maj_in
                    unified_grad[psl][m_merge] = True

    return fill_labels, unified_grad


def _band_gradients_and_blush(
    fill_labels: np.ndarray,
    lab_hi: np.ndarray,
    den_hi: np.ndarray,
    unified_grad: np.ndarray,
    line_mask: np.ndarray,
    l_sc: float,
) -> tuple[np.ndarray, int]:
    """Stage 3: Quantize open merged gradient compartments into smooth gradient-step contour bands."""
    n_macro = int(fill_labels.max())
    f_areas = np.bincount(fill_labels.ravel(), minlength=n_macro + 1)
    dt_comp = distance_transform_edt(~line_mask)
    ug_overlap = np.bincount(fill_labels.ravel(), weights=unified_grad.ravel(), minlength=n_macro + 1)
    objs = find_objects(fill_labels)
    next_id = n_macro + 1
    bands_added = 0

    for cid in range(1, n_macro + 1):
        area_c = int(f_areas[cid])
        if area_c < int(400 * l_sc * l_sc):
            continue
        sl = objs[cid - 1]
        if sl is None:
            continue
        mask_sub = (fill_labels[sl] == cid)
        max_r = float(dt_comp[sl][mask_sub].max())
        # Strictly protect narrow hair crevices from gradient fracturing!
        if max_r < 6.5 * l_sc:
            continue

        ug_cov = float(ug_overlap[cid]) / float(area_c)
        lab_sub = lab_hi[sl]
        dt_sub = distance_transform_edt(mask_sub)
        inner = mask_sub & (dt_sub >= 1.8 * l_sc)
        if int(inner.sum()) < int(140 * l_sc * l_sc):
            continue

        pts_in = lab_sub[inner]
        span = float(np.linalg.norm(np.percentile(pts_in, 95, axis=0) - np.percentile(pts_in, 5, axis=0)))
        if span < 0.018 or (ug_cov < 0.06 and span < 0.028):
            # Still check smooth radial pink cheek blush on large high-L skin regions!
            mean_skin = np.mean(pts_in, axis=0)
            is_pale_skin = (area_c >= int(2200 * l_sc * l_sc)) and (mean_skin[0] > 0.85) and (float(np.percentile(pts_in[:, 1], 98) - np.percentile(pts_in[:, 1], 5)) > 0.012)
            if not is_pale_skin:
                continue

        sigma_g = max(4.0 * l_sc, min(14.0 * l_sc, max_r * 0.28))
        m_f = mask_sub.astype(np.float32)
        w_sm = cv2.GaussianBlur(m_f, (0, 0), sigma_g)
        lab_sm = np.zeros_like(lab_sub)
        for ch in range(3):
            lab_sm[:, :, ch] = cv2.GaussianBlur(lab_sub[:, :, ch] * m_f, (0, 0), sigma_g) / np.maximum(w_sm, 1e-4)

        pts_sm = lab_sm[inner]
        mean_v = pts_sm.mean(axis=0)
        centered = pts_sm - mean_v
        _, eigvecs = np.linalg.eigh(centered.T @ centered / max(1, len(centered)))
        axis1 = eigvecs[:, -1]
        proj_in = centered @ axis1
        p_lo, p_hi = np.percentile(proj_in, [2, 98])
        p_span = float(p_hi - p_lo)
        if p_span < 0.013:
            continue

        n_bands = int(np.clip(np.round(p_span / 0.022) + 1, 3, 10))
        proj_all = (lab_sm - mean_v) @ axis1
        t_norm = np.clip((proj_all - p_lo) / p_span, 0.0, 0.9999)
        band_map = np.floor(t_norm * n_bands).astype(np.int32) + 1
        band_map[~mask_sub] = 0

        # MBO Gaussian curvature smoothing strictly inside mask_sub so contour bands seal smoothly against walls!
        k_mbo = int(max(5, round(sigma_g * 0.8)) | 1)
        probs_b = np.zeros((n_bands + 1, mask_sub.shape[0], mask_sub.shape[1]), np.float32)
        for b_i in range(1, n_bands + 1):
            ind = (band_map == b_i).astype(np.float32)
            probs_b[b_i] = cv2.GaussianBlur(ind, (k_mbo, k_mbo), sigma_g * 0.45) / np.maximum(w_sm, 1e-4)
        band_sm = np.argmax(probs_b[1:], axis=0) + 1

        b_cc = sk_label(np.where(mask_sub, band_sm, 0), connectivity=1)
        b_areas = np.bincount(b_cc.ravel())
        keep_b = b_areas >= int(36 * l_sc * l_sc)
        keep_b[0] = False
        if int(keep_b.sum()) >= 2:
            valid_b = np.where(mask_sub & keep_b[b_cc], b_cc, 0)
            if (~keep_b[b_cc] & mask_sub).any():
                _, (by_n, bx_n) = distance_transform_edt(valid_b == 0, return_indices=True)
                valid_b = valid_b[by_n, bx_n]
            _, b_inv = np.unique(valid_b[mask_sub], return_inverse=True)
            fill_labels[sl][mask_sub] = b_inv + next_id
            next_id += int(b_inv.max()) + 1
            bands_added += int(b_inv.max())

    # Apply single-arc radial blush harmonizer onto clean banded label region RGBs if applicable
    return fill_labels, bands_added


def build_planar_partition(
    sr_hi: np.ndarray,
    fmaps,
    orig_shape: tuple[int, int],
    *,
    unlock_mode: str = "buffer_split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run Stage 2 -> 2.5 -> 3 -> 4 planar Teacher-SVG compartment construction at `sr_hi` resolution.

    Returns:
      - `regions`: int32 4-connected label map `(Hs, Ws)` (0..N-1)
      - `colors`: uint8 `(N, 3)` single solid RGB per closed compartment
      - `line_mask`: bool `(Hs, Ws)` marking Stage 2A line-art contour strips
      - `stats`: dict summary metrics
    """
    Hs, Ws = sr_hi.shape[:2]
    h1, w1 = orig_shape
    l_sc = Hs / float(max(1, h1))

    den_hi = cv2.bilateralFilter(sr_hi, 5, 10, 1.5)
    lab_hi = srgb_to_oklab(den_hi).astype(np.float32)

    gx = np.zeros((Hs, Ws), np.float32)
    gy = np.zeros((Hs, Ws), np.float32)
    gh = np.sqrt(((lab_hi[:, 1:, :] - lab_hi[:, :-1, :]) ** 2).sum(axis=2))
    gv = np.sqrt(((lab_hi[1:, :, :] - lab_hi[:-1, :, :]) ** 2).sum(axis=2))
    gx[:, :-1] = np.maximum(gx[:, :-1], gh)
    gx[:, 1:] = np.maximum(gx[:, 1:], gh)
    gy[:-1, :] = np.maximum(gy[:-1, :], gv)
    gy[1:, :] = np.maximum(gy[1:, :], gv)
    grad_E = np.hypot(gx, gy)

    # Multi-pixel directional step-ridge (±2*l_sc vs ±4*l_sc) catches 2-4px soft cel boundaries
    # (e.g., borderless iris/sclera) where 2x SR dilutes 1px step diffs while ignoring linear gradients.
    step_k = max(2, int(round(2.0 * l_sc)))
    grad_E2 = np.zeros((Hs, Ws), np.float32)
    is_step_ridge = np.zeros((Hs, Ws), bool)
    for dy, dx in ((step_k, 0), (0, step_k), (step_k, step_k), (step_k, -step_k)):
        p1 = np.roll(np.roll(lab_hi, -dy, axis=0), -dx, axis=1)
        p2 = np.roll(np.roll(lab_hi, dy, axis=0), dx, axis=1)
        d_near = np.sqrt(((p1 - p2) ** 2).sum(axis=2))
        q1 = np.roll(np.roll(lab_hi, -2 * dy, axis=0), -2 * dx, axis=1)
        q2 = np.roll(np.roll(lab_hi, 2 * dy, axis=0), 2 * dx, axis=1)
        d_far = np.sqrt(((q1 - q2) ** 2).sum(axis=2))
        dy1, dx1 = int(np.sign(dy)), int(np.sign(dx))
        d_prev = np.roll(np.roll(d_near, dy1, axis=0), dx1, axis=1)
        d_next = np.roll(np.roll(d_near, -dy1, axis=0), -dx1, axis=1)
        is_ridge_peak = (d_near >= d_prev - 0.002) & (d_near >= d_next - 0.002)
        grad_E2 = np.maximum(grad_E2, d_near)
        is_step_ridge |= (d_near >= 0.020) & (d_near > 0.60 * d_far) & is_ridge_peak

    vf = fmaps.line_vector_fields
    vf0 = vf[0] if vf[0].shape == (Hs, Ws) else cv2.resize(vf[0], (Ws, Hs), interpolation=cv2.INTER_LINEAR)
    vf1 = vf[1] if vf[1].shape == (Hs, Ws) else cv2.resize(vf[1], (Ws, Hs), interpolation=cv2.INTER_LINEAR)

    gc95_hi = cv2.resize(fmaps.grad_core_95.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0
    lb5_1x = getattr(fmaps, 'line_buffer_5', None)
    if lb5_1x is not None:
        lb5_hi = cv2.resize(lb5_1x.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0
    else:
        lb5_hi = np.zeros((Hs, Ws), bool)

    safe_grad_hi = distance_transform_edt(gc95_hi) >= (4.0 * l_sc)

    # --- Stage 2A: Closed Line-CNN Contour Strips ---
    line_mask, repaired_heat = _extract_closed_line_mask(vf0, lab_hi, safe_grad_hi, l_sc)
    line_comp_map, n_line_ribbons = _partition_line_ribbons(line_mask, repaired_heat, lab_hi, l_sc)

    # --- Stage 2B: Contour-Guided Watershed (Eliminates 1-2px AA micro-slivers) ---
    if unlock_mode == "direct_unlock":
        # Route B: Unlock grad_core_95 + line_buffer_5 immediately after Line-CNN finishes
        unified_grad = gc95_hi | (lb5_hi & (~line_mask))
        cel_wall = (
            (vf1 >= 0.14)
            | ((grad_E >= 0.024) & (~unified_grad))
            | ((grad_E >= 0.042) & lb5_hi)
            | (grad_E >= 0.065)
        )
    else:
        cel_wall = (
            (vf1 >= 0.14)
            | ((grad_E >= 0.024) & (~safe_grad_hi))
            | (is_step_ridge & (~safe_grad_hi))
            | (grad_E >= 0.065)
        )
    flat_open = (~line_mask) & (~cel_wall) & (grad_E < 0.022)
    f_cc = sk_label(flat_open, connectivity=1)
    f_areas = np.bincount(f_cc.ravel())
    valid_seed = f_areas >= 4
    valid_seed[0] = False
    remap = np.zeros(len(f_areas), np.int32)
    remap[valid_seed] = np.arange(1, int(valid_seed.sum()) + 1, dtype=np.int32)
    markers = remap[f_cc]
    n_fill_seeds = int(valid_seed.sum())

    # Rescue any enclosed non-line pocket >= 2px without a flat seed (eye sparkles, narrow crevices)
    nonline_cc = sk_label(~line_mask, connectivity=1)
    has_seed = np.zeros(nonline_cc.max() + 1, bool)
    nz = markers > 0
    has_seed[nonline_cc[nz]] = True
    nl_areas = np.bincount(nonline_cc.ravel())
    need_id = np.flatnonzero((~has_seed) & (nl_areas >= 2))
    need_id = need_id[need_id > 0]
    if len(need_id) > 0:
        objs_nl = find_objects(nonline_cc)
        for nid in need_id:
            sl = objs_nl[nid - 1]
            if sl is not None:
                markers[sl][nonline_cc[sl] == nid] = n_fill_seeds + 1
                n_fill_seeds += 1

    elev = (grad_E + 0.35 * vf1 + 0.40 * grad_E2 * (~safe_grad_hi)).astype(np.float32)
    fill_labels = watershed(elev, markers=markers, mask=(~line_mask), watershed_line=False)

    dt_fill = distance_transform_edt((~line_mask) & (grad_E < 0.024))
    comp_area = np.bincount(fill_labels.ravel())
    is_core = (fill_labels > 0) & (dt_fill >= 1.2)
    has_core = np.zeros(len(comp_area), bool)
    has_core[fill_labels[is_core]] = True
    sample_mask = np.where(has_core[fill_labels], is_core, fill_labels > 0)
    cnt = np.bincount(fill_labels[sample_mask], minlength=len(comp_area)).astype(np.float32)
    fill_lab = np.stack([
        np.bincount(fill_labels[sample_mask], weights=lab_hi[sample_mask, ch], minlength=len(comp_area)) / np.maximum(cnt, 1.0)
        for ch in range(3)
    ], axis=1).astype(np.float32)

    fill_labels = _union_find_merge(
        fill_labels, fill_lab, ~line_mask,
        dE_always=0.008,
        area_sm=int(16 * l_sc * l_sc), dE_sm=0.032,
        area_tiny=int(4 * l_sc * l_sc), dE_tiny=0.065,
        min_tiny_abs=4,
    )
    _, fill_labels = np.unique(fill_labels, return_inverse=True)
    fill_labels = fill_labels.reshape(Hs, Ws).astype(np.int32)

    # --- Stage 2.5: Post-Line-CNN Buffer Handling / Direct Unlock ---
    if unlock_mode != "direct_unlock":
        fill_labels, unified_grad = _process_line_buffer_5(
            fill_labels, lab_hi, grad_E, gc95_hi, lb5_hi, line_mask, l_sc
        )
        _, fill_labels = np.unique(fill_labels, return_inverse=True)
        fill_labels = fill_labels.reshape(Hs, Ws).astype(np.int32)
    n_macro = int(fill_labels.max())

    # --- Stage 3: Gradient Color Banding -> Smooth Gradient-Step Contours ---
    fill_labels, bands_added = _band_gradients_and_blush(
        fill_labels, lab_hi, den_hi, unified_grad, line_mask, l_sc
    )

    # --- Stage 4: Merge Line Ribbons + Fill/Gradient Compartments -> Single Pure Color per Shape ---
    total_labels = fill_labels.copy()
    if n_line_ribbons > 0:
        base_line_id = int(total_labels.max()) + 1
        total_labels[line_mask] = base_line_id + (line_comp_map[line_mask] - 1)
    _, total_labels = np.unique(total_labels, return_inverse=True)
    total_labels = total_labels.reshape(Hs, Ws).astype(np.int32)
    n_total = int(total_labels.max()) + 1

    weight_mask = np.where(
        line_mask, np.maximum(repaired_heat, 0.1), np.where(dt_fill >= 1.2, dt_fill, 0.2)
    ).astype(np.float32)
    weight_mask[line_mask & (repaired_heat >= 0.40)] *= 4.0

    w_sum = np.bincount(total_labels.ravel(), weights=weight_mask.ravel(), minlength=n_total)
    colors_rgb = np.zeros((n_total, 3), np.uint8)
    for ch in range(3):
        ch_sum = np.bincount(total_labels.ravel(), weights=(sr_hi[:, :, ch].astype(np.float32) * weight_mask).ravel(), minlength=n_total)
        colors_rgb[:, ch] = np.clip(np.round(ch_sum / np.maximum(w_sum, 1e-6)), 0, 255).astype(np.uint8)

    # Snap near-pure grays cleanly without shifting deep dark/navy backgrounds away from source tones
    base_rgb = colors_rgb[total_labels]
    # Relabel strictly 4-connected compartments for valid planar boundary tracing
    regions = sk_label(base_rgb.astype(np.int32) @ np.array([65536, 256, 1]) + 1, connectivity=1) - 1
    colors = np.empty((regions.max() + 1, 3), np.uint8)
    colors[regions.ravel()] = base_rgb.reshape(-1, 3)
    colors = snap_gray(colors, False)

    stats = dict(
        n_line_ribbons=n_line_ribbons,
        n_macro_fill=n_macro,
        n_grad_bands_added=bands_added,
        n_regions_raw=len(colors),
    )
    return regions, colors, line_mask, stats
