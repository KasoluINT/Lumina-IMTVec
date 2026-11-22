from __future__ import annotations

import cv2
import numpy as np

from .color import srgb_to_oklab


def build_line_wall_mask(
    source: np.ndarray,
    fmaps=None,
    heat_repaired: np.ndarray | None = None,
) -> np.ndarray:
    """Build a crisp [H, W] boolean barrier mask from CNN line heatmaps and sharp color-block edges.

    Separates enclosed coloring compartments strictly BEFORE quantization and region merging so
    skin, hair, and cloth colors cannot bleed across outlines or color boundaries.
    """
    h, w = source.shape[:2]
    wall = np.zeros((h, w), bool)
    if fmaps is None and heat_repaired is None:
        return wall

    L_raw = srgb_to_oklab(source)[..., 0].astype(np.float32)

    if heat_repaired is not None:
        wall |= heat_repaired >= 0.44
    elif fmaps is not None:
        if getattr(fmaps, "line_vector_fields", None) is not None:
            vf = fmaps.line_vector_fields
            wall |= cv2.resize(vf[0], (w, h), interpolation=cv2.INTER_AREA) >= 0.44
            wall |= (cv2.resize(vf[1], (w, h), interpolation=cv2.INTER_AREA) >= 0.50)
        elif getattr(fmaps, "line_net_maps", None) is not None:
            lm = fmaps.line_net_maps
            stroke = np.maximum(lm[0], np.maximum(lm[1] * 0.90, lm[3] * 0.86))
            wall |= stroke >= 0.44
        else:
            wall |= np.maximum(fmaps.ink_main_geom, fmaps.fine_strand * 0.85) >= 0.44

    # Include strong flat color-block boundaries (e.g., flower stamen / hair color steps in L2)
    if fmaps is not None and getattr(fmaps, "line_net_maps", None) is not None and fmaps.line_net_maps.shape[0] >= 3:
        l2_bound = fmaps.line_net_maps[2]
        wash = fmaps.mergeable_wash if getattr(fmaps, "mergeable_wash", None) is not None else np.zeros((h, w), np.float32)
        wall |= (l2_bound >= 0.50) & (wash < 0.72)

    # Guard eye iris glints so tiny catchlights inside eyes are never turned into empty wall holes
    if fmaps is not None:
        iris = fmaps.elem_grad_maps[3] if getattr(fmaps, "elem_grad_maps", None) is not None else np.zeros((h, w), np.float32)
        delic = fmaps.delicate_detail if getattr(fmaps, "delicate_detail", None) is not None else np.zeros((h, w), np.float32)
        glint = (L_raw > 0.88) & ((iris > 0.28) | (delic > 0.30))
        if glint.any():
            wall &= ~glint

    return wall


def sanitize_colors_inside_compartments(
    rgb: np.ndarray,
    fmaps=None,
    wall_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Strip 1–1.5px anti-aliased wall contamination strictly WITHIN each wall-enclosed compartment.

    Any boundary-adjacent rim pixel (`0 < dt_wall < 1.6`) only receives color from resolved
    interior core pixels (`dt_wall >= 1.6`) belonging to the exact same 4-connected compartment ID.
    Narrow compartments lacking an interior core remain untouched to protect fine hair wisps.
    """
    if wall_mask is None or not wall_mask.any():
        return rgb

    h, w = rgb.shape[:2]
    compartments = ~wall_mask
    if not compartments.any():
        return rgb

    dt_wall = cv2.distanceTransform(compartments.astype(np.uint8), cv2.DIST_L2, 3)
    n_cc, comp_lbl, comp_stats, _ = cv2.connectedComponentsWithStats(compartments.astype(np.uint8), connectivity=4)

    protect = np.zeros((h, w), bool)
    grad_open = np.zeros((h, w), bool)
    if fmaps is not None:
        if getattr(fmaps, "delicate_detail", None) is not None:
            protect |= fmaps.delicate_detail > 0.36
        if getattr(fmaps, "mergeable_wash", None) is not None:
            protect |= fmaps.mergeable_wash > 0.72
        if getattr(fmaps, "grad_core_95", None) is not None:
            gc = fmaps.grad_core_95
            if gc.shape != (h, w):
                gc = cv2.resize(gc.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            grad_open |= gc

    resolved = compartments & (dt_wall >= 1.6)
    target_rim = compartments & (dt_wall < 1.6) & (~protect)
    out = rgb.copy()

    # Lock small enclosed non-gradient compartments (e.g. narrow hair crevices, stained-glass pockets)
    # to a clean flat solid interior median colour so they never fracture into false gradient steps.
    small_flat_cap = max(2400, int(0.0005 * h * w))
    narrow_flat_cap = max(7200, int(0.0015 * h * w))
    max_narrow_r = 5.5 * max(1.0, (h * w / 900000.0) ** 0.5)
    lab_raw = srgb_to_oklab(rgb).astype(np.float32)
    for cid in range(1, n_cc):
        area = int(comp_stats[cid, cv2.CC_STAT_AREA])
        if area < 24 or area > narrow_flat_cap:
            continue
        x0, y0, bw, bh = (
            int(comp_stats[cid, cv2.CC_STAT_LEFT]),
            int(comp_stats[cid, cv2.CC_STAT_TOP]),
            int(comp_stats[cid, cv2.CC_STAT_WIDTH]),
            int(comp_stats[cid, cv2.CC_STAT_HEIGHT]),
        )
        sub = comp_lbl[y0:y0 + bh, x0:x0 + bw] == cid
        if area > small_flat_cap and float(dt_wall[y0:y0 + bh, x0:x0 + bw][sub].max()) > max_narrow_r:
            continue
        if protect[y0:y0 + bh, x0:x0 + bw][sub].mean() > 0.25 or grad_open[y0:y0 + bh, x0:x0 + bw][sub].mean() > 0.35:
            continue
        core_sub = sub & (dt_wall[y0:y0 + bh, x0:x0 + bw] >= 1.2)
        sample_m = core_sub if int(core_sub.sum()) >= 8 else sub
        l_vals = lab_raw[y0:y0 + bh, x0:x0 + bw, 0][sample_m]
        a_vals = lab_raw[y0:y0 + bh, x0:x0 + bw, 1][sample_m]
        b_vals = lab_raw[y0:y0 + bh, x0:x0 + bw, 2][sample_m]
        if float(np.std(l_vals)) < 0.048 and float(np.hypot(np.std(a_vals), np.std(b_vals))) < 0.032:
            med_rgb = np.rint(np.median(rgb[y0:y0 + bh, x0:x0 + bw][sample_m], axis=0)).astype(np.uint8)
            out[y0:y0 + bh, x0:x0 + bw][sub] = med_rgb

    if not resolved.any() or not target_rim.any():
        return out
    score_map = np.where(resolved, dt_wall, -1.0).astype(np.float32)

    # Iterative strictly same-compartment neighbor propagation (4-neighbors then 8-neighbors)
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _pass in range(2):
        if not target_rim.any():
            break
        best_score = np.full((h, w), -1.0, dtype=np.float32)
        best_rgb = out.copy()
        for dy, dx in offsets:
            y0_dst, y1_dst = max(0, dy), h + min(0, dy)
            x0_dst, x1_dst = max(0, dx), w + min(0, dx)
            y0_src, y1_src = max(0, -dy), h + min(0, -dy)
            x0_src, x1_src = max(0, -dx), w + min(0, -dx)

            same_comp = comp_lbl[y0_dst:y1_dst, x0_dst:x1_dst] == comp_lbl[y0_src:y1_src, x0_src:x1_src]
            cand_res = resolved[y0_src:y1_src, x0_src:x1_src] & same_comp & target_rim[y0_dst:y1_dst, x0_dst:x1_dst]
            if not cand_res.any():
                continue
            cand_sc = score_map[y0_src:y1_src, x0_src:x1_src] - (0.05 if (dy != 0 and dx != 0) else 0.0)
            better = cand_res & (cand_sc > best_score[y0_dst:y1_dst, x0_dst:x1_dst])
            if better.any():
                sub_bs = best_score[y0_dst:y1_dst, x0_dst:x1_dst]
                sub_bs[better] = cand_sc[better]
                sub_br = best_rgb[y0_dst:y1_dst, x0_dst:x1_dst]
                sub_br[better] = out[y0_src:y1_src, x0_src:x1_src][better]

        filled = target_rim & (best_score >= 0.0)
        if not filled.any():
            break
        out[filled] = best_rgb[filled]
        score_map[filled] = best_score[filled]
        resolved |= filled
        target_rim &= ~filled

    return out
