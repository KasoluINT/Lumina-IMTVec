"""Post-CNN Heatmap-Driven Line & Compartment Gradient Topology Engine.

Given the CNN heatmaps (DedicatedLineNet `line_net.onnx`, ElementGradNet `element_grad.onnx`,
FeaturePartUNet `feature_part.onnx`), enforces:
  1. Stage A — Accidental Coloring-Gap ("穿帮白线") Inpainting:
     Removes narrow 1-2.5px bright canvas slits pinched inside hair/clothing masses while preserving
     genuine skin gaps and eye highlights.
  2. Stage B — Anti-Aliasing Halo Worm, Chroma-Bled Tip, & Pinched `A -> B -> A` Elimination:
     Restores thin hair tips whose hue was desaturated by adjacent bright skin, absorbs 1-2px
     intermediate AA worms, and merges bi-laterally enveloped (`A -> B -> A`) or island patches.
  3. Stage C — Level-Set Curve Smoothing of Gradient Band Interfaces:
     Smooths internal gradient boundaries (partial A cap partial B) strictly inside A u B via
     Gaussian level-set filtering, keeping outer compartment boundaries 100% invariant.
  4. Stage D — DedicatedLineNet Sub-Pixel Thin Outer Edge Outline Recovery:
     Recovers 0.8-1.2px thin dark outer contour lines against bright skin/background that were
     weakened by sub-pixel alpha blending (0.65 * L_ink + 0.35 * L_bg) and swallowed into fills.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import label as sk_label

from .color import srgb_to_oklab, oklab_to_srgb


def enforce_global_gradient_topology(
    base: np.ndarray,
    source: np.ndarray,
    thin_lines: np.ndarray | None = None,
    fmaps=None,
    protect_mask: np.ndarray | None = None,
    sigma_interface: float = 2.6,
    wall_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Execute Stages A-D of the post-CNN heatmap-driven line and compartment topology engine."""
    h, w = base.shape[:2]
    out = base.copy()
    k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    lab_raw = srgb_to_oklab(source).astype(np.float32)
    L_raw = lab_raw[..., 0]
    lab_b = srgb_to_oklab(out).astype(np.float32)

    # Eye & facial micro-feature protection guard
    eye_guard = np.zeros((h, w), bool)
    if protect_mask is not None:
        eye_guard |= protect_mask
    if fmaps is not None and getattr(fmaps, "delicate_detail", None) is not None:
        eye_guard |= cv2.dilate((fmaps.delicate_detail > 0.28).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    # --- Stage A: Eliminate 1-2.5px Narrow "穿帮米白色白线" Slits Pinched Inside Chromatic Hair/Cloth ---
    # Strictly preserve clean-cel regions (E0 > 0.52 or cel_ratio > 0.50) and long white dress ribbons (span > 22)!
    cel_guard = np.zeros((h, w), bool)
    if fmaps is not None:
        if getattr(fmaps, "surface_probs", None) is not None:
            cel_guard |= fmaps.surface_probs[0] > 0.52
        if getattr(fmaps, "elem_grad_maps", None) is not None:
            cel_guard |= fmaps.elem_grad_maps[0] > 0.52

    bright_px = (lab_b[..., 0] > 0.85) & (~eye_guard) & (~cel_guard)
    if bright_px.any():
        k3 = np.ones((3, 3), np.uint8)
        thick_bright = cv2.dilate(cv2.erode(bright_px.astype(np.uint8), k3, iterations=1), k3, iterations=1) > 0
        thin_white_slits = bright_px & (~thick_bright)
        # Flank must be chromatic hair/cloth fill (0.52 <= L < 0.80, chroma > 0.016) - NEVER black ribbon outlines!
        chroma_b = np.hypot(lab_b[..., 1], lab_b[..., 2])
        is_chrom_hair = (lab_b[..., 0] >= 0.52) & (lab_b[..., 0] < 0.80) & (chroma_b > 0.016)
        p_dh = np.pad(is_chrom_hair, 2, mode="edge")
        flanked2 = (
            (p_dh[2:-2, :-4] & p_dh[2:-2, 4:])
            | (p_dh[:-4, 2:-2] & p_dh[4:, 2:-2])
            | (p_dh[:-4, 4:] & p_dh[4:, :-4])
            | (p_dh[:-4, :-4] & p_dh[4:, 4:])
        )
        raw_pinched = (cv2.dilate((thin_white_slits & flanked2).astype(np.uint8), k3) > 0) & thin_white_slits
        # Restrict strictly to short local slits (area <= 52, span <= 24px) so long white dress ribbons stay 100% intact
        n_pw, l_pw, st_pw, _ = cv2.connectedComponentsWithStats(raw_pinched.astype(np.uint8), connectivity=8)
        keep_pw = (st_pw[:, cv2.CC_STAT_AREA] <= 52) & (np.maximum(st_pw[:, cv2.CC_STAT_WIDTH], st_pw[:, cv2.CC_STAT_HEIGHT]) <= 24)
        keep_pw[0] = False
        pinched_white = keep_pw[l_pw]

        n_bw, l_bw, st_bw, _ = cv2.connectedComponentsWithStats(bright_px.astype(np.uint8), connectivity=4)
        cand_bw = np.flatnonzero((st_bw[:, cv2.CC_STAT_AREA] <= 52) & (np.maximum(st_bw[:, cv2.CC_STAT_WIDTH], st_bw[:, cv2.CC_STAT_HEIGHT]) <= 24))
        for bi in cand_bw:
            if bi == 0:
                continue
            bx0 = max(0, int(st_bw[bi, cv2.CC_STAT_LEFT]) - 1)
            by0 = max(0, int(st_bw[bi, cv2.CC_STAT_TOP]) - 1)
            bx1 = min(w, int(st_bw[bi, cv2.CC_STAT_LEFT] + st_bw[bi, cv2.CC_STAT_WIDTH]) + 1)
            by1 = min(h, int(st_bw[bi, cv2.CC_STAT_TOP] + st_bw[bi, cv2.CC_STAT_HEIGHT]) + 1)
            m_bi = l_bw[by0:by1, bx0:bx1] == bi
            dil_bi = (cv2.dilate(m_bi.astype(np.uint8), k4) > 0) & (~m_bi)
            if dil_bi.any() and float(is_chrom_hair[by0:by1, bx0:bx1][dil_bi].mean()) >= 0.65:
                pinched_white[by0:by1, bx0:bx1] |= m_bi

        if pinched_white.any():
            fill_pool = is_chrom_hair & (~pinched_white)
            if fill_pool.any():
                _, (iy_w, ix_w) = distance_transform_edt(~fill_pool, return_indices=True)
                out[pinched_white] = out[iy_w[pinched_white], ix_w[pinched_white]]
                lab_b = srgb_to_oklab(out).astype(np.float32)

    # --- Stage B: Source-Fidelity-Guarded Cleanup of AA Rim Worms & Pinched `A -> B -> A` Gradient Patches ---
    barrier = (lab_b[..., 0] < 0.36) | cel_guard
    if wall_mask is not None:
        barrier |= wall_mask
    if fmaps is not None and getattr(fmaps, "delicate_detail", None) is not None:
        barrier |= (fmaps.delicate_detail > 0.28) & (lab_b[..., 0] < 0.62)
    if fmaps is not None and getattr(fmaps, "line_net_maps", None) is not None:
        lm = fmaps.line_net_maps
        barrier |= lm[0] > 0.44
        if lm.shape[0] >= 3:
            barrier |= (lm[2] > 0.45) & (lm[0] < 0.20)
    bright = lab_b[..., 0] > 0.86
    n_g, l_g, st_g, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=4)
    if n_g > 1:
        keep_g = st_g[:, cv2.CC_STAT_AREA] <= 260
        keep_g[0] = False
        barrier |= keep_g[l_g]
    # Never swallow crisp white cel/specular highlights (L > 0.935) or sharp CNN color-block boundaries (L2)!
    barrier |= (lab_b[..., 0] > 0.935)
    if fmaps is not None and getattr(fmaps, "line_net_maps", None) is not None:
        barrier |= fmaps.line_net_maps[2] > 0.45

    # Compute source gradient/wash map strictly guarded away from dense line zones and crisp stained-glass mosaics
    grad_wash = np.zeros((h, w), np.float32)
    if fmaps is not None:
        base_wash = fmaps.mergeable_wash if getattr(fmaps, "mergeable_wash", None) is not None else np.zeros((h, w), np.float32)
        grad_wash = np.maximum(grad_wash, base_wash)
        if getattr(fmaps, "surface_probs", None) is not None:
            grad_wash = np.maximum(grad_wash, fmaps.surface_probs[1] + 0.8 * fmaps.surface_probs[2])
        if getattr(fmaps, "elem_grad_maps", None) is not None:
            # Combine ElementGradNet E1/E2 ONLY where mergeable_wash also supports soft gradient shading,
            # so stained-glass jewel facets / crisp cel mosaics (e.g. Tarot) keep 100% crisp borders!
            e_soft = np.maximum(fmaps.elem_grad_maps[1], fmaps.elem_grad_maps[2])
            wash_gate = np.clip((base_wash - 0.18) / 0.22, 0.0, 1.0)
            grad_wash = np.maximum(grad_wash, e_soft * wash_gate)
        if wall_mask is not None and wall_mask.any():
            a_gw = max(1.0, (h * w) / (1333.0 * 1769.0))
            l_gw = float(np.sqrt(a_gw))
            kd_gw = max(11, int(round(11.0 * l_gw)) | 1)
            w_dens = cv2.boxFilter(wall_mask.astype(np.float32), -1, (kd_gw, kd_gw), normalize=True)
            grad_wash = np.where(wall_mask | (w_dens >= 0.22), np.minimum(grad_wash, 0.18), grad_wash)

    for _iter in range(3):
        packed = (
            (out[..., 0].astype(np.int32) << 16)
            | (out[..., 1].astype(np.int32) << 8)
            | out[..., 2].astype(np.int32)
        )
        regs = sk_label(packed + 1, connectivity=1) - 1
        n_reg = int(regs.max()) + 1
        if n_reg <= 1:
            break

        areas = np.bincount(regs.ravel(), minlength=n_reg)
        reg_rgb = np.zeros((n_reg, 3), np.uint8)
        reg_rgb[regs.ravel()] = out.reshape(-1, 3)
        reg_lab = srgb_to_oklab(reg_rgb).astype(np.float32)
        reg_is_barrier = np.bincount(regs.ravel(), weights=barrier.ravel(), minlength=n_reg) > 0.35 * np.maximum(areas, 1)
        boxes = find_objects(regs + 1)

        reg_is_line = reg_is_barrier.copy()
        for r_i in range(n_reg):
            if reg_is_line[r_i] or areas[r_i] == 0 or reg_lab[r_i, 0] >= 0.60:
                continue
            bx = boxes[r_i]
            if bx is None:
                continue
            span = max(bx[0].stop - bx[0].start, bx[1].stop - bx[1].start)
            if span >= 8 and areas[r_i] <= 3.0 * span:
                reg_is_line[r_i] = True

        pairs = []
        tot_perim = np.zeros(n_reg, np.int32)
        for a, b in ((regs[:-1, :], regs[1:, :]), (regs[:, :-1], regs[:, 1:])):
            diff = a != b
            if diff.any():
                da, db = a[diff], b[diff]
                np.add.at(tot_perim, da, 1)
                np.add.at(tot_perim, db, 1)
                pairs.append(np.column_stack([da, db]))
        if not pairs:
            break
        all_p = np.concatenate(pairs, axis=0)
        all_p = np.vstack([all_p, all_p[:, ::-1]])
        uniq_p, counts_p = np.unique(all_p, axis=0, return_counts=True)
        order_p = np.argsort(uniq_p[:, 0])
        uniq_p, counts_p = uniq_p[order_p], counts_p[order_p]
        splits = np.searchsorted(uniq_p[:, 0], np.arange(n_reg + 1))

        merged_count = 0
        for r_b in np.argsort(areas):
            ar_b = int(areas[r_b])
            if ar_b == 0 or ar_b > 420 or reg_is_line[r_b]:
                continue
            box = boxes[r_b]
            if box is None:
                continue
            s0, s1 = splits[r_b], splits[r_b + 1]
            if s0 == s1:
                continue
            nb_ids = uniq_p[s0:s1, 1]
            nb_cnts = counts_p[s0:s1]
            perim_b = max(1, int(tot_perim[r_b]))
            lb = reg_lab[r_b]
            m_b = regs[box] == r_b
            src_px = source[box][m_b].astype(np.float32)
            rmse_keep = float(np.sqrt(np.mean((src_px - reg_rgb[r_b].astype(np.float32)) ** 2)))
            mean_cel_b = float(np.mean(fmaps.surface_probs[0][box][m_b])) if (fmaps is not None and getattr(fmaps, "surface_probs", None) is not None) else 0.0
            mean_wash_b = float(np.mean(grad_wash[box][m_b])) if fmaps is not None else 0.0

            # Never swallow crisp graphic cel regions (> 30px) in cel artwork (e.g. A5 flower rings)
            if ar_b > 30 and mean_cel_b > 0.32 and mean_wash_b < 0.45:
                continue

            # Case 1A: Tiny Chroma-Bled Tip / Rim (`色相被亮背景虚化污染的窄边`)
            if ar_b <= 48 and len(nb_ids) >= 2:
                bright_nb = (reg_lab[nb_ids, 0] - lb[0] >= 0.20) & (areas[nb_ids] > ar_b)
                fill_nb = (
                    (~reg_is_line[nb_ids])
                    & (areas[nb_ids] > ar_b * 2)
                    & (np.abs(reg_lab[nb_ids, 0] - lb[0]) <= 0.080)
                    & (np.linalg.norm(reg_lab[nb_ids] - lb[None, :], axis=1) <= 0.070)
                )
                if bright_nb.any() and fill_nb.any():
                    f_ids = nb_ids[fill_nb]
                    f_de = np.linalg.norm(reg_lab[f_ids] - lb[None, :], axis=1)
                    pick_a = int(f_ids[np.argmin(f_de)])
                    rmse_m = float(np.sqrt(np.mean((src_px - reg_rgb[pick_a].astype(np.float32)) ** 2)))
                    if rmse_m <= rmse_keep + 10.0:
                        out[box][m_b] = reg_rgb[pick_a]
                        merged_count += 1
                        continue

            # Case 1B: Thin 1-2px Anti-Aliasing Rim Worm
            if ar_b <= 45 and len(nb_ids) >= 2:
                nbs_l = reg_lab[nb_ids, 0]
                if nbs_l.min() < lb[0] - 0.05 and nbs_l.max() > lb[0] + 0.07 and (nbs_l.max() - nbs_l.min()) >= 0.20:
                    if int(cv2.erode(m_b.astype(np.uint8), k4, iterations=1).sum()) == 0:
                        cand_m = (~reg_is_line[nb_ids]) & (areas[nb_ids] > ar_b * 2) & (np.abs(reg_lab[nb_ids, 0] - lb[0]) <= 0.10)
                        if cand_m.any():
                            c_ids = nb_ids[cand_m]
                            c_cnts = nb_cnts[cand_m]
                            pick_a = int(c_ids[np.argmax(c_cnts)])
                            rmse_m = float(np.sqrt(np.mean((src_px - reg_rgb[pick_a].astype(np.float32)) ** 2)))
                            if rmse_m <= rmse_keep + 10.0:
                                out[box][m_b] = reg_rgb[pick_a]
                                merged_count += 1
                                continue

            # Case 2: Pinched / Bi-laterally Enveloped Gradient Patch (`色块 B 两侧被色块 A 包住`)
            # For patches > 40px, restrict strictly to CNN-detected gradient/wash zones (`mean_wash_b > 0.36 & mean_cel_b < 0.38`)
            in_wash = (mean_wash_b > 0.36 and mean_cel_b < 0.38) if fmaps is not None else False
            if ar_b > 40 and not in_wash:
                continue

            non_bar = ~reg_is_line[nb_ids]
            if not non_bar.any():
                continue
            de_all = np.linalg.norm(reg_lab[nb_ids] - lb[None, :], axis=1)
            dc_all = np.hypot(reg_lab[nb_ids, 1] - lb[1], reg_lab[nb_ids, 2] - lb[2])
            compat = non_bar & (de_all < 0.085) & (dc_all < 0.030) & (areas[nb_ids] > ar_b)
            if not compat.any():
                continue
            c_ids = nb_ids[compat]
            c_cnts = nb_cnts[compat]
            best_i = int(np.argmax(c_cnts))
            r_a = int(c_ids[best_i])
            shared_ab = int(c_cnts[best_i])
            wrap_frac = shared_ab / perim_b

            if protect_mask is not None and ar_b >= 20:
                if float(np.mean(protect_mask[box][m_b])) > 0.45:
                    if float(np.linalg.norm(reg_lab[r_a] - lb)) > 0.050:
                        continue

            rmse_m = float(np.sqrt(np.mean((src_px - reg_rgb[r_a].astype(np.float32)) ** 2)))
            max_delta = 10.0 if ar_b <= 30 else 7.5
            if rmse_m - rmse_keep > max_delta:
                continue

            is_pinched = wrap_frac >= 0.54
            if not is_pinched and ar_b < 0.42 * int(areas[r_a]) and shared_ab >= 0.34 * perim_b:
                ya = slice(max(0, box[0].start - 2), min(h, box[0].stop + 2))
                xa = slice(max(0, box[1].start - 2), min(w, box[1].stop + 2))
                sub_r = regs[ya, xa]
                ma = sub_r == r_a
                mb = sub_r == r_b
                u_ab = ma | mb
                u_er = cv2.erode(u_ab.astype(np.uint8), k4, iterations=1) > 0
                u_border = u_ab & (~u_er)
                b_on_border = mb & u_border
                if not b_on_border.any():
                    is_pinched = True
                elif in_wash:
                    ab_interface = (cv2.dilate(ma.astype(np.uint8), k4) > 0) & mb
                    iy, ix = np.where(ab_interface)
                    if len(iy) >= 4:
                        pts = np.column_stack([ix, iy]).astype(np.float32)
                        d0 = np.linalg.norm(pts - pts[0], axis=1)
                        p1 = pts[int(np.argmax(d0))]
                        chord = float(np.max(np.linalg.norm(pts - p1, axis=1)))
                        arc_len = float(len(iy))
                        if arc_len > 1.32 * max(chord, 1.0):
                            is_pinched = True
                    if not is_pinched and len(c_ids) == 1 and ar_b < 140:
                        is_pinched = True

            if is_pinched and wall_mask is not None and ar_b >= 12:
                ab_dil = ((cv2.dilate((regs[box] == r_a).astype(np.uint8), k4) > 0) & m_b)
                if ab_dil.any() and float(wall_mask[box][ab_dil].mean()) >= 0.35:
                    is_pinched = False

            if is_pinched:
                out[box][m_b] = reg_rgb[r_a]
                merged_count += 1

        if merged_count == 0:
            break

    # --- Stage C: Level-Set Curve Smoothing of Gradient Band Interfaces Strictly Inside CNN Gradient Zones ---
    sigma_grad = min(2.0, sigma_interface)
    for _s_iter in range(2):
        packed = (
            (out[..., 0].astype(np.int32) << 16)
            | (out[..., 1].astype(np.int32) << 8)
            | out[..., 2].astype(np.int32)
        )
        regs = sk_label(packed + 1, connectivity=1) - 1
        n_reg = int(regs.max()) + 1
        if n_reg <= 1:
            break
        areas = np.bincount(regs.ravel(), minlength=n_reg)
        reg_rgb = np.zeros((n_reg, 3), np.uint8)
        reg_rgb[regs.ravel()] = out.reshape(-1, 3)
        reg_lab = srgb_to_oklab(reg_rgb).astype(np.float32)
        reg_is_barrier = np.bincount(regs.ravel(), weights=barrier.ravel(), minlength=n_reg) > 0.35 * np.maximum(areas, 1)
        boxes = find_objects(regs + 1)

        reg_is_line = reg_is_barrier.copy()
        for r_i in range(n_reg):
            if reg_is_line[r_i] or areas[r_i] == 0 or reg_lab[r_i, 0] >= 0.60:
                continue
            bx = boxes[r_i]
            if bx is None:
                continue
            span = max(bx[0].stop - bx[0].start, bx[1].stop - bx[1].start)
            if span >= 8 and areas[r_i] <= 3.0 * span:
                reg_is_line[r_i] = True

        pairs = []
        for a, b in ((regs[:-1, :], regs[1:, :]), (regs[:, :-1], regs[:, 1:])):
            diff = a < b
            if diff.any():
                pairs.append(np.column_stack([a[diff], b[diff]]))
        if not pairs:
            break
        uniq_p, counts_p = np.unique(np.concatenate(pairs, axis=0), axis=0, return_counts=True)
        ks = int(round(sigma_grad * 3)) * 2 + 1
        for (ra, rb), cnt in zip(uniq_p, counts_p):
            if cnt < 6 or areas[ra] < 24 or areas[rb] < 24:
                continue
            if reg_is_line[ra] or reg_is_line[rb]:
                continue
            if (reg_lab[ra, 0] > 0.88) != (reg_lab[rb, 0] > 0.88):
                continue
            de = float(np.linalg.norm(reg_lab[ra] - reg_lab[rb]))
            dc = float(np.hypot(reg_lab[ra, 1] - reg_lab[rb, 1], reg_lab[ra, 2] - reg_lab[rb, 2]))
            if de >= 0.115 or dc >= 0.036:
                continue
            ba, bb = boxes[ra], boxes[rb]
            y0 = max(0, min(ba[0].start, bb[0].start) - 4)
            y1 = min(h, max(ba[0].stop, bb[0].stop) + 4)
            x0 = max(0, min(ba[1].start, bb[1].start) - 4)
            x1 = min(w, max(ba[1].stop, bb[1].stop) + 4)
            sub_r = regs[y0:y1, x0:x1]
            ma = sub_r == ra
            mb = sub_r == rb
            u_ab = ma | mb
            if not u_ab.any():
                continue
            if fmaps is not None:
                if float(np.mean(grad_wash[y0:y1, x0:x1][u_ab])) < 0.25:
                    continue
                if float(np.mean(cel_guard[y0:y1, x0:x1][u_ab])) > 0.45:
                    continue
            if wall_mask is not None:
                ab_iface = (cv2.dilate(ma.astype(np.uint8), k4) > 0) & mb
                if ab_iface.any() and float(wall_mask[y0:y1, x0:x1][ab_iface].mean()) >= 0.25:
                    continue
            field = ma.astype(np.float32) - mb.astype(np.float32)
            sm_f = cv2.GaussianBlur(field, (ks, ks), sigma_grad)
            choose_a = u_ab & (sm_f >= 0.0)
            choose_b = u_ab & (sm_f < 0.0)
            if choose_a.any() and choose_b.any():
                sub_out = out[y0:y1, x0:x1]
                sub_out[choose_a] = reg_rgb[ra]
                sub_out[choose_b] = reg_rgb[rb]

    # --- Stage E: Slope-Adaptive Cel-Gradient Puddle/Staircase Consolidation & 1D Compartment Color-Axis Harmonization ---
    wash_area_global = float(fmaps.summary.get("wash_area", 0.0)) if (fmaps is not None and getattr(fmaps, "summary", None)) else 0.0
    out = _harmonize_cel_gradient_compartments(out, source, lab_raw, barrier, grad_wash, cel_guard, wall_mask, wash_area_global=wash_area_global)
    out = _smooth_radial_blush_single_arc(out, lab_raw, barrier, cel_guard, wall_mask)
    return out


def _harmonize_cel_gradient_compartments(
    out: np.ndarray,
    source: np.ndarray,
    lab_raw: np.ndarray,
    barrier: np.ndarray,
    grad_wash: np.ndarray,
    cel_guard: np.ndarray,
    wall_mask: np.ndarray | None = None,
    wash_area_global: float = 0.0,
) -> np.ndarray:
    """Eliminate topographic puddles, dense staircase ribbons, global K-means cross-talk (e.g. cold bedsheet
    gray bleeding into warm sheer stockings), and harsh Mach-band jumps inside smooth gradient compartments."""
    # Never alter stained-glass / intricate linework illustrations where soft gradient wash is a minor accent
    if wash_area_global < 0.16 and float((grad_wash >= 0.55).mean()) < 0.16:
        return out
    h, w = out.shape[:2]
    a_sc = max(1.0, (h * w) / (1333.0 * 1769.0))
    l_sc = float(np.sqrt(a_sc))

    # Measure edge-preserved structural slope ||grad L|| per 1x source pixel strictly on NON-INK
    # pixels so dark ink outlines never spike the rim-shadow slope beside ink edges!
    ink_dil = cv2.dilate((lab_raw[..., 0] < 0.35).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if ink_dil.any() and not ink_dil.all():
        _, (iy_ni, ix_ni) = distance_transform_edt(ink_dil, return_indices=True)
        lab_noink = lab_raw.copy()
        lab_noink[ink_dil] = lab_raw[iy_ni[ink_dil], ix_ni[ink_dil]]
    else:
        lab_noink = lab_raw
    sm_lab = cv2.GaussianBlur(lab_noink, (0, 0), 1.4 * l_sc)
    gx = cv2.Sobel(sm_lab[..., 0], cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(sm_lab[..., 0], cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    gcx = np.hypot(cv2.Sobel(sm_lab[..., 1], cv2.CV_32F, 1, 0, ksize=3, scale=0.125),
                   cv2.Sobel(sm_lab[..., 2], cv2.CV_32F, 1, 0, ksize=3, scale=0.125))
    gcy = np.hypot(cv2.Sobel(sm_lab[..., 1], cv2.CV_32F, 0, 1, ksize=3, scale=0.125),
                   cv2.Sobel(sm_lab[..., 2], cv2.CV_32F, 0, 1, ksize=3, scale=0.125))
    slope = (np.hypot(gx, gy) + 2.0 * np.hypot(gcx, gcy)) * l_sc

    work = out.copy()
    for _pass in range(3):
        packed = (work[..., 0].astype(np.int32) << 16) | (work[..., 1].astype(np.int32) << 8) | work[..., 2].astype(np.int32)
        regs = sk_label(packed + 1, connectivity=1) - 1
        n_r = int(regs.max()) + 1
        if n_r <= 2:
            break
        areas = np.bincount(regs.ravel(), minlength=n_r)
        r_rgb = np.zeros((n_r, 3), np.uint8)
        r_rgb[regs.ravel()] = work.reshape(-1, 3)
        r_lab = srgb_to_oklab(r_rgb).astype(np.float32)

        edge = np.zeros((h, w), bool)
        edge[1:] |= regs[1:] != regs[:-1]; edge[:-1] |= regs[1:] != regs[:-1]
        edge[:, 1:] |= regs[:, 1:] != regs[:, :-1]; edge[:, :-1] |= regs[:, 1:] != regs[:, :-1]
        dist_in = distance_transform_edt(~edge)
        thick = np.zeros(n_r, np.float32)
        np.maximum.at(thick, regs.ravel(), dist_in.ravel())

        boxes_e = find_objects(regs + 1)
        spans_e = np.array([max(bx[0].stop - bx[0].start, bx[1].stop - bx[1].start) if bx is not None else 0 for bx in boxes_e])
        reg_bar = np.bincount(regs.ravel(), weights=barrier.ravel(), minlength=n_r) > 0.35 * np.maximum(areas, 1)
        reg_cel = np.bincount(regs.ravel(), weights=cel_guard.ravel(), minlength=n_r) > 0.45 * np.maximum(areas, 1)
        reg_wash = np.bincount(regs.ravel(), weights=grad_wash.ravel(), minlength=n_r) / np.maximum(areas, 1)
        # Strictly protect ALL thin/elongated lines (dark ink OR pastel/pink/white hair wisps) regardless of L!
        is_line = reg_bar | (r_lab[:, 0] < 0.38) | ((thick <= 3.6 * l_sc) & (spans_e >= 6.0 * l_sc))

        k1_list, k2_list, b_slopes, b_walls = [], [], [], []
        for (a, b, sl_s, w_s) in (
            (regs[:-1, :], regs[1:, :], np.maximum(slope[:-1, :], slope[1:, :]), (wall_mask[:-1, :] | wall_mask[1:, :]) if wall_mask is not None else None),
            (regs[:, :-1], regs[:, 1:], np.maximum(slope[:, :-1], slope[:, 1:]), (wall_mask[:, :-1] | wall_mask[:, 1:]) if wall_mask is not None else None),
        ):
            diff = a != b
            if diff.any():
                va, vb = a[diff].astype(np.uint64), b[diff].astype(np.uint64)
                k1_list.append((va << 32) | vb)
                k2_list.append((vb << 32) | va)
                b_slopes.append(sl_s[diff])
                if w_s is not None:
                    b_walls.append(w_s[diff].astype(np.float32))
        if not k1_list:
            break
        bi_k = np.concatenate(k1_list + k2_list)
        all_s = np.concatenate(b_slopes, axis=0)
        bi_s = np.concatenate([all_s, all_s])
        uniq_k, inv, contact = np.unique(bi_k, return_inverse=True, return_counts=True)
        ra = (uniq_k >> 32).astype(np.int32)
        rb = (uniq_k & 0xffffffff).astype(np.int32)
        mean_sl = np.bincount(inv, weights=bi_s, minlength=len(uniq_k)) / np.maximum(contact, 1)
        if b_walls:
            all_w = np.concatenate(b_walls, axis=0)
            bi_w = np.concatenate([all_w, all_w])
            wall_frac = np.bincount(inv, weights=bi_w, minlength=len(uniq_k)) / np.maximum(contact, 1)
        else:
            wall_frac = np.zeros(len(uniq_k), np.float32)
        perim = np.bincount(ra, weights=contact, minlength=n_r)
        perim_nl = np.bincount(ra[~is_line[rb]], weights=contact[~is_line[rb]], minlength=n_r)

        share = contact / np.maximum(perim[ra], 1)
        share_nl = contact / np.maximum(perim_nl[ra], 1)
        dL = np.abs(r_lab[ra, 0] - r_lab[rb, 0])
        dC = np.hypot(r_lab[ra, 1] - r_lab[rb, 1], r_lab[ra, 2] - r_lab[rb, 2])
        dE = np.hypot(dL, dC * 1.8)

        in_grad = ((reg_wash[ra] >= 0.36) | (reg_wash[rb] >= 0.36)) & (~reg_cel[ra])
        soft_border = in_grad & (~is_line[ra]) & (~is_line[rb]) & (wall_frac < 0.25) & (dC < 0.030) & (mean_sl < 0.015)

        # 1) Absorb isolated puddles / knee bullseyes inside low-slope smooth gradients
        is_puddle = soft_border & (areas[rb] > areas[ra]) & (
            ((share >= 0.72) & (areas[ra] < 4200 * a_sc) & (dL < 0.080))
            | ((share >= 0.48) & (areas[ra] < 1600 * a_sc) & (dL < 0.072))
            | ((areas[ra] < 380 * a_sc) & (dL < 0.080))
        )
        # 2) Absorb jagged ink-edge nibs/bays tucked between an ink outline and a dominant smooth fill
        is_ink_nib = in_grad & (~is_line[ra]) & (~is_line[rb]) & (areas[rb] > areas[ra] * 2.0) & (
            (areas[ra] < 550 * a_sc) & (dC < 0.032) & (dL < 0.16) & (share_nl >= 0.62) & (mean_sl < 0.025)
        )
        # 3) Collapse narrow intermediate staircase ribbons (thick <= 6.5 * l_sc)
        is_narrow_stair = in_grad & (~is_line[ra]) & (~is_line[rb]) & (wall_frac < 0.25) & (dC < 0.030) & (
            (thick[ra] <= 6.5 * l_sc) & (areas[rb] > areas[ra] * 1.4) & (dL < 0.075) & (share >= 0.22) & (mean_sl < 0.018)
        )
        # 4) Consolidate overly dense smooth-gradient staircase steps into clean anime gradient bands
        is_micro_step = soft_border & (areas[rb] > areas[ra]) & (dL < 0.056) & (share >= 0.16)

        ok = is_puddle | is_ink_nib | is_narrow_stair | is_micro_step
        if not ok.any():
            break
        ra_ok, rb_ok, dE_ok, sh_ok = ra[ok], rb[ok], dE[ok], share[ok]
        order = np.lexsort((-sh_ok, dE_ok, ra_ok))
        ra_ok, rb_ok = ra_ok[order], rb_ok[order]
        first = np.r_[True, ra_ok[1:] != ra_ok[:-1]]
        mapping = np.arange(n_r)
        mapping[ra_ok[first]] = rb_ok[first]
        for _ in range(16):
            nxt = mapping[mapping]
            if np.array_equal(nxt, mapping):
                break
            mapping = nxt
        work = r_rgb[mapping[regs]]

    # Final 1D Compartment Chromaticity Alignment & Mach-Band Contrast Damping
    packed = (work[..., 0].astype(np.int32) << 16) | (work[..., 1].astype(np.int32) << 8) | work[..., 2].astype(np.int32)
    regs = sk_label(packed + 1, connectivity=1) - 1
    n_r = int(regs.max()) + 1
    if n_r <= 2:
        return work
    areas = np.bincount(regs.ravel(), minlength=n_r)
    r_rgb = np.zeros((n_r, 3), np.uint8)
    r_rgb[regs.ravel()] = work.reshape(-1, 3)
    r_lab = srgb_to_oklab(r_rgb).astype(np.float32)
    boxes = find_objects(regs + 1)

    edge_f = np.zeros((h, w), bool)
    edge_f[1:] |= regs[1:] != regs[:-1]; edge_f[:-1] |= regs[1:] != regs[:-1]
    edge_f[:, 1:] |= regs[:, 1:] != regs[:, :-1]; edge_f[:, :-1] |= regs[:, 1:] != regs[:, :-1]
    thick_f = np.zeros(n_r, np.float32)
    np.maximum.at(thick_f, regs.ravel(), distance_transform_edt(~edge_f).ravel())
    spans_f = np.array([max(bx[0].stop - bx[0].start, bx[1].stop - bx[1].start) if bx is not None else 0 for bx in boxes])

    reg_bar = np.bincount(regs.ravel(), weights=barrier.ravel(), minlength=n_r) > 0.35 * np.maximum(areas, 1)
    reg_cel = np.bincount(regs.ravel(), weights=cel_guard.ravel(), minlength=n_r) > 0.45 * np.maximum(areas, 1)
    reg_wash = np.bincount(regs.ravel(), weights=grad_wash.ravel(), minlength=n_r) / np.maximum(areas, 1)
    is_line = reg_bar | (r_lab[:, 0] < 0.38) | ((thick_f <= 3.6 * l_sc) & (spans_f >= 6.0 * l_sc))

    parent = np.arange(n_r)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    k1_f, b_slopes, b_walls = [], [], []
    for (a, b, sl_s, w_s) in (
        (regs[:-1, :], regs[1:, :], np.maximum(slope[:-1, :], slope[1:, :]), (wall_mask[:-1, :] | wall_mask[1:, :]) if wall_mask is not None else None),
        (regs[:, :-1], regs[:, 1:], np.maximum(slope[:, :-1], slope[:, 1:]), (wall_mask[:, :-1] | wall_mask[:, 1:]) if wall_mask is not None else None),
    ):
        diff = a < b
        if diff.any():
            k1_f.append((a[diff].astype(np.uint64) << 32) | b[diff].astype(np.uint64))
            b_slopes.append(sl_s[diff])
            if w_s is not None:
                b_walls.append(w_s[diff].astype(np.float32))
    if not k1_f:
        return work

    uk_f, inv, contact = np.unique(np.concatenate(k1_f), return_inverse=True, return_counts=True)
    ra_f = (uk_f >> 32).astype(np.int32)
    rb_f = (uk_f & 0xffffffff).astype(np.int32)
    mean_sl = np.bincount(inv, weights=np.concatenate(b_slopes), minlength=len(uk_f)) / np.maximum(contact, 1)
    if b_walls:
        wall_frac = np.bincount(inv, weights=np.concatenate(b_walls), minlength=len(uk_f)) / np.maximum(contact, 1)
    else:
        wall_frac = np.zeros(len(uk_f), np.float32)

    soft_adj = [[] for _ in range(n_r)]
    for ra, rb, cnt, msl, wf in zip(ra_f, rb_f, contact, mean_sl, wall_frac):
        if cnt >= int(8 * l_sc) and not is_line[ra] and not is_line[rb] and wf < 0.25:
            if (reg_wash[ra] < 0.36 and reg_wash[rb] < 0.36) or (reg_cel[ra] and reg_cel[rb]):
                continue
            dC = float(np.hypot(r_lab[ra, 1] - r_lab[rb, 1], r_lab[ra, 2] - r_lab[rb, 2]))
            dL = float(abs(r_lab[ra, 0] - r_lab[rb, 0]))
            if dC < 0.032 and dL < 0.19:
                parent[find(ra)] = find(rb)
                if msl < 0.022:
                    soft_adj[ra].append((rb, cnt))
                    soft_adj[rb].append((ra, cnt))

    fams = {}
    for r in range(n_r):
        if areas[r] > 0 and not is_line[r]:
            fams.setdefault(find(r), []).append(r)

    harm_lab = r_lab.copy()
    changed_any = False
    k3_er = np.ones((3, 3), np.uint8)
    for _, members in fams.items():
        if len(members) < 2:
            continue
        m_idx = np.array(members, np.int32)
        # Re-sample each band's TRUE interior source median color (eroding away 1px ink AA fringe)
        for r in m_idx:
            bx = boxes[r]
            if bx is not None:
                m = regs[bx] == r
                m_in = cv2.erode(m.astype(np.uint8), k3_er) > 0
                use_m = m_in if int(m_in.sum()) >= 12 else m
                harm_lab[r] = np.median(lab_raw[bx][use_m], axis=0)

        w_m = areas[m_idx].astype(np.float64)
        w_m /= w_m.sum()
        L_m = harm_lab[m_idx, 0].astype(np.float64)
        L_mean = float(np.sum(w_m * L_m))
        L_var = float(np.sum(w_m * (L_m - L_mean) ** 2))
        for ch in (1, 2):
            c_m = harm_lab[m_idx, ch].astype(np.float64)
            c_mean = float(np.sum(w_m * c_m))
            slope_c = float(np.clip(np.sum(w_m * (L_m - L_mean) * (c_m - c_mean)) / max(L_var, 1e-4), -0.10, 0.10))
            harm_lab[m_idx, ch] = (0.85 * (c_mean + slope_c * (L_m - L_mean)) + 0.15 * c_m).astype(np.float32)

        L_new = harm_lab[m_idx, 0].copy()
        for i_m, r in enumerate(m_idx):
            if not soft_adj[r]:
                continue
            nbs = [nb for nb, _ in soft_adj[r]]
            wts = np.array([cnt for _, cnt in soft_adj[r]], np.float32)
            wts /= wts.sum()
            nb_L = float(np.sum(wts * harm_lab[nbs, 0]))
            L_new[i_m] = 0.72 * harm_lab[r, 0] + 0.28 * nb_L
        harm_lab[m_idx, 0] = L_new
        changed_any = True

    if not changed_any:
        res_work = work
    else:
        final_cols = np.clip(oklab_to_srgb(harm_lab), 0, 255).astype(np.uint8)
        final_cols[is_line] = r_rgb[is_line]
        res_work = final_cols[regs]

    # --- Post-Stage-E Simultaneous Multi-Label Family Level-Set Flow ---
    # Smooths ALL internal boundaries inside each smooth-gradient material family simultaneously
    # (zero 3-band triple-junction spikes and zero ragged ink-rim nibs), while strictly locking
    # ink lines, wall_mask, delicate zones, and steep cast-shadow corners (> 2.5x).
    sigma_post = 3.8 * l_sc
    packed_p = (res_work[..., 0].astype(np.int32) << 16) | (res_work[..., 1].astype(np.int32) << 8) | res_work[..., 2].astype(np.int32)
    regs_p = sk_label(packed_p + 1, connectivity=1) - 1
    n_rp = int(regs_p.max()) + 1
    if n_rp > 2:
        areas_p = np.bincount(regs_p.ravel(), minlength=n_rp)
        rgb_p = np.zeros((n_rp, 3), np.uint8)
        rgb_p[regs_p.ravel()] = res_work.reshape(-1, 3)
        lab_p = srgb_to_oklab(rgb_p).astype(np.float32)
        boxes_p = find_objects(regs_p + 1)
        bar_p = np.bincount(regs_p.ravel(), weights=barrier.ravel(), minlength=n_rp) > 0.35 * np.maximum(areas_p, 1)
        cel_p = np.bincount(regs_p.ravel(), weights=cel_guard.ravel(), minlength=n_rp) > 0.45 * np.maximum(areas_p, 1)
        wash_p = np.bincount(regs_p.ravel(), weights=grad_wash.ravel(), minlength=n_rp) / np.maximum(areas_p, 1)
        line_p = bar_p | (lab_p[:, 0] < 0.38) | ((areas_p < 500 * a_sc) & (lab_p[:, 0] >= 0.72) & (wash_p < 0.52))

        # Group connected gradient-band regions belonging to the same material family (`dC < 0.032, dL < 0.19`)
        parent_p = np.arange(n_rp)
        def find_p(x):
            while parent_p[x] != x:
                parent_p[x] = parent_p[parent_p[x]]
                x = parent_p[x]
            return x

        kp_list, bsl_p, bw_p = [], [], []
        for (a, b, sl_s, w_s) in (
            (regs_p[:-1, :], regs_p[1:, :], np.maximum(slope[:-1, :], slope[1:, :]), (wall_mask[:-1, :] | wall_mask[1:, :]) if wall_mask is not None else None),
            (regs_p[:, :-1], regs_p[:, 1:], np.maximum(slope[:, :-1], slope[:, 1:]), (wall_mask[:, :-1] | wall_mask[:, 1:]) if wall_mask is not None else None),
        ):
            diff = a < b
            if diff.any():
                kp_list.append((a[diff].astype(np.uint64) << 32) | b[diff].astype(np.uint64))
                bsl_p.append(sl_s[diff])
                if w_s is not None:
                    bw_p.append(w_s[diff].astype(np.float32))
        if kp_list:
            uk_p, inv_p, cnt_p = np.unique(np.concatenate(kp_list), return_inverse=True, return_counts=True)
            ra_p = (uk_p >> 32).astype(np.int32)
            rb_p = (uk_p & 0xffffffff).astype(np.int32)
            msl_p = np.bincount(inv_p, weights=np.concatenate(bsl_p), minlength=len(uk_p)) / np.maximum(cnt_p, 1)
            wf_p = (np.bincount(inv_p, weights=np.concatenate(bw_p), minlength=len(uk_p)) / np.maximum(cnt_p, 1)) if bw_p else np.zeros(len(uk_p), np.float32)

            for ra, rb, cnt, msl, wf in zip(ra_p, rb_p, cnt_p, msl_p, wf_p):
                if cnt < int(8 * l_sc) or line_p[ra] or line_p[rb] or areas_p[ra] < 32 * a_sc or areas_p[rb] < 32 * a_sc:
                    continue
                if wf >= 0.25 or (cel_p[ra] and cel_p[rb]) or (wash_p[ra] < 0.36 and wash_p[rb] < 0.36):
                    continue
                dC = float(np.hypot(lab_p[ra, 1] - lab_p[rb, 1], lab_p[ra, 2] - lab_p[rb, 2]))
                dL = float(abs(lab_p[ra, 0] - lab_p[rb, 0]))
                if dC < 0.032 and dL < 0.19 and msl < 0.036:
                    parent_p[find_p(ra)] = find_p(rb)

            fams_p = {}
            for r in range(n_rp):
                if areas_p[r] > 0 and not line_p[r]:
                    fams_p.setdefault(find_p(r), []).append(r)

            pad_post = int(round(sigma_post * 3.0))
            for members in fams_p.values():
                if len(members) < 2:
                    continue
                y0 = min(boxes_p[r][0].start for r in members if boxes_p[r] is not None)
                y1 = max(boxes_p[r][0].stop for r in members if boxes_p[r] is not None)
                x0 = min(boxes_p[r][1].start for r in members if boxes_p[r] is not None)
                x1 = max(boxes_p[r][1].stop for r in members if boxes_p[r] is not None)
                y0 = max(0, y0 - pad_post); y1 = min(h, y1 + pad_post)
                x0 = max(0, x0 - pad_post); x1 = min(w, x1 + pad_post)
                sub_r = regs_p[y0:y1, x0:x1]
                # Lock pixels that are outside this family (ink lines, walls, or steep cast shadows)
                fam_mask = np.isin(sub_r, members)
                if not fam_mask.any():
                    continue
                best_prob = np.full(sub_r.shape, -1.0, dtype=np.float32)
                best_lbl = sub_r.copy()
                # Normalized multi-label Gaussian probability vote strictly inside `fam_mask`
                for r in members:
                    ind = (sub_r == r).astype(np.float32)
                    p = cv2.GaussianBlur(ind, (0, 0), sigma_post)
                    upd = fam_mask & (p > best_prob)
                    best_lbl[upd] = r
                    best_prob[upd] = p[upd]
                res_work[y0:y1, x0:x1][fam_mask] = rgb_p[best_lbl[fam_mask]]

    return res_work


def _smooth_radial_blush_single_arc(
    img_in: np.ndarray,
    lab_raw: np.ndarray,
    barrier: np.ndarray,
    cel_guard: np.ndarray,
    wall_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Single-Arc Mean-Curvature Flow strictly for ultra-low-slope radial cheek blushes / soft glows
    so multi-lobed quantization waves collapse into ONE clean arc without shrinking or erasing blush."""
    res_work = img_in.copy()
    h, w = res_work.shape[:2]
    a_sc = max(1.0, (h * w) / (1333.0 * 1769.0))
    l_sc = float(np.sqrt(a_sc))

    ink_dil = cv2.dilate((lab_raw[..., 0] < 0.35).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if ink_dil.any() and not ink_dil.all():
        _, (iy_ni, ix_ni) = distance_transform_edt(ink_dil, return_indices=True)
        lab_noink = lab_raw.copy()
        lab_noink[ink_dil] = lab_raw[iy_ni[ink_dil], ix_ni[ink_dil]]
    else:
        lab_noink = lab_raw
    sm_lab = cv2.GaussianBlur(lab_noink, (0, 0), 1.4 * l_sc)
    gx = cv2.Sobel(sm_lab[..., 0], cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(sm_lab[..., 0], cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    slope = np.hypot(gx, gy) * l_sc

    packed_b = (res_work[..., 0].astype(np.int32) << 16) | (res_work[..., 1].astype(np.int32) << 8) | res_work[..., 2].astype(np.int32)
    regs_b = sk_label(packed_b + 1, connectivity=1) - 1
    n_rb = int(regs_b.max()) + 1
    if n_rb <= 2:
        return res_work
    areas_b = np.bincount(regs_b.ravel(), minlength=n_rb)
    rgb_b = np.zeros((n_rb, 3), np.uint8)
    rgb_b[regs_b.ravel()] = res_work.reshape(-1, 3)
    lab_b = srgb_to_oklab(rgb_b).astype(np.float32)
    boxes_b = find_objects(regs_b + 1)
    bar_b = np.bincount(regs_b.ravel(), weights=barrier.ravel(), minlength=n_rb) > 0.35 * np.maximum(areas_b, 1)
    cel_b = np.bincount(regs_b.ravel(), weights=cel_guard.ravel(), minlength=n_rb) > 0.68 * np.maximum(areas_b, 1)
    line_b = bar_b | (lab_b[:, 0] < 0.38)
    parent_b = np.arange(n_rb)
    def find_b(x):
        while parent_b[x] != x:
            parent_b[x] = parent_b[parent_b[x]]
            x = parent_b[x]
        return x
    kb_list, bsl_b, bwb_list = [], [], []
    for (a, b, sl_s, w_s) in (
        (regs_b[:-1, :], regs_b[1:, :], np.maximum(slope[:-1, :], slope[1:, :]), (wall_mask[:-1, :] | wall_mask[1:, :]) if wall_mask is not None else None),
        (regs_b[:, :-1], regs_b[:, 1:], np.maximum(slope[:, :-1], slope[:, 1:]), (wall_mask[:, :-1] | wall_mask[:, 1:]) if wall_mask is not None else None),
    ):
        diff = a != b
        if diff.any():
            pa = np.minimum(a[diff], b[diff]).astype(np.uint64)
            pb = np.maximum(a[diff], b[diff]).astype(np.uint64)
            kb_list.append((pa << 32) | pb)
            bsl_b.append(sl_s[diff])
            if w_s is not None:
                bwb_list.append(w_s[diff].astype(np.float32))
    if not kb_list:
        return res_work
    uk_b, inv_b, cnt_b = np.unique(np.concatenate(kb_list), return_inverse=True, return_counts=True)
    ra_b = (uk_b >> 32).astype(np.int32)
    rb_b = (uk_b & 0xffffffff).astype(np.int32)
    msl_b = np.bincount(inv_b, weights=np.concatenate(bsl_b), minlength=len(uk_b)) / np.maximum(cnt_b, 1)
    wf_b = (np.bincount(inv_b, weights=np.concatenate(bwb_list), minlength=len(uk_b)) / np.maximum(cnt_b, 1)) if bwb_list else np.zeros(len(uk_b), np.float32)
    for ra, rb, cnt, msl, wf in zip(ra_b, rb_b, cnt_b, msl_b, wf_b):
        if cnt < int(14 * l_sc) or line_b[ra] or line_b[rb] or areas_b[ra] < 120 * a_sc or areas_b[rb] < 120 * a_sc:
            continue
        if wf >= 0.20 or (cel_b[ra] and cel_b[rb]):
            continue
        dC = float(np.hypot(lab_b[ra, 1] - lab_b[rb, 1], lab_b[ra, 2] - lab_b[rb, 2]))
        dL = float(abs(lab_b[ra, 0] - lab_b[rb, 0]))
        if msl < 0.012 and dC < 0.036 and dL < 0.048:
            parent_b[find_b(ra)] = find_b(rb)
    fams_b = {}
    for r in range(n_rb):
        if areas_b[r] > 0 and not line_b[r]:
            fams_b.setdefault(find_b(r), []).append(r)
    sigma_blush = 24.0 * l_sc
    pad_b = int(round(sigma_blush * 2.5))
    for members in fams_b.values():
        if not (2 <= len(members) <= 4):
            continue
        y0 = max(0, min(boxes_b[r][0].start for r in members if boxes_b[r] is not None) - pad_b)
        y1 = min(h, max(boxes_b[r][0].stop for r in members if boxes_b[r] is not None) + pad_b)
        x0 = max(0, min(boxes_b[r][1].start for r in members if boxes_b[r] is not None) - pad_b)
        x1 = min(w, max(boxes_b[r][1].stop for r in members if boxes_b[r] is not None) + pad_b)
        sub_r = regs_b[y0:y1, x0:x1]
        fam_mask = np.isin(sub_r, members)
        if not fam_mask.any():
            continue
        den_f = np.maximum(cv2.GaussianBlur(fam_mask.astype(np.float32), (0, 0), sigma_blush), 1e-6)
        target_a = {r: int((sub_r == r).sum()) for r in members}
        cur_lbl = sub_r.copy()
        weights = {r: 1.0 for r in members}
        for _ in range(4):
            probs = {
                r: cv2.GaussianBlur((cur_lbl == r).astype(np.float32), (0, 0), sigma_blush) / den_f
                for r in members
            }
            for _ in range(8):
                best_prob = np.full(sub_r.shape, -1.0, dtype=np.float32)
                nxt = cur_lbl.copy()
                for r in members:
                    sc_p = probs[r] * np.float32(weights[r])
                    upd = fam_mask & (sc_p > best_prob)
                    nxt[upd] = r
                    best_prob[upd] = sc_p[upd]
                for r in members:
                    cur_a = max(1, int((nxt[fam_mask] == r).sum()))
                    weights[r] = float(np.clip(weights[r] * ((target_a[r] / cur_a) ** 0.5), 0.02, 50.0))
                w_ref = max(1e-6, max(weights.values()))
                for r in members:
                    weights[r] = float(np.clip(weights[r] / w_ref, 0.02, 1.0))
            cur_lbl = nxt
        res_work[y0:y1, x0:x1][fam_mask] = rgb_b[cur_lbl[fam_mask]]

    return res_work
