"""Bridge short breaks in dark ink strokes via path prediction and smooth stroke width."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .color import srgb_to_oklab


def _endpoint_tangent(sk, y, x, steps=6):
    """Estimate outgoing unit tangent at a skeleton endpoint (points away from the stroke)."""
    h, w = sk.shape
    cur = (y, x)
    prev = None
    path = [cur]
    for _ in range(steps):
        cy, cx = cur
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] and (ny, nx) != prev:
                    nxt = (ny, nx)
                    break
            if nxt is not None:
                break
        if nxt is None:
            break
        prev, cur = cur, nxt
        path.append(cur)
    if len(path) < 2:
        return None
    v = np.array([path[0][1] - path[-1][1], path[0][0] - path[-1][0]], np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-4 else None


def stitch_and_smooth_ink(base, source, line_evidence=None, max_gap=8.0):
    """Repair short gaps along thin ink lines (dark and coloured) and regularise stroke width."""
    lab_b = srgb_to_oklab(base)
    dark = lab_b[..., 0] < .28
    edt = distance_transform_edt(dark) if dark.any() else np.full(base.shape[:2], 99.0, np.float32)
    # Operate on thin dark ink strokes (radius <= 3.5 px), plus coloured thin outlines from line_evidence.
    thin_seed = dark & (edt <= 3.5)
    if line_evidence is not None:
        thin_seed |= dark & (line_evidence > .30) & (edt <= 4.5)
        col_dark = (~dark) & (lab_b[..., 0] < .54) & (line_evidence > .35)
        if col_dark.any():
            col_edt = distance_transform_edt(col_dark)
            thin_seed |= col_dark & (col_edt <= 3.0)
    if not thin_seed.any():
        return base
    ink_all = dark | thin_seed

    # Morphological closing (3x3) on thin ink closes 1-px pinches without altering wide fills.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_thin = cv2.morphologyEx(thin_seed.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    pinch_fill = closed_thin & ~ink_all & (lab_b[..., 0] < .85)
    out = base.copy()
    if pinch_fill.any():
        _, (iy, ix) = distance_transform_edt(~thin_seed, return_indices=True)
        out[pinch_fill] = base[iy[pinch_fill], ix[pinch_fill]]
        ink_all = ink_all | pinch_fill

    # Path-prediction bridging between skeleton endpoints of thin ink.
    sk = skeletonize(thin_seed | pinch_fill)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - sk
    ey, ex = np.where(sk & (nb == 1))
    if len(ey) < 2:
        return out

    seed_edt = distance_transform_edt(thin_seed)
    pts, tans, rads, ep_rgb, ep_lab = [], [], [], [], []
    for y, x in zip(ey.tolist(), ex.tolist()):
        t = _endpoint_tangent(sk, y, x)
        if t is not None:
            pts.append((float(x), float(y)))
            tans.append(t)
            rads.append(float(np.clip(seed_edt[y, x], .8, 2.5)))
            ep_rgb.append(out[y, x])
            ep_lab.append(lab_b[y, x])
    if len(pts) < 2:
        return out

    pts = np.asarray(pts, np.float32)
    tans = np.asarray(tans, np.float32)
    rads = np.asarray(rads, np.float32)
    ep_rgb = np.asarray(ep_rgb, np.float32)
    ep_lab = np.asarray(ep_lab, np.float32)
    lab_s = srgb_to_oklab(source)[..., 0].astype(np.float32)
    local_bg = cv2.dilate(lab_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    tree = cKDTree(pts)
    pairs = []
    for i in range(len(pts)):
        for j in tree.query_ball_point(pts[i], max_gap):
            if j <= i:
                continue
            if float(np.linalg.norm(ep_lab[i] - ep_lab[j])) > .11:
                continue
            v = pts[j] - pts[i]
            d = float(np.linalg.norm(v))
            if d < 1.5:
                continue
            u = v / d
            a1 = float(tans[i] @ u)
            a2 = float((-tans[j]) @ u)
            a12 = float(tans[i] @ (-tans[j]))
            if a1 < .76 or a2 < .76 or a12 < .68:
                continue
            coloured = max(float(ep_lab[i, 0]), float(ep_lab[j, 0])) >= .28
            # Require source corroboration for longer gaps (earlier for coloured lines).
            if d > (2.5 if coloured else 4.5):
                ts = np.linspace(.2, .8, 5)
                sy = np.clip(np.round(pts[i, 1] + ts * v[1]).astype(int), 0, out.shape[0] - 1)
                sx = np.clip(np.round(pts[i, 0] + ts * v[0]).astype(int), 0, out.shape[1] - 1)
                if np.mean(local_bg[sy, sx] - lab_s[sy, sx]) < .025:
                    continue
            score = d + 4.0 * (2.0 - a1 - a2)
            pairs.append((score, i, j, d))

    pairs.sort()
    used = set()
    for _, i, j, d in pairs:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        p0, p3 = pts[i], pts[j]
        p1 = p0 + tans[i] * (d / 3.0)
        p2 = p3 + tans[j] * (d / 3.0)
        t = np.linspace(0.0, 1.0, max(6, int(np.ceil(d * 2.0))))[:, None]
        mt = 1.0 - t
        curve = (mt ** 3) * p0 + 3 * (mt ** 2) * t * p1 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3
        r_mid = .5 * (rads[i] + rads[j])
        rad = max(0, int(np.floor(r_mid - .35)))
        fill_col = (ep_rgb[i] if ep_lab[i, 0] <= ep_lab[j, 0] else ep_rgb[j]).astype(np.uint8)
        single = np.zeros(ink_all.shape, np.uint8)
        for cx, cy in np.round(curve).astype(int):
            if 0 <= cy < out.shape[0] and 0 <= cx < out.shape[1]:
                if rad == 0:
                    single[cy, cx] = 1
                else:
                    cv2.circle(single, (cx, cy), rad, 1, -1)
        fill_b = (single > 0) & ~ink_all
        if fill_b.any():
            out[fill_b] = fill_col
            ink_all |= fill_b
    return out


def clean_ink_borders(base, source, line_evidence=None, protect_mask=None):
    """Remove muddy anti-alias rim slivers and tiny cavities along dark ink borders."""
    from skimage.measure import label as cc_label
    from scipy.ndimage import find_objects

    lab_b = srgb_to_oklab(base)
    lab_s = srgb_to_oklab(source)
    ink = lab_b[..., 0] < .26
    if not ink.any():
        return base

    k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    rim = (cv2.dilate(ink.astype(np.uint8), k4, iterations=2) > 0) & ~ink
    if not rim.any():
        return base

    keys = base.astype(np.int32) @ np.array([65536, 256, 1])
    regions = cc_label(keys + 1, connectivity=1) - 1
    n_reg = regions.max() + 1
    areas = np.bincount(regions.ravel(), minlength=n_reg)
    rim_px = np.bincount(regions.ravel(), weights=rim.ravel(), minlength=n_reg)
    # Candidate: small regions (<= 96 px) sitting mostly on the 2-px rim of dark ink.
    cand = (areas <= 96) & (rim_px >= .45 * np.maximum(areas, 1))
    # Exclude dark ink itself and bright white highlights that are genuinely distinct.
    reg_cols = np.empty((n_reg, 3), np.uint8)
    reg_cols[regions.ravel()] = base.reshape(-1, 3)
    reg_lab = srgb_to_oklab(reg_cols)
    cand &= (reg_lab[:, 0] >= .26) & (reg_lab[:, 0] < .92)
    ids = np.flatnonzero(cand)
    if not len(ids):
        return base

    out = base.copy()
    h, w = regions.shape
    sliver_mask = cand[regions]
    clean_fill = ~ink & ~sliver_mask
    _, (fy, fx) = distance_transform_edt(~clean_fill, return_indices=True)
    _, (iy, ix) = distance_transform_edt(~ink, return_indices=True)

    boxes = find_objects(regions + 1)
    for idx in ids:
        box = boxes[idx]
        if box is None:
            continue
        m = regions[box] == idx
        if protect_mask is not None and float(np.mean(protect_mask[box][m])) > 0.35:
            continue
        src_l = float(np.median(lab_s[box][m, 0]))
        le_med = float(np.median(line_evidence[box][m])) if line_evidence is not None else 0.0
        area = int(areas[idx])
        chroma = float(np.hypot(reg_lab[idx, 1], reg_lab[idx, 2]))
        if src_l < .38 or (le_med > .22 and src_l < .54):
            out[box][m] = base[iy[box][m], ix[box][m]]
        elif le_med > .22 and src_l < .72:
            continue
        elif area <= 28 or (chroma < .045 and area <= 72):
            out[box][m] = base[fy[box][m], fx[box][m]]
    return out
