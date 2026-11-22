"""Eye fine channel (prototype). An eye is drawn by the artist as clean layers (iris bands,
pupil, highlights, lid shadow); the global pipeline blurs them (AI clean, smoothing, a palette
shared with the whole image). Inside each eye: original pixels, a palette of the eye's own
colours, only a light mode filter, nothing merged."""
import numpy as np
import cv2

from .bands import fit_palette, assign, soft_vote, edge_strength
from .color import srgb_to_oklab


def eye_patch(post, src, boxes, k=12, feather=0.15):
    """post: label image (work px), src: original at work size. boxes: (x, y, w, h) work px.
    Returns (new label image, mask of replaced pixels)."""
    from .contour import smooth_labels
    out = post.copy()
    H, W = post.shape[:2]
    mask = np.zeros((H, W), bool)
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y)); x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        sub = np.ascontiguousarray(src[y0:y1, x0:x1])
        cen = fit_palette(sub, k)
        lab = assign(sub, cen).reshape(sub.shape[:2])
        cols = np.zeros((lab.max() + 1, 3), np.uint8)
        flat = lab.ravel(); px = sub.reshape(-1, 3)
        for c in np.unique(flat):
            cols[c] = np.median(px[flat == c], 0)
        loc = smooth_labels(cols[lab], sigma=0.6)
        # elliptical mask (an eye sits in the middle of its box), shrunk a little so the patch
        # meets the surrounding skin where both are skin anyway
        m = np.zeros(sub.shape[:2], np.uint8)
        cv2.ellipse(m, ((sub.shape[1] / 2, sub.shape[0] / 2), (sub.shape[1] * (1 - feather), sub.shape[0] * (1 - feather)), 0), 1, -1)
        m = m > 0
        out[y0:y1, x0:x1][m] = loc[m]
        mask[y0:y1, x0:x1] |= m
    return out, mask


def rescue_delicate_rois(quant, src, delicate_map, k=12, thr=0.16, line_evidence=None):
    """Quantize CNN-detected compact delicate ROIs (eyes, small irises, mouth details) with an
    independent local palette from `src` so small subtle colors are not swallowed globally.
    Returns (updated_quant, roi_protect_mask)."""
    if delicate_map is None:
        return quant, np.zeros(quant.shape[:2], bool)
    H, W = quant.shape[:2]
    pad = max(4, round(min(H, W) * 0.010))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    raw_mask = (delicate_map > thr).astype(np.uint8)
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k_dil)
    dilated = cv2.dilate(closed, k_dil) > 0
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(dilated.astype(np.uint8), connectivity=8)

    out = quant.copy()
    protect_roi = np.zeros((H, W), bool)
    min_area = max(100, int(H * W * 0.0004))
    max_area = int(H * W * 0.025)
    prot_max_area = max_area

    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = [int(stats[idx, f]) for f in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
        )]
        if bw > 0.45 * W or bh > 0.45 * H:
            continue
        if float(delicate_map[y:y + bh, x:x + bw][lbl[y:y + bh, x:x + bw] == idx].max()) < 0.42:
            continue

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + bw), min(H, y + bh)
        sub_src = np.ascontiguousarray(src[y0:y1, x0:x1])
        sub_q = np.ascontiguousarray(quant[y0:y1, x0:x1])
        m_roi = (lbl[y0:y1, x0:x1] == idx)
        if area <= prot_max_area:
            # Fill the convex hull of compact facial/eye masks so open lash arcs protect
            # the entire interior iris, sclera, pupil, and highlights from downstream merging.
            pts = np.column_stack(np.where(m_roi))[:, ::-1].astype(np.int32)
            hull_mask = np.zeros_like(m_roi, dtype=np.uint8)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(hull_mask, hull, 1)
                m_prot = m_roi | (hull_mask > 0)
            else:
                m_prot = m_roi
            protect_roi[y0:y1, x0:x1] |= m_prot

        cen = fit_palette(sub_src, k, l_weight=1.1, c_weight=3.2, mask=m_roi)
        if not len(cen):
            continue
        raw_l = assign(sub_src, cen, l_weight=1.1, c_weight=3.2)
        loc_edges = (edge_strength(sub_src) > 0.25) | (delicate_map[y0:y1, x0:x1] > 0.28)
        lab = np.where(loc_edges, raw_l, soft_vote(raw_l, 0.55))

        used = np.unique(lab[m_roi])
        cols = np.zeros((lab.max() + 1, 3), np.uint8)
        for c in used:
            sel = (lab == c) & m_roi
            if sel.any():
                cols[c] = np.median(sub_src[sel], axis=0).astype(np.uint8)
        loc_rgb = cols[lab]

        # Override pixels where global quantization lost subtle chroma/hue or detail,
        # while keeping already-crisp dark outline strokes untouched so thin mouths don't fragment.
        lab_src = srgb_to_oklab(sub_src)
        lab_q = srgb_to_oklab(sub_q)
        lab_loc = srgb_to_oklab(loc_rgb)
        de_q = np.linalg.norm(lab_src - lab_q, axis=2)
        de_loc = np.linalg.norm(lab_src - lab_loc, axis=2)
        dark_stroke = (lab_q[..., 0] < 0.62) & (lab_src[..., 0] < 0.65)
        if line_evidence is not None:
            dark_stroke = dark_stroke & (line_evidence[y0:y1, x0:x1] > 0.18)
        improved = m_roi & (~dark_stroke) & (de_q > 0.022) & (de_loc + 0.005 < de_q)
        if improved.any():
            out[y0:y1, x0:x1][improved] = loc_rgb[improved]
            protect_roi[y0:y1, x0:x1] |= improved

    return out, protect_roi
