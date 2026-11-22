"""Structure-first output: the SVG mirrors how the picture was drawn, not how it looks when
flattened.

    <g id="line-art">     下: 每条线一份几何 (闭合线 = 外环 + 内环 = 有真实空腔; 极细线 = 单环)
    <g id="fills">        中: 空腔填色 (each face one closed even-odd path, holes for islands)
    <g id="line-shading"> 上: 同一份线条几何按弧长分段换色 (一根线多节颜色, 线本身不被切断)

Two flavours, because importers differ:
  structure="structured" : <defs> + <use href> + <clipPath>  (geometry stored once)
  structure="flat"       : plain <path> only, no references    (works in any importer)
"""
import os
import numpy as np
import cv2
from skimage import measure

from .color import srgb_to_oklab, oklab_to_srgb
from .contour import fit_closed, ink_field
from .draw import hysteresis, line_strokes, stroke_outline_d

__all__ = ["vectorize_layered"]


def _hex(lab_or_rgb, is_lab=True):
    a = np.asarray(lab_or_rgb, np.float64)
    rgb = np.clip(oklab_to_srgb(a[None])[0] if is_lab else a, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def _face_path(mask, color, ss=2, level=0.45, tol=0.8, min_area=6.0, sigma=0.6, arcs=True,
               offset=(0, 0)):
    """One closed even-odd path for a face; islands of other faces inside it become holes.

    `offset` lets the caller pass a mask cropped to the face's bounding box (much faster on
    full-page rasters than allocating a full-size mask per face); traces come back in page
    coordinates.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return ""
    y0, y1 = max(0, ys.min() - 3), min(mask.shape[0], ys.max() + 4)
    x0, x1 = max(0, xs.min() - 3), min(mask.shape[1], xs.max() + 4)
    m = mask[y0:y1, x0:x1].astype(np.float32)
    f = cv2.resize(m, None, fx=ss, fy=ss, interpolation=cv2.INTER_LINEAR)
    f = cv2.GaussianBlur(f, (0, 0), sigma * ss)
    f = np.pad(f, 2)
    parts = []
    for c in measure.find_contours(f, level):
        if len(c) < 8:
            continue
        xy = (c[:, ::-1] + 0.5) / ss + np.array([x0 - 2, y0 - 2]) + np.array(offset, float)
        area = 0.5 * abs(np.dot(xy[:-1, 0], xy[1:, 1]) - np.dot(xy[1:, 0], xy[:-1, 1]))
        if area < min_area:
            continue
        if arcs:
            body = fit_arcs_closed(xy, tol=tol)
            if not body:
                continue
            parts.append("M%.2f,%.2f" % tuple(xy[0]) + body + "Z")
        else:
            segs = fit_closed(xy, tol=tol)
            if not segs:
                continue
            parts.append("M%.2f,%.2f" % tuple(segs[0][0]) +
                         "".join("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*s[1], *s[2], *s[3]) for s in segs) + "Z")
    if not parts:
        return ""
    return '<path d="%s" fill="%s" fill-rule="evenodd"/>' % ("".join(parts), color)


def _ring_d(pts, w):
    """Closed stroke -> outer loop + inner loop in ONE path (even-odd): the inside is a cavity."""
    ext = np.vstack([pts[-2:], pts, pts[:2]])
    tang = np.gradient(ext, axis=0)[2:-2]
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    half = (w / 2)[:, None]

    def loop(off):
        p = pts + nrm * off
        q = np.vstack([p, p[:1]])
        return "M" + "L".join("%.2f,%.2f" % tuple(v) for v in q) + "Z"
    return loop(half) + loop(-half)


def _segments(pts, lab, W, H, de=None, min_len=None, max_seg=10):
    """Split a line's centreline where the artwork's colour really changes.

    de adapts to the line's own colour spread: a strand that fades gently gets few segments, a
    lash with several colours gets more."""
    if len(pts) < 3:
        return []
    idx = np.clip(pts.astype(int), [0, 0], [W - 1, H - 1])
    cols = lab[idx[:, 1], idx[:, 0]]
    spread = float(np.percentile(np.linalg.norm(cols - np.median(cols, 0), axis=1), 90))
    if spread < 0.02:                            # a line that really is one colour stays one colour:
        return []                                # no shading segment, the base colour is enough
    if de is None:
        de = float(np.clip(0.45 * spread, 0.03, 0.10))
    if min_len is None:
        min_len = 8.0
    k = 7
    pad = np.vstack([cols[:1]] * (k // 2) + [cols] + [cols[-1:]] * (k // 2))
    ker = np.ones(k) / k
    cols = np.stack([np.convolve(pad[:, i], ker, "valid") for i in range(3)], 1)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    cuts = [0]
    for i in range(1, len(pts)):
        if s[i] - s[cuts[-1]] >= min_len and np.linalg.norm(cols[i] - cols[cuts[-1]]) > de:
            cuts.append(i)
    cuts.append(len(pts) - 1)
    cuts = sorted(set(cuts))
    while len(cuts) - 1 > max_seg and len(cuts) > 2:
        best, bj = 1e9, 1
        for j in range(1, len(cuts) - 1):
            a, b, c = cols[cuts[j - 1]], cols[cuts[j]], cols[cuts[j + 1]]
            v = np.linalg.norm(a - b) + np.linalg.norm(b - c)
            if v < best:
                best, bj = v, j
        cuts.pop(bj)
    return [(a, b, cols[a:b + 1].mean(0)) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 1]


def _band_poly(pts, w, a, b, pad=1.6):
    p = pts[a:b + 1]
    tang = np.gradient(p, axis=0) if len(p) > 1 else np.array([[1.0, 0.0]])
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    hw = (w[a:b + 1] / 2 + pad)[:, None]
    poly = np.vstack([p + nrm * hw, (p - nrm * hw)[::-1]])
    return "M" + "L".join("%.2f,%.2f" % tuple(q) for q in poly) + "Z"


def _cover_poly(pts, w):
    ext = np.vstack([pts[:1], pts, pts[-1:]])
    tang = np.gradient(ext, axis=0)[1:-1]
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    hw = (w / 2)[:, None]
    return np.vstack([pts + nrm * hw, (pts - nrm * hw)[::-1]])


def vectorize_layered(img, weights=None, *, structure="structured", pre_down=1.0, work_scale=2.0,
                      bands=112, ms_sp=6, ms_sr=10, blur_sigma=12.0, thin_thr=0.015,
                      vote_sigma=1.5, min_area=6, seg_max=10, progress=None, weights_dir=None,
                      arcs=True):
    """Return (svg, work_raster). Layers: line-art (bottom) -> fills (cavities) -> shading (top)."""
    import os, re
    from PIL import Image
    from scipy import ndimage as ndi
    from .denoise import ai_clean
    from .pipeline import match_tone
    from .bands import posterize
    from .lines import line_map
    from .strokes2 import line_prob, blob_mask

    report = progress or (lambda *a: None)
    W0, H0 = img.size
    report(0.02, "AI 去颗粒")
    if weights and os.path.exists(weights):
        work = ai_clean(img, weights, pre_down=pre_down, out_scale=work_scale)
        rgb = np.asarray(work.convert("RGB"))
        rgb = match_tone(rgb, np.asarray(img.convert("RGB").resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS)))
    else:
        work = img.convert("RGB").resize((round(W0 * work_scale), round(H0 * work_scale)), Image.LANCZOS)
        rgb = np.asarray(work)
    H, W = rgb.shape[:2]
    report(0.28, "压平")
    flat = cv2.pyrMeanShiftFiltering(rgb, ms_sp, ms_sr) if ms_sp > 0 else rgb
    report(0.34, "明暗分层")
    post = posterize(flat, k=bands, vote_sigma=vote_sigma, blur_sigma=blur_sigma, thin_thr=thin_thr,
                     progress=lambda *a: None, min_px=0)[0]
    # without the usual clean-up the posterised image is speckled and every speck becomes a face
    # (a 480 px crop produced 27k paths); the mode filter + sliver absorption merge those into
    # their neighbours, so faces stay large and the fill layer stays a few thousand shapes
    from .contour import smooth_labels, drop_slivers
    post = smooth_labels(post, sigma=1.0)
    post = drop_slivers(post, keep_dark=0.0)   # no merge_small: it also eats the thin bright
                                              # strands, and those are the ones the user cares about
    report(0.50, "识别线条")
    wd = weights_dir or (os.path.dirname(weights) if weights else "weights")
    dl = min(max(H, W), 1536)
    M = line_map(flat, "manga", detect_long=dl, weights_dir=wd)
    A = line_map(flat, "anime", detect_long=dl, weights_dir=wd)
    P = line_prob(M, A)
    cov = ink_field(rgb, P > 0.5)
    blob = blob_mask(P, cov)
    Pd = hysteresis(np.where(blob, 0, P), weak=0.30)
    report(0.66, "重建线条几何")
    lines = []
    for it in line_strokes(Pd, cov):
        w = np.maximum(it["w"], 0.8)
        pts = it["pts"]
        if len(pts) < 3:
            continue
        closed = (not it["free"][0] and not it["free"][1] and
                  np.hypot(*(pts[0] - pts[-1])) < max(6.0, 4.0 * float(np.median(w))))
        if closed and len(pts) > 12:
            lines.append(dict(d=_ring_d(pts, w), pts=pts, w=w, ring=True))
        else:
            d = stroke_outline_d(pts, w, it["free"], tol=0.5)
            if d:
                lines.append(dict(d=d, pts=pts, w=w, ring=False))
    ink = np.zeros((H, W), np.uint8)
    for ln in lines:
        poly = np.round(_cover_poly(ln["pts"], ln["w"]) * 16).astype(np.int32)
        cv2.fillPoly(ink, [poly], 1, lineType=cv2.LINE_8, shift=4)
    ink = ink > 0
    report(0.76, "空腔填色")
    key = (post[..., 0].astype(np.int64) << 16) | (post[..., 1].astype(np.int64) << 8) | post[..., 2]
    reg = measure.label(key + 1, connectivity=1, background=0)
    fills = []
    # big faces first, small ones last: every face is grown 1 px so that neighbours overlap, and
    # in the wrong order a grown neighbour swallows a 3 px highlight (light patches over the hair)
    area = np.bincount(reg.ravel())
    objs = ndi.find_objects(reg)
    order = [r for r in np.argsort(-area) if r > 0]
    for r in order:
        sl = objs[r - 1]
        if sl is None:
            continue
        m = np.zeros((H, W), bool)
        m[sl] = reg[sl] == r
        if not m.any():
            continue
        # sample the colour BEFORE growing the mask: after the 1 px dilation the first pixel of
        # the mask can belong to the neighbouring face, which painted whole regions in the wrong
        # colour (big light patches over the hair)
        yy, xx = np.argwhere(m)[0]
        col = post[yy, xx]
        # No drop of "small" faces and no subtraction of the ink: every pixel must be covered by
        # some face, otherwise the page shows through as white specks wherever a region is only a
        # few pixels across (dense hair). The line network is painted on top afterwards.
        # grow by a pixel so neighbouring faces overlap: two exact half-pixel edges leave a
        # 1 px white vein between them (that is what made the first prototype look "veined")
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        p = _face_path(m, "#%02x%02x%02x" % tuple(int(v) for v in col), tol=0.8, min_area=0.5, arcs=arcs)
        if p:
            fills.append(p)
    report(0.90, "装配")
    lab = srgb_to_oklab(rgb).astype(np.float32)
    defs, art, shade = [], [], []
    nseg = 0
    for i, ln in enumerate(lines):
        pts, w = ln["pts"], ln["w"]
        idx = np.clip(pts.astype(int), [0, 0], [W - 1, H - 1])
        base = _hex(np.median(lab[idx[:, 1], idx[:, 0]], 0))
        rule = ' fill-rule="evenodd"' if ln["ring"] else ""
        if structure == "flat":
            art.append('<path d="%s" fill="%s"%s/>' % (ln["d"], base, rule))
        else:
            defs.append('<path id="ln%d" d="%s"%s/>' % (i, ln["d"], rule))
            art.append('<use href="#ln%d" fill="%s"/>' % (i, base))
        thin = float(np.median(w)) < 3.5      # 只有细线（发丝、睫毛）才分段上色
        for j, (a, b, seg) in enumerate(_segments(pts, lab, W, H, max_seg=seg_max) if thin else []):
            if b - a < 2:
                continue
            nseg += 1
            if structure == "flat":
                sub = stroke_outline_d(pts[a:b + 1], w[a:b + 1], (False, False), tol=0.6)
                if sub:
                    shade.append('<path d="%s" fill="%s"/>' % (sub, _hex(seg)))
            else:
                cid = "lc%d_%d" % (i, j)
                defs.append('<clipPath id="%s"><path d="%s"/></clipPath>' % (cid, _band_poly(pts, w, a, b)))
                shade.append('<use href="#ln%d" fill="%s" clip-path="url(#%s)"/>' % (i, _hex(seg), cid))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (W, H)]
    if defs:
        parts.append("<defs>" + "".join(defs) + "</defs>")
    # paint order: faces first, then the line network, then the colour segments. Filling first is
    # what makes the seams disappear - each face is grown a pixel and the ink is painted over it.
    parts.append('<g id="fills" fill-rule="evenodd">' + "".join(fills) + "</g>")
    parts.append('<g id="line-art">' + "".join(art) + "</g>")
    if shade:
        parts.append('<g id="line-shading">' + "".join(shade) + "</g>")
    parts.append("</svg>")
    svg = "\n".join(parts)
    svg = re.sub(r"<svg\b([^>]*)>",
                 lambda m: "<svg" + re.sub(r'\s(width|height)="[^"]*"', "", m.group(1)) +
                 f' width="{W0}" height="{H0}" viewBox="0 0 {W} {H}">', svg, count=1)
    report(1.0, "完成")
    return svg, work


def _circle(p):
    """Least-squares circle through p (Kasa). Returns (cx, cy, r) or None."""
    if len(p) < 5:
        return None
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    b = (p ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 1e-6:
        return None
    return cx, cy, float(np.sqrt(r2))


def fit_arcs_closed(xy, tol=0.8, corner_deg=50.0, arc_tol=0.45):
    """Like contour.fit_closed but emits CIRCULAR ARCS where the boundary really is a circle
    (eyes, pupils, rings, round highlights): the reference SVGs use arcs and quadratics heavily,
    and one arc on a near-circular run is shorter and smoother than three cubics."""
    from .contour import _resample, _corners, fit_cubic, _unit
    p = _resample(xy[:-1] if np.allclose(xy[0], xy[-1]) else xy, 1.0)
    if len(p) < 6:
        return ""
    pc = np.r_[p[-2:], p, p[:2]]
    sm = (pc[:-4] + 2 * pc[1:-3] + 3 * pc[2:-2] + 2 * pc[3:-1] + pc[4:]) / 9
    cs = _corners(p, corner_deg)
    if cs:
        sm[cs] = p[cs]
    n = len(sm)
    if not cs:
        cs = sorted({0, int(np.argmax(np.hypot(*(sm - sm[0]).T)))})
    d = []
    for j, c0 in enumerate(cs):
        c1 = cs[(j + 1) % len(cs)]
        idx = np.arange(c0, c1 + (n if c1 <= c0 else 0) + 1) % n
        run = sm[idx]
        if len(run) < 3:
            continue
        circ = _circle(run)
        done = False
        if circ is not None:
            cx, cy, r = circ
            dev = np.abs(np.hypot(run[:, 0] - cx, run[:, 1] - cy) - r)
            ang = np.unwrap(np.arctan2(run[:, 1] - cy, run[:, 0] - cx))
            sweep = float(ang[-1] - ang[0])
            if dev.max() < arc_tol and abs(sweep) < np.radians(330) and r < 4000:
                d.append("A%.2f,%.2f 0 %d %d %.2f,%.2f" %
                         (r, r, 1 if abs(sweep) > np.pi else 0, 1 if sweep > 0 else 0,
                          run[-1][0], run[-1][1]))
                done = True
        if not done:
            t1 = _unit(run[min(2, len(run) - 1)] - run[0])
            t2 = _unit(run[max(-3, -len(run))] - run[-1])
            for s in fit_cubic(run, t1, t2, tol):
                a = _cubic_to_arc(s, arc_tol)
                if a:
                    d.append(a)
                else:
                    d.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*s[1], *s[2], *s[3]))
    return "".join(d)


def _cubic_to_arc(ctrl, tol=0.45, n=9):
    """One cubic whose samples all sit on a circle -> an 'A' command (shorter, exactly circular)."""
    t = np.linspace(0, 1, n)[:, None]
    mt = 1 - t
    pts = (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] + 3 * mt * t ** 2 * ctrl[2] + t ** 3 * ctrl[3]
    circ = _circle(pts)
    if circ is None:
        return None
    cx, cy, r = circ
    dev = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).max()
    if dev > tol or r > 4000:
        return None
    a0 = np.arctan2(ctrl[0][1] - cy, ctrl[0][0] - cx)
    a1 = np.arctan2(ctrl[3][1] - cy, ctrl[3][0] - cx)
    # direction from the middle control points (they sit on the correct side of the chord)
    mid = np.arctan2(ctrl[1][1] - cy, ctrl[1][0] - cx)
    ccw = np.sin(mid - a0) > 0
    sweep = (a1 - a0) % (2 * np.pi) if ccw else -((a0 - a1) % (2 * np.pi))
    if abs(sweep) >= np.radians(195):
        return None
    am = a0 + 0.5 * sweep
    arc_mid_x, arc_mid_y = cx + r * np.cos(am), cy + r * np.sin(am)
    cub_mid = pts[len(pts) // 2]
    if np.hypot(arc_mid_x - cub_mid[0], arc_mid_y - cub_mid[1]) > max(0.8, tol * 1.5):
        return None
    return "A%.2f,%.2f 0 %d %d %.2f,%.2f" % (r, r, 1 if abs(sweep) > np.pi else 0,
                                             1 if sweep > 0 else 0, ctrl[3][0], ctrl[3][1])
