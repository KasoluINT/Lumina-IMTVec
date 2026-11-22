"""Direct Vector-Field Scatter-Graph & Schneider Cubic Bezier SVG Emitter.

Converts the 9-channel output of LineVectorNet (at 2x SR resolution) into clean
vector SVG curves without sequential ant-walking overshoot or blurry binary contours:
  1. Projects sub-pixel `(x + dx, y + dy)` scatter votes onto a 2x-denser density map.
  2. Extracts an 8-connected topological skeleton graph, prunes spurs, splits sharp corners,
     and pairs collinear continuations straight across X-crossings.
  3. Fits each unified curve via `fit_structural` (least-squares 4-arc circles, straight
     `L` segments, and Schneider optimal least-squares Cubic Beziers with Newton reparam).
"""
from __future__ import annotations

from collections import defaultdict
import re
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .color import srgb_to_oklab
from .structural_curves import fit_structural


def _sample_bilinear(arr: np.ndarray, y: float, x: float) -> float:
    h, w = arr.shape[:2]
    yc = min(max(y, 0.0), h - 1.001)
    xc = min(max(x, 0.0), w - 1.001)
    y0 = int(yc)
    x0 = int(xc)
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)
    fy = yc - y0
    fx = xc - x0
    return float(
        (1.0 - fy) * (1.0 - fx) * arr[y0, x0]
        + (1.0 - fy) * fx * arr[y0, x1]
        + fy * (1.0 - fx) * arr[y1, x0]
        + fy * fx * arr[y1, x1]
    )


def _rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification on [N, 2] float32 points."""
    if len(pts) <= 2:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 <= 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        v = p1 - p0
        vlen = float(np.hypot(v[0], v[1]))
        seg = pts[i0 + 1:i1]
        if vlen < 1e-6:
            dists = np.linalg.norm(seg - p0, axis=1)
        else:
            dists = np.abs(v[0] * (seg[:, 1] - p0[1]) - v[1] * (seg[:, 0] - p0[0])) / vlen
        k_rel = int(np.argmax(dists))
        if dists[k_rel] > eps:
            k_abs = i0 + 1 + k_rel
            keep[k_abs] = True
            stack.append((i0, k_abs))
            stack.append((k_abs, i1))
    return pts[keep]


def _rdp_with_radii(pts: np.ndarray, radii: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) <= 2:
        return pts, radii
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 <= 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        v = p1 - p0
        vlen = float(np.hypot(v[0], v[1]))
        seg = pts[i0 + 1:i1]
        if vlen < 1e-6:
            dists = np.linalg.norm(seg - p0, axis=1)
        else:
            dists = np.abs(v[0] * (seg[:, 1] - p0[1]) - v[1] * (seg[:, 0] - p0[0])) / vlen
        k_rel = int(np.argmax(dists))
        if dists[k_rel] > eps:
            k_abs = i0 + 1 + k_rel
            keep[k_abs] = True
            stack.append((i0, k_abs))
            stack.append((k_abs, i1))
    return pts[keep], radii[keep]


def _fit_subspan_cubic_or_line(sub: np.ndarray, closed_loop: bool = False) -> list[str]:
    """Emit straight `L` for linear geometric spans, or clean Catmull-Rom `C` for smooth curves."""
    n = len(sub)
    if n < 2:
        return []
    if n == 2:
        return [f"L{sub[1,0]:.2f},{sub[1,1]:.2f}"]

    # Check maximum perpendicular deviation from straight chord p0 -> p_end
    if not closed_loop:
        chord = sub[-1] - sub[0]
        clen = float(np.hypot(chord[0], chord[1]))
        if clen > 1e-5:
            dev = np.abs(chord[0] * (sub[:, 1] - sub[0, 1]) - chord[1] * (sub[:, 0] - sub[0, 0])) / clen
            if float(dev.max()) <= 0.14:
                return [f"L{sub[-1,0]:.2f},{sub[-1,1]:.2f}"]

    if closed_loop:
        p_ext = np.vstack([sub[-1:], sub, sub[:2]])
        m_end = n
    else:
        p_s = 2.0 * sub[0] - sub[1]
        p_e = 2.0 * sub[-1] - sub[-2]
        p_ext = np.vstack([p_s[None], sub, p_e[None]])
        m_end = n - 1

    cmds = []
    for i in range(m_end):
        pm, p0, p1, pn = p_ext[i], p_ext[i + 1], p_ext[i + 2], p_ext[i + 3]
        c1 = p0 + (p1 - pm) / 6.0
        c2 = p1 - (pn - p0) / 6.0
        cmds.append(f"C{c1[0]:.2f},{c1[1]:.2f},{c2[0]:.2f},{c2[1]:.2f},{p1[0]:.2f},{p1[1]:.2f}")
    return cmds


def _cubic_d_open(p_xy: np.ndarray, closed: bool = False) -> str:
    """Piecewise Corner-Preserving Cubic Bezier / Line SVG path emitter.
    Splits smoothly at sharp corners (>= 38 deg) so diamond petals and geometric contours
    never bow or overshoot, while flowing hair curves remain C1 smooth."""
    n = len(p_xy)
    if n < 2:
        return ""
    if n == 2:
        return f"M{p_xy[0,0]:.2f},{p_xy[0,1]:.2f}L{p_xy[1,0]:.2f},{p_xy[1,1]:.2f}"

    # Detect sharp geometric corner vertices (turn angle >= 38 deg -> dot < 0.78)
    corners = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        p_prev = p_xy[(i - 1) % n]
        p_curr = p_xy[i]
        p_next = p_xy[(i + 1) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        l1, l2 = float(np.hypot(v1[0], v1[1])), float(np.hypot(v2[0], v2[1]))
        if l1 > 1e-4 and l2 > 1e-4:
            if float((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)) < 0.78:
                corners.append(i)

    if not corners:
        cmds = _fit_subspan_cubic_or_line(p_xy, closed_loop=closed)
        return f"M{p_xy[0,0]:.2f},{p_xy[0,1]:.2f}" + "".join(cmds) + ("Z" if closed else "")

    if closed:
        # Rotate array to start cleanly on first corner, split into corner-bounded spans
        c0 = corners[0]
        p_rot = np.roll(p_xy, -c0, axis=0)
        c_rot = [0] + [(c - c0) % n for c in corners[1:]] + [n]
        out = [f"M{p_rot[0,0]:.2f},{p_rot[0,1]:.2f}"]
        for a, b in zip(c_rot[:-1], c_rot[1:]):
            seg = p_rot[a:b + 1] if b < n else np.vstack([p_rot[a:], p_rot[:1]])
            out.extend(_fit_subspan_cubic_or_line(seg, closed_loop=False))
        out.append("Z")
        return "".join(out)

    splits = [0] + corners + [n - 1]
    out = [f"M{p_xy[0,0]:.2f},{p_xy[0,1]:.2f}"]
    for a, b in zip(splits[:-1], splits[1:]):
        if b > a:
            out.extend(_fit_subspan_cubic_or_line(p_xy[a:b + 1], closed_loop=False))
    return "".join(out)


def _line_bez(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Construct a collinear `[4, 2]` Bezier segment from `p0` to `p1`."""
    p0f = np.asarray(p0, dtype=np.float32)
    p1f = np.asarray(p1, dtype=np.float32)
    return np.stack([p0f, (2.0 * p0f + p1f) / 3.0, (p0f + 2.0 * p1f) / 3.0, p1f], axis=0)


def _find_corner_indices(pts_1x: np.ndarray, is_closed: bool, dot_thr: float = 0.78) -> list[int]:
    """Detect sharp vertex indices (`turn >= ~38 deg`) before any coordinate smoothing."""
    n = len(pts_1x)
    w = min(3, max(1, (n - 1) // 3))
    if is_closed and n >= 6:
        corners: list[int] = []
        for i in range(n):
            v1 = pts_1x[i] - pts_1x[(i - w) % n]
            v2 = pts_1x[(i + w) % n] - pts_1x[i]
            n1, n2 = float(np.hypot(v1[0], v1[1])), float(np.hypot(v2[0], v2[1]))
            if n1 > 1e-3 and n2 > 1e-3 and float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)) < dot_thr:
                if not corners or (i - corners[-1]) >= 3:
                    corners.append(i)
        return corners
    corners = [0]
    for i in range(w, n - w):
        v1 = pts_1x[i] - pts_1x[i - w]
        v2 = pts_1x[i + w] - pts_1x[i]
        n1, n2 = float(np.hypot(v1[0], v1[1])), float(np.hypot(v2[0], v2[1]))
        if n1 > 1e-3 and n2 > 1e-3 and float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)) < dot_thr:
            if i - corners[-1] >= 3:
                corners.append(i)
    corners.append(n - 1)
    return corners


def _fit_line_or_structural(pts_1x: np.ndarray, is_closed: bool) -> list[np.ndarray]:
    """Split open/closed polyline at sharp corners, emit exact straight `L` spans when
    collinear within 0.45 px, and fit curved spans via Schneider Beziers / 4-arc circles."""
    n = len(pts_1x)
    if n < 2:
        return []
    if n == 2:
        return [_line_bez(pts_1x[0], pts_1x[1])]

    corners = _find_corner_indices(pts_1x, is_closed=is_closed, dot_thr=0.78)

    def _span_fit(sub: np.ndarray) -> list[np.ndarray]:
        m = len(sub)
        if m < 2:
            return []
        if m == 2:
            return [_line_bez(sub[0], sub[1])]
        p0, p1 = sub[0], sub[-1]
        v = p1 - p0
        vn = float(np.hypot(v[0], v[1]))
        if vn > 1e-4:
            d_perp = np.abs((sub[:, 0] - p0[0]) * v[1] - (sub[:, 1] - p0[1]) * v[0]) / vn
            max_e, rms_e = float(np.max(d_perp)), float(np.sqrt(np.mean(d_perp * d_perp)))
            if (max_e <= 0.45 and rms_e <= 0.22) or (vn >= 10.0 and max_e <= 0.55 and rms_e <= 0.25):
                return [_line_bez(p0, p1)]
        sub_fit = sub.copy()
        if m >= 5:
            sm = sub_fit.copy()
            sm[1:-1] = 0.6 * sub_fit[1:-1] + 0.2 * (sub_fit[:-2] + sub_fit[2:])
            sub_fit = sm
        return fit_structural(sub_fit, scale=0.75, smooth_lines=False)

    if is_closed and len(corners) >= 2:
        segs: list[np.ndarray] = []
        nc = len(corners)
        for ci in range(nc):
            a, b = corners[ci], corners[(ci + 1) % nc]
            sub = pts_1x[a : b + 1] if b > a else np.vstack([pts_1x[a:], pts_1x[: b + 1]])
            segs.extend(_span_fit(sub))
        return segs
    if is_closed:
        pts_fit = np.vstack([pts_1x, pts_1x[:1]]) if not np.allclose(pts_1x[0], pts_1x[-1]) else pts_1x
        return fit_structural(pts_fit, scale=0.70, smooth_lines=True)

    segs = []
    for ci in range(len(corners) - 1):
        segs.extend(_span_fit(pts_1x[corners[ci] : corners[ci + 1] + 1]))
    return segs


def _clamp_monotone_offset(side: np.ndarray, spine: np.ndarray) -> np.ndarray:
    """Prevent inner-elbow offset foldbacks that create even-odd bowtie pinholes in fills."""
    out = side.copy()
    for i in range(1, len(out)):
        ds = spine[i] - spine[i - 1]
        if float(np.hypot(ds[0], ds[1])) <= 1e-5:
            continue
        if float(np.dot(out[i] - out[i - 1], ds)) < 0.0:
            out[i] = out[i - 1]
    return out


def _variable_width_envelope_d(p_xy: np.ndarray, r_arr: np.ndarray, closed: bool = False) -> str:
    """Build smooth corner-safe variable-width Bezier envelopes around `p_xy` with half-width `r_arr`."""
    n = len(p_xy)
    if n < 3:
        return ""

    r_sm = r_arr.astype(np.float32).copy()
    if n >= 5:
        for _ in range(2):
            tmp = r_sm.copy()
            tmp[1:-1] = 0.5 * r_sm[1:-1] + 0.25 * (r_sm[:-2] + r_sm[2:])
            if closed:
                tmp[0] = 0.5 * r_sm[0] + 0.25 * (r_sm[-1] + r_sm[1])
                tmp[-1] = 0.5 * r_sm[-1] + 0.25 * (r_sm[-2] + r_sm[0])
            r_sm = tmp

    def _emit_span(sub_p: np.ndarray, sub_r: np.ndarray, is_loop: bool) -> str:
        m = len(sub_p)
        if m < 2:
            return ""
        if is_loop:
            tang = np.roll(sub_p, -1, axis=0) - np.roll(sub_p, 1, axis=0)
        else:
            tang = np.zeros_like(sub_p)
            if m >= 3:
                tang[1:-1] = sub_p[2:] - sub_p[:-2]
            tang[0] = sub_p[1] - sub_p[0]
            tang[-1] = sub_p[-1] - sub_p[-2]
        tlen = np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-5)
        norm = np.stack([-tang[:, 1] / tlen, tang[:, 0] / tlen], axis=1)
        left = _clamp_monotone_offset(sub_p + norm * sub_r[:, None], sub_p)
        right = _clamp_monotone_offset(sub_p - norm * sub_r[:, None], sub_p)

        l_segs = _fit_line_or_structural(left, is_closed=is_loop)
        r_segs = _fit_line_or_structural(right[::-1], is_closed=is_loop)
        if is_loop:
            return _beziers_to_svg_d(l_segs, closed=True) + _beziers_to_svg_d(r_segs, closed=True)
        d1 = _beziers_to_svg_d(l_segs, closed=False)
        d2 = _beziers_to_svg_d(r_segs, closed=False)
        if not d1 or not d2:
            return ""
        d2_body = re.sub(r"^M[^MLCA]+", "", d2)
        r_end, r_beg = max(0.25, float(sub_r[-1])), max(0.25, float(sub_r[0]))
        return f"{d1}A{r_end:.2f},{r_end:.2f} 0 0 0 {right[-1,0]:.2f},{right[-1,1]:.2f}{d2_body}A{r_beg:.2f},{r_beg:.2f} 0 0 0 {left[0,0]:.2f},{left[0,1]:.2f}Z"

    if closed:
        return _emit_span(p_xy, r_sm, is_loop=True)
    corners = _find_corner_indices(p_xy, is_closed=False, dot_thr=0.76)
    if len(corners) <= 2:
        return _emit_span(p_xy, r_sm, is_loop=False)
    parts = [
        _emit_span(p_xy[corners[i] : corners[i + 1] + 1], r_sm[corners[i] : corners[i + 1] + 1], is_loop=False)
        for i in range(len(corners) - 1)
    ]
    return "".join(p for p in parts if p)


def _beziers_to_svg_d(segs: list[np.ndarray], closed: bool = False) -> str:
    """Format a list of `[4, 2]` Cubic Bezier arrays into compact SVG `L`/`C` commands."""
    if not segs:
        return ""
    out = [f"M{segs[0][0,0]:.2f},{segs[0][0,1]:.2f}"]
    for s in segs:
        p0, c1, c2, p1 = s[0], s[1], s[2], s[3]
        v = p1 - p0
        vn = float(np.hypot(v[0], v[1]))
        if vn > 1e-5:
            d1 = abs(v[0] * (c1[1] - p0[1]) - v[1] * (c1[0] - p0[0])) / vn
            d2 = abs(v[0] * (c2[1] - p0[1]) - v[1] * (c2[0] - p0[0])) / vn
            t1 = ((c1[0] - p0[0]) * v[0] + (c1[1] - p0[1]) * v[1]) / (vn * vn)
            t2 = ((c2[0] - p0[0]) * v[0] + (c2[1] - p0[1]) * v[1]) / (vn * vn)
            if max(d1, d2) <= 0.12 and 0.0 <= min(t1, t2) and max(t1, t2) <= 1.0:
                out.append(f"L{p1[0]:.2f},{p1[1]:.2f}")
                continue
        out.append(f"C{c1[0]:.2f},{c1[1]:.2f},{c2[0]:.2f},{c2[1]:.2f},{p1[0]:.2f},{p1[1]:.2f}")
    if closed:
        out.append("Z")
    return "".join(out)


def trace_vector_fields_to_svg_paths(
    vec_fields: np.ndarray,
    sr_rgb: np.ndarray,
    sr_scale: float = 2.0,
    prob_thr: float = 0.29,
    seed_thr: float = 0.44,
    tapered: bool = False,
    force_color: str | None = None,
    return_centerlines: bool = False,
) -> tuple[list[str], np.ndarray] | tuple[list[str], np.ndarray, list[np.ndarray]]:
    """Convert `LineVectorNet` v4 [9, Hs, Ws] vector fields via sub-pixel scatter collapse,
    topological graph decomposition, and Schneider Cubic Bezier / analytic circle fitting.

    Returns:
      paths_svg      : list of SVG `<path .../>` strings in 1x image coordinate space
      wall_hi        : [Hs, Ws] boolean stroke/boundary wall mask for coloring constraints
      centerlines_sr : optional list of `[N, 2]` float32 SR polylines
    """
    hs, ws = vec_fields.shape[1:]
    prob = vec_fields[0]
    cbound = vec_fields[1]
    junc = vec_fields[3]
    dx_map = (vec_fields[4] - 0.5) * 3.0
    dy_map = (vec_fields[5] - 0.5) * 3.0
    rad_map = vec_fields[8] * 3.2

    wall_hi = (prob >= 0.40) | (cbound >= 0.46)
    sub = 2
    H2, W2 = hs * sub, ws * sub
    ay, ax = np.where(prob >= prob_thr)
    if len(ay) == 0:
        return ([], wall_hi, []) if return_centerlines else ([], wall_hi)

    # 1. Project active CNN pixels via (x + dx, y + dy) onto 2x sub-pixel density grid
    w_pt = prob[ay, ax]
    damp = np.where(junc[ay, ax] < 0.42, 1.0, 0.45)
    px = np.clip((ax + dx_map[ay, ax] * damp) * sub + 0.5, 0.0, W2 - 1.001)
    py = np.clip((ay + dy_map[ay, ax] * damp) * sub + 0.5, 0.0, H2 - 1.001)
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    fx = px - x0
    fy = py - y0
    x1 = np.minimum(x0 + 1, W2 - 1)
    y1 = np.minimum(y0 + 1, H2 - 1)

    dens = np.zeros((H2, W2), dtype=np.float32)
    np.add.at(dens, (y0, x0), w_pt * (1.0 - fx) * (1.0 - fy))
    np.add.at(dens, (y0, x1), w_pt * fx * (1.0 - fy))
    np.add.at(dens, (y1, x0), w_pt * (1.0 - fx) * fy)
    np.add.at(dens, (y1, x1), w_pt * fx * fy)

    # Sharp anti-bridging ridge extraction (prevents 5x5 blur / MORPH_CLOSE from welding parallel bars)
    dens_s = cv2.GaussianBlur(dens, (3, 3), 0.50)
    mean5 = cv2.boxFilter(dens_s, -1, (5, 5), normalize=True)
    mean7 = cv2.boxFilter(dens_s, -1, (7, 7), normalize=True)
    prob_2x = cv2.resize(prob, (W2, H2), interpolation=cv2.INTER_LINEAR)
    junc_2x = cv2.resize(junc, (W2, H2), interpolation=cv2.INTER_LINEAR)
    dense_zone = mean5 > 0.20
    ridge_iso = (~dense_zone) & ((dens_s >= 0.11) | ((dens_s >= 0.045) & ((dens_s - 0.36 * mean7) >= 0.009)))
    ridge_crd = dense_zone & (dens_s >= np.maximum(0.11, 0.94 * mean5))
    junc_br = (junc_2x >= 0.26) & (prob_2x >= 0.34) & (dens_s >= np.where(dense_zone, 0.86 * mean5, 0.035))
    ridge_band = (ridge_iso | ridge_crd | junc_br) & (prob_2x >= min(prob_thr, 0.24))

    n_rb, l_rb, st_rb, _ = cv2.connectedComponentsWithStats(ridge_band.astype(np.uint8), connectivity=8)
    for ri in range(1, n_rb):
        if int(st_rb[ri, cv2.CC_STAT_AREA]) <= 5:
            x0r, y0r, bwr, bhr = [int(st_rb[ri, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            sub_m = l_rb[y0r:y0r + bhr, x0r:x0r + bwr] == ri
            if float(prob_2x[y0r:y0r + bhr, x0r:x0r + bwr][sub_m].max()) < 0.48:
                ridge_band[y0r:y0r + bhr, x0r:x0r + bwr][sub_m] = False

    skel = skeletonize(ridge_band)

    ys_s, xs_s = np.where(skel)
    if len(ys_s) == 0:
        return ([], wall_hi, []) if return_centerlines else ([], wall_hi)

    # 2. Extract 8-connected topological skeleton graph (nodes = endpoints / junctions)
    skel_u8 = skel.astype(np.uint8)
    nbr_cnt = cv2.filter2D(skel_u8, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - skel_u8
    node_mask = skel & (nbr_cnt != 2)
    OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    node_set = set(zip(*np.where(node_mask)))
    skel_set = set(zip(ys_s.tolist(), xs_s.tolist()))

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    chains: list[list[tuple[int, int]]] = []

    def _ek(p1: tuple[int, int], p2: tuple[int, int]):
        return (p1, p2) if p1 <= p2 else (p2, p1)

    for ny, nx in sorted(node_set):
        for dy, dx in OFFS:
            vy, vx = ny + dy, nx + dx
            if (vy, vx) not in skel_set:
                continue
            ek0 = _ek((ny, nx), (vy, vx))
            if ek0 in visited_edges:
                continue
            path = [(ny, nx), (vy, vx)]
            visited_edges.add(ek0)
            prev, cur = (ny, nx), (vy, vx)
            while cur not in node_set:
                nxt = None
                for ody, odx in OFFS:
                    cand = (cur[0] + ody, cur[1] + odx)
                    if cand != prev and cand in skel_set and _ek(cur, cand) not in visited_edges:
                        nxt = cand
                        break
                if nxt is None:
                    break
                visited_edges.add(_ek(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            chains.append(path)

    # Extract isolated node-free closed loops
    for sy, sx in zip(ys_s.tolist(), xs_s.tolist()):
        if (sy, sx) in node_set:
            continue
        inc = [(sy + dy, sx + dx) for dy, dx in OFFS if (sy + dy, sx + dx) in skel_set and _ek((sy, sx), (sy + dy, sx + dx)) not in visited_edges]
        if not inc:
            continue
        path = [(sy, sx), inc[0]]
        visited_edges.add(_ek((sy, sx), inc[0]))
        prev, cur = (sy, sx), inc[0]
        while True:
            nxt = None
            for ody, odx in OFFS:
                cand = (cur[0] + ody, cur[1] + odx)
                if cand != prev and cand in skel_set and _ek(cur, cand) not in visited_edges:
                    nxt = cand
                    break
            if nxt is None:
                break
            visited_edges.add(_ek(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == (sy, sx):
                break
        chains.append(path)

    # 3. Prune noisy skeleton spurs & isolated dust stubs while keeping real star/petal tips
    deg = {p: int(nbr_cnt[p[0], p[1]]) for p in node_set}
    kept_chains = []
    for ch in chains:
        is_closed = len(ch) >= 6 and ch[0] == ch[-1]
        if not is_closed:
            d0, d1 = deg.get(ch[0], 2), deg.get(ch[-1], 2)
            ch_arr = np.asarray(ch, dtype=np.int32)
            m_prob = float(prob_2x[ch_arr[:, 0], ch_arr[:, 1]].mean())
            if ((d0 == 1 and d1 >= 3) or (d1 == 1 and d0 >= 3)) and (len(ch) <= 4 or (len(ch) <= 6 and m_prob < 0.48)):
                continue
            if len(ch) <= 2 or (d0 == 1 and d1 == 1 and (len(ch) <= 6 or (len(ch) <= 9 and m_prob < 0.58))):
                continue
        kept_chains.append(ch)

    split_chains: list[tuple[list[tuple[float, float]], bool]] = []
    for ch in kept_chains:
        pts = np.asarray(ch, dtype=np.float32)[:, ::-1]
        is_closed = len(ch) >= 6 and ch[0] == ch[-1]
        if is_closed:
            split_chains.append((pts.tolist(), True))
            continue
        if len(pts) >= 10:
            cut_idx = [0]
            step_w = 4
            for k in range(step_w, len(pts) - step_w):
                v1 = pts[k] - pts[k - step_w]
                v2 = pts[k + step_w] - pts[k]
                n1, n2 = np.hypot(v1[0], v1[1]), np.hypot(v2[0], v2[1])
                if n1 > 1e-3 and n2 > 1e-3 and float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)) < 0.72:
                    if k - cut_idx[-1] >= 4:
                        cut_idx.append(k)
            cut_idx.append(len(pts) - 1)
            for a, b in zip(cut_idx[:-1], cut_idx[1:]):
                if b > a:
                    split_chains.append((pts[a:b + 1].tolist(), False))
        else:
            split_chains.append((pts.tolist(), False))

    open_chains = [list(pts) for pts, cl in split_chains if not cl]
    closed_chains = [list(pts) for pts, cl in split_chains if cl]

    def _end_vec(pts_list, at_start: bool) -> np.ndarray:
        k = min(len(pts_list) - 1, 5)
        v = (np.array(pts_list[0], np.float32) - np.array(pts_list[k], np.float32)) if at_start else (np.array(pts_list[-1], np.float32) - np.array(pts_list[-1 - k], np.float32))
        return v / max(float(np.hypot(v[0], v[1])), 1e-5)

    hub: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    for ci, pts_l in enumerate(open_chains):
        if len(pts_l) < 2:
            continue
        hub[(int(round(pts_l[0][0])), int(round(pts_l[0][1])))].append((ci, True))
        hub[(int(round(pts_l[-1][0])), int(round(pts_l[-1][1])))].append((ci, False))

    pair_of: dict[tuple[int, bool], tuple[int, bool]] = {}
    for items in hub.values():
        if len(items) < 2:
            continue
        cands = []
        for i in range(len(items)):
            ci, si = items[i]
            vi = _end_vec(open_chains[ci], si)
            for j in range(i + 1, len(items)):
                cj, sj = items[j]
                if ci == cj:
                    continue
                vj = _end_vec(open_chains[cj], sj)
                dot_c = float(-(vi[0] * vj[0] + vi[1] * vj[1]))
                if dot_c >= 0.90:
                    cands.append((dot_c, (ci, si), (cj, sj)))
        cands.sort(reverse=True)
        used_ends = set()
        for _, a, b in cands:
            if a in used_ends or b in used_ends:
                continue
            used_ends.add(a)
            used_ends.add(b)
            pair_of[a] = b
            pair_of[b] = a

    # Vector-level gap bridging across short colinear breaks (<= 22 subgrid px = 11 SR px)
    free_ends = []
    for ci, pts_l in enumerate(open_chains):
        if len(pts_l) < 2:
            continue
        for si in (True, False):
            if (ci, si) not in pair_of:
                p_end = np.array(pts_l[0] if si else pts_l[-1], dtype=np.float32)
                free_ends.append((ci, si, p_end, _end_vec(open_chains[ci], si)))
    if len(free_ends) >= 2:
        gap_cands = []
        cell_sz = 22.0
        grid_fe = defaultdict(list)
        for idx_f, fe in enumerate(free_ends):
            grid_fe[(int(fe[2][1] // cell_sz), int(fe[2][0] // cell_sz))].append(idx_f)
        for idx_a, (ci, si, pa, va) in enumerate(free_ends):
            gy0, gx0 = int(pa[1] // cell_sz), int(pa[0] // cell_sz)
            for dy_g in (-1, 0, 1):
                for dx_g in (-1, 0, 1):
                    for idx_b in grid_fe.get((gy0 + dy_g, gx0 + dx_g), ()):
                        if idx_b <= idx_a:
                            continue
                        cj, sj, pb, vb = free_ends[idx_b]
                        if ci == cj:
                            continue
                        dv = pb - pa
                        dist_e = float(np.hypot(dv[0], dv[1]))
                        if not (1.0 <= dist_e <= 22.0):
                            continue
                        dir_ab = dv / dist_e
                        al_a = float(va[0] * dir_ab[0] + va[1] * dir_ab[1])
                        al_b = float(-(vb[0] * dir_ab[0] + vb[1] * dir_ab[1]))
                        if al_a < 0.76 or al_b < 0.76:
                            continue
                        ts = np.linspace(0.0, 1.0, max(3, int(dist_e) + 1), dtype=np.float32)
                        sxs = np.clip(np.rint(pa[0] + ts * dv[0]).astype(int), 0, W2 - 1)
                        sys_g = np.clip(np.rint(pa[1] + ts * dv[1]).astype(int), 0, H2 - 1)
                        if float(prob_2x[sys_g, sxs].mean()) >= 0.08 or float(prob_2x[sys_g, sxs].max()) >= 0.16:
                            score_g = (al_a + al_b) - 0.03 * dist_e
                            gap_cands.append((score_g, (ci, si), (cj, sj)))
        gap_cands.sort(reverse=True)
        for _, a, b in gap_cands:
            if a in pair_of or b in pair_of:
                continue
            pair_of[a] = b
            pair_of[b] = a

    consumed = set()
    final_curves: list[tuple[np.ndarray, bool]] = []
    for ci in range(len(open_chains)):
        if ci in consumed or len(open_chains[ci]) < 2:
            continue
        cur_i, back_end = ci, True
        seen_c = {ci}
        while (cur_i, back_end) in pair_of:
            pj, pj_meet_end = pair_of[(cur_i, back_end)]
            if pj in seen_c:
                break
            seen_c.add(pj)
            cur_i, back_end = pj, (not pj_meet_end)

        seq = []
        c_idx = cur_i
        enter_at_start = back_end
        vis_local = set()
        is_loop = False
        while c_idx not in vis_local:
            vis_local.add(c_idx)
            consumed.add(c_idx)
            pts_c = open_chains[c_idx] if enter_at_start else open_chains[c_idx][::-1]
            if not seq:
                seq.extend(pts_c)
            elif np.hypot(seq[-1][0] - pts_c[0][0], seq[-1][1] - pts_c[0][1]) > 0.5:
                seq.extend(pts_c)
            else:
                seq.extend(pts_c[1:])
            exit_end = (c_idx, False if enter_at_start else True)
            if exit_end not in pair_of:
                break
            nxt_idx, nxt_start = pair_of[exit_end]
            if nxt_idx in vis_local:
                is_loop = True
                break
            c_idx = nxt_idx
            enter_at_start = nxt_start
        if len(seq) >= 2:
            final_curves.append((np.asarray(seq, dtype=np.float32), is_loop))

    for pts_l in closed_chains:
        if len(pts_l) >= 4:
            final_curves.append((np.asarray(pts_l, dtype=np.float32), True))

    # 4. Snap to sub-pixel CNN scatter cloud, decode v4 width, and fit crisp Lines & Beziers
    L_sr = srgb_to_oklab(sr_rgb)[..., 0].astype(np.float32)
    paths_svg: list[str] = []
    centerlines_sr: list[np.ndarray] = []

    for pts_sub, is_closed in final_curves:
        pts_sr = pts_sub / float(sub)
        if len(pts_sr) < 2:
            continue

        ys_i = np.clip(np.round(pts_sr[:, 1]).astype(int), 0, hs - 1)
        xs_i = np.clip(np.round(pts_sr[:, 0]).astype(int), 0, ws - 1)
        j_v = junc[ys_i, xs_i]
        snap_w = np.where(j_v < 0.30, 0.55, 0.0)[:, None]
        if not is_closed:
            snap_w[0] = 0.0
            snap_w[-1] = 0.0
        pts_sr = pts_sr + snap_w * np.stack([dx_map[ys_i, xs_i], dy_map[ys_i, xs_i]], axis=1)

        centerlines_sr.append(pts_sr)

        ys_c = np.clip(np.round(pts_sr[:, 1]).astype(int), 0, hs - 1)
        xs_c = np.clip(np.round(pts_sr[:, 0]).astype(int), 0, ws - 1)
        Ls = L_sr[ys_c, xs_c]
        r_arr = np.maximum(0.45, rad_map[ys_c, xs_c])

        if force_color is not None:
            hex_col = force_color
            is_bright = False
        else:
            tang_v = np.gradient(pts_sr, axis=0) if len(pts_sr) >= 3 else np.vstack([pts_sr[1:2] - pts_sr[:1]] * len(pts_sr))
            tn_v = np.maximum(np.hypot(tang_v[:, 0], tang_v[:, 1]), 1e-4)
            nv_x, nv_y = -tang_v[:, 1] / tn_v, tang_v[:, 0] / tn_v

            def _sample_L(dpx: float) -> np.ndarray:
                ox = np.clip(np.rint(pts_sr[:, 0] + dpx * nv_x).astype(int), 0, ws - 1)
                oy = np.clip(np.rint(pts_sr[:, 1] + dpx * nv_y).astype(int), 0, hs - 1)
                return L_sr[oy, ox]

            # Bright core within ±1.5px catches thin white/colored hair strands, cat rims & snowflakes
            # even when the CNN polyline sits slightly on the strand's outer edge against a dark background.
            L_core_max = np.maximum(Ls, np.maximum(_sample_L(1.5), _sample_L(-1.5)))
            flp_min = np.minimum(
                np.minimum(np.minimum(_sample_L(2.0), _sample_L(3.2)), _sample_L(4.5)),
                np.minimum(np.minimum(_sample_L(6.0), _sample_L(7.5)), _sample_L(8.8)),
            )
            flm_min = np.minimum(
                np.minimum(np.minimum(_sample_L(-2.0), _sample_L(-3.2)), _sample_L(-4.5)),
                np.minimum(np.minimum(_sample_L(-6.0), _sample_L(-7.5)), _sample_L(-8.8)),
            )
            max_flank = max(float(np.median(flp_min)), float(np.median(flm_min)))
            med_cmax = float(np.median(L_core_max))
            q40_cmax = float(np.quantile(L_core_max, 0.40))
            med_L = float(np.median(Ls))
            is_bright = (
                (med_cmax >= max_flank + 0.08)
                and (q40_cmax >= max_flank + 0.03)
                and (max_flank < 0.40)
                and (len(pts_sr) >= 4 or med_cmax > 0.42)
            )
            cols_c = sr_rgb[ys_c, xs_c]
            if is_bright:
                top_idx = np.argsort(L_core_max)[-max(1, len(Ls) // 3):]
                bright_s = []
                for pi in top_idx:
                    cy_p, cx_p = int(ys_c[pi]), int(xs_c[pi])
                    y0p, y1p = max(0, cy_p - 2), min(hs, cy_p + 3)
                    x0p, x1p = max(0, cx_p - 2), min(ws, cx_p + 3)
                    pl = L_sr[y0p:y1p, x0p:x1p]
                    pm_y, pm_x = np.unravel_index(int(np.argmax(pl)), pl.shape)
                    bright_s.append(sr_rgb[y0p + pm_y, x0p + pm_x])
                rep_rgb = np.median(np.asarray(bright_s, dtype=np.float32), axis=0).astype(np.uint8)
            else:
                # Only sample 3x3 darkest core pixel along verified thin dark troughs (both flanks brighter)
                # so step boundaries beside black backgrounds or shadows never grab neighbor black pixels.
                min_both_flank = float(min(np.median(flp_min), np.median(flm_min)))
                is_true_trough = (min_both_flank >= med_L + 0.03)
                if is_true_trough:
                    core_cols = []
                    core_Ls = []
                    for cy_p, cx_p in zip(ys_c.tolist(), xs_c.tolist()):
                        y0p, y1p = max(0, cy_p - 1), min(hs, cy_p + 2)
                        x0p, x1p = max(0, cx_p - 1), min(ws, cx_p + 2)
                        pl = L_sr[y0p:y1p, x0p:x1p]
                        pm_y, pm_x = np.unravel_index(int(np.argmin(pl)), pl.shape)
                        core_cols.append(sr_rgb[y0p + pm_y, x0p + pm_x])
                        core_Ls.append(float(pl[pm_y, pm_x]))
                    core_Ls_arr = np.asarray(core_Ls, dtype=np.float32)
                    core_cols_arr = np.asarray(core_cols, dtype=np.uint8)
                    med_core_L = float(np.median(core_Ls_arr))
                    pick_q = int((0.28 if med_core_L >= 0.45 else 0.22) * (len(core_Ls_arr) - 1))
                    rep_rgb = core_cols_arr[np.argsort(core_Ls_arr)[max(0, pick_q)]]
                else:
                    pick_q = int((0.40 if med_L >= 0.45 else 0.22) * (len(Ls) - 1))
                    rep_rgb = cols_c[np.argsort(Ls)[max(0, pick_q)]]
            hex_col = "#%02x%02x%02x" % (int(rep_rgb[0]), int(rep_rgb[1]), int(rep_rgb[2]))

        pts_1x = (pts_sr + 0.5) / sr_scale
        sw_arr = np.clip((r_arr * 2.0) / sr_scale, 0.58, 4.80)
        if is_bright:
            if med_L < med_cmax - 0.08:
                sw_arr = np.clip(sw_arr * 0.50, 0.45, 0.82)
            else:
                sw_arr = np.clip(sw_arr * 0.68, 0.52, 1.25)
        sw_cal = float(np.median(sw_arr))

        if (not is_bright) and (tapered or float(sw_arr.max() - sw_arr.min()) >= 0.55) and len(pts_1x) >= 6:
            d_str = _variable_width_envelope_d(pts_1x, sw_arr * 0.5, closed=is_closed)
            if d_str:
                paths_svg.append(f'<path d="{d_str}" fill="{hex_col}"/>')
                continue

        segs = _fit_line_or_structural(pts_1x, is_closed=is_closed)
        d_str = _beziers_to_svg_d(segs, closed=is_closed)
        if d_str:
            paths_svg.append(
                f'<path d="{d_str}" fill="none" stroke="{hex_col}" stroke-width="{sw_cal:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )

    if return_centerlines:
        return paths_svg, wall_hi, centerlines_sr
    return paths_svg, wall_hi
