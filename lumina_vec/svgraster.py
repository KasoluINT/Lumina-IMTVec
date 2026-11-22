"""Minimal rasteriser for path data (M/L/H/V/C/Q/A/Z, absolute): id maps & clean preview rendering."""
import math
import re
import numpy as np
import cv2

_TOK = re.compile(r"[MLHVCQAZmlhvcqaz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _arc_points(pt: np.ndarray, rx: float, ry: float, rot_deg: float, fA: int, fS: int, end: np.ndarray, n_seg: int = 8) -> list[np.ndarray]:
    """Subdivide an SVG elliptical arc `(pt -> end)` into smooth polyline samples."""
    x1, y1 = float(pt[0]), float(pt[1])
    x2, y2 = float(end[0]), float(end[1])
    if rx <= 1e-6 or ry <= 1e-6 or (abs(x1 - x2) < 1e-7 and abs(y1 - y2) < 1e-7):
        return [end]
    phi = math.radians(rot_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) * 0.5, (y1 - y2) * 0.5
    x1p = cp * dx2 + sp * dy2
    y1p = -sp * dx2 + cp * dy2
    rx, ry = abs(rx), abs(ry)
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        s_l = math.sqrt(lam)
        rx *= s_l
        ry *= s_l
    rx2, ry2 = rx * rx, ry * ry
    num = max(0.0, rx2 * ry2 - rx2 * y1p * y1p - ry2 * x1p * x1p)
    den = rx2 * y1p * y1p + ry2 * x1p * x1p
    fac = math.sqrt(num / max(den, 1e-12)) * (-1.0 if fA == fS else 1.0)
    cxp = fac * (rx * y1p / max(ry, 1e-9))
    cyp = fac * (-ry * x1p / max(rx, 1e-9))
    cx = cp * cxp - sp * cyp + (x1 + x2) * 0.5
    cy = sp * cxp + cp * cyp + (y1 + y2) * 0.5

    def _ang(ux: float, uy: float, vx: float, vy: float) -> float:
        d = max(-1.0, min(1.0, (ux * vx + uy * vy) / max(1e-12, math.hypot(ux, uy) * math.hypot(vx, vy))))
        a = math.acos(d)
        return -a if (ux * vy - uy * vx) < 0 else a

    vx0, vy0 = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx1, vy1 = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    t1 = _ang(1.0, 0.0, vx0, vy0)
    dt = _ang(vx0, vy0, vx1, vy1)
    if fS == 0 and dt > 0:
        dt -= 2.0 * math.pi
    elif fS == 1 and dt < 0:
        dt += 2.0 * math.pi

    pts: list[np.ndarray] = []
    for k in range(1, n_seg + 1):
        th = t1 + dt * (k / n_seg)
        ct, st = math.cos(th), math.sin(th)
        px = cp * (rx * ct) - sp * (ry * st) + cx
        py = sp * (rx * ct) + cp * (ry * st) + cy
        pts.append(np.array([px, py], dtype=np.float64))
    return pts


def path_polys(d, n_bez=8):
    toks = _TOK.findall(d)
    polys, cur, pt, start, i, cmd = [], [], np.zeros(2), np.zeros(2), 0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t.upper(); i += 1
            if cmd == "Z":
                if cur:
                    polys.append(np.array(cur)); cur = []
                pt = start.copy()
                continue
            continue
        if cmd == "M":
            if cur:
                polys.append(np.array(cur))
            pt = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
            start = pt.copy(); cur = [pt]; cmd = "L"
        elif cmd == "L":
            pt = np.array([float(toks[i]), float(toks[i + 1])]); i += 2; cur.append(pt)
        elif cmd == "H":
            pt = np.array([float(toks[i]), pt[1]]); i += 1; cur.append(pt)
        elif cmd == "V":
            pt = np.array([pt[0], float(toks[i])]); i += 1; cur.append(pt)
        elif cmd == "Q":
            c = np.array([float(v) for v in toks[i:i + 4]]).reshape(2, 2); i += 4
            t_ = np.linspace(0, 1, n_bez + 1)[1:, None]
            b = ((1 - t_) ** 2) * pt + 2 * (1 - t_) * t_ * c[0] + (t_ ** 2) * c[1]
            cur.extend(list(b)); pt = c[1]
        elif cmd == "C":
            c = np.array([float(v) for v in toks[i:i + 6]]).reshape(3, 2); i += 6
            t_ = np.linspace(0, 1, n_bez + 1)[1:, None]
            b = ((1 - t_) ** 3) * pt + 3 * ((1 - t_) ** 2) * t_ * c[0] + 3 * (1 - t_) * t_ ** 2 * c[1] + t_ ** 3 * c[2]
            cur.extend(list(b)); pt = c[2]
        elif cmd == "A":
            rx, ry, rot = float(toks[i]), float(toks[i + 1]), float(toks[i + 2])
            fa, fs = int(float(toks[i + 3])), int(float(toks[i + 4]))
            end = np.array([float(toks[i + 5]), float(toks[i + 6])])
            i += 7
            cur.extend(_arc_points(pt, rx, ry, rot, fa, fs, end, n_seg=n_bez))
            pt = end
        else:
            i += 1
    if cur:
        polys.append(np.array(cur))
    return polys


def rasterize_ids(ds, W, H, ss=2):
    """int32 map (H*ss x W*ss): 1-based index of the last path covering each pixel (even-odd)."""
    lab = np.zeros((H * ss, W * ss), np.int32)
    for k, d in enumerate(ds, start=1):
        polys = [np.round(p * ss * 16).astype(np.int32) for p in path_polys(d) if len(p) >= 3]
        if not polys:
            continue
        allp = np.vstack(polys) >> 4
        x0, y0 = np.maximum(allp.min(0) - 1, 0); x1, y1 = allp.max(0) + 2
        x1, y1 = min(x1, W * ss), min(y1, H * ss)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0, y0]) * 16
        cv2.fillPoly(m, [p - off for p in polys], 1, lineType=cv2.LINE_8, shift=4)
        lab[y0:y1, x0:x1][m > 0] = k
    return lab
