"""线稿几何：把亚像素轮廓拟合成参考 SVG 那种 L / Q / A / C 混合路径。

参考文件（person_...vector.svg）的规律（从代码读出）：
- 每个色块两条 path：fill 一条 + class="qq3d-gap-filler" 的描边一条，两条 d 完全相同；
- gap-filler：stroke = 该块填充色，stroke-width=2，round join/cap，vector-effect="non-scaling-stroke"；
- 路径由 M / Q / A / L / C 混合组成，圆弧 rx≈ry 带旋转角 → 按段做"最省原语拟合"，
  直线优先，其次二次、圆弧、三次，残差超限才升级；
- 坐标打印成 float32 展开值（149.41000366210938）→ 生成器是 Python/numpy。

本模块：DP 粗分 → 相邻段合并（同一原语还能拟合就并）→ 每段选最省原语 → 输出 d。
"""
import numpy as np

RAW = ("M", "L", "Q", "A", "C")


def _f(x):
    return repr(float(np.float32(x)))


def dp_simplify_idx(pts, tol):
    n = len(pts)
    if n < 3:
        return [0, n - 1]
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = pts[j] - pts[i]
        L = np.hypot(*seg)
        rel = pts[i + 1:j] - pts[i]
        if L < 1e-9:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return [int(i) for i in np.nonzero(keep)[0]]


def _err_line(seg):
    a, b = seg[0], seg[-1]
    v = b - a
    L = np.hypot(*v)
    if L < 1e-9:
        return 1e9
    rel = seg - a
    return float(np.max(np.abs(v[0] * rel[:, 1] - v[1] * rel[:, 0]) / L))


def _err_quad(seg):
    p0, p2 = seg[0], seg[-1]
    t0 = seg[min(2, len(seg) - 1)] - seg[0]
    t1 = seg[-1] - seg[max(-3, -len(seg))]
    n0, n1 = np.hypot(*t0), np.hypot(*t1)
    if n0 < 1e-9 or n1 < 1e-9:
        return 1e9, None
    t0 /= n0; t1 /= n1
    den = t0[0] * t1[1] - t0[1] * t1[0]
    if abs(den) < 1e-9:
        return 1e9, None
    w = p2 - p0
    s = (w[0] * t1[1] - w[1] * t1[0]) / den
    c = p0 + t0 * s
    tt = np.linspace(0, 1, max(12, len(seg)))[:, None]
    q = (1 - tt) ** 2 * p0 + 2 * tt * (1 - tt) * c + tt ** 2 * p2
    err = 0.0
    for pt in seg:
        err = max(err, float(np.min(np.hypot(q[:, 0] - pt[0], q[:, 1] - pt[1]))))
    return err, c


def _err_arc(seg):
    x, y = seg[:, 0], seg[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    bb = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    except np.linalg.LinAlgError:
        return 1e9, None, 0.0
    cx, cy, c = sol
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0 or not np.isfinite(r2):
        return 1e9, None, 0.0
    r = float(np.sqrt(r2))
    err = float(np.max(np.abs(np.hypot(x - cx, y - cy) - r)))
    return err, np.array([cx, cy]), r


def _segment_prim(seg, tol):
    """给一段点选最省的原语，返回 (kind, params, err)"""
    if len(seg) <= 2:
        return "L", None, 0.0
    e_line = _err_line(seg)
    if e_line <= tol * 0.6:
        return "L", None, e_line
    # 原语优先级：直线 → （圆弧 vs 二次，按拟合精度择优）→ 兜底直线。
    # 参考文件里 Q:A ≈ 1.2:1，圆心/圆角段基本都走圆弧，所以弧明显更准时优先用弧。
    e_q, cq = _err_quad(seg)
    e_arc, c, r = _err_arc(seg)
    arc_ok = c is not None and e_arc <= tol and r < 1e4
    quad_ok = cq is not None and e_q <= tol
    if arc_ok and quad_ok:
        if e_arc < 0.55 * e_q:
            ang = np.arctan2(seg[:, 1] - c[1], seg[:, 0] - c[0])
            dd = np.unwrap(ang)
            span = dd[-1] - dd[0]
            return "A", (r, 0.0, 1 if abs(span) > np.pi else 0, 1 if span > 0 else 0), e_arc
        return "Q", cq, e_q
    if quad_ok:
        return "Q", cq, e_q
    if arc_ok:
        ang = np.arctan2(seg[:, 1] - c[1], seg[:, 0] - c[0])
        dd = np.unwrap(ang)
        span = dd[-1] - dd[0]
        return "A", (r, 0.0, 1 if abs(span) > np.pi else 0, 1 if span > 0 else 0), e_arc
    return "L", None, e_line


def fit_runs(pts, tol=0.6):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return [("M", pts[0]), ("L", pts[-1])]
    idx = dp_simplify_idx(pts, tol * 2.0)
    runs = [[idx[k], idx[k + 1]] for k in range(len(idx) - 1)]

    prims = []
    for i, j in runs:
        seg = pts[i:j + 1]
        kind, p, err = _segment_prim(seg, tol)
        prims.append([i, j, kind, p])

    # 合并"同一圆上的相邻弧"（参考文件的圆/圆角通常 2~3 段）
    k = 0
    while k < len(prims) - 1:
        a, b = prims[k], prims[k + 1]
        if a[2] == "A" and b[2] == "A":
            i, j = a[0], b[1]
            seg = pts[i:j + 1]
            kind, p, err = _segment_prim(seg, tol)
            if kind == "A" and abs(p[0] - a[3][0]) < 0.02 * a[3][0] + 0.2:
                c = _arc_center(seg)
                ang = np.arctan2(seg[:, 1] - c[1], seg[:, 0] - c[0])
                span = abs(float(np.unwrap(ang)[-1] - np.unwrap(ang)[0]))
                if span < 5.9:
                    prims[k] = [i, j, kind, p]
                    del prims[k + 1]
                    continue
        k += 1

    out = [("M", pts[prims[0][0]])]
    for i, j, kind, p in prims:
        seg = pts[i:j + 1]
        b = seg[-1]
        if kind == "L":
            out.append(("L", b))
        elif kind == "Q":
            out.append(("Q", p, b))
        else:
            out.append(("A", p[0], p[1], p[2], p[3], b))

    # 整条是一个"起点=终点"的圆弧时，拆成两段，避免 SVG 里的退化弧
    if len(out) == 2 and out[1][0] == "A" and np.hypot(*(out[1][5] - out[0][1])) < 1e-6:
        mid = pts[len(pts) // 2]
        seg1 = pts[:len(pts) // 2 + 1]
        seg2 = np.vstack([pts[len(pts) // 2:], pts[:1]])
        c = _arc_center(pts)
        r = float(np.median(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])))
        out = [("M", pts[0]), ("A", r, 0.0, 0, out[1][4], mid),
               ("A", r, 0.0, 0, out[1][4], pts[0])]
    return out


def _arc_center(seg):
    x, y = seg[:, 0], seg[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    bb = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    return np.array([sol[0], sol[1]])


def to_d(segs, closed=True):
    parts = []
    for s in segs:
        if s[0] == "M":
            parts.append("M %s %s" % (_f(s[1][0]), _f(s[1][1])))
        elif s[0] == "L":
            parts.append("L %s %s" % (_f(s[1][0]), _f(s[1][1])))
        elif s[0] == "Q":
            parts.append("Q %s %s %s %s" % (_f(s[1][0]), _f(s[1][1]), _f(s[2][0]), _f(s[2][1])))
        elif s[0] == "A":
            parts.append("A %s %s %s 0 %d %s %s" % (_f(s[1]), _f(s[1]), _f(s[2]), s[4],
                                                    _f(s[5][0]), _f(s[5][1])))
        elif s[0] == "C":
            parts.append("C %s %s %s %s %s %s" % (_f(s[1][0]), _f(s[1][1]), _f(s[2][0]), _f(s[2][1]),
                                                  _f(s[3][0]), _f(s[3][1])))
    if closed:
        parts.append("Z")
    return " ".join(parts)
