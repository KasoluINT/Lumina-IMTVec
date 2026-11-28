"""Standalone cubic Bézier and circular-arc curve fitting utilities (Schneider 1990)."""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit-length 2D vector, or original vector if near-zero."""
    norm = float(np.hypot(v[0], v[1]))
    return v / norm if norm > 1e-12 else v


def _chord_params(pts: np.ndarray) -> np.ndarray:
    """Chord-length parameterization in [0, 1] for an (N, 2) polyline."""
    diffs = np.diff(pts, axis=0)
    cum = np.concatenate(([0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))))
    total = float(cum[-1])
    return cum / total if total > 0.0 else np.linspace(0.0, 1.0, len(pts))


def _eval_cubic(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    t = u[:, None]
    mt = 1.0 - t
    return (
        (mt ** 3) * ctrl[0]
        + 3.0 * (mt ** 2) * t * ctrl[1]
        + 3.0 * mt * (t ** 2) * ctrl[2]
        + (t ** 3) * ctrl[3]
    )


def _eval_cubic_d1(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    t = u[:, None]
    mt = 1.0 - t
    return (
        3.0 * (mt ** 2) * (ctrl[1] - ctrl[0])
        + 6.0 * mt * t * (ctrl[2] - ctrl[1])
        + 3.0 * (t ** 2) * (ctrl[3] - ctrl[2])
    )


def _eval_cubic_d2(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    t = u[:, None]
    mt = 1.0 - t
    return 6.0 * mt * (ctrl[2] - 2.0 * ctrl[1] + ctrl[0]) + 6.0 * t * (ctrl[3] - 2.0 * ctrl[2] + ctrl[1])


def _fit_one(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Least-squares single cubic Bézier matching endpoint tangents `t1`, `t2`."""
    mt = 1.0 - u
    a1 = np.outer(3.0 * (mt ** 2) * u, t1)
    a2 = np.outer(3.0 * mt * (u ** 2), t2)
    c00 = float((a1 * a1).sum())
    c01 = float((a1 * a2).sum())
    c11 = float((a2 * a2).sum())

    b0 = (mt ** 3) + 3.0 * (mt ** 2) * u
    b1 = 3.0 * mt * (u ** 2) + (u ** 3)
    rhs = pts - np.outer(b0, pts[0]) - np.outer(b1, pts[-1])
    x0 = float((a1 * rhs).sum())
    x1 = float((a2 * rhs).sum())

    det = c00 * c11 - c01 * c01
    chord = float(np.hypot(*(pts[-1] - pts[0])))
    if abs(det) > 1e-12:
        al1 = (x0 * c11 - x1 * c01) / det
        al2 = (c00 * x1 - c01 * x0) / det
    else:
        al1 = al2 = chord / 3.0

    if al1 < 1e-6 * chord or al2 < 1e-6 * chord or al1 > 3.0 * chord or al2 > 3.0 * chord:
        al1 = al2 = chord / 3.0
    return np.array([pts[0], pts[0] + t1 * al1, pts[-1] + t2 * al2, pts[-1]], dtype=float)


def _max_err(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    diff = _eval_cubic(ctrl, u) - pts
    dist = np.hypot(diff[:, 0], diff[:, 1])
    idx = int(dist.argmax())
    return float(dist[idx]), idx


def _newton_reparam(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    q = _eval_cubic(ctrl, u) - pts
    d1 = _eval_cubic_d1(ctrl, u)
    d2 = _eval_cubic_d2(ctrl, u)
    num = (q * d1).sum(axis=1)
    den = (d1 * d1).sum(axis=1) + (q * d2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_next = np.where(np.abs(den) > 1e-12, u - num / den, u)
    return np.clip(u_next, 0.0, 1.0)


def fit_cubic(pts: np.ndarray, t1: np.ndarray, t2: np.ndarray, tol: float, depth: int = 0) -> list[np.ndarray]:
    """Recursively fit an (N, 2) point sequence with cubic Béziers within `tol`."""
    if len(pts) == 2 or depth > 12:
        step = float(np.hypot(*(pts[-1] - pts[0]))) / 3.0
        return [np.array([pts[0], pts[0] + t1 * step, pts[-1] + t2 * step, pts[-1]], dtype=float)]

    u = _chord_params(pts)
    ctrl = _fit_one(pts, u, t1, t2)
    err, split = _max_err(pts, ctrl, u)
    if err < tol:
        return [ctrl]

    if err < tol * 4.0:
        for _ in range(4):
            u = _newton_reparam(pts, ctrl, u)
            ctrl = _fit_one(pts, u, t1, t2)
            err, split = _max_err(pts, ctrl, u)
            if err < tol:
                return [ctrl]

    split = min(max(split, 1), len(pts) - 2)
    tc = _unit(pts[split - 1] - pts[split + 1])
    return fit_cubic(pts[: split + 1], t1, tc, tol, depth + 1) + fit_cubic(
        pts[split:], -tc, t2, tol, depth + 1
    )


def fit_circle_kasa(pts: np.ndarray) -> tuple[float, float, float] | None:
    """Kasa algebraic least-squares circle fit returning `(cx, cy, r)` or None."""
    if len(pts) < 5:
        return None
    a_mat = np.c_[2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(len(pts))]
    b_vec = (pts ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c_val = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c_val + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 1e-6:
        return None
    return cx, cy, float(np.sqrt(r2))


def cubic_to_svg_arc(ctrl: np.ndarray, tol: float = 0.45, n: int = 9) -> str | None:
    """Convert a nearly circular cubic Bézier to an exact SVG `A` arc command."""
    u = np.linspace(0.0, 1.0, n)
    pts = _eval_cubic(ctrl, u)
    circ = fit_circle_kasa(pts)
    if circ is None:
        return None
    cx, cy, r = circ
    dev = float(np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).max())
    if dev > tol or r > 4000.0:
        return None

    a0 = float(np.arctan2(ctrl[0][1] - cy, ctrl[0][0] - cx))
    a1 = float(np.arctan2(ctrl[3][1] - cy, ctrl[3][0] - cx))
    mid = float(np.arctan2(ctrl[1][1] - cy, ctrl[1][0] - cx))
    ccw = np.sin(mid - a0) > 0.0
    sweep = (a1 - a0) % (2.0 * np.pi) if ccw else -((a0 - a1) % (2.0 * np.pi))
    if abs(sweep) >= np.radians(195.0):
        return None

    am = a0 + 0.5 * sweep
    arc_mid_x, arc_mid_y = cx + r * np.cos(am), cy + r * np.sin(am)
    cub_mid = pts[len(pts) // 2]
    if float(np.hypot(arc_mid_x - cub_mid[0], arc_mid_y - cub_mid[1])) > max(0.8, tol * 1.5):
        return None
    return "A%s,%s 0 %d %d %s,%s" % (
        fmt_coord(r),
        fmt_coord(r),
        1 if abs(sweep) > np.pi else 0,
        1 if sweep > 0.0 else 0,
        fmt_coord(ctrl[3][0]),
        fmt_coord(ctrl[3][1]),
    )


def fmt_coord(v: float) -> str:
    """Compact SVG coordinate formatter (0.01 precision at 2x SR = 0.005px at 1x, zero bloat)."""
    s = f"{v:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def fit_closed_circle_loop(pieces: list[np.ndarray], tol: float = 0.38) -> tuple[float, float, float, int] | None:
    """Test whether a closed loop of cubic Béziers is an exact circle within `tol`.

    Order-invariant Kasa fit guarantees that an island's outer loop and its enclosing parent's
    inner hole (reversed traversal of the same chain) produce the exact same `(cx, cy, r)`,
    preserving 100% zero-gap planar water-tightness.

    Returns `(cx, cy, r, sweep_flag)` where `sweep_flag` matches the loop's winding order, or None.
    """
    if not pieces:
        return None
    u = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    sampled = np.vstack([_eval_cubic(s, u) for s in pieces])
    if len(sampled) < 8:
        return None

    # Order-invariant centroid & Kasa fit (sort by rounded angle/coord is unnecessary because
    # normal equations A^T A and A^T b are pure sums, strictly invariant to row permutation!)
    circ = fit_circle_kasa(sampled)
    if circ is None:
        return None
    cx, cy, r = circ
    if r < 1.8 or r > 2000.0:
        return None

    dx = sampled[:, 0] - cx
    dy = sampled[:, 1] - cy
    dists = np.hypot(dx, dy)
    # Tight radial deviation check (scales slightly with radius, capped strictly)
    max_dev = float(np.abs(dists - r).max())
    if max_dev > min(tol, max(0.24, r * 0.065)):
        return None

    # Require full 4-quadrant angular coverage (rejects arcs / folded slivers)
    ang = np.arctan2(dy, dx)
    q = ((ang >= 0.0).astype(int) * 2) + ((dx >= 0.0).astype(int))
    if len(np.unique(q)) < 4:
        return None

    # Signed area via shoelace formula to preserve exact loop winding (CW vs CCW)
    x = sampled[:, 0]
    y = sampled[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area2) < 0.78 * 2.0 * np.pi * (r ** 2):
        return None
    sweep = 1 if area2 > 0.0 else 0
    return cx, cy, r, sweep


def fit_closed_ellipse_loop(pieces: list[np.ndarray], tol: float = 0.42) -> tuple[float, float, float, float, float, int] | None:
    """Order-invariant PCA ellipse fit for a closed loop of cubic Béziers.

    Guarantees identical `(cx, cy, rx, ry, rot_deg)` regardless of CW/CCW traversal order
    so an isolated island and its enclosing parent hole stay 100% zero-gap water-tight.
    """
    if not pieces:
        return None
    u = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    sampled = np.vstack([_eval_cubic(s, u) for s in pieces])
    if len(sampled) < 10:
        return None

    cx, cy = float(sampled[:, 0].mean()), float(sampled[:, 1].mean())
    rel = sampled - np.array([cx, cy])
    cov = (rel.T @ rel) / float(len(sampled))
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Order-invariant principal axes: major axis = idx 1, minor = idx 0; canonicalize sign
    u_maj = eigvecs[:, 1].copy()
    if u_maj[0] < -1e-9 or (abs(u_maj[0]) <= 1e-9 and u_maj[1] < 0.0):
        u_maj = -u_maj
    u_min = np.array([-u_maj[1], u_maj[0]], dtype=float)

    p = rel @ u_maj
    q = rel @ u_min
    rx = float(np.sqrt(2.0 * np.mean(p * p)))
    ry = float(np.sqrt(2.0 * np.mean(q * q)))
    if ry < 1.8 or rx > 2000.0 or (rx / max(ry, 1e-6)) > 4.5 or (rx / max(ry, 1e-6)) < 1.08:
        return None

    norm_r = np.hypot(p / rx, q / ry)
    dev_px = float(np.abs(norm_r - 1.0).max() * ry)
    if dev_px > min(tol, max(0.26, ry * 0.065)):
        return None

    ang = np.arctan2(q / ry, p / rx)
    quad = ((ang >= 0.0).astype(int) * 2) + ((p >= 0.0).astype(int))
    if len(np.unique(quad)) < 4:
        return None

    x = sampled[:, 0]
    y = sampled[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area2) < 0.80 * 2.0 * np.pi * rx * ry:
        return None
    rot_deg = float(np.degrees(np.arctan2(u_maj[1], u_maj[0])))
    sweep = 1 if area2 > 0.0 else 0
    return cx, cy, rx, ry, rot_deg, sweep


def fit_arc_span(cubics: list[np.ndarray], tol: float = 0.35) -> tuple[float, int, int, np.ndarray, np.ndarray] | None:
    """Fit a single circular SVG arc across 1..N consecutive cubic Béziers within `tol`.

    Returns `(r, large_arc, sweep, start_pt, end_pt)` or None.
    """
    if not cubics:
        return None
    u = np.linspace(0.0, 1.0, 7 if len(cubics) == 1 else 5)
    pts_list = [_eval_cubic(cubics[0], u)] + [_eval_cubic(c, u[1:]) for c in cubics[1:]]
    pts = np.vstack(pts_list)
    p0, p1 = cubics[0][0], cubics[-1][3]
    chord2 = float(np.sum((p1 - p0) ** 2))
    if chord2 <= 4.0:
        return None

    circ = fit_circle_kasa(pts)
    if circ is None:
        return None
    cx, cy, r = circ
    if r < 1.5 or r > 4000.0:
        return None
    dev = float(np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).max())
    if dev > tol:
        return None

    # Verify monotonic angular progression & total sweep < 175 deg
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    unwrapped = np.unwrap(ang)
    diffs = np.diff(unwrapped)
    tot_sweep = float(unwrapped[-1] - unwrapped[0])
    if abs(tot_sweep) < np.radians(4.0) or abs(tot_sweep) >= np.radians(175.0):
        return None
    if tot_sweep > 0.0 and float(diffs.min()) < -0.08:
        return None
    if tot_sweep < 0.0 and float(diffs.max()) > 0.08:
        return None

    # Verify arc midpoint matches cubic span midpoint closely
    am = float(unwrapped[0] + 0.5 * tot_sweep)
    arc_mid = np.array([cx + r * np.cos(am), cy + r * np.sin(am)])
    cub_mid = pts[len(pts) // 2]
    if float(np.hypot(*(arc_mid - cub_mid))) > max(0.75, tol * 1.5):
        return None

    large_arc = 1 if abs(tot_sweep) > np.pi else 0
    sweep = 1 if tot_sweep > 0.0 else 0
    return r, large_arc, sweep, p0, p1
