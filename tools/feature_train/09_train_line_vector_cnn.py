"""Train LineVectorNet v4 (9-channel Neural Vector-Field CNN) with shell-free, sub-pixel
anti-staircase supervision on 4x-supersampled synthetic SVG geometry + continuous multi-scale
targets across v1/v2 galleries and E:\\图集, then export weights/line_vector_net.onnx.

Predicted channels in [0, 1]:
  0: stroke_prob  (single unified medial ridge for both 1px hairlines and 3-11px bold/Y strokes)
  1: color_bound  (smooth sub-pixel iso-luminant cel / flat color-block seam wall, haze-free)
  2: end_prob     (open curve tip endpoint heatmap)
  3: cross_junc   (T-junction / X-crossing / Y-hub intersection heatmap)
  4: offset_dx    (sub-pixel X shift to curve center: (v - 0.5) * 8.0 px, +-4.0 px reach)
  5: offset_dy    (sub-pixel Y shift to curve center: (v - 0.5) * 8.0 px, +-4.0 px reach)
  6: dir_cos2t    (tangent field 0.5 * (cos(2*theta) + 1))
  7: dir_sin2t    (tangent field 0.5 * (sin(2*theta) + 1))
  8: half_width   (local stroke half-width r / 6.0 px, covers strokes up to 12 px wide)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import concurrent.futures
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lumina_vec.color import srgb_to_oklab
from lumina_vec.feature_nets import LineVectorNet
from lumina_vec.feature_routing import _SESSION_CACHE, _get_session, _upscale_sr, _WEIGHTS_DEFAULT

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
VEC_V2_CKPT = CACHE_DIR / "line_vector_v2_best.pt"
VEC_V4_CKPT = CACHE_DIR / "line_vector_v4_best.pt"
V4_TILES_DIR = CACHE_DIR / "v4_tiles_cache"
VEC_ONNX = ROOT / "weights" / "line_vector_net.onnx"

OFFSET_SCALE = 8.0
RAD_SCALE = 6.0


def _eval_cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    u = 1.0 - t
    pts = (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3
    deriv = 3 * (u ** 2) * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * (t ** 2) * (p3 - p2)
    norm = np.maximum(np.hypot(deriv[:, 0], deriv[:, 1]), 1e-5)[:, None]
    tang = deriv / norm
    return pts, tang, t[:, 0]


def _bilinear_sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img.astype(np.float32),
        x.astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def generate_synthetic_vector_tile(rng: np.random.Generator, size: int = 384) -> tuple[np.ndarray, np.ndarray, float]:
    """Render a 4x-supersampled synthetic anime/stained-glass/prism illustration tile paired with
    exact analytical vector ground truth `[9, H, W]` (`OFFSET_SCALE=8.0`, `RAD_SCALE=6.0`)."""
    ss = 4
    S = size * ss
    ss_b = 2
    Sb = size * ss_b

    # 1. Backdrop with sharp SDF cel blocks + explicit smooth gradients (hard negatives for haze!)
    is_dark_night = rng.random() < 0.30
    if is_dark_night:
        pal_bg = [
            (6, 6, 10), (12, 12, 22), (18, 14, 30), (26, 20, 48),
            (42, 28, 68), (22, 34, 62), (76, 36, 78), (190, 206, 232),
        ]
        base_col = pal_bg[int(rng.integers(0, 4))]
    else:
        pal_bg = [
            (252, 250, 252), (249, 229, 224), (124, 130, 168), (222, 146, 114),
            (56, 54, 78), (242, 232, 212), (255, 228, 128), (186, 206, 232), (140, 110, 185),
        ]
        base_col = pal_bg[int(rng.integers(0, len(pal_bg)))]
    canvas = np.full((S, S, 3), base_col, dtype=np.float32)

    # Multi-stop soft sky/hair gradient (~45% of tiles) so CNN learns strictly ZERO gradient haze
    if rng.random() < 0.45:
        gx_f = np.linspace(0.0, 1.0, S, dtype=np.float32)[None, :]
        gy_f = np.linspace(0.0, 1.0, S, dtype=np.float32)[:, None]
        ang_g = float(rng.uniform(0, 2 * np.pi))
        ramp = np.clip(0.5 + 0.5 * ((gx_f - 0.5) * np.cos(ang_g) + (gy_f - 0.5) * np.sin(ang_g)) * 1.4, 0.0, 1.0)[..., None]
        c_grad = np.array(pal_bg[int(rng.integers(0, len(pal_bg)))], dtype=np.float32)
        canvas = canvas * (1.0 - 0.55 * ramp) + c_grad * (0.55 * ramp)

    # Sub-pixel continuous Euclidean SDF color-block rim mask at 2x resolution
    bound_2x = np.zeros((Sb, Sb), dtype=np.float32)
    yy_s, xx_s = np.mgrid[0:S, 0:S].astype(np.float32)
    n_blocks = int(rng.integers(2, 6))
    for _ in range(n_blocks):
        cx = float(rng.uniform(0.1 * S, 0.9 * S))
        cy = float(rng.uniform(0.1 * S, 0.9 * S))
        ang = float(rng.uniform(0, 2 * np.pi))
        rx = float(rng.uniform(0.10 * S, 0.50 * S))
        ry = float(rng.uniform(0.05 * S, 0.35 * S))
        col_b = np.array(pal_bg[int(rng.integers(0, len(pal_bg)))], dtype=np.float32)
        dx_r = (xx_s - cx) * np.cos(ang) + (yy_s - cy) * np.sin(ang)
        dy_r = -(xx_s - cx) * np.sin(ang) + (yy_s - cy) * np.cos(ang)
        if rng.random() < 0.58:
            inside = ((dx_r / max(rx, 1.0)) ** 2 + (dy_r / max(ry, 1.0)) ** 2) <= 1.0
            sample_c = canvas[min(S - 1, max(0, int(cy))), min(S - 1, max(0, int(cx)))]
            edge_dist = float(np.linalg.norm(col_b - sample_c) / 255.0)
            canvas[inside] = col_b
            # Continuous SDF rim at 2x scale (zero integer dilate-erode staircase!)
            in_2x = cv2.resize(inside.astype(np.float32), (Sb, Sb), interpolation=cv2.INTER_AREA) >= 0.5
            bound_2x[in_2x] = 0.0
            if edge_dist >= 0.08:
                d_in = distance_transform_edt(in_2x).astype(np.float32) / ss_b
                d_out = distance_transform_edt(~in_2x).astype(np.float32) / ss_b
                d_sdf = np.where(in_2x, np.maximum(0.0, d_in - 0.25), np.maximum(0.0, d_out - 0.25))
                rim_s = np.exp(-0.5 * (d_sdf / 0.68) ** 2).astype(np.float32)
                bound_2x = np.maximum(bound_2x, np.where(d_sdf <= 1.6, rim_s, 0.0))
        else:
            g = np.exp(-0.5 * ((dx_r / max(rx, 1.0)) ** 2 + (dy_r / max(ry, 1.0)) ** 2))[..., None]
            canvas = canvas * (1.0 - 0.55 * g) + col_b * (0.55 * g)

    if rng.random() < 0.28:
        grain = rng.normal(0.0, float(rng.uniform(2.5, 7.5)), size=(S // 4, S // 4, 1)).astype(np.float32)
        grain = cv2.resize(grain, (S, S), interpolation=cv2.INTER_LINEAR)[..., None]
        canvas = np.clip(canvas + grain, 0.0, 255.0)

    canvas_u8 = np.rint(np.clip(canvas, 0, 255)).astype(np.uint8)

    # 2. Vector geometry collection (covers hairlines 0.45px up to bold/Y strokes 4.5px!)
    all_x, all_y, all_tx, all_ty, all_r, all_cid = [], [], [], [], [], []
    end_pts: list[tuple[float, float]] = []
    stroke_pal = [
        (10, 10, 16), (28, 26, 36), (58, 52, 72), (118, 72, 58), (154, 102, 86),
        (78, 86, 126), (176, 128, 118), (242, 236, 218), (250, 250, 254), (255, 224, 188),
    ]

    def _emit_segment(pa: np.ndarray, pb: np.ndarray, r_a: float, r_b: float, col_seg: tuple[int, int, int], cid_val: int, add_ends: bool = False):
        seg_len = float(np.linalg.norm(pb - pa))
        if seg_len < 1.5:
            return cid_val
        n_ev = max(8, int(np.ceil(seg_len * 2.5)))
        ts_e = np.linspace(0.0, 1.0, n_ev, dtype=np.float32)[:, None]
        pts_s = pa[None, :] * (1.0 - ts_e) + pb[None, :] * ts_e
        r_s = (r_a * (1.0 - ts_e[:, 0]) + r_b * ts_e[:, 0]).astype(np.float32)
        tv = (pb - pa) / max(seg_len, 1e-5)
        if abs(r_a - r_b) < 0.15:
            cv2.line(
                canvas_u8,
                (int(round(pa[0] * ss)), int(round(pa[1] * ss))),
                (int(round(pb[0] * ss)), int(round(pb[1] * ss))),
                col_seg,
                max(1, int(round((r_a + r_b) * ss))),
                lineType=cv2.LINE_AA,
            )
        else:
            for k in range(0, n_ev - 1, 2):
                k2 = min(n_ev - 1, k + 2)
                cv2.line(
                    canvas_u8,
                    (int(round(pts_s[k, 0] * ss)), int(round(pts_s[k, 1] * ss))),
                    (int(round(pts_s[k2, 0] * ss)), int(round(pts_s[k2, 1] * ss))),
                    col_seg,
                    max(1, int(round((r_s[k] + r_s[k2]) * ss))),
                    lineType=cv2.LINE_AA,
                )
        in_s = (pts_s[:, 0] >= 0) & (pts_s[:, 0] < size) & (pts_s[:, 1] >= 0) & (pts_s[:, 1] < size)
        if in_s.any():
            m_s = int(in_s.sum())
            all_x.append(pts_s[in_s, 0])
            all_y.append(pts_s[in_s, 1])
            all_tx.append(np.full(m_s, float(tv[0]), dtype=np.float32))
            all_ty.append(np.full(m_s, float(tv[1]), dtype=np.float32))
            all_r.append(r_s[in_s])
            all_cid.append(np.full(m_s, cid_val, dtype=np.int32))
            if add_ends:
                for ep in (pa, pb):
                    if 0 <= ep[0] < size and 0 <= ep[1] < size:
                        end_pts.append((float(ep[0]), float(ep[1])))
            return cid_val + 1
        return cid_val

    cid = 1
    n_groups = int(rng.integers(5, 10))
    for _ in range(n_groups):
        mode = int(rng.integers(0, 7))
        n_strands = int(rng.integers(3, 8)) if mode in (0, 1) else int(rng.integers(1, 4))
        base_x = float(rng.uniform(20, size - 20))
        base_y = float(rng.uniform(20, size - 20))
        base_ang = float(rng.uniform(0, 2 * np.pi))
        length = float(rng.uniform(35, 200))
        bend = float(rng.uniform(-55, 55))
        s_col = stroke_pal[int(rng.integers(0, len(stroke_pal)))]
        bg_lum = float(canvas_u8[min(S - 1, int(base_y * ss)), min(S - 1, int(base_x * ss))].mean())
        if bg_lum < 85:
            if rng.random() < 0.78:
                s_col = stroke_pal[int(rng.integers(7, len(stroke_pal)))]
        elif s_col[0] > 200 and bg_lum > 140:
            s_col = stroke_pal[int(rng.integers(0, 6))]

        if mode == 4:
            # Explicit Stained-Glass Rosette with filled glass petals + varying-width BLACK Y-lead strokes!
            cx_g = float(rng.uniform(0.22 * size, 0.78 * size))
            cy_g = float(rng.uniform(0.22 * size, 0.78 * size))
            rad_outer = float(rng.uniform(44.0, 108.0))
            n_petals = int(rng.integers(8, 17))
            lead_col = (10, 10, 14)
            petal_cols = [(110, 150, 240), (85, 118, 225), (242, 228, 244), (248, 198, 222), (165, 210, 250)]
            r_stem = float(rng.uniform(1.1, 2.2))
            r_yhub = float(rng.uniform(2.4, 4.4))
            r_arm = float(rng.uniform(0.9, 1.8))
            # Draw filled colored petals first, then continuous varying-width Y leadlines on top!
            for pk in range(n_petals):
                a0 = base_ang + 2.0 * np.pi * pk / n_petals
                a_next = base_ang + 2.0 * np.pi * (pk + 1) / n_petals
                a_mid = 0.5 * (a0 + a_next)
                p_col = petal_cols[int(rng.integers(0, len(petal_cols)))]
                p0_v = np.array([cx_g + 0.14 * rad_outer * np.cos(a0), cy_g + 0.14 * rad_outer * np.sin(a0)], np.float32)
                p1_v = np.array([cx_g + 0.52 * rad_outer * np.cos(a0), cy_g + 0.52 * rad_outer * np.sin(a0)], np.float32)
                p2_v = np.array([cx_g + 0.92 * rad_outer * np.cos(a_mid), cy_g + 0.92 * rad_outer * np.sin(a_mid)], np.float32)
                p3_v = np.array([cx_g + 0.52 * rad_outer * np.cos(a_next), cy_g + 0.52 * rad_outer * np.sin(a_next)], np.float32)
                poly_s = np.rint(np.stack([p0_v, p1_v, p2_v, p3_v], axis=0) * ss).astype(np.int32)
                cv2.fillPoly(canvas_u8, [poly_s], p_col, lineType=cv2.LINE_AA)
            for pk in range(n_petals):
                a0 = base_ang + 2.0 * np.pi * pk / n_petals
                a_mid_l = a0 - 0.5 * np.pi / n_petals
                a_mid_r = a0 + 0.5 * np.pi / n_petals
                p_c = np.array([cx_g + 0.12 * rad_outer * np.cos(a0), cy_g + 0.12 * rad_outer * np.sin(a0)], np.float32)
                p_y = np.array([cx_g + 0.52 * rad_outer * np.cos(a0), cy_g + 0.52 * rad_outer * np.sin(a0)], np.float32)
                p_l = np.array([cx_g + 0.92 * rad_outer * np.cos(a_mid_l), cy_g + 0.92 * rad_outer * np.sin(a_mid_l)], np.float32)
                p_r = np.array([cx_g + 0.92 * rad_outer * np.cos(a_mid_r), cy_g + 0.92 * rad_outer * np.sin(a_mid_r)], np.float32)
                cid = _emit_segment(p_c, p_y, r_stem, r_yhub, lead_col, cid)
                cid = _emit_segment(p_y, p_l, r_yhub, r_arm, lead_col, cid)
                cid = _emit_segment(p_y, p_r, r_yhub, r_arm, lead_col, cid)
            continue

        if mode == 5:
            # Prism-style crisp parallel double-bars & straight wide crystal struts!
            n_bars = int(rng.integers(3, 7))
            r_bar = float(rng.uniform(0.95, 3.60))
            if rng.random() < 0.55:
                # Parallel double-bars framing bright ribbon (must NOT weld across bright gap!)
                gap_w = float(rng.uniform(2.8, 6.5)) + r_bar
                ca, sa = np.cos(base_ang), np.sin(base_ang)
                nx_b, ny_b = -sa, ca
                bar_col = (12, 12, 18) if bg_lum > 70 else (246, 232, 236)
                for off_s in (-gap_w * 0.5, gap_w * 0.5):
                    pa_b = np.array([base_x + nx_b * off_s - ca * length * 0.45, base_y + ny_b * off_s - sa * length * 0.45], np.float32)
                    pb_b = np.array([base_x + nx_b * off_s + ca * length * 0.45, base_y + ny_b * off_s + sa * length * 0.45], np.float32)
                    cid = _emit_segment(pa_b, pb_b, min(r_bar, 1.85), min(r_bar, 1.85), bar_col, cid, add_ends=True)
            else:
                for _ in range(n_bars):
                    ang_l = base_ang + float(rng.uniform(-1.3, 1.3))
                    cx_l = base_x + float(rng.uniform(-65, 65))
                    cy_l = base_y + float(rng.uniform(-65, 65))
                    hl = float(rng.uniform(30, 110))
                    pa_l = np.array([cx_l - hl * np.cos(ang_l), cy_l - hl * np.sin(ang_l)], np.float32)
                    pb_l = np.array([cx_l + hl * np.cos(ang_l), cy_l + hl * np.sin(ang_l)], np.float32)
                    cid = _emit_segment(pa_l, pb_l, r_bar, r_bar, s_col, cid, add_ends=True)
            continue

        if mode == 6:
            # Explicit bold dark/light solid rings & thick curved ink bars (r = 1.6..4.4 px)
            rad_x = float(rng.uniform(12.0, 42.0))
            rad_y = rad_x * float(rng.uniform(0.55, 1.10))
            ang_l = float(rng.uniform(0, np.pi))
            ts = np.linspace(0, 2 * np.pi, 73, endpoint=True, dtype=np.float32)
            lx, ly = rad_x * np.cos(ts), rad_y * np.sin(ts)
            pts = np.stack([base_x + lx * np.cos(ang_l) - ly * np.sin(ang_l), base_y + lx * np.sin(ang_l) + ly * np.cos(ang_l)], axis=1)
            d_pts = np.gradient(pts, axis=0)
            t_norm = np.maximum(np.hypot(d_pts[:, 0], d_pts[:, 1]), 1e-5)[:, None]
            tang = d_pts / t_norm
            r0 = float(rng.uniform(1.4, 3.8))
            cv2.polylines(canvas_u8, [np.rint(pts * ss).astype(np.int32)], True, s_col, max(1, int(round(r0 * 2 * ss))), lineType=cv2.LINE_AA)
            in_b = (pts[:, 0] >= 0) & (pts[:, 0] < size) & (pts[:, 1] >= 0) & (pts[:, 1] < size)
            if in_b.any():
                all_x.append(pts[in_b, 0]); all_y.append(pts[in_b, 1])
                all_tx.append(tang[in_b, 0]); all_ty.append(tang[in_b, 1])
                all_r.append(np.full(int(in_b.sum()), r0, dtype=np.float32))
                all_cid.append(np.full(int(in_b.sum()), cid, dtype=np.int32))
                cid += 1
            continue

        for si in range(n_strands):
            if mode == 0:
                perp_off = (si - 0.5 * (n_strands - 1)) * float(rng.uniform(2.5, 8.5))
                ang_j = base_ang + float(rng.uniform(-0.22, 0.22))
                len_j = length * float(rng.uniform(0.75, 1.15))
                p0 = np.array([base_x - np.sin(ang_j) * perp_off, base_y + np.cos(ang_j) * perp_off], np.float32)
                ca, sa = np.cos(ang_j), np.sin(ang_j)
                p3 = p0 + np.array([ca * len_j, sa * len_j], np.float32)
                p1 = p0 + np.array([ca * len_j * 0.33 - sa * bend, sa * len_j * 0.33 + ca * bend], np.float32)
                p2 = p0 + np.array([ca * len_j * 0.67 - sa * bend * 0.4, sa * len_j * 0.67 + ca * bend * 0.4], np.float32)
                r0 = float(rng.uniform(0.50, 1.50))
                taper = True
            elif mode == 1:
                dx_g = (si - 0.5 * n_strands) * float(rng.uniform(5.0, 10.0))
                p0 = np.array([base_x + dx_g, base_y + float(rng.uniform(-4, 4))], np.float32)
                ln = float(rng.uniform(10.0, 26.0))
                p3 = p0 + np.array([np.cos(base_ang) * ln, np.sin(base_ang) * ln], np.float32)
                p1 = p0 + 0.33 * (p3 - p0) + np.array([0, float(rng.uniform(-3, 3))], np.float32)
                p2 = p0 + 0.67 * (p3 - p0) + np.array([0, float(rng.uniform(-3, 3))], np.float32)
                r0 = float(rng.uniform(0.45, 0.90))
                taper = True
            elif mode == 2:
                # Wide bold tapered ink calligraphy / eyelash sweep (r = 1.2..4.2 px!)
                p0 = np.array([base_x, base_y], np.float32)
                ca, sa = np.cos(base_ang), np.sin(base_ang)
                p3 = p0 + np.array([ca * length * 0.75, sa * length * 0.75], np.float32)
                p1 = p0 + np.array([ca * length * 0.25 - sa * bend * 1.1, sa * length * 0.25 + ca * bend * 1.1], np.float32)
                p2 = p0 + np.array([ca * length * 0.50 - sa * bend * 0.8, sa * length * 0.50 + ca * bend * 0.8], np.float32)
                r0 = float(rng.uniform(1.30, 4.20))
                taper = True
            else:
                rad_x = float(rng.uniform(8.0, 28.0))
                rad_y = rad_x * float(rng.uniform(0.55, 1.15))
                ang_l = float(rng.uniform(0, np.pi))
                ts = np.linspace(0, 2 * np.pi, 65, endpoint=True, dtype=np.float32)
                lx, ly = rad_x * np.cos(ts), rad_y * np.sin(ts)
                pts = np.stack([base_x + lx * np.cos(ang_l) - ly * np.sin(ang_l), base_y + lx * np.sin(ang_l) + ly * np.cos(ang_l)], axis=1)
                d_pts = np.gradient(pts, axis=0)
                t_norm = np.maximum(np.hypot(d_pts[:, 0], d_pts[:, 1]), 1e-5)[:, None]
                tang = d_pts / t_norm
                r0 = float(rng.uniform(0.65, 2.20))
                cv2.polylines(canvas_u8, [np.rint(pts * ss).astype(np.int32)], True, s_col, max(1, int(round(r0 * 2 * ss))), lineType=cv2.LINE_AA)
                in_b = (pts[:, 0] >= 0) & (pts[:, 0] < size) & (pts[:, 1] >= 0) & (pts[:, 1] < size)
                if in_b.any():
                    all_x.append(pts[in_b, 0]); all_y.append(pts[in_b, 1])
                    all_tx.append(tang[in_b, 0]); all_ty.append(tang[in_b, 1])
                    all_r.append(np.full(int(in_b.sum()), r0, dtype=np.float32))
                    all_cid.append(np.full(int(in_b.sum()), cid, dtype=np.int32))
                    cid += 1
                continue

            est_len = float(np.linalg.norm(p3 - p0) + abs(bend) + 10.0)
            n_eval = max(24, int(np.ceil(est_len * 2.5)))
            pts, tang, t_par = _eval_cubic_bezier(p0, p1, p2, p3, n_eval)
            if taper:
                r_arr = np.clip(r0 * (np.maximum(np.sin(np.pi * t_par), 0.0) ** 0.35 * 0.72 + 0.32), 0.40, 4.50).astype(np.float32)
            else:
                r_arr = np.full(n_eval, r0, dtype=np.float32)

            for k in range(0, n_eval - 1, 2):
                k2 = min(n_eval - 1, k + 2)
                q0 = (int(round(pts[k, 0] * ss)), int(round(pts[k, 1] * ss)))
                q1 = (int(round(pts[k2, 0] * ss)), int(round(pts[k2, 1] * ss)))
                thick_ss = max(1, int(round(0.5 * (r_arr[k] + r_arr[k2]) * 2.0 * ss)))
                cv2.line(canvas_u8, q0, q1, s_col, thick_ss, lineType=cv2.LINE_AA)

            in_b = (pts[:, 0] >= 0) & (pts[:, 0] < size) & (pts[:, 1] >= 0) & (pts[:, 1] < size)
            if in_b.any():
                all_x.append(pts[in_b, 0]); all_y.append(pts[in_b, 1])
                all_tx.append(tang[in_b, 0]); all_ty.append(tang[in_b, 1])
                all_r.append(r_arr[in_b])
                all_cid.append(np.full(int(in_b.sum()), cid, dtype=np.int32))
                for ep in (pts[0], pts[-1]):
                    if 0 <= ep[0] < size and 0 <= ep[1] < size:
                        end_pts.append((float(ep[0]), float(ep[1])))
                cid += 1

    rgb_out = cv2.resize(canvas_u8, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.30:
        q_jpg = int(rng.integers(85, 98))
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q_jpg])
        rgb_out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    tgt = np.zeros((9, size, size), dtype=np.float32)
    tgt[4] = 0.5
    tgt[5] = 0.5
    tgt[6] = 0.5
    tgt[7] = 0.5
    bound_1x = np.clip(cv2.resize(bound_2x, (size, size), interpolation=cv2.INTER_AREA), 0.0, 1.0)

    if len(all_x) == 0:
        tgt[1] = bound_1x
        return rgb_out, tgt, 1.0

    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)
    txs = np.concatenate(all_tx)
    tys = np.concatenate(all_ty)
    rs = np.concatenate(all_r)
    cids = np.concatenate(all_cid)

    cand = np.zeros((size, size), dtype=np.uint8)
    px_i = np.clip(np.round(xs).astype(np.int32), 0, size - 1)
    py_i = np.clip(np.round(ys).astype(np.int32), 0, size - 1)
    cand[py_i, px_i] = 1
    cand = cv2.dilate(cand, np.ones((13, 13), np.uint8)) > 0
    cy, cx = np.where(cand)

    tree = cKDTree(np.column_stack([ys, xs]))
    dists, idxs = tree.query(np.column_stack([cy.astype(np.float32), cx.astype(np.float32)]), k=min(8, len(xs)))
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    d0 = dists[:, 0]
    i0 = idxs[:, 0]
    nx_p, ny_p = xs[i0], ys[i0]
    ntx, nty = txs[i0], tys[i0]
    nr = rs[i0]
    ncid = cids[i0]

    # Single smooth centerline peak even for r = 4.5 px strokes (ZERO hollow cavity / 抽壳!)
    sigma_p = np.maximum(0.65, 0.46 * nr)
    s_prob = np.where(d0 <= np.maximum(2.4, nr + 1.1), np.exp(-0.5 * (d0 / sigma_p) ** 2), 0.0)
    tgt[0, cy, cx] = s_prob
    tgt[4, cy, cx] = np.clip((nx_p - cx) / OFFSET_SCALE + 0.5, 0.0, 1.0)
    tgt[5, cy, cx] = np.clip((ny_p - cy) / OFFSET_SCALE + 0.5, 0.0, 1.0)
    cos2t = ntx * ntx - nty * nty
    sin2t = 2.0 * ntx * nty
    tgt[6, cy, cx] = np.clip(0.5 * cos2t + 0.5, 0.0, 1.0)
    tgt[7, cy, cx] = np.clip(0.5 * sin2t + 0.5, 0.0, 1.0)
    tgt[8, cy, cx] = np.clip(nr / RAD_SCALE, 0.0, 1.0)

    if dists.shape[1] > 1:
        ang_dot = np.abs(ntx[:, None] * txs[idxs] + nty[:, None] * tys[idxs])
        diff_c = (cids[idxs] != ncid[:, None]) & (dists <= 2.6) & (ang_dot < 0.88)
        has_x = diff_c.any(axis=1)
        if has_x.any():
            first_x = np.argmax(diff_c[has_x], axis=1)
            d_other = dists[has_x, first_x]
            tgt[3, cy[has_x], cx[has_x]] = np.clip(np.exp(-0.5 * (d_other / 1.15) ** 2), 0.0, 1.0)

    if end_pts:
        e_map = np.zeros((size, size), np.float32)
        for ex, ey in end_pts:
            e_map[int(round(np.clip(ey, 0, size - 1))), int(round(np.clip(ex, 0, size - 1)))] = 1.0
        tgt[2] = np.clip(cv2.GaussianBlur(e_map, (7, 7), 1.0) * 6.28, 0.0, 1.0)

    tgt[1] = np.clip(bound_1x * np.clip(1.0 - tgt[0] * 1.35, 0.0, 1.0), 0.0, 1.0)
    return rgb_out, tgt, 1.0


def build_real_vector_targets_v4(sr_2x: np.ndarray) -> np.ndarray:
    """Build smooth sub-pixel, shell-free, haze-free 9-channel supervision targets on 2x SR real artworks."""
    h, w = sr_2x.shape[:2]
    sm = cv2.bilateralFilter(sr_2x, 5, 14, 5)
    lab = srgb_to_oklab(sm).astype(np.float32)
    L, a_c, b_c = lab[..., 0], lab[..., 1], lab[..., 2]
    L_raw = srgb_to_oklab(sr_2x)[..., 0].astype(np.float32)
    C3 = np.stack([L, a_c * 1.85, b_c * 1.85], axis=-1).astype(np.float32)

    # 1. Rotation-invariant Scharr structure tensor on smooth OKLab
    gx_c = cv2.Scharr(C3, cv2.CV_32F, 1, 0, scale=1.0 / 32.0)
    gy_c = cv2.Scharr(C3, cv2.CV_32F, 0, 1, scale=1.0 / 32.0)
    jxx = cv2.GaussianBlur((gx_c * gx_c).sum(axis=-1), (5, 5), 1.0)
    jyy = cv2.GaussianBlur((gy_c * gy_c).sum(axis=-1), (5, 5), 1.0)
    jxy = cv2.GaussianBlur((gx_c * gy_c).sum(axis=-1), (5, 5), 1.0)

    tr = jxx - jyy
    norm2 = np.maximum(np.hypot(tr, 2.0 * jxy), 1e-6)
    cos2n = tr / norm2
    sin2n = (2.0 * jxy) / norm2
    ang_n = 0.5 * np.arctan2(sin2n, cos2n)
    nx_u = np.cos(ang_n).astype(np.float32)
    ny_u = np.sin(ang_n).astype(np.float32)
    cos2t = -cos2n
    sin2t = -sin2n
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 2. Multi-scale Black-Hat stroke body (1px hairlines AND 3-11px wide strokes/Y-wedges)
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark_void_core = cv2.dilate(L, k15) < 0.24
    inside_dark_void = (L < 0.25) & (cv2.dilate(dark_void_core.astype(np.uint8), k9) > 0)

    bh_max = np.zeros((h, w), dtype=np.float32)
    for diam in (5, 9, 13):
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
        closed_L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, ke)
        bh_r = np.maximum(0.0, closed_L - L)
        if diam >= 9:
            bh_r = np.where(inside_dark_void, 0.0, bh_r)
        bh_max = np.maximum(bh_max, bh_r)

    Lm1 = _bilinear_sample(L, yy - 1.15 * ny_u, xx - 1.15 * nx_u)
    Lp1 = _bilinear_sample(L, yy + 1.15 * ny_u, xx + 1.15 * nx_u)
    Lm2 = _bilinear_sample(L, yy - 2.2 * ny_u, xx - 2.2 * nx_u)
    Lp2 = _bilinear_sample(L, yy + 2.2 * ny_u, xx + 2.2 * nx_u)
    d_min1 = np.minimum(Lm1, Lp1) - L
    d_min2 = np.minimum(Lm2, Lp2) - L
    is_thin_valley = (Lm1 >= L - 0.004) & (Lp1 >= L - 0.004)
    thin_val_str = np.where(
        is_thin_valley,
        np.clip((np.maximum(d_min1, d_min2 * 0.85) - 0.012) / 0.075, 0.0, 1.0),
        0.0,
    )
    denom_p = Lm1 - 2.0 * L + Lp1
    delta_thin = np.where(
        np.abs(denom_p) > 0.010,
        0.5 * 1.15 * (Lm1 - Lp1) / np.maximum(np.abs(denom_p), 0.010),
        0.0,
    )
    delta_thin = np.clip(delta_thin, -1.25, 1.25)

    dark_stroke_u8 = ((bh_max >= 0.085) & (~inside_dark_void)).astype(np.uint8)
    dark_union_u8 = np.maximum(dark_stroke_u8, (thin_val_str >= 0.30).astype(np.uint8))

    dark_prob = np.zeros((h, w), dtype=np.float32)
    dx_dark = delta_thin * nx_u
    dy_dark = delta_thin * ny_u
    local_r = np.where(thin_val_str >= 0.20, 0.85, 0.0).astype(np.float32)

    if dark_union_u8.any():
        du_2x = cv2.resize(dark_union_u8.astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_LINEAR) >= 0.48
        dt_2x = (cv2.distanceTransform(du_2x.astype(np.uint8), cv2.DIST_L2, 5) / 2.0).astype(np.float32)
        sk_2x = skeletonize(du_2x)
        del du_2x
        ys_sk, xs_sk = np.where(sk_2x)
        del sk_2x
        if len(ys_sk) > 0:
            sk_pts_sr = np.column_stack([ys_sk.astype(np.float32) * 0.5, xs_sk.astype(np.float32) * 0.5])
            sk_radii = dt_2x[ys_sk, xs_sk]
            del dt_2x, ys_sk, xs_sk
            cand_m = cv2.dilate(dark_union_u8, np.ones((7, 7), np.uint8)) > 0
            cy_m, cx_m = np.where(cand_m)
            del cand_m
            if len(cy_m) > 0:
                tree_sk = cKDTree(sk_pts_sr)
                d_m, idx_m = tree_sk.query(np.column_stack([cy_m.astype(np.float32), cx_m.astype(np.float32)]), workers=1)
                d_skel = np.full((h, w), 99.0, dtype=np.float32)
                off_y = np.zeros((h, w), dtype=np.float32)
                off_x = np.zeros((h, w), dtype=np.float32)
                r_at_sk = np.zeros((h, w), dtype=np.float32)
                d_skel[cy_m, cx_m] = d_m.astype(np.float32)
                off_y[cy_m, cx_m] = sk_pts_sr[idx_m, 0] - cy_m.astype(np.float32)
                off_x[cy_m, cx_m] = sk_pts_sr[idx_m, 1] - cx_m.astype(np.float32)
                r_at_sk[cy_m, cx_m] = sk_radii[idx_m]

                r_sm = cv2.GaussianBlur(np.clip(r_at_sk, 0.55, 5.8), (3, 3), 0.7)
                local_r = np.where(dark_union_u8 > 0, r_sm, local_r)

                sigma_r = np.maximum(0.68, 0.46 * local_r)
                in_band = (dark_union_u8 > 0) | (d_skel <= local_r + 1.0)
                body_prob = np.where(in_band & (bh_max >= 0.05), np.exp(-0.5 * (d_skel / sigma_r) ** 2), 0.0)
                body_prob *= np.clip(bh_max / 0.085, 0.65, 1.0)

                thin_prob = np.where(is_thin_valley, thin_val_str * np.exp(-0.5 * (delta_thin / 0.72) ** 2), 0.0)
                dark_prob = np.maximum(body_prob, thin_prob)

                use_sk = (local_r >= 1.20) & (dark_prob >= 0.10)
                dx_dark = np.where(use_sk, off_x, delta_thin * nx_u)
                dy_dark = np.where(use_sk, off_y, delta_thin * ny_u)
                del d_skel, off_y, off_x, r_at_sk, body_prob, thin_prob
        else:
            del dt_2x, ys_sk, xs_sk

    # 3. Bright wisps on dark ground + colored thin ridges along continuous normal
    Lrm1 = _bilinear_sample(L_raw, yy - 1.15 * ny_u, xx - 1.15 * nx_u)
    Lrp1 = _bilinear_sample(L_raw, yy + 1.15 * ny_u, xx + 1.15 * nx_u)
    Lrm2 = _bilinear_sample(L_raw, yy - 2.3 * ny_u, xx - 2.3 * nx_u)
    Lrp2 = _bilinear_sample(L_raw, yy + 2.3 * ny_u, xx + 2.3 * nx_u)
    flank_hi = np.maximum(Lrm1, Lrp1)
    flank_hi2 = np.maximum(Lrm2, Lrp2)
    d_br = np.maximum(L_raw - flank_hi, (L_raw - flank_hi2) * 0.85)
    is_br = (L_raw >= flank_hi - 0.005) & (flank_hi2 < 0.44) & (d_br > 0.050)
    br_str = np.where(is_br, np.clip((d_br - 0.040) / 0.13, 0.0, 1.0), 0.0)
    br_cc = (br_str > 0.20).astype(np.uint8)
    n_b, lb, stb, _ = cv2.connectedComponentsWithStats(br_cc, connectivity=8)
    for bi in range(1, n_b):
        if int(stb[bi, cv2.CC_STAT_AREA]) <= 4:
            br_str[lb == bi] = 0.0
    denom_b = Lrm1 - 2.0 * L_raw + Lrp1
    delta_br = np.where(np.abs(denom_b) > 0.012, 0.5 * 1.15 * (Lrm1 - Lrp1) / np.maximum(np.abs(denom_b), 0.012), 0.0)
    delta_br = np.clip(delta_br, -1.2, 1.2)

    Cm1 = _bilinear_sample(C3, yy - 1.15 * ny_u, xx - 1.15 * nx_u)
    Cp1 = _bilinear_sample(C3, yy + 1.15 * ny_u, xx + 1.15 * nx_u)
    Cm2 = _bilinear_sample(C3, yy - 2.3 * ny_u, xx - 2.3 * nx_u)
    Cp2 = _bilinear_sample(C3, yy + 2.3 * ny_u, xx + 2.3 * nx_u)
    de1 = np.minimum(np.linalg.norm(Cm1 - C3, axis=-1), np.linalg.norm(Cp1 - C3, axis=-1))
    de2 = np.minimum(np.linalg.norm(Cm2 - C3, axis=-1), np.linalg.norm(Cp2 - C3, axis=-1))
    de_flank = np.linalg.norm(Cm2 - Cp2, axis=-1)
    col_ridge = np.where((de1 >= 0.022) & (de2 >= 0.032) & (de_flank <= 1.25 * de2), np.clip((de2 - 0.025) / 0.075, 0.0, 1.0), 0.0)

    thin_aux = np.maximum(br_str * 0.94, col_ridge * 0.88)
    stroke_prob = np.maximum(dark_prob, thin_aux)

    # 4. Crisp sub-pixel color-block seam (`color_bound`): smooth dots/petals intact, zero gradient haze!
    g_edge = np.linalg.norm(Cp1 - Cm1, axis=-1)
    Cm26 = _bilinear_sample(C3, yy - 2.6 * ny_u, xx - 2.6 * nx_u)
    Cp26 = _bilinear_sample(C3, yy + 2.6 * ny_u, xx + 2.6 * nx_u)
    Cm50 = _bilinear_sample(C3, yy - 5.0 * ny_u, xx - 5.0 * nx_u)
    Cp50 = _bilinear_sample(C3, yy + 5.0 * ny_u, xx + 5.0 * nx_u)
    wide_span = np.linalg.norm(Cp50 - Cm50, axis=-1)
    flank_drift = np.maximum(np.linalg.norm(Cm26 - Cm1, axis=-1), np.linalg.norm(Cp26 - Cp1, axis=-1))
    not_grad_ramp = (flank_drift <= 0.55 * g_edge) | (wide_span <= 1.35 * g_edge)

    ge_m1 = _bilinear_sample(g_edge, yy - ny_u, xx - nx_u)
    ge_p1 = _bilinear_sample(g_edge, yy + ny_u, xx + nx_u)
    denom_e = ge_m1 - 2.0 * g_edge + ge_p1
    delta_e = np.where(denom_e < -0.006, 0.5 * (ge_m1 - ge_p1) / np.minimum(denom_e, -0.006), 0.0)
    is_sharp_step = (g_edge >= 0.030) & not_grad_ramp & (g_edge >= ge_m1 - 0.006) & (g_edge >= ge_p1 - 0.006)
    cbound = np.where(is_sharp_step, np.clip((g_edge - 0.022) / 0.065, 0.0, 1.0) * np.exp(-0.5 * (delta_e / 0.85) ** 2), 0.0)
    # Only suppress directly on dark stroke centerlines (3x3), leaving adjacent petal/bead rims strong!
    stroke_supp = cv2.dilate((stroke_prob > 0.42).astype(np.float32), np.ones((3, 3), np.uint8))
    cbound = np.clip(cbound * (1.0 - np.clip(stroke_supp * 1.15, 0.0, 1.0)), 0.0, 1.0)

    use_br = (br_str * 0.94 >= dark_prob) & (br_str * 0.94 >= col_ridge * 0.88)
    dx = np.where(dark_prob >= thin_aux, dx_dark, np.where(use_br, delta_br * nx_u, delta_thin * nx_u))
    dy = np.where(dark_prob >= thin_aux, dy_dark, np.where(use_br, delta_br * ny_u, delta_thin * ny_u))
    dx = np.where(stroke_prob >= 0.08, np.clip(dx, -4.0, 4.0), 0.0)
    dy = np.where(stroke_prob >= 0.08, np.clip(dy, -4.0, 4.0), 0.0)

    tgt = np.zeros((9, h, w), dtype=np.float32)
    tgt[0] = np.clip(stroke_prob, 0.0, 1.0)
    tgt[1] = cbound
    tgt[4] = np.clip(dx / OFFSET_SCALE + 0.5, 0.0, 1.0)
    tgt[5] = np.clip(dy / OFFSET_SCALE + 0.5, 0.0, 1.0)
    tgt[6] = np.clip(0.5 * cos2t + 0.5, 0.0, 1.0)
    tgt[7] = np.clip(0.5 * sin2t + 0.5, 0.0, 1.0)

    r_out = np.where(dark_prob > 0.15, local_r, np.where(stroke_prob > 0.15, 0.85, 0.0))
    tgt[8] = np.clip(r_out / RAD_SCALE, 0.0, 1.0)

    mask_s = tgt[0] >= 0.44
    if mask_s.any():
        sk = skeletonize(mask_s).astype(np.uint8)
        nbr = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - sk
        ep = ((sk > 0) & (nbr == 1)).astype(np.float32)
        jn = ((sk > 0) & (nbr >= 3)).astype(np.float32)
        tgt[2] = np.clip(cv2.GaussianBlur(ep, (7, 7), 1.0) * 6.28, 0.0, 1.0)
        tgt[3] = np.clip(cv2.GaussianBlur(jn, (7, 7), 1.0) * 6.28, 0.0, 1.0)
    return tgt


class LineVectorDataset(Dataset):
    def __init__(self, items, train: bool = True):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y_u8, geom_w = self.items[idx]
        if self.train:
            k = int(np.random.randint(0, 4))
            if k:
                x = np.rot90(x, k).copy()
                y_u8 = np.rot90(y_u8, k, axes=(1, 2)).copy()
                if k in (1, 3):
                    ox = y_u8[4].astype(np.float32) - 127.5
                    oy = y_u8[5].astype(np.float32) - 127.5
                    c2 = y_u8[6].astype(np.float32) - 127.5
                    s2 = y_u8[7].astype(np.float32) - 127.5
                    if k == 1:
                        y_u8[4] = np.clip(127.5 + oy, 0, 255).astype(np.uint8)
                        y_u8[5] = np.clip(127.5 - ox, 0, 255).astype(np.uint8)
                    else:
                        y_u8[4] = np.clip(127.5 - oy, 0, 255).astype(np.uint8)
                        y_u8[5] = np.clip(127.5 + ox, 0, 255).astype(np.uint8)
                    y_u8[6] = np.clip(127.5 - c2, 0, 255).astype(np.uint8)
                    y_u8[7] = np.clip(127.5 - s2, 0, 255).astype(np.uint8)
                else:
                    y_u8[4] = 255 - y_u8[4]
                    y_u8[5] = 255 - y_u8[5]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
                y_u8 = y_u8[:, :, ::-1].copy()
                y_u8[4] = 255 - y_u8[4]
                y_u8[7] = 255 - y_u8[7]
            if np.random.rand() < 0.5:
                x = x[::-1, :]
                y_u8 = y_u8[:, ::-1, :].copy()
                y_u8[5] = 255 - y_u8[5]
                y_u8[7] = 255 - y_u8[7]
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
            if np.random.rand() < 0.35:
                xf = np.power(np.clip(xf, 1e-4, 1.0), float(np.random.uniform(0.90, 1.12)))
            xf = np.clip(xf * float(np.random.uniform(0.93, 1.07)) + float(np.random.uniform(-0.025, 0.025)), 0.0, 1.0)
        else:
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
        yf = np.ascontiguousarray(y_u8).astype(np.float32) / 255.0
        return (
            torch.from_numpy(xf).permute(2, 0, 1).contiguous(),
            torch.from_numpy(yf).contiguous(),
            torch.tensor(geom_w, dtype=torch.float32),
        )


def _vector_loss(pred: torch.Tensor, target: torch.Tensor, geom_w: torch.Tensor):
    p_cls = pred[:, :4]
    t_cls = target[:, :4]
    p_c = p_cls.clamp(1e-4, 1.0 - 1e-4)
    l1_c = torch.abs(p_cls - t_cls)
    bce_c = -(t_cls * torch.log(p_c) + (1.0 - t_cls) * torch.log(1.0 - p_c))
    # Strongly penalize background false-positives (t < 0.02) so gradients never bloom into haze!
    pos_w = torch.where(t_cls > 0.02, 1.0 + 2.4 * t_cls, 1.35)
    inter = (p_cls * t_cls).sum(dim=(2, 3))
    denom = (p_cls + t_cls).sum(dim=(2, 3)) + 1e-4
    dice = 1.0 - (2.0 * inter / denom)
    loss_cls = 0.40 * (l1_c * pos_w).mean() + 0.35 * (bce_c * pos_w).mean() + 0.25 * dice.mean()

    t_stroke = target[:, 0:1]
    t_junc = target[:, 3:4]
    active = (t_stroke > 0.15).float()
    gw = geom_w[:, None, None, None]
    m_off = active * gw * (0.4 + 0.6 * t_stroke)
    m_ang = active * gw * (1.0 - 0.75 * t_junc)

    p_off = pred[:, 4:6]
    t_off = target[:, 4:6]
    loss_off = (torch.nn.functional.smooth_l1_loss(p_off, t_off, beta=0.04, reduction="none") * m_off).sum() / torch.clamp(m_off.sum() * 2.0, min=1.0)

    p_ang = pred[:, 6:8]
    t_ang = target[:, 6:8]
    p_v = (p_ang - 0.5) * 2.0
    t_v = (t_ang - 0.5) * 2.0
    cos_err = 1.0 - (p_v[:, 0:1] * t_v[:, 0:1] + p_v[:, 1:2] * t_v[:, 1:2])
    loss_ang = (
        (torch.abs(p_ang - t_ang) * m_ang).sum() / torch.clamp(m_ang.sum() * 2.0, min=1.0)
        + 0.35 * (torch.clamp(cos_err, min=0.0) * m_ang).sum() / torch.clamp(m_ang.sum(), min=1.0)
    )

    p_rad = pred[:, 8:9]
    t_rad = target[:, 8:9]
    loss_rad = (torch.abs(p_rad - t_rad) * m_off).sum() / torch.clamp(m_off.sum(), min=1.0)

    total = loss_cls + 1.8 * loss_off + 1.1 * loss_ang + 1.4 * loss_rad
    return total, loss_cls, loss_off, loss_ang, loss_rad


def _synth_worker(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    seed_i, tile_sz = args
    rng_w = np.random.default_rng(seed_i)
    rgb, tgt, gw = generate_synthetic_vector_tile(rng_w, size=tile_sz)
    y_u8 = np.rint(np.clip(tgt, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb, y_u8, float(gw)


def _real_crop_worker(args) -> tuple[bool, list[tuple[np.ndarray, np.ndarray, float]]]:
    p2_str, is_val, gw_g, gw_c, n_c_tr, seed_i, tile_sz = args
    sr_2x = np.asarray(Image.open(p2_str).convert("RGB"))
    rng_l = np.random.default_rng(seed_i)
    tgt_2x = build_real_vector_targets_v4(sr_2x)
    y_u8_2x = np.rint(np.clip(tgt_2x, 0.0, 1.0) * 255.0).astype(np.uint8)
    h2, w2 = sr_2x.shape[:2]
    out_items = []
    r_g = cv2.resize(sr_2x, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)
    y_g = np.stack([cv2.resize(y_u8_2x[c], (tile_sz, tile_sz), interpolation=cv2.INTER_AREA) for c in range(9)], axis=0)
    out_items.append((r_g, y_g, gw_g))
    if min(h2, w2) >= tile_sz:
        ys, xs = np.where(tgt_2x[0] > 0.45)
        n_c = 1 if is_val else n_c_tr
        half = tile_sz // 2
        for _ in range(n_c):
            if len(ys):
                pk = int(rng_l.integers(0, len(ys)))
                y0 = int(np.clip(int(ys[pk]) - half + int(rng_l.integers(-48, 49)), 0, h2 - tile_sz))
                x0 = int(np.clip(int(xs[pk]) - half + int(rng_l.integers(-48, 49)), 0, w2 - tile_sz))
            else:
                y0 = int(rng_l.integers(0, h2 - tile_sz + 1))
                x0 = int(rng_l.integers(0, w2 - tile_sz + 1))
            out_items.append((
                sr_2x[y0:y0 + tile_sz, x0:x0 + tile_sz].copy(),
                y_u8_2x[:, y0:y0 + tile_sz, x0:x0 + tile_sz].copy(),
                gw_c,
            ))
    return is_val, out_items


def _load_or_build_v4_dataset(tile_sz: int = 384):
    req = ("tr_x.npy", "tr_y.npy", "tr_w.npy", "va_x.npy", "va_y.npy", "va_w.npy")
    if V4_TILES_DIR.is_dir() and all((V4_TILES_DIR / f).is_file() for f in req):
        print(f"Loading cached v4 dataset from {V4_TILES_DIR}...", flush=True)
        tr_x = np.load(V4_TILES_DIR / "tr_x.npy")
        tr_y = np.load(V4_TILES_DIR / "tr_y.npy")
        tr_w = np.load(V4_TILES_DIR / "tr_w.npy")
        va_x = np.load(V4_TILES_DIR / "va_x.npy")
        va_y = np.load(V4_TILES_DIR / "va_y.npy")
        va_w = np.load(V4_TILES_DIR / "va_w.npy")
        return (
            [(tr_x[i], tr_y[i], float(tr_w[i])) for i in range(len(tr_x))],
            [(va_x[i], va_y[i], float(va_w[i])) for i in range(len(va_x))],
        )

    V4_TILES_DIR.mkdir(parents=True, exist_ok=True)
    syn_cache = V4_TILES_DIR / "syn_cache.npz"
    if syn_cache.is_file():
        print(f"Loading cached synthetic tiles from {syn_cache}...", flush=True)
        sc_d = np.load(syn_cache)
        tr_items = [(sc_d["tr_x"][i], sc_d["tr_y"][i], float(sc_d["tr_w"][i])) for i in range(len(sc_d["tr_x"]))]
        va_items = [(sc_d["va_x"][i], sc_d["va_y"][i], float(sc_d["va_w"][i])) for i in range(len(sc_d["va_x"]))]
    else:
        n_syn_tr, n_syn_va = 1500, 180
        t0 = time.time()
        print(f"Generating {n_syn_tr} train + {n_syn_va} val 4x-supersampled synthetic {tile_sz}x{tile_sz} tiles...", flush=True)
        jobs = [(30000 + i, tile_sz) for i in range(n_syn_tr + n_syn_va)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
            syn_items = list(ex.map(_synth_worker, jobs, chunksize=16))
        va_items = list(syn_items[:n_syn_va])
        tr_items = list(syn_items[n_syn_va:])
        np.savez(
            syn_cache,
            tr_x=np.stack([it[0] for it in tr_items], axis=0),
            tr_y=np.stack([it[1] for it in tr_items], axis=0),
            tr_w=np.array([it[2] for it in tr_items], dtype=np.float32),
            va_x=np.stack([it[0] for it in va_items], axis=0),
            va_y=np.stack([it[1] for it in va_items], axis=0),
            va_w=np.array([it[2] for it in va_items], dtype=np.float32),
        )
        print(f"Synthetic SVG tiles ready & cached in {time.time() - t0:.1f}s", flush=True)

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    holdout_ids = {2, 4}
    val_ids = {7, 14, 19, 27, 33, 41, 46}
    v2_mpath = CACHE_DIR / "v2_gallery" / "v2_manifest.json"
    v2_manifest = json.loads(v2_mpath.read_text(encoding="utf-8")) if v2_mpath.is_file() else []
    e_mpath = CACHE_DIR / "e_gallery" / "e_manifest.json"
    e_manifest = json.loads(e_mpath.read_text(encoding="utf-8")) if e_mpath.is_file() else []

    sr_cache_dir = CACHE_DIR / "v2_gallery" / "sr_2x"
    sr_cache_dir.mkdir(parents=True, exist_ok=True)
    e_sr_dir = CACHE_DIR / "e_gallery" / "sr_2x"
    e_sr_dir.mkdir(parents=True, exist_ok=True)

    work_specs = []
    need_sr = []
    for item in manifest:
        if item["id"] in holdout_ids:
            continue
        out_sr = sr_cache_dir / f"v1_{item['id']:03d}.png"
        if not out_sr.is_file():
            need_sr.append((Path(item["rgb_path"]), out_sr))
        work_specs.append((str(out_sr), item["id"] in val_ids, 0.55, 0.72, 2))

    for vi, item in enumerate(v2_manifest):
        is_val = (vi % 10 == 0)
        if item.get("kind") == "svg_2x":
            work_specs.append((str(item["rgb_path"]), is_val, 0.72, 0.85, 4))
        else:
            out_sr = sr_cache_dir / f"v2_{vi:03d}.png"
            if not out_sr.is_file():
                need_sr.append((Path(item["rgb_path"]), out_sr))
            work_specs.append((str(out_sr), is_val, 0.60, 0.74, 3))

    for ei, item in enumerate(e_manifest):
        is_val = (ei % 12 == 0)
        n2x = item.get("native_2x_path")
        if n2x and Path(n2x).is_file():
            work_specs.append((str(n2x), is_val, 0.65, 0.78, 3))
        else:
            out_sr = e_sr_dir / f"e3_{ei:03d}.png"
            if not out_sr.is_file():
                need_sr.append((Path(item["rgb_path"]), out_sr))
            work_specs.append((str(out_sr), is_val, 0.58, 0.72, 3))

    if need_sr:
        print(f"Upscaling {len(need_sr)} remaining raster images...", flush=True)
        sess_sr = _get_session(_WEIGHTS_DEFAULT / "realesr_anime.onnx")
        for src_p, dst_p in need_sr:
            rgb = np.asarray(Image.open(src_p).convert("RGB"))
            sc_sr = 2.0 if max(rgb.shape[:2]) <= 1200 else 1.4
            sr_2x = _upscale_sr(rgb, sc_sr, sess_sr)
            Image.fromarray(sr_2x).save(dst_p, compress_level=1)
        del sess_sr
    _SESSION_CACHE.clear()
    gc.collect()

    t_real = time.time()
    print(f"Extracting shell-free continuous v4 targets across {len(work_specs)} 2x artworks (4 low-RAM workers)...", flush=True)
    r_jobs = [(p2, iv, gwg, gwc, nc, 60000 + i, tile_sz) for i, (p2, iv, gwg, gwc, nc) in enumerate(work_specs)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for done_i, (iv, items) in enumerate(ex.map(_real_crop_worker, r_jobs, chunksize=4), 1):
            (va_items if iv else tr_items).extend(items)
            if done_i % 50 == 0 or done_i == len(r_jobs):
                print(f"  real {done_i}/{len(r_jobs)} ({time.time() - t_real:.1f}s)", flush=True)

    V4_TILES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(V4_TILES_DIR / "tr_x.npy", np.stack([it[0] for it in tr_items], axis=0))
    np.save(V4_TILES_DIR / "tr_y.npy", np.stack([it[1] for it in tr_items], axis=0))
    np.save(V4_TILES_DIR / "tr_w.npy", np.array([it[2] for it in tr_items], dtype=np.float32))
    np.save(V4_TILES_DIR / "va_x.npy", np.stack([it[0] for it in va_items], axis=0))
    np.save(V4_TILES_DIR / "va_y.npy", np.stack([it[1] for it in va_items], axis=0))
    np.save(V4_TILES_DIR / "va_w.npy", np.array([it[2] for it in va_items], dtype=np.float32))
    print(f"Cached v4 dataset to {V4_TILES_DIR}", flush=True)
    return tr_items, va_items


def main():
    tile_sz = 384
    tr_items, va_items = _load_or_build_v4_dataset(tile_sz=tile_sz)
    print(f"Dataset summary: train={len(tr_items)}, val={len(va_items)}", flush=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    tr_loader = DataLoader(LineVectorDataset(tr_items, train=True), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    va_loader = DataLoader(LineVectorDataset(va_items, train=False), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    base_dim = 48
    net = LineVectorNet(base=base_dim, out_ch=9).to(device)
    if VEC_V2_CKPT.is_file():
        # Warm-start encoder/decoder backbone ONLY; reinitialize head_cls & head_vec cleanly for v4 scales & zero shells!
        sd_v2 = torch.load(VEC_V2_CKPT, map_location=device, weights_only=True)
        sd_cur = net.state_dict()
        loaded_keys = [k for k, v in sd_v2.items() if not k.startswith(("head_cls.", "head_vec.")) and k in sd_cur and sd_cur[k].shape == v.shape]
        for k in loaded_keys:
            sd_cur[k] = sd_v2[k]
        net.load_state_dict(sd_cur)
        print(f"Warm-started v4 backbone ({len(loaded_keys)}/{len(sd_cur)} tensors) from v2; fresh heads!", flush=True)

    epochs = 18
    lr = 1.0e-3
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val = 1e9
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        net.train()
        tr_tot = 0.0
        n_steps = len(tr_loader)
        for step_i, (bx, by, bgw) in enumerate(tr_loader, 1):
            bx, by, bgw = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), bgw.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = net(bx)
            loss, _, _, _, _ = _vector_loss(pred.float(), by, bgw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tr_tot += float(loss.item()) * len(bx)
            if ep == 1 and (step_i == 10 or step_i % 60 == 0):
                res_gb = (torch.cuda.max_memory_reserved(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
                print(f"  [Ep 1 step {step_i}/{n_steps}] loss={float(loss.item()):.4f} | VRAM res={res_gb:.2f}GB | {time.time()-t_ep:.1f}s", flush=True)
        sch.step()

        net.eval()
        va_tot, va_cls, off_px_acc, ang_deg_acc, rad_px_acc, n_syn_eval = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for bx, by, bgw in va_loader:
                bx, by, bgw = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), bgw.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    pred = net(bx)
                pf = pred.float()
                tot, l_c, _, _, _ = _vector_loss(pf, by, bgw)
                va_tot += float(tot.item()) * len(bx)
                va_cls += float(l_c.item()) * len(bx)

                syn_m = bgw > 0.9
                if syn_m.any():
                    ps = pf[syn_m]
                    ts = by[syn_m]
                    act = (ts[:, 0] > 0.25) & (ts[:, 3] < 0.25)
                    if act.any():
                        dx_e = torch.abs((ps[:, 4] - ts[:, 4]) * OFFSET_SCALE)[act].mean()
                        dy_e = torch.abs((ps[:, 5] - ts[:, 5]) * OFFSET_SCALE)[act].mean()
                        off_px_acc += float(0.5 * (dx_e + dy_e)) * int(syn_m.sum())
                        th_p = 0.5 * torch.atan2((ps[:, 7] - 0.5) * 2.0, (ps[:, 6] - 0.5) * 2.0)
                        th_t = 0.5 * torch.atan2((ts[:, 7] - 0.5) * 2.0, (ts[:, 6] - 0.5) * 2.0)
                        dth = torch.abs(th_p - th_t)
                        dth = torch.minimum(dth, np.pi - dth)
                        ang_deg_acc += float((dth[act] * (180.0 / np.pi)).mean()) * int(syn_m.sum())
                        rad_px_acc += float((torch.abs(ps[:, 8] - ts[:, 8]) * RAD_SCALE)[act].mean()) * int(syn_m.sum())
                        n_syn_eval += int(syn_m.sum())

        va_tot /= max(1, len(va_items))
        va_cls /= max(1, len(va_items))
        off_px = off_px_acc / max(1, n_syn_eval)
        ang_deg = ang_deg_acc / max(1, n_syn_eval)
        rad_px = rad_px_acc / max(1, n_syn_eval)
        if va_tot <= best_val:
            best_val = va_tot
            torch.save(net.state_dict(), VEC_V4_CKPT)
        print(
            f"Ep {ep:02d}/{epochs} ({time.time() - t_ep:.1f}s) | tr={tr_tot/len(tr_items):.4f} "
            f"val={va_tot:.4f} cls={va_cls:.4f} | off={off_px:.3f}px ang={ang_deg:.2f}deg rad={rad_px:.3f}px",
            flush=True,
        )

    net.load_state_dict(torch.load(VEC_V4_CKPT, map_location="cpu", weights_only=True))
    net.cpu().eval()
    dummy = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    VEC_ONNX.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(VEC_ONNX),
        input_names=["rgb"],
        output_names=["vec_fields"],
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(VEC_ONNX), providers=["CPUExecutionProvider"])
    pt_out = net(dummy).detach().numpy()
    ort_out = sess.run(None, {"rgb": dummy.numpy()})[0]
    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"Exported {VEC_ONNX} ({VEC_ONNX.stat().st_size / 1024:.1f} KB) | PT-vs-ORT max_diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
