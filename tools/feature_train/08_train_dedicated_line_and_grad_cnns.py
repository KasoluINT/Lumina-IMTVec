"""Phase 6: Train DedicatedLineNet (4-ch line & AA-halo recognition) and ElementGradNet
(6-element semantic taxonomy + 2-ch smooth gradient flow direction field) on the expanded
multi-gallery teacher dataset + exemplar SVG ground-truth pairs.

Exports:
  - weights/line_net.onnx
  - weights/element_grad.onnx
"""
from __future__ import annotations

import concurrent.futures
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "feature_train"))

from lumina_vec.color import srgb_to_oklab
from lumina_vec.feature_nets import DedicatedLineNet, ElementGradNet

import importlib
ds_mod = importlib.import_module("02_build_datasets")

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
DEFECT_DIR = CACHE_DIR / "defect_mining"
LINE_ONNX = ROOT / "weights" / "line_net.onnx"
ELEM_ONNX = ROOT / "weights" / "element_grad.onnx"
LINE_CKPT = CACHE_DIR / "dedicated_line_best.pt"
LINE_V2_CKPT = CACHE_DIR / "dedicated_line_v2_best.pt"
ELEM_CKPT = CACHE_DIR / "element_grad_best.pt"


def build_line_and_elem_targets(
    rgb: np.ndarray,
    anime_u8: np.ndarray,
    manga_u8: np.ndarray,
    alpha_u8: np.ndarray,
    stem: str = "",
    style: str = "mixed",
) -> tuple[np.ndarray, np.ndarray]:
    """Build:
      1. line_tgt [4, H, W] float32 in [0, 1]:
         L0: structural_contour    (outer contours & strong compartment dividers)
         L1: internal_colored_line (low-contrast internal hair bang lines, cloth folds, colored arcs)
         L2: free_wisp_strand      (open-ended thin hair wisps, eyelashes, fine strands)
         L3: aa_halo_rim           (1-2px anti-aliasing transition rim around lines)
      2. elem_tgt [8, H, W] float32 in [0, 1]:
         E0: flat_cel
         E1: grad_directional      (Type I directional ribbon gradients: hair locks, skin shading, folds)
         E2: grad_radial           (Type II concentric ring gradients: cheek blush, spherical highlights, orbs)
         E3: eye_iris_core         (eye iris bands, pupils, specular highlights)
         E4: micro_ornament_feature(mouths, hairpins, jewelry/gems, flowers, lace, text/logos)
         E5: textured_wash_foliage (watercolor texture, foliage, ice shards, background grain)
         G0: grad_dir_cos_norm     (0.5 * (cos(theta) + 1) of smooth colour gradient direction)
         G1: grad_dir_sin_norm     (0.5 * (sin(theta) + 1) of smooth colour gradient direction)
    """
    h, w = rgb.shape[:2]
    smooth = cv2.bilateralFilter(rgb, 5, 16, 5)
    lab = srgb_to_oklab(smooth).astype(np.float32)
    L, a_c, b_c = lab[..., 0], lab[..., 1], lab[..., 2]
    anime = anime_u8.astype(np.float32) / 255.0
    manga = manga_u8.astype(np.float32) / 255.0
    fg = (alpha_u8 > 128) & np.any(rgb < 246, axis=2)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    loc_min3 = cv2.erode(L, np.ones((3, 3), np.uint8))
    loc_max3 = cv2.dilate(L, np.ones((3, 3), np.uint8))
    loc_max5 = cv2.dilate(L, k5)
    loc_max_L = cv2.dilate(L, k7)

    # --- 1. Dedicated 4-ch 3D-Oklab Line & Color-Boundary Targets (ZERO hair-fill flood!) ---
    lab_raw = srgb_to_oklab(rgb).astype(np.float32)
    L_raw = lab_raw[..., 0]
    lab_3d = np.stack([L, a_c * 1.90, b_c * 1.90], axis=-1).astype(np.float32)
    p1 = np.pad(L, 1, mode="edge")
    p2 = np.pad(L, 2, mode="edge")
    pr1 = np.pad(L_raw, 1, mode="edge")
    pr2 = np.pad(L_raw, 2, mode="edge")
    c1 = np.pad(lab_3d, ((1, 1), (1, 1), (0, 0)), mode="edge")
    c2 = np.pad(lab_3d, ((2, 2), (2, 2), (0, 0)), mode="edge")
    lraw_min3 = cv2.erode(L_raw, np.ones((3, 3), np.uint8))
    lraw_max5 = cv2.dilate(L_raw, k5)
    dirs = (
        ((slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None)), (slice(2, -2), slice(None, -4)), (slice(2, -2), slice(4, None))),
        ((slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)), (slice(None, -4), slice(2, -2)), (slice(4, None), slice(2, -2))),
        ((slice(None, -2), slice(2, None)), (slice(2, None), slice(None, -2)), (slice(None, -4), slice(4, None)), (slice(4, None), slice(None, -4))),
        ((slice(None, -2), slice(None, -2)), (slice(2, None), slice(2, None)), (slice(None, -4), slice(None, -4)), (slice(4, None), slice(4, None))),
    )
    valley_1px = np.zeros((h, w), np.float32)
    trough_1px = np.zeros((h, w), np.float32)
    color_ridge_1px = np.zeros((h, w), np.float32)
    color_step_1px = np.zeros((h, w), np.float32)
    for d1a, d1b, d2a, d2b in dirs:
        sa1, sb1 = p1[d1a], p1[d1b]
        sa2, sb2 = p2[d2a], p2[d2b]
        # True 1-2px dark valley peak: BOTH perpendicular flanks are strictly brighter than center!
        s_mono = (sa1 >= L - 0.002) & (sb1 >= L - 0.002)
        s_lo1 = np.minimum(sa1, sb1) - L
        s_lo2 = np.minimum(sa2, sb2) - L
        v_str = np.where(s_mono, np.maximum(s_lo1, s_lo2 * 0.85), 0.0)
        thr_v = np.where(np.maximum(anime, manga) > 0.06, 0.007, 0.016)
        valley_1px = np.maximum(valley_1px, np.clip((v_str - thr_v) / 0.052, 0.0, 1.0))

        # 1px outer contour trough between darker stroke/hair and bright skin/bg (alpha-blended thin lines)
        ra1, rb1 = pr1[d1a], pr1[d1b]
        ra2, rb2 = pr2[d2a], pr2[d2b]
        lo1 = np.minimum(ra1, rb1) - L_raw
        lo2 = np.minimum(ra2, rb2) - L_raw
        hi1 = np.maximum(ra1, rb1) - L_raw
        hi2 = np.maximum(ra2, rb2) - L_raw
        is_tr = (L_raw <= lraw_min3 + 0.010) & ((lo1 >= 0.001) | ((lo1 >= -0.002) & (lo2 >= 0.008))) & (hi1 >= 0.045) & (hi2 >= 0.070)
        t_str = np.where(is_tr, np.clip((hi2 - 0.05) / 0.16, 0.0, 1.0), 0.0)
        trough_1px = np.maximum(trough_1px, t_str)

        # 3D Oklab Symmetric Colored Line Ridge (catches bright/yellow/peach/pink/cyan thin outlines!)
        ca1, cb1 = c1[d1a], c1[d1b]
        ca2, cb2 = c2[d2a], c2[d2b]
        de_c1 = np.minimum(np.linalg.norm(ca1 - lab_3d, axis=-1), np.linalg.norm(cb1 - lab_3d, axis=-1))
        de_c2 = np.minimum(np.linalg.norm(ca2 - lab_3d, axis=-1), np.linalg.norm(cb2 - lab_3d, axis=-1))
        de_flank2 = np.linalg.norm(ca2 - cb2, axis=-1)
        is_c_ridge = (de_c1 >= 0.022) & (de_c2 >= 0.032) & (de_flank2 <= 1.25 * de_c2)
        color_ridge_1px = np.maximum(color_ridge_1px, np.where(is_c_ridge, np.clip((de_c2 - 0.026) / 0.075, 0.0, 1.0), 0.0))

        # 3D Oklab Directional Step-Edge NMS (clean cel/shadow color-block boundaries, zero gradient ramp flood!)
        g1 = np.linalg.norm(ca1 - cb1, axis=-1)
        g2 = np.linalg.norm(ca2 - cb2, axis=-1)
        flank_a = np.linalg.norm(ca2 - ca1, axis=-1)
        flank_b = np.linalg.norm(cb2 - cb1, axis=-1)
        g1_pad = np.pad(g1, 1, mode="edge")
        ga1, gb1 = g1_pad[d1a], g1_pad[d1b]
        is_step_peak = (
            (g1 >= ga1 - 1e-4)
            & (g1 >= gb1 - 1e-4)
            & (g1 >= 0.026)
            & (g2 >= 0.038)
            & (g1 >= 0.52 * np.maximum(g2, 1e-4))
            & (np.maximum(flank_a, flank_b) <= 0.65 * g1)
        )
        color_step_1px = np.maximum(color_step_1px, np.where(is_step_peak, np.clip((g2 - 0.030) / 0.095, 0.0, 1.0), 0.0))

    trough_1px *= ((lraw_max5 - L_raw) >= 0.12).astype(np.float32)

    # Symmetric 1-2px bright-on-dark ridge (captures flying white/pastel wisps on black backgrounds)
    bright_ridge_1px = np.zeros((h, w), np.float32)
    for d1a, d1b, d2a, d2b in dirs:
        f1 = np.maximum(pr1[d1a], pr1[d1b])
        f2 = np.maximum(pr2[d2a], pr2[d2b])
        d_flank = np.maximum(L_raw - f1, (L_raw - f2) * 0.85)
        is_br = (L_raw >= f1 - 0.005) & (f2 < 0.42) & (d_flank > 0.055)
        bright_ridge_1px = np.maximum(bright_ridge_1px, np.where(is_br, np.clip((d_flank - 0.045) / 0.14, 0.0, 1.0), 0.0))

    # Remove tiny isolated <= 3px specks across extracted ridge/step maps
    for arr_cl in (valley_1px, color_ridge_1px, color_step_1px, bright_ridge_1px):
        vm = (arr_cl > 0.14).astype(np.uint8)
        n_v, lv, stv, _ = cv2.connectedComponentsWithStats(vm, connectivity=8)
        for vi in range(1, n_v):
            if int(stv[vi, cv2.CC_STAT_AREA]) <= 3 and max(int(stv[vi, cv2.CC_STAT_WIDTH]), int(stv[vi, cv2.CC_STAT_HEIGHT])) <= 3:
                arr_cl[lv == vi] = 0.0

    fg_l = (alpha_u8 > 128) & (np.any(rgb < 246, axis=2) | (bright_ridge_1px > 0.16))

    # Contrast-boost clean anime & manga teacher line maps while strictly excluding flat/dense dark hair interiors
    anime_sharp = np.clip((anime - 0.07) / 0.42, 0.0, 1.0)
    manga_sharp = np.clip((manga - 0.09) / 0.45, 0.0, 1.0)
    dark_valid_line = (L >= 0.36) | (valley_1px > 0.16) | (trough_1px > 0.16) | ((loc_max3 - L) >= 0.12)
    is_ridge_or_edge = (
        dark_valid_line
        & (
            (valley_1px > 0.12)
            | (trough_1px > 0.15)
            | ((loc_max3 - L) >= 0.032)
        )
    )
    dark_drop = np.clip((loc_max3 - L) / 0.14, 0.0, 1.0)

    # L0: dark ink & hair contours (0 inside dense dark hair locks!)
    L0_teacher = np.maximum(manga_sharp * np.clip((loc_max3 - L) / 0.05, 0.0, 1.0), anime_sharp * dark_drop)
    L0 = np.clip(np.maximum(L0_teacher, trough_1px * np.clip(manga * 2.5 + anime * 2.5, 0.30, 1.0)), 0.0, 1.0)
    L0 = np.where(fg_l & is_ridge_or_edge, L0, 0.0)

    # L1: colored / iso-luminant / bright-on-dark thin lines & internal colored dividers
    L1_raw = np.maximum(anime_sharp * 0.90 * is_ridge_or_edge.astype(np.float32), np.maximum(valley_1px * 0.90, np.maximum(color_ridge_1px * 0.95, bright_ridge_1px * 0.92)))
    L1 = np.clip(np.where(fg_l, L1_raw, 0.0), 0.0, 1.0)

    # L2: distinct 1px color-block step boundaries (A5 flower petals/stamen, test1 hair shadow blocks, cel partitions)
    L2 = np.clip(np.where(fg, color_step_1px, 0.0), 0.0, 1.0)

    # L3: free wisp strands & eyelashes (strictly outside solid dark fills!)
    solid_dark = cv2.erode((L < 0.45).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    thin_zone = ~cv2.dilate(solid_dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    L3_raw = np.maximum(anime_sharp * 0.92, np.maximum(valley_1px * 0.90, trough_1px * 0.88))
    L3 = np.clip(np.maximum(L3_raw * thin_zone.astype(np.float32) * is_ridge_or_edge.astype(np.float32), bright_ridge_1px * 0.92), 0.0, 1.0)
    L3 = np.where(fg_l, L3, 0.0)

    line_tgt = np.stack([L0, L1, L2, L3], axis=0).astype(np.float32)

    # --- 2. Dedicated 8-ch Element & Gradient Topology Targets ---
    line_union = np.clip(np.maximum(L0, np.maximum(L1, L3)), 0.0, 1.0)
    non_line = 1.0 - line_union

    # Smooth macro colour gradient field (sigma=6px) for gradient magnitude & flow direction
    L_sm = cv2.GaussianBlur(L, (0, 0), 5.0)
    a_sm = cv2.GaussianBlur(a_c, (0, 0), 5.0)
    b_sm = cv2.GaussianBlur(b_c, (0, 0), 5.0)
    gx_L = cv2.Sobel(L_sm, cv2.CV_32F, 1, 0, ksize=3)
    gy_L = cv2.Sobel(L_sm, cv2.CV_32F, 0, 1, ksize=3)
    gx_a = cv2.Sobel(a_sm, cv2.CV_32F, 1, 0, ksize=3)
    gy_a = cv2.Sobel(a_sm, cv2.CV_32F, 0, 1, ksize=3)
    gx_b = cv2.Sobel(b_sm, cv2.CV_32F, 1, 0, ksize=3)
    gy_b = cv2.Sobel(b_sm, cv2.CV_32F, 0, 1, ksize=3)

    # Dominant gradient vector combining lightness and chroma
    gx_tot = gx_L + 1.5 * np.where(gx_L * gx_a >= 0, gx_a, -gx_a) + 1.5 * np.where(gx_L * gx_b >= 0, gx_b, -gx_b)
    gy_tot = gy_L + 1.5 * np.where(gy_L * gy_a >= 0, gy_a, -gy_a) + 1.5 * np.where(gy_L * gy_b >= 0, gy_b, -gy_b)
    g_mag = np.hypot(gx_tot, gy_tot)
    G0 = np.clip(0.5 * (np.divide(gx_tot, g_mag + 1e-5) + 1.0), 0.0, 1.0)
    G1 = np.clip(0.5 * (np.divide(gy_tot, g_mag + 1e-5) + 1.0), 0.0, 1.0)

    # Facial & eye detection
    delicate = ds_mod.detect_facial_delicate(rgb, np.maximum(anime, manga), fg=fg, stem=stem)

    # Separate E2 (grad_radial: cheek blush, spherical chest/shoulder highlights, glowing orbs)
    # vs E3 (eye_iris_core) vs E4 (micro_ornament_feature: mouths, pins, jewels, flowers, lace)
    chroma = np.hypot(a_c, b_c)
    skin = fg & (L > 0.60) & (L < 0.975) & (a_c > -0.002) & (a_c < 0.095) & (b_c > -0.005) & (b_c < 0.105) & (chroma < 0.115)
    skin_dt = distance_transform_edt(skin)
    skin_bg_a = cv2.GaussianBlur(a_c, (0, 0), max(12.0, min(h, w) * 0.025))
    skin_bg_L = cv2.GaussianBlur(L, (0, 0), max(12.0, min(h, w) * 0.025))

    # E2: Concentric Radial Diffusion (cheek blush ovals + spherical skin highlights + glowing halos)
    blush_oval = skin & (skin_dt >= max(8.0, 0.012 * min(h, w))) & ((a_c - skin_bg_a) > 0.012) & (a_c > 0.025)
    sphere_hi = skin & (skin_dt >= max(10.0, 0.015 * min(h, w))) & ((L - skin_bg_L) > 0.028) & (L > 0.85)
    E2 = cv2.GaussianBlur((blush_oval | sphere_hi).astype(np.float32), (0, 0), 4.0)
    E2 = np.clip(E2 * 1.4 * non_line, 0.0, 1.0)

    # E3: Eye iris & specular glints (delicate regions with strong internal contrast or chromatic iris)
    E3 = np.clip(delicate * (1.0 - 0.7 * E2), 0.0, 1.0)

    # E4: Micro ornaments & accessories (high local chroma/detail compact islands: hairpins, gems, flowers, lace)
    lap = np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3))
    high_detail_zone = cv2.GaussianBlur((lap > 0.025).astype(np.float32), (0, 0), 3.5)
    gem_or_pin = (chroma > 0.085) & (high_detail_zone > 0.18) & (~skin)
    E4 = np.clip(np.maximum(delicate * 0.8, gem_or_pin.astype(np.float32) * 0.85) * non_line, 0.0, 1.0)

    # E0 (flat_cel) vs E1 (grad_directional) vs E5 (textured_wash_foliage)
    macro_step = cv2.GaussianBlur(g_mag, (0, 0), 4.0)
    micro_grain = cv2.GaussianBlur(lap, (0, 0), 3.0)

    E0 = np.where((~fg) | ((macro_step < 0.012) & (micro_grain < 0.008)), 1.0, 0.0).astype(np.float32) * non_line
    E1 = np.clip(((macro_step - 0.010) / 0.055) * (1.0 - 0.6 * E2) * non_line, 0.0, 1.0)
    if style in ("soft_gradient", "mixed"):
        E1 = np.maximum(E1, np.where(fg & (macro_step >= 0.007), 0.75 * non_line, 0.0).astype(np.float32))

    E5 = np.clip(((micro_grain - 0.008) / 0.025) * (1.0 - 0.7 * E3) * (1.0 - 0.5 * E4) * non_line, 0.0, 1.0)
    if style == "grain_paint":
        E5 = np.maximum(E5, 0.65 * fg.astype(np.float32) * non_line)

    elem_tgt = np.stack([E0, E1, E2, E3, E4, E5, G0, G1], axis=0).astype(np.float32)
    return line_tgt, elem_tgt


class DualTargetDataset(Dataset):
    def __init__(self, items, train: bool = True):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y_line, y_elem, w_line = self.items[idx]
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
                y_line = y_line[:, :, ::-1]
                y_elem = y_elem[:, :, ::-1].copy()
                # Flip cos(theta) horizontally: cos_new = 1.0 - cos_old
                y_elem[6] = 1.0 - y_elem[6]
                w_line = w_line[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[::-1, :]
                y_line = y_line[:, ::-1, :]
                y_elem = y_elem[:, ::-1, :].copy()
                # Flip sin(theta) vertically: sin_new = 1.0 - sin_old
                y_elem[7] = 1.0 - y_elem[7]
                w_line = w_line[:, ::-1, :]
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
            xf = np.clip(xf * float(np.random.uniform(0.93, 1.07)) + float(np.random.uniform(-0.025, 0.025)), 0.0, 1.0)
        else:
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0

        return (
            torch.from_numpy(xf).permute(2, 0, 1).contiguous(),
            torch.from_numpy(np.ascontiguousarray(y_line).astype(np.float32) / 255.0).contiguous(),
            torch.from_numpy(np.ascontiguousarray(y_elem).astype(np.float32) / 255.0).contiguous(),
            torch.from_numpy(np.ascontiguousarray(w_line).astype(np.float32)).contiguous(),
        )


def _loss_fn(pred: torch.Tensor, target: torch.Tensor, wgt: torch.Tensor | None = None, edge_focus: bool = False) -> torch.Tensor:
    l1 = torch.abs(pred - target)
    p_c = pred.clamp(1e-4, 1.0 - 1e-4)
    bce = -(target * torch.log(p_c) + (1.0 - target) * torch.log(1.0 - p_c))
    if edge_focus:
        pos_w = 1.0 + 2.2 * target
        l1 = l1 * pos_w
        bce = bce * pos_w
    if wgt is not None:
        l1 = l1 * wgt
        bce = bce * wgt
    inter = (pred * target).sum(dim=(2, 3))
    denom = (pred + target).sum(dim=(2, 3)) + 1e-4
    dice = 1.0 - (2.0 * inter / denom)
    return 0.45 * l1.mean() + 0.35 * bce.mean() + 0.25 * dice.mean()


def _prep_teacher_item(args):
    rgb_path, teacher_path, style, is_val, line_only, n_crops, seed_i = args
    rng_w = np.random.default_rng(seed_i)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    npz = np.load(teacher_path)
    anime, manga, alpha = npz["anime"], npz["manga"], npz["alpha"]
    stem = Path(rgb_path).stem.split("_", 1)[-1]
    y_l, y_e = build_line_and_elem_targets(rgb, anime, manga, alpha, stem=stem, style=style)
    y_l_u8 = np.rint(np.clip(y_l, 0, 1) * 255).astype(np.uint8)
    y_e_u8 = np.zeros((8, 1, 1), np.uint8) if line_only else np.rint(np.clip(y_e, 0, 1) * 255).astype(np.uint8)

    out = []
    r512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    yl512 = np.stack([cv2.resize(y_l_u8[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(4)], axis=0)
    ye512 = y_e_u8 if line_only else np.stack([cv2.resize(y_e_u8[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(8)], axis=0)
    out.append((r512, yl512, ye512, np.ones((1, 512, 512), np.float32)))

    h, w = rgb.shape[:2]
    if min(h, w) >= 512:
        score_map = y_l[0] + y_l[1] + y_l[3] + (y_e[3] if not line_only else 0.0)
        ys, xs = np.where(score_map > np.quantile(score_map, 0.80))
        nc = 1 if is_val else n_crops
        for _ in range(nc):
            if len(ys):
                pick = int(rng_w.integers(0, len(ys)))
                y0 = int(np.clip(int(ys[pick]) - 256 + int(rng_w.integers(-64, 65)), 0, h - 512))
                x0 = int(np.clip(int(xs[pick]) - 256 + int(rng_w.integers(-64, 65)), 0, w - 512))
            else:
                y0 = int(rng_w.integers(0, h - 512 + 1))
                x0 = int(rng_w.integers(0, w - 512 + 1))
            out.append((
                rgb[y0:y0 + 512, x0:x0 + 512].copy(),
                y_l_u8[:, y0:y0 + 512, x0:x0 + 512].copy(),
                y_e_u8 if line_only else y_e_u8[:, y0:y0 + 512, x0:x0 + 512].copy(),
                np.ones((1, 512, 512), np.float32),
            ))
    return is_val, out


def main():
    line_only = "--line-only" in sys.argv
    rng = np.random.default_rng(19)
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    holdout_ids = {2, 4}  # Hold out test1 & test12 strictly
    val_ids = {7, 14, 19, 27, 33, 41, 46}
    e_mpath = CACHE_DIR / "e_gallery" / "e_manifest.json"
    e_manifest = json.loads(e_mpath.read_text(encoding="utf-8")) if e_mpath.is_file() else []

    jobs = []
    for item in manifest:
        if item["id"] in holdout_ids:
            continue
        jobs.append((item["rgb_path"], item["teacher_path"], item["style"], item["id"] in val_ids, line_only, 6, 1000 + item["id"]))
    for ei, item in enumerate(e_manifest):
        jobs.append((item["rgb_path"], item["teacher_path"], "mixed", (ei % 12 == 0), line_only, 4, 5000 + ei))

    tr_items, va_items = [], []
    t_prep = time.time()
    print(f"Building 512x512 multi-scale tiles via 8 workers from {len(jobs)} images (line_only={line_only})...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for is_val, items in ex.map(_prep_teacher_item, jobs, chunksize=4):
            (va_items if is_val else tr_items).extend(items)
    print(f"Teacher tiles ready in {time.time() - t_prep:.1f}s: train={len(tr_items)}, val={len(va_items)}", flush=True)

    # 3. Add Exemplar SVG Ground-Truth Defect Tiles (a1, a2, A3, A4, bbb)
    for npz_p in sorted(DEFECT_DIR.glob("diff_*.npz")):
        d = np.load(npz_p)
        src = d["src"]
        clutter = d["clutter"].astype(np.float32) / 255.0
        missing = d["missing"].astype(np.float32) / 255.0
        ref_edge = d["ref_edge"].astype(np.float32) / 255.0
        h, w = src.shape[:2]
        pseudo_a = np.rint(np.clip(np.maximum(ref_edge, missing), 0, 1) * 255).astype(np.uint8)
        y_l, y_e = build_line_and_elem_targets(src, pseudo_a, pseudo_a, np.full((h, w), 255, np.uint8), stem=npz_p.stem)
        # Boost missed lines strongly on L0/L1/L3, suppress false clutter lines
        y_l[0] = np.clip(np.maximum(y_l[0], missing * 0.9), 0, 1)
        y_l[1] = np.clip(np.maximum(y_l[1], missing * 0.95), 0, 1)
        y_l[3] = np.clip(np.maximum(y_l[3], missing * 0.95), 0, 1)
        sup = clutter > 0.5
        y_l[0][sup] = 0.0; y_l[1][sup] = 0.0; y_l[3][sup] = 0.0
        # Where clutter occurred on smooth gradients, boost E1 (grad_directional) & E5 (wash)
        y_e[1] = np.clip(np.maximum(y_e[1], clutter * 0.85), 0, 1)
        y_e[5] = np.clip(np.maximum(y_e[5], clutter * 0.85), 0, 1)

        y_l_u8 = np.rint(np.clip(y_l, 0, 1) * 255).astype(np.uint8)
        y_e_u8 = np.zeros((8, 1, 1), np.uint8) if line_only else np.rint(np.clip(y_e, 0, 1) * 255).astype(np.uint8)
        w_map = (1.0 + 3.5 * missing + 2.5 * clutter)[None].astype(np.float32)

        r512 = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)
        yl512 = np.stack([cv2.resize(y_l_u8[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(4)], axis=0)
        ye512 = y_e_u8 if line_only else np.stack([cv2.resize(y_e_u8[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(8)], axis=0)
        w512 = cv2.resize(w_map[0], (512, 512), interpolation=cv2.INTER_AREA)[None]
        tr_items.append((r512, yl512, ye512, w512))

        if min(h, w) >= 512:
            err = clutter + 1.6 * missing
            ys, xs = np.where(err > np.quantile(err, 0.88))
            for _ in range(10):
                pick = int(rng.integers(0, len(ys)))
                y0 = int(np.clip(int(ys[pick]) - 256 + int(rng.integers(-64, 65)), 0, h - 512))
                x0 = int(np.clip(int(xs[pick]) - 256 + int(rng.integers(-64, 65)), 0, w - 512))
                tr_items.append((
                    src[y0:y0 + 512, x0:x0 + 512],
                    y_l_u8[:, y0:y0 + 512, x0:x0 + 512],
                    y_e_u8 if line_only else y_e_u8[:, y0:y0 + 512, x0:x0 + 512],
                    w_map[:, y0:y0 + 512, x0:x0 + 512],
                ))

    print(f"Dataset ready: train={len(tr_items)} tiles, val={len(va_items)} tiles", flush=True)
    torch.backends.cudnn.benchmark = True
    tr_loader = DataLoader(DualTargetDataset(tr_items, train=True), batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    va_loader = DataLoader(DualTargetDataset(va_items, train=False), batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line_base = 64
    line_net = DedicatedLineNet(base=line_base, out_ch=4).to(device)
    if LINE_CKPT.is_file():
        print(f"Warm-starting DedicatedLineNet v2 from {LINE_CKPT}...", flush=True)
        line_net.load_state_dict(torch.load(LINE_CKPT, map_location=device, weights_only=True))
        epochs = 8 if line_only else 10
        lr_l = 6.0e-4
    else:
        epochs = 16 if line_only else 12
        lr_l = 1.4e-3
    elem_net = None if line_only else ElementGradNet(base=48, out_ch=8).to(device)

    opt_l = torch.optim.AdamW(line_net.parameters(), lr=lr_l, weight_decay=2e-4)
    opt_e = None if line_only else torch.optim.AdamW(elem_net.parameters(), lr=1.2e-3, weight_decay=2e-4)
    sch_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_l, T_max=epochs)
    sch_e = None if line_only else torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=epochs)

    best_l, best_e = 1e9, 1e9
    for ep in range(1, epochs + 1):
        t0 = time.time()
        line_net.train()
        if elem_net is not None:
            elem_net.train()
        tr_ll, tr_el = 0.0, 0.0
        for step_i, (bx, byl, bye, bw) in enumerate(tr_loader, 1):
            bx, byl, bye, bw = bx.to(device, non_blocking=True), byl.to(device, non_blocking=True), bye.to(device, non_blocking=True), bw.to(device, non_blocking=True)

            opt_l.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pl = line_net(bx)
            loss_l = _loss_fn(pl.float(), byl, bw, edge_focus=True)
            loss_l.backward()
            torch.nn.utils.clip_grad_norm_(line_net.parameters(), 2.0)
            opt_l.step()
            tr_ll += float(loss_l.item()) * len(bx)
            if ep == 1 and (step_i == 10 or step_i % 80 == 0):
                res_gb = (torch.cuda.max_memory_reserved(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
                print(f"  [Ep 1 step {step_i}/{len(tr_loader)}] loss={float(loss_l.item()):.4f} | VRAM_res={res_gb:.2f}GB | {time.time()-t0:.1f}s", flush=True)

            if elem_net is not None:
                opt_e.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    pe = elem_net(bx)
                loss_e = _loss_fn(pe.float(), bye, None)
                loss_e.backward()
                torch.nn.utils.clip_grad_norm_(elem_net.parameters(), 2.0)
                opt_e.step()
                tr_el += float(loss_e.item()) * len(bx)

        sch_l.step()
        if sch_e is not None:
            sch_e.step()

        line_net.eval()
        if elem_net is not None:
            elem_net.eval()
        va_ll, va_el = 0.0, 0.0
        with torch.no_grad():
            for bx, byl, bye, bw in va_loader:
                bx, byl, bye = bx.to(device, non_blocking=True), byl.to(device, non_blocking=True), bye.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    pl = line_net(bx)
                    pe = elem_net(bx) if elem_net is not None else None
                va_ll += float(_loss_fn(pl.float(), byl, None, edge_focus=True).item()) * len(bx)
                if pe is not None:
                    va_el += float(_loss_fn(pe.float(), bye).item()) * len(bx)
        va_ll /= max(1, len(va_items))
        va_el /= max(1, len(va_items))
        if va_ll <= best_l:
            best_l = va_ll
            torch.save(line_net.state_dict(), LINE_V2_CKPT)
        if elem_net is not None and va_el <= best_e:
            best_e = va_el
            torch.save(elem_net.state_dict(), ELEM_CKPT)
        msg = f"Ep {ep:02d}/{epochs} ({time.time()-t0:.1f}s) | LineNet tr={tr_ll/len(tr_loader.dataset):.4f} val={va_ll:.4f}"
        if elem_net is not None:
            msg += f" | ElemGradNet tr={tr_el/len(tr_loader.dataset):.4f} val={va_el:.4f}"
        print(msg, flush=True)

    # Export trained model(s) to ONNX Runtime
    dummy = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    export_list = [(DedicatedLineNet, LINE_V2_CKPT, LINE_ONNX, 4, line_base)]
    if not line_only:
        export_list.append((ElementGradNet, ELEM_CKPT, ELEM_ONNX, 8, 48))
    for net_cls, ckpt, out_p, out_ch, b_dim in export_list:
        net = net_cls(base=b_dim, out_ch=out_ch).cpu().eval()
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        torch.onnx.export(
            net, dummy, str(out_p),
            input_names=["rgb"], output_names=["maps"],
            dynamic_axes={"rgb": {0: "b", 2: "h", 3: "w"}, "maps": {0: "b", 2: "h", 3: "w"}},
            opset_version=17, dynamo=False,
        )
        sess = ort.InferenceSession(str(out_p), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        diff = float(np.max(np.abs(net(dummy).detach().numpy() - sess.run(None, {"rgb": dummy.numpy()})[0])))
        print(f"Exported {out_p.name} ({out_p.stat().st_size/1024:.1f} KB), PT-vs-ORT diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
