"""Closed-Region Gradient CNN (ElementGradNet v2) + Clean Margin-Suppressed LineVectorNet v3 Training.

Solves two structural issues spotted in visual verification:
1. ElementGradNet v1 predicted pixelwise local gradient magnitude, creating holes inside
   smooth highlight plateaus (e.g. stocking highlight centers). Here we build per-compartment
   CLOSED-REGION gradient ground truth so an entire smooth-shading compartment is labeled
   solidly 1.0 edge-to-edge right up to structural line walls, while crisp cel / stained-glass
   compartments remain flat E0 = 1.0.
2. LineVectorNet predicted low-confidence purple haze (0.12 - 0.38) flanking seams on smooth
   gradient slopes. Here we fine-tune LineVectorNet with strong asymmetric negative penalties on
   gradient flanks and soft shadow rims plus hard-negative gradient-seam tiles, while preserving
   all crisp Teacher-SVG hair strands and stained-glass borders.

Usage:
  python 13_train_closed_grad_and_clean_line.py --preview
  python 13_train_closed_grad_and_clean_line.py --train-grad
  python 13_train_closed_grad_and_clean_line.py --train-line
  python 13_train_closed_grad_and_clean_line.py --train-all
"""
from __future__ import annotations

import concurrent.futures
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "feature_train"))

from lumina_vec.color import srgb_to_oklab
from lumina_vec.feature_nets import ElementGradNet, LineVectorNet

m08 = importlib.import_module("08_train_dedicated_line_and_grad_cnns")

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
DEFECT_DIR = CACHE_DIR / "defect_mining"
V2_CACHE = CACHE_DIR / "v2_tiles_cache"
T_CACHE = CACHE_DIR / "teacher_svg_cache"

ELEM_CKPT_IN = CACHE_DIR / "element_grad_best.pt"
ELEM_CKPT_OUT = CACHE_DIR / "element_grad_v2_best.pt"
ELEM_ONNX_OUT = ROOT / "weights" / "element_grad.onnx"

LINE_CKPT_IN = CACHE_DIR / "line_vector_teacher_best.pt"
LINE_CKPT_OUT = CACHE_DIR / "line_vector_v3_best.pt"
LINE_ONNX_OUT = ROOT / "weights" / "line_vector_net.onnx"


def _exterior_fg_mask(rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Foreground mask that removes exterior white canvas borders via border flood-fill,
    while PRESERVING interior white dresses, eyes, and bright specular highlights."""
    h, w = rgb.shape[:2]
    fg = alpha_u8 > 128
    near_white = (np.min(rgb, axis=2) >= 246).astype(np.uint8)
    n_nw, nw_cc, _, _ = cv2.connectedComponentsWithStats(near_white, connectivity=4)
    if n_nw > 1:
        border_ids = set(np.unique(np.concatenate([
            nw_cc[0, :], nw_cc[-1, :], nw_cc[:, 0], nw_cc[:, -1]
        ]))) - {0}
        if border_ids:
            ext_bg = np.isin(nw_cc, list(border_ids))
            # Only treat border-connected white regions covering > 0.5% image area as canvas backdrop
            for bid in border_ids:
                m_b = nw_cc == bid
                if int(m_b.sum()) >= max(64, int(0.002 * h * w)):
                    fg &= (~m_b)
    return fg


def build_closed_elem_targets(
    rgb: np.ndarray,
    anime_u8: np.ndarray,
    manga_u8: np.ndarray,
    alpha_u8: np.ndarray,
    stem: str = "",
    style: str = "mixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build line_tgt [4,H,W], closed-compartment elem_tgt [8,H,W], and region weight w_elem [1,H,W].

    Segments non-line interiors into closed compartments bounded by structural contours,
    colored lines, cel color-step boundaries, and 1px OKLab cliffs (>= 0.028).
    Inside each compartment, fits a 2D quadratic OKLab surface to distinguish genuine smooth
    gradient compartments (wide or narrow, 0 holes) from flat cel/stained-glass facets (E0)
    and textured grain wash (E5).
    """
    line_tgt, elem_base = m08.build_line_and_elem_targets(rgb, anime_u8, manga_u8, alpha_u8, stem=stem, style=style)
    _, _, E2_b, E3_b, E4_b, E5_b, G0, G1 = [elem_base[i] for i in range(8)]

    h, w = rgb.shape[:2]
    fg = _exterior_fg_mask(rgb, alpha_u8)
    smooth = cv2.bilateralFilter(rgb, 5, 14, 4)
    lab_raw = srgb_to_oklab(rgb).astype(np.float32)
    lab_sm = srgb_to_oklab(smooth).astype(np.float32)
    L_sm, a_sm, b_sm = lab_sm[..., 0], lab_sm[..., 1], lab_sm[..., 2]

    # Sharp 1px OKLab cliff seals cel shadow edges & stained-glass mosaic boundaries cleanly
    dx = np.linalg.norm(lab_sm[:, 1:] - lab_sm[:, :-1], axis=-1)
    dy = np.linalg.norm(lab_sm[1:, :] - lab_sm[:-1, :], axis=-1)
    cliff = np.zeros((h, w), np.float32)
    cliff[:, :-1] = np.maximum(cliff[:, :-1], dx)
    cliff[:, 1:] = np.maximum(cliff[:, 1:], dx)
    cliff[:-1, :] = np.maximum(cliff[:-1, :], dy)
    cliff[1:, :] = np.maximum(cliff[1:, :], dy)

    line_union = np.clip(np.maximum(line_tgt[0], np.maximum(line_tgt[1], line_tgt[3])), 0.0, 1.0)
    step_wall = line_tgt[2] >= 0.22
    cliff_wall = cliff >= 0.028
    line_wall = line_union >= 0.24
    wall = (~fg) | line_wall | step_wall | cliff_wall

    L_low = cv2.GaussianBlur(L_sm, (0, 0), 2.2)
    hf_grain = np.abs(lab_raw[..., 0] - L_low)

    interior = (~wall).astype(np.uint8)
    n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=4)

    E1_closed = np.zeros((h, w), np.float32)
    E0_closed = np.where(fg, 1.0, 0.0).astype(np.float32)
    E5_closed = np.zeros((h, w), np.float32)

    k3 = np.ones((3, 3), np.uint8)
    wall_near = cv2.dilate((line_wall | step_wall | cliff_wall).astype(np.uint8), k3) > 0

    for cid in range(1, n_cc):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        x0, y0, bw, bh = (
            int(stats[cid, cv2.CC_STAT_LEFT]),
            int(stats[cid, cv2.CC_STAT_TOP]),
            int(stats[cid, cv2.CC_STAT_WIDTH]),
            int(stats[cid, cv2.CC_STAT_HEIGHT]),
        )
        sub_cc = cc_map[y0:y0 + bh, x0:x0 + bw] == cid
        if area >= 120:
            sub_core = sub_cc & (~wall_near[y0:y0 + bh, x0:x0 + bw])
            if int(sub_core.sum()) < 32:
                sub_core = sub_cc
        else:
            sub_core = sub_cc

        ys_c, xs_c = np.where(sub_core)
        n_pts = len(ys_c)
        if n_pts < 24:
            continue

        # Subsample up to 1500 interior points for fast, exact 2D quadratic surface fit
        if n_pts > 1500:
            step = n_pts // 1500
            ys_s, xs_s = ys_c[::step], xs_c[::step]
        else:
            ys_s, xs_s = ys_c, xs_c

        Ls = L_sm[y0:y0 + bh, x0:x0 + bw][ys_s, xs_s]
        As = a_sm[y0:y0 + bh, x0:x0 + bw][ys_s, xs_s]
        Bs = b_sm[y0:y0 + bh, x0:x0 + bw][ys_s, xs_s]
        gr_s = hf_grain[y0:y0 + bh, x0:x0 + bw][ys_s, xs_s]

        l_span = float(np.quantile(Ls, 0.96) - np.quantile(Ls, 0.04))
        a_span = float(np.quantile(As, 0.96) - np.quantile(As, 0.04))
        b_span = float(np.quantile(Bs, 0.96) - np.quantile(Bs, 0.04))
        e_span = float(np.sqrt(l_span**2 + 2.2 * (a_span**2 + b_span**2)))

        if e_span < 0.036:
            # Strictly flat cel / stained-glass facet
            continue

        yn = (ys_s.astype(np.float32) / max(1.0, bh - 1.0)) * 2.0 - 1.0
        xn = (xs_s.astype(np.float32) / max(1.0, bw - 1.0)) * 2.0 - 1.0
        design = np.column_stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn])
        Y_mat = np.column_stack([Ls, As * 1.48, Bs * 1.48])
        coef, _, _, _ = np.linalg.lstsq(design, Y_mat, rcond=None)
        pred = design @ coef
        res = Y_mat - pred
        var_tot = float(np.var(Y_mat, axis=0).sum()) + 1e-8
        var_res = float(np.var(res, axis=0).sum())
        r2_smooth = 1.0 - var_res / var_tot
        res_rms = float(np.sqrt(np.mean(np.sum(res * res, axis=1))))
        gr_q60 = float(np.quantile(gr_s, 0.60))

        # Check grain paint vs smooth vector gradient
        is_heavy_grain = (gr_q60 >= 0.020) and (r2_smooth < 0.45)
        is_smooth_grad = (
            (e_span >= 0.040)
            and (r2_smooth >= 0.48)
            and (res_rms <= max(0.038, 0.34 * e_span))
            and (not is_heavy_grain)
        )

        # Narrow compartments (< 1500 px) require clean quadratic gradient fit
        if area < 1500 and (r2_smooth < 0.68 or e_span < 0.042):
            is_smooth_grad = False

        if is_smooth_grad:
            E1_closed[y0:y0 + bh, x0:x0 + bw][sub_cc] = 1.0
            E0_closed[y0:y0 + bh, x0:x0 + bw][sub_cc] = 0.0
        elif gr_q60 >= 0.017 or (style == "grain_paint" and res_rms >= 0.018):
            E5_closed[y0:y0 + bh, x0:x0 + bw][sub_cc] = np.clip((max(gr_q60, res_rms * 0.7) - 0.013) / 0.024, 0.35, 1.0)
            E0_closed[y0:y0 + bh, x0:x0 + bw][sub_cc] = 0.0

    # Dilate E1_closed by 2px right up to line walls so closed shape hits ink boundaries cleanly
    e1_grow = cv2.dilate((E1_closed > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    non_wall_border = fg & (~line_wall) & (~cliff_wall)
    E1_closed = np.where(e1_grow & non_wall_border, np.maximum(E1_closed, 0.95), E1_closed)
    E1_closed = np.clip(E1_closed * (1.0 - 0.85 * line_union), 0.0, 1.0)
    E0_closed = np.where(E1_closed > 0.25, 0.0, E0_closed * (1.0 - line_union))
    E5_final = np.clip(np.maximum(E5_closed, E5_b * (1.0 - E1_closed)) * (1.0 - line_union), 0.0, 1.0)

    E1_closed = np.clip(cv2.GaussianBlur(E1_closed, (3, 3), 0.7), 0.0, 1.0) * fg.astype(np.float32) * (1.0 - 0.85 * line_union)
    E0_closed = np.clip(cv2.GaussianBlur(E0_closed, (3, 3), 0.7), 0.0, 1.0) * fg.astype(np.float32) * (1.0 - 0.85 * line_union)

    elem_tgt = np.stack([E0_closed, E1_closed, E2_b, E3_b, E4_b, E5_final, G0, G1], axis=0).astype(np.float32)

    # Extra loss focus across flat centers inside E1_closed compartments (where pixel slope is small!)
    L_macro = cv2.GaussianBlur(lab_raw[..., 0], (0, 0), 4.0)
    gx_m = cv2.Sobel(L_macro, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy_m = cv2.Sobel(L_macro, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    macro_slope = np.hypot(gx_m, gy_m)
    smooth_center = (E1_closed > 0.7) & (macro_slope < 0.010)
    w_elem = (1.0 + 1.6 * smooth_center.astype(np.float32) + 0.8 * (E1_closed > 0.7).astype(np.float32))[None].astype(np.float32)
    return line_tgt, elem_tgt, w_elem


def run_preview():
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    by_stem = {x["stem"]: x for x in manifest}
    picks = ["test4", "test14", "test11", "test18"]
    out_dir = Path(r"C:\Users\30266\.gemini\antigravity\brain\439cca1c-a31d-4f73-81c2-1f9c82b1e720")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    th = 420
    for stem in picks:
        item = by_stem[stem]
        rgb = np.asarray(Image.open(item["rgb_path"]).convert("RGB"))
        npz = np.load(item["teacher_path"])
        _, old_e = m08.build_line_and_elem_targets(rgb, npz["anime"], npz["manga"], npz["alpha"], stem=stem, style=item["style"])
        _, new_e, _ = build_closed_elem_targets(rgb, npz["anime"], npz["manga"], npz["alpha"], stem=stem, style=item["style"])

        h, w = rgb.shape[:2]
        tw = max(240, int(round(w * th / h)))
        p_rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)

        def _hm(arr):
            u8 = np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
            col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
            col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
            return cv2.resize(col, (tw, th), interpolation=cv2.INTER_AREA)

        p_old_e1 = _hm(old_e[1])
        p_new_e1 = _hm(new_e[1])
        p_new_e0 = _hm(new_e[0])

        for img, title in [(p_rgb, f"{stem} RGB"), (p_old_e1, "Old E1 (holes)"), (p_new_e1, "Closed E1 (new)"), (p_new_e0, "Flat Cel E0")]:
            cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.concatenate([p_rgb, p_old_e1, p_new_e1, p_new_e0], axis=1))

    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 24, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    grid = np.concatenate(padded, axis=0)
    out_p = out_dir / "scratch_closed_grad_gt_preview.jpg"
    Image.fromarray(grid).save(out_p, quality=92)
    print(f"Saved preview to {out_p} ({grid.shape[1]}x{grid.shape[0]})", flush=True)


def _prep_closed_elem_item(args):
    rgb_path, teacher_path, style, is_val, n_crops, seed_i = args
    rng_w = np.random.default_rng(seed_i)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    npz = np.load(teacher_path)
    stem = Path(rgb_path).stem.split("_", 1)[-1]
    y_l, y_e, w_e = build_closed_elem_targets(rgb, npz["anime"], npz["manga"], npz["alpha"], stem=stem, style=style)
    y_e_u8 = np.rint(np.clip(y_e, 0, 1) * 255).astype(np.uint8)
    w_u8 = np.rint(np.clip(w_e[0] / 3.5, 0, 1) * 255).astype(np.uint8)

    out = []
    r512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    ye512 = np.stack([cv2.resize(y_e_u8[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(8)], axis=0)
    we512 = cv2.resize(w_u8, (512, 512), interpolation=cv2.INTER_AREA)[None]
    out.append((r512, ye512, we512))

    h, w = rgb.shape[:2]
    if min(h, w) >= 512:
        score = y_e[1] + 0.5 * y_e[0] + y_e[2]
        ys, xs = np.where(score > np.quantile(score, 0.72))
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
                y_e_u8[:, y0:y0 + 512, x0:x0 + 512].copy(),
                w_u8[None, y0:y0 + 512, x0:x0 + 512].copy(),
            ))
    return is_val, out


class ClosedElemDataset(Dataset):
    def __init__(self, items, train: bool = True):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y_e, w_e = self.items[idx]
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
                y_e = y_e[:, :, ::-1].copy()
                y_e[6] = 255 - y_e[6]
                w_e = w_e[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[::-1, :]
                y_e = y_e[:, ::-1, :].copy()
                y_e[7] = 255 - y_e[7]
                w_e = w_e[:, ::-1, :]
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
            xf = np.clip(xf * float(np.random.uniform(0.93, 1.07)) + float(np.random.uniform(-0.025, 0.025)), 0.0, 1.0)
        else:
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0

        wf = np.ascontiguousarray(w_e).astype(np.float32) * (3.5 / 255.0)
        return (
            torch.from_numpy(xf).permute(2, 0, 1).contiguous(),
            torch.from_numpy(np.ascontiguousarray(y_e).astype(np.float32) / 255.0).contiguous(),
            torch.from_numpy(wf).contiguous(),
        )


def _elem_loss(pred: torch.Tensor, target: torch.Tensor, wgt: torch.Tensor) -> torch.Tensor:
    l1 = torch.abs(pred - target) * wgt
    p_c = pred.clamp(1e-4, 1.0 - 1e-4)
    bce = -(target * torch.log(p_c) + (1.0 - target) * torch.log(1.0 - p_c)) * wgt
    # Focus weight 1.8x on E0 and E1 closed-region segmentation channels
    ch_w = torch.tensor([1.6, 2.0, 1.0, 1.0, 1.0, 1.1, 0.75, 0.75], device=pred.device).view(1, 8, 1, 1)
    l_pix = (0.45 * l1 + 0.40 * bce) * ch_w

    # Dice on E0..E5 semantic channels
    inter = (pred[:, :6] * target[:, :6]).sum(dim=(2, 3))
    denom = (pred[:, :6] + target[:, :6]).sum(dim=(2, 3)) + 1e-4
    dice = (1.0 - (2.0 * inter / denom)) * ch_w[:, :6, 0, 0]

    # Spatial consistency (TV smoothness) on E1 so closed compartments stay solid without noisy pinholes
    tv_pred_e1 = (torch.abs(pred[:, 1:2, :, 1:] - pred[:, 1:2, :, :-1]).mean() + torch.abs(pred[:, 1:2, 1:, :] - pred[:, 1:2, :-1, :]).mean())
    tv_tgt_e1 = (torch.abs(target[:, 1:2, :, 1:] - target[:, 1:2, :, :-1]).mean() + torch.abs(target[:, 1:2, 1:, :] - target[:, 1:2, :-1, :]).mean())
    l_solid = torch.relu(tv_pred_e1 - tv_tgt_e1)

    return l_pix.mean() + 0.30 * dice.mean() + 0.25 * l_solid


ELEM_CACHE = CACHE_DIR / "closed_elem_tiles_v2.npz"


def _cap_vram(limit_gb: float = 8.5) -> None:
    """Strictly cap CUDA process memory allocation well below 10 GB on Windows WDDM."""
    if torch.cuda.is_available():
        tot_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        frac = min(0.82, float(limit_gb) / max(tot_gb, 1.0))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        torch.cuda.reset_peak_memory_stats(0)
        print(f"[VRAM Cap] Limiting process CUDA memory to {limit_gb:.1f} GB ({frac * 100:.1f}% of {tot_gb:.2f} GB)", flush=True)


def _load_or_build_closed_elem_tiles():
    if ELEM_CACHE.is_file():
        print(f"[ElementGradNet v2] Loading cached closed-region tiles from {ELEM_CACHE}...", flush=True)
        d = np.load(ELEM_CACHE)
        tr_items = list(zip(d["tr_x"], d["tr_ye"], d["tr_we"]))
        va_items = list(zip(d["va_x"], d["va_ye"], d["va_we"]))
        print(f"[ElementGradNet v2] Loaded cached tiles: train={len(tr_items)}, val={len(va_items)}", flush=True)
        return tr_items, va_items

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    holdout_ids = {2, 4}  # test1, test12 held out
    val_ids = {7, 14, 19, 27, 33, 41, 46}
    e_mpath = CACHE_DIR / "e_gallery" / "e_manifest.json"
    e_manifest = json.loads(e_mpath.read_text(encoding="utf-8")) if e_mpath.is_file() else []

    jobs = []
    for item in manifest:
        if item["id"] in holdout_ids:
            continue
        jobs.append((item["rgb_path"], item["teacher_path"], item["style"], item["id"] in val_ids, 6, 2000 + item["id"]))
    for ei, item in enumerate(e_manifest):
        jobs.append((item["rgb_path"], item["teacher_path"], "mixed", (ei % 12 == 0), 2, 6000 + ei))

    tr_items, va_items = [], []
    t0 = time.time()
    print(f"[ElementGradNet v2] Building closed-region tiles from {len(jobs)} images (4 workers)...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for is_val, items in ex.map(_prep_closed_elem_item, jobs, chunksize=4):
            (va_items if is_val else tr_items).extend(items)
    print(f"[ElementGradNet v2] Tiles ready in {time.time() - t0:.1f}s: train={len(tr_items)}, val={len(va_items)}", flush=True)

    np.savez(
        ELEM_CACHE,
        tr_x=np.stack([x[0] for x in tr_items], axis=0),
        tr_ye=np.stack([x[1] for x in tr_items], axis=0),
        tr_we=np.stack([x[2] for x in tr_items], axis=0),
        va_x=np.stack([x[0] for x in va_items], axis=0),
        va_ye=np.stack([x[1] for x in va_items], axis=0),
        va_we=np.stack([x[2] for x in va_items], axis=0),
    )
    print(f"[ElementGradNet v2] Saved tile cache to {ELEM_CACHE}", flush=True)
    return tr_items, va_items


def train_grad_cnn():
    import gc
    _cap_vram(8.5)
    tr_items, va_items = _load_or_build_closed_elem_tiles()

    tr_loader = DataLoader(ClosedElemDataset(tr_items, True), batch_size=3, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(ClosedElemDataset(va_items, False), batch_size=3, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    net = ElementGradNet(base=48, out_ch=8).to(device)
    if ELEM_CKPT_IN.is_file():
        print(f"Warm-starting ElementGradNet from {ELEM_CKPT_IN}", flush=True)
        net.load_state_dict(torch.load(ELEM_CKPT_IN, map_location=device, weights_only=True))

    epochs = 12
    grad_accum = 3
    opt = torch.optim.AdamW(net.parameters(), lr=3.5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 1e9
    t_tr = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        tr_l = 0.0
        opt.zero_grad(set_to_none=True)
        n_steps = len(tr_loader)
        for step_i, (bx, by, bw) in enumerate(tr_loader):
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = net(bx)
            loss = _elem_loss(pred.float(), by, bw)
            tr_l += float(loss.item()) * len(bx)
            (loss / grad_accum).backward()
            if ((step_i + 1) % grad_accum == 0) or ((step_i + 1) == n_steps):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        sch.step()

        net.eval()
        va_l, va_e1_dice = 0.0, 0.0
        with torch.no_grad():
            for bx, by, bw in va_loader:
                bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(bx)
                pf = pred.float()
                l = _elem_loss(pf, by, bw)
                va_l += float(l.item()) * len(bx)
                inter = (pf[:, 1] * by[:, 1]).sum(dim=(1, 2))
                denom = (pf[:, 1] + by[:, 1]).sum(dim=(1, 2)) + 1e-4
                va_e1_dice += float((2.0 * inter / denom).mean().item()) * len(bx)
        va_l /= max(1, len(va_items))
        va_e1_dice /= max(1, len(va_items))
        peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        mark = ""
        if va_l < best_val:
            best_val = va_l
            torch.save(net.state_dict(), ELEM_CKPT_OUT)
            mark = " *best"
        print(f"Elem Ep {ep:02d}/{epochs} tr={tr_l/len(tr_items):.4f} val={va_l:.4f} E1_dice={va_e1_dice:.4f} peak={peak_gb:.2f}GB {time.time()-t_tr:.1f}s{mark}", flush=True)

    # Export ONNX
    net.load_state_dict(torch.load(ELEM_CKPT_OUT, map_location=device, weights_only=True))
    net.eval().cpu()
    dummy = torch.rand(1, 3, 512, 512)
    torch.onnx.export(
        net,
        dummy,
        str(ELEM_ONNX_OUT),
        input_names=["rgb"],
        output_names=["maps"],
        dynamic_axes={"rgb": {0: "b", 2: "h", 3: "w"}, "maps": {0: "b", 2: "h", 3: "w"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        pt_out = net(dummy).numpy()
    sess = ort.InferenceSession(str(ELEM_ONNX_OUT), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"rgb": dummy.numpy()})[0]
    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"Exported {ELEM_ONNX_OUT} ({ELEM_ONNX_OUT.stat().st_size / 1024:.1f} KB), PT-vs-ORT diff={diff:.2e}", flush=True)
    del net, opt, tr_loader, va_loader, tr_items, va_items
    gc.collect()
    torch.cuda.empty_cache()


def _synth_grad_seam_tiles(n_tiles: int = 160, sz: int = 384, seed: int = 777):
    """Synthesize hard-negative soft gradient slopes + subtle internal seam lines where the
    flanking gradient slope MUST have target line=0.00 (eliminating purple margin fog around seams)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)

    for _ in range(n_tiles):
        ang = float(rng.uniform(0, np.pi))
        ca, sa = np.cos(ang), np.sin(ang)
        u = (xx - sz * 0.5) * ca + (yy - sz * 0.5) * sa
        v = -(xx - sz * 0.5) * sa + (yy - sz * 0.5) * ca

        # Smooth base gradient field with cylindrical highlight (mimics stocking / skin / hair gradient)
        c1 = rng.uniform(0.18, 0.55, size=3).astype(np.float32)
        c2 = np.clip(c1 + rng.uniform(0.15, 0.42, size=3), 0, 0.96).astype(np.float32)
        t = 0.5 + 0.5 * np.tanh(v / rng.uniform(45.0, 120.0) + 0.25 * np.sin(u / rng.uniform(60.0, 140.0)))
        bg = c1[None, None, :] * (1.0 - t[..., None]) + c2[None, None, :] * t[..., None]

        tgt = np.zeros((9, sz, sz), np.float32)
        tgt[4] = 0.5
        tgt[5] = 0.5
        tgt[6] = 0.5
        tgt[7] = 0.5

        # 60% chance: draw 1 or 2 crisp curved seam/fold ink lines across the gradient with clean sharp centerline
        if rng.random() < 0.65:
            line_mask = np.zeros((sz, sz), np.float32)
            thick = float(rng.uniform(0.9, 1.8))
            cx0 = float(rng.uniform(sz * 0.2, sz * 0.8))
            amp = float(rng.uniform(12.0, 45.0))
            pts = []
            for yi in range(0, sz, 4):
                xi = cx0 + amp * np.sin((yi / sz) * np.pi * rng.uniform(1.0, 2.2))
                pts.append([int(round(xi)), yi])
            pts_a = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            # Soft shadow rim flanking one side of the ink line (this MUST NOT trigger purple line fog!)
            rim_side = np.exp(-0.5 * (np.clip(xx - cx0, 0, 40) / rng.uniform(8.0, 22.0)) ** 2) * rng.uniform(0.06, 0.16)
            bg = np.clip(bg - rim_side[..., None] * np.array([0.8, 0.85, 0.9], np.float32), 0.0, 1.0)
            # Draw antialiased dark seam line
            cv2.polylines(line_mask, [pts_a], False, 1.0, max(1, int(round(thick * 2))), lineType=cv2.LINE_AA)
            line_col = c1 * rng.uniform(0.18, 0.45)
            bg = bg * (1.0 - line_mask[..., None]) + line_col[None, None, :] * line_mask[..., None]

            sk = (line_mask > 0.35)
            ys_k, xs_k = np.where(sk)
            if len(ys_k) > 5:
                from scipy.spatial import cKDTree
                pts_yx = np.column_stack([ys_k.astype(np.float32), xs_k.astype(np.float32)])
                coords = np.column_stack([yy.ravel(), xx.ravel()])
                dist, idx = cKDTree(pts_yx).query(coords)
                dist = dist.reshape(sz, sz)
                idx = idx.reshape(sz, sz)
                ny = pts_yx[idx, 0]
                nx = pts_yx[idx, 1]
                prob = np.exp(-0.5 * (dist / max(0.75, thick * 0.72)) ** 2)
                prob[dist > 2.8] = 0.0
                active = dist <= 2.6
                tgt[0] = prob.astype(np.float32)
                tgt[4] = np.where(active, np.clip((nx - xx) / 3.0 + 0.5, 0.0, 1.0), 0.5)
                tgt[5] = np.where(active, np.clip((ny - yy) / 3.0 + 0.5, 0.0, 1.0), 0.5)
                tgt[8] = np.where(active, np.clip(thick / 3.2, 0.0, 1.0), 0.0)

        # Tiny realistic compression/dither noise
        if rng.random() < 0.5:
            bg = np.clip(bg + rng.normal(0.0, 0.004, size=bg.shape).astype(np.float32), 0.0, 1.0)
        x_u8 = np.rint(np.clip(bg, 0, 1) * 255).astype(np.uint8)
        y_u8 = np.rint(np.clip(tgt, 0, 1) * 255).astype(np.uint8)
        xs.append(x_u8)
        ys.append(y_u8)
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


class CleanLineDataset(Dataset):
    def __init__(self, svg_x, svg_y, v2_x, v2_y, synth_x, synth_y, train: bool = True):
        self.svg_x, self.svg_y = svg_x, svg_y
        self.v2_x, self.v2_y = v2_x, v2_y
        self.synth_x, self.synth_y = synth_x, synth_y
        self.train = train
        self.n_svg = len(svg_x)
        self.n_v2 = len(v2_x)
        self.n_syn = len(synth_x)

    def __len__(self):
        return self.n_svg + self.n_v2 + self.n_syn

    def __getitem__(self, i):
        if i < self.n_svg:
            x, y, gw, is_svg, is_syn = self.svg_x[i], self.svg_y[i], 1.10, True, False
        elif i < self.n_svg + self.n_v2:
            j = i - self.n_svg
            x, y, gw, is_svg, is_syn = self.v2_x[j], self.v2_y[j], 0.85, False, False
        else:
            k = i - self.n_svg - self.n_v2
            x, y, gw, is_svg, is_syn = self.synth_x[k], self.synth_y[k], 1.25, False, True

        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]; y = y[:, :, ::-1].copy()
                y[4] = 255 - y[4]; y[7] = 255 - y[7]
            if np.random.rand() < 0.5:
                x = x[::-1, :]; y = y[:, ::-1, :].copy()
                y[5] = 255 - y[5]; y[7] = 255 - y[7]
            x = np.ascontiguousarray(x)
            if is_svg and np.random.rand() < 0.5:
                if np.random.rand() < 0.6:
                    x = cv2.GaussianBlur(x, (0, 0), sigmaX=float(np.random.uniform(0.30, 0.65)))
                if np.random.rand() < 0.45:
                    ok_enc, buf = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(np.random.randint(85, 97))])
                    if ok_enc:
                        x = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            xf = x.astype(np.float32) / 255.0
            xf = np.clip(xf * float(np.random.uniform(0.93, 1.07)) + float(np.random.uniform(-0.025, 0.025)), 0.0, 1.0)
        else:
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
        yf = np.ascontiguousarray(y).astype(np.float32) / 255.0
        return (
            torch.from_numpy(xf).permute(2, 0, 1).contiguous(),
            torch.from_numpy(yf).contiguous(),
            torch.tensor(gw, dtype=torch.float32),
            torch.tensor(1.0 if is_syn else 0.0, dtype=torch.float32),
        )


def _clean_line_loss(pred: torch.Tensor, target: torch.Tensor, g_weight: torch.Tensor, is_syn: torch.Tensor):
    gw = g_weight.view(-1, 1, 1, 1)
    p0 = pred[:, 0:1].clamp(1e-4, 1.0 - 1e-4)
    t0 = target[:, 0:1]

    # Asymmetric false-positive suppression in zero-line margin zones (kills purple gradient fog!)
    zero_bg = (t0 < 0.05).float()
    near_rim = F.max_pool2d((t0 > 0.35).float(), kernel_size=13, stride=1, padding=6) * zero_bg
    # Heavy penalty on low-confidence purple fog (0.06 - 0.40) outside real strokes
    fog_fp = torch.relu(p0 - 0.05) * zero_bg
    w_map = (1.0 + 3.2 * (t0 > 0.35).float() + 3.5 * near_rim + 2.5 * is_syn.view(-1, 1, 1, 1) * zero_bg) * gw

    bce = -(t0 * torch.log(p0) + (1.0 - t0) * torch.log(1.0 - p0)) * w_map
    inter = (p0 * t0 * gw).sum(dim=(2, 3))
    denom = ((p0 + t0) * gw).sum(dim=(2, 3)) + 1e-4
    dice = (1.0 - (2.0 * inter / denom)).mean()
    l_fog = (fog_fp * fog_fp * (1.0 + 3.0 * near_rim)).mean()
    l_prob = 0.45 * bce.mean() + 0.45 * dice + 0.25 * (torch.abs(p0 - t0) * gw).mean() + 1.8 * l_fog

    l_cls = (torch.abs(pred[:, 1:4] - target[:, 1:4]) * gw).mean()
    active = (t0 > 0.25).float() * gw
    denom_a = active.sum() + 1.0
    l_off = (F.smooth_l1_loss(pred[:, 4:6], target[:, 4:6], reduction="none", beta=0.05) * active).sum() / denom_a
    l_ang = (F.smooth_l1_loss(pred[:, 6:8], target[:, 6:8], reduction="none", beta=0.05) * active).sum() / denom_a
    l_rad = (F.smooth_l1_loss(pred[:, 8:9], target[:, 8:9], reduction="none", beta=0.04) * active).sum() / denom_a

    # Keep angle weight 0 on synthetic tiles where tangent angle channel wasn't supervised
    syn_mask = (1.0 - is_syn).view(-1, 1, 1, 1)
    l_ang = (F.smooth_l1_loss(pred[:, 6:8], target[:, 6:8], reduction="none", beta=0.05) * active * syn_mask).sum() / (
        (active * syn_mask).sum() + 1.0
    )

    tot = 1.40 * l_prob + 0.85 * l_cls + 2.5 * l_off + 1.5 * l_ang + 1.4 * l_rad
    return tot, float(l_prob.item()), float(l_fog.item())


def train_line_cnn():
    import gc
    _cap_vram(8.5)
    d = np.load(T_CACHE / "teacher_tiles.npz")
    svg_tr_x, svg_tr_y, svg_va_x, svg_va_y = d["tr_x"], d["tr_y"], d["va_x"], d["va_y"]
    v2_tr_x = np.load(V2_CACHE / "tr_x.npy", mmap_mode="r")
    v2_tr_y = np.load(V2_CACHE / "tr_y.npy", mmap_mode="r")
    v2_va_x = np.load(V2_CACHE / "va_x.npy", mmap_mode="r")
    v2_va_y = np.load(V2_CACHE / "va_y.npy", mmap_mode="r")

    rng = np.random.default_rng(108)
    n_v2_use = min(len(v2_tr_x), max(len(svg_tr_x), 650))
    idx_v2 = rng.choice(len(v2_tr_x), size=n_v2_use, replace=False)
    v2_tr_xs = np.array(v2_tr_x[idx_v2])
    v2_tr_ys = np.array(v2_tr_y[idx_v2])
    v2_va_xs = np.array(v2_va_x[:64])
    v2_va_ys = np.array(v2_va_y[:64])

    print("[LineVectorNet v3] Generating synthetic gradient-seam hard negatives...", flush=True)
    syn_tr_x, syn_tr_y = _synth_grad_seam_tiles(n_tiles=220, sz=384, seed=777)
    syn_va_x, syn_va_y = _synth_grad_seam_tiles(n_tiles=32, sz=384, seed=999)

    tr_ds = CleanLineDataset(svg_tr_x, svg_tr_y, v2_tr_xs, v2_tr_ys, syn_tr_x, syn_tr_y, train=True)
    va_ds = CleanLineDataset(svg_va_x, svg_va_y, v2_va_xs, v2_va_ys, syn_va_x, syn_va_y, train=False)

    tr_loader = DataLoader(tr_ds, batch_size=3, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=3, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    net = LineVectorNet(base=48, out_ch=9).to(device)
    if LINE_CKPT_IN.is_file():
        print(f"Warm-starting LineVectorNet from {LINE_CKPT_IN}", flush=True)
        net.load_state_dict(torch.load(LINE_CKPT_IN, map_location=device, weights_only=True))

    epochs = 10
    grad_accum = 3
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-4, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 1e9
    t0 = time.time()
    print(f"[LineVectorNet v3] Fine-tuning margin fog suppression: svg={len(svg_tr_x)}, v2={n_v2_use}, syn={len(syn_tr_x)}, va={len(va_ds)}", flush=True)
    for ep in range(1, epochs + 1):
        net.train()
        tr_l, tr_f = 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        n_steps = len(tr_loader)
        for step_i, (bx, by, bgw, bsyn) in enumerate(tr_loader):
            bx, by, bgw, bsyn = bx.to(device), by.to(device), bgw.to(device), bsyn.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = net(bx)
            loss, _, lfog = _clean_line_loss(pred.float(), by, bgw, bsyn)
            tr_l += float(loss.item()) * len(bx)
            tr_f += lfog * len(bx)
            (loss / grad_accum).backward()
            if ((step_i + 1) % grad_accum == 0) or ((step_i + 1) == n_steps):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        sch.step()

        net.eval()
        va_l, va_p, va_f = 0.0, 0.0, 0.0
        with torch.no_grad():
            for bx, by, bgw, bsyn in va_loader:
                bx, by, bgw, bsyn = bx.to(device), by.to(device), bgw.to(device), bsyn.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(bx)
                l, lp, lfog = _clean_line_loss(pred.float(), by, bgw, bsyn)
                va_l += float(l.item()) * len(bx)
                va_p += lp * len(bx)
                va_f += lfog * len(bx)
        va_l /= max(1, len(va_ds))
        va_p /= max(1, len(va_ds))
        va_f /= max(1, len(va_ds))
        peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        mark = ""
        if va_l < best_val:
            best_val = va_l
            torch.save(net.state_dict(), LINE_CKPT_OUT)
            mark = " *best"
        print(f"Line Ep {ep:02d}/{epochs} tr={tr_l/len(tr_ds):.4f} val={va_l:.4f} (prob={va_p:.4f}, fog={va_f:.5f}) peak={peak_gb:.2f}GB {time.time()-t0:.1f}s{mark}", flush=True)

    net.load_state_dict(torch.load(LINE_CKPT_OUT, map_location=device, weights_only=True))
    net.eval().cpu()
    dummy = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        net,
        dummy,
        str(LINE_ONNX_OUT),
        input_names=["rgb"],
        output_names=["out"],
        dynamic_axes={"rgb": {0: "batch", 2: "height", 3: "width"}, "out": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        pt_out = net(dummy).numpy()
    sess = ort.InferenceSession(str(LINE_ONNX_OUT), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"rgb": dummy.numpy()})[0]
    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"Exported {LINE_ONNX_OUT} ({LINE_ONNX_OUT.stat().st_size / 1024:.1f} KB), PT-vs-ORT diff={diff:.2e}", flush=True)
    del net, opt, tr_loader, va_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--preview" in args or not args:
        run_preview()
    if "--train-grad" in args or "--train-all" in args:
        train_grad_cnn()
    if "--train-line" in args or "--train-all" in args:
        train_line_cnn()
