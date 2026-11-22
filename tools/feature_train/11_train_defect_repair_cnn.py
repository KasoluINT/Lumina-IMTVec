"""Phase 5: Train Feature-Guided Pre-SVG Defect Detection & Restoration CNN (`DefectRepairNet`).

Trains `weights/defect_repair_net.onnx` (`base=40`, 7-channel input: RGB + FeaturePartUNet 4 channels ->
4-channel output: [defect_prob, clean_R, clean_G, clean_B]).

Combines:
  1) Real artwork mining across `e_gallery` + `v2_gallery` + Tarot using surgical dark-sandwiched
     sliver detection protected by `FeaturePartUNet.delicate_detail`.
  2) Exact synthetic defect injection simulating the user's 6 visual criteria:
     - Hazy, semi-transparent pale-pink / off-white diagonal & curved bright slivers ('亮棱')
       with soft blurred edges, uneven brightness ripples, overlaid over dark ink / borders.
     - Trapped dim scratches & tiny bright dust specks pinched inside dark pockets.
     - Hard negative preservation (eye catchlights, white beads, fine hair wisps -> defect=0).

Usage:
    python tools/feature_train/11_train_defect_repair_cnn.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lumina_vec.color import srgb_to_oklab
from lumina_vec.feature_nets import DefectRepairNet
from lumina_vec.feature_routing import _SESSION_CACHE, _get_session, _run_dense_unet

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
DEF_CACHE = CACHE_DIR / "defect_tiles_cache"
WEIGHTS_DIR = ROOT / "weights"
CKPT_OUT = CACHE_DIR / "defect_repair_best.pt"
ONNX_OUT = WEIGHTS_DIR / "defect_repair_net.onnx"
ART_DIR = Path(r"C:\Users\30266\.gemini\antigravity\brain\439cca1c-a31d-4f73-81c2-1f9c82b1e720")
TAROT_IMG = Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图集2\图集\2ED9A5F713EFAA837875DB903C72AA94.png")


def _opposite_dark_sandwich_blocked(dark_wall: np.ndarray, blocker: np.ndarray, max_r: int = 5) -> np.ndarray:
    h, w = dark_wall.shape
    pad = max_r
    dp = np.pad(dark_wall, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
    bp = np.pad(blocker, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
    sandwiched = np.zeros((h, w), dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        pos_hit = np.zeros((h, w), dtype=bool)
        neg_hit = np.zeros((h, w), dtype=bool)
        pos_alive = np.ones((h, w), dtype=bool)
        neg_alive = np.ones((h, w), dtype=bool)
        for step in range(1, max_r + 1):
            sy, sx = dy * step, dx * step
            b_p = bp[pad + sy : pad + sy + h, pad + sx : pad + sx + w]
            d_p = dp[pad + sy : pad + sy + h, pad + sx : pad + sx + w]
            pos_hit |= (pos_alive & d_p & (~b_p))
            pos_alive &= (~b_p)

            b_n = bp[pad - sy : pad - sy + h, pad - sx : pad - sx + w]
            d_n = dp[pad - sy : pad - sy + h, pad - sx : pad - sx + w]
            neg_hit |= (neg_alive & d_n & (~b_n))
            neg_alive &= (~b_n)
        sandwiched |= (pos_hit & neg_hit)
    return sandwiched


def detect_and_repair_teacher(
    rgb: np.ndarray,
    parts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Surgical real-artwork teacher identifying hazy dark-sandwiched bright slivers & trapped
    micro-cavities while protecting delicate details, white beads, and solid colored facets."""
    h, w = rgb.shape[:2]
    ok = srgb_to_oklab(rgb).astype(np.float32)
    L = ok[..., 0]
    C = np.hypot(ok[..., 1], ok[..., 2])

    prot = np.zeros((h, w), dtype=bool)
    if parts is not None:
        # Protect delicate eye catchlights/details strongly, plus gently shield hair highlights if weak
        prot = (parts[2] > 0.32)

    dark = L < 0.23
    dark_wall = (L < 0.23) | ((L < 0.29) & (C < 0.08))
    dt_nd = cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3)
    near_bright = cv2.dilate((L >= 0.72).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0

    thick_bright = (dt_nd > 2.0) & (L >= 0.68)
    bright_hi = (L >= 0.72) | thick_bright
    _, lb, _, _ = cv2.connectedComponentsWithStats(bright_hi.astype(np.uint8), connectivity=8)
    bright_strut = np.zeros((h, w), dtype=bool)
    thick_labels = set(np.unique(lb[thick_bright])) - {0}
    if thick_labels:
        bright_strut = np.isin(lb, list(thick_labels))

    thick_color = (dt_nd > 2.45) & (L >= 0.25)
    solid_facet = bright_strut | thick_color

    thick_bright_rim = (cv2.dilate(bright_strut.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~dark)
    thick_color_rim = (cv2.dilate(thick_color.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (L >= 0.32)
    edge_prot = solid_facet | thick_bright_rim | thick_color_rim

    sandwiched = _opposite_dark_sandwich_blocked(dark_wall, solid_facet, max_r=5)
    sliver_cand = sandwiched & (~dark) & (L < 0.80) & near_bright & (~edge_prot) & (~prot)

    dark2 = L < 0.27
    non_dark2 = (~dark2).astype(np.uint8)
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(non_dark2, connectivity=4)
    dt_nd2 = cv2.distanceTransform(non_dark2, cv2.DIST_L2, 3)
    enclosed_sliver = np.zeros((h, w), dtype=bool)
    for hi in range(1, nh):
        area = int(sh[hi, cv2.CC_STAT_AREA])
        if area < 2 or area > 190:
            continue
        xb, yb, wb, hb = [int(sh[hi, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        if xb <= 0 or yb <= 0 or xb + wb >= w or yb + hb >= h:
            continue
        y0, y1, x0, x1 = max(0, yb - 2), min(h, yb + hb + 2), max(0, xb - 2), min(w, xb + wb + 2)
        m = lh[y0:y1, x0:x1] == hi
        if prot[y0:y1, x0:x1][m].mean() > 0.20 or solid_facet[y0:y1, x0:x1][m].any():
            continue
        max_r = float(dt_nd2[y0:y1, x0:x1][m].max())
        aspect = max(wb, hb) / max(1.0, min(wb, hb))
        if (max_r <= 1.15 and area <= 24) or (max_r <= 2.15 and aspect >= 1.9 and area <= 160):
            enclosed_sliver[y0:y1, x0:x1][m] = True

    nt, lt, st, _ = cv2.connectedComponentsWithStats(sliver_cand.astype(np.uint8), connectivity=8)
    sliver_clean = np.zeros((h, w), dtype=bool)
    for ti in range(1, nt):
        if st[ti, cv2.CC_STAT_AREA] >= 5:
            sliver_clean[lt == ti] = True

    defect = sliver_clean | enclosed_sliver
    defect_dil = (cv2.dilate(defect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~edge_prot) & (~prot) & (L < 0.78)

    out = rgb.copy()
    mask_f = defect_dil.astype(np.float32)
    if defect_dil.any():
        valid_dark = (L <= 0.19) & (~defect_dil)
        if not valid_dark.any():
            valid_dark = (L < 0.25) & (~defect_dil)
        if valid_dark.any():
            _, idx = distance_transform_edt(~valid_dark, return_indices=True)
            dark_fill = out[idx[0], idx[1]].astype(np.float32)
        else:
            dark_fill = np.zeros_like(out, dtype=np.float32)
        dark_sm = cv2.GaussianBlur(dark_fill, (3, 3), 0.6)
        d_in = cv2.distanceTransform(defect_dil.astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(d_in / 1.0, 0.0, 1.0)
        out_f = out.astype(np.float32)
        out_f = np.where(defect_dil[..., None], alpha[..., None] * dark_sm + (1.0 - alpha[..., None]) * out_f, out_f)
        out = np.clip(np.rint(out_f), 0, 255).astype(np.uint8)
        mask_f = np.where(defect_dil, np.clip(0.45 + 0.55 * alpha, 0.0, 1.0), 0.0).astype(np.float32)

    return out, mask_f


def inject_synthetic_defects(
    clean_rgb: np.ndarray,
    parts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlay realistic hazy semi-transparent pale-pink / cream slivers ('亮棱') onto dark ink
    strokes/wedges of `clean_rgb`, following all 6 criteria, while avoiding `delicate_detail`.
    Returns (corrupted_rgb, gt_clean_rgb, gt_defect_mask_f32)."""
    h, w = clean_rgb.shape[:2]
    ok = srgb_to_oklab(clean_rgb).astype(np.float32)
    L = ok[..., 0]

    # Prefer placing synthetic slivers along/near dark ink outlines (L < 0.26) away from delicate_detail
    dark_elig = (L < 0.26) & (parts[2] < 0.25)
    # Dilate slightly so thin dark lines and adjacent edges also host diagonal sliver overlays
    ink_zone = cv2.dilate(dark_elig.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.where(ink_zone)
    if len(ys) < 40:
        # Fallback: place random slanted soft sliver over any non-delicate dark pixel or skip
        return clean_rgb.copy(), clean_rgb.copy(), np.zeros((h, w), np.float32)

    alpha_map = np.zeros((h, w), dtype=np.float32)
    sliver_color = np.zeros((h, w, 3), dtype=np.float32)
    n_defects = int(rng.integers(3, 9))

    for _ in range(n_defects):
        pick = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[pick]), int(xs[pick])
        angle = float(rng.uniform(0.0, np.pi))
        length = float(rng.uniform(12.0, 48.0))
        dx = float(np.cos(angle) * length * 0.5)
        dy = float(np.sin(angle) * length * 0.5)
        p0 = (int(round(cx - dx)), int(round(cy - dy)))
        p1 = (int(round(cx + dx)), int(round(cy + dy)))
        # Curved 3-point bend occasionally
        mx = int(round(cx + float(rng.uniform(-4.5, 4.5))))
        my = int(round(cy + float(rng.uniform(-4.5, 4.5))))
        pts = np.array([p0, (mx, my), p1], dtype=np.int32)

        thick = int(rng.integers(1, 4))
        tmp = np.zeros((h, w), dtype=np.float32)
        cv2.polylines(tmp, [pts], False, 1.0, thickness=thick, lineType=cv2.LINE_AA)

        # Hazy gradient edge blur (Criterion 2 & 4: no sharp hard edge, gradual fade)
        sig = float(rng.uniform(0.65, 1.45))
        tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=sig)
        if tmp.max() > 1e-4:
            tmp /= max(float(tmp.max()), 1.0)

        # Uneven internal pixel intensity fluctuation (Criterion 6)
        ripple = 0.78 + 0.36 * cv2.GaussianBlur(rng.uniform(0.0, 1.0, (h, w)).astype(np.float32), (0, 0), 1.4)
        peak_alpha = float(rng.uniform(0.32, 0.78))
        # Restrict mainly to dark-to-transition neighborhoods so we simulate slivers overlaid on dark lines/wedges
        local_dark = cv2.dilate((L < 0.35).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        m_a = np.clip(tmp * ripple * peak_alpha, 0.0, 0.88) * local_dark.astype(np.float32)
        m_a *= (parts[2] < 0.30).astype(np.float32)

        # Pale pink / off-white / maroon-tinted sliver RGB (Criterion 1)
        if rng.random() < 0.72:
            # Pale pink / cream off-white
            col = np.array([
                float(rng.uniform(210, 255)),
                float(rng.uniform(160, 238)),
                float(rng.uniform(175, 245)),
            ], dtype=np.float32)
        else:
            # Soft maroon / dusty rose haze
            col = np.array([
                float(rng.uniform(130, 195)),
                float(rng.uniform(75, 125)),
                float(rng.uniform(85, 145)),
            ], dtype=np.float32)

        better = m_a > alpha_map
        alpha_map = np.maximum(alpha_map, m_a)
        sliver_color[better] = col

    # Also optionally inject 1-2 tiny enclosed bright pinpoint specks over dark lines
    for _ in range(int(rng.integers(0, 4))):
        pick = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[pick]), int(xs[pick])
        if L[cy, cx] < 0.20 and parts[2, cy, cx] < 0.20:
            tmp = np.zeros((h, w), dtype=np.float32)
            cv2.circle(tmp, (cx, cy), int(rng.integers(1, 3)), 1.0, -1, lineType=cv2.LINE_AA)
            tmp = cv2.GaussianBlur(tmp, (0, 0), 0.7) * float(rng.uniform(0.45, 0.85))
            col = np.array([rng.uniform(215, 255), rng.uniform(180, 245), rng.uniform(195, 250)], dtype=np.float32)
            better = tmp > alpha_map
            alpha_map = np.maximum(alpha_map, tmp)
            sliver_color[better] = col

    corrupted = clean_rgb.astype(np.float32) * (1.0 - alpha_map[..., None]) + sliver_color * alpha_map[..., None]
    corrupted_u8 = np.clip(np.rint(corrupted), 0, 255).astype(np.uint8)
    gt_mask = np.clip(alpha_map / 0.55, 0.0, 1.0).astype(np.float32)
    return corrupted_u8, clean_rgb.copy(), gt_mask


def build_or_load_dataset(tile_sz: int = 384):
    DEF_CACHE.mkdir(parents=True, exist_ok=True)
    cache_p = DEF_CACHE / "defect_dataset.npz"
    if cache_p.is_file():
        print(f"Loading cached defect dataset from {cache_p}...", flush=True)
        d = np.load(cache_p)
        return {k: d[k] for k in d.files}

    _SESSION_CACHE.clear()
    sess_part = _get_session(WEIGHTS_DIR / "feature_part.onnx")
    if sess_part is None:
        raise RuntimeError("Missing weights/feature_part.onnx")

    e_man = json.loads((CACHE_DIR / "e_gallery" / "e_manifest.json").read_text(encoding="utf-8"))
    v2_man = json.loads((CACHE_DIR / "v2_gallery" / "v2_manifest.json").read_text(encoding="utf-8"))
    e_sr_dir = CACHE_DIR / "e_gallery" / "sr_2x"
    v2_sr_dir = CACHE_DIR / "v2_gallery" / "sr_2x"

    paths_2x = []
    for vi, item in enumerate(v2_man):
        p = Path(item["rgb_path"]) if item.get("kind") == "svg_2x" else (v2_sr_dir / f"v2_{vi:03d}.png")
        if p.is_file():
            paths_2x.append(p)
    for ei, item in enumerate(e_man):
        n2 = item.get("native_2x_path")
        p = Path(n2) if (n2 and Path(n2).is_file()) else (e_sr_dir / f"e3_{ei:03d}.png")
        if p.is_file():
            paths_2x.append(p)
        if len(paths_2x) >= 195:
            break

    print(f"Mining FeaturePart-guided defect tiles across {len(paths_2x)} artworks...", flush=True)
    xs_rgb, xs_parts, ys_clean, ys_mask, is_val = [], [], [], [], []
    rng = np.random.default_rng(777)
    t0 = time.time()

    for idx, p in enumerate(paths_2x):
        sr_2x = np.asarray(Image.open(p).convert("RGB"))
        h2, w2 = sr_2x.shape[:2]
        if min(h2, w2) < tile_sz:
            continue
        # Run FeaturePartUNet at 1x resolution matching sr_clean_1x semantics, then upsample to 2x
        img_1x = cv2.resize(sr_2x, (max(64, w2 // 2), max(64, h2 // 2)), interpolation=cv2.INTER_AREA)
        parts_1x = _run_dense_unet(sess_part, img_1x, out_ch=4, hi_weight=0.70)
        parts_2x = np.stack([cv2.resize(parts_1x[c], (w2, h2), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=0)

        for ci in range(4):
            y0 = int(rng.integers(0, h2 - tile_sz + 1))
            x0 = int(rng.integers(0, w2 - tile_sz + 1))
            crop = sr_2x[y0:y0 + tile_sz, x0:x0 + tile_sz]
            pcrop = parts_2x[:, y0:y0 + tile_sz, x0:x0 + tile_sz]
            if float((pcrop[0] > 0.30).mean()) < 0.01:
                continue

            # Clean real defects first so clean base is pure, recording real mined sample when non-trivial
            clean_real, mask_real = detect_and_repair_teacher(crop, parts=pcrop)
            p_u8 = np.rint(np.clip(pcrop, 0, 1) * 255).astype(np.uint8)
            val_flag = 1 if (idx % 12 == 0 and ci == 0) else 0

            if float((mask_real > 0.3).sum()) >= 20:
                xs_rgb.append(crop)
                xs_parts.append(p_u8)
                ys_clean.append(clean_real)
                ys_mask.append(np.rint(np.clip(mask_real, 0, 1) * 255).astype(np.uint8))
                is_val.append(val_flag)

            # Synthetic defect injection on cleaned real artwork (78% positive defect, 22% hard negative!)
            if rng.random() < 0.78:
                corr, tgt_c, tgt_m = inject_synthetic_defects(clean_real, pcrop, rng)
            else:
                corr, tgt_c, tgt_m = clean_real, clean_real, np.zeros((tile_sz, tile_sz), np.float32)

            xs_rgb.append(corr)
            xs_parts.append(p_u8)
            ys_clean.append(tgt_c)
            ys_mask.append(np.rint(np.clip(tgt_m, 0, 1) * 255).astype(np.uint8))
            is_val.append(val_flag)

        if (idx + 1) % 35 == 0:
            print(f"  processed {idx+1}/{len(paths_2x)} artworks (tiles={len(xs_rgb)}, {time.time()-t0:.1f}s)", flush=True)

    _SESSION_CACHE.clear()
    out = {
        "xs_rgb": np.stack(xs_rgb, axis=0),
        "xs_parts": np.stack(xs_parts, axis=0),
        "ys_clean": np.stack(ys_clean, axis=0),
        "ys_mask": np.stack(ys_mask, axis=0),
        "is_val": np.array(is_val, dtype=np.uint8),
    }
    np.savez(cache_p, **out)
    print(f"Saved defect dataset to {cache_p}: total={len(xs_rgb)}, val={int(out['is_val'].sum())}", flush=True)
    return out


class DefectDataset(Dataset):
    def __init__(self, d: dict, indices: np.ndarray, train: bool = True):
        self.xr = d["xs_rgb"][indices]
        self.xp = d["xs_parts"][indices]
        self.yc = d["ys_clean"][indices]
        self.ym = d["ys_mask"][indices]
        self.train = train

    def __len__(self):
        return len(self.xr)

    def __getitem__(self, i):
        xr = self.xr[i]
        xp = self.xp[i]
        yc = self.yc[i]
        ym = self.ym[i]
        if self.train:
            if np.random.rand() < 0.5:
                xr = xr[:, ::-1]; yc = yc[:, ::-1]
                xp = xp[:, :, ::-1]; ym = ym[:, ::-1]
            if np.random.rand() < 0.5:
                xr = xr[::-1, :]; yc = yc[::-1, :]
                xp = xp[:, ::-1, :]; ym = ym[::-1, :]
            k = int(np.random.randint(0, 4))
            if k:
                xr = np.rot90(xr, k); yc = np.rot90(yc, k)
                xp = np.rot90(xp, k, axes=(1, 2)); ym = np.rot90(ym, k)

        in_7 = np.concatenate([
            np.ascontiguousarray(xr).astype(np.float32).transpose(2, 0, 1) / 255.0,
            np.ascontiguousarray(xp).astype(np.float32) / 255.0,
        ], axis=0)
        tgt_c = np.ascontiguousarray(yc).astype(np.float32).transpose(2, 0, 1) / 255.0
        tgt_m = np.ascontiguousarray(ym).astype(np.float32)[None] / 255.0
        return torch.from_numpy(in_7), torch.from_numpy(tgt_c), torch.from_numpy(tgt_m)


def defect_loss(pred: torch.Tensor, tgt_rgb: torch.Tensor, tgt_m: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    pm = pred[:, 0:1].clamp(1e-4, 1.0 - 1e-4)
    prgb = pred[:, 1:4]

    # Weighted BCE + soft Dice for sparse defect masks
    pos_w = 1.0 + 4.5 * (tgt_m > 0.15).float()
    bce = (-(tgt_m * torch.log(pm) + (1.0 - tgt_m) * torch.log(1.0 - pm)) * pos_w).mean()
    inter = (pm * tgt_m).sum(dim=(2, 3))
    denom = (pm + tgt_m).sum(dim=(2, 3)) + 1e-3
    dice = (1.0 - (2.0 * inter / denom)).mean()
    l_mask = 0.55 * bce + 0.45 * dice

    # Clean RGB restoration loss: 12x weighted on defect pixels, 0.35x identity stability outside
    w_rgb = 0.35 + 12.0 * tgt_m
    l_rgb = (torch.abs(prgb - tgt_rgb) * w_rgb).mean()
    return l_mask + 1.8 * l_rgb, float(l_mask.item()), float(l_rgb.item())


def main():
    d = build_or_load_dataset(tile_sz=384)
    val_m = d["is_val"] == 1
    tr_idx = np.where(~val_m)[0]
    va_idx = np.where(val_m)[0]
    if len(va_idx) < 10:
        va_idx = tr_idx[:40]
        tr_idx = tr_idx[40:]

    tr_loader = DataLoader(DefectDataset(d, tr_idx, train=True), batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(DefectDataset(d, va_idx, train=False), batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    net = DefectRepairNet(base=40).to(device)
    epochs = 16
    opt = torch.optim.AdamW(net.parameters(), lr=9e-4, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 1e9
    t0 = time.time()
    print(f"Training DefectRepairNet(base=40): train={len(tr_idx)}, val={len(va_idx)} tiles...", flush=True)
    for ep in range(1, epochs + 1):
        net.train()
        tr_l = 0.0
        for bx, bc, bm in tr_loader:
            bx, bc, bm = bx.to(device), bc.to(device), bm.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = net(bx)
            loss, _, _ = defect_loss(pred.float(), bc, bm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tr_l += float(loss.item()) * len(bx)
        sch.step()

        net.eval()
        va_l, va_m, va_r = 0.0, 0.0, 0.0
        with torch.no_grad():
            for bx, bc, bm in va_loader:
                bx, bc, bm = bx.to(device), bc.to(device), bm.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(bx)
                l, lm, lr_ = defect_loss(pred.float(), bc, bm)
                va_l += float(l.item()) * len(bx)
                va_m += lm * len(bx)
                va_r += lr_ * len(bx)
        va_l /= max(1, len(va_idx))
        va_m /= max(1, len(va_idx))
        va_r /= max(1, len(va_idx))
        mark = ""
        if va_l < best_val:
            best_val = va_l
            torch.save(net.state_dict(), CKPT_OUT)
            mark = " *best"
        print(
            f"Ep {ep:02d}/{epochs} tr={tr_l/len(tr_idx):.4f} val={va_l:.4f} "
            f"(mask={va_m:.4f}, rgb={va_r:.4f}) {time.time()-t0:.1f}s{mark}",
            flush=True,
        )

    # Export ONNX
    net.load_state_dict(torch.load(CKPT_OUT, map_location=device, weights_only=True))
    net.eval().cpu()
    dummy = torch.randn(1, 7, 512, 512)
    torch.onnx.export(
        net,
        dummy,
        str(ONNX_OUT),
        input_names=["feat_in"],
        output_names=["out"],
        dynamic_axes={"feat_in": {0: "batch", 2: "height", 3: "width"}, "out": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        pt_out = net(dummy).numpy()
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"feat_in": dummy.numpy()})[0]
    diff = float(np.max(np.abs(pt_out - ort_out)))
    kb = ONNX_OUT.stat().st_size / 1024.0
    print(f"Exported {ONNX_OUT} ({kb:.1f} KB), PT-vs-ORT max diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
