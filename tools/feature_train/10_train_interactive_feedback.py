"""Interactive Human-in-the-Loop Positive-Feedback Trainer for LineVectorNet (v4.2+).

Features:
  1. Unified v4.2 Teacher: preserves 100% of v2's complete outer contours & delicate hairlines,
     collapses uniform-dark Rosette Y-wedges into crisp single lines without bridging beaded rings,
     and swallows tiny trapped highlight slivers (Prism bevel) into unified clean strokes.
  2. Interactive Desktop Picker with Mouse-Wheel Zoom (up to 6x NEAREST), Left-Drag Pan,
     Single-Row Full-Width Focus buttons (Prism / Rosette / Hair / test12), Heatmap/Scatter toggle,
     and directional "Retry this stage" or "Advance with positive feedback" actions.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset
import tkinter as tk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "feature_train"))

from lumina_vec.color import srgb_to_oklab
from lumina_vec.feature_nets import LineVectorNet
from lumina_vec.feature_routing import _SESSION_CACHE, _get_session, _upscale_sr, _WEIGHTS_DEFAULT

m09 = importlib.import_module("09_train_line_vector_cnn")

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
INTER_DIR = CACHE_DIR / "interactive_v43"
VEC_V2_CKPT = CACHE_DIR / "backup_pre_v4" / "ckpt" / "line_vector_v2_best.pt"
VEC_V4_CKPT = CACHE_DIR / "line_vector_v4_best.pt"
VEC_ONNX = ROOT / "weights" / "line_vector_net.onnx"

TAROT_IMG = Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图集2\图集\2ED9A5F713EFAA837875DB903C72AA94.png")
TEST12_IMG = Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图片集\test12.jpg")

OFFSET_SCALE = 8.0
RAD_SCALE = 6.0


def _build_v43_targets(
    sr_2x: np.ndarray,
    vf2_2x: np.ndarray,
    vf4_2x: np.ndarray,
    collapse_slivers: bool = True,
    sliver_max_r: float = 1.45,
    bh_thr: float = 0.09,
    dt_hi: float = 3.55,
) -> np.ndarray:
    """Build v4.3 CleanFuse teacher [9, H, W]:
    1) Keeps 100% of v2's complete outer contours, beaded rings, purple lattice & delicate hairlines.
    2) Erases faint trapped dim scratches inside bright-bordered dark wedges (Prism V-hairpin maroon streak)
       while keeping Rosette outer purple rings unbroken (near_bright_facet guard).
    3) Smoothly fuses crisp single-line v4 predictions exclusively inside genuine double-wall hollow Y-hubs
       (Rosette inner star), blending all 9 channels (prob, dir, offset, rad) in [OFFSET_SCALE=8, RAD_SCALE=6].
    4) Optionally swallows sub-pixel enclosed highlight slivers without merging crystal facets."""
    h, w = sr_2x.shape[:2]
    sm = cv2.bilateralFilter(sr_2x, 3, 10, 3)
    L = srgb_to_oklab(sm)[..., 0].astype(np.float32)
    p2 = vf2_2x[0].copy()
    p4 = vf4_2x[0].copy()

    # Rescale v2 offsets/radii into v4 target space (v2 used 3.0 / 3.2; v4 uses 8.0 / 6.0)
    dx2 = (vf2_2x[4] - 0.5) * 3.0
    dy2 = (vf2_2x[5] - 0.5) * 3.0
    rad2 = vf2_2x[8] * 3.2
    dx4 = (vf4_2x[4] - 0.5) * OFFSET_SCALE
    dy4 = (vf4_2x[5] - 0.5) * OFFSET_SCALE
    rad4 = vf4_2x[8] * RAD_SCALE

    # 1. Scratch kill inside dark wedges (Prism V-hairpin maroon streak):
    # Close strictly dark mask (L < 0.22) with k9 -> faint trapped streaks (L < 0.42) inside bright-bordered
    # dark wedges (near L >= 0.62) are cleared cleanly; outer Rosette dim-purple arcs stay untouched.
    dark_dim = L < 0.22
    k9e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark_closed = cv2.morphologyEx(dark_dim.astype(np.uint8), cv2.MORPH_CLOSE, k9e) > 0
    trapped_raw = dark_closed & (~dark_dim) & (L < 0.42)
    bright_wall = cv2.dilate((L >= 0.62).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    trapped_filt = np.zeros_like(trapped_raw)
    nt, lt, st, _ = cv2.connectedComponentsWithStats((trapped_raw & (~bright_wall)).astype(np.uint8), connectivity=8)
    for ti in range(1, nt):
        if st[ti, cv2.CC_STAT_AREA] >= 3:
            trapped_filt[lt == ti] = True
    near_bright_facet = cv2.dilate((L >= 0.62).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    scratch_kill = (cv2.dilate(trapped_filt.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~bright_wall) & near_bright_facet
    p2_sk = np.where(scratch_kill, np.minimum(p2, 0.03), p2)

    # 2. Deep dark void exclusion (outer dark background & Rosette outer bead gap)
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark_void_core = cv2.dilate(L, k15) < 0.22
    inside_dark_void = (L < 0.23) & (cv2.dilate(dark_void_core.astype(np.uint8), k9) > 0)

    # 3. Black-hat for narrow dark corridors
    bh_max = np.zeros((h, w), dtype=np.float32)
    for diam in (5, 7):
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
        closed_L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, ke)
        bh_r = np.maximum(0.0, closed_L - L)
        bh_r = np.where(inside_dark_void, 0.0, bh_r)
        bh_max = np.maximum(bh_max, bh_r)
    dark_thick = (bh_max >= bh_thr) & (L < 0.20) & (~inside_dark_void)

    # 4. Tiny enclosed pockets (dark hubs + optional sub-pixel bright slivers)
    r2_sk = p2_sk >= 0.34
    inv2 = (~r2_sk).astype(np.uint8)
    n_h, lbl_h, st_h, _ = cv2.connectedComponentsWithStats(inv2, connectivity=4)
    dt_h = cv2.distanceTransform(inv2, cv2.DIST_L2, 3)
    sliver_fill = np.zeros((h, w), dtype=bool)
    dark_fill = np.zeros((h, w), dtype=bool)
    for hi in range(1, n_h):
        area = int(st_h[hi, cv2.CC_STAT_AREA])
        if area < 2 or area > 240:
            continue
        xb, yb, wb, hb = [int(st_h[hi, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        y0, y1, x0, x1 = max(0, yb - 4), min(h, yb + hb + 4), max(0, xb - 4), min(w, xb + wb + 4)
        m = lbl_h[y0:y1, x0:x1] == hi
        max_r = float(dt_h[y0:y1, x0:x1][m].max())
        mean_L = float(L[y0:y1, x0:x1][m].mean())
        if mean_L < 0.19 and max_r <= 3.6 and area <= 180:
            dark_fill[y0:y1, x0:x1][m] = True
        elif collapse_slivers and sliver_max_r > 0 and max_r <= sliver_max_r and area <= 85 and mean_L < 0.72:
            sliver_fill[y0:y1, x0:x1][m] = True

    # 5. Selective v4 fusion exclusively inside hollow double-wall Y-corridors
    collapse_body = dark_thick | dark_fill
    dt_body = cv2.distanceTransform((r2_sk | collapse_body).astype(np.uint8), cv2.DIST_L2, 5)
    corridor_cand = (dt_body >= 1.45) & (dt_body <= dt_hi) & collapse_body
    hollow_core = corridor_cand & (p2_sk < 0.24) & (p4 >= 0.34)
    sk_excl = cv2.dilate(scratch_kill.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hollow_core = hollow_core & (~sk_excl)
    n_hc, lbl_hc, st_hc, _ = cv2.connectedComponentsWithStats(hollow_core.astype(np.uint8), connectivity=8)
    hollow_valid = np.zeros_like(hollow_core)
    for ci in range(1, n_hc):
        if st_hc[ci, cv2.CC_STAT_AREA] >= 6:
            hollow_valid[lbl_hc == ci] = True

    hollow_touch = cv2.dilate(hollow_valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    corr_mask = (dt_body >= 1.35) & (dt_body <= dt_hi + 0.3) & collapse_body
    n_cm, lbl_cm, _, _ = cv2.connectedComponentsWithStats(corr_mask.astype(np.uint8), connectivity=8)
    fuse_seed = np.zeros_like(corr_mask)
    for ci in range(1, n_cm):
        m_c = lbl_cm == ci
        if (m_c & hollow_touch).any():
            fuse_seed[m_c] = True
    fuse_zone = (cv2.dilate(fuse_seed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (~sk_excl) & (~inside_dark_void)

    prob_out = p2_sk.copy()
    dx = dx2.copy()
    dy = dy2.copy()
    rad = rad2.copy()
    ch13 = vf2_2x[1:4].copy()
    ch68 = vf2_2x[6:8].copy()

    if fuse_zone.any():
        p4_boost = np.clip(np.power(np.clip(p4, 0.0, 1.0), 0.78) * 1.12, 0.0, 1.0)
        bdist = cv2.distanceTransform(fuse_zone.astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(bdist / 2.2, 0.0, 1.0)
        a_f = np.where(fuse_zone, alpha, 0.0)
        prob_out = a_f * p4_boost + (1.0 - a_f) * p2_sk
        dx = a_f * dx4 + (1.0 - a_f) * dx2
        dy = a_f * dy4 + (1.0 - a_f) * dy2
        rad = a_f * rad4 + (1.0 - a_f) * rad2
        ch13 = a_f[None] * vf4_2x[1:4] + (1.0 - a_f[None]) * vf2_2x[1:4]
        ch68 = a_f[None] * vf4_2x[6:8] + (1.0 - a_f[None]) * vf2_2x[6:8]

    # 6. Swallow tiny sub-pixel enclosed bright slivers away from V-hairpin scratch zone
    if sliver_fill.any():
        sf_dil = (cv2.dilate(sliver_fill.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~sk_excl)
        sf_2x = cv2.resize((sf_dil & (r2_sk | sliver_fill)).astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_LINEAR) >= 0.45
        sk_sf = skeletonize(sf_2x)
        sy, sx = np.where(sk_sf)
        if len(sy) > 0:
            s_pts = np.column_stack([sy.astype(np.float32) * 0.5, sx.astype(np.float32) * 0.5])
            cy_s, cx_s = np.where(sf_dil)
            if len(cy_s) > 0:
                d_s, idx_s = cKDTree(s_pts).query(np.column_stack([cy_s.astype(np.float32), cx_s.astype(np.float32)]), workers=1)
                p_s = np.exp(-0.5 * (d_s / 0.85) ** 2)
                prob_out[cy_s, cx_s] = np.maximum(prob_out[cy_s, cx_s] * 0.35, p_s)
                in_s = d_s <= 2.0
                dx[cy_s[in_s], cx_s[in_s]] = np.clip(s_pts[idx_s[in_s], 1] - cx_s[in_s], -4.0, 4.0)
                dy[cy_s[in_s], cx_s[in_s]] = np.clip(s_pts[idx_s[in_s], 0] - cy_s[in_s], -4.0, 4.0)

    tgt = np.zeros((9, h, w), dtype=np.float32)
    tgt[0] = np.clip(prob_out, 0.0, 1.0)
    tgt[1:4] = ch13
    tgt[4] = np.clip(dx / OFFSET_SCALE + 0.5, 0.0, 1.0)
    tgt[5] = np.clip(dy / OFFSET_SCALE + 0.5, 0.0, 1.0)
    tgt[6:8] = ch68
    tgt[8] = np.clip(rad / RAD_SCALE, 0.0, 1.0)
    return tgt


def _prepare_stage_pools(tile_sz: int = 384):
    """Build or load compact pools with v4.3 CleanFuse variants (Clean / Balanced / Agg-Sliver)."""
    INTER_DIR.mkdir(parents=True, exist_ok=True)
    pool_p = INTER_DIR / "pools_v43.npz"
    if pool_p.is_file():
        print(f"Loading cached v4.3 pools from {pool_p}...", flush=True)
        d = np.load(pool_p)
        return {k: d[k] for k in d.files}

    syn_p = CACHE_DIR / "v4_tiles_cache" / "syn_cache.npz"
    if not syn_p.is_file():
        raise RuntimeError(f"Missing {syn_p}. Run 09_train_line_vector_cnn.py first to populate syn_cache.")
    sc = np.load(syn_p)
    syn_x, syn_y = sc["tr_x"][:850], sc["tr_y"][:850]

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
        if len(paths_2x) >= 190:
            break

    print(f"Extracting v4.3 CleanFuse targets (v2+v4) across {len(paths_2x)} artworks on GPU...", flush=True)
    v2_net = LineVectorNet(base=48, out_ch=9).cuda().eval()
    v2_net.load_state_dict(torch.load(VEC_V2_CKPT, map_location="cuda", weights_only=True))
    v4_net = LineVectorNet(base=48, out_ch=9).cuda().eval()
    v4_net.load_state_dict(torch.load(VEC_V4_CKPT, map_location="cuda", weights_only=True))

    real_x_list, real_clean_list, real_bal_list, real_agg_list = [], [], [], []
    rng = np.random.default_rng(2026)
    t0 = time.time()
    for idx, p in enumerate(paths_2x):
        sr_2x = np.asarray(Image.open(p).convert("RGB"))
        h2, w2 = sr_2x.shape[:2]
        if min(h2, w2) < tile_sz:
            continue
        for _ in range(3):
            y0 = int(rng.integers(0, h2 - tile_sz + 1))
            x0 = int(rng.integers(0, w2 - tile_sz + 1))
            crop = sr_2x[y0:y0 + tile_sz, x0:x0 + tile_sz]
            ten = torch.from_numpy((crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vf2 = v2_net(ten).float().cpu().numpy()[0]
                vf4 = v4_net(ten).float().cpu().numpy()[0]
            if float((vf2[0] > 0.40).mean()) < 0.012:
                continue
            t_clean = _build_v43_targets(crop, vf2, vf4, collapse_slivers=True, sliver_max_r=1.05, bh_thr=0.095, dt_hi=3.45)
            t_bal = _build_v43_targets(crop, vf2, vf4, collapse_slivers=True, sliver_max_r=1.45, bh_thr=0.09, dt_hi=3.55)
            t_agg = _build_v43_targets(crop, vf2, vf4, collapse_slivers=True, sliver_max_r=1.95, bh_thr=0.08, dt_hi=3.85)
            real_x_list.append(crop)
            real_clean_list.append(np.rint(np.clip(t_clean, 0, 1) * 255).astype(np.uint8))
            real_bal_list.append(np.rint(np.clip(t_bal, 0, 1) * 255).astype(np.uint8))
            real_agg_list.append(np.rint(np.clip(t_agg, 0, 1) * 255).astype(np.uint8))
        if (idx + 1) % 30 == 0:
            print(f"  processed {idx+1}/{len(paths_2x)} artworks ({len(real_x_list)} tiles, {time.time()-t0:.1f}s)", flush=True)

    del v2_net, v4_net
    torch.cuda.empty_cache()

    out = {
        "syn_x": syn_x, "syn_y": syn_y,
        "real_x": np.stack(real_x_list, axis=0),
        "real_clean": np.stack(real_clean_list, axis=0),
        "real_bal": np.stack(real_bal_list, axis=0),
        "real_agg": np.stack(real_agg_list, axis=0),
    }
    np.savez(pool_p, **out)
    print(f"Saved pools to {pool_p} (syn={len(syn_x)}, real={len(out['real_x'])})", flush=True)
    return out


class _StageDataset(Dataset):
    def __init__(self, xs, ys, gws, train: bool = True):
        self.xs, self.ys, self.gws, self.train = xs, ys, gws, train

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x, y, gw = self.xs[i], self.ys[i], self.gws[i]
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
                y = y[:, :, ::-1].copy()
                y[4] = 255 - y[4]; y[7] = 255 - y[7]
            if np.random.rand() < 0.5:
                x = x[::-1, :]
                y = y[:, ::-1, :].copy()
                y[5] = 255 - y[5]; y[7] = 255 - y[7]
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
            xf = np.clip(xf * float(np.random.uniform(0.94, 1.06)) + float(np.random.uniform(-0.02, 0.02)), 0.0, 1.0)
        else:
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
        yf = np.ascontiguousarray(y).astype(np.float32) / 255.0
        return torch.from_numpy(xf).permute(2, 0, 1).contiguous(), torch.from_numpy(yf).contiguous(), torch.tensor(gw, dtype=torch.float32)


def _train_candidate(
    name: str,
    init_state: dict,
    pools: dict,
    syn_frac: float,
    real_mode: str,
    pos_anchor_net: LineVectorNet | None,
    sharp_boost: float,
    y_collapse_boost: float,
    epochs: int,
    lr: float,
    out_ckpt: Path,
) -> dict:
    device = torch.device("cuda")
    net = LineVectorNet(base=48, out_ch=9).to(device)
    net.load_state_dict(init_state, strict=False)

    n_syn = len(pools["syn_x"])
    n_real = len(pools["real_x"])
    n_s_use = min(n_syn, int(round(920 * syn_frac)))
    n_r_use = min(n_real, 920 - n_s_use)

    rng = np.random.default_rng(abs(hash(name + str(time.time()))) % (2**31))
    si = rng.choice(n_syn, size=n_s_use, replace=False)
    ri = rng.choice(n_real, size=n_r_use, replace=False)

    tr_x = np.concatenate([pools["syn_x"][si], pools["real_x"][ri]], axis=0)
    if real_mode == "agg":
        real_y_src = pools["real_agg"]
    elif real_mode == "clean" and "real_clean" in pools:
        real_y_src = pools["real_clean"]
    else:
        real_y_src = pools["real_bal"]
    tr_y = np.concatenate([pools["syn_y"][si], real_y_src[ri]], axis=0).copy()
    tr_gw = np.concatenate([np.full(n_s_use, 1.0, np.float32), np.full(n_r_use, 0.78, np.float32)])

    if pos_anchor_net is not None:
        pos_anchor_net.eval()
        with torch.no_grad():
            for s0 in range(n_s_use, len(tr_x), 16):
                batch = torch.from_numpy((tr_x[s0:s0+16].astype(np.float32)/255.0).transpose(0,3,1,2)).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pa = pos_anchor_net(batch).float().cpu().numpy()
                pa_u8 = np.rint(np.clip(pa, 0, 1) * 255).astype(np.uint8)
                blend_m = (pa[:, 0] >= 0.60) | ((pa[:, 0] < 0.04) & (tr_y[s0:s0+16, 0] < 60))
                for k in range(len(pa)):
                    bm = blend_m[k]
                    for ch in range(9):
                        tr_y[s0 + k, ch, bm] = np.rint(0.65 * pa_u8[k, ch, bm] + 0.35 * tr_y[s0 + k, ch, bm]).astype(np.uint8)

    if sharp_boost > 1.0:
        p0 = tr_y[:, 0].astype(np.float32) / 255.0
        p0 = np.power(p0, 1.0 + 0.30 * (sharp_boost - 1.0))
        tr_y[:, 0] = np.rint(np.clip(p0, 0, 1) * 255).astype(np.uint8)
    elif y_collapse_boost > 1.0:
        p0 = tr_y[:, 0].astype(np.float32) / 255.0
        p0 = np.power(p0, 1.0 / (1.0 + 0.25 * (y_collapse_boost - 1.0)))
        tr_y[:, 0] = np.rint(np.clip(p0, 0, 1) * 255).astype(np.uint8)

    loader = DataLoader(_StageDataset(tr_x, tr_y, tr_gw, train=True), batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        tot = 0.0
        for bx, by, bgw in loader:
            bx, by, bgw = bx.to(device), by.to(device), bgw.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = net(bx)
            loss, _, _, _, _ = m09._vector_loss(pred.float(), by, bgw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tot += float(loss.item()) * len(bx)
        sch.step()
        print(f"  [{name}] ep {ep}/{epochs} loss={tot/len(tr_x):.4f} ({time.time()-t0:.1f}s)", flush=True)

    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(sd, out_ckpt)
    del net, opt
    torch.cuda.empty_cache()
    return sd


def _eval_candidate_crops(state_dict: dict, sr_tarot: np.ndarray, sr_t12: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    device = torch.device("cuda")
    net = LineVectorNet(base=48, out_ch=9).to(device).eval()
    net.load_state_dict(state_dict)

    def _pred_crop(crop_sr: np.ndarray) -> np.ndarray:
        h, w = crop_sr.shape[:2]
        pad_h, pad_w = max(0, 384 - h), max(0, 384 - w)
        inp = np.pad(crop_sr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        ten = torch.from_numpy((inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = net(ten).float().cpu().numpy()[0]
        return o[:, :h, :w]

    boxes_t = [
        (1060 * 2, 1210 * 2, 1175 * 2, 1325 * 2),  # Prism bevel/sliver
        (150 * 2, 1455 * 2, 330 * 2, 1635 * 2),    # Rosette center Y-star + outer beads
        (360 * 2, 420 * 2, 540 * 2, 600 * 2),      # HairWisps / NightSky
    ]
    H12, W12 = sr_t12.shape[:2]
    cx12, cy12 = W12 // 2, int(H12 * 0.35)
    box12 = (cx12 - 180, cy12 - 180, cx12 + 180, cy12 + 180)

    crops_sr = [sr_tarot[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes_t] + [sr_t12[box12[1]:box12[3], box12[0]:box12[2]]]
    hmaps, scatters = [], []
    for cs in crops_sr:
        vf = _pred_crop(cs)
        p = vf[0]
        hm = cv2.cvtColor(cv2.applyColorMap(np.clip(p * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        hmaps.append(hm)

        hs, ws = p.shape
        sub = 2
        dens = np.zeros((hs * sub, ws * sub), np.float32)
        ay, ax = np.where(p >= 0.30)
        if len(ay):
            dx = (vf[4, ay, ax] - 0.5) * OFFSET_SCALE
            dy = (vf[5, ay, ax] - 0.5) * OFFSET_SCALE
            damp = np.where(vf[3, ay, ax] < 0.42, 1.0, 0.45)
            px = np.clip((ax + dx * damp) * sub + 0.5, 0, ws * sub - 1.001)
            py = np.clip((ay + dy * damp) * sub + 0.5, 0, hs * sub - 1.001)
            x0, y0 = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
            np.add.at(dens, (y0, x0), p[ay, ax])
        dens_s = cv2.GaussianBlur(dens, (3, 3), 0.45)
        line_u8 = np.clip((1.0 - np.clip(dens_s / 1.1, 0, 1)) * 255, 0, 255).astype(np.uint8)
        scatters.append(np.stack([line_u8, line_u8, line_u8], axis=-1))

    del net
    torch.cuda.empty_cache()
    return hmaps, scatters


def _render_grid(
    orig_crops: list[np.ndarray],
    cand_results: dict[str, tuple[list[np.ndarray], list[np.ndarray], str]],
    show_scatter: bool,
    row_filter: int | None = None,
) -> Image.Image:
    """Render full crisp 360px/tile grid (either all 4 rows or 1 zoomed focus row)."""
    tw = 360
    names = ["1:Prism斜棱吞并", "2:Rosette花心+外珠", "3:Hair/夜空", "4:test12五官"]
    keys = ["A", "B", "C", "D"]
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    rows = [row_filter] if row_filter is not None else [0, 1, 2, 3]
    grid = Image.new("RGB", (tw * 5, tw * len(rows)), (16, 16, 20))
    for out_r, r_i in enumerate(rows):
        im_o = Image.fromarray(cv2.resize(orig_crops[r_i], (tw, tw), interpolation=cv2.INTER_NEAREST))
        d = ImageDraw.Draw(im_o)
        d.rectangle([0, 0, tw, 26], fill=(22, 22, 28))
        d.text((6, 4), f"Orig ({names[r_i]})", fill=(250, 250, 255), font=font)
        grid.paste(im_o, (0, out_r * tw))
        for c_i, k in enumerate(keys):
            hms, scs, desc = cand_results[k]
            arr = scs[r_i] if show_scatter else hms[r_i]
            im_c = Image.fromarray(cv2.resize(arr, (tw, tw), interpolation=cv2.INTER_NEAREST))
            dc = ImageDraw.Draw(im_c)
            dc.rectangle([0, 0, tw, 26], fill=(22, 22, 28))
            dc.text((6, 4), f"[{k}] {desc[:20]}", fill=(255, 220, 110), font=font)
            grid.paste(im_c, ((c_i + 1) * tw, out_r * tw))
    return grid


def _open_picker_window(
    stage_idx: int,
    orig_crops: list[np.ndarray],
    cand_results: dict[str, tuple[list[np.ndarray], list[np.ndarray], str]],
) -> tuple[str, list[str], str]:
    """Interactive zoomable/pannable Tkinter selector.
    Returns (winner_key, feedback_tags, action) where action in ('next', 'retry', 'finish').
    """
    _render_grid(orig_crops, cand_results, show_scatter=False).save(INTER_DIR / f"stage_{stage_idx}_heatmap.jpg", quality=94)
    _render_grid(orig_crops, cand_results, show_scatter=True).save(INTER_DIR / f"stage_{stage_idx}_scatter.jpg", quality=94)

    state = {"key": "A", "tags": [], "action": "next"}
    root = tk.Tk()
    root.title(f"Lumina-Vec v4.2 正反馈选优 - 第 {stage_idx} 阶段（支持滚轮放大/左键拖拽/单行特写）")
    root.geometry("1460x920")
    root.configure(bg="#141418")
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))

    view_mode = tk.StringVar(value="heat")
    row_sel = tk.IntVar(value=-1)  # -1 = all rows, 0..3 = focus row
    var_merge_sliver = tk.BooleanVar(value=False)
    var_keep_lines = tk.BooleanVar(value=False)
    var_y_collapse = tk.BooleanVar(value=False)
    var_sharp = tk.BooleanVar(value=False)

    # Cache base PIL images
    cache_imgs: dict[tuple[bool, int | None], Image.Image] = {}
    for sc_b in (False, True):
        for rf in (None, 0, 1, 2, 3):
            cache_imgs[(sc_b, rf)] = _render_grid(orig_crops, cand_results, show_scatter=sc_b, row_filter=rf)

    # Top controls bar (Row focus + Mode + Zoom reset)
    top_bar = tk.Frame(root, bg="#1b1b22")
    top_bar.pack(fill="x", padx=8, pady=6)
    tk.Label(top_bar, text="滚轮放大(最高6x) | 左键拖拽 | 单行聚焦：", fg="#a6f7b2", bg="#1b1b22", font=("Microsoft YaHei UI", 10, "bold")).pack(side="left", padx=6)

    canvas_state = {"zoom": 1.0, "off_x": 0.0, "off_y": 0.0, "drag_x": 0, "drag_y": 0, "tk_img": None}
    canvas = tk.Canvas(root, bg="#101014", highlightthickness=0, cursor="fleur")
    canvas.pack(fill="both", expand=True, padx=8, pady=2)

    lbl_zoom = tk.Label(top_bar, text="缩放: 100%", fg="#ffd86b", bg="#1b1b22", font=("Microsoft YaHei UI", 10))
    lbl_zoom.pack(side="right", padx=10)

    def _current_base() -> Image.Image:
        rf = None if row_sel.get() < 0 else row_sel.get()
        return cache_imgs[(view_mode.get() == "scat", rf)]

    def _redraw(fit_first: bool = False):
        cw = max(100, canvas.winfo_width())
        ch = max(100, canvas.winfo_height())
        base = _current_base()
        if fit_first:
            z = min(cw / base.width, ch / base.height)
            canvas_state["zoom"] = z
            canvas_state["off_x"] = (cw - base.width * z) * 0.5
            canvas_state["off_y"] = (ch - base.height * z) * 0.5
        z = canvas_state["zoom"]
        nw = max(16, int(round(base.width * z)))
        nh = max(16, int(round(base.height * z)))
        interp = Image.Resampling.NEAREST if z >= 1.0 else Image.Resampling.BILINEAR
        disp = base.resize((nw, nh), interp)
        canvas_state["tk_img"] = ImageTk.PhotoImage(disp)
        canvas.delete("all")
        canvas.create_image(int(canvas_state["off_x"]), int(canvas_state["off_y"]), anchor="nw", image=canvas_state["tk_img"])
        lbl_zoom.configure(text=f"缩放: {int(round(z * 100))}%")

    def _on_wheel(ev):
        old_z = canvas_state["zoom"]
        factor = 1.22 if ev.delta > 0 else (1.0 / 1.22)
        new_z = float(np.clip(old_z * factor, 0.25, 6.0))
        ratio = new_z / old_z
        canvas_state["off_x"] = ev.x - (ev.x - canvas_state["off_x"]) * ratio
        canvas_state["off_y"] = ev.y - (ev.y - canvas_state["off_y"]) * ratio
        canvas_state["zoom"] = new_z
        _redraw(False)

    def _on_press(ev):
        canvas_state["drag_x"] = ev.x
        canvas_state["drag_y"] = ev.y

    def _on_drag(ev):
        canvas_state["off_x"] += ev.x - canvas_state["drag_x"]
        canvas_state["off_y"] += ev.y - canvas_state["drag_y"]
        canvas_state["drag_x"] = ev.x
        canvas_state["drag_y"] = ev.y
        _redraw(False)

    canvas.bind("<MouseWheel>", _on_wheel)
    canvas.bind("<ButtonPress-1>", _on_press)
    canvas.bind("<B1-Motion>", _on_drag)
    canvas.bind("<Configure>", lambda _e: _redraw(True) if canvas_state["tk_img"] is None else None)

    for r_val, r_txt in [(-1, "全部4行"), (0, "只看1:Prism斜棱"), (1, "只看2:Rosette花心"), (2, "只看3:Hair夜空"), (3, "只看4:test12")]:
        tk.Radiobutton(
            top_bar, text=r_txt, variable=row_sel, value=r_val,
            command=lambda: _redraw(True), fg="#f2f2f8", bg="#1b1b22", selectcolor="#2e6fdb",
            indicatoron=0, padx=8, pady=3, font=("Microsoft YaHei UI", 9),
        ).pack(side="left", padx=2)
    tk.Button(top_bar, text="重置视口", command=lambda: _redraw(True), bg="#323240", fg="#fff", relief="flat", padx=8, pady=3, font=("Microsoft YaHei UI", 9)).pack(side="left", padx=6)

    mid_bar = tk.Frame(root, bg="#1c1c24")
    mid_bar.pack(fill="x", padx=8, pady=4)
    tk.Radiobutton(mid_bar, text="热力图 (stroke_prob)", variable=view_mode, value="heat", command=lambda: _redraw(False), fg="#ffd86b", bg="#1c1c24", selectcolor="#2c2c38", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left", padx=8)
    tk.Radiobutton(mid_bar, text="散点聚拢线稿 (Scatter)", variable=view_mode, value="scat", command=lambda: _redraw(False), fg="#8be9fd", bg="#1c1c24", selectcolor="#2c2c38", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left", padx=8)
    tk.Checkbutton(mid_bar, text="第1行窄亮棱还需吞并", variable=var_merge_sliver, fg="#ff9b9b", bg="#1c1c24", selectcolor="#2c2c38", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=8)
    tk.Checkbutton(mid_bar, text="外圈/细线残缺需补回", variable=var_keep_lines, fg="#a6f7b2", bg="#1c1c24", selectcolor="#2c2c38", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=8)
    tk.Checkbutton(mid_bar, text="花心Y字还需再收拢", variable=var_y_collapse, fg="#ffcb80", bg="#1c1c24", selectcolor="#2c2c38", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=8)
    tk.Checkbutton(mid_bar, text="线条略发虚需更锋利", variable=var_sharp, fg="#a5d6ff", bg="#1c1c24", selectcolor="#2c2c38", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=8)

    btn_row = tk.Frame(root, bg="#141418")
    btn_row.pack(fill="x", padx=8, pady=6)

    def _finish_pick(k: str, action: str):
        tags = []
        if var_merge_sliver.get():
            tags.append("merge_sliver")
        if var_keep_lines.get():
            tags.append("keep_lines")
        if var_y_collapse.get():
            tags.append("fix_hollow")
        if var_sharp.get():
            tags.append("fix_sharp")
        state["key"], state["tags"], state["action"] = k, tags, action
        root.destroy()

    colors = {"A": "#2e6fdb", "B": "#1f8f5f", "C": "#c96f1d", "D": "#7e4bc4"}
    for k in ("A", "B", "C", "D"):
        desc = cand_results[k][2]
        f_b = tk.Frame(btn_row, bg="#141418")
        f_b.pack(side="left", expand=True, fill="x", padx=4)
        tk.Button(
            f_b, text=f"选 [{k}] 作为正反馈进入下一轮 ({desc})",
            bg=colors[k], fg="white", font=("Microsoft YaHei UI", 9, "bold"),
            relief="flat", pady=6, cursor="hand2", command=lambda kk=k: _finish_pick(kk, "next"),
        ).pack(fill="x")
        tk.Button(
            f_b, text=f"直接导出 [{k}] 为最终 ONNX",
            bg="#262632", fg="#a6f7b2", font=("Microsoft YaHei UI", 8),
            relief="flat", pady=3, cursor="hand2", command=lambda kk=k: _finish_pick(kk, "finish"),
        ).pack(fill="x", pady=(3, 0))

    tk.Button(
        btn_row, text="都不满意：\n按左侧勾选方向\n重跑本轮候选",
        bg="#8a2c3a", fg="white", font=("Microsoft YaHei UI", 9, "bold"),
        relief="flat", padx=10, cursor="hand2", command=lambda: _finish_pick("A", "retry"),
    ).pack(side="right", fill="y", padx=(6, 0))

    root.after(120, lambda: _redraw(True))
    root.bell()
    root.mainloop()
    return state["key"], state["tags"], state["action"]


def main():
    pools = _prepare_stage_pools(tile_sz=384)

    rgb_t = np.asarray(Image.open(TAROT_IMG).convert("RGB"))
    rgb_12 = np.asarray(Image.open(TEST12_IMG).convert("RGB"))
    sess_sr = _get_session(_WEIGHTS_DEFAULT / "realesr_anime.onnx")
    sr_tarot = _upscale_sr(rgb_t, 2.0, sess_sr)
    sr_t12 = _upscale_sr(rgb_12, 2.0, sess_sr)
    del sess_sr
    _SESSION_CACHE.clear()

    boxes_t = [(1060 * 2, 1210 * 2, 1175 * 2, 1325 * 2), (150 * 2, 1455 * 2, 330 * 2, 1635 * 2), (360 * 2, 420 * 2, 540 * 2, 600 * 2)]
    H12, W12 = sr_t12.shape[:2]
    cx12, cy12 = W12 // 2, int(H12 * 0.35)
    box12 = (cx12 - 180, cy12 - 180, cx12 + 180, cy12 + 180)
    orig_crops = [sr_tarot[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes_t] + [sr_t12[box12[1]:box12[3], box12[0]:box12[2]]]

    sd_v2 = torch.load(VEC_V2_CKPT, map_location="cpu", weights_only=True)
    sd_v4 = torch.load(VEC_V4_CKPT, map_location="cpu", weights_only=True) if VEC_V4_CKPT.is_file() else sd_v2

    # Start Stage 1 from clean v2 backbone + v4.2 targets so NO residual D-dropout carries over!
    anchor_sd = sd_v4
    anchor_desc = "v4参考基线"
    sharp_boost = 1.05
    y_boost = 1.05
    syn_frac = 0.50
    prefer_agg = False
    history = []

    stage = 1
    while stage <= 5:
        print(f"\n=== Stage {stage}: Training Positive-Feedback Candidates ===", flush=True)
        base_init = sd_v2 if (stage == 1) else anchor_sd
        pos_net = None
        if stage > 1:
            pos_net = LineVectorNet(base=48, out_ch=9).cuda()
            pos_net.load_state_dict(anchor_sd)

        ep_s = 4 if stage == 1 else 3
        lr_s = 6.0e-4 if stage == 1 else 2.8e-4
        sd_A = _train_candidate(
            f"St{stage}_A_Unified", base_init, pools, syn_frac=syn_frac,
            real_mode="agg" if prefer_agg else "bal",
            pos_anchor_net=pos_net, sharp_boost=sharp_boost * 1.10, y_collapse_boost=y_boost,
            epochs=ep_s, lr=lr_s, out_ckpt=INTER_DIR / f"st{stage}_A.pt",
        )
        sd_B = _train_candidate(
            f"St{stage}_B_AggSliver", base_init, pools, syn_frac=syn_frac, real_mode="agg",
            pos_anchor_net=pos_net, sharp_boost=sharp_boost * 1.25, y_collapse_boost=max(1.20, y_boost * 1.20),
            epochs=ep_s, lr=lr_s, out_ckpt=INTER_DIR / f"st{stage}_B.pt",
        )
        sd_C = _train_candidate(
            f"St{stage}_C_SharpClean", base_init, pools, syn_frac=min(0.70, syn_frac + 0.12), real_mode="clean",
            pos_anchor_net=pos_net, sharp_boost=max(1.30, sharp_boost * 1.35), y_collapse_boost=1.0,
            epochs=ep_s, lr=lr_s, out_ckpt=INTER_DIR / f"st{stage}_C.pt",
        )
        if pos_net is not None:
            del pos_net
            torch.cuda.empty_cache()

        cands = {
            "A": (sd_A, "均衡融合(V夹净轮廓+单线Y+保外框)"),
            "B": (sd_B, "强吞微亮棱+利落单线Y"),
            "C": (sd_C, "高锐高保真(V夹保细节+单线Y)"),
            "D": (anchor_sd, anchor_desc),
        }
        cand_eval = {k: (*_eval_candidate_crops(sd_k, sr_tarot, sr_t12), desc_k) for k, (sd_k, desc_k) in cands.items()}

        print(f"Popping up Stage {stage} Zoomable Visual Picker...", flush=True)
        winner_k, tags, action = _open_picker_window(stage, orig_crops, cand_eval)
        print(f"User action={action} winner={winner_k} tags={tags}", flush=True)

        history.append({"stage": stage, "winner": winner_k, "tags": tags, "action": action})
        (INTER_DIR / "feedback_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        if "merge_sliver" in tags:
            prefer_agg = True
            y_boost = min(1.8, y_boost + 0.25)
        if "keep_lines" in tags:
            prefer_agg = False
            syn_frac = max(0.35, syn_frac - 0.10)
        if "fix_hollow" in tags:
            y_boost = min(1.8, y_boost + 0.25)
        if "fix_sharp" in tags:
            sharp_boost = min(2.0, sharp_boost + 0.28)

        if action == "retry":
            continue
        anchor_sd, anchor_desc = cands[winner_k][0], f"S{stage}-{winner_k}胜出"
        if action == "finish":
            break
        stage += 1

    best_ckpt = CACHE_DIR / "line_vector_v4_interactive_best.pt"
    torch.save(anchor_sd, best_ckpt)
    net_out = LineVectorNet(base=48, out_ch=9).cpu().eval()
    net_out.load_state_dict(anchor_sd)
    dummy = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    torch.onnx.export(net_out, dummy, str(VEC_ONNX), input_names=["rgb"], output_names=["vec_fields"], opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(VEC_ONNX), providers=["CPUExecutionProvider"])
    diff = float(np.max(np.abs(net_out(dummy).detach().numpy() - sess.run(None, {"rgb": dummy.numpy()})[0])))
    print(f"Exported Positive-Feedback ONNX -> {VEC_ONNX} (PT-vs-ORT diff={diff:.2e})", flush=True)


if __name__ == "__main__":
    main()
