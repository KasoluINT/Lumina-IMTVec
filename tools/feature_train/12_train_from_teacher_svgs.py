"""Phase 6: Train LineVectorNet with 78 High-Precision Teacher SVGs + V2 Domain Retention.

Renders every Teacher SVG (`C:\\Users\\30266\\Desktop\\自建模型\\图片处理SVG\\教师SVG`) via headless Edge
into pixel-exact RGB and 24-bit polygon ID maps (cached under `work_cache/teacher_svg_cache`),
extracts double-wall-free ('无抽壳') single medial ink centerlines + exact sub-pixel offsets,
tangents, and stroke half-widths, cleanly separating structural lines (`ch 0`) from soft cel-shading
boundaries (`ch 1: color_bound`), and trains `LineVectorNet` initialized from V2 (`OFFSET_SCALE=3.0`,
`RAD_SCALE=3.2`) mixed with `v2_tiles_cache` for stained-glass & fine-line retention.

Usage:
    python tools/feature_train/12_train_from_teacher_svgs.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import importlib
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "feature_train"))

from lumina_vec.color import srgb_to_oklab
from lumina_vec.feature_nets import LineVectorNet

TEACHER_SVG_DIR = Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\教师SVG")
CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
T_CACHE = CACHE_DIR / "teacher_svg_cache"
V2_CACHE = CACHE_DIR / "v2_tiles_cache"
V2_CKPT = CACHE_DIR / "backup_v2" / "ckpt" / "line_vector_v2_best.pt"
CKPT_OUT = CACHE_DIR / "line_vector_teacher_best.pt"
ONNX_OUT = ROOT / "weights" / "line_vector_net.onnx"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

OFFSET_SCALE = 3.0
RAD_SCALE = 3.2


def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
    hx = hx.strip().lstrip("#")
    if len(hx) == 6:
        return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
    return (255, 255, 255)


def _edge_render_svg(svg_txt: str, rw: int, rh: int, bg_hex: str, tmp_tag: str) -> np.ndarray:
    T_CACHE.mkdir(parents=True, exist_ok=True)
    svg_p = T_CACHE / f"_r_{tmp_tag}.svg"
    html_p = T_CACHE / f"_r_{tmp_tag}.html"
    png_p = T_CACHE / f"_r_{tmp_tag}.png"
    svg_p.write_text(svg_txt, encoding="utf-8")
    html = (
        f"<!doctype html><style>*{{margin:0;padding:0;overflow:hidden}}"
        f"body{{width:{rw}px;height:{rh}px;background:{bg_hex}}}"
        f"img{{width:{rw}px;height:{rh}px;display:block}}</style>"
        f"<img src={svg_p.resolve().as_uri()}>"
    )
    html_p.write_text(html, encoding="utf-8")
    try:
        subprocess.run([
            EDGE_EXE, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=1", f"--window-size={rw},{rh}",
            f"--screenshot={str(png_p)}", html_p.resolve().as_uri(),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        im = np.array(Image.open(png_p).convert("RGB"))[:rh, :rw]
    finally:
        for p in (svg_p, html_p, png_p):
            if p.exists():
                p.unlink()
    return im


def _extract_svg_targets(im_rgb: np.ndarray, id_map: np.ndarray, fills: list[tuple[int, int, int]]) -> np.ndarray:
    """Build exact 9-ch V2-scale ground-truth targets from rendered SVG RGB + exact 24-bit polygon IDs.
    Zero '抽壳' double walls: thin dark ink polygons skeletonize to 1 medial axis with exact half-width!
    """
    h, w = im_rgb.shape[:2]
    fills_arr = np.array(fills, dtype=np.uint8)
    ok_pal = srgb_to_oklab(fills_arr[None])[0]
    L_pal = ok_pal[:, 0]

    # Exact boundaries between distinct fill polygons
    diff_h = id_map[:, :-1] != id_map[:, 1:]
    diff_v = id_map[:-1, :] != id_map[1:, :]
    bound_mask = np.zeros((h, w), dtype=bool)
    bound_mask[:, :-1] |= diff_h
    bound_mask[:, 1:] |= diff_h
    bound_mask[:-1, :] |= diff_v
    bound_mask[1:, :] |= diff_v

    ok_img = ok_pal[id_map]
    de_h = np.linalg.norm(ok_img[:, :-1] - ok_img[:, 1:], axis=-1)
    de_v = np.linalg.norm(ok_img[:-1, :] - ok_img[1:, :], axis=-1)
    max_de = np.zeros((h, w), dtype=np.float32)
    max_de[:, :-1] = np.maximum(max_de[:, :-1], de_h)
    max_de[:, 1:] = np.maximum(max_de[:, 1:], de_h)
    max_de[:-1, :] = np.maximum(max_de[:-1, :], de_v)
    max_de[1:, :] = np.maximum(max_de[1:, :], de_v)

    # Dark ink fill polygons: medial skeletonize ONLY genuine narrow/medium ink strokes (dt <= 4.3px),
    # keeping wide solid dark outfits strictly as outer boundary contours (ZERO interior skeleton artifacts!)
    dark_px = L_pal[id_map] < 0.28
    dt_dark = cv2.distanceTransform(dark_px.astype(np.uint8), cv2.DIST_L2, 3)
    wide_dark_core = dt_dark > 4.4
    wide_dark_zone = cv2.dilate(wide_dark_core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    thin_ink = dark_px & (~wide_dark_zone)

    # Smooth thin_ink slightly on 2x subgrid to make skeleton ultra-smooth
    thin_2x = cv2.resize(thin_ink.astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_LINEAR) >= 0.45
    dt_2x = (cv2.distanceTransform(thin_2x.astype(np.uint8), cv2.DIST_L2, 5) / 2.0).astype(np.float32)
    sk_2x = skeletonize(thin_2x)

    # Boundaries NOT touching thin_ink -> split into high-contrast structural contours vs cel-shading bounds
    dark_ink_halo = cv2.dilate(thin_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pure_edge = bound_mask & (~dark_ink_halo)
    hi_edge = pure_edge & (max_de >= 0.14)
    cel_edge = pure_edge & (max_de < 0.14)

    # Gather centerline points for sparse exact cKDTree target synthesis
    sy_i, sx_i = np.where(sk_2x)
    pts_ink = np.column_stack([sy_i.astype(np.float32) * 0.5, sx_i.astype(np.float32) * 0.5]) if len(sy_i) else np.zeros((0, 2), np.float32)
    rad_ink = np.clip(dt_2x[sy_i, sx_i], 0.65, 3.2) if len(sy_i) else np.zeros((0,), np.float32)

    ey_h, ex_h = np.where(hi_edge)
    pts_edge = np.column_stack([ey_h.astype(np.float32), ex_h.astype(np.float32)]) if len(ey_h) else np.zeros((0, 2), np.float32)
    rad_edge = np.full(len(ey_h), 0.75, np.float32)

    all_pts = np.concatenate([pts_ink, pts_edge], axis=0)
    all_rad = np.concatenate([rad_ink, rad_edge], axis=0)

    tgt = np.zeros((9, h, w), dtype=np.float32)
    tgt[4] = 0.5
    tgt[5] = 0.5
    tgt[6] = 0.5
    tgt[7] = 0.5

    if len(all_pts) >= 4:
        # Sub-pixel tangent via finite difference on smoothed combined binary map + local structure tensor
        line_bin = (sk_2x[::2, ::2] | hi_edge).astype(np.float32)
        sm = cv2.GaussianBlur(line_bin, (5, 5), 1.1)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (5, 5), 1.2)
        jxy = cv2.GaussianBlur(gx * gy, (5, 5), 1.2)
        jyy = cv2.GaussianBlur(gy * gy, (5, 5), 1.2)
        # Tangent orientation perpendicular to max gradient eigenvector = 0.5 * atan2(-2*jxy, jyy - jxx)
        phi = np.arctan2(-2.0 * jxy, jyy - jxx)
        cos2t = np.cos(phi)
        sin2t = np.sin(phi)

        # Endpoints & junctions on 1x skeleton
        sk_1x = skeletonize((sk_2x[::2, ::2] | hi_edge).astype(bool))
        sk_u8 = sk_1x.astype(np.uint8)
        nbr = cv2.filter2D(sk_u8, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - sk_u8
        end_m = cv2.GaussianBlur((sk_1x & (nbr == 1)).astype(np.float32), (5, 5), 0.9)
        junc_m = cv2.GaussianBlur((sk_1x & (nbr >= 3)).astype(np.float32), (5, 5), 0.9)
        if end_m.max() > 0:
            end_m = np.clip(end_m / (end_m.max() * 0.65 + 1e-4), 0.0, 1.0)
        if junc_m.max() > 0:
            junc_m = np.clip(junc_m / (junc_m.max() * 0.65 + 1e-4), 0.0, 1.0)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        coords = np.column_stack([yy.ravel(), xx.ravel()])
        dist, idx = cKDTree(all_pts).query(coords, workers=4)
        dist = dist.reshape(h, w)
        idx = idx.reshape(h, w)

        ny = all_pts[idx, 0]
        nx = all_pts[idx, 1]
        nr = all_rad[idx]

        # Stroke prob: sharp Gaussian peak matching radius
        sig = np.clip(0.55 + 0.25 * nr, 0.72, 1.25)
        prob = np.exp(-0.5 * (dist / sig) ** 2)
        prob[dist > 4.5] = 0.0

        niy = np.clip(np.round(ny).astype(int), 0, h - 1)
        nix = np.clip(np.round(nx).astype(int), 0, w - 1)
        dx = np.clip(nx - xx, -OFFSET_SCALE * 0.5, OFFSET_SCALE * 0.5)
        dy = np.clip(ny - yy, -OFFSET_SCALE * 0.5, OFFSET_SCALE * 0.5)

        active = dist <= 3.8
        tgt[0] = np.clip(prob, 0.0, 1.0)
        tgt[2] = np.where(active, end_m[niy, nix], 0.0)
        tgt[3] = np.where(active, junc_m[niy, nix], 0.0)
        tgt[4] = np.where(active, np.clip(dx / OFFSET_SCALE + 0.5, 0.0, 1.0), 0.5)
        tgt[5] = np.where(active, np.clip(dy / OFFSET_SCALE + 0.5, 0.0, 1.0), 0.5)
        tgt[6] = np.where(active, 0.5 * (cos2t[niy, nix] + 1.0), 0.5)
        tgt[7] = np.where(active, 0.5 * (sin2t[niy, nix] + 1.0), 0.5)
        tgt[8] = np.where(active, np.clip(nr / RAD_SCALE, 0.0, 1.0), 0.0)

    # Channel 1: crisp color_bound covers BOTH cel_edge AND hi_edge cleanly
    cb_raw = (pure_edge).astype(np.float32)
    tgt[1] = np.clip(cv2.GaussianBlur(cb_raw, (3, 3), 0.75) * 1.45, 0.0, 1.0)
    return tgt


def build_or_load_teacher_cache(tile_sz: int = 384):
    T_CACHE.mkdir(parents=True, exist_ok=True)
    npz_p = T_CACHE / "teacher_tiles.npz"
    if npz_p.is_file():
        print(f"Loading cached Teacher-SVG tiles from {npz_p}...", flush=True)
        d = np.load(npz_p)
        return d["tr_x"], d["tr_y"], d["va_x"], d["va_y"]

    svgs = sorted(TEACHER_SVG_DIR.glob("*.svg"))
    if not svgs:
        raise RuntimeError(f"No SVGs found in {TEACHER_SVG_DIR}")

    # Hold out 2 unique subject hashes cleanly for validation
    subjects = sorted(set(s.name.split("_")[1] for s in svgs))
    val_subjects = set(subjects[-2:])
    print(f"Extracting single-line targets from {len(svgs)} Teacher SVGs ({len(subjects)} subjects, val={val_subjects})...", flush=True)

    tr_x, tr_y, va_x, va_y = [], [], [], []
    rng = np.random.default_rng(2026)
    t0 = time.time()

    for si, sp in enumerate(svgs):
        txt = sp.read_text(encoding="utf-8", errors="ignore")
        root = ET.fromstring(txt)
        w = int(float(root.attrib.get("width", 1000)))
        h = int(float(root.attrib.get("height", 1000)))
        sc = min(1.0, 1280.0 / max(w, h))
        rw, rh = max(384, int(round(w * sc))), max(384, int(round(h * sc)))

        r_id = ET.fromstring(txt)
        fills = [(255, 255, 255)]
        pid = 1
        for el in r_id.iter():
            if el.tag.endswith("path"):
                if el.get("class", "") == "qq3d-gap-filler":
                    el.set("stroke", "none")
                    el.set("fill", "none")
                else:
                    col_s = el.get("fill", "#ffffff")
                    fills.append(_hex_to_rgb(col_s) if col_s.startswith("#") else (255, 255, 255))
                    el.set("fill", f"#{(pid >> 16) & 255:02x}{(pid >> 8) & 255:02x}{pid & 255:02x}")
                    el.set("stroke", "none")
                    el.set("shape-rendering", "crispEdges")
                    pid += 1

        im_rgb = _edge_render_svg(txt, rw, rh, "#ffffff", f"{si}_rgb")
        im_id = _edge_render_svg(ET.tostring(r_id, encoding="unicode"), rw, rh, "#000000", f"{si}_id")
        id_map = (im_id[..., 0].astype(np.int32) << 16) | (im_id[..., 1].astype(np.int32) << 8) | im_id[..., 2].astype(np.int32)
        id_map = np.clip(id_map, 0, len(fills) - 1)

        tgt = _extract_svg_targets(im_rgb, id_map, fills)
        tgt_u8 = np.rint(np.clip(tgt, 0, 1) * 255).astype(np.uint8)

        subj = sp.name.split("_")[1]
        is_v = subj in val_subjects
        n_crops = 5 if not is_v else 3
        kept = 0
        for _ in range(n_crops * 3):
            y0 = int(rng.integers(0, max(1, rh - tile_sz + 1)))
            x0 = int(rng.integers(0, max(1, rw - tile_sz + 1)))
            c_x = im_rgb[y0:y0 + tile_sz, x0:x0 + tile_sz]
            c_y = tgt_u8[:, y0:y0 + tile_sz, x0:x0 + tile_sz]
            if c_x.shape[0] < tile_sz or c_x.shape[1] < tile_sz:
                continue
            if float((c_y[0] > 90).mean()) < 0.015:
                continue
            if is_v:
                va_x.append(c_x); va_y.append(c_y)
            else:
                tr_x.append(c_x); tr_y.append(c_y)
            kept += 1
            if kept >= n_crops:
                break

        if (si + 1) % 10 == 0:
            print(f"  parsed {si+1}/{len(svgs)} SVGs -> tr={len(tr_x)}, va={len(va_x)} ({time.time()-t0:.1f}s)", flush=True)

    tr_x_a = np.stack(tr_x, axis=0)
    tr_y_a = np.stack(tr_y, axis=0)
    va_x_a = np.stack(va_x, axis=0)
    va_y_a = np.stack(va_y, axis=0)
    np.savez(npz_p, tr_x=tr_x_a, tr_y=tr_y_a, va_x=va_x_a, va_y=va_y_a)
    print(f"Saved Teacher SVG tiles to {npz_p}: tr={len(tr_x_a)}, va={len(va_x_a)}", flush=True)
    return tr_x_a, tr_y_a, va_x_a, va_y_a


class MixedLineDataset(Dataset):
    def __init__(self, svg_x, svg_y, v2_x, v2_y, train: bool = True, svg_weight: float = 1.0, v2_weight: float = 0.85):
        self.svg_x, self.svg_y = svg_x, svg_y
        self.v2_x, self.v2_y = v2_x, v2_y
        self.train = train
        self.svg_w = svg_weight
        self.v2_w = v2_weight
        self.n_svg = len(svg_x)
        self.n_v2 = len(v2_x)

    def __len__(self):
        return self.n_svg + self.n_v2

    def __getitem__(self, i):
        if i < self.n_svg:
            x = self.svg_x[i]
            y = self.svg_y[i]
            gw = self.svg_w
            is_svg = True
        else:
            j = i - self.n_svg
            x = self.v2_x[j]
            y = self.v2_y[j]
            gw = self.v2_w
            is_svg = False

        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]; y = y[:, :, ::-1].copy()
                y[4] = 255 - y[4]; y[7] = 255 - y[7]
            if np.random.rand() < 0.5:
                x = x[::-1, :]; y = y[:, ::-1, :].copy()
                y[5] = 255 - y[5]; y[7] = 255 - y[7]

            # Subtle real-domain degradation on 55% of crisp SVG tiles so CNN generalizes cleanly to 2x SR PNG/JPGs
            x = np.ascontiguousarray(x)
            if is_svg and np.random.rand() < 0.55:
                if np.random.rand() < 0.65:
                    sig = float(np.random.uniform(0.32, 0.72))
                    x = cv2.GaussianBlur(x, (0, 0), sigmaX=sig)
                if np.random.rand() < 0.50:
                    q = int(np.random.randint(84, 97))
                    ok_enc, buf = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
                    if ok_enc:
                        x = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            xf = x.astype(np.float32) / 255.0
            xf = np.clip(xf * float(np.random.uniform(0.93, 1.07)) + float(np.random.uniform(-0.025, 0.025)), 0.0, 1.0)
        else:
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
        yf = np.ascontiguousarray(y).astype(np.float32) / 255.0
        return torch.from_numpy(xf).permute(2, 0, 1).contiguous(), torch.from_numpy(yf).contiguous(), torch.tensor(gw, dtype=torch.float32)


def _vector_loss_v2(pred: torch.Tensor, target: torch.Tensor, g_weight: torch.Tensor):
    gw = g_weight.view(-1, 1, 1, 1)
    p0 = pred[:, 0:1].clamp(1e-4, 1.0 - 1e-4)
    t0 = target[:, 0:1]
    cw_p = 1.0 + 3.0 * (t0 > 0.35).float()
    bce = (-(t0 * torch.log(p0) + (1.0 - t0) * torch.log(1.0 - p0)) * cw_p * gw).mean()
    inter = (p0 * t0 * gw).sum(dim=(2, 3))
    denom = ((p0 + t0) * gw).sum(dim=(2, 3)) + 1e-4
    dice = (1.0 - (2.0 * inter / denom)).mean()
    l_prob = 0.50 * bce + 0.50 * dice + 0.25 * (torch.abs(p0 - t0) * gw).mean()

    l_cls = (torch.abs(pred[:, 1:4] - target[:, 1:4]) * gw).mean()

    active = (t0 > 0.25).float() * gw
    denom_a = active.sum() + 1.0
    l_off = (F.smooth_l1_loss(pred[:, 4:6], target[:, 4:6], reduction="none", beta=0.05) * active).sum() / denom_a

    # Symmetric tangent angle cos2t/sin2t
    l_ang = (F.smooth_l1_loss(pred[:, 6:8], target[:, 6:8], reduction="none", beta=0.05) * active).sum() / denom_a
    l_rad = (F.smooth_l1_loss(pred[:, 8:9], target[:, 8:9], reduction="none", beta=0.04) * active).sum() / denom_a

    tot = 1.30 * l_prob + 0.85 * l_cls + 2.8 * l_off + 1.6 * l_ang + 1.4 * l_rad
    return tot, float(l_prob.item()), float(l_off.item()), float(l_ang.item()), float(l_rad.item())


def main():
    svg_tr_x, svg_tr_y, svg_va_x, svg_va_y = build_or_load_teacher_cache(tile_sz=384)
    v2_tr_x = np.load(V2_CACHE / "tr_x.npy")
    v2_tr_y = np.load(V2_CACHE / "tr_y.npy")
    v2_va_x = np.load(V2_CACHE / "va_x.npy")
    v2_va_y = np.load(V2_CACHE / "va_y.npy")

    # Balance ~1:1 between Teacher-SVG crops & V2 retention cache
    rng = np.random.default_rng(42)
    n_v2_use = min(len(v2_tr_x), max(len(svg_tr_x), 650))
    idx_v2 = rng.choice(len(v2_tr_x), size=n_v2_use, replace=False)

    tr_ds = MixedLineDataset(svg_tr_x, svg_tr_y, v2_tr_x[idx_v2], v2_tr_y[idx_v2], train=True, svg_weight=1.05, v2_weight=0.80)
    va_ds = MixedLineDataset(svg_va_x, svg_va_y, v2_va_x[:64], v2_va_y[:64], train=False)

    tr_loader = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    net = LineVectorNet(base=48, out_ch=9).to(device)
    if V2_CKPT.is_file():
        print(f"Initializing LineVectorNet from clean V2 checkpoint: {V2_CKPT}", flush=True)
        net.load_state_dict(torch.load(V2_CKPT, map_location=device, weights_only=True))

    epochs = 16
    opt = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 1e9
    t0 = time.time()
    print(f"Training Teacher-SVG LineVectorNet (V2 scales 3.0/3.2): svg_tr={len(svg_tr_x)}, v2_tr={n_v2_use}, va={len(va_ds)}", flush=True)
    for ep in range(1, epochs + 1):
        net.train()
        tr_l = 0.0
        for bx, by, bgw in tr_loader:
            bx, by, bgw = bx.to(device), by.to(device), bgw.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = net(bx)
            loss, _, _, _, _ = _vector_loss_v2(pred.float(), by, bgw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tr_l += float(loss.item()) * len(bx)
        sch.step()

        net.eval()
        va_l, va_p, va_o = 0.0, 0.0, 0.0
        with torch.no_grad():
            for bx, by, bgw in va_loader:
                bx, by, bgw = bx.to(device), by.to(device), bgw.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(bx)
                l, lp, lo, _, _ = _vector_loss_v2(pred.float(), by, bgw)
                va_l += float(l.item()) * len(bx)
                va_p += lp * len(bx)
                va_o += lo * len(bx)
        va_l /= max(1, len(va_ds))
        va_p /= max(1, len(va_ds))
        va_o /= max(1, len(va_ds))
        mark = ""
        if va_l < best_val:
            best_val = va_l
            torch.save(net.state_dict(), CKPT_OUT)
            mark = " *best"
        print(f"Ep {ep:02d}/{epochs} tr={tr_l/len(tr_ds):.4f} val={va_l:.4f} (prob={va_p:.4f}, off={va_o:.4f}) {time.time()-t0:.1f}s{mark}", flush=True)

    # Export ONNX to weights/line_vector_net.onnx
    net.load_state_dict(torch.load(CKPT_OUT, map_location=device, weights_only=True))
    net.eval().cpu()
    dummy = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        net,
        dummy,
        str(ONNX_OUT),
        input_names=["rgb"],
        output_names=["out"],
        dynamic_axes={"rgb": {0: "batch", 2: "height", 3: "width"}, "out": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        pt_out = net(dummy).numpy()
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"rgb": dummy.numpy()})[0]
    diff = float(np.max(np.abs(pt_out - ort_out)))
    kb = ONNX_OUT.stat().st_size / 1024.0
    print(f"Exported {ONNX_OUT} ({kb:.1f} KB), PT-vs-ORT max diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
