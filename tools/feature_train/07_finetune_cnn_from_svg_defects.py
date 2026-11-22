"""Phase 5c: Reverse fine-tuning of FeaturePartUNet using mined SVG-vs-Exemplar defect maps
(clutter -> mergeable_wash, missing_line -> ink/strand/detail, wrong_tint -> delicate_detail)
mixed 50/50 with original 图片集 tiles so generalisation improves without forgetting."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.feature_nets import FeaturePartUNet

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
TILES_JSON = CACHE_DIR / "datasets" / "part_tiles.json"
DEFECT_DIR = CACHE_DIR / "defect_mining"
CKPT_IN = CACHE_DIR / "feature_part_best.pt"
CKPT_OUT = CACHE_DIR / "feature_part_ft_best.pt"
WEIGHTS_OUT = ROOT / "weights" / "feature_part.onnx"


def build_defect_tiles(rng, stage1_ckpt: Path, device: torch.device):
    """Crop 512x512 hard-example tiles centred on SVG defects (clutter, missing), keeping ch2
    (delicate_detail) strictly semantic (compact facial features from Stage-1 teacher) instead
    of polluting ch2 with cloth/kimono color diffs."""
    base_net = FeaturePartUNet(base=48, out_ch=4).to(device).eval()
    base_net.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    tiles = []
    for npz_p in sorted(DEFECT_DIR.glob("diff_*.npz")):
        d = np.load(npz_p)
        src = d["src"]
        clutter = d["clutter"].astype(np.float32) / 255.0
        missing = d["missing"].astype(np.float32) / 255.0
        ref_edge = d["ref_edge"].astype(np.float32) / 255.0
        wrong_tint = d["wrong_tint"].astype(np.float32) / 255.0
        h, w = src.shape[:2]

        # Query Stage-1 semantic model at 512x512 so ch2 preserves face/eye/mouth semantics cleanly.
        s512 = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            x_t = torch.from_numpy(s512.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            p1 = base_net(x_t)[0].cpu().numpy()
        base_ch1 = cv2.resize(p1[1], (w, h), interpolation=cv2.INTER_LINEAR)
        base_ch2 = cv2.resize(p1[2], (w, h), interpolation=cv2.INTER_LINEAR)

        # Build refined target [4, H, W] and weight [4, H, W]
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        dark_mask = np.clip((0.40 - gray) / 0.30, 0.0, 1.0)
        t_ch0 = np.clip(np.maximum(ref_edge * dark_mask, missing * dark_mask), 0.0, 1.0)
        # Exemplar thin lines, flying hair wisps & missing contours belong cleanly to ch1 (fine_strand)
        t_ch1 = np.clip(np.maximum(base_ch1 * 0.65, np.maximum(ref_edge * (1.0 - 0.35 * dark_mask), missing * 0.95)), 0.0, 1.0)
        # Keep ch2 strictly on facial/delicate zones (boosted only if missing/wrong_tint lies INSIDE facial mask)
        facial_mask = cv2.dilate((base_ch2 > 0.20).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        t_ch2 = np.clip(np.where(facial_mask, np.maximum(base_ch2, 0.85 * wrong_tint), base_ch2 * 0.85), 0.0, 1.0)
        # Clutter edges where exemplar stayed flat MUST be merged (ch3=1, ch0/1=0)
        line_block = np.maximum(t_ch0, np.maximum(t_ch1, t_ch2))
        ref_smooth = 1.0 - cv2.dilate((ref_edge > 0.20).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        t_ch3 = np.clip(np.maximum(clutter, 0.65 * ref_smooth) * (1.0 - 0.85 * line_block), 0.0, 1.0)

        # Suppress false positive clutter on structural lines & strands (do NOT erase facial base_ch2)
        sup = clutter > 0.5
        t_ch0[sup] = 0.0; t_ch1[sup] = 0.0; t_ch2[sup & (~facial_mask)] = 0.0

        tgt = (np.clip(np.stack([t_ch0, t_ch1, t_ch2, t_ch3], axis=0), 0.0, 1.0) * 255).astype(np.uint8)
        wgt = np.ones((4, h, w), np.float32)
        wgt[0] += 3.5 * missing + 2.5 * clutter
        wgt[1] += 4.5 * missing + 2.5 * clutter
        wgt[2] += np.where(facial_mask, 3.0 * wrong_tint + 2.0 * missing, 1.5 * clutter)
        wgt[3] += 3.5 * clutter + 2.5 * missing

        r_full = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)
        y_full = np.stack([cv2.resize(tgt[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(4)], axis=0)
        w_full = np.stack([cv2.resize(wgt[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(4)], axis=0)
        tiles.append((r_full, y_full, w_full))

        if min(h, w) >= 512:
            err_score = clutter + 1.5 * missing + (wrong_tint * facial_mask.astype(np.float32))
            ys, xs = np.where(err_score > np.quantile(err_score, 0.88))
            for _ in range(14):
                if len(ys):
                    pick = int(rng.integers(0, len(ys)))
                    cy, cx = int(ys[pick]), int(xs[pick])
                    y0 = int(np.clip(cy - 256 + int(rng.integers(-64, 65)), 0, h - 512))
                    x0 = int(np.clip(cx - 256 + int(rng.integers(-64, 65)), 0, w - 512))
                else:
                    y0 = int(rng.integers(0, h - 512 + 1))
                    x0 = int(rng.integers(0, w - 512 + 1))
                tiles.append((src[y0:y0+512, x0:x0+512], tgt[:, y0:y0+512, x0:x0+512], wgt[:, y0:y0+512, x0:x0+512]))
    return tiles


class MixedDefectDataset(Dataset):
    def __init__(self, orig_items, defect_tiles, train=True):
        self.train = train
        self.samples = []
        for m in orig_items:
            d = np.load(m["path"])
            self.samples.append((d["x"], d["y"], np.ones((4, 512, 512), np.float32)))
        for x, y, w in defect_tiles:
            self.samples.append((x, y, w))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, w = self.samples[idx]
        if self.train:
            if np.random.rand() < 0.5:
                x, y, w = x[:, ::-1], y[:, :, ::-1], w[:, :, ::-1]
            if np.random.rand() < 0.5:
                x, y, w = x[::-1, :], y[:, ::-1, :], w[:, ::-1, :]
            k = int(np.random.randint(0, 4))
            if k:
                x = np.rot90(x, k)
                y = np.rot90(y, k, axes=(1, 2))
                w = np.rot90(w, k, axes=(1, 2))
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
            yf = np.ascontiguousarray(y).astype(np.float32) / 255.0
            wf = np.ascontiguousarray(w).astype(np.float32)
            xf = np.clip(xf * float(np.random.uniform(0.92, 1.08)) + float(np.random.uniform(-0.03, 0.03)), 0.0, 1.0)
        else:
            xf = x.astype(np.float32) / 255.0
            yf = y.astype(np.float32) / 255.0
            wf = w.astype(np.float32)
        return (
            torch.from_numpy(xf).permute(2, 0, 1).contiguous(),
            torch.from_numpy(yf).contiguous(),
            torch.from_numpy(wf).contiguous(),
        )


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, wgt: torch.Tensor) -> torch.Tensor:
    cw = torch.tensor([1.0, 1.4, 1.8, 1.1], device=pred.device).view(1, 4, 1, 1) * wgt
    l1 = torch.abs(pred - target) * cw
    p_c = pred.clamp(1e-4, 1.0 - 1e-4)
    bce = -(target * torch.log(p_c) + (1.0 - target) * torch.log(1.0 - p_c)) * cw
    inter = (pred * target).sum(dim=(2, 3))
    denom = (pred + target).sum(dim=(2, 3)) + 1e-4
    dice = 1.0 - (2.0 * inter / denom)
    return 0.55 * l1.mean() + 0.30 * bce.mean() + 0.25 * dice.mean()


def main():
    rng = np.random.default_rng(7)
    meta = json.loads(TILES_JSON.read_text(encoding="utf-8"))
    tr_orig = [m for m in meta if not m["is_val"]]
    va_orig = [m for m in meta if m["is_val"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defect_tiles = build_defect_tiles(rng, CKPT_IN, device)
    print(f"Loaded orig train={len(tr_orig)}, val={len(va_orig)}, defect hard tiles={len(defect_tiles)}", flush=True)

    tr_loader = DataLoader(MixedDefectDataset(tr_orig, defect_tiles, train=True), batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(MixedDefectDataset(va_orig, [], train=False), batch_size=4, shuffle=False, num_workers=0)

    model = FeaturePartUNet(base=48, out_ch=4).to(device)
    model.load_state_dict(torch.load(CKPT_IN, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=6.0e-4, weight_decay=2e-4)
    epochs = 14
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = 1e9
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_l = 0.0
        for bx, by, bw in tr_loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(bx)
            loss = weighted_loss(pred.float(), by, bw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_l += float(loss.item()) * len(bx)
        sched.step()

        model.eval()
        va_l = 0.0
        ch_mae = np.zeros(4, np.float64)
        with torch.no_grad():
            for bx, by, bw in va_loader:
                bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    pred = model(bx)
                pred_f = pred.float()
                va_l += float(weighted_loss(pred_f, by, bw).item()) * len(bx)
                ch_mae += torch.abs(pred_f - by).mean(dim=(0, 2, 3)).cpu().numpy() * len(bx)
        va_l /= max(1, len(va_orig))
        ch_mae /= max(1, len(va_orig))
        if va_l <= best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), CKPT_OUT)
        print(
            f"FT Ep {ep:02d}/{epochs} ({time.time()-t0:.1f}s) tr={tr_l/len(tr_loader.dataset):.4f} "
            f"val={va_l:.4f} mae={[round(float(v), 4) for v in ch_mae]}",
            flush=True,
        )

    best_model = FeaturePartUNet(base=48, out_ch=4).cpu().eval()
    best_model.load_state_dict(torch.load(CKPT_OUT, map_location="cpu"))
    dummy = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    torch.onnx.export(
        best_model, dummy, str(WEIGHTS_OUT),
        input_names=["rgb"], output_names=["maps"],
        dynamic_axes={"rgb": {0: "b", 2: "h", 3: "w"}, "maps": {0: "b", 2: "h", 3: "w"}},
        opset_version=17, dynamo=False,
    )
    sess = ort.InferenceSession(str(WEIGHTS_OUT), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    diff = float(np.max(np.abs(best_model(dummy).detach().numpy() - sess.run(None, {"rgb": dummy.numpy()})[0])))
    print(f"Re-exported {WEIGHTS_OUT} ({WEIGHTS_OUT.stat().st_size/1024:.1f} KB), PT-vs-ORT diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
