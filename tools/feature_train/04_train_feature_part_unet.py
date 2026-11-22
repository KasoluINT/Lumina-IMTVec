"""Phase 3b: Train FeaturePartUNet on RTX 4070 SUPER and export ONNX (with DirectML verification)."""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.feature_nets import FeaturePartUNet

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
TILES_JSON = CACHE_DIR / "datasets" / "part_tiles.json"
WEIGHTS_OUT = ROOT / "weights" / "feature_part.onnx"
CKPT_OUT = CACHE_DIR / "feature_part_best.pt"


class TileDataset(Dataset):
    def __init__(self, items, train=True):
        self.train = train
        self.xs, self.ys = [], []
        for m in items:
            d = np.load(m["path"])
            self.xs.append(d["x"])
            self.ys.append(d["y"])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
                y = y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[::-1, :]
                y = y[:, ::-1, :]
            k = int(np.random.randint(0, 4))
            if k:
                x = np.rot90(x, k)
                y = np.rot90(y, k, axes=(1, 2))
            xf = np.ascontiguousarray(x).astype(np.float32) / 255.0
            yf = np.ascontiguousarray(y).astype(np.float32) / 255.0
            gain = float(np.random.uniform(0.90, 1.10))
            bias = float(np.random.uniform(-0.04, 0.04))
            xf = np.clip(xf * gain + bias, 0.0, 1.0)
            if np.random.rand() < 0.25:
                xf = xf[..., np.random.permutation(3)]
        else:
            xf = x.astype(np.float32) / 255.0
            yf = y.astype(np.float32) / 255.0
        return torch.from_numpy(xf).permute(2, 0, 1).contiguous(), torch.from_numpy(yf).contiguous()


def multi_channel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cw = torch.tensor([1.0, 1.3, 2.0, 1.0], device=pred.device).view(1, 4, 1, 1)
    l1 = torch.abs(pred - target) * cw
    p_c = pred.clamp(1e-4, 1.0 - 1e-4)
    bce = -(target * torch.log(p_c) + (1.0 - target) * torch.log(1.0 - p_c)) * cw
    inter = (pred * target).sum(dim=(2, 3))
    denom = (pred + target).sum(dim=(2, 3)) + 1e-4
    dice = 1.0 - (2.0 * inter / denom)
    return 0.65 * l1.mean() + 0.35 * bce.mean() + 0.25 * (dice * cw.view(1, 4)).mean()


def main():
    meta = json.loads(TILES_JSON.read_text(encoding="utf-8"))
    tr_items = [m for m in meta if not m["is_val"]]
    va_items = [m for m in meta if m["is_val"]]
    print(f"Preloading tiles into RAM: train={len(tr_items)}, val={len(va_items)}...", flush=True)

    tr_loader = DataLoader(TileDataset(tr_items, train=True), batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(TileDataset(va_items, train=False), batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeaturePartUNet(base=48, out_ch=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=2e-4)
    epochs = 24
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = 1e9
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        tr_l = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(bx)
            loss = multi_channel_loss(pred.float(), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_l += float(loss.item()) * len(bx)
        sched.step()

        model.eval()
        va_l = 0.0
        ch_mae = np.zeros(4, np.float64)
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    pred = model(bx)
                pred_f = pred.float()
                va_l += float(multi_channel_loss(pred_f, by).item()) * len(bx)
                ch_mae += torch.abs(pred_f - by).mean(dim=(0, 2, 3)).cpu().numpy() * len(bx)
        va_l /= max(1, len(va_items))
        ch_mae /= max(1, len(va_items))
        if va_l <= best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), CKPT_OUT)
        print(
            f"Ep {ep:02d}/{epochs} ({time.time()-t_ep:.1f}s) tr={tr_l/len(tr_items):.4f} val={va_l:.4f} best={best_loss:.4f} "
            f"mae={[round(float(v), 4) for v in ch_mae]}",
            flush=True,
        )

    best_model = FeaturePartUNet(base=48, out_ch=4).cpu().eval()
    best_model.load_state_dict(torch.load(CKPT_OUT, map_location="cpu"))
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    torch.onnx.export(
        best_model,
        dummy,
        str(WEIGHTS_OUT),
        input_names=["rgb"],
        output_names=["maps"],
        dynamic_axes={"rgb": {0: "b", 2: "h", 3: "w"}, "maps": {0: "b", 2: "h", 3: "w"}},
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(WEIGHTS_OUT), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    pt_y = best_model(dummy).detach().numpy()
    ort_y = sess.run(None, {"rgb": dummy.numpy()})[0]
    diff = float(np.max(np.abs(pt_y - ort_y)))
    print(f"Exported {WEIGHTS_OUT} ({WEIGHTS_OUT.stat().st_size/1024:.1f} KB), max PT-vs-ORT diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
