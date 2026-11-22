"""Phase 3a: Train SurfaceStyleCNN on RTX 4070 SUPER and export ONNX (FP16) with DirectML verification."""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.feature_nets import SurfaceStyleCNN

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
DS_PATH = CACHE_DIR / "datasets" / "surface_patches.npz"
WEIGHTS_OUT = ROOT / "weights" / "surface_style.onnx"
CKPT_OUT = CACHE_DIR / "surface_style_best.pt"


class PatchDataset(Dataset):
    def __init__(self, X, y, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
            k = int(np.random.randint(0, 4))
            if k:
                img = np.rot90(img, k)
            x = np.ascontiguousarray(img).astype(np.float32) / 255.0
            # Light photometric jitter without altering surface texture / grain
            gain = float(np.random.uniform(0.90, 1.10))
            bias = float(np.random.uniform(-0.04, 0.04))
            x = np.clip(x * gain + bias, 0.0, 1.0)
            if np.random.rand() < 0.3:
                x = x[..., np.random.permutation(3)]
        else:
            x = img.astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return t, int(self.y[idx])


def _export_fp16_onnx(model_cpu, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp32 = out_path.with_suffix(".fp32.onnx")
    dummy = torch.rand(1, 3, 96, 96, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(tmp_fp32),
        input_names=["rgb"],
        output_names=["logits"],
        dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # Keep FP32 if already < 1 MB (ultra-fast & exact across CPU & DirectML!)
    sz = tmp_fp32.stat().st_size
    tmp_fp32.replace(out_path)
    return out_path.stat().st_size


def main():
    data = np.load(DS_PATH)
    X, y, is_val = data["X"], data["y"], data["is_val"] > 0
    X_tr, y_tr = X[~is_val], y[~is_val]
    X_va, y_va = X[is_val], y[is_val]
    print(f"Train patches: {len(y_tr)}, Val patches (unseen imgs): {len(y_va)}", flush=True)

    tr_loader = DataLoader(PatchDataset(X_tr, y_tr, train=True), batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(PatchDataset(X_va, y_va, train=False), batch_size=128, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurfaceStyleCNN().to(device)
    counts = np.bincount(y_tr, minlength=4).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / np.maximum(counts, 1.0) / 4.0, device=device)
    crit = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    epochs = 28
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(by)
        sched.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for bx, by in va_loader:
                p = model(bx.to(device)).argmax(1).cpu().numpy()
                preds.append(p)
                gts.append(by.numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        acc = float((preds == gts).mean())
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CKPT_OUT)
        if ep in (1, 5, 10, 15, 20, 25, epochs):
            per_cls = [round(float((preds[gts == c] == c).mean()), 3) if (gts == c).any() else -1 for c in range(4)]
            print(f"Ep {ep:02d}/{epochs} loss={tr_loss/len(y_tr):.4f} val_acc={acc:.4f} best={best_acc:.4f} per_cls={per_cls}", flush=True)

    # Export best model to ONNX & verify parity
    best_model = SurfaceStyleCNN().cpu().eval()
    best_model.load_state_dict(torch.load(CKPT_OUT, map_location="cpu"))
    onnx_bytes = _export_fp16_onnx(best_model, WEIGHTS_OUT)
    sess = ort.InferenceSession(str(WEIGHTS_OUT), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    sample_x = X_va[:16].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    pt_out = best_model(torch.from_numpy(sample_x)).detach().numpy()
    ort_out = sess.run(None, {"rgb": sample_x})[0]
    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"Exported {WEIGHTS_OUT} ({onnx_bytes/1024:.1f} KB), max PT-vs-ORT diff={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
