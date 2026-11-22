"""Phase 4: Inspect ONNX feature maps on unseen benchmark images (A4, A5, bbb)
and audit deployment package size against the 500 MB budget."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.feature_routing import infer_feature_maps

BENCH_DIR = Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比")
OUT_DIR = ROOT / "tools" / "feature_train" / "work_cache" / "eval_maps"


def render_maps(name: str, rgb: np.ndarray, fm) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tw, th = 220, int(round(220 * rgb.shape[0] / rgb.shape[1]))
    panels = [("RGB", Image.fromarray(rgb).resize((tw, th), Image.LANCZOS))]
    for title, arr in [
        ("ch0:ink_geom", fm.ink_main_geom),
        ("ch1:strand", fm.fine_strand),
        ("ch2:delicate", fm.delicate_detail),
        ("ch3:wash", fm.mergeable_wash),
        ("surf:cel", fm.surface_probs[0]),
        ("surf:grad+grn", np.clip(fm.surface_probs[1] + fm.surface_probs[2], 0, 1)),
    ]:
        hm = cv2.applyColorMap((np.clip(arr, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
        panels.append((title, Image.fromarray(cv2.resize(hm, (tw, th), interpolation=cv2.INTER_AREA))))
    out = Image.new("RGB", (len(panels) * (tw + 4), th + 22), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    for i, (t, im) in enumerate(panels):
        out.paste(im, (i * (tw + 4) + 2, 18))
        dr.text((i * (tw + 4) + 4, 2), t, fill=(200, 0, 0))
    p = OUT_DIR / f"cnn_maps_{name}.jpg"
    out.save(p, quality=90)
    return p


def main():
    for name, fname in [("A5", "A5.png"), ("A4", "A4.png"), ("bbb", "bbb.png"), ("a2", "a2.png")]:
        rgb = np.asarray(Image.open(BENCH_DIR / fname).convert("RGB"))
        fm = infer_feature_maps(rgb)
        if fm is None:
            print(f"{name}: feature_routing returned None")
            continue
        p = render_maps(name, rgb, fm)
        print(f"{name}: summary={fm.summary} -> {p}", flush=True)

    # Deployment size audit
    code_bytes = sum(
        p.stat().st_size for p in (ROOT / "lumina_vec").rglob("*.py")
    ) + sum((ROOT / f).stat().st_size for f in ("webui.py", "index.html") if (ROOT / f).exists())
    onnx_bytes = sum(p.stat().st_size for p in (ROOT / "weights").glob("*.onnx"))
    print(f"\n[Deployment Size Audit]")
    print(f"  Project code (.py/.html): {code_bytes / 1e6:.2f} MB")
    print(f"  Distilled ONNX weights:   {onnx_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
