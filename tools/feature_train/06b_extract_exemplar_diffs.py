"""Phase 5b: Extract inverse supervision maps directly from (Project SVG vs Exemplar SVG) pairs.
Where Project has noisy/chaotic boundaries absent in Exemplar -> push mergeable_wash=1, ink/strand/detail=0.
Where Exemplar has clean strokes/boundaries missed by Project -> push ink_geom/fine_strand/delicate_detail=1.
Also renders visual diff overlays so we can inspect them directly."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.color import srgb_to_oklab

SCRATCH = Path(r"C:\Users\30266\.gemini\antigravity\brain\439cca1c-a31d-4f73-81c2-1f9c82b1e720\scratch")
BENCH_DIR = Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比")
OUT_DIR = ROOT / "tools" / "feature_train" / "work_cache" / "defect_mining"

PAIRS = [
    ("a1", "a1.jpg", SCRATCH / "a1cnn"),
    ("a2", "a2.png", SCRATCH / "a2cnn2"),
    ("A3", "A3.png", SCRATCH / "a3reg"),
    ("A4", "A4.png", SCRATCH / "a4cnn2"),
    ("bbb", "bbb.png", SCRATCH / "bbbcnn"),
]


def _edge_strength_oklab(rgb: np.ndarray) -> np.ndarray:
    lab = srgb_to_oklab(rgb).astype(np.float32)
    gx = np.sqrt(
        cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3) ** 2
        + 2.2 * cv2.Sobel(lab[..., 1], cv2.CV_32F, 1, 0, ksize=3) ** 2
        + 2.2 * cv2.Sobel(lab[..., 2], cv2.CV_32F, 1, 0, ksize=3) ** 2
    )
    gy = np.sqrt(
        cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3) ** 2
        + 2.2 * cv2.Sobel(lab[..., 1], cv2.CV_32F, 0, 1, ksize=3) ** 2
        + 2.2 * cv2.Sobel(lab[..., 2], cv2.CV_32F, 0, 1, ksize=3) ** 2
    )
    return np.hypot(gx, gy)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, orig_n, sub in PAIRS:
        if not (sub / "proj_full.png").exists():
            print(f"Skipping {name}, missing {sub}", flush=True)
            continue
        src = np.asarray(Image.open(BENCH_DIR / orig_n).convert("RGB"))
        proj = np.asarray(Image.open(sub / "proj_full.png").convert("RGB"))
        ref = np.asarray(Image.open(sub / "ref_full.png").convert("RGB"))

        e_proj = _edge_strength_oklab(proj)
        e_ref = _edge_strength_oklab(ref)
        # Allow 2-px tolerance around exemplar boundaries
        ref_dil = cv2.dilate((e_ref > 0.12).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        proj_dil = cv2.dilate((e_proj > 0.12).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

        # 1. Excess clutter edges in proj absent in ref (colour banding / fragment clutter -> should merge!)
        clutter_mask = (e_proj > 0.14) & (~ref_dil)
        # 2. Missing boundaries/lines in ref absent in proj (broken hair strands / missing thin lines!)
        missing_mask = (e_ref > 0.16) & (~proj_dil)
        # 3. Large colour discrepancy (>0.06 OKLab) due to over-simplification or wrong tint
        lab_p = srgb_to_oklab(proj)
        lab_r = srgb_to_oklab(ref)
        color_err = np.linalg.norm(lab_p - lab_r, axis=-1)
        wrong_tint = (color_err > 0.06) & (~clutter_mask)

        # Save compressed supervision pack for reverse CNN training
        np.savez_compressed(
            OUT_DIR / f"diff_{name}.npz",
            src=src,
            clutter=(np.clip(clutter_mask.astype(np.float32), 0, 1) * 255).astype(np.uint8),
            missing=(np.clip(missing_mask.astype(np.float32), 0, 1) * 255).astype(np.uint8),
            ref_edge=(np.clip(e_ref / 0.6, 0, 1) * 255).astype(np.uint8),
            wrong_tint=(np.clip((color_err - 0.04) / 0.12, 0, 1) * 255).astype(np.uint8),
        )
        print(
            f"{name}: clutter={clutter_mask.mean():.4f}, missing={missing_mask.mean():.4f}, "
            f"wrong_tint={wrong_tint.mean():.4f}",
            flush=True,
        )

        # Visual defect overlay (Red = clutter/should merge, Green = missing line/detail, Blue = color err)
        h, w = src.shape[:2]
        tw, th = 280, int(round(280 * h / w))
        overlay = (src.astype(np.float32) * 0.45).astype(np.uint8)
        overlay[clutter_mask] = [255, 50, 50]
        overlay[missing_mask] = [40, 245, 80]
        summary_rows.append((name, Image.fromarray(src).resize((tw, th)), Image.fromarray(overlay).resize((tw, th)), Image.fromarray(ref).resize((tw, th))))

    if summary_rows:
        mh = max(r[1].height for r in summary_rows) + 22
        canvas = Image.new("RGB", (284 * 3, len(summary_rows) * mh), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for idx, (name, im_s, im_o, im_r) in enumerate(summary_rows):
            y = idx * mh
            canvas.paste(im_s, (2, y + 18))
            canvas.paste(im_o, (286, y + 18))
            canvas.paste(im_r, (570, y + 18))
            dr.text((4, y + 2), f"{name} orig", fill=(0, 0, 0))
            dr.text((288, y + 2), "Red=clutter/merge  Green=missed_line", fill=(200, 0, 0))
            dr.text((572, y + 2), f"{name} exemplar", fill=(0, 0, 0))
        out_p = OUT_DIR / "svg_defect_diff.jpg"
        canvas.save(out_p, quality=90)
        print("Saved:", out_p, flush=True)


if __name__ == "__main__":
    main()
