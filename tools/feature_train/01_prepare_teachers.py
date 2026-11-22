"""Phase 1: Scan dataset images, standardize RGB, run teacher models (netG + erika) on GPU,
and export curated visual inspection sheets for surface patches and fine-detail regions."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.lines import line_map

DATASET_DIR = Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图片集")
WEIGHTS_DIR = Path(r"D:\Lumina-Vec-0.9.2\weights")
CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
TEACHER_DIR = CACHE_DIR / "teachers"
RGB_DIR = CACHE_DIR / "rgb_work"

# Visual style classification curated from inspecting dataset_sheet_1.jpg & dataset_sheet_2.jpg
CLEAN_CEL_STEMS = {
    "3UQAI(TY2(NWDG%84AQXV[1_tmb", "test3", "test5", "test6", "test16", "test17",
    "test19", "test20", "test42", "test47", "test50",
    # Stained-glass set (crisp hard fills + thick black leadlines)
    "test9", "test11", "test15", "test31", "test33", "test34", "test37", "test38", "test39", "test45",
}
SOFT_GRADIENT_STEMS = {
    "test14", "test21", "test22", "test25", "test26", "test28", "test30",
    "test32", "test40", "test41", "test43", "test46", "test48", "test7", "test8",
}
GRAIN_PAINT_STEMS = {
    "test1", "test2", "test4", "test12", "test13", "test18", "test23",
    "test24", "test35", "test36", "test44",
}


def load_clean_rgb(path: Path, max_side: int = 1600):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = np.asarray(im)[..., 3]
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        rgb_im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        alpha = None
        rgb_im = im.convert("RGB")
    w, h = rgb_im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        nw, nh = round(w * s), round(h * s)
        rgb_im = rgb_im.resize((nw, nh), Image.LANCZOS)
        if alpha is not None:
            alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.asarray(rgb_im, np.uint8), alpha


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    RGB_DIR.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(p for p in DATASET_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)
    print(f"Scanning {len(files)} images in {DATASET_DIR}...", flush=True)

    manifest = []
    for idx, p in enumerate(files):
        stem = p.stem
        rgb, alpha = load_clean_rgb(p, max_side=1600)
        h, w = rgb.shape[:2]

        if stem in CLEAN_CEL_STEMS:
            style = "clean_cel"
        elif stem in SOFT_GRADIENT_STEMS:
            style = "soft_gradient"
        elif stem in GRAIN_PAINT_STEMS:
            style = "grain_paint"
        else:
            style = "mixed"

        rgb_path = RGB_DIR / f"{idx:03d}_{stem}.png"
        Image.fromarray(rgb).save(rgb_path)

        # Save teacher maps as 8-bit PNG pair inside npz or dual PNGs (compact & fast)
        t_path = TEACHER_DIR / f"{idx:03d}_{stem}.npz"
        if not t_path.exists():
            dl = min(1536, max(h, w))
            m_anime = line_map(rgb, kind="anime", detect_long=dl, weights_dir=str(WEIGHTS_DIR))
            m_manga = line_map(rgb, kind="manga", detect_long=dl, weights_dir=str(WEIGHTS_DIR))
            np.savez_compressed(
                t_path,
                anime=(np.clip(m_anime, 0, 1) * 255).astype(np.uint8),
                manga=(np.clip(m_manga, 0, 1) * 255).astype(np.uint8),
                alpha=(alpha if alpha is not None else np.full((h, w), 255, np.uint8)),
            )

        # Quick grain measurement inside non-white pixels
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        fg = (alpha > 128) if alpha is not None else (gray < 0.98)
        grain = float(np.median(lap[fg])) if fg.any() else 0.0

        manifest.append({
            "id": idx,
            "stem": stem,
            "orig_path": str(p),
            "rgb_path": str(rgb_path),
            "teacher_path": str(t_path),
            "width": w,
            "height": h,
            "style": style,
            "grain_lap": round(grain, 5),
            "has_alpha": alpha is not None,
        })
        if (idx + 1) % 10 == 0 or idx + 1 == len(files):
            print(f"  [{idx+1}/{len(files)}] {stem} ({w}x{h}) style={style} grain={grain:.4f}", flush=True)

    m_path = CACHE_DIR / "manifest.json"
    m_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest ({len(manifest)} items) -> {m_path}", flush=True)


if __name__ == "__main__":
    main()
