"""Phase 1c: Ingest, deduplicate, prepare native 2x crisp crops (when >= 2000px), and run
teacher models (netG + erika) on GPU for the 268 high-resolution images in `E:\\图集`,
strictly holding out all evaluation benchmarks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.lines import line_map

E_DIR = Path(r"E:\图集")
TEACHER_WEIGHTS = Path(r"D:\Lumina-Vec-0.9.2\weights")
CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
OUT_ROOT = CACHE_DIR / "e_gallery"
RGB_1X_DIR = OUT_ROOT / "rgb_1x"
NATIVE_2X_DIR = OUT_ROOT / "native_2x"
TEACHER_DIR = OUT_ROOT / "teachers"

EVAL_HOLDOUT_PATHS = [
    CACHE_DIR / "rgb_work" / "002_test1.png",
    CACHE_DIR / "rgb_work" / "004_test12.png",
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\A4.png"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\A5.png"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\回归1.jpg"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\bbb.png"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图集2\图集\2ED9A5F713EFAA837875DB903C72AA94.png"),
]


def _dhash_vec(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L").resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float32)
    return ((g - g.mean()) / (g.std() + 1e-5)).ravel()


def _load_rgb_alpha(path: Path) -> tuple[Image.Image, np.ndarray | None]:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            alpha = np.asarray(im)[..., 3].copy()
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, im).convert("RGB")
        else:
            alpha = None
            rgb = im.convert("RGB")
    return rgb, alpha


def _resize_max(rgb: Image.Image, alpha: np.ndarray | None, max_side: int) -> tuple[np.ndarray, np.ndarray | None]:
    w, h = rgb.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        nw, nh = round(w * s), round(h * s)
        rgb_r = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        if alpha is not None:
            alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        return np.asarray(rgb_r, np.uint8), alpha
    return np.asarray(rgb, np.uint8), alpha


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RGB_1X_DIR.mkdir(parents=True, exist_ok=True)
    NATIVE_2X_DIR.mkdir(parents=True, exist_ok=True)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)

    holdout_vecs: list[tuple[str, np.ndarray]] = []
    for p in EVAL_HOLDOUT_PATHS:
        if p.is_file():
            with Image.open(p) as im:
                holdout_vecs.append((p.name, _dhash_vec(im)))

    seen_vecs: list[tuple[str, np.ndarray]] = []
    for mp in (CACHE_DIR / "manifest.json", CACHE_DIR / "v2_gallery" / "v2_manifest.json"):
        if mp.is_file():
            for it in json.loads(mp.read_text(encoding="utf-8")):
                p = Path(it["rgb_path"])
                if p.is_file():
                    with Image.open(p) as im:
                        seen_vecs.append((it.get("stem", p.stem), _dhash_vec(im)))

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(p for p in E_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    print(f"Scanning {len(files)} images in {E_DIR} (seen={len(seen_vecs)}, holdout={len(holdout_vecs)})...", flush=True)

    manifest = []
    dup_skip, hold_skip = 0, 0
    t0 = time.time()
    for idx, p in enumerate(files):
        rgb_orig, alpha_orig = _load_rgb_alpha(p)
        ow, oh = rgb_orig.size
        vec = _dhash_vec(rgb_orig)

        if any(float(np.dot(vec, hv) / len(vec)) > 0.90 for _, hv in holdout_vecs):
            hold_skip += 1
            print(f"  [holdout skip] {p.name}", flush=True)
            continue
        if any(float(np.dot(vec, sv) / len(vec)) > 0.94 for _, sv in seen_vecs):
            dup_skip += 1
            continue
        seen_vecs.append((p.stem, vec))

        rgb_1x, alpha_1x = _resize_max(rgb_orig, alpha_orig, max_side=1600)
        h1, w1 = rgb_1x.shape[:2]
        rgb_1x_path = RGB_1X_DIR / f"eg_{idx:03d}_{p.stem[:32]}.png"
        if not rgb_1x_path.is_file():
            Image.fromarray(rgb_1x).save(rgb_1x_path, compress_level=1)

        # Save native 2x image (max_side=2400) directly when source is >= 2000px to avoid SR distortion
        native_2x_path = None
        if max(ow, oh) >= 2000:
            rgb_2x, _ = _resize_max(rgb_orig, None, max_side=2400)
            native_2x_path = NATIVE_2X_DIR / f"eg2x_{idx:03d}_{p.stem[:32]}.png"
            if not native_2x_path.is_file():
                Image.fromarray(rgb_2x).save(native_2x_path, compress_level=1)

        t_path = TEACHER_DIR / f"eg_{idx:03d}_{p.stem[:32]}.npz"
        if not t_path.is_file():
            dl = min(1536, max(h1, w1))
            m_anime = line_map(rgb_1x, kind="anime", detect_long=dl, weights_dir=str(TEACHER_WEIGHTS))
            m_manga = line_map(rgb_1x, kind="manga", detect_long=dl, weights_dir=str(TEACHER_WEIGHTS))
            np.savez_compressed(
                t_path,
                anime=(np.clip(m_anime, 0, 1) * 255).astype(np.uint8),
                manga=(np.clip(m_manga, 0, 1) * 255).astype(np.uint8),
                alpha=(alpha_1x if alpha_1x is not None else np.full((h1, w1), 255, np.uint8)),
            )

        manifest.append({
            "id": f"eg_{idx:03d}",
            "stem": p.stem,
            "orig_path": str(p),
            "rgb_path": str(rgb_1x_path),
            "native_2x_path": str(native_2x_path) if native_2x_path is not None else None,
            "teacher_path": str(t_path),
            "width": w1,
            "height": h1,
            "orig_wh": [ow, oh],
            "has_alpha": alpha_orig is not None,
        })
        if len(manifest) % 20 == 0 or (idx + 1) == len(files):
            print(f"  [{idx+1}/{len(files)}] kept={len(manifest)}, dup={dup_skip}, hold={hold_skip} ({time.time()-t0:.1f}s)", flush=True)

    out_m = OUT_ROOT / "e_manifest.json"
    out_m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done! Kept {len(manifest)} unique images (dup={dup_skip}, hold={hold_skip}) -> {out_m}", flush=True)


if __name__ == "__main__":
    main()
