"""Phase 1b: Ingest, render, deduplicate, and audit the v2 expansion gallery
(`SVG图` pure-vector SVGs + `图集` high-res illustrations), strictly holding out
evaluation benchmarks (`test1`, `test12`, `A4`, `A5`, `reg1`, `bbb`, `Tarot`).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GALLERY_ROOT = Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图集2")
SVG_DIR = GALLERY_ROOT / "SVG图"
RASTER_DIR = GALLERY_ROOT / "图集"
CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
V2_DIR = CACHE_DIR / "v2_gallery"
SVG_OUT_DIR = V2_DIR / "svg_2x"
RASTER_OUT_DIR = V2_DIR / "raster_1x"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

EVAL_HOLDOUT_PATHS = [
    CACHE_DIR / "rgb_work" / "002_test1.png",
    CACHE_DIR / "rgb_work" / "004_test12.png",
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\A4.png"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\A5.png"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\回归1.jpg"),
    Path(r"C:\Users\30266\Desktop\自建模型\图片处理SVG\处理对比\bbb.png"),
    ROOT / "data" / "uploads" / "1744656b83574d3b9234f6293116ccb2.source",
]


def _dhash_vec(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L").resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float32)
    return ((g - g.mean()) / (g.std() + 1e-5)).ravel()


def _parse_svg_dims(svg_path: Path, max_side: int = 2200) -> tuple[int, int]:
    root = ET.parse(svg_path).getroot()
    vb = root.attrib.get("viewBox", "")
    parts = [float(x) for x in re.split(r"[\s,]+", vb.strip()) if x]
    if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
        vw, vh = parts[2], parts[3]
    else:
        vw, vh = 1600.0, 1600.0
    s = max_side / max(vw, vh)
    w = max(64, int(round(vw * s / 2.0) * 2))
    h = max(64, int(round(vh * s / 2.0) * 2))
    return w, h


def _render_svg_edge(svg_path: Path, out_png: Path, w: int, h: int, tmp_html: Path) -> None:
    uri = svg_path.resolve().as_uri()
    html = (
        f"<!doctype html><style>*{{margin:0;padding:0;overflow:hidden}}"
        f"body{{width:{w}px;height:{h}px;background:#fff}}"
        f"img{{width:{w}px;height:{h}px;object-fit:contain;display:block}}</style>"
        f'<img src="{uri}">'
    )
    tmp_html.write_text(html, encoding="utf-8")
    try:
        subprocess.run(
            [
                EDGE_EXE,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                f"--window-size={w},{h}",
                f"--screenshot={str(out_png)}",
                tmp_html.resolve().as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if tmp_html.exists():
            tmp_html.unlink()
    with Image.open(out_png) as im:
        rgb = im.convert("RGB")
        if rgb.size != (w, h):
            rgb = rgb.crop((0, 0, w, h))
        rgb.save(out_png, optimize=False)


def _load_rgb_clean(path: Path, max_side: int = 1600) -> Image.Image:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, im).convert("RGB")
        else:
            rgb = im.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        rgb = rgb.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    return rgb


def main():
    V2_DIR.mkdir(parents=True, exist_ok=True)
    SVG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load perceptual signatures of existing dataset + held-out benchmarks
    holdout_vecs: list[tuple[str, np.ndarray]] = []
    for p in EVAL_HOLDOUT_PATHS:
        if p.is_file():
            with Image.open(p) as im:
                holdout_vecs.append((p.name, _dhash_vec(im)))

    seen_vecs: list[tuple[str, np.ndarray]] = []
    manifest_v1 = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    for it in manifest_v1:
        p = Path(it["rgb_path"])
        if p.is_file():
            with Image.open(p) as im:
                seen_vecs.append((it["stem"], _dhash_vec(im)))

    entries = []
    tmp_html = V2_DIR / "_render_tmp.html"
    t0 = time.time()

    # 1. Render & deduplicate SVG图 (native crisp 2x resolution, max_side=2200)
    svg_files = sorted(SVG_DIR.glob("*.svg"))
    print(f"Rendering & deduplicating {len(svg_files)} SVGs from {SVG_DIR}...", flush=True)
    kept_svg, skipped_dup, skipped_holdout = 0, 0, 0
    for idx, p in enumerate(svg_files):
        out_png = SVG_OUT_DIR / f"svg_{idx:03d}_{p.stem[:40]}.png"
        if not out_png.is_file():
            w, h = _parse_svg_dims(p, max_side=2200)
            _render_svg_edge(p, out_png, w, h, tmp_html)
        with Image.open(out_png) as im:
            w, h = im.size
            vec = _dhash_vec(im)

        if any(float(np.dot(vec, hv) / len(vec)) > 0.90 for _, hv in holdout_vecs):
            skipped_holdout += 1
            out_png.unlink(missing_ok=True)
            continue
        if any(float(np.dot(vec, sv) / len(vec)) > 0.94 for _, sv in seen_vecs):
            skipped_dup += 1
            out_png.unlink(missing_ok=True)
            continue

        seen_vecs.append((p.stem, vec))
        kept_svg += 1
        entries.append({
            "id": f"svg_{idx:03d}",
            "kind": "svg_2x",
            "stem": p.stem,
            "orig_path": str(p),
            "rgb_path": str(out_png),
            "width": w,
            "height": h,
        })
    print(f"  SVG图 kept={kept_svg}, dup_skipped={skipped_dup}, holdout_skipped={skipped_holdout} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Ingest & deduplicate high-res raster images in 图集 (root only, skip 低质量二次元插画 thumbnails)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    raster_files = sorted(p for p in RASTER_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)
    print(f"Ingesting & deduplicating {len(raster_files)} high-res rasters from {RASTER_DIR}...", flush=True)
    kept_r, skipped_r_dup, skipped_r_hold = 0, 0, 0
    for idx, p in enumerate(raster_files):
        im_clean = _load_rgb_clean(p, max_side=1600)
        vec = _dhash_vec(im_clean)
        if any(float(np.dot(vec, hv) / len(vec)) > 0.90 for _, hv in holdout_vecs):
            skipped_r_hold += 1
            print(f"  [holdout skip] {p.name}", flush=True)
            continue
        if any(float(np.dot(vec, sv) / len(vec)) > 0.94 for _, sv in seen_vecs):
            skipped_r_dup += 1
            continue
        out_png = RASTER_OUT_DIR / f"gal_{idx:03d}_{p.stem[:32]}.png"
        im_clean.save(out_png)
        seen_vecs.append((p.stem, vec))
        kept_r += 1
        entries.append({
            "id": f"gal_{idx:03d}",
            "kind": "raster_1x",
            "stem": p.stem,
            "orig_path": str(p),
            "rgb_path": str(out_png),
            "width": im_clean.width,
            "height": im_clean.height,
        })
    print(f"  图集 kept={kept_r}, dup_skipped={skipped_r_dup}, holdout_skipped={skipped_r_hold}", flush=True)

    out_manifest = V2_DIR / "v2_manifest.json"
    out_manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(entries)} unique v2 images -> {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
