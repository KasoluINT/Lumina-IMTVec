"""Phase 5a: Vectorize selected hard images from 图片集 and compute SVG-vs-Exemplar defect maps
(missing lines vs fragmented clutter) for reverse CNN fine-tuning & visual inspection."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.color import srgb_to_oklab

PY = r"D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SCRATCH = Path(r"C:\Users\30266\.gemini\antigravity\brain\439cca1c-a31d-4f73-81c2-1f9c82b1e720\scratch")
DATASET_DIR = Path(r"C:\Users\30266\Desktop\自建模型\图片处理AI\图片集")
OUT_DIR = ROOT / "tools" / "feature_train" / "work_cache" / "defect_mining"


def render_svg_png(svg_path: Path, png_path: Path, w: int, h: int):
    html_p = OUT_DIR / "tmp_render.html"
    svg_u = str(svg_path).replace("\\", "/")
    html_p.write_text(
        f'<html><body style="margin:0;background:#fff"><img src="file:///{svg_u}" width="{w}" height="{h}"></body></html>',
        encoding="utf-8",
    )
    subprocess.run(
        [EDGE, "--headless=new", "--disable-gpu", f"--screenshot={png_path}", f"--window-size={w},{h}", "file:///" + str(html_p).replace("\\", "/")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_svg_lines_only(svg_path: Path, png_path: Path, w: int, h: int):
    """Extract fill paths as white-filled stroke boundaries to visualize SVG region splits cleanly."""
    import re
    txt = svg_path.read_text(encoding="utf-8", errors="ignore")
    # Convert all fill paths to white fill + original colour 1.2px stroke so region borders appear starkly
    def _rep(m):
        tag = m.group(0)
        fm = re.search(r'fill="(#[0-9a-fA-F]{6})"', tag)
        col = fm.group(1) if fm else "#000000"
        if col.lower() in ("#ffffff", "#fefefe", "#fdfdfd"):
            col = "#cccccc"
        tag = re.sub(r'fill="[^"]*"', 'fill="none"', tag)
        tag = re.sub(r'stroke="[^"]*"', f'stroke="{col}"', tag)
        if 'stroke="' not in tag:
            tag = tag[:-2] + f' stroke="{col}" stroke-width="1.0"/>' if tag.endswith("/>") else tag[:-1] + f' stroke="{col}" stroke-width="1.0">'
        return tag
    # Simplify via white background SVG with thin strokes
    line_svg = re.sub(r"<path\b[^>]*>", _rep, txt)
    tmp_svg = OUT_DIR / (svg_path.stem + "_lines.svg")
    tmp_svg.write_text(line_svg, encoding="utf-8")
    render_svg_png(tmp_svg, png_path, w, h)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Vectorize 4 hard stress images from 图片集
    hard_imgs = ["test1.png", "test11.jpg", "test12.jpg", "test26.png"]
    rows = []
    for fname in hard_imgs:
        src_p = DATASET_DIR / fname
        im = Image.open(src_p).convert("RGB")
        w, h = im.size
        svg_p = OUT_DIR / f"{Path(fname).stem}.svg"
        png_p = OUT_DIR / f"{Path(fname).stem}_proj.png"
        ln_p = OUT_DIR / f"{Path(fname).stem}_lines.png"
        if not svg_p.exists():
            print(f"Vectorizing {fname} ({w}x{h})...", flush=True)
            subprocess.run(
                [PY, "-m", "lumina_vec.pipeline", str(src_p), str(svg_p), "--structure", "print-regions", "--bands", "84", "--gradient-mode", "clean", "--smooth-lines", "--no-ai"],
                check=True,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )
        if not png_p.exists():
            render_svg_png(svg_p, png_p, w, h)
        if not ln_p.exists():
            render_svg_lines_only(svg_p, ln_p, w, h)

        # Build 3-panel + zoomed center crop row for visual reading
        pj = Image.open(png_p).convert("RGB")
        pl = Image.open(ln_p).convert("RGB")
        tw, th = 260, int(round(260 * h / w))
        # Also zoom center 45% crop
        cx0, cy0, cx1, cy1 = int(0.25 * w), int(0.15 * h), int(0.75 * w), int(0.60 * h)
        zw, zh = 260, int(round(260 * (cy1 - cy0) / (cx1 - cx0)))
        canvas = Image.new("RGB", (tw * 3 + 12, max(th, zh) * 2 + 44), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for c, (t, img) in enumerate([("orig", im), ("proj_color", pj), ("proj_line", pl)]):
            canvas.paste(img.resize((tw, th), Image.LANCZOS), (c * (tw + 4), 20))
            canvas.paste(img.crop((cx0, cy0, cx1, cy1)).resize((zw, zh), Image.LANCZOS), (c * (tw + 4), th + 40))
            dr.text((c * (tw + 4) + 4, 2), f"{Path(fname).stem} {t}", fill=(200, 0, 0))
            dr.text((c * (tw + 4) + 4, th + 22), f"zoom {t}", fill=(200, 0, 0))
        out_cmp = OUT_DIR / f"inspect_{Path(fname).stem}.jpg"
        canvas.save(out_cmp, quality=90)
        print("Saved:", out_cmp, flush=True)


if __name__ == "__main__":
    main()
