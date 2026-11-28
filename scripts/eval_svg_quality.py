"""Perceptual & structural SVG quality benchmark (inspired by StarVector SVG-Bench & RLRF).

Evaluates a generated SVG against its source raster image by rendering the SVG at intrinsic
resolution via headless Edge and computing:
  - DinoScore (DINOv2-base cosine similarity in [0, 1], optional when torch+transformers exist)
  - Structural Edge F1 (Sobel/Canny contour preservation)
  - OKLab ΔE (mean & p95 perceptual color error) and PSNR
  - SVG Compactness (paths, line/fill split, unique hex colors, KB, arc/primitive commands)
  - RLRF-style Efficiency Index: DinoScore - 0.02 * log10(max(10, paths))
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from lumina_vec.color import srgb_to_oklab

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
_DINO_CACHE = {}


def _svg_size(svg_text: str) -> tuple[int, int]:
    head = svg_text[:6000]
    m = re.search(r'<svg[^>]*width="([0-9.]+)"[^>]*height="([0-9.]+)"', head)
    if m:
        return int(round(float(m.group(1)))), int(round(float(m.group(2))))
    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', head)
    if m:
        return int(round(float(m.group(1)))), int(round(float(m.group(2))))
    return 1024, 1024


def render_svg_edge(svg_path: pathlib.Path, out_png: pathlib.Path, w: int, h: int, bg: str = "#ffffff") -> Image.Image:
    w_cap, h_cap = min(w, 4096), min(h, 4096)
    html_path = out_png.with_suffix(".html")
    html_path.write_text(
        f'<!doctype html><html><body style="margin:0;width:{w_cap}px;height:{h_cap}px;overflow:hidden;background:{bg}">'
        f'<img src="{svg_path.resolve().as_uri()}" style="display:block;width:{w_cap}px;height:{h_cap}px"></body></html>',
        encoding="utf-8",
    )
    subprocess.run(
        [
            EDGE,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={w_cap},{h_cap}",
            f"--screenshot={out_png}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    return Image.open(out_png).convert("RGB")


def compute_dino_score(im_a: Image.Image, im_b: Image.Image) -> float | None:
    """DINOv2-base cosine similarity mapped to [0, 1] (StarVector SVG-Bench metric)."""
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoImageProcessor, AutoModel
    except Exception:
        return None

    if "model" not in _DINO_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "facebook/dinov2-base"
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        _DINO_CACHE["model"] = model
        _DINO_CACHE["proc"] = proc
        _DINO_CACHE["device"] = device

    model = _DINO_CACHE["model"]
    proc = _DINO_CACHE["proc"]
    device = _DINO_CACHE["device"]
    with torch.no_grad():
        fa = model(**proc(images=im_a, return_tensors="pt").to(device)).last_hidden_state.mean(dim=1)
        fb = model(**proc(images=im_b, return_tensors="pt").to(device)).last_hidden_state.mean(dim=1)
        sim = nn.CosineSimilarity(dim=1)(fa, fb).item()
    return float((sim + 1.0) / 2.0)


def compute_edge_f1(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    ga = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
    ea = cv2.Canny(ga, 60, 150) > 0
    eb = cv2.Canny(gb, 60, 150) > 0
    # 1px dilation tolerance (standard boundary F1)
    k = np.ones((3, 3), np.uint8)
    ea_d = cv2.dilate(ea.astype(np.uint8), k) > 0
    eb_d = cv2.dilate(eb.astype(np.uint8), k) > 0
    tp_a = float((ea & eb_d).sum())
    tp_b = float((eb & ea_d).sum())
    prec = tp_b / max(1.0, float(eb.sum()))
    rec = tp_a / max(1.0, float(ea.sum()))
    return float(2.0 * prec * rec / max(1e-8, prec + rec))


def evaluate_svg(src_path: pathlib.Path, svg_path: pathlib.Path, *, use_dino: bool = True, out_png: pathlib.Path | None = None) -> dict:
    src_raw = Image.open(src_path)
    if src_raw.mode in ("RGBA", "LA") or (src_raw.mode == "P" and "transparency" in src_raw.info):
        rgba = src_raw.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        src_rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        src_rgb = src_raw.convert("RGB")

    svg_txt = svg_path.read_text(encoding="utf-8", errors="ignore")
    w, h = _svg_size(svg_txt)

    tmp_dir = None
    if out_png is None:
        tmp_dir = tempfile.TemporaryDirectory()
        png_path = pathlib.Path(tmp_dir.name) / "render.png"
    else:
        png_path = out_png

    try:
        ren_rgb = render_svg_edge(svg_path, png_path, w, h, bg="#ffffff")
        if ren_rgb.size != src_rgb.size:
            ren_rgb = ren_rgb.resize(src_rgb.size, Image.Resampling.LANCZOS)

        a = np.asarray(src_rgb)
        b = np.asarray(ren_rgb)
        mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
        psnr = 99.0 if mse <= 0 else float(10.0 * math.log10((255.0 ** 2) / mse))

        lab_a = srgb_to_oklab(a).astype(np.float32)
        lab_b = srgb_to_oklab(b).astype(np.float32)
        de = np.linalg.norm(lab_a - lab_b, axis=2)

        edge_f1 = compute_edge_f1(a, b)
        dino = compute_dino_score(src_rgb, ren_rgb) if use_dino else None

        n_paths = svg_txt.count("<path")
        n_filled = len(re.findall(r'<path\b[^>]*\bfill="#[0-9a-fA-F]{6}"', svg_txt))
        n_colors = len(set(m.lower() for m in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg_txt)))
        n_arcs = len(re.findall(r'A[0-9.]', svg_txt))
        n_circ_loops = svg_txt.count('data-primitive="circle"')
        n_ell_loops = svg_txt.count('data-primitive="ellipse"')
        kb = svg_path.stat().st_size / 1024.0
        rlrf_idx = (dino - 0.02 * math.log10(max(10, n_filled))) if dino is not None else None

        return {
            "image": src_path.name,
            "svg": svg_path.name,
            "size": [src_rgb.width, src_rgb.height],
            "psnr": round(psnr, 2),
            "oklab_de_mean": round(float(de.mean()), 4),
            "oklab_de_p95": round(float(np.percentile(de, 95)), 4),
            "edge_f1": round(edge_f1, 4),
            "dino_score": round(dino, 5) if dino is not None else None,
            "rlrf_efficiency": round(rlrf_idx, 5) if rlrf_idx is not None else None,
            "total_paths": n_paths,
            "filled_regions": n_filled,
            "unique_colors": n_colors,
            "arc_cmds": n_arcs,
            "circle_primitives": n_circ_loops,
            "ellipse_primitives": n_ell_loops,
            "svg_kb": round(kb, 1),
        }
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def main():
    ap = argparse.ArgumentParser(description="Evaluate SVG perceptual & structural quality (DinoScore + EdgeF1 + OKLab dE + Compactness)")
    ap.add_argument("image", help="source raster image")
    ap.add_argument("svg", help="generated SVG path")
    ap.add_argument("--no-dino", action="store_true", help="skip DINOv2 score")
    ap.add_argument("--out-png", default=None, help="optional path to save rendered PNG")
    a = ap.parse_args()
    res = evaluate_svg(
        pathlib.Path(a.image),
        pathlib.Path(a.svg),
        use_dino=not a.no_dino,
        out_png=pathlib.Path(a.out_png) if a.out_png else None,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
