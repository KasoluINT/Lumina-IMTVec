# Copyright 2026 MESFET5198 (Lumina-Vec, Apache License 2.0)
# Modified 2026 by KasoluINT (Lumina-IMTVec): replaced legacy pipeline with CNN planar engine facade.
# See LICENSE-APACHE-2.0 and NOTICE for full license and attribution notices.
"""Top-level vectorization facade routing cleanly into the CNN two-layer planar engine."""
from __future__ import annotations

from PIL import Image

from .presets import PRESETS
from .print_partition import vectorize_print


def grain_level(img: Image.Image) -> float:
    """Estimate high-frequency noise level on native luminance for metadata/logging."""
    import cv2
    import numpy as np

    grey = np.asarray(img.convert("L"), dtype=np.float32)
    return float(np.mean(np.abs(grey - cv2.medianBlur(grey.astype(np.uint8), 3))))


def auto_params(img: Image.Image) -> tuple[str, float, dict]:
    """Default automatic route uses the CNN two-layer planar vectorization preset."""
    return "print-regions", grain_level(img), dict(PRESETS["print-regions"])


def fine_params(img: Image.Image, params: dict) -> dict:
    """Return copy of params (kept for CLI/worker compatibility)."""
    return dict(params)


def vectorize(
    img: Image.Image,
    weights=None,
    *,
    bands: int = 84,
    progress=None,
    gradient_mode: str = "clean",
    refit_outline: bool = True,
    redraw_source: bool = False,
    smooth_lines: bool = True,
    feature_routing: bool = True,
    sr_scale: float = 2.0,
    ai_repaint: bool = False,
    core_mode: bool | None = None,
    **kwargs,
):
    """Vectorize an image (RGB or RGBA with transparency) into a two-layer gapless SVG."""
    return vectorize_print(
        img,
        bands=bands or 84,
        progress=progress,
        gradient_mode=gradient_mode,
        refit_outline=refit_outline,
        redraw_source=redraw_source,
        smooth_lines=smooth_lines,
        feature_routing=feature_routing,
        sr_scale=sr_scale,
        ai_repaint=ai_repaint,
        core_mode=core_mode,
    )


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Lumina-IMTVec: illustration to dual-layer planar vector SVG")
    ap.add_argument("input", help="input raster image path (PNG/JPG/WebP, supports RGBA transparency)")
    ap.add_argument("output", help="output SVG file path")
    ap.add_argument("--preset", default="print-regions", choices=["auto", *PRESETS])
    ap.add_argument("--bands", type=int, default=None)
    ap.add_argument("--gradient-mode", default="clean", choices=["clean", "layered"])
    ap.add_argument("--sr-scale", type=float, default=2.0)
    ap.add_argument("--ai-repaint", action="store_true", help="enable Ink-Locked Flat repaint + streamlined core mode")
    ap.add_argument("--no-smooth-lines", action="store_true")
    ap.add_argument("--no-feature-routing", action="store_true")
    a = ap.parse_args()

    img = Image.open(a.input)
    img.load()
    preset = "print-regions" if a.preset == "auto" else a.preset
    params = dict(PRESETS[preset])
    if a.bands is not None:
        params["bands"] = a.bands
    params["gradient_mode"] = a.gradient_mode
    params["sr_scale"] = a.sr_scale
    params["ai_repaint"] = bool(a.ai_repaint)
    if a.no_smooth_lines:
        params["smooth_lines"] = False
    if a.no_feature_routing:
        params["feature_routing"] = False

    svg, _ = vectorize(img, **params)
    Path(a.output).write_text(svg, encoding="utf-8")
    print(f"Wrote {a.output} ({len(svg) / 1024:.1f} KB, {svg.count('<path')} paths)")


if __name__ == "__main__":
    main()
