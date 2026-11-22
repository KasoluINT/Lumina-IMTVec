"""Render an SVG at its INTRINSIC size (so nothing is cropped), downscale to the source image
size and report PSNR / mean abs diff."""
import sys, os, re, subprocess, json, math, tempfile
import numpy as np
from PIL import Image

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def svg_size(path):
    t = open(path, encoding="utf-8", errors="ignore").read(6000)
    m = re.search(r'<svg[^>]*width="([0-9.]+)"[^>]*height="([0-9.]+)"', t)
    if m:
        return int(float(m.group(1))), int(float(m.group(2)))
    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', t)
    return (int(float(m.group(1))), int(float(m.group(2)))) if m else (1200, 1200)


def render(svg, png, w=None, h=None):
    if w is None or h is None:
        w, h = svg_size(svg)
    w, h = min(w, 6000), min(h, 6000)
    prof = tempfile.mkdtemp()
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--user-data-dir=" + prof, "--window-size=%d,%d" % (w, h),
                    "--screenshot=" + png, "file:///" + os.path.abspath(svg).replace("\\", "/")],
                   capture_output=True, timeout=300)
    return png


def psnr(a, b):
    d = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if d <= 0 else 10 * math.log10(255.0 ** 2 / d)


if __name__ == "__main__":
    img, svg = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    src = Image.open(img).convert("RGB")
    w, h = svg_size(svg)
    png = out or os.path.join(tempfile.gettempdir(), "psnr-" + os.path.basename(svg) + ".png")
    render(svg, png, w, h)
    got = Image.open(png).convert("RGB").resize(src.size, Image.LANCZOS)
    a, b = np.asarray(src), np.asarray(got)
    print(json.dumps(dict(image=os.path.basename(img), svg=os.path.basename(svg),
                          intrinsic=[w, h], psnr=round(psnr(a, b), 2),
                          mean_abs=round(float(np.abs(a.astype(float) - b.astype(float)).mean()), 2))))
