"""Full-coverage colour tracing with an explicit colour-retention setting."""
import math
import re
import numpy as np
from PIL import Image
from collections import defaultdict


def retained_image(image, color_retention):
    """Use 2 to 256 levels per RGB channel; 100 leaves all source RGB values intact."""
    if isinstance(color_retention, bool) or not math.isfinite(color_retention) or not 0 <= color_retention <= 100:
        raise ValueError("色彩保留程度必须在 0–100% 之间。")
    rgb = image.convert("RGB")
    if color_retention == 100:
        return rgb
    levels = max(2, round(2 ** (1 + 7 * color_retention / 100)))
    values = np.asarray(rgb, dtype=np.float64)
    bins = np.rint(values * (levels - 1) / 255)
    quantized = np.rint(bins * 255 / (levels - 1)).astype(np.uint8)
    return Image.fromarray(quantized)


def vectorize_source(image, progress=None, color_retention=100, hair_repair=False):
    """Preserve the quantized raster grid as vector paths at the input dimensions."""
    work = retained_image(image, color_retention)
    hair_paths = []
    if hair_repair:
        from .hair_repair import repair_hair
        if progress:
            progress(0.02, "发丝修复 · 检查明暗细线与连续性")
        background, hair_paths, _, _ = repair_hair(np.asarray(work), allow_gradients=color_retention == 100)
        work = retained_image(Image.fromarray(background), color_retention)
        if color_retention < 100:
            def quantize_fill(match):
                value = tuple(bytes.fromhex(match.group(1)))
                color = retained_image(Image.new('RGB', (1,1), value), color_retention).getpixel((0,0))
                return 'fill="#%02x%02x%02x"' % color
            hair_paths = [re.sub(r'fill="#([0-9a-f]{6})"', quantize_fill, p) for p in hair_paths]
    if progress:
        progress(0.1, f"完整绘制 · 色彩保留 {color_retention:g}%")
    pixels = np.asarray(work, dtype=np.uint32)
    colors = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
    paths = defaultdict(list)
    active = {}

    def finish(key, rectangle):
        left, right, color = key
        top, bottom = rectangle
        width, height = right - left, bottom - top
        paths[color].append(f"M{left},{top}h{width}v{height}h-{width}Z")

    for y, row in enumerate(colors):
        boundaries = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
        current = {}
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            key = (int(left), int(right), int(row[left]))
            previous = active.pop(key, None)
            current[key] = (previous[0] if previous else y, y + 1)
        for key, rectangle in active.items():
            finish(key, rectangle)
        active = current
        if progress and y % 128 == 0:
            progress(0.1 + 0.8 * y / work.height, "完整绘制 · 合并同色区域")
    for key, rectangle in active.items():
        finish(key, rectangle)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{work.width}" height="{work.height}" '
             f'viewBox="0 0 {work.width} {work.height}" shape-rendering="crispEdges">']
    parts.extend(f'<path fill="#{color:06x}" d="{"".join(rectangles)}"/>'
                 for color, rectangles in paths.items())
    if hair_repair:
        parts.append(f'<g id="repaired-hair" data-count="{len(hair_paths)}" shape-rendering="geometricPrecision">')
        parts.extend(hair_paths)
        parts.append('</g>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    if progress:
        progress(1.0, "完整绘制 · 完成")
    return svg, work
