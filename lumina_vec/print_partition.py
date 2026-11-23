"""Non-overlapping two-layer CNN planar SVG reconstruction with full RGBA transparency support."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import os
import re

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import label

from .boundary_corridor import SourceCorridor
from .planar_partition import build_planar_partition
from .shared_regions import trace_regions
from .structural_curves import fit_compact_outline, refit_sketch_chain, smooth_outline_junctions
from .svg_layers import boundary_then_fill


def _prepare_alpha_source(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract an alpha-safe RGB canvas and optional 1x alpha channel `(H, W) uint8`.

    When an input image has transparent regions (`alpha < 250`):
      - Bleeds nearest visible pixel colors outward across the first ~6px outside the
        silhouette so SR / MeanShift / CNN filters never pull black or white edge fringes
        into semi-transparent anti-aliased strokes.
      - Smoothly transitions the far exterior (> 6px) to a uniform contrasting backdrop
        so line detectors see clean outer contours without phantom background artifacts.
      - Returns `alpha_1x` so final planar labeling masks transparent pixels back to `-1`
        (zero background `<path>` emitted in SVG, full alpha preserved in PNG preview).
    """
    has_alpha = ("A" in img.getbands()) or ("transparency" in img.info)
    if not has_alpha:
        return np.asarray(img.convert("RGB")), None

    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[:, :, 3]
    if np.all(alpha >= 250):
        return rgba[:, :, :3].copy(), None

    rgb_raw = rgba[:, :, :3]
    visible = alpha >= 128
    if not visible.any():
        return np.full_like(rgb_raw, 255), alpha

    # Bleed nearest visible RGB outward across the transparent exterior (eliminates black/white fringe)
    dist_out, (iy, ix) = distance_transform_edt(~visible, return_indices=True)
    bled = rgb_raw[iy, ix].astype(np.float32)

    # Pick a clean uniform backdrop for far exterior (> 6px) contrasting with silhouette border
    border_ring = (~visible) & (dist_out <= 2.5)
    border_lum = float(bled[border_ring].mean()) if border_ring.any() else 128.0
    backdrop_val = 248.0 if border_lum < 210.0 else 32.0

    # Smooth transition from edge-bled color (0..4px) to clean uniform backdrop (6px+)
    blend = np.clip((dist_out - 3.0) / 4.0, 0.0, 1.0)[:, :, None]
    comp = np.where(
        visible[:, :, None],
        rgb_raw.astype(np.float32),
        bled * (1.0 - blend) + backdrop_val * blend,
    )
    return np.clip(np.rint(comp), 0, 255).astype(np.uint8), alpha


def compact_fragments(labels: np.ndarray, colors: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Absorb sub-pixel micro-specks (`< scale^2`) into best adjacent visible neighbor while preserving `-1` transparency."""
    if len(colors) == 0:
        return labels, colors
    has_trans = bool((labels < 0).any())
    vis = labels >= 0
    if not vis.any():
        return labels, colors

    area = np.bincount(labels[vis].ravel(), minlength=len(colors))
    boxes = find_objects(np.where(vis, labels + 1, 0))
    span = np.array([max(s.stop - s.start for s in b) if b is not None else 0 for b in boxes])

    k1, k2 = [], []
    for a, b in [(labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])]:
        different = (a != b) & (a >= 0) & (b >= 0)
        if not different.any():
            continue
        va = a[different].astype(np.uint64)
        vb = b[different].astype(np.uint64)
        k1.append((va << 32) | vb)
        k2.append((vb << 32) | va)
    if not k1:
        return labels, colors

    uniq_k, contacts = np.unique(np.concatenate(k1 + k2), return_counts=True)
    src = (uniq_k >> 32).astype(np.int32)
    dst = (uniq_k & 0xFFFFFFFF).astype(np.int32)
    valid = (area[src] < scale ** 2) & (span[src] < 4 * scale) & (area[dst] >= scale ** 2)
    src, dst, contacts = src[valid], dst[valid], contacts[valid]
    if len(src) == 0:
        return labels, colors

    distance = np.linalg.norm(colors[src].astype(float) - colors[dst], axis=1)
    order = np.lexsort((-contacts, distance, src))
    mapping = np.arange(len(colors), dtype=np.int32)
    seen = set()
    for a_id, b_id in zip(src[order].tolist(), dst[order].tolist()):
        if a_id not in seen:
            mapping[a_id] = b_id
            seen.add(a_id)

    palette, ids = np.unique(colors, axis=0, return_inverse=True)
    if has_trans:
        field = np.full(labels.shape, -1, dtype=np.int32)
        field[vis] = ids[mapping[labels[vis]]]
        regions = label(np.where(vis, field + 1, 0), connectivity=1, background=0).astype(np.int32) - 1
        n_reg = int(regions.max()) + 1
        values = np.zeros((max(0, n_reg), 3), np.uint8)
        if n_reg > 0:
            r_vis = regions >= 0
            values[regions[r_vis]] = palette[field[r_vis]]
        return regions, values

    field = ids[mapping[labels]]
    regions = label(field + 1, connectivity=1).astype(np.int32) - 1
    values = np.empty((regions.max() + 1, 3), np.uint8)
    values[regions.ravel()] = palette[field.ravel()]
    return regions, values


def trace_binary_ink(img: Image.Image, rgb: np.ndarray, outline_layer: bool):
    """Trace exact pure black/white pixel artwork without deleting pixels or filling holes."""
    if "A" in img.getbands() or "transparency" in img.info:
        alpha = np.asarray(img.convert("RGBA"))[..., 3]
        if not np.all((alpha == 0) | (alpha == 255)):
            return None
        visible = alpha == 255
    else:
        visible = np.ones(rgb.shape[:2], bool)
    black = np.all(rgb == 0, axis=2) & visible
    white = np.all(rgb == 255, axis=2) & visible
    if np.any(visible & ~(black | white)):
        return None
    if not visible.any():
        h, w = rgb.shape[:2]
        empty = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"/>'
        return empty, img.convert("RGBA")
    regions = label(black.astype(np.uint8) + 2 * white, connectivity=1, background=0).astype(np.int32) - 1
    palette = np.empty(regions.max() + 1, dtype=bool)
    palette[regions[visible]] = white[visible]
    svg, _ = trace_regions(regions, np.where(palette, "#ffffff", "#000000"), exact_grid=True)
    if outline_layer:
        start = svg.index(">") + 1
        svg = svg[:start] + '<g id="region-boundaries"/><g id="region-colours">' + svg[start:-6] + "</g></svg>"
    return svg, img.convert("RGB" if visible.all() else "RGBA")


def _vectorize_clean_planar(
    source: np.ndarray,
    fmaps,
    orig_shape: tuple[int, int],
    *,
    alpha_1x: np.ndarray | None = None,
    smooth_lines: bool = True,
    refit_outline: bool = True,
    locate_boundaries: bool = True,
    outline_layer: bool = True,
    core_mode: bool = False,
    progress=None,
):
    """Run lossless planar reconstruction directly at SR resolution with alpha transparency preservation."""
    step = progress or (lambda *_: None)
    h_orig, w_orig = orig_shape
    h, w = source.shape[:2]
    if h == h_orig and w == w_orig and (h * w) <= 4_000_000:
        sr_in = cv2.resize(source, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4 if core_mode else cv2.INTER_CUBIC)
    else:
        sr_in = source
    hs, ws = sr_in.shape[:2]

    visible_hi = None
    if alpha_1x is not None:
        alpha_hi = alpha_1x if alpha_1x.shape == (hs, ws) else cv2.resize(alpha_1x, (ws, hs), interpolation=cv2.INTER_LINEAR)
        visible_hi = alpha_hi >= 128

    step(0.45 if core_mode else 0.25, "核心线稿骨架提取 + 平涂色区分割" if core_mode else "Stage 2->3：无损线条骨架提取 + 保护区解锁 + 渐变色阶轮廓划分")
    regions, colors, line_mask, p_stats = build_planar_partition(
        sr_in, fmaps, (h_orig, w_orig), core_mode=core_mode, visible_mask=visible_hi
    )
    print("[planar_partition]", p_stats, flush=True)

    step(0.68 if core_mode else 0.62, "整理细碎色区")
    regions, colors = compact_fragments(regions, colors, scale=max(1, int(round(hs / max(1, h_orig)))))

    step(0.80 if core_mode else 0.76, "核心阶段：SVG 共享矢量边界拟合" if core_mode else "Stage 4：SVG共享矢量边界拟合")
    fitting = {}
    corridor = None
    metadata = {
        "data-planar-engine": "core-ai-v1" if core_mode else "v1",
        "data-raw-compartments": str(p_stats.get("n_regions_raw", len(colors))),
    }
    if refit_outline:
        fitting = dict(
            structural_fitter=partial(fit_compact_outline, smooth_lines=smooth_lines),
            secondary_fitter=partial(refit_sketch_chain, smooth_lines=smooth_lines),
            secondary_network=partial(smooth_outline_junctions, smooth_lines=smooth_lines),
        )
        if locate_boundaries:
            corridor = SourceCorridor(sr_in, colors)
            if core_mode:
                fitting.update(chain_locator=corridor, source_scale=1)
            else:
                gc95_hi = cv2.resize(fmaps.grad_core_95.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0
                vf0 = fmaps.line_vector_fields[0]
                vf0_hi = vf0 if vf0.shape == (hs, ws) else cv2.resize(vf0, (ws, hs), interpolation=cv2.INTER_LINEAR)

                def _grad_guarded_corridor(pts, right, left, sc):
                    if min(right, left) >= 0 and len(pts) >= 4:
                        ys_c = np.clip(np.round(pts[:, 1]).astype(int), 0, hs - 1)
                        xs_c = np.clip(np.round(pts[:, 0]).astype(int), 0, ws - 1)
                        if float(np.mean(gc95_hi[ys_c, xs_c])) > 0.75 and float(np.max(vf0_hi[ys_c, xs_c])) < 0.24:
                            return pts
                    return corridor(pts, right, left, sc)

                fitting.update(chain_locator=_grad_guarded_corridor, source_scale=1)

    chain_scale_fn = None
    if (not core_mode) and smooth_lines and hasattr(fmaps, "delicate_detail"):
        del_hi = fmaps.delicate_detail if fmaps.delicate_detail.shape == (hs, ws) else cv2.resize(fmaps.delicate_detail, (ws, hs), interpolation=cv2.INTER_LINEAR)
        wash_hi = fmaps.mergeable_wash if fmaps.mergeable_wash.shape == (hs, ws) else cv2.resize(fmaps.mergeable_wash, (ws, hs), interpolation=cv2.INTER_LINEAR)

        def _chain_scale(pts, base_scale, s_scale):
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, hs - 1)
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, ws - 1)
            if float(np.mean(del_hi[ys, xs])) > 0.35:
                return base_scale * 0.85
            if float(np.mean(wash_hi[ys, xs])) > 0.48:
                return base_scale * 1.40
            return base_scale

        chain_scale_fn = _chain_scale

    tol = 0.36 if smooth_lines else 0.28
    smoothing = 0.95 if smooth_lines else 0.75
    struct_sc = 1.35 if smooth_lines else 1.15
    svg, stats = trace_regions(
        regions,
        ["#%02x%02x%02x" % tuple(c) for c in colors],
        tol=tol,
        seam=0,
        geometry_guard=True,
        smoothing=smoothing,
        structural_scale=struct_sc,
        compact_commands=True,
        chain_scale_fn=chain_scale_fn,
        **fitting,
    )
    svg = re.sub(r'width="\d+" height="\d+"', f'width="{w_orig}" height="{h_orig}"', svg, count=1)

    if outline_layer and len(colors) > 0:
        step(0.96, "生成教师SVG结构（gap-filler轮廓 + 线稿轮廓约束单色填充层）")
        vis = regions >= 0
        r_areas = np.bincount(regions[vis].ravel(), minlength=len(colors))
        r_lm = np.bincount(
            regions[vis].ravel(),
            weights=line_mask[vis].astype(np.float32).ravel(),
            minlength=len(colors),
        ) / np.maximum(r_areas, 1)
        line_region_ids = set(np.flatnonzero(r_lm >= 0.55).tolist())
        metadata["data-line-art-paths"] = str(len(line_region_ids))
        metadata["data-color-fill-paths"] = str(len(colors) - len(line_region_ids))
        metadata["data-segments-before-refit"] = str(stats.get("segments_before_secondary", 0))
        metadata["data-segments-after-refit"] = str(stats.get("curve_segments", 0))
        if corridor is not None:
            metadata["data-source-chains-considered"] = str(corridor.considered)
            metadata["data-source-chains-moved"] = str(corridor.moved)
        svg = boundary_then_fill(svg, hidden_strokes=(), metadata=metadata, line_region_ids=line_region_ids)

    if visible_hi is not None:
        prev_rgba = np.zeros((hs, ws, 4), np.uint8)
        vis = regions >= 0
        if len(colors) > 0 and vis.any():
            prev_rgba[vis, :3] = colors[regions[vis]]
            prev_rgba[vis, 3] = 255
        preview = Image.fromarray(prev_rgba, mode="RGBA").resize((w_orig, h_orig), Image.Resampling.LANCZOS)
    else:
        preview = Image.fromarray(colors[regions]).resize((w_orig, h_orig), Image.Resampling.LANCZOS)

    step(1.0, "完成")
    return svg, preview


def vectorize_print(
    img: Image.Image,
    bands: int = 112,
    recover_strands: bool = True,
    progress=None,
    outline_layer: bool = True,
    gradient_mode: str = "clean",
    refit_outline: bool = True,
    redraw_source: bool = False,
    smooth_lines: bool = True,
    feature_routing: bool = True,
    sr_scale: float | None = None,
    ai_repaint: bool = False,
    core_mode: bool | None = None,
    *,
    locate_boundaries: bool = True,
):
    """Main CNN two-layer planar vectorization entry point with full transparency preservation."""
    if core_mode is None:
        core_mode = bool(ai_repaint)
    step = progress or (lambda *_: None)

    source, alpha_1x = _prepare_alpha_source(img)
    h, w = source.shape[:2]
    h_orig, w_orig = h, w

    exact = trace_binary_ink(img, source, outline_layer)
    if exact is not None:
        step(1.0, "完成：保留纯黑白像素边界")
        return exact

    if ai_repaint:
        from .ai_repaint import ink_locked_flat_repaint

        vis_1x = (alpha_1x >= 128) if alpha_1x is not None else None
        rep_img, rep_info = ink_locked_flat_repaint(
            Image.fromarray(source),
            progress=lambda pct, msg: step(0.01 + 0.17 * (pct / 100.0), msg),
            visible_mask=vis_1x,
        )
        print("[ai_repaint]", rep_info, flush=True)
        source = np.asarray(rep_img.convert("RGB"))

    from .feature_routing import infer_feature_maps, render_vector_heatmap_png

    step(
        0.20 if ai_repaint else 0.004,
        "核心线条 CNN 识别（极简模式）" if core_mode else "整图倍率放大 + 去杂色平滑锐化 + CNN 热力线条识别",
    )
    fmaps = infer_feature_maps(source, sr_scale=sr_scale, core_mode=core_mode)
    if fmaps is None or getattr(fmaps, "line_vector_fields", None) is None:
        raise RuntimeError("缺少核心 CNN ONNX 权重（weights/line_vector_net.onnx）")

    print("[feature_routing]", fmaps.summary, flush=True)
    hm_out = os.environ.get("LUMINA_HEATMAP_OUT", "").strip()
    if hm_out:
        try:
            render_vector_heatmap_png(source, hm_out, vec_fields=fmaps.line_vector_fields)
        except Exception:
            pass

    if getattr(fmaps, "sr_hi_rgb", None) is not None:
        source = fmaps.sr_hi_rgb
        h, w = source.shape[:2]
        fmaps = replace(
            fmaps,
            ink_main_geom=cv2.resize(fmaps.ink_main_geom, (w, h), interpolation=cv2.INTER_LINEAR),
            fine_strand=cv2.resize(fmaps.fine_strand, (w, h), interpolation=cv2.INTER_LINEAR),
            delicate_detail=cv2.resize(fmaps.delicate_detail, (w, h), interpolation=cv2.INTER_LINEAR),
            mergeable_wash=cv2.resize(fmaps.mergeable_wash, (w, h), interpolation=cv2.INTER_LINEAR),
            surface_probs=np.stack(
                [cv2.resize(fmaps.surface_probs[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)],
                axis=0,
            ),
            line_net_maps=(
                np.stack(
                    [cv2.resize(fmaps.line_net_maps[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)],
                    axis=0,
                )
                if fmaps.line_net_maps is not None
                else None
            ),
            elem_grad_maps=(
                np.stack(
                    [cv2.resize(fmaps.elem_grad_maps[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(8)],
                    axis=0,
                )
                if fmaps.elem_grad_maps is not None
                else None
            ),
            grad_core_95=(
                cv2.resize(fmaps.grad_core_95.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if getattr(fmaps, "grad_core_95", None) is not None
                else None
            ),
            line_buffer_5=(
                cv2.resize(fmaps.line_buffer_5.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if getattr(fmaps, "line_buffer_5", None) is not None
                else None
            ),
            sr_clean_1x=source,
        )
    elif fmaps.sr_clean_1x is not None:
        source = fmaps.sr_clean_1x

    return _vectorize_clean_planar(
        source,
        fmaps,
        (h_orig, w_orig),
        alpha_1x=alpha_1x,
        smooth_lines=smooth_lines,
        refit_outline=refit_outline,
        locate_boundaries=locate_boundaries,
        outline_layer=outline_layer,
        core_mode=core_mode,
        progress=progress,
    )
