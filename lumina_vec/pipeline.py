"""AI-assisted raster -> SVG vectorizer.

Pipeline (see README for why each step exists):
  1. AI de-grain:  box-downscale by `pre_down` (averages film grain away) -> Real-ESRGAN
                   anime model x4 (re-synthesises crisp line art) -> resample to `work_scale`.
  2. Flatten:      mean-shift filtering collapses residual painterly texture into flat colour.
  2b. Bands (optional, --bands N): OKLab k-means palette; low-contrast pixels are labelled
                   from an edge-masked, distance-adaptive blur + spatial vote, so gradients become
                   smooth contour-like steps; lines/texture keep per-pixel labels (lumina_vec/bands.py).
  3. Trace:        vtracer 1.0 hierarchical (stacked) clustering + spline fitting.
  4. Output:       SVG at the input's pixel size (viewBox keeps the working resolution).
"""
import argparse, json, math, os, re, time
import numpy as np, cv2, vtracer
from PIL import Image

from .presets import (  # preset tables live in a data-only module so the web
    PRESETS, GRAIN_THRESHOLD, MAX_WORK_PX, FLAT_THRESHOLD,   # front-end can import them without torch
)


def grain_level(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"))
    return float(np.abs(g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16)).mean())


def add_gap_fillers(svg):
    """Display-only seam cover for shared-edge (cutout) output: before each fill path, a
    same-colour 2px non-scaling stroke in Lumina's exact `qq3d-gap-filler` layout, which
    Lumina strips on import (core/vector_engine.py fast filter), so it never reaches the print."""
    def rep(m):
        tag = m.group(0)
        d = re.search(r'\sd="([^"]*)"', tag); f = re.search(r'\sfill="(#[0-9a-fA-F]{6})"', tag)
        if not d or not f:
            return tag
        gf = (f'<path class="qq3d-gap-filler" d="{d.group(1)}" fill="none" stroke="{f.group(1)}" '
              'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>')
        tr = re.search(r'\stransform="([^"]*)"', tag)
        if tr:
            gf = f'<g transform="{tr.group(1)}">{gf}</g>'
        return gf + "\n" + tag
    return re.sub(r"<path\b[^>]*/>", rep, svg)


def gradient_share(img: Image.Image) -> float:
    """Share of pixels in gentle colour gradients (shading), measured at ~1 MP after a median
    filter (ignores grain). Flat cel art ~0.1, shaded illustrations 0.3-0.5."""
    W, H = img.size; s = min(1.0, (1e6 / (W * H)) ** 0.5)
    a = np.asarray(img.convert("RGB").resize((max(1, round(W * s)), max(1, round(H * s))), Image.BOX))
    lab = cv2.cvtColor(cv2.medianBlur(a, 5).astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    g = np.sqrt((cv2.Sobel(lab, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab, cv2.CV_32F, 0, 1) ** 2).sum(-1))
    return float(((g >= 2) & (g < 12)).mean())


def auto_params(img: Image.Image):
    """Pick preset from measured grain, and shrink the working size of very large images.
    Calibrated on 3 images: heavy-grain flat anime (8.8), painterly (1.9), clean cel (0.2)."""
    grain = grain_level(img)
    if grain >= GRAIN_THRESHOLD:
        name = "shading"
    else:
        name = "cel" if gradient_share(img) < FLAT_THRESHOLD else "shading-detail"
    params = dict(PRESETS[name])
    W, H = img.size
    cap = (MAX_WORK_PX[name] / (W * H)) ** 0.5
    if params["work_scale"] > cap:
        params["work_scale"] = round(max(0.25, cap), 2)
    return name, grain, params


FINE_MAX_PX = 6e6              # fine mode: working-size cap (memory / time)


def fine_params(img: Image.Image, params: dict) -> dict:
    """精细模式 (tested on 5 illustrations, Nov 2026): line layer v3 on; for images that are not
    grainy, no 2x downscale before the AI clean (small images lost hair strands and wallpaper
    lines there), a 2x working size (measured knee: 1x->2x completes lines, beyond 2x nothing
    but more nodes), and thin faint structures protected from gradient smoothing. Grainy
    images keep their preset's downscale and smoothing (grain would pass as 'thin structure')."""
    p = dict(params)
    if p.get("source_detail"):
        return p
    p.update(ink=True, line_style="lens", region_smooth=True)
    if grain_level(img) < GRAIN_THRESHOLD:
        W, H = img.size
        p.update(pre_down=1.0, thin_thr=0.015,
                 work_scale=round(min(2.0, (FINE_MAX_PX / (W * H)) ** 0.5), 2))
    return p


def match_tone(clean, ref, sigma=6.0):
    """clean + blur(ref) - blur(clean) in OKLab: local mean colour of `ref`, detail of `clean`.
    Blurs are edge-aware enough at this sigma for flat anime fills; a large sigma keeps the
    grain of `ref` out."""
    from .color import srgb_to_oklab, oklab_to_srgb
    a = srgb_to_oklab(clean).astype(np.float32); b = srgb_to_oklab(ref).astype(np.float32)
    d = cv2.GaussianBlur(b, (0, 0), sigma) - cv2.GaussianBlur(a, (0, 0), sigma)
    return np.clip(oklab_to_srgb(a + d), 0, 255).astype(np.uint8)


def vectorize(img: Image.Image, weights=None, pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
              filter_speckle=12, layer_difference=10, simplify=1.5, color_precision=6, ai=True,
              clean=None,
              bands=0, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4, detail_overlay=True, ink=False, progress=None,
              print_mm=0.0, line_mm=0.42, lines="anime", weights_dir=None, hierarchical_mode=None, contour_lines=True,
              tracer="contour", label_sigma=1.0, thin_thr=0.0, line_style="v3",
              palette_n=0, palette_de=2.5, region_color=False, region_smooth=False, tone_match=True,
              eye_boxes=None, eye_colors=12, source_detail=False, color_retention=100, hair_repair=False,
              structure="stacked", gradient_mode="layered", refit_outline=True, redraw_source=False, smooth_lines=False, feature_routing=True, sr_scale=2.0):
    if structure == "print-regions":
        from .print_partition import vectorize_print
        return vectorize_print(img, bands=bands or 112, progress=progress, gradient_mode=gradient_mode,
                               refit_outline=refit_outline, redraw_source=redraw_source, smooth_lines=smooth_lines,
                               feature_routing=feature_routing, sr_scale=sr_scale)
    if feature_routing and clean is None:
        from .feature_routing import repair_image_defects_onnx
        repaired_np, d_ratio = repair_image_defects_onnx(np.asarray(img.convert("RGB")), weights_dir=weights_dir)
        if d_ratio > 0.0:
            if img.mode == "RGBA":
                r_im = Image.fromarray(repaired_np).convert("RGBA")
                r_im.putalpha(img.getchannel("A"))
                img = r_im
            else:
                img = Image.fromarray(repaired_np)
    if structure in ("regions", "lumina"):
        # ★ 主模式（参考 SVG 逆向配方）：放大 2 倍 -> 量化到 palette_n 色（默认 160）
        # -> vtracer spline 描摹（一色块一闭合 path，包住的部分是空腔）-> 颜色吸附回色板。
        # 渐变里过度的中间色被并掉（腮红压平到接近肤色），眼影/发丝描边这类强边界保留；
        # 边界是拟合出的长曲线，不是逐像素折线。
        # 参考 SVG 的生成策略：量化 -> 每块亚像素轮廓 -> L/Q/A 分段拟合 ->
        # 成对输出（fill + class="qq3d-gap-filler" 的同色描边）
        from .regionsvg import vectorize_paired
        return vectorize_paired(img, colors=palette_n or 106, tol=0.6, sigma=2.2,
                                min_area=20, scale=1.5, gap_filler=True, gap_width=2,
                                progress=progress)
    if structure == "fidelity":
        # 保真优先：不做颜色合并，2.6 倍描摹；路径数很多，但 PSNR 最高
        from .regionsvg import vectorize_lumina
        return vectorize_lumina(img, colors=0, layer_diff=1, simplify=0.8, speckle=2,
                                gap=False, progress=progress)
    if structure in ("structured", "flat"):
        # 分层结构：封闭色块（带空腔）+ 连续线条几何 + 同一根线的分段着色
        # structured = <defs>/<use>/<clipPath>（几何只存一份）；flat = 纯 <path>（兼容性最好）
        from .layered import vectorize_layered
        return vectorize_layered(
            img, weights, structure=structure, pre_down=pre_down, work_scale=work_scale,
            bands=bands or 112, ms_sp=ms_sp, ms_sr=ms_sr,
            blur_sigma=blur_sigma if blur_sigma and blur_sigma > 0 else 6.0,
            thin_thr=thin_thr, vote_sigma=vote_sigma, progress=progress)
    if source_detail:
        if print_mm:
            raise ValueError("保留原色模式请清空打印尺寸；打印优化请使用其他预设。")
        from .source_detail import vectorize_source
        return vectorize_source(img, progress, color_retention, hair_repair)
    W, H = img.size
    # stage cost model (seconds per megapixel, CPU), calibrated on two test images; the AI step
    # scales with its (downscaled) input, everything else with the working resolution
    mp_in = W * H / pre_down ** 2 / 1e6
    mp_w = W * H * work_scale ** 2 / 1e6
    stages = [("ai", "AI 去颗粒", 12.5 * mp_in if (ai and clean is None) else 0.2),
              ("flat", "压平", 1.0 * mp_w),
              ("ink", "线条", 4.0 * mp_w if (bands and ink) else 0),
              ("bands", "明暗分层", 6.4 * mp_w if bands else 0),
              ("trace", "矢量描摹", 2.5 * mp_w),
              ("detail", "细节补回", 0.4 * mp_w if (bands and detail_overlay) else 0.05)]
    total = sum(w for *_, w in stages)
    start, acc = {}, 0.0
    for key, _, wgt in stages:
        start[key] = acc / total; acc += wgt
    names = {k: n for k, n, _ in stages}; weight = {k: w / total for k, _, w in stages}

    def report(key, frac=0.0, sub=None):
        if progress:
            label = names[key] + (f" · {sub}" if sub else "")
            progress(min(1.0, start[key] + weight[key] * frac), label)

    report("ai")
    if clean is not None:   # AI step already done elsewhere (python -m lumina_vec.denoise)
        work = clean.convert("RGB").resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    elif ai:
        if os.path.isfile(weights):
            try:
                from .denoise import ai_clean
                work = ai_clean(img, weights, pre_down=pre_down, out_scale=work_scale,
                                progress=lambda f: report("ai", f, f"{f * 100:.0f}%"))
            except ImportError:
                work = None
        else:
            work = None
        if work is None:
            from .feature_routing import realesr_dejpeg_onnx
            src_np = np.asarray(img.convert("RGB"))
            cleaned = realesr_dejpeg_onnx(src_np, weights_dir=weights_dir, pre_down=pre_down)
            base_im = Image.fromarray(cleaned) if cleaned is not None else img.convert("RGB")
            work = base_im.resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    else:
        work = img.convert("RGB").resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    rgb = np.asarray(work.convert("RGB"))
    if tone_match and (ai or clean is not None):
        # Real-ESRGAN brightens and desaturates (pale skin went whiter: L 93.9 -> 95.2, chroma
        # 1.93 -> 1.77). Keep its detail/denoising, take the low-frequency tone from the original
        rgb = match_tone(rgb, np.asarray(img.convert("RGB").resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS)))
        work = Image.fromarray(rgb)
    clean = rgb
    report("flat")
    if ms_sp > 0:
        rgb = cv2.pyrMeanShiftFiltering(rgb, ms_sp, ms_sr)
    h, w = rgb.shape[:2]
    palette = None
    ink_paths = []
    ink_cover = None
    bright = None
    # print mode: smallest feature one `line_mm` extrusion can lay down, in working pixels
    min_px = line_mm * max(w, h) / print_mm if print_mm else 0.0
    hierarchical = hierarchical_mode or "stacked"
    if min_px:
        filter_speckle = max(filter_speckle, int(math.ceil(min_px)))
        # stacked shapes are fitted independently and leave slivers where their curves disagree
        # (thousands of sub-extrusion islands); a cutout mosaic shares every boundary
        hierarchical = "cutout"
        detail_overlay = False      # printable details are already in the cleaned label map
    if bands and ink:
        from .ink import ink_mask, fill_under, trace_ink, trace_lines
        report("ink")
        wdir = weights_dir or (os.path.dirname(weights) if weights else "weights")
        v3 = (line_style in ("v3", "stroke", "lens", "lens5", "lens7") and lines and not min_px
              and all(os.path.exists(os.path.join(wdir, f)) for f in ("netG.pth", "erika.pth")))
        if v3:
            # line layer v3 (lumina_vec/linework.py): manga_line + lineart_anime say WHERE, the
            # original's ink says HOW WIDE (capped near the centreline: dark fills stay fills),
            # one outline per line network traced at 2x, each piece coloured from the original.
            # Print mode keeps the older path below (minimum stroke width / length handling).
            from .linework import line_field, trace_field, color_paths
            geo = clean if grain_level(img) >= GRAIN_THRESHOLD else \
                np.asarray(img.convert("RGB").resize((w, h), Image.LANCZOS))
            report("ink", 0.05, "AI 识别线条")
            if line_style in ("lens", "lens5", "lens7"):
                # v6: each line = ONE continuous path (chained cubic edges, lens_layer in draw.py);
                # lens5 keeps the older one-path-per-piece version for A/B
                from .lines import line_map
                from .draw import lens_layer, lens_layer_legacy
                dl = min(max(h, w), 1536)
                M = line_map(clean, "manga", detect_long=dl, weights_dir=wdir)
                A = line_map(clean, "anime", detect_long=dl, weights_dir=wdir)
                if eye_boxes:
                    # inside the eyes the eye channel draws everything (lashes included, from
                    # the original pixels): lens strokes made lash clusters into thick blobs
                    em = np.zeros((h, w), np.uint8)
                    for bx, by, bw, bh in eye_boxes:
                        sx, sy = w / W, h / H
                        cv2.ellipse(em, ((float((bx + bw / 2) * sx), float((by + bh / 2) * sy)),
                                         (float(bw * sx * 0.85), float(bh * sy * 0.85)), 0), 1, -1)
                    M = np.where(em > 0, 0, M); A = np.where(em > 0, 0, A)
                report("ink", 0.6, "绘制线条")
                ink_paths, _ = (lens_layer_legacy(geo, M, A) if line_style == "lens5"
                                else lens_layer(geo, M, A, continuity=line_style == "lens7"))
                # colour under the lines is repaired where the original actually has ink
                # (the drawn strokes' own outline left grey patches at dense lashes)
                fld, _ = line_field(geo, clean, weights_dir=wdir, maps=(M, A))
                im = fld > 0.5
            elif line_style == "stroke":
                # v4: strokes (lumina_vec/strokes2.py) - continuous, tapered, one colour each
                from .lines import line_map
                from .strokes2 import stroke_layer
                dl = min(max(h, w), 1536)
                M = line_map(clean, "manga", detect_long=dl, weights_dir=wdir)
                A = line_map(clean, "anime", detect_long=dl, weights_dir=wdir)
                report("ink", 0.6, "重建笔画")
                ink_paths, im = stroke_layer(geo, M, A)
            else:
                fld, _ = line_field(geo, clean, weights_dir=wdir)
                report("ink", 0.7, "描摹线条")
                ink_paths = color_paths(trace_field(fld), geo)
                # colour fill is repaired ONLY under the drawn lines: texture the line layer did
                # not take (knit, wallpaper lines) stays in the colour layer
                im = fld > 0.5
        elif lines and os.path.exists(os.path.join(wdir, "netG.pth")):
            # AI line model decides WHERE lines are; the original gives their width; strokes are
            # redrawn from centreline + smoothed width -> even, smooth lines
            from .lines import line_map, stroke_mask, redraw_strokes
            report("ink", 0.05, "AI 识别线条")
            lm = line_map(clean, "anime", detect_long=min(3072, max(clean.shape[:2])), weights_dir=wdir)
            report("ink", 0.6, "重绘线条")
            # print mode: a stroke shorter than a few extrusion widths prints as a blob
            im = stroke_mask(clean, lm, min_len=max(6, int(round(4 * min_px))) if min_px else 6,
                             rel_thr=0.06 if not contour_lines else 0.035)
            rd = redraw_strokes(im, min_radius=min_px / 2 if min_px else 0.0)
            # lines never paint over highlights: pixels clearly lighter than their surroundings
            from .color import srgb_to_oklab as _ok
            _L = _ok(clean)[..., 0].astype(np.float32)
            # highlights = SMALL blobs lighter than their surroundings (white top-hat). A median
            # comparison also flagged ordinary fill next to every dark line, leaving slivers
            _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            bright = cv2.morphologyEx(_L, cv2.MORPH_TOPHAT, _k) > 0.10
            bright = cv2.dilate(bright.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if blur_sigma <= 0:     # flat cel art: hard fills already keep highlights; protecting
                bright[:] = False   # its white outlines/rims only leaves slivers beside lines
            im &= ~bright
            rd &= ~cv2.resize(bright.astype(np.uint8), (rd.shape[1], rd.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            report("ink", 0.8, "描摹线条")
            if contour_lines:
                # geometry from CONTINUOUS values of the original (its anti-aliasing places the
                # edge sub-pixel); the AI-cleaned image only where the original is grainy
                from .ink import trace_lines_contour
                from .contour import ink_field
                geo = clean if grain_level(img) >= GRAIN_THRESHOLD else \
                    np.asarray(img.convert("RGB").resize((w, h), Image.LANCZOS))
                # WHERE: the line model's own map, gated low (0.15): its thin faint outlines peak
                # at ~0.5 and a mask built from local darkness missed them (outlines merge with
                # dark shading into blobs). HOW WIDE / HOW DARK: the original's ink coverage
                # (no highlight mask here: coverage comes from darkness, so highlights get none,
                # while the dilated top-hat mask also ate every line beside white hair)
                # Print mode keeps the conservative stroke mask: model-gated hairlines add
                # thousands of sub-extrusion islands
                gate = im if min_px else lm > 0.15
                fld = ink_field(geo, gate)
                fld[bright] = 0
                im = gate & (fld > 0.35)
                if min_px:      # print mode: never thinner than one extrusion line
                    rd_cov = cv2.resize(rd.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                    fld = np.maximum(fld, rd_cov)
                ink_paths = trace_lines_contour(geo, fld, im)
                im = im | (fld > 0.3)
            else:
                ink_paths = trace_lines(clean, im, rd)
                im = im | (cv2.resize(rd.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0)
        else:                       # no line model: colour rule (dark, neutral, thin)
            im = ink_mask(clean)
            ink_paths = trace_ink(clean, im)
        sm_fill = blur_sigma > 0
        rgb, clean = fill_under(rgb, im, bright, sm_fill), fill_under(clean, im, bright, sm_fill)
        ink_cover = im
    if bands:
        from .bands import posterize
        # contours come from the (pre-mean-shift) cleaned image: mean-shift plateaus are blocky
        rgb, palette = posterize(rgb, k=bands, vote_sigma=vote_sigma, l_weight=l_weight, c_weight=c_weight,
                                 edge_thr=edge_thr, blur_sigma=blur_sigma, smooth_src=clean,
                                 progress=lambda f, s: report("bands", f, s), min_px=min_px,
                                 cover=ink_cover, thin_thr=thin_thr)
        color_precision, layer_difference = 8, 1   # keep every band as its own layer
    # print mode: the cutout mosaic is still more printable (its stacked shapes leave sub-0.7 px
    # slivers where two fitted curves trace the same edge); contour tracer for display output
    if bands and tracer == "contour" and not min_px:
        # own tracer: mode-filtered label map -> stacked sub-pixel contours -> cubic Beziers.
        # Each visible boundary belongs to exactly one shape, so no seams/slivers (no cutout needed)
        from .contour import smooth_labels, drop_slivers, region_paths
        report("trace", 0.0, "平滑色块边界")
        extra = []
        if detail_overlay:
            # small details (beauty mark, pupil glints) are taken BEFORE the mode filter, which
            # removes anything a few pixels across
            from .overlay import trace_details
            extra = trace_details(rgb, min_px=min_px)
            detail_overlay = False          # added below together with the lines
        if label_sigma > 0:
            rgb = smooth_labels(rgb, sigma=label_sigma)
        rgb = drop_slivers(rgb, keep_dark=0.08 if not ink else 0.0)
        if region_color:
            # colour pipeline: shapes from the bands, colour per region from a 50 % copy of the
            # (line-free) cleaned image, similar neighbours merged, one global palette for all
            from .palette import region_pipeline
            report("trace", 0.05, "区域取色")
            n_det = len(extra)
            rgb, recol = region_pipeline(rgb, clean, extra + list(ink_paths), n=palette_n or 128,
                                         barrier=ink_cover if ink_cover is not None and ink_cover.shape == rgb.shape[:2] else None)
            rgb = drop_slivers(rgb, keep_dark=0.08 if not ink else 0.0)
            extra, ink_paths = recol[:n_det], recol[n_det:]
        elif palette_n:
            # one global palette for every layer; same-colour neighbours then merge into one
            # region in region_paths (connected components of equal colour)
            from .palette import unify
            report("trace", 0.05, "统一调色板")
            n_det = len(extra)
            rgb, recol, _ = unify(rgb, extra + list(ink_paths), n=palette_n, min_de=palette_de)
            rgb = drop_slivers(rgb, keep_dark=0.08 if not ink else 0.0)
            extra, ink_paths = recol[:n_det], recol[n_det:]
        eye_mask = None
        if eye_boxes:
            # eyes: their own fine channel - original pixels (no AI clean / smoothing), own
            # palette, nothing merged; written into the label map so there is no patch edge
            from .eyes import eye_patch
            sx, sy = w / W, h / H
            geo0 = np.asarray(img.convert("RGB").resize((w, h), Image.LANCZOS))
            rgb, eye_mask = eye_patch(rgb, geo0, [(bx * sx, by * sy, bw * sx, bh * sy) for bx, by, bw, bh in eye_boxes],
                                      k=eye_colors)
        if region_smooth:
            # colour regions drawn like the lines: small/thin bands merged, each boundary a few
            # smooth curves between real corners (lumina_vec/regions2.py)
            from .regions2 import merge_small, region_paths_smooth
            report("trace", 0.1, "合并碎色块")
            rgb = merge_small(rgb, protect=eye_mask)
            report("trace", 0.4, "绘制色块边界")
            items = [(0, p) for p in region_paths_smooth(rgb)]
        else:
            items = region_paths(rgb, min_area=max(6, int(min_px * min_px)) if min_px else 6,
                                 progress=lambda f: report("trace", 0.1 + 0.9 * f, f"{f * 100:.0f}%"))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n'
               + "\n".join([p for _, p in items] + extra) + "\n</svg>")
        hierarchical = "stacked"
    else:
        report("trace", 0.0, "这一步无法细分，请稍候")
        cfg = vtracer.Config(filter_speckle=filter_speckle, layer_difference=layer_difference,
                             color_precision=color_precision, simplify=simplify, palette=palette,
                             hierarchical=hierarchical)
        svg = cfg.convert_pixels(np.dstack([rgb, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
    report("detail")
    if hierarchical == "cutout":
        svg = add_gap_fillers(svg)       # colour layer only; lines/details go on top afterwards
    if bands and detail_overlay:
        from .overlay import trace_details
        extra = trace_details(rgb, min_px=min_px)
        svg = svg.replace("</svg>", "\n".join(extra) + "\n</svg>")
    if ink_paths:
        svg = svg.replace("</svg>", "\n".join(ink_paths) + "\n</svg>")
    report("detail", 1.0)
    svg = re.sub(r"<svg\b([^>]*)>", lambda m: "<svg" + re.sub(r'\s(width|height)="[^"]*"', "", m.group(1))
                 + f' width="{W}" height="{H}" viewBox="0 0 {w} {h}">', svg, count=1)
    return svg, work


def main():
    ap = argparse.ArgumentParser(description="AI-assisted raster -> SVG vectorizer")
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--preset", choices=["auto", *PRESETS], default="auto")
    ap.add_argument("--hair-repair", action="store_true", help="source-detail: reconstruct conservative continuous fine strokes")
    ap.add_argument("--weights", default="weights/realesr-animevideov3.pth")
    ap.add_argument("--no-ai", action="store_true", help="skip the Real-ESRGAN clean step")
    ap.add_argument("--save-clean", default=None, help="also save the AI-cleaned raster here")
    ap.add_argument("--clean", default=None,
                    help="use an image already cleaned by `python -m lumina_vec.denoise` (skips the AI step)")
    for k, v in PRESETS["illustration"].items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    ap.add_argument("--print-mm", type=float, default=0.0,
                    help="print mode: printed length of this image's long side in mm (crop first if you only "
                         "print part of it). Removes colour features narrower than one extrusion line")
    ap.add_argument("--line-mm", type=float, default=0.42, help="extrusion line width in mm (default 0.42)")
    ap.add_argument("--lines", choices=["anime", "none"], default=None,
                    help="AI line model for the line layer (default anime if weights/netG.pth exists)")
    ap.add_argument("--tracer", choices=["contour", "vtracer"], default="contour",
                    help="colour-layer tracer for band presets (default: own sub-pixel contour tracer)")
    ap.add_argument("--fine", action="store_true",
                    help="精细模式: line layer v3 + no AI downscale + 2x work size + thin-line protection")
    ap.add_argument("--redraw-source", action="store_true",
                    help="Smooth similar source colours locally while tracing against the original edges (print-regions)")
    ap.add_argument("--smooth-lines", action="store_true",
                    help="Suppress line jaggedness and fit jagged polylines into smooth clean curves/lines (print-regions)")
    ap.add_argument("--no-feature-routing", action="store_true",
                    help="Disable ONNX CNN feature-routed conditioning (print-regions)")
    ap.add_argument("--structure", choices=["regions", "fidelity", "stacked", "structured", "flat", "lumina", "print-regions"], default=None,
                    help="输出结构: stacked (默认) | structured (封闭色块 + 连续线几何 + 分段着色, "
                         "defs/use/clipPath) | flat (同样分层, 只用 <path>)")
    ap.add_argument("--line-style", choices=["lens", "lens5", "lens7", "stroke", "v3", "contour"], default=None,
                    help="line layer: lens (fine mode default), lens5 (old per-piece), "
                         "lens7 (experimental hair continuity), stroke, v3, contour")
    ap.add_argument("--thin-thr", type=float, default=None,
                    help="protect faint thin structures from gradient smoothing (0 = off, fine mode 0.015)")
    ap.add_argument("--palette", type=int, default=None,
                    help="experimental: one global palette of N colours for all layers (0 = off)")
    ap.add_argument("--region-color", type=int, choices=[0, 1], default=None,
                    help="experimental colour pipeline: per-region colour from a 50%% image, merge, palette")
    ap.add_argument("--region-smooth", type=int, choices=[0, 1], default=None,
                    help="colour regions with smooth boundaries + merged small bands (fine mode: on)")
    ap.add_argument("--eye", action="append", default=None, metavar="X,Y,W,H",
                    help="eye box in original pixels (repeatable): drawn with the eye fine channel")
    ap.add_argument("--ink", type=int, choices=[0, 1], default=None,
                    help="separate smoothed outline layer (default: on for the grainy `shading` preset)")
    ap.add_argument('--gradient-mode', choices=['layered', 'clean'], default=None,
                    help='纯色色区渐变策略：layered 保留色阶，clean 纯净色块')
    a = ap.parse_args()
    src = Image.open(a.input)
    if a.preset == "auto":
        chosen, grain, params = auto_params(src)
        print(json.dumps(dict(auto_preset=chosen, grain=round(grain, 2), work_scale=params["work_scale"])))
    else:
        params = dict(PRESETS[a.preset])
    if a.fine:
        params = fine_params(src, params)
    if params.get("source_detail"):
        params["hair_repair"] = a.hair_repair
    elif a.hair_repair:
        ap.error("--hair-repair requires --preset source-detail")
    if a.line_style:
        params["line_style"] = a.line_style
    if a.eye:
        params["eye_boxes"] = [tuple(float(v) for v in e.split(",")) for e in a.eye]
    if a.region_smooth is not None:
        params["region_smooth"] = bool(a.region_smooth)
    if a.region_color is not None:
        params["region_color"] = bool(a.region_color)
    if a.palette is not None:
        params["palette_n"] = a.palette
    if a.thin_thr is not None:
        params["thin_thr"] = a.thin_thr
    if a.ink is not None:
        params["ink"] = bool(a.ink)
    if a.lines:
        params["lines"] = None if a.lines == "none" else a.lines
    if a.print_mm:
        params["print_mm"], params["line_mm"] = a.print_mm, a.line_mm
        if a.bands is None and params.get("bands"):
            params["bands"] = 56            # fewer colours -> fewer filament recipes -> fewer splits per layer
    if a.structure:
        params["structure"] = a.structure
    if a.redraw_source:
        params['redraw_source'] = True
    if a.smooth_lines:
        params['smooth_lines'] = True
    if a.no_feature_routing:
        params['feature_routing'] = False
    if a.gradient_mode:
        params['gradient_mode'] = a.gradient_mode
    for k in params:
        if k in ("ink", "print_mm", "line_mm", "lines", "line_style", "thin_thr", "palette_n", "region_color",
                 "region_smooth", "eye_boxes", "structure"):
            continue
        v = getattr(a, k, None)
        if v is not None:
            params[k] = v
    t = time.time()
    svg, work = vectorize(src, a.weights, ai=not a.no_ai, tracer=a.tracer,
                          clean=Image.open(a.clean) if a.clean else None, **params)
    open(a.output, "w").write(svg)
    if a.save_clean:
        work.save(a.save_clean)
    print(json.dumps(dict(paths=svg.count("<path"), kb=round(len(svg) / 1024),
                          secs=round(time.time() - t, 1), **params)))


if __name__ == "__main__":
    main()
