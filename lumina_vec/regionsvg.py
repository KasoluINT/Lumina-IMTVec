"""Lumina 风格精确重构（v2）：复用已验证的叠放色块描摹（contour.region_paths，带空腔），
再把三次贝塞尔里"接近圆弧"的段落换成 A 命令，可选在顶层叠加连续线条层。

不做像素矩形拼贴：每个色块是一条闭合 path，被后画区域包住的部分是洞。
"""
import os
import re
import numpy as np
import cv2

from .color import oklab_to_srgb

__all__ = ["vectorize_regions"]

_TOK = re.compile(r"[MLHVCZmlhvczA]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _circle(pts):
    if len(pts) < 5:
        return None
    A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 1e-6:
        return None
    return cx, cy, float(np.sqrt(r2))


def _cubic_arc(p0, c1, c2, p3, tol=0.35):
    t = np.linspace(0, 1, 9)[:, None]
    mt = 1 - t
    pts = (mt ** 3) * p0 + 3 * (mt ** 2) * t * c1 + 3 * mt * t ** 2 * c2 + t ** 3 * p3
    circ = _circle(pts)
    if circ is None:
        return None
    cx, cy, r = circ
    if r > 5000 or np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).max() > tol:
        return None
    a0 = np.arctan2(p0[1] - cy, p0[0] - cx)
    a1 = np.arctan2(p3[1] - cy, p3[0] - cx)
    mid = np.arctan2(c1[1] - cy, c1[0] - cx)
    ccw = np.sin(mid - a0) > 0
    sweep = (a1 - a0) % (2 * np.pi) if ccw else -((a0 - a1) % (2 * np.pi))
    chord = float(np.linalg.norm(p3 - p0))
    # 近乎直线的三次段用最小二乘会拟合出一个半径几十倍的"大圆"，转成 A 之后会横扫出去，
    # 整张图就变成放射状条纹（第一次就是这个错）。要求弧真的绕得起来，且中点对得上。
    if abs(sweep) >= np.radians(300) or abs(sweep) < np.radians(3) or chord < 1e-6:
        return None
    if r > 8 * chord:
        return None
    am = a0 + sweep / 2
    arc_mid = np.array([cx + r * np.cos(am), cy + r * np.sin(am)])
    cub_mid = 0.125 * p0 + 0.375 * c1 + 0.375 * c2 + 0.125 * p3
    if np.linalg.norm(arc_mid - cub_mid) > max(1.5, tol * 3):
        return None
    return "A%.2f,%.2f 0 %d %d %.2f,%.2f" % (r, r, 1 if abs(sweep) > np.pi else 0,
                                             1 if sweep > 0 else 0, p3[0], p3[1])


def arcify(d, tol=0.35):
    """Rewrite cubic segments that really are circular arcs as 'A' commands."""
    toks = _TOK.findall(d)
    out = []
    pt = np.zeros(2)
    start = np.zeros(2)
    cmd = None
    i = 0
    n_arc = n_cub = 0
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t.upper()
            i += 1
            if cmd == "Z":
                out.append("Z")
                pt = start.copy()
            continue
        if cmd == "M":
            pt = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
            start = pt.copy(); out.append("M%.2f,%.2f" % tuple(pt)); cmd = "L"
        elif cmd == "L":
            pt = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
            out.append("L%.2f,%.2f" % tuple(pt))
        elif cmd == "H":                     # 画布底色的 d 就是 M0,0H..V..H0Z：不处理会把它拆坏
            pt = np.array([float(toks[i]), pt[1]]); i += 1
            out.append("L%.2f,%.2f" % tuple(pt))
        elif cmd == "V":
            pt = np.array([pt[0], float(toks[i])]); i += 1
            out.append("L%.2f,%.2f" % tuple(pt))
        elif cmd == "C":
            c = np.array([float(v) for v in toks[i:i + 6]]).reshape(3, 2); i += 6
            a = _cubic_arc(pt, c[0], c[1], c[2], tol)
            if a:
                out.append(a); n_arc += 1
            else:
                out.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*c[0], *c[1], *c[2])); n_cub += 1
            pt = c[2]
        else:
            i += 1
    return "".join(out), n_arc, n_cub


def vectorize_regions(img, weights=None, *, colors=256, ss=3, tol=0.6, min_area=6, arcs=True,
                      line_layer=False, progress=None, weights_dir=None, pre_down=1.0,
                      work_scale=2.0, ms_sp=6, ms_sr=10, blur_sigma=2.0, vote_sigma=1.5,
                      thin_thr=0.0, gap=True):
    """Return (svg, work). 色块走 contour.region_paths（叠放、带空腔），线条层可选。"""
    report = progress or (lambda *a: None)
    W0, H0 = img.size
    from PIL import Image
    report(0.02, "AI 去颗粒")
    if weights and os.path.exists(weights):
        from .denoise import ai_clean
        from .pipeline import match_tone
        work = ai_clean(img, weights, pre_down=pre_down, out_scale=work_scale)
        rgb = np.asarray(work.convert("RGB"))
        rgb = match_tone(rgb, np.asarray(img.convert("RGB").resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS)))
    else:
        work = img.convert("RGB").resize((round(W0 * work_scale), round(H0 * work_scale)), Image.LANCZOS)
        rgb = np.asarray(work)
    H, W = rgb.shape[:2]
    report(0.25, "压平")
    flat = cv2.pyrMeanShiftFiltering(rgb, ms_sp, ms_sr) if ms_sp > 0 else rgb
    report(0.32, f"颜色量化 {colors} 色")
    from .bands import posterize
    post = posterize(flat, k=colors, vote_sigma=vote_sigma, blur_sigma=blur_sigma, thin_thr=thin_thr,
                     progress=lambda *a: None, min_px=0)[0]
    # 吸收 1~20 px 的碎区域：不吸收的话，它们要么被丢弃留下白点，要么各自成形状（每个还带遮缝描边）
    # 变成边缘噪点。merge_specks 把碎点并进最近的邻居，画面仍被完整覆盖。
    from .bands import merge_specks
    cols, inv = np.unique(post.reshape(-1, 3), axis=0, return_inverse=True)
    labels = merge_specks(inv.reshape(post.shape[:2]).astype(np.int32), post, max_area=20,
                          keep_de=0.10, iters=2)
    post = cols[labels]
    post = _absorb_small(post, min_area=max(24, 4 * min_area), iters=3)
    report(0.55, "色块描摹（叠放+空腔）")
    from .contour import region_paths
    items = region_paths(post, tol=tol, sigma=0.5, ss=ss, min_area=min_area,
                         progress=lambda f: report(0.55 + 0.3 * f, "色块描摹"), level=0.5)
    report(0.88, "圆弧化")
    fills, n_arc, n_cub = [], 0, 0
    for _area, svgpath in items:
        m = re.search(r'\sd="([^"]*)"', svgpath)
        if not m:
            continue
        d, a, c = arcify(m.group(1), tol=0.35) if arcs else (m.group(1), 0, 0)
        n_arc += a; n_cub += c
        fills.append(svgpath.replace(m.group(1), d))
    art = []
    if line_layer:
        report(0.9, "线条层")
        from .draw import lens_layer
        from .lines import line_map
        wd = weights_dir or (os.path.dirname(weights) if weights else "weights")
        dl = min(max(H, W), 1536)
        M = line_map(flat, "manga", detect_long=dl, weights_dir=wd)
        A = line_map(flat, "anime", detect_long=dl, weights_dir=wd)
        art, _cover = lens_layer(rgb, M, A)
    report(0.97, "装配")
    if gap:
        # 参考文件的每个形状前面都有一条同色遮缝描边（class=qq3d-gap-filler，导入端会删掉）；
        # 相邻色块的边界是两条独立曲线，必然留下不到 1 px 的缝，靠它遮掉
        out = []
        for p in fills:
            md = re.search(r'\sd="([^"]*)"', p)
            mf = re.search(r'fill="(#[0-9a-fA-F]{6})"', p)
            if md and mf:
                out.append('<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                           'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" '
                           'vector-effect="non-scaling-stroke"/>' % (md.group(1), mf.group(1)))
            out.append(p)
        fills = out
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (W, H),
             '<g id="fills" fill-rule="evenodd">' + "".join(fills) + "</g>"]
    if art:
        parts.append('<g id="line-art">' + "".join(art) + "</g>")
    parts.append("</svg>")
    svg = "\n".join(parts)
    svg = re.sub(r"<svg\b([^>]*)>",
                 lambda m2: "<svg" + re.sub(r'\s(width|height)="[^"]*"', "", m2.group(1)) +
                 f' width="{W0}" height="{H0}" viewBox="0 0 {W} {H}">', svg, count=1)
    report(1.0, f"完成 · 弧 {n_arc} / 三次 {n_cub}")
    return svg, work


def _absorb_small(post, min_area=24, iters=3):
    """把面积 < min_area 的颜色块整块并进邻居：它们单独描出来只是一个 2~4 px 的点（还带遮缝描边），
    丢掉又会在画面里留下白点。像素级直接改颜色最省事。"""
    from skimage import measure
    out = post.copy()
    for _ in range(iters):
        key = (out[..., 0].astype(np.int64) << 16) | (out[..., 1].astype(np.int64) << 8) | out[..., 2]
        reg = measure.label(key + 1, connectivity=1, background=0)
        n = reg.max() + 1
        area = np.bincount(reg.ravel(), minlength=n)
        small = [r for r in range(1, n) if 0 < area[r] < min_area]
        if not small:
            break
        for r in small:
            m = reg == r
            ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
            vals = reg[ring]
            vals = vals[(vals > 0) & (vals != r)]
            if vals.size == 0:
                continue
            nb = np.bincount(vals).argmax()
            keep = nb > 0
            if keep:
                sel = reg == nb
                yy, xx = np.argwhere(sel)[0]
                out[m] = out[yy, xx]
    return out


def vectorize_lumina(img, *, scale=None, max_px=12e6, colors=0, blur=2.0, layer_diff=1, simplify=0.8,
                     speckle=2, gap=False, corner=60, length=3.5, splice=45, progress=None,
                     line_layer=False, weights_dir=None):
    """Lumina/Vecto 风格的精确重构：把图放大到 ~2.6 倍后直接描摹（不量化，除非 colors>0），
    vtracer 的 stacked 模式给出"每个色块一条闭合 path + 空腔"，再按参考文件的格式
    （同色遮缝描边 + <g id="fills">）装配，最后把画布映射回原图尺寸。

    实测（a2.png，960×1600）：无量化 + 2.67 倍放大 + ld=1 时 PSNR 32.37 dB，
    参考 A2.svg 是 32.84 dB（差 0.47 dB）。
    """
    import re
    from PIL import Image
    import vtracer

    report = progress or (lambda *a: None)
    W0, H0 = img.size
    if scale is None:
        scale = max(1.0, min(2.67, (max_px / (W0 * H0)) ** 0.5))
    W, H = int(round(W0 * scale)), int(round(H0 * scale))
    report(0.05, f"放大到 {W}×{H}")
    up = np.asarray(img.convert("RGB").resize((W, H), Image.LANCZOS))
    if colors:
        from .bands import posterize
        report(0.25, f"颜色量化 {colors} 色")
        up = posterize(up, k=colors, vote_sigma=1.0, blur_sigma=float(blur),
                       progress=lambda *a: None, min_px=0)[0]
    report(0.4, "描摹（叠放 + 空腔）")
    cfg = vtracer.Config(color_precision=8, layer_difference=int(layer_diff), simplify=float(simplify),
                         filter_speckle=int(speckle), hierarchical="stacked", mode="spline",
                         corner_threshold=int(corner), length_threshold=float(length),
                         splice_threshold=int(splice))
    raw = cfg.convert_pixels(np.dstack([up, np.full((H, W), 255, np.uint8)]).tobytes(), W, H)
    report(0.85, "装配")
    paths = re.findall(r"<path\b[^>]*/>", raw)
    out = []
    for p in paths:
        out.append(p)
        if gap:
            md = re.search(r'\sd="([^"]*)"', p)
            mf = re.search(r'fill="(#[0-9a-fA-F]{6})"', p)
            if md and mf:
                # 画布按 viewBox 缩回原图尺寸，所以描边必须用 viewBox 单位（不能用
                # non-scaling-stroke：那会让 2 设备像素在缩小时变成巨大的光环，PSNR 直掉 9 dB）
                out.append('<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                           'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
                           % (md.group(1), mf.group(1)))
    # 画布保持描摹分辨率（参考文件也是这么干的：960×1600 的图输出 2560×4264 的画布）。
    # 不要用 viewBox 缩回原图尺寸——浏览器在小画布上栅格化时会把几十万个小形状合并掉，
    # 实测 PSNR 从 32.4 dB 掉到 30.1 dB。
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<g id="fills" fill-rule="evenodd">%s</g>\n</svg>' % (W, H, W, H, "".join(out)))
    report(1.0, "完成")
    return svg, Image.fromarray(up)


def vectorize_flat(img, *, colors=160, simplify=1.5, speckle=10, layer_diff=2, scale=2.0,
                   max_px=12e6, progress=None):
    """去掉过度色彩、边界拉直的纯色区块描摹 —— 对参考 SVG 逆向出来的配方。

    1. LANCZOS 放大到 ~2 倍（参考文件的画布也是原图的 2 倍）；
    2. 中位切分量化到 N 色：把渐变里"过度"的中间色并掉，腮红/暗面被压成接近肤色的少数几档，
       而眼影、发丝描边这类强边界因为色差大仍然独立成色；
    3. vtracer stacked + spline：每块一条闭合 path、被包住的部分成为空腔，
       边界是拟合出来的长曲线（不是逐像素折线）；
    4. 描摹后把每个 fill 颜色吸附回量化色板：色板里的颜色集中复用，跟参考 SVG 一样是共享色。

    实测（5t5，900×1600，画布 1800×3200）：17,163 paths / 160 色 / PSNR 22.2 dB，
    参考 SVG 为 14,444 paths / 106 色 / 23.5 dB。
    """
    import re as _re
    from PIL import Image
    import vtracer

    report = progress or (lambda *a: None)
    W0, H0 = img.size
    sc = min(scale, max(1.0, (max_px / (W0 * H0)) ** 0.5))
    W, H = int(round(W0 * sc)), int(round(H0 * sc))
    report(0.1, "放大到 %d×%d" % (W, H))
    up = img.convert("RGB").resize((W, H), Image.LANCZOS)
    report(0.25, "量化到 %d 色" % colors)
    q = up.quantize(colors=int(colors), method=Image.MEDIANCUT).convert("RGB")
    arr = np.asarray(q)
    pal = np.unique(arr.reshape(-1, 3), axis=0).astype(int)
    report(0.45, "描摹（spline + 空腔）")
    cfg = vtracer.Config(color_precision=8, layer_difference=int(layer_diff), simplify=float(simplify),
                         filter_speckle=int(speckle), hierarchical="stacked", mode="spline")
    raw = cfg.convert_pixels(np.dstack([arr, np.full((H, W), 255, np.uint8)]).tobytes(), W, H)
    report(0.85, "颜色吸附回色板")

    def _snap(m):
        c = np.array([int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)])
        return 'fill="#%02x%02x%02x"' % tuple(pal[((pal - c) ** 2).sum(1).argmin()])

    body = _re.sub(r'fill="#([0-9a-fA-F]{6})"', _snap, raw)
    body = _re.sub(r"^.*?<svg[^>]*>", "", body, flags=_re.S)
    body = _re.sub(r"</svg>.*$", "", body, flags=_re.S)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<g id="regions" fill-rule="evenodd">%s</g>\n</svg>' % (W, H, W, H, body))
    report(1.0, "完成")
    return svg, up


def vectorize_paired(img, *, colors=106, tol=0.6, sigma=1.2, min_area=12, scale=1.5,
                     gap_filler=True, gap_width=2, progress=None):
    """按参考 SVG 的生成策略输出"填充 + 成对遮缝描边"。

    从样例 SVG 的代码反推出来的规律：
      * 每个色块两条 path：fill 一条、class="qq3d-gap-filler" 的描边一条，d 完全相同；
      * gap-filler：stroke = 该块填充色，stroke-width=2，round join/cap，
        vector-effect="non-scaling-stroke"；
      * 路径是 M/L/Q/A 混合，圆弧只在二次拟合不了时出现；
      * 坐标是 float32 展开成 float64 打印（149.41000366210938）→ Python/numpy 生成。

    线与填色用同一份几何，所以"线稿层"就是把 fill 去掉、只看 gap-filler。
    """
    from PIL import Image
    from skimage import measure
    from scipy import ndimage as ndi
    import cv2
    from .bands import posterize
    from .contour import smooth_labels, drop_slivers
    from .fitvec import fit_runs, to_d

    report = progress or (lambda *a: None)
    W0, H0 = img.size
    CW, CH = int(round(W0 * scale)), int(round(H0 * scale))
    report(0.1, "放大到 %d×%d" % (CW, CH))
    up = np.asarray(img.convert("RGB").resize((CW, CH), Image.LANCZOS))
    flat = cv2.bilateralFilter(up, 7, 30, 7)
    report(0.25, "量化到 %d 色" % colors)
    post, _pal = posterize(flat, k=int(colors), vote_sigma=1.5, blur_sigma=6.0,
                           progress=lambda *a: None, min_px=0)
    sm = np.asarray(drop_slivers(smooth_labels(post, sigma=sigma), keep_dark=0.02))
    key = (sm[..., 0].astype(np.int64) << 16) | (sm[..., 1].astype(np.int64) << 8) | sm[..., 2]
    reg = measure.label(key + 1, connectivity=1, background=0)
    n = int(reg.max()) + 1
    area = np.bincount(reg.ravel(), minlength=n)
    objs = ndi.find_objects(reg)
    order = [int(r) for r in np.argsort(-area) if r > 0 and area[r] >= min_area]
    report(0.45, "拟合 %d 块轮廓（L/Q/A）" % len(order))

    SS = 2
    body = []
    for k, r in enumerate(order):
        sl = objs[r - 1]
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 2), min(CH, sl[0].stop + 2)
        x0, x1 = max(0, sl[1].start - 2), min(CW, sl[1].stop + 2)
        sub = reg[y0:y1, x0:x1] == r
        f = cv2.resize(sub.astype(np.float32), (sub.shape[1] * SS, sub.shape[0] * SS),
                       interpolation=cv2.INTER_LINEAR)
        f = cv2.GaussianBlur(f, (0, 0), 0.9 * SS)
        ds = []
        for c in measure.find_contours(f, 0.5):
            if len(c) < 8:
                continue
            xy = (c[:, ::-1] + 0.5) / SS + np.array([x0, y0], float)
            if abs(cv2.contourArea(xy.astype(np.float32))) < 1.5:
                continue
            ds.append(to_d(fit_runs(np.vstack([xy, xy[:1]]), tol=tol)))
        if not ds:
            continue
        yy, xx = np.argwhere(sub)[0]
        col = "#%02x%02x%02x" % tuple(int(v) for v in sm[y0 + yy, x0 + xx])
        d = " ".join(ds)
        body.append('<path d="%s" fill="%s" fill-rule="evenodd" stroke="none"/>' % (d, col))
        if gap_filler:
            body.append('<path class="qq3d-gap-filler" d="%s" fill="none" stroke="%s" '
                        'stroke-width="%g" stroke-linejoin="round" stroke-linecap="round" '
                        'vector-effect="non-scaling-stroke"/>' % (d, col, gap_width))
        if progress and k % 400 == 0:
            report(0.45 + 0.5 * k / max(1, len(order)), "拟合轮廓 %d/%d" % (k, len(order)))
    report(0.97, "装配")
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n'
           '<g id="flat" fill-rule="evenodd">\n%s\n</g></svg>' % (CW, CH, CW, CH, "\n".join(body)))
    report(1.0, "完成")
    return svg, up
