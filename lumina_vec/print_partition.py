"""Quantized, non-overlapping solid SVG regions with optional strand recovery."""
import re
import numpy as np,cv2
from PIL import Image
from scipy.ndimage import find_objects
from skimage.measure import label
from .bands import posterize
from .shared_regions import segment,trace_regions
from .palette_regions import restore_interior_details
from .strand_reconstruction import extract_branches,supported_pairs,rebuild_partition
from .svg_layers import boundary_then_fill
from .outline_stages import snap_gray


def compact_fragments(labels,colors,scale):
    area=np.bincount(labels.ravel());boxes=find_objects(labels+1)
    span=np.array([max(s.stop-s.start for s in b) for b in boxes])
    k1,k2=[],[]
    for a,b in [(labels[:-1],labels[1:]),(labels[:,:-1],labels[:,1:])]:
        different=a!=b;va=a[different].astype(np.uint64);vb=b[different].astype(np.uint64)
        k1.append((va<<32)|vb);k2.append((vb<<32)|va)
    uniq_k,contacts=np.unique(np.concatenate(k1+k2),return_counts=True)
    src=(uniq_k>>32).astype(np.int32);dst=(uniq_k&0xffffffff).astype(np.int32)
    valid=(area[src]<scale**2)&(span[src]<4*scale)&(area[dst]>=scale**2)
    src,dst,contacts=src[valid],dst[valid],contacts[valid]
    distance=np.linalg.norm(colors[src].astype(float)-colors[dst],axis=1)
    order=np.lexsort((-contacts,distance,src));mapping=np.arange(len(colors));seen=set()
    for a,b in zip(src[order],dst[order]):
        if a not in seen:mapping[a]=b;seen.add(a)
    palette,ids=np.unique(colors,axis=0,return_inverse=True);field=ids[mapping[labels]]
    regions=label(field+1,connectivity=1)-1
    values=np.empty((regions.max()+1,3),np.uint8);values[regions.ravel()]=palette[field.ravel()]
    return regions,values


def trace_binary_ink(img, rgb, outline_layer):
    """Trace exact black/white artwork without deleting pixels or filling holes."""
    if 'A' in img.getbands() or 'transparency' in img.info:
        alpha=np.asarray(img.convert('RGBA'))[...,3]
        if not np.all((alpha==0)|(alpha==255)):
            return None
        visible=alpha==255
    else:
        visible=np.ones(rgb.shape[:2],bool)
    black=np.all(rgb==0,axis=2)&visible
    white=np.all(rgb==255,axis=2)&visible
    if np.any(visible&~(black|white)):
        return None
    # Two visible colours may share edges; label transparency as exterior.
    regions=label(black.astype(np.uint8)+2*white,connectivity=1)-1
    palette=np.empty(regions.max()+1,dtype=bool)
    palette[regions[visible]]=white[visible]
    svg,_=trace_regions(regions,np.where(palette,'#ffffff','#000000'),exact_grid=True)
    if outline_layer:
        start=svg.index('>')+1
        svg=svg[:start]+'<g id="region-boundaries"/><g id="region-colours">'+svg[start:-6]+'</g></svg>'
    return svg,img.convert('RGB' if visible.all() else 'RGBA')


def _vectorize_clean_planar(source, fmaps, orig_shape, *, smooth_lines=True, refit_outline=True, locate_boundaries=True, outline_layer=True, progress=None):
    """Run lossless Teacher-SVG Stage 2->3->4 planar reconstruction directly at SR resolution."""
    from functools import partial
    from .planar_partition import build_planar_partition
    from .structural_curves import fit_compact_outline, refit_sketch_chain, smooth_outline_junctions
    from .boundary_corridor import SourceCorridor

    step = progress or (lambda *_: None)
    h_orig, w_orig = orig_shape
    h, w = source.shape[:2]
    if h == h_orig and w == w_orig and (h * w) <= 4_000_000:
        # Ensure clean 2x super-sampled geometry grid for smooth sub-pixel hairlines on small/medium inputs
        sr_in = cv2.resize(source, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_CUBIC)
    else:
        sr_in = source
    hs, ws = sr_in.shape[:2]

    step(.25, 'Stage 2->3：无损线条骨架提取 + 保护区解锁 + 渐变色阶轮廓划分')
    regions, colors, line_mask, p_stats = build_planar_partition(sr_in, fmaps, (h_orig, w_orig))
    print('[planar_partition]', p_stats, flush=True)

    step(.62, '整理细碎色区')
    regions, colors = compact_fragments(regions, colors, scale=max(1, int(round(hs / max(1, h_orig)))))

    step(.76, 'Stage 4：SVG共享矢量边界拟合')
    fitting = {}
    corridor = None
    metadata = {'data-planar-engine': 'v1', 'data-raw-compartments': str(p_stats.get('n_regions_raw', len(colors)))}
    if refit_outline:
        fitting = dict(
            structural_fitter=partial(fit_compact_outline, smooth_lines=smooth_lines),
            secondary_fitter=partial(refit_sketch_chain, smooth_lines=smooth_lines),
            secondary_network=partial(smooth_outline_junctions, smooth_lines=smooth_lines),
        )
        if locate_boundaries:
            corridor = SourceCorridor(sr_in, colors)
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
    if smooth_lines and hasattr(fmaps, 'delicate_detail'):
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
        regions, ['#%02x%02x%02x' % tuple(c) for c in colors],
        tol=tol, seam=0, geometry_guard=True, smoothing=smoothing,
        structural_scale=struct_sc, compact_commands=True,
        chain_scale_fn=chain_scale_fn, **fitting,
    )
    svg = re.sub(r'width="\d+" height="\d+"', f'width="{w_orig}" height="{h_orig}"', svg, count=1)

    if outline_layer:
        step(.96, '生成教师SVG结构（gap-filler轮廓 + 线稿轮廓约束单色填充层）')
        r_areas = np.bincount(regions.ravel(), minlength=len(colors))
        r_lm = np.bincount(regions.ravel(), weights=line_mask.astype(np.float32).ravel(), minlength=len(colors)) / np.maximum(r_areas, 1)
        line_region_ids = set(np.flatnonzero(r_lm >= 0.55).tolist())
        metadata['data-line-art-paths'] = str(len(line_region_ids))
        metadata['data-color-fill-paths'] = str(len(colors) - len(line_region_ids))
        metadata['data-segments-before-refit'] = str(stats.get('segments_before_secondary', 0))
        metadata['data-segments-after-refit'] = str(stats.get('curve_segments', 0))
        if corridor is not None:
            metadata['data-source-chains-considered'] = str(corridor.considered)
            metadata['data-source-chains-moved'] = str(corridor.moved)
        svg = boundary_then_fill(svg, hidden_strokes=(), metadata=metadata, line_region_ids=line_region_ids)

    preview = Image.fromarray(colors[regions]).resize((w_orig, h_orig), Image.Resampling.LANCZOS)
    step(1., '完成')
    return svg, preview


def vectorize_print(img,bands=112,recover_strands=True,progress=None,outline_layer=True,gradient_mode='layered',refit_outline=True,redraw_source=False,smooth_lines=False,feature_routing=True,sr_scale=None,*,locate_boundaries=True):
    if gradient_mode not in ('layered','clean'):
        raise ValueError('gradient_mode must be layered or clean')
    step=progress or (lambda *_:None);source=np.asarray(img.convert('RGB'));h,w=source.shape[:2]
    h_orig,w_orig=h,w
    exact=trace_binary_ink(img,source,outline_layer)
    if exact is not None:
        step(1.,'完成：保留纯黑白像素边界')
        return exact
    rgb=source
    fmaps=None
    if feature_routing and gradient_mode=='clean':
        import os
        from dataclasses import replace
        from .feature_routing import infer_feature_maps, realesr_dejpeg_onnx, render_vector_heatmap_png
        step(.004,'整图倍率放大 + 去杂色平滑锐化 + CNN 热力线条识别')
        fmaps=infer_feature_maps(source, sr_scale=sr_scale)
        if fmaps is not None:
            print('[feature_routing]',fmaps.summary,flush=True)
            hm_out = os.environ.get('LUMINA_HEATMAP_OUT', '').strip()
            if hm_out and getattr(fmaps, 'line_vector_fields', None) is not None:
                try:
                    render_vector_heatmap_png(source, hm_out, vec_fields=fmaps.line_vector_fields)
                except Exception:
                    pass
            if getattr(fmaps, 'sr_hi_rgb', None) is not None:
                source = fmaps.sr_hi_rgb
                rgb = fmaps.sr_hi_rgb
                h, w = source.shape[:2]
                fmaps = replace(
                    fmaps,
                    ink_main_geom=cv2.resize(fmaps.ink_main_geom, (w, h), interpolation=cv2.INTER_LINEAR),
                    fine_strand=cv2.resize(fmaps.fine_strand, (w, h), interpolation=cv2.INTER_LINEAR),
                    delicate_detail=cv2.resize(fmaps.delicate_detail, (w, h), interpolation=cv2.INTER_LINEAR),
                    mergeable_wash=cv2.resize(fmaps.mergeable_wash, (w, h), interpolation=cv2.INTER_LINEAR),
                    surface_probs=np.stack([cv2.resize(fmaps.surface_probs[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=0),
                    line_net_maps=np.stack([cv2.resize(fmaps.line_net_maps[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=0) if fmaps.line_net_maps is not None else None,
                    elem_grad_maps=np.stack([cv2.resize(fmaps.elem_grad_maps[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(8)], axis=0) if fmaps.elem_grad_maps is not None else None,
                    grad_core_95=(cv2.resize(fmaps.grad_core_95.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0) if getattr(fmaps, 'grad_core_95', None) is not None else None,
                    line_buffer_5=(cv2.resize(fmaps.line_buffer_5.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0) if getattr(fmaps, 'line_buffer_5', None) is not None else None,
                    sr_clean_1x=source,
                )
            elif fmaps.sr_clean_1x is not None:
                source = fmaps.sr_clean_1x
                rgb = fmaps.sr_clean_1x
            if fmaps.summary['grain_ratio'] >= .22:
                cleaned_sr = realesr_dejpeg_onnx(rgb, fmaps=fmaps)
                if cleaned_sr is not None and cleaned_sr.shape == rgb.shape:
                    rgb = cleaned_sr
    elif feature_routing:
        from .feature_routing import repair_image_defects_onnx
        step(.004,'CNN 图像缺陷预修复')
        source, _ = repair_image_defects_onnx(source)
        rgb = source
    if gradient_mode == 'clean' and feature_routing and fmaps is not None and getattr(fmaps, 'line_vector_fields', None) is not None:
        return _vectorize_clean_planar(
            source, fmaps, (h_orig, w_orig),
            smooth_lines=smooth_lines, refit_outline=refit_outline,
            locate_boundaries=locate_boundaries, outline_layer=outline_layer, progress=progress,
        )
    already_sr_cleaned = fmaps is not None and getattr(fmaps, 'sr_hi_rgb', None) is not None
    if redraw_source or (smooth_lines and not already_sr_cleaned) or (fmaps is not None and fmaps.summary['grain_ratio'] >= .22):
        from .source_redraw import redraw_illustration
        step(.007,'保边平滑颜色表面')
        rgb=redraw_illustration(rgb, fmaps=fmaps)
    line_evidence=None
    cnn_lines=None
    wall_mask=None
    heat_repaired=None
    heat_bridged=None
    if gradient_mode=='clean':
        from .source_lines import source_line_evidence
        from .color import srgb_to_oklab
        step(.01,'从原图提取明暗线条证据')
        line_evidence=source_line_evidence(source)
        if fmaps is not None:
            from .two_layer_engine import repair_line_heatmap
            from .compartment_coloring import build_line_wall_mask, sanitize_colors_inside_compartments
            L_raw_src = srgb_to_oklab(source)[..., 0].astype(np.float32)
            if getattr(fmaps, 'line_vector_fields', None) is not None:
                vf0 = fmaps.line_vector_fields[0]
                vf1 = fmaps.line_vector_fields[1]
                if vf0.shape != (h, w):
                    vf0 = cv2.resize(vf0, (w, h), interpolation=cv2.INTER_AREA)
                    vf1 = cv2.resize(vf1, (w, h), interpolation=cv2.INTER_AREA)
                stroke_raw = np.clip(np.maximum(vf0, vf1 * 0.85), 0.0, 1.0)
            elif getattr(fmaps, 'line_net_maps', None) is not None:
                lm = fmaps.line_net_maps
                l0_p, l1_p, l3_p = lm[0], lm[1], lm[3]
                l1_c = np.where((l0_p > 0.14) | (l1_p > 0.50), l1_p * 0.92, l1_p * 0.48)
                l3_c = np.where((l0_p > 0.14) | (l3_p > 0.56), l3_p * 0.88, l3_p * 0.44)
                stroke_raw = np.maximum(l0_p, np.maximum(l1_c, l3_c))
            else:
                stroke_raw = np.maximum(fmaps.ink_main_geom, fmaps.fine_strand * 0.80)
            iris_p = fmaps.elem_grad_maps[3] if getattr(fmaps, 'elem_grad_maps', None) is not None else np.zeros((h, w), np.float32)
            grad_e1 = fmaps.elem_grad_maps[1] if getattr(fmaps, 'elem_grad_maps', None) is not None else np.zeros((h, w), np.float32)
            cel_e0 = fmaps.elem_grad_maps[0] if getattr(fmaps, 'elem_grad_maps', None) is not None else np.zeros((h, w), np.float32)
            wash_p = fmaps.mergeable_wash if getattr(fmaps, 'mergeable_wash', None) is not None else np.zeros((h, w), np.float32)
            glint_ex = (L_raw_src > 0.88) & ((iris_p > 0.28) | (fmaps.delicate_detail > 0.30))
            # Never expand wash_ex with E1 so LineVectorNet line bridging is 100% never blocked on colored hair/eyes
            wash_ex = (wash_p >= 0.68) & (stroke_raw < 0.48)
            heat_repaired, heat_bridged = repair_line_heatmap(stroke_raw, L_raw_src, exclude_mask=(glint_ex | wash_ex), max_gap=18)
            wall_mask = build_line_wall_mask(source, fmaps=fmaps, heat_repaired=heat_repaired)
            rgb = sanitize_colors_inside_compartments(rgb, fmaps=fmaps, wall_mask=wall_mask)

            # Strict Line-Priority Safe Gradient Isolation (`线条绝对优先权`):
            # 1) LineVectorNet owns a 2px-buffered corridor shield (`line_shield`) around ALL detected strokes/seams;
            # 2) Any high line-density neighborhood (dense hair wisps, eyelashes, lace, dense curves) is shielded;
            # 3) Only open, line-free smooth-gradient interiors (`safe_grad_zone`) trigger strong gradient smoothing
            #    and suppress false legacy `fine_strand` rim shadows.
            l_sc_w = float(np.sqrt(max(1.0, (h * w) / float(h_orig * w_orig))))
            has_vec = getattr(fmaps, 'line_vector_fields', None) is not None
            k_sh = max(3, int(round(2.0 * l_sc_w)) | 1)
            if has_vec:
                vf_base = np.maximum(fmaps.line_vector_fields[0], fmaps.line_vector_fields[1] * 0.85)
                if vf_base.shape != (h, w):
                    vf_base = cv2.resize(vf_base, (w, h), interpolation=cv2.INTER_AREA)
                line_core = (vf_base >= 0.40) | (heat_repaired >= 0.42) | heat_bridged | wall_mask
                weak_line_guard = (vf_base >= 0.35) | (heat_repaired >= 0.36) | ((vf_base >= 0.24) & (L_raw_src < 0.36))
            else:
                line_core = (heat_repaired >= 0.42) | heat_bridged | wall_mask
                weak_line_guard = (heat_repaired >= 0.36) | ((heat_repaired >= 0.24) & (L_raw_src < 0.36)) | wall_mask
            line_shield = cv2.dilate(line_core.astype(np.uint8), np.ones((k_sh, k_sh), np.uint8)) > 0
            k_den = max(11, int(round(11.0 * l_sc_w)) | 1)
            line_density = cv2.boxFilter(line_core.astype(np.float32), -1, (k_den, k_den), normalize=True)
            # Three-Zone Spatial Partition:
            # 1) `safe_grad_zone`: 95%-volume uniformly shrunk closed gradient cores (skip Line CNN clutter!)
            # 2) `line_shield` & `weak_line_guard`: 5% outer buffer ring + non-gradient zones own crisp lines/hair
            # 3) `irregular_zone`: residual unshielded 5% buffer pockets sealed cleanly into enclosing fill compartments
            grad_core_hi = getattr(fmaps, 'grad_core_95', None)
            line_buf_hi = getattr(fmaps, 'line_buffer_5', None)
            if grad_core_hi is not None and grad_core_hi.shape == (h, w):
                closed_grad_core = grad_core_hi
            else:
                closed_grad_core = (grad_e1 > 0.64) & (cel_e0 < 0.30) & ((wash_p >= 0.32) | (grad_e1 >= 0.78))
            safe_grad_zone = closed_grad_core & (~line_shield) & (~weak_line_guard) & (line_density < 0.22)
            irregular_zone = (line_buf_hi if (line_buf_hi is not None and line_buf_hi.shape == (h, w)) else np.zeros((h, w), bool)) & (~line_shield) & (~safe_grad_zone)
            grad_e2 = fmaps.elem_grad_maps[2] if getattr(fmaps, 'elem_grad_maps', None) is not None else np.zeros((h, w), np.float32)
            # Only lock strictly flat high-confidence cel blocks so soft radial cheek blushes smooth evenly
            cel_lock = (cel_e0 >= 0.68) & (~closed_grad_core) & (grad_e2 < 0.15)

            eff_strand = np.where(safe_grad_zone, 0.0, fmaps.fine_strand) if has_vec else fmaps.fine_strand
            eff_ink_g = np.where(safe_grad_zone, 0.0, fmaps.ink_main_geom) if has_vec else fmaps.ink_main_geom
            # Slope-protected compartment flow smoothing inside 95% gradient interiors & irregular buffer transitions:
            L_sm_sl = cv2.GaussianBlur(L_raw_src, (0, 0), 1.3 * l_sc_w)
            gx_l = cv2.Sobel(L_sm_sl, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
            gy_l = cv2.Sobel(L_sm_sl, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
            slope_l = np.hypot(gx_l, gy_l) * l_sc_w
            hard_edge = wall_mask | line_shield | cel_lock | (slope_l > 0.024) | (fmaps.delicate_detail > 0.24) | (eff_strand > 0.26)
            flow_m = (~cv2.dilate(hard_edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.float32)
            wall_only_m = (~cv2.dilate((wall_mask | cel_lock | (fmaps.delicate_detail > 0.24) | (eff_strand > 0.26)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.float32)
            grad_comb = np.maximum(wash_p, np.where(safe_grad_zone, grad_e1 * 0.90, 0.0))
            grad_comb = np.where(cel_lock, np.minimum(grad_comb, 0.20), grad_comb)
            grad_base = np.clip((np.maximum(grad_comb, fmaps.surface_probs[1]) - 0.24) / 0.34, 0.0, 0.85) * (~line_shield) * (~cel_lock)
            grad_z = grad_base * np.maximum(flow_m, wall_only_m * np.clip(1.0 - slope_l / 0.032, 0.0, 0.75))
            if (grad_z > 0.05).any():
                sig_f = 9.0 * l_sc_w
                num_f = cv2.GaussianBlur(rgb.astype(np.float32) * flow_m[..., None], (0, 0), sig_f)
                den_f = cv2.GaussianBlur(flow_m, (0, 0), sig_f)[..., None]
                sm_rgb = np.where(den_f > 1e-3, num_f / np.maximum(den_f, 1e-3), rgb.astype(np.float32))
                rgb = np.clip(np.rint(rgb.astype(np.float32) * (1.0 - grad_z[..., None]) + sm_rgb * grad_z[..., None]), 0, 255).astype(np.uint8)
            raw_le = np.clip(.85 * eff_strand + .35 * eff_ink_g, 0, 1)
            line_evidence = np.where(safe_grad_zone, np.maximum(heat_repaired * 0.90, raw_le), np.maximum(line_evidence, raw_le))
            cnn_lines = ((eff_strand > .32) | (eff_ink_g > .38) | heat_bridged | (heat_repaired >= .44)) & (np.maximum(line_evidence, heat_repaired) > .18)
            delicate_pin = (fmaps.delicate_detail > .25) & (L_raw_src < .74) & (line_evidence > .20) & (~safe_grad_zone)
            pin_all = cnn_lines | delicate_pin
            if pin_all.any():
                rgb = np.where(pin_all[..., None], source, rgb)
    step(.02,'量化颜色');cv2.setRNGSeed(14)
    k_det = 32 if smooth_lines else 48
    if fmaps is not None and (fmaps.summary['delicate_area'] > .005 or fmaps.summary['detail_ratio'] > .22):
        k_det = max(k_det, 40)
    quant,_=posterize(rgb,k=bands,vote_sigma=1.,blur_sigma=0.,speck_area=8,
                      speck_keep_de=.04,
                      k_detail=k_det,
                      k_pastel=max(6,bands//12) if gradient_mode=='clean' else 0,thin_thr=.015,
                      detail_mask=(fmaps.delicate_detail > .32) if fmaps is not None else None,
                      k_feat=24,
                      edge_boost=(cnn_lines | delicate_pin) if cnn_lines is not None else None)
    roi_protect = np.zeros(source.shape[:2], bool)
    roi_changed = np.zeros(source.shape[:2], bool)
    if fmaps is not None and fmaps.delicate_detail.max() >= .38:
        from .eyes import rescue_delicate_rois
        q_res, roi_protect = rescue_delicate_rois(quant, source, fmaps.delicate_detail, k=12, thr=.16, line_evidence=line_evidence)
        roi_changed = (q_res != quant).any(axis=2)
        quant = q_res
    unified_strokes = np.zeros(source.shape[:2], bool)
    stroke_band = np.zeros(source.shape[:2], bool)
    if cnn_lines is not None and pin_all.any():
        # Consolidate fragmented 1-2px AA gaps along thin heatmap ridges (`heat_repaired` / `heat_bridged`)
        # before simplify_quantized & segment so diagonal hair strands stay 4-connected and continuous.
        q_L = srgb_to_oklab(quant)[..., 0]
        k3_u = np.ones((3, 3), np.uint8)
        k7_u = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        solid_fill = cv2.erode(((q_L < .50) & (~cnn_lines)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        solid_guard = cv2.dilate(solid_fill.astype(np.uint8), k3_u) > 0
        sgz_loc0 = safe_grad_zone if 'safe_grad_zone' in locals() else np.zeros(source.shape[:2], bool)
        if heat_repaired is not None:
            hr_max3 = cv2.dilate(heat_repaired, k3_u)
            src_span7 = cv2.dilate(L_raw_src, k7_u) - cv2.erode(L_raw_src, k7_u)
            hb_loc = heat_bridged if heat_bridged is not None else np.zeros(source.shape[:2], bool)
            is_heat_ridge = (((heat_repaired >= 0.36) & (heat_repaired >= hr_max3 - 0.08)) | hb_loc) & (~sgz_loc0) & (~roi_protect) & (~solid_guard) & (src_span7 >= 0.04)
            is_core_dark = is_heat_ridge & (q_L < cv2.dilate(q_L, k7_u) - 0.03)
            if is_core_dark.any():
                dists_hr, lbl_hr = cv2.distanceTransformWithLabels((~is_core_dark).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                ys_c, xs_c = np.where(is_core_dark)
                gap_hr = is_heat_ridge & (~is_core_dark) & (dists_hr <= 9.0)
                if gap_hr.any():
                    gy_g, gx_g = np.where(gap_hr)
                    ni_hr = lbl_hr[gy_g, gx_g] - 1
                    ny_g, nx_g = ys_c[ni_hr], xs_c[ni_hr]
                    ok_hr = q_L[ny_g, nx_g] < q_L[gy_g, gx_g] - 0.02
                    quant[gy_g[ok_hr], gx_g[ok_hr]] = quant[ny_g[ok_hr], nx_g[ok_hr]]
                    q_L = srgb_to_oklab(quant)[..., 0]
            stroke_band = is_heat_ridge
            d1_s = stroke_band[:-1, :-1] & stroke_band[1:, 1:] & (~stroke_band[:-1, 1:]) & (~stroke_band[1:, :-1])
            if d1_s.any():
                ys_d, xs_d = np.where(d1_s)
                pt_d = heat_repaired[ys_d, xs_d + 1] >= heat_repaired[ys_d + 1, xs_d]
                by_d = np.where(pt_d, ys_d, ys_d + 1); bx_d = np.where(pt_d, xs_d + 1, xs_d)
                quant[by_d, bx_d] = quant[ys_d, xs_d]; stroke_band[by_d, bx_d] = True
            d2_s = stroke_band[:-1, 1:] & stroke_band[1:, :-1] & (~stroke_band[:-1, :-1]) & (~stroke_band[1:, 1:])
            if d2_s.any():
                ys_d, xs_d = np.where(d2_s)
                pt_d = heat_repaired[ys_d, xs_d] >= heat_repaired[ys_d + 1, xs_d + 1]
                by_d = np.where(pt_d, ys_d, ys_d + 1); bx_d = np.where(pt_d, xs_d, xs_d + 1)
                quant[by_d, bx_d] = quant[ys_d, xs_d + 1]; stroke_band[by_d, bx_d] = True
            unified_strokes |= stroke_band

        loc_L = cv2.dilate(q_L, k7_u)
        loc_min_L = cv2.erode(q_L, k7_u)
        is_wisp_or_dark = ((loc_L - q_L >= .08) | (q_L - loc_min_L >= .08) | (eff_strand > .42))
        thin_pin = (cnn_lines & (~roi_protect) & is_wisp_or_dark & (q_L < .96)) | (
            delicate_pin & (loc_L - q_L >= .10) & (q_L < .74)
        )
        thin_pin &= ~solid_guard
        n_dc, d_lbl, d_st, _ = cv2.connectedComponentsWithStats(thin_pin.astype(np.uint8), connectivity=8)
        for idx in range(1, n_dc):
            ar = int(d_st[idx, cv2.CC_STAT_AREA])
            if 4 <= ar <= 600:
                x, y, bw, bh = [int(d_st[idx, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
                sub_m = (d_lbl[y:y + bh, x:x + bw] == idx)
                sub_q = quant[y:y + bh, x:x + bw]
                sub_l = q_L[y:y + bh, x:x + bw][sub_m]
                if float(sub_l.max() - sub_l.min()) <= 0.34:
                    px = sub_q[sub_m]
                    pct = 0.20 if float(np.median(sub_l)) < 0.72 else 0.32
                    sub_q[sub_m] = px[np.argsort(sub_l)[max(0, int(pct * (len(sub_l) - 1)))]]
                    unified_strokes[y:y + bh, x:x + bw] |= sub_m
    palette=np.unique(quant.reshape(-1,3),axis=0)
    prot_m = ((fmaps.delicate_detail > .38) | roi_changed | unified_strokes) if fmaps is not None else None
    wash_comb = None
    if fmaps is not None:
        surf_wash = (fmaps.surface_probs[1] + 0.85 * fmaps.surface_probs[2]) * (1.0 - np.maximum(eff_ink_g if 'eff_ink_g' in locals() else fmaps.ink_main_geom, fmaps.delicate_detail))
        wash_comb = np.maximum(fmaps.mergeable_wash, surf_wash)
        # Never force-boost `wash_comb` across genuine multi-band gradient cores (that would cause
        # region_merge `in_wash` to swallow legitimate gradient color steps like hair/skin shading);
        # keep cel_lock guarded and suppress wash inside line_shield.
        if 'cel_lock' in locals():
            wash_comb = np.where(cel_lock, np.minimum(wash_comb, 0.25), wash_comb)
        if 'line_shield' in locals():
            wash_comb = np.where(line_shield, wash_comb * 0.25, wash_comb)
    if gradient_mode=='clean':
        from .region_cleanup import simplify_quantized
        step(.15,'合并弱色差与渐变碎区')
        wash_m = (wash_comb > .42) if wash_comb is not None else None
        quant,_=simplify_quantized(rgb,quant,line_evidence=line_evidence,evidence_rgb=source,smooth_lines=smooth_lines,protect_mask=prot_m,wash_mask=wash_m)
    step(.25,'构建闭合色区')
    regions,hexcolors=segment(quant,precision=8,difference=1,speckle=2,representative='mode')
    colors=np.array([list(bytes.fromhex(c[1:])) for c in hexcolors],np.uint8)
    if gradient_mode=='layered':
        regions,colors,_=restore_interior_details(source,regions,colors,palette)
    base=colors[regions]
    restored_strands = np.zeros(source.shape[:2], bool)
    if gradient_mode=='clean' and cnn_lines is not None:
        b_lab_pre = srgb_to_oklab(base).astype(np.float32)
        q_lab_pre = srgb_to_oklab(quant).astype(np.float32)
        bL_pre, qL_pre = b_lab_pre[..., 0], q_lab_pre[..., 0]
        lost_sb = stroke_band & ((bL_pre - qL_pre >= 0.025) | (qL_pre - bL_pre >= 0.045))
        if lost_sb.any():
            base[lost_sb] = quant[lost_sb]
            # Unify <6px multi-shade specks strictly on restored `lost_sb` pixels along `stroke_band`
            for _ in range(3):
                sb_L = srgb_to_oklab(base)[..., 0].astype(np.float32)
                packed_sb = (base[..., 0].astype(np.int32) << 16) | (base[..., 1].astype(np.int32) << 8) | base[..., 2].astype(np.int32)
                lbl_sb = label(packed_sb + 1, connectivity=1) - 1
                ar_sb = np.bincount(lbl_sb.ravel())
                tiny_sb = lost_sb & (ar_sb[lbl_sb] < 6)
                if not tiny_sb.any():
                    break
                good_sb = stroke_band & (~tiny_sb) & (sb_L < 0.78)
                if not good_sb.any():
                    break
                dt_sb, lbl_dt = cv2.distanceTransformWithLabels((~good_sb).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
                gy_s, gx_s = np.where(tiny_sb & (dt_sb <= 5.0))
                if not len(gy_s):
                    break
                ys_g, xs_g = np.where(good_sb)
                ni_sb = lbl_dt[gy_s, gx_s] - 1
                ty_sb, tx_sb = ys_g[ni_sb], xs_g[ni_sb]
                ok_sb = sb_L[ty_sb, tx_sb] <= sb_L[gy_s, gx_s] + 0.03
                if not ok_sb.any():
                    break
                base[gy_s[ok_sb], gx_s[ok_sb]] = base[ty_sb[ok_sb], tx_sb[ok_sb]]
        stroke_band = lost_sb | (stroke_band & (srgb_to_oklab(base)[..., 0] < 0.76))
        if stroke_band.any():
            stroke_band_cols = base[stroke_band].copy()
            stroke_band_L = srgb_to_oklab(base)[..., 0][stroke_band].astype(np.float32)
            def _restore_lost_sb():
                cur_L_sb = srgb_to_oklab(base)[..., 0][stroke_band]
                lost_idx = np.abs(cur_L_sb - stroke_band_L) > 0.035
                if lost_idx.any():
                    ys_sb, xs_sb = np.where(stroke_band)
                    base[ys_sb[lost_idx], xs_sb[lost_idx]] = stroke_band_cols[lost_idx]
        else:
            stroke_band_cols = None
            def _restore_lost_sb():
                return None
        # Restore thin 1px diagonal CNN strands & bright-on-dark wisps pruned by vtracer(speckle=2)
        # while strictly rejecting anti-alias gray halos straddling intact black/white edges.
        b_lab = srgb_to_oklab(base).astype(np.float32)
        q_lab = srgb_to_oklab(quant).astype(np.float32)
        bL, qL = b_lab[..., 0], q_lab[..., 0]
        k3 = np.ones((3, 3), np.uint8)
        bL_max3 = cv2.dilate(bL, k3)
        bL_min3 = cv2.erode(bL, k3)
        bL_med5 = cv2.medianBlur(np.rint(np.clip(bL, 0.0, 1.0) * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
        # Strict extrema-anchored filament restore: recovers erased faint dark/bright 1px strands
        # cleanly along neighbor gaps (+0.14 margin) while blocking AA edge-shoulder confetti.
        not_aa_step = ~((bL_min3 <= qL - 0.09) & (bL_max3 >= qL + 0.09))
        bright_wiped = (qL - bL >= 0.042) & (qL >= bL_med5 + 0.04) & (qL >= bL_max3 - 0.14)
        dark_wiped = (bL - qL >= 0.042) & (qL <= bL_med5 - 0.04) & (qL <= bL_min3 + 0.14)
        sgz_loc = safe_grad_zone if 'safe_grad_zone' in locals() else np.zeros(source.shape[:2], bool)
        cand_mask = (unified_strokes | (cnn_lines & (~sgz_loc))) & not_aa_step & (bright_wiped | dark_wiped)
        if cand_mask.any():
            n_mr, lbl_mr, st_mr, _ = cv2.connectedComponentsWithStats(cand_mask.astype(np.uint8), connectivity=8)
            for ri in range(1, n_mr):
                ar_r = int(st_mr[ri, cv2.CC_STAT_AREA])
                x_r, y_r, bw_r, bh_r = [int(st_mr[ri, f]) for f in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
                sub = (lbl_mr[y_r:y_r + bh_r, x_r:x_r + bw_r] == ri)
                if ar_r >= 4 or unified_strokes[y_r:y_r + bh_r, x_r:x_r + bw_r][sub].any():
                    restored_strands[y_r:y_r + bh_r, x_r:x_r + bw_r][sub] = True
                    # Snap each restored strand component to ONE representative quant color to avoid confetti fragments
                    ql_s = qL[y_r:y_r + bh_r, x_r:x_r + bw_r][sub]
                    bl_s = bL[y_r:y_r + bh_r, x_r:x_r + bw_r][sub]
                    px_s = quant[y_r:y_r + bh_r, x_r:x_r + bw_r][sub]
                    pct = 0.72 if float(np.median(ql_s)) > float(np.median(bl_s)) else 0.28
                    col = px_s[np.argsort(ql_s)[min(len(ql_s) - 1, max(0, int(pct * (len(ql_s) - 1))))]]
                    base[y_r:y_r + bh_r, x_r:x_r + bw_r][sub] = col
            if restored_strands.any():
                # Bridge 1px 8-connected diagonal corners into 4-connected filaments using the snapped color
                d1 = restored_strands[:-1, :-1] & restored_strands[1:, 1:] & (~restored_strands[:-1, 1:]) & (~restored_strands[1:, :-1])
                if d1.any():
                    ys_b, xs_b = np.where(d1)
                    pt = line_evidence[ys_b, xs_b + 1] >= line_evidence[ys_b + 1, xs_b]
                    by = np.where(pt, ys_b, ys_b + 1); bx = np.where(pt, xs_b + 1, xs_b)
                    base[by, bx] = base[ys_b, xs_b]
                    restored_strands[by, bx] = True
                d2 = restored_strands[:-1, 1:] & restored_strands[1:, :-1] & (~restored_strands[:-1, :-1]) & (~restored_strands[1:, 1:])
                if d2.any():
                    ys_b, xs_b = np.where(d2)
                    pt = line_evidence[ys_b, xs_b] >= line_evidence[ys_b + 1, xs_b + 1]
                    by = np.where(pt, ys_b, ys_b + 1); bx = np.where(pt, xs_b, xs_b + 1)
                    base[by, bx] = base[ys_b, xs_b + 1]
                    restored_strands[by, bx] = True
    if gradient_mode=='clean':
        from .detail_protection import restore_thin_details
        step(.35,'恢复高对比细节')
        base,_=restore_thin_details(source,base,palette)
        from .outline_export import indexed,remove_weak_islands
        from .outline_stages import select_outline
        step(.4,'选定线稿与填色共用的色区')
        stage, reason, removed = 0, '', 0
        if refit_outline:
            regions,colors=indexed(base)
            base,stage,reason=select_outline(regions,colors,source,for_sketch=False)
            regions,colors=indexed(base)
            regions,colors,removed=remove_weak_islands(regions,colors,source,consolidate_dark=True)
            base=colors[regions]
            _restore_lost_sb()
    if gradient_mode=='clean' or smooth_lines:
        from .ink_stitch import stitch_and_smooth_ink, clean_ink_borders
        step(.45,'缝合墨线断点与平滑线宽')
        base=stitch_and_smooth_ink(base,source,line_evidence=line_evidence)
        pin_prot = (pin_all | restored_strands | stroke_band) if cnn_lines is not None else None
        base=clean_ink_borders(base,source,line_evidence=line_evidence,protect_mask=pin_prot)
        if cnn_lines is not None:
            _restore_lost_sb()
    if smooth_lines:
        wash_global = float(fmaps.summary.get('wash_area', 0.0)) if (fmaps is not None and getattr(fmaps, 'summary', None)) else 1.0
        wall_merge = (wall_mask | restored_strands | stroke_band) if wall_mask is not None else None
        if gradient_mode != 'clean' or wash_global >= 0.16:
            from .region_merge import merge_transition_patches
            step(.47,'合并过渡色碎块为整片色区')
            merge_stats={}
            prot_merge = np.maximum(fmaps.delicate_detail, np.maximum(roi_protect.astype(np.float32) * .70, (restored_strands | stroke_band).astype(np.float32) * .95)) if fmaps is not None else None
            strand_prot = np.where(~safe_grad_zone, np.maximum(heat_repaired, np.maximum(eff_strand, stroke_band.astype(np.float32))), 0.0) if (fmaps is not None and heat_repaired is not None and 'safe_grad_zone' in locals()) else (np.maximum(fmaps.fine_strand, heat_repaired) if (fmaps is not None and heat_repaired is not None) else None)
            base=merge_transition_patches(
                base,source,stats=merge_stats,
                protect_map=prot_merge,
                strand_map=strand_prot,
                wash_map=wash_comb,
                cel_map=fmaps.surface_probs[0] if fmaps is not None else None,
                wall_mask=wall_merge,
                area_scale=max(1.0, (h * w) / float(h_orig * w_orig)),
            )
            print('[region_merge]',merge_stats,flush=True)
            if cnn_lines is not None:
                _restore_lost_sb()
        if gradient_mode == 'clean':
            from .gradient_topology import enforce_global_gradient_topology
            step(.50,'基于CNN热力图执行拓扑净化与梯度重构')
            base = enforce_global_gradient_topology(base, source, fmaps=fmaps, protect_mask=(roi_protect | restored_strands | stroke_band), wall_mask=wall_merge)
            if cnn_lines is not None:
                _restore_lost_sb()
    scale = 2 if (h > h_orig or w * h > 2000000) else 4
    if recover_strands and gradient_mode=='layered':
        step(.4,'提取细线');branches=extract_branches(source,quant,base,palette)
        step(.55,'验证线条连接');pairs=supported_pairs(source,base,palette,branches)
        work=rebuild_partition(base,source,branches,pairs,scale)
    else:work=cv2.resize(base,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    regions=label(work.astype(np.int32)@np.array([65536,256,1])+1,connectivity=1)-1
    colors=np.empty((regions.max()+1,3),np.uint8);colors[regions.ravel()]=work.reshape(-1,3)
    palette=np.unique(np.vstack([palette,colors]),axis=0)
    gray_palette=snap_gray(palette,False)
    assert set(map(tuple,colors))<=set(map(tuple,palette))|set(map(tuple,gray_palette))
    step(.7,'整理细碎色区')
    regions,colors=compact_fragments(regions,snap_gray(colors,False),scale)
    assert set(map(tuple,colors))<=set(map(tuple,gray_palette))
    step(.8,'拟合共享边界')
    fitting={}
    metadata=None
    corridor=None
    if refit_outline and gradient_mode=='clean':
        from functools import partial
        from .structural_curves import fit_compact_outline,refit_sketch_chain,smooth_outline_junctions
        fitting=dict(structural_fitter=partial(fit_compact_outline,smooth_lines=smooth_lines),
                     secondary_fitter=partial(refit_sketch_chain,smooth_lines=smooth_lines),
                     secondary_network=partial(smooth_outline_junctions,smooth_lines=smooth_lines))
        if locate_boundaries:
            from .boundary_corridor import SourceCorridor
            corridor=SourceCorridor(source,colors)
            if fmaps is not None and getattr(fmaps, 'elem_grad_maps', None) is not None:
                sgz_g = safe_grad_zone if 'safe_grad_zone' in locals() else np.zeros((h, w), bool)
                hr_g = heat_repaired if heat_repaired is not None else np.zeros((h, w), np.float32)
                def _grad_guarded_corridor(pts, right, left, sc):
                    if min(right, left) >= 0 and len(pts) >= 4:
                        ys_c = np.clip(np.round(pts[:, 1] / sc).astype(int), 0, h - 1)
                        xs_c = np.clip(np.round(pts[:, 0] / sc).astype(int), 0, w - 1)
                        # Only bypass SourceCorridor when >=75% of the chain lies strictly inside line-free safe_grad_zone
                        if float(np.mean(sgz_g[ys_c, xs_c])) > 0.75 and float(np.max(hr_g[ys_c, xs_c])) < 0.24:
                            return pts
                    return corridor(pts, right, left, sc)
                fitting.update(chain_locator=_grad_guarded_corridor,source_scale=scale)
            else:
                fitting.update(chain_locator=corridor,source_scale=scale)
        metadata={'data-stage':str(stage),'data-stop-reason':reason,'data-weak-islands-removed':str(removed)}
    # Boundary samples use the enlarged grid; keep smoothing in source-pixel units.
    tol = (.32 if smooth_lines else .25) * scale
    smoothing = (.85 if smooth_lines else .65) * scale
    chain_scale_fn = None
    if fmaps is not None and smooth_lines:
        def chain_scale_fn(pts, base_scale, s_scale):
            ys = np.clip(np.round(pts[:, 1] / s_scale).astype(int), 0, h - 1)
            xs = np.clip(np.round(pts[:, 0] / s_scale).astype(int), 0, w - 1)
            d_del = float(np.mean(fmaps.delicate_detail[ys, xs]))
            if d_del > .35:
                return base_scale * .85
            d_wash = float(np.mean(wash_comb[ys, xs]))
            if d_wash > .48:
                return base_scale * 1.45
            d_ink = float(np.mean(fmaps.ink_main_geom[ys, xs]))
            if d_ink > .38:
                return base_scale * 1.15
            return base_scale
    svg,stats=trace_regions(regions,['#%02x%02x%02x'%tuple(c) for c in colors],tol=tol,seam=0,geometry_guard=True,smoothing=smoothing,
                            structural_scale=scale if (gradient_mode=='clean' or smooth_lines) else None,compact_commands=True,
                            chain_scale_fn=chain_scale_fn,**fitting)
    svg=re.sub(r'width="\d+" height="\d+"',f'width="{w_orig}" height="{h_orig}"',svg,count=1)
    if outline_layer:
        step(.96,'生成教师SVG结构（先加载线稿边界层 + 再填充线稿约束色块层）')
        line_region_ids = None
        if gradient_mode == 'clean':
            from .color import srgb_to_oklab
            c_lab = srgb_to_oklab(colors).astype(np.float32)
            r_areas = np.bincount(regions.ravel(), minlength=len(colors))
            r_boxes = find_objects(regions + 1)
            r_spans = np.array([max(b[0].stop - b[0].start, b[1].stop - b[1].start) if b is not None else 0 for b in r_boxes])
            is_line_reg = (c_lab[:, 0] < 0.36) & (r_areas <= 4.5 * scale * r_spans)
            is_line_reg |= (c_lab[:, 0] < 0.605) & (r_spans >= 4 * scale) & (r_areas <= 2.4 * scale * r_spans)
            if fmaps is not None:
                cnn_l = np.maximum(fmaps.ink_main_geom, fmaps.fine_strand)
                if getattr(fmaps, "line_vector_fields", None) is not None:
                    vf0 = fmaps.line_vector_fields[0]
                    vf1 = fmaps.line_vector_fields[1]
                    if vf0.shape != (h, w):
                        vf0 = cv2.resize(vf0, (w, h), interpolation=cv2.INTER_AREA)
                        vf1 = cv2.resize(vf1, (w, h), interpolation=cv2.INTER_AREA)
                    cnn_l = np.maximum(cnn_l, np.maximum(vf0, vf1 * 0.8))
                elif getattr(fmaps, "line_net_maps", None) is not None:
                    lm = fmaps.line_net_maps
                    cnn_l = np.maximum(cnn_l, np.maximum(lm[0], np.maximum(lm[1], lm[2] * 0.7)))
                cnn_l_up = cv2.resize(cnn_l, (regions.shape[1], regions.shape[0]), interpolation=cv2.INTER_NEAREST)
                r_cnn = np.bincount(regions.ravel(), weights=cnn_l_up.ravel(), minlength=len(colors)) / np.maximum(r_areas, 1)
                is_line_reg |= (c_lab[:, 0] < 0.605) & (r_cnn > 0.42) & (r_areas <= 2.8 * scale * r_spans)
            line_region_ids = set(np.flatnonzero(is_line_reg).tolist())
            if metadata is not None:
                metadata['data-line-art-paths'] = str(len(line_region_ids))
                metadata['data-color-fill-paths'] = str(len(colors) - len(line_region_ids))
        if metadata is not None:
            metadata['data-segments-before-refit']=str(stats['segments_before_secondary'])
            metadata['data-segments-after-refit']=str(stats['curve_segments'])
            if corridor is not None:
                metadata['data-source-chains-considered']=str(corridor.considered)
                metadata['data-source-chains-moved']=str(corridor.moved)
                metadata['data-source-tonal-chains']=str(corridor.tonal_chains)
        svg=boundary_then_fill(svg,hidden_strokes=(),
                               metadata=metadata,
                               line_region_ids=line_region_ids)
    preview=Image.fromarray(colors[regions]).resize((w_orig,h_orig),Image.Resampling.LANCZOS)
    step(1.,'完成');return svg,preview
