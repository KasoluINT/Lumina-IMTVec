"""Consolidate weak shading distinctions before constructing vector boundaries."""
import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from skimage.measure import label
from .color import srgb_to_oklab
from .bands import soft_vote


def _structure_coherence(evidence):
    """Multi-channel OKLab structure tensor coherence (high on lines, low on grain)."""
    jxx = jxy = jyy = 0
    for c, w in ((0, 1.0), (1, 1.8), (2, 1.8)):
        gx = cv2.Sobel(evidence[..., c], cv2.CV_32F, 1, 0, ksize=3) * w
        gy = cv2.Sobel(evidence[..., c], cv2.CV_32F, 0, 1, ksize=3) * w
        jxx = jxx + gx * gx
        jxy = jxy + gx * gy
        jyy = jyy + gy * gy
    jxx = cv2.GaussianBlur(jxx, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(jxy, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(jyy, (0, 0), 2.0)
    return np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-6)


def _coherent_binary_mask(raw_mask, coh, min_span=8, min_coh=.45):
    """Filter out small isotropic grain blobs while keeping elongated/coherent strokes."""
    raw = raw_mask.astype(np.uint8)
    num, comp, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if num <= 1:
        return raw.astype(bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    spans = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    mean_coh = np.bincount(comp.ravel(), weights=coh.ravel(), minlength=num) / np.maximum(areas, 1)
    keep = (spans >= min_span) | ((areas >= 4) & (mean_coh >= min_coh))
    keep[0] = False
    return keep[comp]


def simplify_quantized(rgb, quantized, color_distance=.045, shading_radius=6., speck_area=48, line_evidence=None, *, evidence_rgb=None, smooth_lines=False, protect_mask=None, wash_mask=None):
    palette, inverse, counts = np.unique(quantized.reshape(-1, 3), axis=0,
                                        return_inverse=True, return_counts=True)
    field = inverse.reshape(quantized.shape[:2])
    lab = srgb_to_oklab(palette.reshape(1, -1, 3)).reshape(-1, 3)
    source = srgb_to_oklab(rgb).astype(np.float32)
    evidence_rgb = rgb if evidence_rgb is None else evidence_rgb
    evidence = source if evidence_rgb is rgb else srgb_to_oklab(evidence_rgb).astype(np.float32)
    grain_l = float(np.median(np.abs(cv2.Laplacian(evidence[..., 0], cv2.CV_32F, ksize=3))))
    clean_art = grain_l < .004
    eff_color_dist = min(color_distance, .016) if clean_art else color_distance
    eff_shade_r = min(shading_radius, 1.8) if clean_art else shading_radius
    grad_thr = .12 if clean_art else .22
    weak_cap = 80 if clean_art else 1200
    weak_de = .028 if clean_art else .075

    coh = _structure_coherence(evidence)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ridge = np.maximum(cv2.morphologyEx(evidence[..., 0], cv2.MORPH_BLACKHAT, kernel),
                       cv2.morphologyEx(evidence[..., 0], cv2.MORPH_TOPHAT, kernel))
    ridge_a = np.maximum(cv2.morphologyEx(evidence[..., 1], cv2.MORPH_BLACKHAT, kernel),
                         cv2.morphologyEx(evidence[..., 1], cv2.MORPH_TOPHAT, kernel))
    ridge_b = np.maximum(cv2.morphologyEx(evidence[..., 2], cv2.MORPH_BLACKHAT, kernel),
                         cv2.morphologyEx(evidence[..., 2], cv2.MORPH_TOPHAT, kernel))
    ridge_c = ridge + 2.0 * np.hypot(ridge_a, ridge_b)
    faint_arc = _coherent_binary_mask(ridge_c > (.028 if clean_art else .038), coh, min_span=6 if clean_art else 9, min_coh=.45)
    line_mask = faint_arc
    if line_evidence is not None:
        if line_evidence.shape!=field.shape:raise ValueError('Line evidence dimensions must match the source')
        line_mask = line_mask | _coherent_binary_mask(line_evidence > .35, coh, min_span=6 if clean_art else 8, min_coh=.45)
    if len(palette) > 1:
        lab_w = lab * np.array([1.0, 1.8, 1.8], np.float32)
        groups = fcluster(linkage(pdist(lab_w), method='complete'), eff_color_dist, criterion='distance')
        # A common palette family is not a common semantic region. Choose the
        # representative separately for each connected family from its local source.
        family_field=groups[field]
        if line_mask is not None:
            from .source_lines import source_line_evidence
            _,polarity=source_line_evidence(evidence_rgb,return_polarity=True)
            # Separate dark cores, bright cores and background before choosing
            # local representatives; protecting them after consolidation is too late.
            family_field=family_field*3+np.where(line_mask,polarity,0)
        components = label(family_field, connectivity=1) - 1
        area = np.bincount(components.ravel())
        means = np.stack([np.bincount(components.ravel(), weights=source[..., c].ravel()) / area
                          for c in range(3)], axis=1)
        family = np.empty(len(area),np.int32);family[components.ravel()] = groups[field].ravel()
        representative = np.empty(len(area),np.int32)
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            ids = np.flatnonzero(family == group)
            distance = np.sum((means[ids,None,:]-lab[members][None,:,:])**2,axis=2)
            representative[ids] = members[np.argmin(distance,axis=1)]
        field = representative[components]
    # Protect narrow luminance/chromatic ridges and strong colour edges, including isoluminant ones.
    smooth_ev = cv2.GaussianBlur(evidence, (0, 0), .9)
    gradient = np.zeros(field.shape, np.float32)
    for channel in range(3):
        gx = cv2.Sobel(smooth_ev[..., channel], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(smooth_ev[..., channel], cv2.CV_32F, 0, 1)
        gradient += gx*gx + gy*gy
    protected = cv2.dilate(((ridge > (.030 if clean_art else .050)) | (gradient > grad_thr**2)).astype(np.uint8),
                           np.ones((3, 3), np.uint8)).astype(bool)
    if line_mask is not None:
        protected |= cv2.dilate(line_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if protect_mask is not None:
        protected |= protect_mask
    # Vote for whole existing colours, never blur RGB into extra printing colours.
    voted = soft_vote(field, eff_shade_r)
    field = np.where(protected, field, voted)
    for _ in range(2):
        regions = label(field + 1, connectivity=1) - 1
        area = np.bincount(regions.ravel())
        own = np.empty(len(area), np.int32); own[regions.ravel()] = field.ravel()
        protected_ratio = np.bincount(regions.ravel(), weights=protected.ravel()) / area
        feat_prot_ratio = np.zeros(len(area)) if protect_mask is None else np.bincount(regions.ravel(), weights=protect_mask.ravel()) / area
        wash_ratio = np.zeros(len(area)) if wash_mask is None else np.bincount(regions.ravel(), weights=wash_mask.ravel()) / area
        line_ratio = np.zeros(len(area)) if line_mask is None else np.bincount(regions.ravel(),weights=line_mask.ravel())/area
        means = np.stack([np.bincount(regions.ravel(), weights=source[..., c].ravel()) / area
                          for c in range(3)], axis=1)
        k1, k2 = [], []
        for a, b in [(regions[:-1], regions[1:]), (regions[:, :-1], regions[:, 1:])]:
            mask = a != b
            va, vb = a[mask].astype(np.uint64), b[mask].astype(np.uint64)
            k1.append((va << 32) | vb)
            k2.append((vb << 32) | va)
        bi_k = np.concatenate(k1 + k2)
        if not len(bi_k): break
        uniq_k, length = np.unique(bi_k, return_counts=True)
        a = (uniq_k >> 32).astype(np.int32)
        b = (uniq_k & 0xffffffff).astype(np.int32)
        ab_scale = np.array([1.0, 1.8, 1.8], np.float32)
        distance = np.linalg.norm(lab[own[a]] - lab[own[b]], axis=1)
        dist_w = np.linalg.norm((lab[own[a]] - lab[own[b]]) * ab_scale, axis=1)
        mean_w = np.linalg.norm((means[a] - means[b]) * ab_scale, axis=1)
        small = (area[a] < speck_area) & (dist_w < .055) & (line_ratio[a] < .2) & (feat_prot_ratio[a] < .15)
        # In the clean style, subordinate weak shading islands may disappear;
        # delicate feature zones stay protected, while CNN wash zones may simplify slightly further.
        eff_weak_cap = np.where(wash_ratio[a] > .60, np.maximum(weak_cap, 1500), weak_cap)
        eff_weak_de = np.where(wash_ratio[a] > .60, np.maximum(weak_de, .065), weak_de)
        weak_shading = ((area[a] < np.minimum(eff_weak_cap, .22 * area[b])) & (protected_ratio[a] < .2)
                        & (feat_prot_ratio[a] < .15) & (dist_w < eff_weak_de) & (mean_w < .06))
        valid = (small | weak_shading) & (area[b] > area[a])
        a, b, distance, length = a[valid], b[valid], distance[valid], length[valid]
        order = np.lexsort((-length, distance, a))
        mapping = np.arange(len(area)); seen = set()
        for i in order:
            if a[i] not in seen:
                mapping[a[i]] = b[i]; seen.add(a[i])
        # Follow only merges whose final colour remains close to the starting region.
        for _ in range(32):
            following = mapping[mapping]
            accept = np.linalg.norm(lab[own] - lab[own[following]], axis=1) < .055
            updated = np.where(accept, following, mapping)
            if np.array_equal(updated, mapping): break
            mapping = updated
        field = own[mapping[regions]]
    # Unify connected ink stroke cores so one stroke is not split into alternating near-black/coloured shades.
    if line_mask is not None and len(palette) > 1:
        cores = [(line_mask & (lab[field, 0] < .48), .05)]
        for core_mask, max_de in cores:
            if not core_mask.any():
                continue
            n_cc, comp_cc = cv2.connectedComponents(core_mask.astype(np.uint8), connectivity=8)
            if n_cc <= 1:
                continue
            pix_l = field[core_mask]
            pix_c = comp_cc[core_mask]
            keys, cnts = np.unique((pix_c.astype(np.int64) << 20) | pix_l.astype(np.int64), return_counts=True)
            cc_id = keys >> 20
            order_k = np.lexsort((keys, -cnts, cc_id))
            keys = keys[order_k]; cc_id = cc_id[order_k]
            first_k = np.r_[True, cc_id[1:] != cc_id[:-1]]
            mode_lbl = np.zeros(n_cc, np.int32)
            mode_lbl[cc_id[first_k]] = (keys[first_k] & 0xfffff).astype(np.int32)
            target = mode_lbl[comp_cc]
            close = np.linalg.norm(lab[field] - lab[target], axis=2) <= max_de
            field = np.where(core_mask & close, target, field)
    result = palette[field]
    return result, dict(colors_before=len(palette), colors_after=len(np.unique(field)),
                        protected_fraction=float(protected.mean()),
                        source_line_constraints=line_evidence is not None,
                        changed_fraction=float(np.any(result != quantized, axis=2).mean()))
