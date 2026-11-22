"""Stained-glass simplification: absorb small / low-contrast transition patches into a
larger neighbour so each visual shape is one flat colour, while keeping ink strokes and
high-contrast details (eye highlights, white pattern veins) as their own regions."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

from .color import srgb_to_oklab


def _index(rgb):
    keys = rgb.astype(np.int32) @ np.array([65536, 256, 1])
    labels = label(keys + 1, connectivity=1) - 1
    colors = np.empty((labels.max() + 1, 3), np.uint8)
    colors[labels.ravel()] = rgb.reshape(-1, 3)
    return labels, colors


def _adjacency(labels, n=None, wall_mask=None):
    k1, k2 = [], []
    wall_flags = []
    for a, b, w_edge in (
        (
            labels[:-1],
            labels[1:],
            (wall_mask[:-1] | wall_mask[1:]) if wall_mask is not None else None,
        ),
        (
            labels[:, :-1],
            labels[:, 1:],
            (wall_mask[:, :-1] | wall_mask[:, 1:]) if wall_mask is not None else None,
        ),
    ):
        m = a != b
        va = a[m].astype(np.uint64)
        vb = b[m].astype(np.uint64)
        k1.append((va << 32) | vb)
        k2.append((vb << 32) | va)
        if w_edge is not None:
            wall_flags.append(w_edge[m].astype(np.int32))
    bi_k = np.concatenate(k1 + k2)
    if wall_mask is not None:
        wf = np.concatenate(wall_flags)
        bi_w = np.concatenate([wf, wf])
        uniq_k, inv, contact = np.unique(bi_k, return_inverse=True, return_counts=True)
        wall_contact = np.bincount(inv, weights=bi_w, minlength=len(uniq_k))
        return (uniq_k >> 32).astype(np.int32), (uniq_k & 0xffffffff).astype(np.int32), contact, wall_contact
    uniq_k, contact = np.unique(bi_k, return_counts=True)
    return (uniq_k >> 32).astype(np.int32), (uniq_k & 0xffffffff).astype(np.int32), contact, np.zeros(len(uniq_k), dtype=np.float64)


def merge_transition_patches(
    base,
    source,
    *,
    passes=4,
    stats=None,
    protect_map=None,
    strand_map=None,
    wash_map=None,
    cel_map=None,
    wall_mask=None,
    area_scale=1.0,
):
    """Return a simplified copy of `base` (H x W x 3 uint8 partition)."""
    src_lab = srgb_to_oklab(source)
    out = base.copy()
    history = []
    a_sc = float(max(0.25, area_scale))
    l_sc = float(np.sqrt(a_sc))
    for _ in range(passes):
        labels, colors = _index(out)
        n = len(colors)
        area = np.bincount(labels.ravel(), minlength=n)
        lab = srgb_to_oklab(colors)
        # Thickness: distance from the region boundary, max per region.
        edge = np.zeros(labels.shape, bool)
        edge[1:] |= labels[1:] != labels[:-1]; edge[:-1] |= labels[1:] != labels[:-1]
        edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]; edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
        thick = ndi.maximum(ndi.distance_transform_edt(~edge), labels, np.arange(n)) + 1
        pix_err = np.linalg.norm(src_lab - lab[labels], axis=-1)
        src_err = np.bincount(labels.ravel(), weights=pix_err.ravel(), minlength=n) / np.maximum(area, 1)
        feat_prot = np.zeros(n) if protect_map is None else np.bincount(labels.ravel(), weights=protect_map.ravel(), minlength=n) / np.maximum(area, 1)
        feat_strand = np.zeros(n) if strand_map is None else np.bincount(labels.ravel(), weights=strand_map.ravel(), minlength=n) / np.maximum(area, 1)
        feat_wash = np.zeros(n) if wash_map is None else np.bincount(labels.ravel(), weights=wash_map.ravel(), minlength=n) / np.maximum(area, 1)
        feat_cel = np.zeros(n) if cel_map is None else np.bincount(labels.ravel(), weights=cel_map.ravel(), minlength=n) / np.maximum(area, 1)

        a, b, contact, wall_contact = _adjacency(labels, n=n, wall_mask=wall_mask)
        de = np.linalg.norm(lab[a] - lab[b], axis=1)
        perim = np.bincount(a, weights=contact, minlength=n)
        share = contact / np.maximum(perim[a], 1)
        wall_cross_frac = wall_contact / np.maximum(contact, 1)
        # Highest contrast & neighbour concentration: enclosed clean features stay.
        max_de = np.zeros(n); np.maximum.at(max_de, a, de)
        min_de = np.full(n, np.inf); np.minimum.at(min_de, a, de)
        max_share = np.zeros(n); np.maximum.at(max_share, a, share)

        boxes = ndi.find_objects(labels + 1)
        spans = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) if s is not None else 0 for s in boxes])
        # Protect dark ink and coloured thin ink outlines, plus elongated thin details & CNN hairline strands.
        ink = ((lab[:, 0] < .30) & (thick <= 3.5 * l_sc)) | ((lab[:, 0] < .50) & (thick <= 3.0 * l_sc) & (spans >= 7 * l_sc) & (max_de >= .08))
        thin_stroke = (thick <= 2.6 * l_sc) & (spans >= 5 * l_sc) & (area <= 3.2 * l_sc * spans) & (max_de >= .040)
        if strand_map is not None:
            # Preserve faint thin CNN-detected hair curves (including light gray/brown on white) without blocking bulky patches
            thin_stroke |= (feat_strand >= 0.36) & (thick <= 2.6 * l_sc) & (spans >= 3.5 * l_sc) & (area <= 3.6 * l_sc * spans) & (max_de >= .032)
        cnn_keep = feat_prot >= .36

        aa, ab = area[a], area[b]
        crossed_wall = (wall_mask is not None) & (aa >= 12 * a_sc) & (wall_cross_frac >= 0.35)
        light_l = lab[a, 0] > .35
        base_valid = (ab > aa) & ~ink[a] & ~thin_stroke[a] & ~cnn_keep[a] & ~crossed_wall
        valid = base_valid & ~(ink[b] & light_l)
        # Size / contrast ladder (CNN wash zones allow wider stained-glass transition simplification).
        in_wash = (feat_wash[a] >= .42) & (feat_cel[a] < .30) & (src_err[a] >= .012)
        crisp_cel = ((feat_cel[a] >= .28) | ((src_err[a] < .018) & (aa >= 180 * a_sc))) & (src_err[a] < .018) & (aa >= 40 * a_sc) & (de >= .032)
        med_de = np.where(in_wash, .078, .055)
        lrg_de = np.where(in_wash, .056, .035)
        tiny = (aa < 30 * a_sc) & ((spans[a] < 5 * l_sc) | (de < .045))
        small = (aa < 220 * a_sc) & (de < .085) & ~crisp_cel
        medium = (aa < np.where(in_wash, 2000 * a_sc, 900 * a_sc)) & (de < med_de) & (share > .14) & ~crisp_cel
        large = (aa < np.where(in_wash, 5500 * a_sc, 2500 * a_sc)) & (de < lrg_de) & (share > .22) & ~crisp_cel
        # Keep accents and genuine cel-shaded islands (only protect non-cel islands if inside CNN delicate zone).
        accent = (min_de[a] > .11) & (aa >= 6 * a_sc)
        cel_or_feat = (feat_prot[a] >= .22) | (feat_cel[a] >= .28) | (src_err[a] < .012) if (protect_map is not None or cel_map is not None) else True
        clean_island = (max_share[a] >= .78) & (src_err[a] < .016) & (aa >= 18 * a_sc) & (de >= .022) & cel_or_feat
        # Eliminate stray pinholes inside fills/outlines outside delicate zones and flat cel artwork.
        pinhole = base_valid & (ab >= np.maximum(aa * 4, 80 * a_sc)) & (max_share[a] >= .88) & (aa <= 55 * a_sc) & (feat_prot[a] < .25) & (feat_cel[a] < .60)
        ok = pinhole | (valid & (tiny | small | medium | large) & ~accent & ~clean_island)
        if not ok.any():
            break
        a2, b2, de2, sh2 = a[ok], b[ok], de[ok], share[ok]
        # Prefer the closest colour, then the longest shared border.
        order = np.lexsort((-sh2, de2, a2))
        a2, b2 = a2[order], b2[order]
        first = np.r_[True, a2[1:] != a2[:-1]]
        mapping = np.arange(n)
        mapping[a2[first]] = b2[first]
        # Targets are strictly larger, so chains terminate; resolve them.
        for _ in range(64):
            nxt = mapping[mapping]
            if np.array_equal(nxt, mapping):
                break
            mapping = nxt
        merged = int((mapping != np.arange(n)).sum())
        history.append((n, merged))
        out = colors[mapping[labels]]
        if merged < max(20, n * .01):
            break
    if stats is not None:
        stats['merge_passes'] = history
        stats['regions_after'] = int(_index(out)[0].max() + 1)
    return out
