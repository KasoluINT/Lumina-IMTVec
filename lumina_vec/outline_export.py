"""Export the selected partition as closed, guarded linework."""
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_propagation, find_objects

from .color import srgb_to_oklab

from .outline_stages import select_outline
from .shared_regions import trace_regions
from .structural_curves import fit_compact_outline, refit_sketch_chain, smooth_outline_junctions
from .svg_layers import boundary_then_fill

NS = '{http://www.w3.org/2000/svg}'


def indexed(rgb):
    keys = rgb.astype(np.int32) @ np.array([65536, 256, 1])
    regions = label(keys + 1, connectivity=1) - 1
    colors = np.empty((regions.max() + 1, 3), np.uint8)
    colors[regions.ravel()] = rgb.reshape(-1, 3)
    return regions, colors


def _reflection_halo(labels, original, palette, index, target, box):
    """Return the weak outer pixels that can connect to an adjacent shadow."""
    rows, cols = box
    crop = (slice(max(0,rows.start-1),rows.stop+1), slice(max(0,cols.start-1),cols.stop+1))
    field = labels[crop]
    component = field == index
    axis = palette[index] - palette[target]
    axis /= np.linalg.norm(axis)
    signal = (original[crop] - palette[target]) @ axis
    values = signal[component]
    peak = np.quantile(values, .90)
    if peak < .18 or np.quantile(values, .20) > .66 * peak:
        return None
    core = component & (signal >= .68 * peak)
    if core.sum() < 3:
        return None
    weak = component & ~binary_fill_holes(core)
    shadow = field == target
    halo = binary_propagation(binary_dilation(shadow) & weak, mask=weak)
    if label(component & ~halo,connectivity=1).max() > 1:
        # Peel only pixels reachable from the surrounding shadow, retaining
        # the connection between adjacent bright peaks in the same glint.
        halo = np.zeros_like(component)
        rejected = np.zeros_like(component)
        while True:
            candidates = binary_dilation(shadow | halo) & weak & ~halo & ~rejected
            if not candidates.any():
                break
            y,x = np.unravel_index(np.argmin(np.where(candidates,signal,np.inf)), signal.shape)
            halo[y,x] = True
            if label(component & ~halo,connectivity=1).max() > 1:
                halo[y,x] = False
                rejected[y,x] = True
    return (crop, halo, target) if halo.sum() >= 2 else None


def remove_weak_islands(labels, colors, source, *, consolidate_dark=False):
    """Suppress tiny, compact near-colour islands unsupported by the source."""
    sizes = np.bincount(labels.ravel())
    boxes = find_objects(labels+1)
    small = np.array([size<=16 and max(s.stop-s.start for s in box)<=6
                      for size,box in zip(sizes,boxes)])
    palette = srgb_to_oklab(colors)
    dark = np.zeros(len(colors),dtype=bool)
    compact_glint = np.zeros(len(colors),dtype=bool)
    weak_patch = np.zeros(len(colors),dtype=bool)
    if consolidate_dark:
        dims = np.array([[s.stop-s.start for s in box] for box in boxes])
        short,long = dims.min(1),dims.max(1)
        # Keep long thin ink, and open or very sparse shapes, even in shadows.
        dark = ((palette[:,0] < .37)&(sizes<=280)&(long<=38)
                &(long<=3.3*short)&(sizes>=.24*short*long))
        compact_glint = ((sizes>=12)&(sizes<=65)&(long<=13)&(long<=2.6*short)
                         &(sizes>=.25*short*long))
        weak_patch = ((sizes<=48)&(long<=11)&(long<=2.5*short)
                      &(sizes>=.32*short*long))
    ids = np.flatnonzero(small|dark|compact_glint|weak_patch)
    if not len(ids):
        return labels, colors, 0
    original = srgb_to_oklab(source)
    height,width = labels.shape
    replace = np.arange(len(colors))
    halos = []
    for idx in ids:
        ya,xa = boxes[idx]
        y,x = np.where(labels[ya,xa]==idx)
        y,x = y+ya.start,x+xa.start
        neighbors = []
        contrasts = []
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            yi,xi = y+dy,x+dx
            valid = (yi>=0)&(yi<height)&(xi>=0)&(xi<width)
            y0,x0,yi,xi = y[valid],x[valid],yi[valid],xi[valid]
            other = labels[yi,xi]
            edge = other!=idx
            neighbors.extend(other[edge])
            differences = np.linalg.norm(original[y0,x0]-original[yi,xi],axis=1)
            contrasts.extend(differences[edge])
        if len(neighbors)<=2:
            continue
        other,counts = np.unique(neighbors,return_counts=True)
        lead = counts.argmax()
        target = other[lead]
        difference=np.linalg.norm(palette[idx]-palette[target])
        luminance=abs(palette[idx,0]-palette[target,0])
        observed=np.quantile(contrasts,.8)
        q95 = np.quantile(contrasts,.95)
        compact=sizes[target]>=max(25,3*sizes[idx]) and counts[lead]>=.64*len(neighbors)
        if small[idx] and compact and difference<=.071 and observed<=.080 and luminance<=.050:
            replace[idx]=target
            continue
        if consolidate_dark and observed<=.044 and q95<=.065:
            # A nearly enclosed weak island can merge even when its immediate
            # parent is only slightly larger. Stronger edges keep the old rule.
            enclosed = (weak_patch[idx] and counts[lead]>=.94*len(neighbors)
                        and sizes[target]>=1.4*sizes[idx] and difference<=.055)
            shadow = (dark[idx] and compact and sizes[idx]<=96
                      and palette[target,0]<.24 and difference<=.092)
            muted = (weak_patch[idx] and counts[lead]>=.55*len(neighbors)
                     and sizes[target]>=4*sizes[idx] and difference<=.075)
            if enclosed or shadow or muted:
                replace[idx]=target
                continue
        if compact_glint[idx] and observed >= .085:
            shadow_neighbors = ((palette[other,0]<.24) & (sizes[other]>=2.5*sizes[idx])
                                & (palette[idx,0]-palette[other,0]>.20)
                                & (counts>=.10*len(neighbors)))
            proposals = []
            for dark_target in other[shadow_neighbors]:
                halo = _reflection_halo(labels,original,palette,idx,dark_target,boxes[idx])
                if halo is not None:
                    proposals.append(halo)
            if proposals:
                halos.append(max(proposals,key=lambda proposal:proposal[1].sum()))
        # Check the original over all sides so a weak side of a glass facet
        # doesn't erase it when one of its other sides is strongly contrasted.
        if (not dark[idx] or palette[target,0]>=.39 or not compact or difference>.048
                or observed>.060 or q95>.100):
            continue
        replace[idx]=target
    count=int(np.count_nonzero(replace!=np.arange(len(colors))))
    if not count and not halos:
        return labels, colors, count
    # All merges point to a strictly larger region, so resolving parents
    # also removes the inner rings of merged outer shadows.
    while not np.array_equal(replace,replace[replace]):
        replace=replace[replace]
    result = colors[replace[labels]]
    for crop, halo, target in halos:
        result[crop][halo] = colors[replace[target]]
    labels,colors = indexed(result)
    return labels,colors,count


def trace_outline(selected, scale=4, source=None):
    """Fit the chosen linework twice without admitting new boundaries."""
    labels, colors = indexed(selected)
    removed = 0
    if source is not None:
        labels, colors, removed = remove_weak_islands(labels, colors, source)
    grid = cv2.resize(labels, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    sketch, stats = trace_regions(
        grid, ['#%02x%02x%02x' % tuple(c) for c in colors],
        geometry_guard=True, structural_scale=scale,
        structural_fitter=fit_compact_outline, secondary_fitter=refit_sketch_chain,
        secondary_network=smooth_outline_junctions,
    )
    doc = ET.fromstring(boundary_then_fill(sketch))
    outlines = doc.find(f'{NS}g[@id="region-boundaries"]')
    for path in list(outlines):
        if path.get('stroke') == '#7f7f7f':
            outlines.remove(path)
    outlines.set('data-weak-islands-removed', str(removed))
    outlines.set('data-segments-before-refit', str(stats['segments_before_secondary']))
    outlines.set('data-segments-after-refit', str(stats['curve_segments']))
    return outlines


def build_outline(svg, partition, source):
    labels, colors = indexed(partition)
    selected, stage, reason = select_outline(labels, colors, source)
    root = ET.fromstring(svg)
    h, w = selected.shape[:2]
    x, y, vw, vh = map(float, root.get('viewBox').split())
    if x or y or vw / w != vh / h:
        raise ValueError('Outline must share the fill viewBox')
    outlines = trace_outline(selected, int(vw / w), source=source)
    outlines.set('data-stage', str(stage))
    outlines.set('data-stop-reason', reason)
    previous = root.find(f'{NS}g[@id="region-boundaries"]')
    index = list(root).index(previous)
    root.remove(previous)
    root.insert(index, outlines)
    return ET.tostring(root, encoding='unicode')
