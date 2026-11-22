"""Reconstruct an illustration before quantization using its observed edge directions."""
import cv2
import numpy as np

from .color import oklab_to_srgb, srgb_to_oklab


def _orient(light):
    dx = cv2.Sobel(light,cv2.CV_32F,1,0,ksize=3,scale=.125)
    dy = cv2.Sobel(light,cv2.CV_32F,0,1,ksize=3,scale=.125)
    xx,yy,xy = [cv2.boxFilter(v,-1,(5,5)) for v in (dx*dx,dy*dy,dx*dy)]
    anisotropy = np.hypot(xx-yy,2*xy)
    normal = .5*np.arctan2(2*xy,xx-yy)
    return np.cos(normal),np.sin(normal),anisotropy/np.maximum(xx+yy,1e-7)


def redraw_illustration(rgb, fmaps=None):
    """Clean the colour surface and retain locally coherent dark/bright strokes."""
    h,w = rgb.shape[:2]
    if min(h,w)<8:
        return rgb.copy()
    lab = srgb_to_oklab(rgb).astype(np.float32)
    grain_mode = fmaps is not None and (fmaps.summary.get("grain_ratio", 0.0) >= 0.25)
    base = cv2.bilateralFilter(lab, 11 if grain_mode else 9, .090 if grain_mode else .075, 4.2 if grain_mode else 3.1, borderType=cv2.BORDER_REPLICATE)
    base = cv2.bilateralFilter(base, 9, .070 if grain_mode else .060, 3.4 if grain_mode else 2.6, borderType=cv2.BORDER_REPLICATE)
    light = lab[...,0].copy()
    nx,ny,coherence = _orient(light)
    y,x = np.indices((h,w),dtype=np.float32)

    def at(image,u,v):
        return cv2.remap(image,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

    # Sample both sides of a thin feature. It must be lighter or darker
    # than both flanks to be restored as a stroke rather than an edge halo.
    side_a = at(light,x-2*nx,y-2*ny)
    side_b = at(light,x+2*nx,y+2*ny)
    ridge = np.maximum(np.minimum(side_a-light,side_b-light),
                       np.minimum(light-side_a,light-side_b))
    ridge_weight = np.clip((ridge-.027)/.078,0,1)*np.clip((coherence-.32)/.38,0,1)
    broad_edge = np.abs(side_a-side_b)
    edge_weight = np.clip((broad_edge-.055)/.14,0,1)*np.clip((coherence-.44)/.42,0,1)

    # Follow the tangent for denoising narrow structures while rejecting
    # samples of a different colour, especially at crossings.
    signal = base.copy()
    mass = np.ones((h,w),np.float32)
    for step in (-2.5,-1.25,1.25,2.5):
        neighbor = at(base,x-step*ny,y+step*nx)
        diff = np.linalg.norm(neighbor-base,axis=2)
        weight = .6*np.clip(1-diff/.065,0,1)*coherence
        signal += neighbor*weight[...,None]
        mass += weight
    signal /= mass[...,None]
    restore = np.maximum(.95*ridge_weight,.76*edge_weight)
    # A bright point with marked contrast can be important even when it
    # is round. Preserve it without extrapolating an invented contour.
    pin = (light>.88)&(ridge>.13)
    restore[pin] = np.maximum(restore[pin],.85)
    # Preserve more of the observed colour where a group of coherent
    # edges indicates eyelashes, hair or lettering. Flat regions still
    # receive the full surface filter.
    density = cv2.boxFilter(np.maximum(ridge_weight,edge_weight),-1,(11,11))
    detail_floor = .12 + .60*np.clip((density-.10)/.38,0,1)
    restore = np.maximum(restore,detail_floor)
    if fmaps is not None:
        # Lock CNN-detected delicate details and fine strands; smooth washes more strongly.
        restore = np.maximum(restore, np.clip(0.9 * fmaps.delicate_detail, 0.0, 0.92))
        restore = np.maximum(restore, np.clip(0.7 * fmaps.fine_strand, 0.0, 0.85))
    # Allow stronger de-blocking on non-protected washes when JPEG/grain noise is present.
    mix_max = 0.85 if grain_mode else 0.50
    if fmaps is not None:
        wash_boost = np.clip(fmaps.mergeable_wash + fmaps.surface_probs[2], 0.0, 1.0)
        mix_w = (0.50 + (mix_max - 0.50) * wash_boost) * (1.0 - restore)
    else:
        mix_w = mix_max * (1.0 - restore)
    mixed = lab + (signal-lab)*mix_w[...,None]
    return np.rint(oklab_to_srgb(mixed)).clip(0,255).astype(np.uint8)
