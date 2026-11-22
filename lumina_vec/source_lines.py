"""Source-space dark and bright line evidence, not an SVG overlay."""
import numpy as np
import cv2


def source_line_evidence(rgb,radius=2,return_polarity=False):
    gray=(rgb.max(axis=2).astype(np.float32)+rgb.min(axis=2).astype(np.float32))/2
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
    kernel=(x*x+y*y<=radius*radius).astype(np.uint8)
    high=cv2.dilate(gray,kernel,borderType=cv2.BORDER_REPLICATE)
    low=cv2.erode(gray,kernel,borderType=cv2.BORDER_REPLICATE)
    dark=np.divide(gray,high,out=np.ones_like(gray),where=high>1e-6)
    bright=np.divide(255-gray,255-low,out=np.ones_like(gray),where=255-low>1e-6)
    # Same input levels as the investigated dodge recipe (96..245).
    dark=1-np.clip((dark*255-96)/(245-96),0,1)
    bright=1-np.clip((bright*255-96)/(245-96),0,1)
    strength=np.maximum(dark,bright)
    strength[high-low<3]=0
    # Faint chromatic thin ridges (e.g. pale gold/pink arcs on light paper): morphological
    # tophat/blackhat responds only to 1-3 px lines/strands and stays near zero on region steps.
    from .color import srgb_to_oklab
    lab = srgb_to_oklab(rgb)
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 3, 2 * radius + 3))
    c_ridge = np.zeros_like(gray)
    for ch in (1, 2):
        v = lab[..., ch].astype(np.float32)
        th = np.abs(cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kr))
        bh = np.abs(cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, kr))
        c_ridge = np.maximum(c_ridge, np.maximum(th, bh))
    chroma_line = np.clip((c_ridge - 0.025) / 0.065, 0.0, 1.0)
    strength = np.maximum(strength, chroma_line * 0.85)
    if return_polarity:
        polarity=np.where(strength>.35,np.where(dark>=bright,1,2),0).astype(np.uint8)
        return strength.astype(np.float32),polarity
    return strength.astype(np.float32)
