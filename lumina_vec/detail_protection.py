"""Recover source-supported thin dark/light details without restoring weak shading."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import find_objects, distance_transform_edt
from skimage.measure import label
from .color import srgb_to_oklab


def restore_thin_details(rgb,base,palette):
    light=srgb_to_oklab(rgb)[...,0].astype(np.float32)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    ridge=np.maximum(cv2.morphologyEx(light,cv2.MORPH_BLACKHAT,k),cv2.morphologyEx(light,cv2.MORPH_TOPHAT,k))
    base_light=srgb_to_oklab(base)[...,0]
    bright=(ridge>.06)&(light>.8)&(light-base_light>.08)
    dark=(ridge>.06)&(light<.55)&(base_light-light>.08)
    kinds=np.where(bright,1,np.where(dark,2,0))
    components=label(kinds,connectivity=2,background=0)
    boundary=np.zeros(light.shape,bool)
    change=np.any(base[1:]!=base[:-1],axis=2);boundary[1:]|=change;boundary[:-1]|=change
    change=np.any(base[:,1:]!=base[:,:-1],axis=2);boundary[:,1:]|=change;boundary[:,:-1]|=change
    interior=distance_transform_edt(~boundary)
    result=base.copy();mask=np.zeros(light.shape,bool);tree=cKDTree(palette)
    for index,box in enumerate(find_objects(components),1):
        if box is None:continue
        selected=components[box]==index;area=int(selected.sum())
        if area>512:continue
        # Do not turn ordinary antialiased colour boundaries into tiny new regions.
        if np.median(interior[box][selected])<=1:continue
        observations=rgb[box][selected].astype(float);previous=base[box][selected].astype(float)
        distance,chosen=tree.query(np.median(observations,axis=0))
        if distance>30:continue
        color=palette[chosen]
        improvement=np.mean(np.sum((observations-previous)**2,axis=1)-np.sum((observations-color)**2,axis=1))
        if improvement<400:continue
        if area==1 and np.max(np.abs(light[box][selected]-base_light[box][selected]))<.2:continue
        result[box][selected]=color;mask[box][selected]=True
    return result,mask
