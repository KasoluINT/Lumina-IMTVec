"""Colour-mode seeds with confidence margins for closed illustration regions."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label
from scipy.ndimage import distance_transform_edt


def extract_palette(rgb):
    hi=cv2.dilate(rgb,np.ones((3,3),np.uint8));lo=cv2.erode(rgb,np.ones((3,3),np.uint8))
    core=np.max(hi.astype(np.int16)-lo,axis=2)<=2
    if core.mean()<.1: core=np.max(hi.astype(np.int16)-lo,axis=2)<=6
    packed=(rgb[:,:,0].astype(np.uint32)<<16)|(rgb[:,:,1].astype(np.uint32)<<8)|rgb[:,:,2]
    keys,counts=np.unique(packed[core],return_counts=True)
    palette=[];minimum=max(8,round(rgb.shape[0]*rgb.shape[1]*.000005))
    for i in np.argsort(counts)[::-1]:
        if counts[i]<minimum: break
        k=int(keys[i]);c=np.array([k>>16,(k>>8)&255,k&255],float)
        if not palette or np.min(np.linalg.norm(c-np.array(palette),axis=1))>4:
            palette.append(c)
    if len(palette)<2: raise ValueError('Insufficient stable colour modes')
    return np.array(palette,np.uint8),dict(core_fraction=float(core.mean()),palette=len(palette))


def seeded_regions(rgb,palette,min_seed=3):
    pixels=rgb.reshape(-1,3);tree=cKDTree(palette)
    index=np.empty(len(pixels),np.int32);valid=np.empty(len(pixels),bool)
    for start in range(0,len(pixels),250000):
        d,ids=tree.query(pixels[start:start+250000],k=2)
        index[start:start+len(d)]=ids[:,0]
        valid[start:start+len(d)]=(d[:,0]<1.5)|((d[:,0]<15)&(d[:,1]>d[:,0]*3+4))
    index=index.reshape(rgb.shape[:2]);valid=valid.reshape(index.shape)
    markers=np.where(valid,index+1,0).astype(np.int32)
    components=label(markers,connectivity=1,background=0)
    area=np.bincount(components.ravel());markers[area[components]<min_seed]=0
    markers=cv2.watershed(rgb.copy(),markers)
    for _ in range(8):
        ys,xs=np.where(markers<=0)
        if not len(ys): break
        best=np.full(len(ys),np.inf);chosen=np.zeros(len(ys),np.int32)
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny=np.clip(ys+dy,0,len(rgb)-1);nx=np.clip(xs+dx,0,rgb.shape[1]-1)
            ids=markers[ny,nx]
            cost=((rgb[ys,xs].astype(float)-palette[np.maximum(ids-1,0)])**2).sum(axis=1)
            improve=(ids>0)&(cost<best)
            chosen[improve]=ids[improve];best[improve]=cost[improve]
        markers[ys,xs]=chosen
    if (markers<=0).any(): raise ValueError('Unassigned region pixels')
    regions=label(markers,connectivity=1)-1
    colors=np.zeros((regions.max()+1,3),np.uint8)
    colors[regions.ravel()]=palette[markers.ravel()-1]
    return regions,colors


def restore_interior_details(rgb,regions,colors,palette):
    """Recover high-confidence colour islands missing inside a larger flat region."""
    d,ids=cKDTree(palette).query(rgb.reshape(-1,3),k=2)
    d=d.reshape((*regions.shape,2));ids=ids.reshape((*regions.shape,2))
    predicted=colors[regions]
    error=np.linalg.norm(rgb.astype(float)-predicted,axis=2)
    boundary=np.zeros(regions.shape,bool)
    boundary[:,1:]|=regions[:,1:]!=regions[:,:-1];boundary[:,:-1]|=regions[:,1:]!=regions[:,:-1]
    boundary[1:]|=regions[1:]!=regions[:-1];boundary[:-1]|=regions[1:]!=regions[:-1]
    interior=distance_transform_edt(~boundary)>1.5
    confident=(d[:,:,0]<20)&(d[:,:,1]>d[:,:,0]*2+8)&(error-d[:,:,0]>25)&interior
    components=label(np.where(confident,ids[:,:,0]+1,0),background=0,connectivity=1)
    area=np.bincount(components.ravel())
    keep=confident&((area[components]>=2)|((d[:,:,0]<3)&(error>80)))
    work=predicted.copy();work[keep]=palette[ids[:,:,0][keep]]
    packed=(work[:,:,0].astype(np.int32)<<16)|(work[:,:,1].astype(np.int32)<<8)|work[:,:,2]
    result=label(packed+1,connectivity=1)-1
    values=np.zeros((result.max()+1,3),np.uint8);values[result.ravel()]=work.reshape(-1,3)
    return result,values,dict(restored_pixels=int(keep.sum()),restored_components=int(len(np.unique(components[keep]))))
