"""Relocate an existing shared edge using local source colour transitions."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .color import srgb_to_oklab
from .tone_boundary import track_tone_boundary


class SourceCorridor:
    def __init__(self, source, colors):
        self.source = np.ascontiguousarray(srgb_to_oklab(source))
        self.palette = srgb_to_oklab(colors)
        self.considered = self.moved = 0
        self.tonal_chains = 0

    def _profile(self, xy, normal, offsets):
        coords = xy[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        loc = [coords[...,1]-.5,coords[...,0]-.5]
        return np.stack([map_coordinates(self.source[...,c],loc,order=1,mode='nearest')
                         for c in range(3)],axis=-1)

    def __call__(self, points, right, left, scale):
        if min(right,left)<0 or len(points)<4 or np.array_equal(points[0],points[-1]):
            return points
        direction = self.palette[right]-self.palette[left]
        contrast = np.linalg.norm(direction)
        if contrast < .026:
            return points
        dist = np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))/scale]
        strong_edge = dist[-1] >= 10 and np.linalg.norm(points[-1]-points[0]) >= .45*dist[-1]*scale
        tonal_edge = (dist[-1] >= 65 and .033 <= contrast <= .076
                      and np.linalg.norm(direction[1:]) < .009)
        if not strong_edge and not tonal_edge:
            return points
        stations = np.linspace(0,dist[-1],max(9,int(dist[-1]/.60)+1))
        xy = np.column_stack([np.interp(stations,dist,p) for p in points.T])/scale
        tangent = gaussian_filter1d(np.gradient(xy,axis=0),2.,axis=0,mode='nearest')
        normal = tangent[:,::-1]*[-1,1]
        normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
        offsets = np.arange(-3.,3.01,.25)
        value = self._profile(xy,normal,offsets)
        axis = direction/contrast
        center = (self.palette[right]+self.palette[left])*.5
        diff = value-center
        proj = diff @ axis
        # The two materials must still be identifiable across the corridor;
        # a much brighter reflection or a third colour is not a replacement edge.
        dx = proj[:,2:]-proj[:,:-2]
        mid = proj[:,1:-1]
        residual = np.linalg.norm(diff[:,1:-1]-mid[...,None]*axis,axis=-1)
        candidates = offsets[1:-1]
        far = abs(candidates)
        allowed = ((dx > max(.006,.15*contrast)) & (abs(mid)<max(.016,.36*contrast))
                   & (residual<max(.025,.45*contrast)) & (far<=1.75))
        rank = dx - .40*abs(mid) - far*(.006+.09*contrast)
        best = np.argmax(np.where(allowed,rank,-np.inf),axis=1)
        valid = allowed.any(axis=1)
        raw = np.where(valid,candidates[best],0.)
        # Move in runs of agreeing observations only. Isolated strong pixels
        # along otherwise steady gradients do not become kinks in the contour.
        mass = gaussian_filter1d(valid.astype(float),2.,mode='nearest')
        smooth = gaussian_filter1d(raw,2.,mode='nearest')/np.maximum(mass,.1)
        confidence = np.clip((mass-.51)/.30,0,1)
        fade = np.clip(np.minimum(stations,stations[-1]-stations)/3.2,0,1)
        shift = smooth*confidence*fade if strong_edge else np.zeros(len(stations))
        displacements = shift[:,None]*normal
        if tonal_edge:
            tonal_tangent = gaussian_filter1d(np.gradient(xy,axis=0),10.,axis=0,mode='nearest')
            tonal_normal = tonal_tangent[:,::-1]*[-1,1]
            tonal_normal /= np.maximum(np.linalg.norm(tonal_normal,axis=1,keepdims=True),1e-8)
            tonal_shift = track_tone_boundary(self._profile,xy,tonal_normal,direction,center,stations)
            if np.max(np.abs(tonal_shift)) >= .5:
                displacements += tonal_shift[:,None] * (1-confidence[:,None]) * tonal_normal
                self.tonal_chains += 1
        self.considered += 1
        if np.max(np.linalg.norm(displacements,axis=1))<.22:
            return points
        self.moved += 1
        result = points + np.column_stack([np.interp(dist,stations,p) for p in displacements.T])*scale
        # Preserve original corners whose turn is corroborated on both sides.
        sharp = np.sum(tangent[2:]*tangent[:-2],axis=1)
        norm = np.linalg.norm(tangent[2:],axis=1)*np.linalg.norm(tangent[:-2],axis=1)
        if len(sharp):
            corner = np.r_[False, sharp<.37*norm,False].astype(float)
            preserve = np.interp(dist,stations,corner)>.35
            result[preserve] = points[preserve]
        result[[0,-1]] = points[[0,-1]]
        return result
