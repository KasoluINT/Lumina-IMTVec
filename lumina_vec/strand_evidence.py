"""Source evidence for joining thin regions; no geometry is modified here."""
import numpy as np
from scipy.ndimage import map_coordinates


def fit_joint_colors(observed, background, palette, maximum_error=150):
    observed=np.asarray(observed,float);background=np.asarray(background,float)
    fits=[]
    for color in np.asarray(palette,float):
        delta=color-background
        alpha=np.sum((observed-background)*delta,axis=-1)/np.maximum(np.sum(delta**2,axis=-1),1)
        if not .65<np.percentile(alpha,90)<1.2:continue
        error=float(np.mean(np.sum((observed-background-np.clip(alpha,0,1)[...,None]*delta)**2,axis=-1)))
        if error<=maximum_error:fits.append(dict(color=color.astype(int).tolist(),residual=error))
    return sorted(fits,key=lambda x:x['residual'])


def gap_evidence(rgb,background,centers,color,radius=1.25,local_background=False):
    centers=np.asarray(centers,float)
    tangent=np.gradient(centers,axis=0)
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
    locations=centers[:,None,:]+np.linspace(-radius,radius,11)[None,:,None]*normal[:,None,:]
    def sample(image):
        return np.stack([map_coordinates(np.asarray(image,float)[:,:,c],[locations[:,:,1]-.5,locations[:,:,0]-.5],order=1,mode='nearest') for c in range(3)],axis=-1)
    obs=sample(rgb);bg=sample(background)
    flank_ok=np.ones(len(centers),bool)
    if local_background:
        sides=centers[:,None,:]+np.array([-4.,-3.,3.,4.])[None,:,None]*normal[:,None,:]
        samples=np.stack([map_coordinates(np.asarray(rgb,float)[:,:,c],[sides[:,:,1]-.5,sides[:,:,0]-.5],order=1,mode='nearest') for c in range(3)],axis=-1)
        bg=np.broadcast_to(np.median(samples,axis=1)[:,None,:],obs.shape)
        left=np.mean(samples[:,:2],axis=1);right=np.mean(samples[:,2:],axis=1)
        contrast=np.linalg.norm(np.asarray(color,float)-bg[:,0],axis=1)
        similar=np.linalg.norm(left-right,axis=1)<8+.35*contrast
        # A strand can cross a background boundary. Require a distinct thin
        # signal relative to BOTH sides instead of demanding equal backgrounds.
        ridge=np.ones(obs.shape[:2],bool)
        for side in (left,right):
            change=np.asarray(color,float)-side
            den=np.sum(change**2,axis=1)
            coverage=np.sum((obs-side[:,None,:])*change[:,None,:],axis=-1)/np.maximum(den[:,None],1)
            residual=np.sum((obs-side[:,None,:]-np.clip(coverage,0,1)[:,:,None]*change[:,None,:])**2,axis=-1)
            ridge&=(den[:,None]>64)&(coverage>.12)&(coverage<1.2)&(residual<100)
        flank_ok=similar|ridge.any(axis=1)
    delta=np.asarray(color,float)-bg
    alpha=np.sum((obs-bg)*delta,axis=-1)/np.maximum(np.sum(delta**2,axis=-1),1)
    error=np.sum((obs-bg-np.clip(alpha,0,1)[:,:,None]*delta)**2,axis=-1)
    valid=(alpha>.12)&(alpha<1.2)&(error<100)&(np.sum((obs-bg)**2,axis=-1)>64)
    supported=valid.any(axis=1)&flank_ok
    spacing=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).mean())
    runs=''.join('1' if x else '0' for x in supported).split('1')
    longest=max(map(len,runs),default=0)*spacing
    fraction=float(supported.mean())
    # A long real gap and a modelling failure must not silently become a join.
    status='supported' if fraction>=.95 and longest<=1 else ('ambiguous' if fraction>=.8 else 'unsupported')
    return dict(status=status,gap_support=fraction,longest_unsupported=float(longest),supported_samples=supported.tolist())
