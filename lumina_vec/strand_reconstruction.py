"""Recover thin source-supported branches inside a fixed quantizer palette."""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import binary_opening,binary_dilation,gaussian_filter1d
from skimage.measure import label,regionprops
from skimage.morphology import skeletonize
from .strand_evidence import fit_joint_colors,gap_evidence

def extract_branches(rgb,q,base,palette):
    missing=np.sum((rgb.astype(float)-base)**2,axis=2)-np.sum((rgb.astype(float)-q)**2,axis=2)>100
    missing &= ~binary_dilation(binary_opening(missing,structure=np.ones((3,3))),iterations=1)
    components=label(missing,connectivity=2)
    records=[]
    for region in regionprops(components):
        if region.area<12:continue
        sk=skeletonize(region.image);points=set(map(tuple,np.argwhere(sk)+np.array(region.bbox[:2])))
        graph={}
        for y,x in points:
            ns=[]
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if not (dy or dx) or (y+dy,x+dx) not in points:continue
                    if dy and dx and ((y+dy,x) in points or (y,x+dx) in points):continue
                    ns.append((y+dy,x+dx))
            graph[y,x]=ns
        pruned=0
        for _ in range(8):
            remove=set()
            for endpoint in [p for p,ns in graph.items() if len(ns)==1]:
                walk=[endpoint];previous=None;node=endpoint
                while len(walk)<=6:
                    choices=[p for p in graph[node] if p!=previous]
                    if not choices:break
                    previous,node=node,choices[0]
                    if len(graph[node])>2:
                        remove.update(walk);break
                    if len(graph[node])!=2:break
                    walk.append(node)
            if not remove:break
            pruned+=len(remove)
            graph={p:[n for n in ns if n not in remove] for p,ns in graph.items() if p not in remove}
        points=set(graph)
        ends=[p for p,ns in graph.items() if len(ns)==1]
        if len(ends)!=2 or any(len(ns)>2 for ns in graph.values()):continue
        chain=[ends[0]];previous=None
        while True:
            choices=[p for p in graph[chain[-1]] if p!=previous]
            if not choices:break
            previous=chain[-1];chain.append(choices[0])
        if len(chain)!=len(points) or len(chain)<12:continue
        yy,xx=region.coords.T;obs=rgb[yy,xx].astype(float);bg=base[yy,xx].astype(float)
        best=None
        for c in palette.astype(float):
            delta=c-bg;den=np.sum(delta*delta,axis=1)
            alpha=np.sum((obs-bg)*delta,axis=1)/np.maximum(den,1)
            residual=np.mean(np.sum((obs-(bg+np.clip(alpha,0,1)[:,None]*delta))**2,axis=1))
            if np.percentile(alpha,90)<.65 or np.percentile(alpha,90)>1.2:continue
            if (alpha<.1).mean()>.2:continue
            if best is None or residual<best[0]:best=(residual,c,np.clip(alpha,0,1))
        if best is None or best[0]>100:continue
        residual,color,alpha=best
        xy=np.array(chain,float)[:,::-1]+.5
        length=np.linalg.norm(np.diff(xy,axis=0),axis=1).sum();width=float(np.clip(alpha.sum()/length,.3,2.5))
        sm=gaussian_filter1d(xy,1.,axis=0);sm[[0,-1]]=xy[[0,-1]]
        sample_xy=np.column_stack([xx+.5,yy+.5])
        bins=cKDTree(sm).query(sample_xy)[1]
        mass=np.bincount(bins,weights=alpha,minlength=len(sm))
        tangent=np.gradient(sm,axis=0)
        step=np.maximum(np.linalg.norm(tangent,axis=1),.5)
        normals=np.column_stack([-tangent[:,1],tangent[:,0]])/step[:,None]
        offsets=np.sum((sample_xy-sm[bins])*normals[bins],axis=1)
        moment=np.bincount(bins,weights=alpha*offsets,minlength=len(sm))
        smooth_mass=gaussian_filter1d(mass,2,mode='nearest')
        shift=gaussian_filter1d(moment,2,mode='nearest')/np.maximum(smooth_mass,.01)
        sm+=normals*np.clip(shift,-.5,.5)[:,None]
        widths=np.clip(smooth_mass/gaussian_filter1d(step,2,mode='nearest'),.15,2.5)
        boundary=np.vstack([sm+normals*widths[:,None]/2,(sm-normals*widths[:,None]/2)[::-1]])
        records.append(dict(component=region.label,pruned_spur_pixels=pruned,endpoints=xy[[0,-1]].tolist(),centerline=sm.tolist(),widths=widths.tolist(),support_pixels=region.coords.tolist(),length=float(length),width=width,width_range=[float(widths.min()),float(widths.max())],color=color.tolist(),residual=residual))
    return records


def supported_pairs(rgb,base,palette,branches,max_gap=40):
    """Join supported endpoints into same-colour chains, without branches or cycles."""
    candidates=[]
    for i,a in enumerate(branches):
        ca=np.array(a['centerline'])
        for j in range(i+1,len(branches)):
            b=branches[j];cb=np.array(b['centerline'])
            for ia in (0,-1):
                for ib in (0,-1):
                    p,q=ca[ia],cb[ib];delta=q-p;distance=np.linalg.norm(delta)
                    if not 2<distance<max_gap:continue
                    ta=p-ca[3 if ia==0 else -4];tb=cb[3 if ib==0 else -4]-q
                    ta/=np.linalg.norm(ta);tb/=np.linalg.norm(tb)
                    if min(ta@delta/distance,tb@delta/distance)<.8:continue
                    pixels=np.array(a['support_pixels']+b['support_pixels']);yy,xx=pixels.T
                    fits=fit_joint_colors(rgb[yy,xx],base[yy,xx],palette)
                    if not fits:continue
                    t=np.linspace(0,1,int(distance*2)+1)[:,None]
                    found=None
                    for fit in fits:
                        for tension in (.2,.35,.5):
                            curve=(1-t)**3*p+3*(1-t)**2*t*(p+ta*distance*tension)+3*(1-t)*t**2*(q-tb*distance*tension)+t**3*q
                            evidence=gap_evidence(rgb,base,curve,fit['color'],local_background=True)
                            if evidence['status']=='supported':
                                found=dict(a=i,b=j,end_a=ia,end_b=ib,gap=curve.tolist(),color=fit['color'],residual=fit['residual'],distance=float(distance),evidence=evidence)
                                break
                        if found:break
                    if found:candidates.append(found)
    return select_connections(candidates)


def select_connections(candidates):
    """An endpoint is used once; the other endpoint remains available to continue."""
    candidates=sorted(candidates,key=lambda p:(p['residual'],p['distance']))
    used=set();selected=[];parent={};colors={}
    def root(i):
        parent.setdefault(i,i)
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for p in candidates:
        ea=(p['a'],p['end_a']);eb=(p['b'],p['end_b'])
        if ea in used or eb in used:continue
        a,b=root(p['a']),root(p['b']);color=tuple(p['color'])
        if a==b:continue
        if any(r in colors and colors[r]!=color for r in (a,b)):continue
        selected.append(p);used.update((ea,eb));parent[b]=a;colors[a]=color
    return selected


def rebuild_partition(base,rgb,branches,pairs,scale=4):
    """Insert reconstructed bands into one colour partition, never overlay SVGs."""
    work=cv2.resize(base,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    def band(xy,widths,color):
        tangent=np.gradient(xy,axis=0)
        normal=np.column_stack([-tangent[:,1],tangent[:,0]])/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-6)
        polygon=np.vstack([xy+normal*widths[:,None]/2,(xy-normal*widths[:,None]/2)[::-1]])
        cv2.fillPoly(work,[np.rint(polygon*scale).astype(np.int32)],tuple(map(int,color)))
    used={i for p in pairs for i in (p['a'],p['b'])}
    for i,b in enumerate(branches):
        if i not in used:band(np.array(b['centerline']),np.array(b['widths']),b['color'])
    links={}
    for p in pairs:
        links[(p['a'],p['end_a'])]=(p['b'],p['end_b'],p,False)
        links[(p['b'],p['end_b'])]=(p['a'],p['end_a'],p,True)
    visited=set()
    for start in sorted(used):
        if start in visited:continue
        free=[e for e in (0,-1) if (start,e) not in links]
        if not free:continue
        entry=free[0];index=start;pieces=[];width_pieces=[];color=None
        while True:
            branch=branches[index];visited.add(index)
            xy=np.array(branch['centerline']);widths=np.array(branch['widths'])
            if entry==-1:xy=xy[::-1];widths=widths[::-1]
            exit_end=-1 if entry==0 else 0
            link=links.get((index,exit_end))
            if color is None:color=np.array(link[2]['color'])
            yy,xx=np.array(branch['support_pixels']).T;bg=base[yy,xx].astype(float);obs=rgb[yy,xx].astype(float)
            delta=color-bg;alpha=np.clip(np.sum((obs-bg)*delta,axis=1)/np.maximum(np.sum(delta**2,axis=1),1),0,1)
            widths*=np.clip(alpha.sum()/branch['length']/branch['width'],.4,2.5)
            if pieces:
                # The preceding connector ends at this branch's entry point.
                width_pieces[-1]=np.linspace(width_pieces[-2][-1],widths[0],len(pieces[-1])+2)[1:-1]
            pieces.append(xy);width_pieces.append(widths)
            if link is None:break
            index,entry,p,reverse=link
            assert index not in visited, 'Cyclic strand connection'
            assert tuple(p['color'])==tuple(color), 'Mixed-colour strand connection'
            gap=np.array(p['gap'])
            if reverse:gap=gap[::-1]
            pieces.append(gap[1:-1]);width_pieces.append(np.zeros(len(gap)-2))
        xy=np.vstack(pieces);widths=np.concatenate(width_pieces)
        band(gaussian_filter1d(xy,.7,axis=0),np.clip(widths,.15,2.5),color)
    assert visited==used, 'Strand connections must form open chains'
    return work
