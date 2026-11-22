"""Planar colour regions assembled from curves fitted once per shared boundary."""
from collections import defaultdict
import re
import xml.etree.ElementTree as ET
import numpy as np
import vtracer
import cv2
from scipy.ndimage import gaussian_filter1d, median, map_coordinates
from .contour import fit_cubic, _unit


def _pixel_loops(d):
    tokens=re.findall(r'[MLHVZmlhvz]|-?\d+(?:\.\d+)?',d)
    point=np.zeros(2);start=point.copy();loop=[];i=0;cmd=None
    while i<len(tokens):
        if tokens[i].isalpha():
            cmd=tokens[i];i+=1
            if cmd.upper()=='Z':
                yield np.array(loop+[start.copy()],int)
                point=start.copy();loop=[];continue
        n=1 if cmd.upper() in ('H','V') else 2
        values=np.array([float(v) for v in tokens[i:i+n]]);i+=n
        q=point.copy()
        if cmd.upper() in ('M','L'): q=values+(point if cmd.islower() else 0)
        else:
            axis=0 if cmd.upper()=='H' else 1
            q[axis]=values[0]+(point[axis] if cmd.islower() else 0)
        point=q
        if cmd.upper()=='M': start=q.copy();cmd='l' if cmd.islower() else 'L'
        loop.append(q.copy())


def segment(rgb, precision=7, difference=12, speckle=1, representative='median'):
    """Use the clusterer only; its pixel contours are an intermediate label map."""
    h,w=rgb.shape[:2]
    cfg=vtracer.Config(color_precision=precision,layer_difference=difference,
                       filter_speckle=speckle,hierarchical='cutout',mode='pixel')
    raw=cfg.convert_pixels(np.dstack([rgb,np.full((h,w),255,np.uint8)]).tobytes(),w,h)
    labels=np.full((h,w),-1,np.int32)
    colors=[]
    for path in ET.fromstring(raw):
        if not path.tag.endswith('path'): continue
        index=len(colors);colors.append(path.get('fill'))
        events=defaultdict(list)
        for xy in _pixel_loops(path.get('d','')):
            if not np.array_equal(xy[0],xy[-1]): xy=np.vstack([xy,xy[0]])
            for a,b in zip(xy[:-1],xy[1:]):
                if a[0]!=b[0]: continue
                for y in range(max(0,min(a[1],b[1])),min(h,max(a[1],b[1]))):
                    events[y].append(int(a[0]))
        for y,xs in events.items():
            xs.sort()
            assert len(xs)%2==0,(path.attrib,y,xs)
            for a,b in zip(xs[::2],xs[1::2]): labels[y,max(0,a):min(w,b)]=index
    assert (labels>=0).all(), 'Clustering left uncovered pixels'
    # Cluster averages are contaminated by antialiased boundary pixels.
    # Recover each region's representative source colour before fitting geometry.
    if representative=='mode':
        # Channel-wise medians can invent RGB triplets absent from an already
        # quantized image. Select one existing complete RGB colour per region.
        pixels=rgb.astype(np.int64)
        packed=(pixels[:,:,0]<<16)|(pixels[:,:,1]<<8)|pixels[:,:,2]
        keys,counts=np.unique((labels.astype(np.int64)<<24)|packed,return_counts=True)
        region=keys>>24;order=np.lexsort((keys,-counts,region));keys=keys[order];region=region[order]
        first=np.r_[True,region[1:]!=region[:-1]];chosen=keys[first]&0xffffff
        values=np.zeros((len(colors),3),np.uint8)
        values[region[first]]=np.column_stack([chosen>>16,(chosen>>8)&255,chosen&255])
    elif representative=='median':
        values=np.stack([median(rgb[:,:,c],labels,np.arange(len(colors))) for c in range(3)],axis=1)
    else:raise ValueError('Unknown region colour representative')
    colors=['#%02x%02x%02x'%tuple(np.rint(v).astype(int)) for v in values]
    return labels,colors


def boundary_graph(labels):
    h,w=labels.shape
    edges=[];adj=defaultdict(list)
    def add(x0,y0,x1,y1,right,left):
        a=y0*(w+1)+x0;b=y1*(w+1)+x1
        i=len(edges);edges.append((a,b,int(right),int(left)))
        adj[a].append(i);adj[b].append(i)
    # Horizontal east-facing edges have the lower region on their right.
    pad=np.pad(labels,1,constant_values=-1)
    ys,xs=np.where(pad[1:,1:-1]!=pad[:-1,1:-1])
    for y,x in zip(ys.tolist(),xs.tolist()): add(x,y,x+1,y,pad[y+1,x+1],pad[y,x+1])
    # Vertical south-facing edges have the western region on their right.
    ys,xs=np.where(pad[1:-1,1:]!=pad[1:-1,:-1])
    for y,x in zip(ys.tolist(),xs.tolist()): add(x,y,x,y+1,pad[y+1,x],pad[y+1,x+1])
    junctions={v for v,es in adj.items() if len(es)!=2 or
               set(edges[es[0]][2:])!=set(edges[es[1]][2:])}
    seen=set();chains=[]
    def walk(start,edge):
        vertices=[start];current=start
        first=edges[edge]
        right,left=first[2:] if first[0]==start else first[2:][::-1]
        while edge not in seen:
            seen.add(edge);a,b,_,_=edges[edge]
            current=b if current==a else a;vertices.append(current)
            if current in junctions or current==start: break
            edge=next(e for e in adj[current] if e not in seen)
        points=np.array([(v%(w+1),v//(w+1)) for v in vertices],float)
        chains.append(dict(vertices=vertices,points=points,right=right,left=left))
    for v in sorted(junctions):
        for edge in adj[v]:
            if edge not in seen: walk(v,edge)
    for edge,(a,b,r,l) in enumerate(edges):
        if edge not in seen: walk(a,edge)
    return chains,len(edges)


def refine_labels(labels,colors,rgb,iterations=2):
    """Refine existing regions at subpixel scale without creating antialias colour bands."""
    h,w=rgb.shape[:2]
    labels=cv2.resize(labels,(w,h),interpolation=cv2.INTER_NEAREST)
    palette=np.array([list(bytes.fromhex(c[1:])) for c in colors],np.float32)
    pixels=rgb.astype(np.float32)
    for _ in range(iterations):
        padded=np.pad(labels,1,mode='edge')
        best=((pixels-palette[labels])**2).sum(axis=2)
        result=labels.copy()
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            candidate=padded[1+dy:h+1+dy,1+dx:w+1+dx]
            error=((pixels-palette[candidate])**2).sum(axis=2)
            improve=error+1<best
            result[improve]=candidate[improve];best[improve]=error[improve]
        # Only move simple boundary points: never sever a thin strand, join two
        # disconnected regions, create a hole, or erase an isolated detail.
        ring=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        simple=np.zeros(256,bool)
        for code in range(1,255):
            bits=[bool(code&(1<<i)) for i in range(8)]
            simple[code]=bool(code & 0b10101010) and sum(bits[i] and not bits[i-1] for i in range(8))==1
        for py,px in [(0,0),(0,1),(1,0),(1,1)]:
            padded=np.pad(labels,1,constant_values=-1)
            current=labels[py::2,px::2];proposed=result[py::2,px::2]
            before=np.zeros(current.shape,np.uint8);after=before.copy()
            for bit,(dy,dx) in enumerate(ring):
                neighbor=padded[1+dy+py:h+1+dy:2,1+dx+px:w+1+dx:2]
                before|=((neighbor==current).astype(np.uint8)<<bit)
                after|=((neighbor==proposed).astype(np.uint8)<<bit)
            accept=(current!=proposed)&simple[before]&simple[after]
            current[accept]=proposed[accept]
    return labels


def _fit(points,tol,smoothing=.65):
    if len(points)<=2:
        d=points[-1]-points[0]
        return [np.array([points[0],points[0]+d/3,points[0]+d*2/3,points[-1]])]
    # Fix junctions exactly; all regions subsequently reuse these same control points.
    sm=gaussian_filter1d(points,smoothing,axis=0,mode='nearest')
    sm[0]=points[0];sm[-1]=points[-1]
    if np.array_equal(points[0],points[-1]):
        k=len(points)//2
        return _fit(points[:k+1],tol,smoothing)+_fit(points[k:],tol,smoothing)
    return fit_cubic(sm,_unit(sm[1]-sm[0]),_unit(sm[-2]-sm[-1]),tol)


def _subpixel(points,right,left,rgb,colors):
    if right<0 or left<0 or len(points)<4: return points
    a=np.array(list(bytes.fromhex(colors[left][1:])),float)
    b=np.array(list(bytes.fromhex(colors[right][1:])),float)
    delta=b-a;den=np.dot(delta,delta)
    if den<100: return points
    tangent=np.gradient(points,axis=0)
    normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
    offsets=np.linspace(-1,1,9)
    locations=points[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    samples=np.stack([map_coordinates(rgb[:,:,c], [locations[:,:,1]-.5,locations[:,:,0]-.5],order=1,mode='nearest') for c in range(3)],axis=2)
    alpha=((samples-a)*delta).sum(axis=2)/den
    crosses=(alpha[:,:-1]<=.5)&(alpha[:,1:]>=.5)
    shifts=offsets[:-1]+(.5-alpha[:,:-1])/np.maximum(alpha[:,1:]-alpha[:,:-1],1e-6)*.25
    candidates=np.where(crosses,np.abs(shifts),np.inf)
    at=candidates.argmin(axis=1)
    shift=np.where(crosses.any(axis=1),shifts[np.arange(len(points)),at],0)
    shift=np.clip(shift,-.75,.75)
    result=points+normal*gaussian_filter1d(shift,.7)[:,None]
    result[[0,-1]]=points[[0,-1]]
    return result


def _marching(points,right,left,rgb,colors):
    """Locate a boundary between neighbouring pixel centres from their RGB coverage."""
    if right<0 or left<0 or len(points)<3: return points
    a=np.array(list(bytes.fromhex(colors[left][1:])),float)
    b=np.array(list(bytes.fromhex(colors[right][1:])),float)
    delta=b-a;den=np.dot(delta,delta)
    centers=(points[:-1]+points[1:])/2
    direction=np.diff(points,axis=0)
    normal=np.stack([-direction[:,1],direction[:,0]],axis=1)
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
    if den>=64:
        cover=[]
        for side in (-.5,.5):
            p=centers+normal*side
            samples=np.stack([map_coordinates(rgb[:,:,c],[p[:,1]-.5,p[:,0]-.5],order=1,mode='nearest') for c in range(3)],axis=1)
            cover.append(((samples-a)*delta).sum(axis=1)/den)
        lo,hi=cover
        good=(hi-lo>.05)&(lo<=.65)&(hi>=.35)
        offset=np.where(good,np.clip((.5-lo)/np.maximum(hi-lo,.05)-.5,-.5,.5),0)
        centers+=normal*offset[:,None]
    return np.vstack([points[0],centers,points[-1]])


def _coverage_quantile(alpha,normal):
    a=np.max(np.abs(normal),axis=1);b=np.min(np.abs(normal),axis=1)
    q=np.minimum(alpha,1-alpha)
    tail=q<b/np.maximum(2*a,1e-8)
    value=np.where(tail,-(a+b)/2+np.sqrt(np.maximum(2*a*b*q,0)),a*(q-.5))
    return np.where(alpha<=.5,value,-value)


def _coverage_boundary(points,right,left,rgb,colors):
    if right<0 or left<0 or len(points)<4: return points
    a=np.array(list(bytes.fromhex(colors[left][1:])),float)
    b=np.array(list(bytes.fromhex(colors[right][1:])),float)
    delta=b-a;den=np.dot(delta,delta)
    if den<100: return _marching(points,right,left,rgb,colors)
    centers=(points[:-1]+points[1:])/2
    tangent=gaussian_filter1d(np.diff(points,axis=0),1.5,axis=0,mode='nearest')
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
    normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
    origin=np.floor(centers).astype(int)
    weighted=np.zeros(len(centers));weights=weighted.copy()
    h,w=rgb.shape[:2]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            x=np.clip(origin[:,0]+dx,0,w-1);y=np.clip(origin[:,1]+dy,0,h-1)
            sample=rgb[y,x]
            alpha=((sample-a)*delta).sum(axis=1)/den
            residual=np.linalg.norm(sample-(a+alpha[:,None]*delta),axis=1)
            relative=np.stack([x+.5,y+.5],axis=1)-centers
            along=(relative*tangent).sum(axis=1);across=(relative*normal).sum(axis=1)
            distance=_coverage_quantile(np.clip(alpha,0,1),normal)
            shift=across-distance
            valid=(alpha>.015)&(alpha<.985)&(residual<12)&(np.abs(shift)<1)
            weight=np.exp(-along**2/1.3)*np.minimum(alpha,1-alpha)*valid/(1+residual*.2)
            weighted+=shift*weight;weights+=weight
    shift=np.divide(weighted,weights,out=np.zeros_like(weights),where=weights>1e-5)
    centers+=normal*shift[:,None]
    return np.vstack([points[0],centers,points[-1]])


def _compact_command(curve, tol_q=0.25, tol_a=0.35):
    a,b,c,d=curve
    v=d-a
    square=v@v
    if square>0:
        u,w=(b-a)@v,(c-a)@v
        if (0<=u<=w<=square
                and abs(v[0]*(b-a)[1]-v[1]*(b-a)[0])<=1e-8
                and abs(v[0]*(c-a)[1]-v[1]*(c-a)[0])<=1e-8):
            return 'L%.3f,%.3f'%tuple(d)
    if tol_a>0 and square>4.0:
        from .layered import _cubic_to_arc
        arc=_cubic_to_arc(curve,tol=tol_a,n=9)
        if arc is not None:
            return arc
    # Exact max deviation of degree-reduced quadratic Q(q) from cubic (a,b,c,d):
    # (sqrt(3)/36) * ||d - 3c + 3b - a||.
    if np.linalg.norm(d-3*c+3*b-a)*0.0481125<=tol_q:
        q=(3*b-a+3*c-d)*0.25
        return 'Q%.3f,%.3f %.3f,%.3f'%(q[0],q[1],d[0],d[1])
    return 'C%.3f,%.3f %.3f,%.3f %.3f,%.3f'%tuple(curve[1:].ravel())


def trace_regions(labels,colors,tol=.35,seam=0,rgb=None,placement='normal',geometry_guard=False,smoothing=.65,structural_scale=None,source_scale=1.,structural_fitter=None,secondary_fitter=None, secondary_network=None, *, exact_grid=False, compact_commands=False, chain_locator=None, chain_scale_fn=None):
    chains,atomic_edges=boundary_graph(labels)
    faces=defaultdict(list)
    for i,c in enumerate(chains):
        locate={'normal':_subpixel,'marching':_marching,'coverage':_coverage_boundary}[placement]
        fit_points=locate(c['points']/source_scale,c['right'],c['left'],rgb,colors)*source_scale if rgb is not None else c['points']
        if exact_grid or c['right']<0 or c['left']<0:
            p=c['points']
            turns=np.flatnonzero(np.any(np.diff(p,n=2,axis=0)!=0,axis=1))+1
            p=p[np.r_[0,turns,len(p)-1]]
            c['curves']=[np.array([a,a+(b-a)/3,a+(b-a)*2/3,b]) for a,b in zip(p[:-1],p[1:])]
        else:
            original_points=fit_points
            if chain_locator is not None:
                fit_points=chain_locator(fit_points,c['right'],c['left'],source_scale)
            if structural_scale is not None:
                from .structural_curves import fit_structural
                fitter=structural_fitter or fit_structural
                c_scale = chain_scale_fn(fit_points, structural_scale, source_scale) if chain_scale_fn is not None else structural_scale
                c['_scale'] = c_scale
                c['curves']=fitter(fit_points,c_scale)
                if fit_points is not original_points and np.max(
                        np.linalg.norm(fit_points-original_points,axis=1))>2*source_scale:
                    c['relocation']=(fit_points, fitter, c_scale)
            else:
                c['curves']=_fit(fit_points,tol,smoothing)
        if c['right']>=0: faces[c['right']].append((i,1))
        if c['left']>=0: faces[c['left']].append((i,-1))
    guard_stats={}
    guard_cache={}
    if geometry_guard:
        from .geometry_guard import guard
        guard_stats=guard(chains, cached=guard_cache)
    if secondary_fitter is not None or secondary_network is not None:
        prior=[chain['curves'] for chain in chains]
        before=sum(map(len,prior))
        for chain in chains:
            if secondary_fitter is not None and min(chain['left'], chain['right']) >= 0:
                chain['curves']=secondary_fitter(chain['curves'], chain.get('_scale', structural_scale))
        if secondary_network is not None:
            secondary_network(chains, structural_scale)
        if geometry_guard:
            guard_stats.update({f'secondary_{key}': value for key, value in guard(chains,prior=prior,cached=guard_cache).items()})
        guard_stats['segments_before_secondary']=before
    def oriented(ref):
        i,d=ref;c=chains[i]
        if d==1: return c['vertices'][0],c['vertices'][-1],c['points'],c['curves']
        return c['vertices'][-1],c['vertices'][0],c['points'][::-1],[s[::-1] for s in c['curves'][::-1]]
    paths=[];loops=0
    for face,refs in faces.items():
        outgoing=defaultdict(list)
        for ref in refs: outgoing[oriented(ref)[0]].append(ref)
        used=set();parts=[]
        for initial in refs:
            if initial in used: continue
            start=oriented(initial)[0];ref=initial
            pieces=[];walk_refs=[]
            while ref not in used:
                used.add(ref);a,b,p,segs=oriented(ref)
                pieces.extend(segs)
                walk_refs.append(ref)
                if b==start: break
                opts=[r for r in outgoing[b] if r not in used]
                assert opts, 'Open region boundary'
                direction=p[-1]-p[-2]
                def turn(r):
                    q=oriented(r)[2];v=q[1]-q[0]
                    return np.arctan2(direction[0]*v[1]-direction[1]*v[0],np.dot(direction,v))
                ref=max(opts,key=turn)
            assert b==start
            # A face can touch itself at a graph junction. Emit separate rings
            # there instead of a figure-eight contour that importers reject.
            pending=[];vertices=[start];positions={start:0}
            for edge_ref in walk_refs:
                _,end,_,_=oriented(edge_ref);pending.append(edge_ref)
                if end in positions:
                    k=positions[end];cycle=pending[k:]
                    pieces=[s for r in cycle for s in oriented(r)[3]]
                    loops+=1
                    if exact_grid:
                        parts.append('M%d,%d'%tuple(pieces[0][0]))
                        parts.extend('L%d,%d'%tuple(s[-1]) for s in pieces[:-1])
                    else:
                        parts.append('M%.3f,%.3f'%tuple(pieces[0][0]))
                        if compact_commands:
                            sc=max(1.0,float(source_scale))
                            parts.extend(_compact_command(s,tol_q=.25*sc,tol_a=.35*sc) for s in pieces)
                        else:
                            parts.extend('C%.3f,%.3f %.3f,%.3f %.3f,%.3f'%tuple(s[1:].ravel()) for s in pieces)
                    parts.append('Z');pending=pending[:k];vertices=vertices[:k+1]
                    positions={v:i for i,v in enumerate(vertices)}
                else:positions[end]=len(vertices);vertices.append(end)
            assert not pending
        stroke=f' stroke="{colors[face]}" stroke-width="{seam}" stroke-linejoin="round" paint-order="stroke fill"' if seam else ''
        paths.append(f'<path id="region-{face}" fill="{colors[face]}" fill-rule="evenodd"{stroke} d="{"".join(parts)}"/>')
    h,w=labels.shape
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'+''.join(paths)+'</svg>'
    stats=dict(regions=len(paths),loops=loops,shared_chains=sum(c['right']>=0 and c['left']>=0 for c in chains),chains=len(chains),atomic_edges=atomic_edges,curve_segments=sum(len(c['curves']) for c in chains))
    stats.update(guard_stats)
    return svg,stats
