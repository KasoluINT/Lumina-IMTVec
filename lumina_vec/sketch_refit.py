"""Evidence-gated second fitting for Lumina's generated outline SVG layer."""
from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely import LineString, Polygon

from .color import srgb_to_oklab
from .contour import fit_cubic, _unit
from .geometry_guard import flatten
from .structural_curves import line

NS = '{http://www.w3.org/2000/svg}'
OPS = re.compile(r'([MLCZ])([^MLCZ]*)')
COLOUR = .027


def sample(curves):
    return np.vstack([curves[0][0], *[flatten(c, .09) for c in curves]])


def uniform(p, step=.75):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-7]
    return np.column_stack([np.interp(np.linspace(0, s[-1], max(3, round(s[-1]/step)+1)),
                                      s[keep], col[keep]) for col in p.T])


def suppress_short_reversals(points):
    """Replace briefly reversed samples when the broader tangents agree."""
    p = uniform(points,.7)
    flank = 7
    repairs = []
    for window in (16,24,32,40):
        middle = np.arange(window+flank,len(p)-window-flank,2)
        if not len(middle):
            continue
        start,end = middle-window,middle+window
        left = p[start]-p[start-flank]
        right = p[end+flank]-p[end]
        delta = p[end]-p[start]
        length = np.linalg.norm(delta,axis=1)
        before = left/np.maximum(np.linalg.norm(left,axis=1)[:,None],1e-8)
        after = right/np.maximum(np.linalg.norm(right,axis=1)[:,None],1e-8)
        axis = delta/np.maximum(length[:,None],1e-8)
        dot = lambda a,b: np.sum(a*b,axis=1)
        eligible = (length>=7)&(dot(before,axis)>.84)&(dot(after,axis)>.84)&(dot(before,after)>.79)
        for k in np.flatnonzero(eligible):
            a,b = start[k],end[k]
            if any(a<=last and first<=b for first,last,_ in repairs):
                continue
            span = p[a:b+1]
            progress = (span-p[a]) @ axis[k]
            if (np.max(np.maximum.accumulate(progress)-progress)<1.2
                    or progress.min()<-.15 or progress.max()>length[k]+.15):
                continue
            normal = axis[k][::-1]*[-1,1]
            across = (span-p[a]) @ normal
            if np.quantile(np.abs(across),.9)>4.1:
                continue
            # A bin contributes once: extra points from a detour must not
            # carry more weight than the rest of the boundary.
            bins = np.floor(np.clip(progress,0,length[k])/1.15).astype(int)
            middle_points = [np.median(span[bins==n],axis=0) for n in np.unique(bins)[1:-1]]
            q = np.vstack([p[a],*middle_points,p[b]])
            q = q[np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-7]]
            fitted = fit_cubic(q,before[k],-after[k],.95)
            candidate = uniform(sample(fitted),.35)
            if ((np.diff(candidate,axis=0) @ axis[k]).min()>0
                    and cKDTree(span).query(candidate)[0].max()<=2.4):
                repairs.append((a,b,candidate))
    if not repairs:
        return points
    pieces = []
    cursor = 0
    for a,b,q in sorted(repairs,key=lambda item:item[0]):
        pieces.extend((p[cursor:a],q))
        cursor = b+1
    return np.vstack([*pieces,p[cursor:]])


def corner_fit(points, closed, tolerance=.55):
    """Use cyclic neighbours at the seam and distinguish a corner from gradual turning."""
    p = uniform(points if closed else suppress_short_reversals(points))
    if closed:
        p = p[:-1]
    n = len(p)
    if n < 8:
        return None
    i = np.arange(n)
    turns = np.ones(n)
    valid = i if closed else i[3:-3]
    for stride in (2, 3):
        a, b = (valid-stride) % n, (valid+stride) % n
        left, right = p[valid]-p[a], p[b]-p[valid]
        dot = (left*right).sum(1)/np.maximum(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1), 1e-8)
        turns[valid] = np.minimum(turns[valid], dot) if stride == 2 else np.maximum(turns[valid], dot)
    stride = 5
    broad = i if closed and n>2*stride else i[stride:-stride]
    left, right = p[broad]-p[(broad-stride)%n], p[(broad+stride)%n]-p[broad]
    agreement = (left*right).sum(1)/np.maximum(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1),1e-8)
    turns[broad[agreement>.72]] = 1.
    corners = []
    for k in np.argsort(turns):
        if turns[k] > .40:
            break
        if all((min(abs(k-j), n-abs(k-j)) if closed else abs(k-j)) > 4 for j in corners):
            corners.append(int(k))
    if closed:
        # At a corner, start from the corner itself. A smooth ring gets a
        # shared tangent across its seam and an antipodal fitting anchor.
        pivot = min(corners) if corners else 0
        corners = [(j-pivot) % n for j in corners]
        p = np.roll(p, -pivot, axis=0)
        p = np.vstack([p, p[0]])
    last = len(p)-1
    breaks = sorted(set([0, last, *corners] + ([last//2] if closed and len(corners)<2 else [])))
    result = []
    for a,b in zip(breaks[:-1], breaks[1:]):
        q = p[a:b+1]
        delta = q[-1]-q[0]
        length = np.linalg.norm(delta)
        if length > 0:
            u = delta/length
            t = (q-q[0]) @ u
            across = np.linalg.norm(q-q[0]-t[:,None]*u,axis=1)
            if np.max(across) <= .26 and np.all(np.diff(t)>=-1e-6):
                result.append(line(q[0],q[-1]))
                continue
        def tangent(k, sign):
            direction = (p[(k+sign) % n]-p[(k-sign) % n] if closed and k % n not in corners
                         else p[min(last,max(0,k+sign))]-p[k])
            return _unit(direction)
        result.extend(fit_cubic(q,tangent(a,1),tangent(b,-1),tolerance))
    return result


@dataclass
class Part:
    owner: object
    d: str
    color: np.ndarray
    points: np.ndarray
    closed: bool
    count: int
    scale: np.ndarray

    def __post_init__(self):
        p = uniform(self.points)
        self.center = p.mean(0)
        _,_,vt = np.linalg.svd(p-self.center,full_matrices=False)
        self.axis = vt[0]
        self.span = float(np.ptp(p @ self.axis))
        self.width = float(np.ptp(p @ vt[1]))
        self.thin = self.closed and self.span>=10 and 1.3<=self.width<=8 and self.span>=4*self.width
        self.chip = 1.3<=self.span<=6 and self.width<=3.5


def parse(owner, d, color, scale):
    points, head = [], None
    curves = []
    for op,value in OPS.findall(d):
        if op=='Z':
            if not np.array_equal(head,points[0]):
                curves.append(line(head,points[0]))
            continue
        q = np.fromstring(value.replace(',',' '),sep=' ').reshape(-1,2)/scale
        if len(q) != (3 if op=='C' else 1):
            return None
        if op=='M':
            points = q
        elif op=='L':
            curves.append(line(head,q[0]))
        else:
            curves.append(np.vstack([head,q]))
        head = q[-1]
    if not curves:
        return None
    p = sample(curves)
    if np.linalg.norm(np.diff(p,axis=0),axis=1).sum() < 1:
        return None
    return Part(owner,d,color,p,d.endswith('Z'),len(curves),scale)


def encode(curves, scale, closed):
    f = 'M%.3f,%.3f' % tuple(curves[0][0]*scale)
    for c in curves:
        if np.max(np.abs(c-line(c[0],c[-1]))) < 1e-6:
            f += 'L%.3f,%.3f' % tuple(c[-1]*scale)
        else:
            f += 'C%.3f,%.3f %.3f,%.3f %.3f,%.3f' % tuple((c[1:]*scale).ravel())
    return f + ('Z' if closed else '')


def tip_pair(a,b):
    """Find facing tips. Overlapping parallel objects are not continuations."""
    direction = a.axis * (1 if (b.center-a.center) @ a.axis>=0 else -1)
    side = direction[::-1]*[-1,1]
    left, right = a.points @ direction, b.points @ direction
    gap = right.min()-left.max()
    if not -.75<=gap<=12 or (not b.chip and abs(direction @ b.axis)<.94):
        return None
    start = a.points[left>=left.max()-.85].mean(0)
    end = b.points[right<=right.min()+.85].mean(0)
    if abs((start-end) @ side)>max(1.1,min(a.width,b.width)*.60):
        return None
    return start,end,direction


def merge_closed(a,b,full):
    tips = tip_pair(a,b)
    if tips is None or np.linalg.norm(a.color-b.color)>COLOUR:
        return None
    start,end,direction = tips
    # Keep evidence local, with a narrow axial corridor and without dilation.
    # A run of incompatible pixels between the tips therefore vetoes a merge.
    height,width = full.shape[:2]
    q = np.vstack([a.points,b.points])
    origin = np.maximum(0,np.floor(q.min(0)-3).astype(int))
    hi = np.minimum((width,height),np.ceil(q.max(0)+3).astype(int))
    x,y = origin
    w,h = hi-origin
    if min(w,h)<2 or w*h>20000:
        return None
    yy,xx = np.mgrid[y:y+h,x:x+w]
    coords = np.stack([xx+.5,yy+.5],axis=-1)
    normal = direction[::-1]*[-1,1]
    midpoint = (start+end)/2
    axial = (coords-midpoint) @ direction
    lateral = (coords-midpoint) @ normal
    lab = full[y:y+h,x:x+w]
    near = np.maximum(np.linalg.norm(lab-a.color,axis=2),np.linalg.norm(lab-b.color,axis=2))<=COLOUR
    corridor = (np.abs(axial)<=np.linalg.norm(end-start)/2+3.5) & (np.abs(lateral)<=max(1.,min(a.width,b.width)*.55))
    _,groups = cv2.connectedComponents((near & corridor).astype(np.uint8),connectivity=4)
    def mask(part):
        raster = np.zeros((h,w),np.uint8)
        cv2.fillPoly(raster,[np.rint((part.points-origin-.5)*16).astype(np.int32)],1,shift=4)
        # Centre samples must lie in the actual SVG region.
        return raster
    ma,mb = mask(a),mask(b)
    cross = (set(groups[ma!=0]) & set(groups[mb!=0])) - {0}
    if len(cross)!=1:
        return None
    bridge = groups==next(iter(cross))
    if np.sum(bridge & ~((ma|mb).astype(bool)))<1:
        return None
    # At finer resolution preserve the vector sides; only the missing bridge
    # comes from full-stage pixel evidence.
    high = np.zeros((h*4,w*4),np.uint8)
    for part in (a,b):
        cv2.fillPoly(high,[np.rint((part.points-origin)*4-.5).astype(np.int32)],1)
    high |= np.repeat(np.repeat(bridge.astype(np.uint8),4,axis=0),4,axis=1)
    contours,_ = cv2.findContours(high,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=1 or len(contours[0])<4:
        return None
    contour = (contours[0][:,0]+.5)/4+origin
    contour = np.vstack([contour,contour[0]])
    curves = corner_fit(contour,True,.40)
    if not curves:
        return None
    new = sample(curves)
    shape = Polygon(new)
    if not shape.is_valid or abs(shape.area-high.sum()/16)>max(2,high.sum()/160):
        return None
    if not geometry_matches(contour,new,.95):
        return None
    return curves,new


def geometry_matches(before,after,error):
    return LineString(after).is_simple and LineString(before).hausdorff_distance(LineString(after))<=error


def open_satellite(guide,route,tree,chip,full):
    if np.linalg.norm(guide.color-chip.color)>.038:
        return False
    p = uniform(chip.points)
    distances, indices = tree.query(p)
    if np.max(distances)>3.10 or np.mean(distances)>2.65:
        return False
    a,b = int(indices.min()),int(indices.max())
    if a<2 or b>len(route)-3 or (b-a)*.35<.60*chip.span:
        return False
    if chip.span>1.8*chip.width and abs(_unit(route[b+2]-route[a-2]) @ chip.axis)<.78:
        return False
    # Check the original full-stage boundary, including its two sides. A
    # route that goes through an unrelated interior colour is not evidence.
    h,w = full.shape[:2]
    lo = np.maximum(0,np.floor(p.min(0)-4).astype(int))
    hi = np.minimum((w,h),np.ceil(p.max(0)+4).astype(int))
    x,y = lo
    x1,y1 = hi
    crop = full[y:y1,x:x1]
    near = np.linalg.norm(crop-chip.color,axis=2)<=.032
    edge = (cv2.dilate(near.astype(np.uint8),np.ones((3,3),np.uint8)) !=
            cv2.erode(near.astype(np.uint8),np.ones((3,3),np.uint8)))
    # The same supported component has to follow the guide all the way
    # through the chip's projected extent, as well as overlap the chip.
    _,ids = cv2.connectedComponents(edge.astype(np.uint8),connectivity=8)
    for component in np.unique(ids[edge]):
        ey,ex = np.where(ids==component)
        q = np.column_stack([ex+.5+x,ey+.5+y])
        local = cKDTree(q)
        if (local.query(route[a:b+1])[0].max()<=1.95
                and local.query(p)[0].mean()<=1.15):
            return True
    return False


def contact_edges(full, start, end, radius, ink, flank=None):
    """Read matching unit boundary edges from a narrow complete-stage corridor."""
    h,w = full.shape[:2]
    lo = np.maximum(0,np.floor(np.minimum(start,end)-radius-1).astype(int))
    hi = np.minimum((w,h),np.ceil(np.maximum(start,end)+radius+1).astype(int))
    x,y = lo
    x1,y1 = hi
    crop = full[y:y1,x:x1]
    ink_distance=np.linalg.norm(crop-ink,axis=2)
    flank_distance=np.linalg.norm(crop-flank,axis=2) if flank is not None else None
    edges=[]
    axis=_unit(end-start)
    length=np.linalg.norm(end-start)
    for horizontal in (True,False):
        a,b = (crop[:-1],crop[1:]) if horizontal else (crop[:,:-1],crop[:,1:])
        da,db = (ink_distance[:-1],ink_distance[1:]) if horizontal else (ink_distance[:,:-1],ink_distance[:,1:])
        eligible = (np.minimum(da,db)<=.045)&(np.linalg.norm(a-b,axis=2)>=.036)
        if flank_distance is not None:
            fa,fb = (flank_distance[:-1],flank_distance[1:]) if horizontal else (flank_distance[:,:-1],flank_distance[:,1:])
            eligible &= ((da<=.045)&(fb<=.065))|((db<=.045)&(fa<=.065))
        for row,col in np.argwhere(eligible):
            point=np.array([x+col,y+row]) + ([0,1] if horizontal else [1,0])
            other=point+([1,0] if horizontal else [0,1])
            relative=(point+other)/2-start
            advance=relative @ axis
            side=np.linalg.norm(relative-advance*axis)
            if -1.7<=advance<=length+1.7 and side<=radius:
                edges.append((tuple(point),tuple(other)))
    return edges


def flank_colour(part):
    value=part.owner.get('data-flank','')
    if re.fullmatch(r'#[0-9a-fA-F]{6}',value):
        return srgb_to_oklab(np.array([list(bytes.fromhex(value[1:]))]))[0]
    return None


def band_satellite(guide, route, tree, chip, full):
    """Require both visible boundary traces and a common transition colour."""
    a,b = flank_colour(guide),flank_colour(chip)
    if a is None or b is None:
        return False
    colours=np.array([guide.color,a,chip.color,b])
    pair_distance=np.linalg.norm(colours[:2,None]-colours[None,2:],axis=2)
    if pair_distance.min()>.025:
        return False
    # Only the local transition band qualifies. A different hue region or
    # steep unrelated contrast between the paths vetoes consolidation.
    shared_left,shared_right=np.unravel_index(pair_distance.argmin(),pair_distance.shape)
    shared=colours[shared_left]
    shoulders=colours[[1-shared_left,3-shared_right]]
    if np.linalg.norm(shoulders-shared,axis=1).min()>.17:
        return False
    samples=uniform(chip.points,.7)
    distance,index=tree.query(samples)
    start,end=int(index.min()),int(index.max())
    if end-start<5 or distance.max()>5 or np.median(distance)>3.6:
        return False
    direction=_unit(route[end]-route[start])
    if abs(chip.axis @ direction)<.90 or (end-start)*.35<.6*chip.span:
        return False
    if start==0 or end==len(route)-1:
        return False
    for u,v,trace in ((guide.color,a,route[start:end+1]),(chip.color,b,samples)):
        edges=contact_edges(full,route[start],route[end],6,u,v)
        if not edges:
            return False
        points=np.asarray(edges).reshape(-1,2)
        if np.quantile(cKDTree(points).query(trace)[0],.9)>1.8:
            return False
    # Sample the band between the two confirmed edges. The third colour
    # may be a nearby quantisation of a flank, never another object.
    h,w=full.shape[:2]
    between=np.stack([samples+(route[index]-samples)*t for t in (.25,.5,.75)])
    x=np.clip(between[...,0].astype(int),0,w-1)
    y=np.clip(between[...,1].astype(int),0,h-1)
    residual=np.linalg.norm(full[y,x][...,None,:]-colours,axis=-1).min(axis=-1)
    return np.quantile(residual,.9)<=.04 and residual.max()<=.075


def open_bridge(a,b):
    start,end=a[-1],b[0]
    gap=np.linalg.norm(end-start)
    if gap<.15:
        return np.empty((0,2))
    if gap>10:
        return None
    direction=_unit(end-start)
    a_dir=_unit(start-a[max(0,len(a)-5)])
    b_dir=_unit(b[min(len(b)-1,4)]-end)
    if min(a_dir @ direction,b_dir @ direction)<.80 or a_dir @ b_dir<.83:
        return None
    return direction


def supported_open_join(a,b,full,ink,flank):
    if LineString(a).intersects(LineString(b)) and np.linalg.norm(a[-1]-b[0])>.15:
        return None
    direction=open_bridge(a,b)
    if direction is None:
        return None
    if not len(direction):
        return np.vstack([a,b[1:]])
    from collections import defaultdict,deque
    start,end=a[-1],b[0]
    graph=defaultdict(list)
    for left,right in contact_edges(full,start,end,1.55,ink,flank):
        graph[left].append(right)
        graph[right].append(left)
    if not graph:
        return None
    nodes=np.asarray(list(graph))
    origins=nodes[np.linalg.norm(nodes-start,axis=1)<=1.45]
    targets=set(map(tuple,nodes[np.linalg.norm(nodes-end,axis=1)<=1.45]))
    previous={tuple(o):None for o in origins}
    todo=deque(previous)
    while todo:
        node=todo.popleft()
        if node in targets:
            bridge=[]
            while node is not None:
                bridge.append(node)
                node=previous[node]
            return np.vstack([a,bridge[::-1],b])
        for neighbor in graph[node]:
            if neighbor not in previous and (np.subtract(neighbor,node) @ direction)>=-.45:
                previous[neighbor]=node
                todo.append(neighbor)
    return None


def join_open_parts(parts, full, removed, replacements, stats):
    indices=[i for i,p in enumerate(parts) if not p.closed and p.span>=3]
    if len(indices)<2:
        return
    endpoints=cKDTree([point for i in indices for point in parts[i].points[[0,-1]]])
    for i in sorted(indices,key=lambda k:-parts[k].span):
        if i in removed or parts[i].span<8:
            continue
        a=parts[i]
        flank=flank_colour(a)
        route=uniform(a.points)
        for _ in range(12):
            accepted=False
            for reverse in (False,True):
                p=route[::-1] if reverse else route
                best=[]
                for key in endpoints.query_ball_point(p[-1],10):
                    j, side=indices[key//2],key%2
                    if j==i or j in removed or j in replacements:
                        continue
                    b=parts[j]
                    if np.linalg.norm(a.color-b.color)>COLOUR:
                        continue
                    other_flank=flank_colour(b)
                    if flank is not None and other_flank is not None and np.linalg.norm(flank-other_flank)>.052:
                        continue
                    q=uniform(b.points if side==0 else b.points[::-1])
                    joined=supported_open_join(p,q,full,a.color,flank)
                    if joined is not None:
                        best.append((np.linalg.norm(p[-1]-q[0]),j,joined))
                best.sort(key=lambda item:(item[0],item[1]))
                if not best or (len(best)>1 and best[1][0]-best[0][0]<.85):
                    continue
                _,j,joined=best[0]
                if reverse:
                    joined=joined[::-1]
                fit=corner_fit(joined,False)
                if not fit or not geometry_matches(joined,sample(fit),1.4):
                    continue
                route=uniform(sample(fit))
                replacements[i]=encode(fit,a.scale,False)
                removed.add(j)
                stats['open_merges']+=1
                accepted=True
                break
            if not accepted:
                break


def refit_outline_layer(svg, full_rgb, *, layer='region-boundaries'):
    """Replace only outline subpaths; the input pre-stop colour field is evidence."""
    root = ET.fromstring(svg)
    group = root.find(f'{NS}g[@id="{layer}"]')
    if group is None:
        raise ValueError('Missing generated outline layer')
    h,w = full_rgb.shape[:2]
    x,y,vw,vh = map(float,root.get('viewBox').split())
    if x or y or group.get('transform'):
        raise ValueError('Outline and evidence must share an origin')
    scale = np.array([vw/w,vh/h])
    palette,ids = np.unique(np.ascontiguousarray(full_rgb).view('V3').ravel(),return_inverse=True)
    full = srgb_to_oklab(palette.view(np.uint8).reshape(-1,3))[ids.reshape(h,w)]
    entries, parts = {}, []
    for owner in group.findall(NS+'path'):
        stroke = owner.get('stroke','')
        d = owner.get('d','')
        if owner.get('data-frame') or not re.fullmatch(r'#[0-9a-fA-F]{6}',stroke) or not re.fullmatch(r'M[MLCZ0-9., \-]+',d) or owner.get('transform'):
            continue
        color = srgb_to_oklab(np.array([list(bytes.fromhex(stroke[1:]))]))[0]
        ids = []
        for raw in re.findall(r'M[^M]+',d):
            p = parse(owner,raw,color,scale)
            ids.append((len(parts) if p else None,raw))
            if p:
                parts.append(p)
        entries[owner] = ids
    removed, replacements = set(), {}
    stats = dict(thin_candidates=sum(p.thin for p in parts), closed_merges=0, isolated_absorbed=0,
                 closed_refits=0, open_refits=0, open_merges=0, examples=[])
    join_open_parts(parts,full,removed,replacements,stats)
    # Some confirmed open paths (such as a recovered long boundary) use
    # their own group. Their geometry stays fixed; tiny matching satellites
    # can be replaced by that route only with support from the full field.
    guides = [(i,p) for i,p in enumerate(parts) if not p.closed and p.span>=10]
    for g in root.findall(NS+'g'):
        if g.get('id')!='recovered-finger-contour':
            continue
        m = re.fullmatch(r'scale\(([0-9.]+)\)',g.get('transform',''))
        if not m:
            continue
        for owner in g.findall(NS+'path'):
            stroke = owner.get('stroke','')
            if re.fullmatch(r'#[0-9a-fA-F]{6}',stroke) and owner.get('d','').count('M')==1:
                color = srgb_to_oklab(np.array([list(bytes.fromhex(stroke[1:]))]))[0]
                p = parse(owner,owner.get('d'),color,scale/float(m[1]))
                if p and not p.closed:
                    guides.append((None,p))
    chips = [i for i,p in enumerate(parts)
             if (p.chip or (p.span<=22 and p.width<=4 and p.span>=3*p.width))
             and (not p.closed or len(entries[p.owner])==1)]
    chip_tree = cKDTree([parts[i].center for i in chips]) if chips else None
    for guide_index,guide in sorted(guides,key=lambda g:-g[1].span):
        if guide_index in removed:
            continue
        if guide_index in replacements:
            guide=parse(guide.owner,replacements[guide_index],guide.color,scale)
        if chip_tree is None:
            break
        route = uniform(guide.points,.35)
        route_tree = cKDTree(route)
        for j in {chips[k] for row in chip_tree.query_ball_point(route,5) for k in row}-removed:
            chip = parts[j]
            if j==guide_index or j in replacements or chip.span>guide.span*.65:
                continue
            if ((chip.chip and open_satellite(guide,route,route_tree,chip,full))
                    or band_satellite(guide,route,route_tree,chip,full)):
                removed.add(j)
                stats['isolated_absorbed'] += 1
                if len(stats['examples'])<40:
                    stats['examples'].append(dict(kept=guide.owner.get('id') or guide.owner.get('stroke'),removed=chip.owner.get('id'),
                        bounds=np.round([chip.points.min(0),chip.points.max(0)],1).tolist(),chip=True))
    # A compound path can contain holes. Do not infer that its individual
    # rings enclose the stroke colour; refit those rings in place only.
    pool = [i for i,p in enumerate(parts) if p.closed and (p.thin or p.chip) and len(entries[p.owner])==1]
    tips = [parts[i].center+s*parts[i].axis*parts[i].span/2 for i in pool for s in (-1,1)]
    tree = cKDTree(tips) if tips else None
    for i in sorted(pool,key=lambda i:-parts[i].span):
        a = parts[i]
        if not a.thin or i in removed:
            continue
        used = 0
        for _ in range(8):
            near = set()
            for sign in (-1,1):
                near.update(tree.query_ball_point(a.center+sign*a.axis*a.span/2,16))
            proposals = []
            for j in {pool[k//2] for k in near}-{i}-removed:
                b = parts[j]
                if j in replacements or (used==0 and b.span>a.span):
                    continue
                pair = tip_pair(a,b)
                if pair is None:
                    continue
                proposals.append((np.linalg.norm(pair[0]-pair[1]),j,b))
            for _,j,b in sorted(proposals,key=lambda p:(p[0],p[1])):
                result = merge_closed(a,b,full)
                if result is None:
                    continue
                curves,points = result
                replacements[i] = encode(curves,scale,True)
                removed.add(j)
                stats['isolated_absorbed'] += int(b.chip)
                stats['closed_merges'] += int(not b.chip)
                if len(stats['examples'])<40:
                    stats['examples'].append(dict(kept=a.owner.get('id'),removed=b.owner.get('id'),
                        bounds=np.round(np.vstack([points.min(0),points.max(0)]),1).tolist(),chip=bool(b.chip)))
                a = Part(a.owner,a.d,a.color,points,True,len(curves),scale)
                used += 1
                break
            else:
                break
    for i,p in enumerate(parts):
        if i in removed or i in replacements or p.count<4:
            continue
        fit_points = p.points if p.closed else suppress_short_reversals(p.points)
        curves = corner_fit(fit_points,p.closed)
        if not curves or (fit_points is p.points and len(curves)>=p.count):
            continue
        after = sample(curves)
        if not geometry_matches(fit_points,after,.80):
            continue
        replacements[i] = encode(curves,scale,p.closed)
        stats['closed_refits' if p.closed else 'open_refits'] += 1
    for owner, ids in entries.items():
        d = ''.join(replacements.get(i,raw) for i,raw in ids if i not in removed)
        if d:
            owner.set('d',d)
        else:
            group.remove(owner)
    ET.register_namespace('',NS[1:-1])
    return ET.tostring(root,encoding='unicode'),stats


