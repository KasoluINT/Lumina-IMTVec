"""Reject fitted shared boundaries that cross or touch away from junctions."""
import numpy as np
from shapely import LineString,Point,STRtree
from .contour import fit_cubic,_unit

def flatten(c,tolerance=.015,depth=0):
    if np.max(np.abs(c-c[0]))<1e-12:return [c[-1]]
    chord=c[-1]-c[0];length=np.linalg.norm(chord)
    distance=np.max(np.abs((c[1:-1,0]-c[0,0])*chord[1]-(c[1:-1,1]-c[0,1])*chord[0]))/max(length,1e-12)
    if depth>=16 or (length>0 and distance<=tolerance and np.all((c[1:-1]-c[0])@chord>=0) and np.all((c[1:-1]-c[-1])@chord<=0)):
        return [c[-1]]
    a=(c[:-1]+c[1:])/2;b=(a[:-1]+a[1:])/2;m=(b[0]+b[1])/2
    return flatten(np.array([c[0],a[0],b[0],m]),tolerance,depth+1)+flatten(np.array([m,b[1],a[2],c[3]]),tolerance,depth+1)


def _flatten_chain_fast(curves, tolerance=0.015):
    """Exact iterative vectorized equivalent of concatenating [curves[0][0]] + sum(flatten(round(c, 3)))."""
    if not curves:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.round(np.asarray(curves, dtype=np.float64), 3)  # (N, 4, 2)
    n_c = len(arr)
    # Track segment indices iteratively in left-to-right DFS order:
    # Start with [0, 1, ..., n_c - 1]
    cur = arr
    for _depth in range(16):
        c0 = cur[:, 0]
        c1 = cur[:, 1]
        c2 = cur[:, 2]
        c3 = cur[:, 3]
        tiny = np.max(np.abs(cur - c0[:, None, :]), axis=(1, 2)) < 1e-12
        chord = c3 - c0
        length = np.hypot(chord[:, 0], chord[:, 1])
        denom = np.maximum(length, 1e-12)
        d1 = np.abs((c1[:, 0] - c0[:, 0]) * chord[:, 1] - (c1[:, 1] - c0[:, 1]) * chord[:, 0])
        d2 = np.abs((c2[:, 0] - c0[:, 0]) * chord[:, 1] - (c2[:, 1] - c0[:, 1]) * chord[:, 0])
        dist = np.maximum(d1, d2) / denom
        dot1a = (c1[:, 0] - c0[:, 0]) * chord[:, 0] + (c1[:, 1] - c0[:, 1]) * chord[:, 1]
        dot2a = (c2[:, 0] - c0[:, 0]) * chord[:, 0] + (c2[:, 1] - c0[:, 1]) * chord[:, 1]
        dot1b = (c1[:, 0] - c3[:, 0]) * chord[:, 0] + (c1[:, 1] - c3[:, 1]) * chord[:, 1]
        dot2b = (c2[:, 0] - c3[:, 0]) * chord[:, 0] + (c2[:, 1] - c3[:, 1]) * chord[:, 1]
        done = tiny | (
            (length > 0.0)
            & (dist <= tolerance)
            & (dot1a >= 0.0)
            & (dot2a >= 0.0)
            & (dot1b <= 0.0)
            & (dot2b <= 0.0)
        )
        if done.all():
            break
        split_idx = np.flatnonzero(~done)
        s = cur[split_idx]
        a01 = 0.5 * (s[:, 0] + s[:, 1])
        a12 = 0.5 * (s[:, 1] + s[:, 2])
        a23 = 0.5 * (s[:, 2] + s[:, 3])
        b0 = 0.5 * (a01 + a12)
        b1 = 0.5 * (a12 + a23)
        m = 0.5 * (b0 + b1)
        left = np.stack([s[:, 0], a01, b0, m], axis=1)
        right = np.stack([m, b1, a23, s[:, 3]], axis=1)
        # Interleave unchanged `done` items and split (left, right) pairs preserving exact order
        m_cur = len(cur)
        m_split = len(split_idx)
        nxt = np.empty((m_cur + m_split, 4, 2), dtype=np.float64)
        # Position offset of each original index in `nxt`: index i shifts by count(split_idx < i)
        is_split = ~done
        shift = np.zeros(m_cur, dtype=np.int32)
        if m_cur > 1:
            shift[1:] = np.cumsum(is_split[:-1])
        kept_idx = np.flatnonzero(done)
        nxt[kept_idx + shift[kept_idx]] = cur[kept_idx]
        sp_pos = split_idx + shift[split_idx]
        nxt[sp_pos] = left
        nxt[sp_pos + 1] = right
        cur = nxt

    out = np.empty((len(cur) + 1, 2), dtype=np.float64)
    out[0] = arr[0, 0]
    out[1:] = cur[:, 3]
    return out

def _linear(points):
    changes=np.flatnonzero(np.any(np.diff(points,n=2,axis=0)!=0,axis=1))+1
    p=points[np.r_[0,changes,len(points)-1]]
    return [np.array([a,a+(b-a)/3,a+(b-a)*2/3,b]) for a,b in zip(p[:-1],p[1:])]

def _refit(points):
    if len(points)<=2:return _linear(points)
    if np.array_equal(points[0],points[-1]):
        k=len(points)//2
        return _refit(points[:k+1])+_refit(points[k:])
    return fit_cubic(points,_unit(points[1]-points[0]),_unit(points[-2]-points[-1]),.12)

def guard(chains, prior=None, cached=None):
    refitted=set();reverted=set();history=[]
    if cached is None:
        cached={}
    local_attempts={}
    end_pts = [set(map(tuple, c['points'][[0, -1]])) for c in chains]
    for iteration in range(8):
        lines=[]
        changed_set = None
        if prior is not None:
            changed_set = {i for i, c in enumerate(chains) if c['curves'] is not prior[i]}
        for index,chain in enumerate(chains):
            if changed_set is not None and index in changed_set:
                cached.pop(index, None)
            if index not in cached:
                cached[index]=LineString(_flatten_chain_fast(chain['curves']))
            lines.append(cached[index])
        check_indices = range(len(lines)) if changed_set is None else changed_set
        bad={i for i in check_indices if not lines[i].is_simple}
        contacts={}
        tree=STRtree(lines)
        for i in check_indices:
            line = lines[i]
            for j in tree.query(line,predicate='intersects'):
                j_int = int(j)
                if changed_set is None and j_int <= i:
                    continue
                if changed_set is not None and j_int in changed_set and j_int <= i:
                    continue
                if j_int == i:
                    continue
                hit=line.intersection(lines[j_int])
                common=end_pts[i]&end_pts[j_int]
                points=list(hit.geoms) if hit.geom_type=='MultiPoint' else [hit]
                allowed=all(p.geom_type=='Point' and any(p.distance(Point(q))<1e-7 for q in common) for p in points)
                if not allowed:
                    bad.update((i,j_int))
                    if prior is None:
                        for k in (i,j_int):
                            if 'relocation' in chains[k]:
                                p=hit.representative_point()
                                if not p.is_empty:
                                    contacts.setdefault(k,[]).append((p.x,p.y))
        if prior is not None:
            # The initial graph already passed validation. Only roll back
            # proposed curves; keep valid neighbours at their prior geometry.
            rollback = {i for i in bad if chains[i]['curves'] is not prior[i]}
            history.append(dict(iteration=iteration, invalid_chains=len(bad), rollbacks=len(rollback)))
            if not bad:
                return dict(geometry_guard_refitted=0, geometry_guard_reverted=len(reverted),
                            geometry_guard_history=history)
            if not rollback:
                raise ValueError('The initial boundary geometry is invalid')
            for i in rollback:
                chains[i]['curves'] = prior[i]
                cached.pop(i,None)
            reverted.update(rollback)
            continue
        repaired=set()
        for i, points in contacts.items():
            chain=chains[i]
            if i in refitted or not lines[i].is_simple or local_attempts.get(i,0)>=2:
                continue
            proposed,fitter,scale=chain['relocation']
            original=chain['points']
            stations=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(original,axis=0),axis=1))]
            weight=np.ones(len(original))
            # Restore only the neighbourhood of each collision. Unaffected
            # spans can keep the placement supported by the source image.
            for point in points:
                index=np.argmin(np.sum((proposed-point)**2,axis=1))
                distance=np.abs(stations-stations[index])/scale
                weight=np.minimum(weight,np.clip((distance-4)/14,0,1))
            adjusted=original+(proposed-original)*weight[:,None]
            chain['relocation']=(adjusted,fitter,scale)
            chain['curves']=fitter(adjusted,scale)
            cached.pop(i,None)
            local_attempts[i]=local_attempts.get(i,0)+1
            repaired.add(i)
        new=bad-refitted-repaired;fallback=(bad&refitted)-reverted
        history.append(dict(iteration=iteration,invalid_chains=len(bad),new_refits=len(new),new_reversions=len(fallback)))
        if not bad:return dict(geometry_guard_refitted=len(refitted),geometry_guard_reverted=len(reverted),geometry_guard_history=history)
        if not new and not fallback and not repaired:raise ValueError(f'Unresolved boundary crossings: {len(bad)}')
        for i in new:chains[i]['curves']=_refit(chains[i]['points']);cached.pop(i,None)
        for i in fallback:chains[i]['curves']=_linear(chains[i]['points']);cached.pop(i,None)
        refitted.update(new);reverted.update(fallback)
    if prior is not None:
        for i,chain in enumerate(chains):
            if chain['curves'] is not prior[i]:
                reverted.add(i)
                chain['curves'] = prior[i]
        return dict(geometry_guard_refitted=0, geometry_guard_reverted=len(reverted),
                    geometry_guard_history=history)
    raise ValueError('Boundary guard did not converge')
