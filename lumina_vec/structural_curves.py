"""Fit geometric contours at source scale instead of following every grid step."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .bezier_fit import fit_cubic, _unit, _fit_one, _chord_params
from scipy.spatial import cKDTree


def line(a,b):
    return np.array([a,a+(b-a)/3,a+2*(b-a)/3,b])


def fit_structural(points,scale,*,smooth_lines=False):
    points=np.asarray(points,float)
    closed=np.array_equal(points[0],points[-1])
    if len(points)<=2:return [line(points[0],points[-1])]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if distance[-1]==0:return [line(points[0],points[-1])]
    count=max(3,int(np.ceil(distance[-1]/scale))+1)
    at=np.linspace(0,distance[-1],count)
    p=np.column_stack([np.interp(at,distance,points[:,c]) for c in (0,1)])
    # Only a complete, already near-circular loop can use this primitive. Partial
    # irises and occluded arcs retain their junctions and use constrained Beziers.
    if closed and len(p)>=12:
        q=p[:-1];center0=q.mean(axis=0);v=q-center0
        solution=np.linalg.lstsq(np.column_stack([2*v,np.ones(len(v))]),np.sum(v*v,axis=1),rcond=None)[0]
        center=center0+solution[:2];radii=np.linalg.norm(q-center,axis=1);radius=np.mean(radii)
        residual=np.max(np.abs(radii-radius))
        angles=np.unwrap(np.arctan2(p[:,1]-center[1],p[:,0]-center[0]));sweep=angles[-1]-angles[0]
        if radius>2*scale and residual<.45*scale and abs(abs(sweep)-2*np.pi)<.05:
            # Use the exact graph start/end point, not a displaced circle point.
            start=angles[0];direction=1 if sweep>0 else -1;curves=[]
            for i in range(4):
                a=start+direction*i*np.pi/2;b=a+direction*np.pi/2
                u=np.array([np.cos(a),np.sin(a)]);w=np.array([np.cos(b),np.sin(b)])
                k=direction*4/3*np.tan(np.pi/8)*radius
                curves.append(np.array([center+radius*u,center+radius*u+k*np.array([-u[1],u[0]]),center+radius*w-k*np.array([-w[1],w[0]]),center+radius*w]))
            curves[0][0]=points[0];curves[-1][-1]=points[-1]
            return curves
    sigma = 1.25 if smooth_lines else 1.0
    step = 3
    dot_thr = .45
    line_tol = (.45 if smooth_lines else .35) * scale
    fit_tol = (.70 if smooth_lines else .55) * scale
    sm=gaussian_filter1d(p,sigma,axis=0,mode='wrap' if closed else 'nearest')
    sm[[0,-1]]=points[[0,-1]]
    # Detect sustained corners several source pixels apart, not grid stair corners.
    corners=[]
    for i in range(step,len(p)-step):
        left=_unit(p[i]-p[i-step]);right=_unit(p[i+step]-p[i])
        if left@right<dot_thr:
            if not corners or i-corners[-1]>step:corners.append(i)
    stops=sorted(set([0,len(p)-1]+corners+([len(p)//2] if closed else [])))
    for i in corners:sm[i]=p[i]
    result=[]
    for a,b in zip(stops[:-1],stops[1:]):
        piece=sm[a:b+1];delta=piece[-1]-piece[0];length=np.linalg.norm(delta)
        if length>0:
            across=np.abs((piece[:,0]-piece[0,0])*delta[1]-(piece[:,1]-piece[0,1])*delta[0])/length
            along=(piece-piece[0])@delta/(length*length)
            if across.max()<=line_tol and along.min()>=0 and along.max()<=1:
                result.append(line(piece[0],piece[-1]));continue
        result.extend(fit_cubic(piece,_unit(piece[1]-piece[0]),_unit(piece[-2]-piece[-1]),fit_tol))
    return result


def simple_span(points, scale, *, smooth_lines=False):
    start, end = points[[0, -1]]
    chord = end - start
    length = np.linalg.norm(chord)
    if length < 10 * scale:
        return None
    tangent = chord / length
    normal = tangent[::-1] * [-1, 1]
    local = points - start
    along = local @ tangent
    across = local @ normal
    if along.min() < 0 or along.max() > length + 1e-8:
        return None
    if np.max(np.maximum.accumulate(along) - along) > .45 * scale:
        return None
    if np.max(np.abs(across)) <= .90 * scale:
        return line(start, end), 'line'

    step = np.linspace(0, length, max(6, round(length / scale) + 1))
    order = np.argsort(along, kind='stable')
    height = np.interp(step, along[order], across[order])
    if len(step) >= 9:
        span = 4
        direction = np.arctan2(height[span:] - height[:-span], step[span:] - step[:-span])
        if np.max(np.abs(direction[span:] - direction[:-span])) > .67:
            return None

    t = step / length
    # Prefer a single quadratic for a steady turn. Its cubic elevation keeps
    # it compatible with the shared-edge guard and reverse traversal.
    q_basis = 2*t*(1-t)
    q_offset = (q_basis @ height) / (q_basis @ q_basis)
    u = along / length
    q_error = np.abs(across - 2*u*(1-u)*q_offset)
    q_max = (.95 if smooth_lines else .83) * scale
    q_rms = (.45 if smooth_lines else .40) * scale
    if (abs(q_offset) <= length*.32 and q_error.max() <= q_max
            and np.sqrt(np.mean(q_error**2)) <= q_rms):
        q = (start+end)/2 + q_offset*normal
        return np.array([start,(start+2*q)/3,(end+2*q)/3,end]), 'quadratic'
    basis = 3 * np.column_stack(((1-t)**2*t, (1-t)*t**2))
    offsets = np.linalg.lstsq(basis, height, rcond=None)[0]
    # Monotone along the chord by construction, with at most one inflection.
    if np.max(np.abs(np.diff(np.r_[0., offsets, 0.]))) > length * .25:
        return None
    t = along / length
    predicted = 3*t*(1-t)*((1-t)*offsets[0] + t*offsets[1])
    error = np.abs(across - predicted)
    c_max = (1.25 if smooth_lines else 1.15) * scale
    c_rms = (.62 if smooth_lines else .57) * scale
    if error.max() > c_max or np.sqrt(np.mean(error**2)) > c_rms:
        return None
    controls = line(start, end)
    controls[1:3] += offsets[:, None] * normal
    return controls, 'cubic'


def fit_compact_outline(points, scale, *, smooth_lines=False):
    curves = fit_structural(points, scale, smooth_lines=smooth_lines)
    if len(curves) > 1:
        compact = simple_span(points, scale, smooth_lines=smooth_lines)
        if compact is not None:
            return [compact[0]]
    return curves


def _sample_cubic(curve, step):
    length = np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()
    t = np.linspace(0, 1, max(5, int(np.ceil(length / step)) + 1))
    basis = np.column_stack(((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3))
    return basis @ curve


def _persistent_corner(points, index, *, smooth_lines=False):
    """A true corner is visible on both short and long baselines."""
    n = len(points)
    if not 2 <= index < n-2:
        return False
    checks = ((3, .70), (6, .60)) if smooth_lines else ((2, .73), (5, .64))
    for radius, threshold in checks:
        before = _unit(points[index]-points[max(0,index-radius)])
        after = _unit(points[min(n-1,index+radius)]-points[index])
        if before @ after > threshold:
            return False
    return True


def _station_fit(points, first, last, *, smooth_lines=False):
    """Fit one arc by forward stations, keeping the existing end tangents."""
    vector = points[-1]-points[0]
    length = np.linalg.norm(vector)
    if length < 2.8:
        return None
    direction = vector/length
    # Protect turning tips and tight hairpins.
    if min(direction @ first, direction @ last) < .44:
        return None
    progress=(points-points[0]) @ direction
    if (progress.min()<-.29 or progress.max()>length+.29
            or np.max(np.maximum.accumulate(progress)-progress)>.85):
        return None
    if max(abs((points-points[0]) @ (direction[::-1]*[-1,1])))>max(2.1,length*.27):
        return None
    # Distinct stations give a short grid oscillation the same vote as a
    # straight span. Each end has a fixed point and tangent direction.
    t=np.clip(progress/length,0,1)
    bins=np.floor(t*max(5,round(length/.65))).astype(int)
    ord_b=np.argsort(bins, kind='stable')
    sb=bins[ord_b]; sp=points[ord_b]
    splits=np.r_[0, np.flatnonzero(sb[1:]!=sb[:-1])+1, len(sb)]
    q=np.empty((len(splits)-1, 2), dtype=np.float64)
    for k_b in range(len(splits)-1):
        s0, s1 = splits[k_b], splits[k_b+1]
        if s1 - s0 == 1:
            q[k_b] = sp[s0]
        elif s1 - s0 == 2:
            q[k_b] = 0.5 * (sp[s0] + sp[s0+1])
        else:
            q[k_b] = np.median(sp[s0:s1], axis=0)
    if len(q)<4:
        return None
    q = np.vstack([points[0], q[1:-1], points[-1]])
    control = _fit_one(q, _chord_params(q), first, -last)
    steps=np.diff(control,axis=0) @ direction
    if steps.min()<-1e-8:
        return None
    fitted=_sample_cubic(control,.28)
    if len(points) * len(fitted) <= 16384:
        dx = points[:, None, 0] - fitted[None, :, 0]
        dy = points[:, None, 1] - fitted[None, :, 1]
        d2 = dx * dx + dy * dy
        forward = np.sqrt(d2.min(axis=1))
        backward = np.sqrt(d2.min(axis=0))
    else:
        forward=cKDTree(fitted).query(points)[0]
        backward=cKDTree(points).query(fitted)[0]
    f_max = 1.18 if smooth_lines else 1.04
    f_q90 = .72 if smooth_lines else .65
    b_max = 1.05 if smooth_lines else .95
    if (forward.max()>f_max or np.quantile(forward, .9)>f_q90
            or backward.max()>b_max):
        return None
    return control


def refit_sketch_chain(curves, scale, *, smooth_lines=False):
    """Fit compatible spans of one initial shared curve without moving junctions."""
    original = [curve / scale for curve in curves]
    repaired_geometry = False
    samples = [_sample_cubic(curve,.40) for curve in original]
    points = np.vstack([samples[0], *[s[1:] for s in samples[1:]]])
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if distance[-1]<3.2:
        return curves
    from .sketch_refit import corner_fit, geometry_matches, sample, suppress_short_reversals, uniform
    if not np.array_equal(points[0],points[-1]):
        repaired=suppress_short_reversals(points)
        if repaired is not points:
            smooth=corner_fit(repaired,False)
            if smooth and geometry_matches(repaired,sample(smooth),.8):
                repaired_geometry = True
                original=smooth
                samples=[_sample_cubic(curve,.40) for curve in smooth]
                points=np.vstack([samples[0],*[s[1:] for s in samples[1:]]])
    if len(original)<2:
        return [curve*scale for curve in original]
    stops=np.r_[0,np.cumsum([len(s)-1 for s in samples])]
    spaced=uniform(points,.55)
    nearby=cKDTree(spaced).query(points[stops])[1]
    corners={i for i in range(1,len(stops)-1) if _persistent_corner(spaced,nearby[i],smooth_lines=smooth_lines)}
    max_span = 22 if smooth_lines else 18
    result=[]
    i=0
    while i<len(original):
        limit=min(len(original),i+max_span)
        limit=next((k for k in range(i+1,limit) if k in corners),limit)
        for j in range(limit,i+1,-1):
            span=points[stops[i]:stops[j]+1]
            if len(span)<7:
                continue
            first=_unit(original[i][1]-original[i][0])
            last=_unit(original[j-1][3]-original[j-1][2])
            control=None
            if smooth_lines:
                chord=span[-1]-span[0];clen=np.linalg.norm(chord)
                if clen>0:
                    across=np.abs((span[:,0]-span[0,0])*chord[1]-(span[:,1]-span[0,1])*chord[0])/clen
                    along=(span-span[0])@chord/(clen*clen)
                    if across.max()<=.42 and along.min()>=-.02 and along.max()<=1.02:
                        control=line(span[0],span[-1])
            if control is None:
                control=_station_fit(span,first,last,smooth_lines=smooth_lines)
            # Keep the previous fitter as a fallback if the joint directions
            # allow its straight or gently curved solution.
            if control is None:
                simple=simple_span(span,.92 if smooth_lines else .86,smooth_lines=smooth_lines)
                if simple is not None:
                    candidate=simple[0]
                    if min(_unit(candidate[1]-candidate[0]) @ first,
                           _unit(candidate[3]-candidate[2]) @ last)>(.91 if smooth_lines else .94):
                        control=candidate
            if control is not None:
                result.append(control*scale)
                i=j
                break
        else:
            result.append(original[i]*scale)
            i+=1
    return result if repaired_geometry or len(result)<len(curves) else curves


def _is_straight_span(c, scale):
    chord = c[3] - c[0]
    length = np.linalg.norm(chord)
    if length < 5.0 * scale:
        return False
    u = chord / length
    n = np.array([-u[1], u[0]])
    for p in (c[1], c[2]):
        d_chord = abs((p - c[0]) @ n)
        t = ((p - c[0]) @ u) / length
        if d_chord > .30 * scale or t < -.05 or t > 1.05:
            return False
    return True


def _smooth_handles(a, b, axis, scale, *, smooth_lines=False):
    """Adjust only the handles on the joined endpoint when the change is local."""
    span_a = np.linalg.norm(a[3]-a[0])
    span_b = np.linalg.norm(b[3]-b[0])
    if min(span_a,span_b)<.43*scale:
        return None
    if min(_unit(a[3]-a[0]) @ axis,_unit(b[3]-b[0]) @ axis)<.52:
        return None
    # Never bend long straight spans (e.g. clock hands) into bowed curves at junctions.
    if _is_straight_span(a, scale) or _is_straight_span(b, scale):
        return None
    ca,cb = a.copy(),b.copy()
    for curve,i,anchor,direction,length in (
            (ca,2,3,-axis,span_a),(cb,1,0,axis,span_b)):
        arm = np.linalg.norm(curve[i]-curve[anchor])
        arm = np.clip(arm,.19*length,.44*length)
        curve[i] = curve[anchor] + direction*arm
    # For one moved control point the curve displacement is at most 4/9
    # of the handle shift. Keep the fitted edge in its original corridor.
    limit = min(2.15 * scale, .25 * min(span_a, span_b))
    if max(np.linalg.norm(ca[2]-a[2]),np.linalg.norm(cb[1]-b[1])) > limit:
        return None
    return ca,cb


def smooth_outline_junctions(chains, scale, *, smooth_lines=False):
    """Continue the strongest tangent through junctions without moving any vertex."""
    from collections import defaultdict
    endpoints = defaultdict(list)
    updates = {}

    def curves_at(index):
        return updates.get(index,chains[index]['curves'])

    for index,chain in enumerate(chains):
        if min(chain['left'],chain['right'])<0:
            continue
        curves = chain['curves']
        # A short alternating turn has agreeing upstream and downstream
        # directions, unlike genuine corners and turning tips.
        for j in range(1,len(curves)):
            a,b = curves_at(index)[j-1:j+1]
            ta,tb = _unit(a[3]-a[2]),_unit(b[1]-b[0])
            if ta @ tb>.985:
                continue
            previous=_unit(a[3]-a[0])
            following=_unit(b[3]-b[0])
            if previous @ following<(.70 if smooth_lines else .73):
                continue
            joined=_smooth_handles(a,b,_unit(previous+following),scale,smooth_lines=smooth_lines)
            if joined is not None:
                editable=updates.setdefault(index,list(curves))
                editable[j-1],editable[j]=joined
        points=chain['points']
        if chain['vertices'][0]==chain['vertices'][-1]:
            a,b=curves_at(index)[-1],curves_at(index)[0]
            first,last=_unit(a[3]-a[0]),_unit(b[3]-b[0])
            if len(curves)>1 and first @ last>.76:
                joined=_smooth_handles(a,b,_unit(first+last),scale,smooth_lines=smooth_lines)
                if joined is not None:
                    editable=updates.setdefault(index,list(curves))
                    editable[-1],editable[0]=joined
            continue
        for side in (0,-1):
            span=points if side==0 else points[::-1]
            reach=min(len(span)-1,max(2,round(3*scale)))
            chord=span[reach]-span[0]
            if np.linalg.norm(chord)>=1.3*scale:
                endpoints[chain['vertices'][side]].append((index,side,_unit(chord)))

    for items in endpoints.values():
        if len(items)<2:
            continue
        pairs=sorted((float(a[2] @ b[2]),k,l)
                     for k,a in enumerate(items) for l,b in enumerate(items) if k<l)
        similarity,i,j=pairs[0]
        if similarity>-.76 or (len(pairs)>1 and pairs[1][0]-similarity<.11):
            continue
        first,sa,da=items[i]
        second,sb,db=items[j]
        incoming=curves_at(first)[sa]
        outgoing=curves_at(second)[sb]
        a=incoming[::-1] if sa==0 else incoming
        b=outgoing[::-1] if sb==-1 else outgoing
        joined=_smooth_handles(a,b,_unit(db-da),scale,smooth_lines=smooth_lines)
        if joined is None:
            continue
        ca,cb=joined
        updates.setdefault(first,list(chains[first]['curves']))[sa]=ca[::-1] if sa==0 else ca
        updates.setdefault(second,list(chains[second]['curves']))[sb]=cb[::-1] if sb==-1 else cb

    for i,curves in updates.items():
        chains[i]['curves']=curves

