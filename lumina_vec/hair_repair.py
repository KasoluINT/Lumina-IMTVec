"""Conservative reconstruction of long, thin light and dark strokes."""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .draw import _resample_stroke, stroke_outline_d
from .strokes2 import build_strokes


def _sample(field, points):
    return ndi.map_coordinates(field, [points[:, 1] - 0.5, points[:, 0] - 0.5],
                               order=1, mode="nearest")


def _regional_strokes(items, rgb):
    """Split only persistent colour steps; smooth shading and pixel noise stay connected."""
    for strand, item in enumerate(items):
        points = _resample_stroke(item['xy'])
        colors = np.stack([_sample(rgb[:,:,c], points) for c in range(3)],axis=1)
        smooth = ndi.median_filter(colors,size=(5,1),mode='nearest')
        candidates = []
        for i in np.flatnonzero(np.max(np.abs(np.diff(smooth,axis=0)),axis=1)>8):
            if i < 18 or i > len(points)-20:
                continue
            left, right = smooth[i-12:i-2], smooth[i+3:i+13]
            a,b = np.median(left,axis=0),np.median(right,axis=0)
            step = float(np.max(np.abs(a-b)))
            scatter = max(np.percentile(np.abs(left-a),85),np.percentile(np.abs(right-b),85))
            if step >= 28 and scatter < min(12,step/4):
                candidates.append((step/(1+scatter),int(i)))
        cuts = []
        for _,i in sorted(candidates,reverse=True):
            if all(abs(i-j)>=24 for j in cuts):
                cuts.append(i)
        cuts.sort()
        start = 0
        for region,end in enumerate(cuts+[len(points)-1]):
            segment = points[start:end+1]
            if region:
                segment = np.vstack([(points[start-1]+points[start])/2,segment])
            if end < len(points)-1:
                segment = np.vstack([segment,(points[end]+points[end+1])/2])
            yield dict(item,xy=segment,strand=strand,region=region,regions=len(cuts)+1,
                       free=(item['free'][0] if region==0 else False,
                             item['free'][1] if end==len(points)-1 else False))
            start=end+1


def repair_hair(rgb, allow_gradients=True):
    """Return the cleared background, separate stroke paths and a diagnostic edit mask."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    rgb_float = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    paths, records = [], []
    erase = np.zeros(gray.shape, np.uint8)
    owners = np.full(gray.shape,-1,np.int32)
    background = rgb.copy()
    for polarity, operation, sign in (("dark", cv2.MORPH_BLACKHAT, -1),
                                      ("bright", cv2.MORPH_TOPHAT, 1)):
        response = cv2.morphologyEx(gray, operation, kernel)
        peak = cv2.dilate(response, np.ones((5, 5), np.uint8))
        field = np.where(response >= 7, response / np.maximum(peak, 16), 0).astype(np.float32)
        mask = ((field > 0.45) & (response > 10)).astype(np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        candidates = build_strokes(field, ss=1, min_len=12, bridge_px=2, continuity=True)
        for item in _regional_strokes(candidates, rgb_float):
            owner = item['strand']*2+(sign>0)
            raw = _resample_stroke(item["xy"])
            length = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
            if length < (18 if item['regions']>1 else 35) or np.linalg.norm(raw[0] - raw[-1]) < 4:
                continue
            points = ndi.gaussian_filter1d(raw, 0.8, axis=0, mode="nearest")
            points[[0, -1]] = raw[[0, -1]]
            tangent = np.gradient(points, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)[:, None]
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
            offsets = np.arange(-4, 4.01, 0.25)
            profile = np.stack([_sample(gray, points + normal * off) for off in offsets], axis=1)
            side_gray = (profile[:, 0] + profile[:, -1]) / 2
            weights = np.maximum(sign * (profile-side_gray[:, None]), 0)
            shift = (weights*offsets).sum(axis=1)/np.maximum(weights.sum(axis=1),1e-6)
            points += normal * ndi.gaussian_filter1d(np.clip(shift,-1,1),1)[:, None]
            width = 2 * _sample(distance, points)
            if np.median(width) > 4.5 or np.percentile(width, 90) > 7:
                continue
            center = _sample(gray, points)
            sides = np.stack([_sample(gray, points + normal * offset) for offset in (-4.5, 4.5)])
            contrast = sign * (center - sides)
            if np.mean(np.min(contrast, axis=0) > 9) < 0.78:
                continue
            colors = np.stack([_sample(rgb_float[:, :, c], points) for c in range(3)], axis=1)
            variation = np.max(np.percentile(colors, 90, axis=0) - np.percentile(colors, 10, axis=0))
            chord = points[-1]-points[0]
            projection = (points-points[0]) @ chord / np.dot(chord,chord)
            gradient = allow_gradients and variation > 25 and np.min(np.diff(projection)) > -0.002
            if variation > 65 and not gradient:
                continue
            ink = center <= np.percentile(center, 40) if sign < 0 else center >= np.percentile(center, 60)
            color = np.median(colors[ink], axis=0)
            line_colors = ndi.gaussian_filter1d(colors, 2, axis=0) if gradient else None
            if gradient:
                # A single fill gradient preserves colour changes without splitting the path.
                stops = np.unique(np.r_[0, np.arange(8,len(points)-1,8), len(points)-1])
                stops = stops[np.r_[True,np.diff(projection[stops])>1e-5]]
                line_colors = np.stack([np.interp(projection,projection[stops],line_colors[stops,c]) for c in range(3)],axis=1)
            # Integrate the antialiased cross section rather than thresholding its width.
            samples = np.stack([_sample(gray, points + normal * off) for off in offsets], axis=1)
            side_gray = sides.mean(axis=0)
            ink_gray = (line_colors if gradient else color) @ np.array([0.299,0.587,0.114])
            if gradient:
                ink_gray = ink_gray[:,None]
            # Integrated opacity preserves the weight of subpixel/antialiased hairs.
            opacity = np.clip(sign*(samples-side_gray[:,None]) /
                              np.maximum(sign*(ink_gray-side_gray[:,None]),1),0,1)
            measured = opacity.sum(axis=1)*0.25
            width = ndi.gaussian_filter1d(np.clip(measured, 0.45, 5), 1.4)
            # Keep the original end widths; free tips taper over only two pixels.
            for end, free in enumerate(item["free"]):
                if free:
                    if end == 0:
                        width[:3] *= [0.0, 0.6, 1.0]
                    else:
                        width[-3:] *= [1.0, 0.6, 0.0]
            shape = stroke_outline_d(points, width, item["free"], tol=0.2, allow_two_points=True)
            if not shape:
                continue
            local = np.zeros_like(erase)
            poly = np.vstack([points + normal * (width[:, None] / 2 + 0.8),
                              (points - normal * (width[:, None] / 2 + 0.8))[::-1]])
            cv2.fillPoly(local, [np.round((poly - 0.5) * 16).astype(np.int32)], 1, shift=4)
            # Clear only pixels supported by the source ridge.
            local &= (response > 4).astype(np.uint8)
            if np.any((local > 0) & (erase > 0) & (owners != owner)):
                # Ambiguous crossings remain in the exact background in this first pass.
                continue
            # Check each replacement against its source before committing the erase.
            # This rejects boundaries, crossings and wide highlights that resemble ridges.
            ys, xs = np.nonzero(local)
            if not len(xs):
                continue
            x0, x1 = max(0, xs.min()-6), min(rgb.shape[1], xs.max()+7)
            y0, y1 = max(0, ys.min()-6), min(rgb.shape[0], ys.max()+7)
            patch = rgb[y0:y1, x0:x1]
            patch_mask = local[y0:y1, x0:x1]
            cleared = cv2.inpaint(patch, patch_mask, 3, cv2.INPAINT_TELEA)
            polygon = np.vstack([points + normal * width[:, None] / 2,
                                 (points - normal * width[:, None] / 2)[::-1]])
            alpha = np.zeros((patch.shape[0]*8, patch.shape[1]*8), np.uint8)
            coords = (polygon - [x0, y0]) * 8 - 0.5
            cv2.fillPoly(alpha, [np.round(coords * 16).astype(np.int32)], 255, shift=4)
            alpha = cv2.resize(alpha, (patch.shape[1],patch.shape[0]), interpolation=cv2.INTER_AREA)/255.0
            if gradient:
                py,px = np.indices(patch.shape[:2])
                uv = ((px+x0+.5-points[0,0])*chord[0]+(py+y0+.5-points[0,1])*chord[1])/np.dot(chord,chord)
                paint = np.stack([np.interp(uv,projection[stops],line_colors[stops,c]) for c in range(3)],axis=2)
            else:
                paint = color
            preview = cleared*(1-alpha[:,:,None]) + paint*alpha[:,:,None]
            affected = (alpha > 0) | (patch_mask > 0)
            error = np.abs(preview-patch).mean(axis=2)[affected]
            if error.mean() > 10 or np.percentile(error,95) > 32:
                continue
            background[y0:y1, x0:x1][patch_mask > 0] = cleared[patch_mask > 0]
            index = len(paths)
            fill = "#%02x%02x%02x" % tuple(np.round(color).astype(int))
            definition = ''
            if gradient:
                definition = (f'<defs><linearGradient id="strand-color-{index}" gradientUnits="userSpaceOnUse" '
                              f'x1="{points[0,0]:.3f}" y1="{points[0,1]:.3f}" x2="{points[-1,0]:.3f}" y2="{points[-1,1]:.3f}">')
                for stop in stops:
                    stop_color = '#%02x%02x%02x' % tuple(np.round(line_colors[stop]).astype(int))
                    definition += f'<stop offset="{np.clip(projection[stop],0,1):.5f}" stop-color="{stop_color}"/>'
                definition += '</linearGradient></defs>'
                fill = f'url(#strand-color-{index})'
            paths.append(definition+f'<path id="hair-{index}" data-polarity="{polarity}" '
                         f'data-strand="{polarity}-{item["strand"]}" data-region="{item["region"]}" '
                         f'data-length="{length:.2f}" fill="{fill}" d="{shape}"/>')
            records.append(dict(id=index, polarity=polarity, length=length,
                                points=points.tolist(), width=width.tolist(), color=fill,
                                local_mae=float(error.mean()), gradient=bool(gradient),
                                strand=f'{polarity}-{item["strand"]}',region=item['region'],regions=item['regions']))
            erase |= local
            owners[local > 0] = owner
    return background, paths, erase, records
