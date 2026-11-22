"""Select outline detail and retain the unpruned colour evidence."""
import numpy as np

from .color import srgb_to_oklab


def gray_classes(rgb):
    lab = srgb_to_oklab(rgb)
    chroma = np.linalg.norm(lab[..., 1:], axis=-1)
    # Warm tints (skin, cream, pale yellow) are never treated as neutral gray.
    warm = (lab[..., 1] > .001) | (lab[..., 2] > .003)
    neutral = (chroma <= .006) & ~(warm & (chroma >= .0045))
    luma = rgb @ np.array([.2126, .7152, .0722])
    classes = np.where(luma <= 18.0, 0, np.where(luma >= 246.0, 2, 1))
    return np.where(neutral, classes, -1), luma


def snap_gray(rgb, for_sketch):
    level, luma = gray_classes(rgb)
    result = rgb.copy()
    if for_sketch:
        middle = np.full(luma.shape, 127)
        shade = np.where(level == 0, 0, np.where(level == 2, 255, middle))
        result[level >= 0] = shade[level >= 0, None]
    else:
        # In colour mode, only snap true near-black and near-white; preserve mid-tones.
        result[level == 0] = 0
        result[level == 2] = 255
    return result


def interface_evidence(labels, source, palette_lab):
    lab = srgb_to_oklab(source)
    chroma = lab[..., 1:]
    magnitude = np.linalg.norm(chroma, axis=2)
    # Boost only measured chroma and taper the gain for almost neutral pixels.
    gain = 1 + 2 * np.clip((magnitude - .012) / .045, 0, 1)
    boosted = chroma * np.minimum(gain, .27 / np.maximum(magnitude, 1e-8))[..., None]
    pairs, samples = [], []
    for axis in (0, 1):
        grid = np.moveaxis(labels, axis, 0)
        image = np.moveaxis(lab, axis, 0)
        vivid = np.moveaxis(boosted, axis, 0)
        row, col = np.nonzero(grid[:-1] != grid[1:])
        left, right = np.maximum(row - 1, 0), np.minimum(row + 2, len(grid)-1)
        pairs.append(np.sort(np.column_stack((grid[row, col], grid[row+1, col])), axis=1))
        delta = image[left, col] - image[right, col]
        chromatic = np.linalg.norm(vivid[left, col] - vivid[right, col], axis=1)
        neutral = np.abs(delta[:, 0])
        samples.append(np.column_stack((chromatic, neutral, np.linalg.norm(delta[:, 1:], axis=1))))
    edges, inverse, lengths = np.unique(
        np.concatenate(pairs), axis=0, return_inverse=True, return_counts=True)
    samples = np.concatenate(samples)
    inv = inverse.ravel()
    mean = np.column_stack([
        np.bincount(inv, weights=np.minimum(s, .24)) / lengths for s in samples.T
    ])
    delta = np.linalg.norm(palette_lab[edges[:, 0]] - palette_lab[edges[:, 1]], axis=1)
    # A saturated edge is a strong seed only when the two actual fill colours
    # also differ. Later stages admit less pronounced colour and light changes.
    vivid = np.minimum(.73 * mean[:, 0], 1.5 * delta)
    light = np.minimum(mean[:, 1], 1.4 * delta)
    natural = np.minimum(.73 * mean[:, 2], 1.5 * delta)
    return edges, lengths, vivid, natural, light, delta


def sweep(labels, colors, edges, lengths, vivid, natural, light, delta):
    n = len(colors)
    lab = srgb_to_oklab(colors)
    gray, _ = gray_classes(colors)
    shade_only = ((gray[edges[:, 0]] >= 0) & (gray[edges[:, 0]] == gray[edges[:, 1]]))
    low, high = lab.copy(), lab.copy()
    area = np.bincount(labels.ravel()).copy()
    largest = area.copy()
    representative = np.arange(n)
    parent = np.arange(n)
    protected = [set() for _ in range(n)]
    for a, b in edges[(vivid >= .090) | (light >= .105)]:
        protected[a].add(b)
        protected[b].add(a)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    thresholds = [0, .012, .018, .025, .033, .043, .056, .073, .095, .123, .160]
    gains = np.linspace(0, 1, len(thresholds))
    first = np.zeros(len(edges), int)
    for stage, (threshold, gain) in enumerate(zip(thresholds, gains)):
        chromatic = natural + gain * (vivid - natural)
        # Previously admitted edges stay admitted in the following stage,
        # even when its chroma boost has been reduced.
        first[np.maximum(chromatic, light) + .25 * delta >= threshold] = stage
    score = np.maximum(vivid, light) + .25 * delta
    order = np.lexsort((score, first))
    cursor = 0
    stages = []
    # Build nested partitions from detailed to coarse, then read backwards as
    # progressively added boundary layers. Preserve near-neutral hard edges.
    for stage, (threshold, gain) in enumerate(zip(thresholds, gains)):
        while cursor < len(order) and first[order[cursor]] < stage:
            index = order[cursor]
            cursor += 1
            if np.any(gray[edges[index]] >= 0):
                # Neutral levels are handled by the separate sketch and fill rules.
                continue
            a, b = map(root, edges[index])
            if a == b or b in protected[a]:
                continue
            lo, hi = np.minimum(low[a], low[b]), np.maximum(high[a], high[b])
            span = (hi - lo) * np.array([1.0, 1.8, 1.8])
            if np.linalg.norm(span) > min(.086, .014 + 1.3 * threshold):
                continue
            if area[a] < area[b]:
                a, b = b, a
            next_rep = representative[b] if largest[b] > largest[a] else representative[a]
            locked_neighbors = protected[a] | protected[b]
            # Distinct components with the same fill fuse during export.
            # Keep both sides of a protected edge different after recolouring.
            if any(np.array_equal(colors[next_rep], colors[representative[neighbor]])
                   for neighbor in locked_neighbors):
                continue
            parent[b] = a
            area[a] += area[b]
            low[a], high[a] = lo, hi
            for neighbor in protected[b]:
                protected[neighbor].remove(b)
                protected[neighbor].add(a)
            protected[a] = locked_neighbors
            protected[b].clear()
            if largest[b] > largest[a]:
                largest[a] = largest[b]
                representative[a] = representative[b]
        roots = np.array([root(i) for i in range(n)])
        mapped = roots[edges]
        retained = (mapped[:, 0] != mapped[:, 1]) & ~shade_only
        distinct, indices = np.unique(np.sort(mapped[retained], axis=1), axis=0, return_inverse=True)
        contact = np.bincount(indices.ravel(), weights=lengths[retained])
        stages.append((dict(
            threshold=threshold, saturation_mix=round(float(gain), 2),
            regions=len(np.unique(roots)), interfaces=len(distinct),
            short_interfaces=int((contact <= 12).sum()), boundary_length=int(contact.sum()),
            chromatic_length=int(lengths[retained & (vivid >= .09)].sum()),
            light_length=int(lengths[retained & (light >= .105)].sum()),
        ), representative[roots].copy()))
    return stages[::-1]


def stop_stage(stages):
    # A large increase in short interfaces also needs to be dominated by
    # short features relative to the amount of new boundary being admitted.
    stats = [s for s, _ in stages]
    for index in range(1, len(stats)):
        old, new = stats[index-1:index+1]
        short = new['short_interfaces'] - old['short_interfaces']
        length = new['boundary_length'] - old['boundary_length']
        new['short_per_1000_added_pixels'] = round(1000 * short / max(length, 1), 2)
        new['added_short_interfaces'] = short
    floor = max(25, .02 * stats[0]['short_interfaces'])
    second = None
    for index in range(1, len(stats)-1):
        row = stats[index]
        gain = row['added_short_interfaces']
        previous = [max(0, r['added_short_interfaces']) for r in stats[1:index]]
        baseline = float(np.median(previous)) if previous else 0
        if gain < max(floor, 2.4 * baseline) or row['short_per_1000_added_pixels'] < 65:
            continue
        following = stats[index+1]
        next_gain = following['added_short_interfaces']
        if (max(floor, .60 * gain) <= next_gain <= 2.4 * gain
                and following['short_per_1000_added_pixels'] >= 65):
            second = index + 1
            break
    if second is not None:
        declines = 0
        second_gain = stats[second]['added_short_interfaces']
        for index in range(second+1, len(stats)):
            gain = stats[index]['added_short_interfaces']
            if (index == second+1 and gain >= max(floor, .60 * second_gain)
                    and stats[index]['short_per_1000_added_pixels'] >= 65):
                return second, 'before-third-jump'
            previous_gain = stats[index-1]['added_short_interfaces']
            declines = declines + 1 if gain < previous_gain else 0
            if declines == 3:
                return index, 'third-decline-after-second-jump'
        return second, 'no-confirmed-third-event'
    # Without an abrupt transition, retain a conservative stage for inspection.
    return next(i for i, r in enumerate(stats) if r['threshold'] <= .012), 'no-clear-jump'


def select_outline(labels, colors, source, *, for_sketch=True):
    evidence = interface_evidence(labels, source, srgb_to_oklab(colors))
    stages = sweep(labels, colors, *evidence)
    selected, reason = stop_stage(stages)
    mapping = stages[selected][1]
    return snap_gray(colors[mapping[labels]], for_sketch), selected, reason

