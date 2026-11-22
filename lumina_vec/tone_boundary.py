"""Track a shared colour boundary through a broad tonal transition."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def track_tone_boundary(profile, xy, normal, direction, center, stations):
    """Select one coherent tone level, with the graph junctions kept fixed."""
    offsets = np.arange(-12., 12.01, .5)
    contrast = np.linalg.norm(direction)
    axis = direction / contrast
    colors = profile(xy, normal, offsets)
    relative = colors - center
    tone = gaussian_filter1d(relative @ axis, 3., axis=0, mode='nearest')
    chroma_error = np.linalg.norm(relative - (relative @ axis)[..., None] * axis, axis=-1)

    # Prefer a small expansion of the darker side in a gradual transition.
    # Isolated edges with a sharp step or a third colour use the existing fit.
    target = .005 * np.sign(direction[0])
    flank = 6
    change = tone[:, 2*flank:] - tone[:, :-2*flank]
    samples = tone[:, flank:-flank]
    steps = np.lib.stride_tricks.sliding_window_view(
        np.abs(np.diff(tone, axis=1)), 2*flank, axis=1).max(axis=-1)
    offsets = offsets[flank:-flank]
    weak = ((change > .0025) & (change < .028) & (steps < .006)
            & (chroma_error[:, flank:-flank] < .019)
            & (np.abs(samples - target) < .013))
    cost = np.abs(samples - target) / .004 + .035 * np.abs(offsets)
    cost = np.where(weak, cost, 8. + .15 * np.abs(offsets))
    zero = len(offsets) // 2
    supported = gaussian_filter1d(weak.any(axis=1).astype(float), 4., mode='nearest')
    if np.count_nonzero(supported > .68) < 28:
        return np.zeros(len(xy))
    # Where source evidence drops out, move back toward the existing boundary.
    cost[:, zero] = np.minimum(cost[:, zero], 1.8 + 2.0 * supported)
    cost[[0, -1], :] = np.inf
    cost[[0, -1], zero] = 0

    jumps = np.abs(offsets[:, None] - offsets)
    continuity = np.where(jumps <= 1., 1.2*jumps**2, np.inf)
    previous = cost[0]
    traceback = np.empty(cost.shape, dtype=np.int16)
    indexes = np.arange(len(offsets))
    for i in range(1, len(cost)):
        choices = previous[:, None] + continuity
        traceback[i] = choices.argmin(axis=0)
        previous = cost[i] + choices[traceback[i], indexes]
    selected = np.empty(len(cost), dtype=np.int16)
    selected[-1] = zero
    for i in range(len(cost)-1, 0, -1):
        selected[i-1] = traceback[i, selected[i]]
    shift = gaussian_filter1d(offsets[selected], 6., mode='nearest')
    fade = np.clip(np.minimum(stations, stations[-1]-stations)/12., 0, 1)
    return shift * fade
