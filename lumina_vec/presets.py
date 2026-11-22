"""Preset tables and thresholds: pure data, no heavy imports.

Kept out of `pipeline` so the web front-end can validate preset names and show
the preset list without pulling in numpy / OpenCV / vtracer / torch. The
front-end is meant to run on a small always-on box; the conversion itself runs
on the machine that has the GPU (see `lumina_vec.server`).
"""

_BANDS = dict(bands=112, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4)
_OFF = dict(bands=0, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4)

PRESETS = {
    "print-regions": dict(structure="print-regions", bands=84, gradient_mode="clean",
                          smooth_lines=True, feature_routing=True),
    "source-detail": dict(source_detail=True),
    # tuned on anime / illustration with heavy grain
    "illustration": dict(pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
                         filter_speckle=12, layer_difference=10, simplify=1.5, **_OFF),
    # illustration + explicit shading bands: smooth light/dark transitions (more colours/paths)
    "shading": dict(pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
                    filter_speckle=4, layer_difference=10, simplify=1.5, ink=True, **_BANDS),
    # painterly / detailed art (light grain, fine texture) + shading bands
    "shading-detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                           filter_speckle=4, layer_difference=16, simplify=1.0, ink=False, **_BANDS),
    # ^ clean painterly art: its own lines are already clean; redrawing them made eyes/lashes heavier
    # clean cel/flat-colour art (hard-edged fills): no gradient smoothing (it put halos around
    # highlights), cutout tracing (shared edges, no slivers), fewer colours
    "cel": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10, filter_speckle=4, layer_difference=16,
                simplify=1.0, ink=True, **{**_BANDS, "bands": 48, "blur_sigma": 0.0},
                detail_overlay=False, hierarchical_mode="cutout"),
    # fewer, larger shapes; loses tiny strokes (mouth lines etc.)
    "flat": dict(pre_down=2.0, work_scale=1.5, ms_sp=10, ms_sr=16,
                 filter_speckle=24, layer_difference=12, simplify=1.5, **_OFF),
    # keep more small detail, bigger file
    "detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                   filter_speckle=8, layer_difference=16, simplify=1.0, **_OFF),
}

PRESET_NAMES = ("auto", *PRESETS)

GRAIN_THRESHOLD = 4.0          # mean |img - median3(img)| on native-resolution grey
MAX_WORK_PX = {"shading": 14e6, "shading-detail": 6e6, "cel": 6e6}
FLAT_THRESHOLD = 0.18          # share of pixels in gentle gradients; below = flat cel art
