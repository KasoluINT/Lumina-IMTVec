"""Pure ONNX Runtime (DirectML / CPU) inference and feature-routed conditioning maps.

Zero PyTorch dependency at deployment time: loads `weights/surface_style.onnx` and
`weights/feature_part.onnx` when present, predicting semantic & surface heatmaps used
to guide quantization, line protection, ink stitching, and stained-glass region merging.
"""
from dataclasses import dataclass
import gc
from pathlib import Path
import os

import cv2
import numpy as np

from .vram_guard import compute_runtime_profile

_WEIGHTS_DEFAULT = Path(__file__).resolve().parent.parent / "weights"
_SESSION_CACHE: dict[str, object] = {}


@dataclass
class FeatureMaps:
    ink_main_geom: np.ndarray     # float32 [H, W] in [0, 1]
    fine_strand: np.ndarray       # float32 [H, W] in [0, 1]
    delicate_detail: np.ndarray   # float32 [H, W] in [0, 1]
    mergeable_wash: np.ndarray    # float32 [H, W] in [0, 1]
    surface_probs: np.ndarray     # float32 [4, H, W] (cel, grad, grain, detail)
    summary: dict
    line_net_maps: np.ndarray | None = None   # float32 [4, H, W] (dark_ink, colored_line, color_block_bound, free_wisp)
    elem_grad_maps: np.ndarray | None = None  # float32 [8, H, W] (E0..E5, G0=cos, G1=sin)
    sr_clean_1x: np.ndarray | None = None     # uint8 [H, W, 3] pre-denoised SR->1x clean image if realesr_anime ran
    line_vector_fields: np.ndarray | None = None  # float32 [9, Hs, Ws] unpooled SR vector-field primitives
    sr_hi_rgb: np.ndarray | None = None       # uint8 [Hs, Ws, 3] native SR image matching line_vector_fields
    sr_scale: float = 1.0                     # SR scale factor between sr_hi_rgb and 1x input
    defect_prob: np.ndarray | None = None     # float32 [Hs, Ws] pre-SVG DefectRepairNet probability map
    grad_core_95: np.ndarray | None = None    # bool [H, W] uniform 95%-volume shrunk closed gradient interior
    line_buffer_5: np.ndarray | None = None   # bool [H, W] uniform outer 5% ring reserved as line buffer zone
    part_seg_probs: np.ndarray | None = None  # float32 [8, H, W] anime_part_seg.onnx semantic subject probabilities


def release_session(path: Path | str) -> None:
    """Release a cached ONNX session and trigger GC so DirectML/CUDA frees VRAM immediately."""
    key = str(path)
    if key in _SESSION_CACHE:
        _SESSION_CACHE.pop(key, None)
        gc.collect()


def clear_session_cache() -> None:
    """Evict all cached ONNX sessions and collect garbage to return GPU memory."""
    if _SESSION_CACHE:
        _SESSION_CACHE.clear()
        gc.collect()


def _get_session(
    path: Path,
    *,
    force_cpu: bool = False,
    cuda_mem_limit_bytes: int | None = None,
    low_vram: bool = False,
):
    key = f"{path}|cpu={int(force_cpu)}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    if not path.is_file():
        return None
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        if low_vram:
            try:
                so.add_session_config_entry("session.disable_prepacking", "1")
            except Exception:
                pass
        if force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            avail = ort.get_available_providers()
            providers = []
            if "DmlExecutionProvider" in avail:
                providers.append("DmlExecutionProvider")
            if "CUDAExecutionProvider" in avail:
                if cuda_mem_limit_bytes and cuda_mem_limit_bytes > 0:
                    providers.append((
                        "CUDAExecutionProvider",
                        {"gpu_mem_limit": str(int(cuda_mem_limit_bytes)), "arena_extend_strategy": "kSameAsRequested"},
                    ))
                else:
                    providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        _SESSION_CACHE[key] = sess
        return sess
    except Exception:
        return None


def _safe_run(sess, feed: dict, sess_path: Path | None = None):
    """Run ONNX session and gracefully fall back to CPU if GPU/DirectML hits VRAM exhaustion."""
    try:
        return sess.run(None, feed), sess
    except Exception as exc:
        msg = str(exc).lower()
        is_mem = any(k in msg for k in ("memory", "alloc", "outofmemory", "887a0005", "dml", "vram", "resource"))
        if sess_path is None or not is_mem:
            raise
        print(f"[vram_guard] GPU memory fallback -> CPU for {sess_path.name}: {str(exc)[:140]}", flush=True)
        clear_session_cache()
        cpu_sess = _get_session(sess_path, force_cpu=True)
        if cpu_sess is None:
            raise
        return cpu_sess.run(None, feed), cpu_sess


def _run_dense_unet(
    sess,
    rgb: np.ndarray,
    out_ch: int = 4,
    hi_weight: float = 0.72,
    skip_mask: np.ndarray | None = None,
    sess_path: Path | None = None,
) -> tuple[np.ndarray, int, int] | np.ndarray:
    """Run dense UNet using a constant (1, 3, 512, 512) tensor shape so DirectML never
    reallocates dynamic shape arenas or exhausts GPU memory -> [out_ch, H, W].

    When `skip_mask` is supplied, 512x512 tiles lying >= 99.2% inside the 95% shrunk gradient
    core skip CNN forward evaluation cleanly to save compute.
    """
    h, w = rgb.shape[:2]
    tile = 512
    g_maps = None
    if hi_weight < 0.999:
        g_img = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA if max(h, w) > tile else cv2.INTER_CUBIC)
        gx = (g_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        g_res, sess = _safe_run(sess, {"rgb": gx}, sess_path=sess_path)
        g_out = g_res[0][0]  # [out_ch, 512, 512]
        g_maps = np.stack([cv2.resize(g_out[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(out_ch)], axis=0)
        if max(h, w) <= 720 and skip_mask is None:
            return np.clip(g_maps, 0.0, 1.0)

    stride = 432
    acc = np.zeros((out_ch, h, w), np.float32)
    w_acc = np.zeros((h, w), np.float32)
    wy = np.hanning(tile)[:, None].astype(np.float32) * 0.96 + 0.04
    wx = np.hanning(tile)[None, :].astype(np.float32) * 0.96 + 0.04
    win = wy * wx

    ys = list(range(0, max(1, h - tile + 1), stride))
    if len(ys) == 0 or (ys[-1] + tile < h and h > tile):
        ys.append(max(0, h - tile))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if len(xs) == 0 or (xs[-1] + tile < w and w > tile):
        xs.append(max(0, w - tile))

    skipped = 0
    total_tiles = len(ys) * len(xs)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            th, tw = y1 - y0, x1 - x0
            if skip_mask is not None and float(skip_mask[y0:y1, x0:x1].mean()) >= 0.992:
                skipped += 1
                w_acc[y0:y1, x0:x1] += win[:th, :tw]
                continue
            sub = rgb[y0:y1, x0:x1]
            if sub.shape[0] < tile or sub.shape[1] < tile:
                sub = cv2.copyMakeBorder(sub, 0, tile - sub.shape[0], 0, tile - sub.shape[1], cv2.BORDER_REFLECT_101)
            tx = (sub.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            o_res, sess = _safe_run(sess, {"rgb": tx}, sess_path=sess_path)
            o = o_res[0][0]
            acc[:, y0:y1, x0:x1] += o[:, :th, :tw] * win[:th, :tw]
            w_acc[y0:y1, x0:x1] += win[:th, :tw]

    t_maps = acc / np.maximum(w_acc[None], 1e-4)
    if g_maps is not None:
        out = np.clip(hi_weight * t_maps + (1.0 - hi_weight) * g_maps, 0.0, 1.0)
    else:
        out = np.clip(t_maps, 0.0, 1.0)
    if skip_mask is not None:
        return out, skipped, total_tiles
    return out


def _run_surface_cnn(
    sess,
    rgb: np.ndarray,
    grid: int = 10,
    batch_size: int = 32,
    sess_path: Path | None = None,
) -> np.ndarray:
    """Evaluate 96x96 SurfaceStyleCNN in bounded mini-batches -> upsampled [4, H, W] softmax probs."""
    h, w = rgb.shape[:2]
    ps = 96
    if h < ps or w < ps:
        pad = np.full((max(h, ps), max(w, ps), 3), 255, np.uint8)
        pad[:h, :w] = rgb
        rgb_w = pad
    else:
        rgb_w = rgb
    H, W = rgb_w.shape[:2]
    gh = min(grid, max(2, H // ps + 1))
    gw = min(grid, max(2, W // ps + 1))
    ys = np.linspace(0, H - ps, gh, dtype=int)
    xs = np.linspace(0, W - ps, gw, dtype=int)

    patches = []
    for y0 in ys:
        for x0 in xs:
            patches.append(rgb_w[y0:y0+ps, x0:x0+ps].astype(np.float32).transpose(2, 0, 1) / 255.0)
    batch = max(1, int(batch_size))
    out_chunks = []
    for i in range(0, len(patches), batch):
        inp = np.stack(patches[i:i + batch], axis=0)
        res, sess = _safe_run(sess, {"rgb": inp}, sess_path=sess_path)
        out_chunks.append(res[0])
    logits = np.concatenate(out_chunks, axis=0)  # [gh*gw, 4]
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs_grid = probs.reshape(gh, gw, 4).transpose(2, 0, 1)
    return np.stack([cv2.resize(probs_grid[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=0)


def _upscale_sr(
    rgb: np.ndarray,
    target_scale: float,
    sess_sr,
    *,
    tile: int = 384,
    pad: int = 12,
    sess_path: Path | None = None,
) -> np.ndarray:
    """Run realesr_anime.onnx with constant-shape square tiles tuned to GPU VRAM budget."""
    h, w = rgb.shape[:2]
    sh, sw = int(round(h * target_scale)), int(round(w * target_scale))
    out_sr = np.zeros((sh, sw, 3), np.uint8)
    fixed_s = tile + 2 * pad
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(h, y0 + tile + pad), min(w, x0 + tile + pad)
            sub = rgb[ya:yb, xa:xb]
            sub_h, sub_w = sub.shape[:2]
            if sub_h < fixed_s or sub_w < fixed_s:
                sub = cv2.copyMakeBorder(sub, 0, fixed_s - sub_h, 0, fixed_s - sub_w, cv2.BORDER_REFLECT_101)
            patch = sub.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            sr_res, sess_sr = _safe_run(sess_sr, {"rgb": patch}, sess_path=sess_path)
            sr4 = np.clip(sr_res[0][0].transpose(1, 2, 0), 0.0, 1.0)
            ty, tx = (y0 - ya) * 4, (x0 - xa) * 4
            th_raw, tw_raw = min(h, y0 + tile) - y0, min(w, x0 + tile) - x0
            crop4 = (sr4[ty:ty + th_raw * 4, tx:tx + tw_raw * 4] * 255.0).clip(0, 255).astype(np.uint8)
            dy0 = int(round(y0 * target_scale))
            dx0 = int(round(x0 * target_scale))
            dy1 = int(round(min(h, y0 + tile) * target_scale))
            dx1 = int(round(min(w, x0 + tile) * target_scale))
            out_sr[dy0:dy1, dx0:dx1] = cv2.resize(crop4, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_AREA)
    return out_sr


def _blend_chunked_u8(
    base_u8: np.ndarray,
    over: np.ndarray,
    alpha_2d: np.ndarray,
    *,
    over_is_chw01: bool = False,
    chunk_rows: int = 512,
) -> np.ndarray:
    """Blend `base_u8` with `over` via `alpha_2d` in bounded row strips to prevent multi-GB OOM on large images."""
    h = base_u8.shape[0]
    out = base_u8.copy()
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        a = alpha_2d[y0:y1]
        if not np.any(a > 1e-4):
            continue
        a3 = a[..., None].astype(np.float32, copy=False)
        b = base_u8[y0:y1].astype(np.float32)
        if over_is_chw01:
            o = over[:, y0:y1].transpose(1, 2, 0).astype(np.float32, copy=False) * 255.0
        else:
            o = over[y0:y1].astype(np.float32)
        blended = b * (1.0 - a3) + o * a3
        out[y0:y1] = np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)
    return out


def _run_defect_repair_cnn(sess_def, sr_rgb: np.ndarray, parts_1x: np.ndarray, sess_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run feature-guided DefectRepairNet (fixed 512x512 DirectML-safe tiled inference) on native-res/SR RGB
    BEFORE line/vector extraction so hazy bright slivers ('亮棱') and trapped scratches are repaired cleanly."""
    hs, ws = sr_rgb.shape[:2]
    parts_hi = np.stack([cv2.resize(parts_1x[c], (ws, hs), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=0)

    tile, stride = 512, 432
    acc = np.zeros((4, hs, ws), np.float32)
    w_acc = np.zeros((hs, ws), np.float32)
    wy = np.hanning(tile)[:, None].astype(np.float32) * 0.96 + 0.04
    wx = np.hanning(tile)[None, :].astype(np.float32) * 0.96 + 0.04
    win = wy * wx

    ys = list(range(0, max(1, hs - tile + 1), stride))
    if ys[-1] + tile < hs:
        ys.append(max(0, hs - tile))
    xs = list(range(0, max(1, ws - tile + 1), stride))
    if xs[-1] + tile < ws:
        xs.append(max(0, ws - tile))

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(hs, y0 + tile), min(ws, x0 + tile)
            sub_rgb = sr_rgb[y0:y1, x0:x1]
            sub_p = parts_hi[:, y0:y1, x0:x1]
            if sub_rgb.shape[0] < tile or sub_rgb.shape[1] < tile:
                pb, pr = tile - sub_rgb.shape[0], tile - sub_rgb.shape[1]
                sub_rgb = cv2.copyMakeBorder(sub_rgb, 0, pb, 0, pr, cv2.BORDER_REFLECT_101)
                sub_p = np.pad(sub_p, ((0, 0), (0, pb), (0, pr)), mode="reflect")
            ten_7 = np.concatenate([
                sub_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0,
                sub_p.astype(np.float32),
            ], axis=0)[None]
            out_res, sess_def = _safe_run(sess_def, {"feat_in": ten_7}, sess_path=sess_path)
            out = out_res[0][0]
            th, tw = y1 - y0, x1 - x0
            acc[:, y0:y1, x0:x1] += out[:, :th, :tw] * win[:th, :tw]
            w_acc[y0:y1, x0:x1] += win[:th, :tw]

    for y0 in range(0, hs, 512):
        y1 = min(hs, y0 + 512)
        w_sl = np.maximum(w_acc[None, y0:y1], 1e-4)
        acc[:, y0:y1] = np.clip(acc[:, y0:y1] / w_sl, 0.0, 1.0)
    del w_acc
    d_prob = acc[0].copy()
    # Hard semantic shield: protect eyes, catchlights & delicate detail pixels
    for y0 in range(0, hs, 512):
        y1 = min(hs, y0 + 512)
        d_prob[y0:y1] *= (1.0 - np.clip((parts_hi[2, y0:y1] - 0.32) / 0.25, 0.0, 1.0))
    del parts_hi
    repaired = _blend_chunked_u8(
        sr_rgb,
        acc[1:4],
        np.clip((d_prob - 0.35) / 0.45, 0.0, 1.0),
        over_is_chw01=True,
    )
    return repaired, d_prob


def _pre_sr_denoise_1x(rgb_1x: np.ndarray, parts_1x: np.ndarray, surf_1x: np.ndarray) -> np.ndarray:
    """Suppress weak 1x compression/shading mottling in flat interiors BEFORE RealESR-Anime upscaling
    while strictly protecting delicate eye details, fine hair strands, dense curves, and soft gradients."""
    from .color import srgb_to_oklab

    ok = srgb_to_oklab(rgb_1x).astype(np.float32)
    L = ok[..., 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.hypot(gx, gy)
    flat_w = np.clip((0.065 - g) / 0.045, 0.0, 1.0)
    edge_dense = cv2.boxFilter((g > 0.06).astype(np.float32), -1, (5, 5), normalize=True)
    dense_prot = np.clip((edge_dense - 0.16) / 0.14, 0.0, 1.0)
    prot = np.clip(
        np.maximum(
            np.maximum(parts_1x[2] * 1.4, np.clip((parts_1x[1] - 0.12) / 0.28, 0.0, 1.0)),
            np.maximum(surf_1x[1] * 1.15, dense_prot),
        ),
        0.0,
        1.0,
    )
    w_eff = (flat_w * (1.0 - prot))[..., None].astype(np.float32)
    sm = cv2.bilateralFilter(rgb_1x, d=7, sigmaColor=22, sigmaSpace=3.2)
    return np.clip(np.rint(rgb_1x.astype(np.float32) * (1.0 - w_eff) + sm.astype(np.float32) * w_eff), 0, 255).astype(np.uint8)


def _flatten_sr_for_lines(sr_rgb: np.ndarray, parts_1x: np.ndarray, surf_1x: np.ndarray) -> np.ndarray:
    """Flatten residual intra-compartment color mottling (~95%+ fidelity) BEFORE LineVectorNet
    so ghost shading streaks inside petals don't turn into sticky/bridging linework, while strictly
    preserving fine hair strands and dense parallel curve bundles."""
    from .color import srgb_to_oklab

    hs, ws = sr_rgb.shape[:2]
    delic_hi = cv2.resize(parts_1x[2], (ws, hs), interpolation=cv2.INTER_LINEAR)
    strand_hi = cv2.resize(parts_1x[1], (ws, hs), interpolation=cv2.INTER_LINEAR)
    cel_hi = cv2.resize(surf_1x[0], (ws, hs), interpolation=cv2.INTER_LINEAR)
    grad_hi = cv2.resize(surf_1x[1], (ws, hs), interpolation=cv2.INTER_LINEAR)

    L = srgb_to_oklab(sr_rgb)[..., 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.hypot(gx, gy)
    edge_dense = cv2.boxFilter((g > 0.06).astype(np.float32), -1, (11, 11), normalize=True)
    dense_prot = np.clip((edge_dense - 0.12) / 0.12, 0.0, 1.0)

    sm_1x = cv2.resize(sr_rgb, (max(1, ws // 2), max(1, hs // 2)), interpolation=cv2.INTER_AREA)
    ms_1x = cv2.pyrMeanShiftFiltering(sm_1x, sp=6, sr=34, maxLevel=1)
    ms = cv2.resize(ms_1x, (ws, hs), interpolation=cv2.INTER_LINEAR)
    f_flat = cv2.bilateralFilter(ms, d=7, sigmaColor=34, sigmaSpace=5)
    f_mild = cv2.bilateralFilter(sr_rgb, d=5, sigmaColor=16, sigmaSpace=3.5)

    prot_soft = np.clip(
        np.maximum(
            np.maximum(
                grad_hi * np.clip((0.35 - cel_hi) / 0.25, 0.0, 1.0) * 1.35,
                np.clip((strand_hi - 0.14) / 0.28, 0.0, 1.0),
            ),
            dense_prot,
        ),
        0.0,
        1.0,
    )[..., None].astype(np.float32)
    f = f_flat.astype(np.float32) * (1.0 - prot_soft) + f_mild.astype(np.float32) * prot_soft

    ink_core = np.clip((0.28 - L) / 0.14, 0.0, 1.0)
    keep_w = np.clip(
        np.maximum(
            np.maximum(ink_core, (delic_hi - 0.25) / 0.35),
            np.maximum(np.clip((strand_hi - 0.18) / 0.30, 0.0, 1.0), dense_prot),
        ),
        0.0,
        0.97,
    )[..., None].astype(np.float32)
    return np.clip(np.rint(f * (1.0 - keep_w) + sr_rgb.astype(np.float32) * keep_w), 0, 255).astype(np.uint8)


def _reprocess_after_zoom(zoomed_rgb: np.ndarray, sess_def=None, parts_patch: np.ndarray | None = None) -> np.ndarray:
    """Re-process an upscaled (5x or 10x) image patch BEFORE feeding to LineVectorNet:
    1) Morphological shock edge-sharpening: collapses stretched interpolation transition ramps & Lanczos ringing halos
       back into crisp step edges.
    2) DefectRepairNet + 95% edge-preserving bilateral de-noise: cleans any magnified residual noise/mottling
       so the CNN heatmap has ZERO halo ('光晕')."""
    from .color import srgb_to_oklab

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(zoomed_rgb, k3, iterations=1)
    di = cv2.dilate(zoomed_rgb, k3, iterations=1)
    mid = 0.5 * (er.astype(np.float32) + di.astype(np.float32))
    diff_span = np.max(di.astype(np.float32) - er.astype(np.float32), axis=2, keepdims=True)
    edge_w = np.clip((diff_span - 18.0) / 36.0, 0.0, 0.78)
    snapped = np.where(zoomed_rgb.astype(np.float32) >= mid, di.astype(np.float32), er.astype(np.float32))
    sharp = np.clip(np.rint(zoomed_rgb.astype(np.float32) * (1.0 - edge_w) + snapped * edge_w), 0, 255).astype(np.uint8)

    if sess_def is not None and parts_patch is not None:
        sharp, _ = _run_defect_repair_cnn(sess_def, sharp, parts_patch)

    sm = cv2.bilateralFilter(sharp, d=7, sigmaColor=24, sigmaSpace=4.0)
    L = srgb_to_oklab(sharp)[..., 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.hypot(gx, gy)
    keep_edge = np.clip((g - 0.04) / 0.06, 0.0, 0.95)[..., None].astype(np.float32)
    return np.clip(np.rint(sm.astype(np.float32) * (1.0 - keep_edge) + sharp.astype(np.float32) * keep_edge), 0, 255).astype(np.uint8)


def _downsample_zoomed_vf(vf_hi: np.ndarray, hs: int, ws: int, zoom_ratio: float) -> np.ndarray:
    """Downsample 5x/10x LineVectorNet output back to 2x SR (hs, ws) with zero halo skirt."""
    out = np.empty((9, hs, ws), dtype=np.float32)
    for c in (0, 1, 2, 3):
        ch = vf_hi[c].copy()
        ch = np.where(ch >= 0.22, (ch - 0.16) / 0.84, 0.0).astype(np.float32)
        ch_2x = cv2.resize(ch, (ws, hs), interpolation=cv2.INTER_AREA)
        if c == 0:
            ch_2x = np.where(ch_2x >= 0.12, np.clip(ch_2x * 1.48, 0.0, 1.0), 0.0)
        out[c] = ch_2x
    for c in (4, 5, 6, 7):
        out[c] = cv2.resize(vf_hi[c], (ws, hs), interpolation=cv2.INTER_LINEAR)
    rad_2x = cv2.resize(vf_hi[8], (ws, hs), interpolation=cv2.INTER_AREA) / float(zoom_ratio)
    out[8] = np.clip(rad_2x * 1.20, 0.0, 1.0)
    return out


def _restore_tone_1x(hi_rgb: np.ndarray, orig_rgb_1x: np.ndarray) -> np.ndarray:
    """Anchor SR low-frequency OKLab tone (luminance & chroma) back to the untouched 1x source
    so RealESR-Anime never washes out skin warmth or raises white exposure."""
    from .color import srgb_to_oklab, oklab_to_srgb

    hs, ws = hi_rgb.shape[:2]
    h1, w1 = orig_rgb_1x.shape[:2]
    up = orig_rgb_1x if (hs, ws) == (h1, w1) else cv2.resize(orig_rgb_1x, (ws, hs), interpolation=cv2.INTER_LINEAR)
    orig_lab = srgb_to_oklab(up).astype(np.float32)
    hi_lab = srgb_to_oklab(hi_rgb).astype(np.float32)
    diff = np.clip(orig_lab - hi_lab, -0.06, 0.06)
    sig = 3.0 * max(1.0, hs / max(1.0, float(h1)))
    delta = cv2.GaussianBlur(diff, (0, 0), sig)
    delta[..., 0] = np.clip(delta[..., 0], -0.045, 0.045)
    delta[..., 1:] = np.clip(delta[..., 1:], -0.025, 0.025)
    return np.clip(np.rint(oklab_to_srgb(hi_lab + delta)), 0, 255).astype(np.uint8)


def _compute_grad_core_and_buffer(
    elem_maps: np.ndarray,
    parts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build strict closed-region 95%-volume gradient cores and 5% outer perimeter line buffer rings.

    For each closed irregular gradient compartment G_i:
      - Requires high ElementGradNet confidence (`E1 >= 0.64`, `E0 < 0.30`) so flat cel / mosaic
        compartments are never falsely absorbed into gradient processing.
      - Computes Euclidean distance transform from internal boundaries (preserving canvas borders)
        and shrinks each G_i inward at its 5%-area distance quantile -> preserving ~95% interior
        volume (`grad_core_95`) while leaving a uniform 5% ring (`line_buffer_5`) for line safety.
    """
    h, w = elem_maps.shape[1:]
    grad_cand = (
        (elem_maps[1] >= 0.64)
        & (elem_maps[0] < 0.30)
        & ((parts[3] >= 0.32) | (elem_maps[1] >= 0.78))
        & (parts[2] < 0.38)
    )
    k3 = np.ones((3, 3), np.uint8)
    grad_clean = cv2.morphologyEx(grad_cand.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    if not grad_clean.any():
        empty = np.zeros((h, w), bool)
        return empty, empty.copy()

    # Replicate outer borders before EDT so gradient regions cut off by canvas edges don't get buffer bands at crop borders
    pad_u8 = cv2.copyMakeBorder(grad_clean.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    dist = cv2.distanceTransform(pad_u8, cv2.DIST_L2, 5)[1:-1, 1:-1]

    n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(grad_clean.astype(np.uint8), connectivity=4)
    grad_core_95 = np.zeros((h, w), bool)
    line_buffer_5 = np.zeros((h, w), bool)
    # Gradients are genuinely wide/open surfaces, NEVER narrow hair crevices or tiny flat pockets
    min_open_area = max(1400, int(0.0010 * h * w))
    min_inscribed_r = 7.0

    for cid in range(1, n_cc):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_open_area:
            continue
        x0, y0, bw, bh = (
            int(stats[cid, cv2.CC_STAT_LEFT]),
            int(stats[cid, cv2.CC_STAT_TOP]),
            int(stats[cid, cv2.CC_STAT_WIDTH]),
            int(stats[cid, cv2.CC_STAT_HEIGHT]),
        )
        sub = cc_map[y0:y0 + bh, x0:x0 + bw] == cid
        d_vals = dist[y0:y0 + bh, x0:x0 + bw][sub]
        if float(d_vals.max()) < min_inscribed_r:
            continue
        r_5 = float(np.clip(np.quantile(d_vals, 0.05), 1.5, 6.0))
        core_sub = sub & (dist[y0:y0 + bh, x0:x0 + bw] > r_5)
        if not core_sub.any():
            continue
        grad_core_95[y0:y0 + bh, x0:x0 + bw] |= core_sub
        line_buffer_5[y0:y0 + bh, x0:x0 + bw] |= (sub & (~core_sub))

    return grad_core_95, line_buffer_5


def _suppress_dense_highlight_plateaus(
    vf: np.ndarray,
    sr_rgb: np.ndarray,
    grad_core_hi: np.ndarray | None = None,
) -> np.ndarray:
    """Clean residual dark-background gap glow inside dense multi-line plateaus so closely packed
    parallel curves stay cleanly separated without local ROI patch resampling."""
    from .color import srgb_to_oklab

    vf = vf.copy()
    prob = vf[0].astype(np.float32)
    lab = srgb_to_oklab(sr_rgb).astype(np.float32)
    L = lab[..., 0]

    p_density7 = cv2.boxFilter((prob > 0.45).astype(np.float32), -1, (7, 7), normalize=True)
    p_mean9 = cv2.boxFilter(prob, -1, (9, 9), normalize=True)
    plateau_w = np.clip((p_density7 - 0.42) / 0.24, 0.0, 1.0) * np.clip((p_mean9 - 0.34) / 0.22, 0.0, 1.0)

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    ax_s = cv2.Sobel(lab[..., 1], cv2.CV_32F, 1, 0, ksize=3) / 8.0
    ay_s = cv2.Sobel(lab[..., 1], cv2.CV_32F, 0, 1, ksize=3) / 8.0
    bx_s = cv2.Sobel(lab[..., 2], cv2.CV_32F, 1, 0, ksize=3) / 8.0
    by_s = cv2.Sobel(lab[..., 2], cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.sqrt(gx * gx + gy * gy + 2.5 * (ax_s * ax_s + ay_s * ay_s + bx_s * bx_s + by_s * by_s))

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    L_open = cv2.morphologyEx(L, cv2.MORPH_OPEN, k9)
    L_close = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k9)
    tophat = np.maximum(0.0, L - L_open)
    blackhat = np.maximum(0.0, L_close - L)
    th_max9 = cv2.dilate(tophat, k9)
    bh_max9 = cv2.dilate(blackhat, k9)

    is_dark_sky_bundle = np.clip((0.26 - L_open) / 0.08, 0.0, 1.0) * np.clip((th_max9 - 0.15) / 0.15, 0.0, 1.0)
    rel_th = tophat / np.maximum(th_max9, 1e-4)
    rel_bh = blackhat / np.maximum(bh_max9, 1e-4)
    keep_in_bundle = np.clip(
        np.maximum(
            (rel_th - 0.16) / 0.28,
            (grad - 0.055) / 0.065 * np.clip((rel_th - 0.08) / 0.18, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )

    g_max7 = cv2.dilate(grad, k7)
    rel_g = grad / np.maximum(g_max7, 1e-4)
    # Strictly preserve 1-2px dark ink valleys (blackhat) & bright/colored ridges inside dense hair bundles
    bh5 = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k5)
    th5 = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k5)
    lab3 = np.stack([L, lab[..., 1] * 1.8, lab[..., 2] * 1.8], axis=-1)
    ridge_3d = np.linalg.norm(lab3 - cv2.GaussianBlur(lab3, (5, 5), 1.1), axis=-1)
    bh_l = np.maximum(bh5, th5)
    valley_ridge_shield = np.maximum.reduce([
        np.clip((bh5 - 0.012) / 0.016, 0.0, 1.0),
        np.clip((th5 - 0.012) / 0.016, 0.0, 1.0),
        np.clip((ridge_3d - 0.016) / 0.020, 0.0, 1.0),
    ])
    keep_general = np.clip(
        np.maximum.reduce([(rel_g - 0.18) / 0.30, (rel_th - 0.25) / 0.30, (rel_bh - 0.22) / 0.28, valley_ridge_shield]),
        0.10,
        1.0,
    )

    keep_mask = keep_in_bundle * is_dark_sky_bundle + keep_general * (1.0 - is_dark_sky_bundle)
    vf0_clean = np.clip(prob * (1.0 - plateau_w + plateau_w * keep_mask), 0.0, 1.0)

    # Suppress thick amorphous shadow-rim clouds in `vf[0]` along soft-shading seams (never touching dark ink or crisp lines)
    thick_cloud = cv2.dilate(cv2.erode((vf0_clean > 0.36).astype(np.uint8), k9), k9) > 0
    if thick_cloud.any():
        sharp_core = (L < 0.33) | (vf[1] >= 0.34) | (grad > 0.042) | (bh_l >= 0.028) | (ridge_3d >= 0.030)
        shadow_cloud = thick_cloud & (~sharp_core) & (L >= 0.33)
        vf0_clean = np.where(shadow_cloud, np.minimum(vf0_clean, np.maximum(vf[1], 0.14)), vf0_clean)

    # Prune low-confidence purple fog (0.04 - 0.38) flanking gradient seams or floating un-backed
    # ONLY where local 5x5 blackhat/tophat/3D-OKLab ridge evidence is genuinely absent.
    ridge_support = (
        (vf0_clean >= 0.39)
        | (vf[1] >= 0.33)
        | (L < 0.33)
        | (grad >= 0.038)
        | (bh_l >= 0.018)
        | (ridge_3d >= 0.022)
    )
    weak_fog_cand = (vf0_clean > 0.04) & (vf0_clean < 0.39) & (~ridge_support) & (L_open >= 0.24)
    if weak_fog_cand.any():
        near_strong_line = cv2.dilate((vf0_clean >= 0.46).astype(np.uint8), k9) > 0
        detached_haze = ~(cv2.dilate((vf0_clean >= 0.39).astype(np.uint8), k5) > 0)
        flank_haze = near_strong_line & (bh_l < 0.018) & (ridge_3d < 0.022) & (L >= 0.33)
        drop_fog = weak_fog_cand & (detached_haze | flank_haze)
        vf0_clean = np.where(drop_fog, np.minimum(vf0_clean * 0.15, vf[1] * 0.40), vf0_clean)

    # Inside strict 95%-volume shrunk gradient cores, zero out weak ridge-free haze while preserving
    # the entire outer 5% line-buffer ring untouched.
    if grad_core_hi is not None and grad_core_hi.shape == vf0_clean.shape:
        core_weak = grad_core_hi & (vf0_clean > 0.03) & (vf0_clean < 0.44) & (vf[1] < 0.34) & (L >= 0.31) & (bh_l < 0.025) & (ridge_3d < 0.028)
        if core_weak.any():
            vf0_clean = np.where(core_weak, vf0_clean * 0.12, vf0_clean)

    vf[0] = vf0_clean
    vf[8] = vf[8] * (1.0 - 0.35 * plateau_w)
    return vf


def _run_anime_part_seg_onnx(
    sess,
    rgb: np.ndarray,
    sess_path: Path | None = None,
) -> np.ndarray:
    """Run static 512x512 aspect-preserving `anime_part_seg.onnx` with automatic 2nd-pass
    character-ROI zoom for small-character full-scene illustrations -> float32 [8, H, W]."""
    h, w = rgb.shape[:2]
    size = 512
    inp_name = sess.get_inputs()[0].name

    def _infer_box(img_box: np.ndarray, cur_sess) -> tuple[np.ndarray, object]:
        bh, bw = img_box.shape[:2]
        sc = size / float(max(bh, bw))
        nh = max(1, min(size, int(round(bh * sc))))
        nw = max(1, min(size, int(round(bw * sc))))
        interp = cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR
        r_im = cv2.resize(img_box, (nw, nh), interpolation=interp)
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        canvas = np.full((size, size, 3), 128, dtype=np.uint8)
        canvas[y0:y0 + nh, x0:x0 + nw] = r_im
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        res, cur_sess = _safe_run(cur_sess, {inp_name: x}, sess_path=sess_path)
        out = res[0][0]  # [8, 512, 512]
        cropped = out[:, y0:y0 + nh, x0:x0 + nw]
        probs_box = np.stack(
            [cv2.resize(cropped[c], (bw, bh), interpolation=cv2.INTER_LINEAR) for c in range(8)],
            axis=0,
        )
        return np.clip(probs_box, 0.0, 1.0).astype(np.float32), cur_sess

    def _blend_roi(dst: np.ndarray, roi_probs: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> None:
        """Smoothly replace blurry global probabilities inside a sharper zoomed ROI."""
        wy = np.hanning(y1 - y0)[:, None].astype(np.float32)
        wx = np.hanning(x1 - x0)[None, :].astype(np.float32)
        wb = np.clip((wy * wx) * 5.0, 0.0, 1.0)
        dst[:, y0:y1, x0:x1] = (1.0 - wb[None]) * dst[:, y0:y1, x0:x1] + wb[None] * roi_probs

    probs, sess = _infer_box(rgb, sess)

    # Pass 2: Character foreground ROI zoom (feathered replacement erases blurry global bleed!)
    fg = probs[1:].max(axis=0) >= 0.40
    fg_ratio = float(fg.mean())
    if 0.01 <= fg_ratio <= 0.35:
        ys, xs = np.where(fg)
        cy0, cy1 = int(ys.min()), int(ys.max()) + 1
        cx0, cx1 = int(xs.min()), int(xs.max()) + 1
        bh, bw = cy1 - cy0, cx1 - cx0
        pad_y = max(int(bh * 0.35), int(h * 0.08))
        pad_x = max(int(bw * 0.35), int(w * 0.08))
        ry0, ry1 = max(0, cy0 - pad_y), min(h, cy1 + pad_y)
        rx0, rx1 = max(0, cx0 - pad_x), min(w, cx1 + pad_x)
        if (ry1 - ry0) >= 96 and (rx1 - rx0) >= 96 and (ry1 - ry0) * (rx1 - rx0) < 0.72 * h * w:
            p_roi, sess = _infer_box(rgb[ry0:ry1, rx0:rx1], sess)
            _blend_roi(probs, p_roi, ry0, ry1, rx0, rx1)

    # Pass 3: Tight Head/Face ROI zoom (hair + headwear + face_skin + eyes_mouth)
    # Resolves eyes, eyebrows, lips, and crisp hair/forehead edges even on wide 4K/13MP scenes!
    head = probs[1:5].max(axis=0) >= 0.38
    head_ratio = float(head.mean())
    if 0.001 <= head_ratio <= 0.18:
        ys, xs = np.where(head)
        hy0, hy1 = int(ys.min()), int(ys.max()) + 1
        hx0, hx1 = int(xs.min()), int(xs.max()) + 1
        hh, hw = hy1 - hy0, hx1 - hx0
        pad_y = max(int(hh * 0.45), 64)
        pad_x = max(int(hw * 0.45), 64)
        ry0, ry1 = max(0, hy0 - pad_y), min(h, hy1 + pad_y)
        rx0, rx1 = max(0, hx0 - pad_x), min(w, hx1 + pad_x)
        if (ry1 - ry0) >= 96 and (rx1 - rx0) >= 96 and (ry1 - ry0) * (rx1 - rx0) < 0.65 * h * w:
            p_head, sess = _infer_box(rgb[ry0:ry1, rx0:rx1], sess)
            _blend_roi(probs, p_head, ry0, ry1, rx0, rx1)

    return probs


def infer_feature_maps(
    rgb: np.ndarray,
    weights_dir: str | Path | None = None,
    sr_scale: float | None = None,
    core_mode: bool = False,
) -> FeatureMaps | None:
    """Whole-image magnification (default 2x, optional 5x / 10x) -> Pre-flatten closed-region
    Gradient CNN (95% core + 5% line buffer) -> Skip pure-gradient tiles during LineVectorNet.

        When `core_mode=True` (for AI-repainted clean flat/cel inputs), bypasses redundant JPEG/raw-noise
    repair passes (`_pre_sr_denoise_1x`, `DefectRepairNet`, `_reprocess_after_zoom`, `_suppress_dense_highlight_plateaus`)
    while retaining core `RealESR 2x + _flatten_sr_for_lines + ElementGradNet + LineVectorNet + LineNet`.
    """
    wdir = Path(weights_dir) if weights_dir else Path(os.environ.get("LUMINA_WEIGHTS_DIR", str(_WEIGHTS_DEFAULT)))
    p_sr = wdir / "realesr_anime.onnx"
    p_def = wdir / "defect_repair_net.onnx"
    p_line = wdir / "line_net.onnx"
    p_vec = wdir / "line_vector_net.onnx"
    p_elem = wdir / "element_grad.onnx"
    p_part = wdir / "feature_part.onnx"
    p_surf = wdir / "surface_style.onnx"
    p_partseg = wdir / "anime_part_seg.onnx"
    if not any(p.is_file() for p in (p_line, p_vec, p_elem, p_part, p_surf)):
        return None

    profile = compute_runtime_profile()
    cuda_lim = int(profile.vram_budget_mb * 1024 * 1024) if profile.vram_budget_mb > 0 else None
    sess_kw = dict(force_cpu=profile.force_cpu, cuda_mem_limit_bytes=cuda_lim, low_vram=profile.evict_per_stage)
    print(
        f"[vram_guard] gpu={profile.gpu_name or 'CPU'} vram_budget={profile.vram_budget_mb:.0f}MB "
        f"ram_avail={profile.ram_avail_mb:.0f}MB sr_tile={profile.sr_tile} evict={profile.evict_per_stage} force_cpu={profile.force_cpu}",
        flush=True,
    )

    h, w = rgb.shape[:2]
    sr_clean_1x = None
    feat_rgb = rgb
    sr_hi = None
    work_scale = 1.0

    req_zoom = float(sr_scale) if sr_scale is not None else float(os.environ.get("LUMINA_SR_SCALE", "2.0") or 2.0)
    if req_zoom not in (2.0, 5.0, 10.0):
        req_zoom = 2.0
    eff_zoom = req_zoom

    # 1. Pre-analyze 1x semantic/surface priors (bypassed in core_mode since Ink-Locked Flat input is already clean cel)
    if core_mode:
        parts_pre = np.zeros((4, h, w), np.float32)
        surf_pre = np.zeros((4, h, w), np.float32)
        surf_pre[0] = 1.0
    else:
        sess_part = _get_session(p_part, **sess_kw)
        parts_pre = _run_dense_unet(sess_part, rgb, out_ch=4, hi_weight=0.70, sess_path=p_part) if sess_part is not None else np.zeros((4, h, w), np.float32)
        if profile.evict_per_stage and sess_part is not None:
            release_session(f"{p_part}|cpu={int(profile.force_cpu)}")

        sess_surf = _get_session(p_surf, **sess_kw)
        surf_pre = _run_surface_cnn(sess_surf, rgb, batch_size=profile.surface_batch, sess_path=p_surf) if sess_surf is not None else np.full((4, h, w), 0.25, np.float32)
        if profile.evict_per_stage and sess_surf is not None:
            release_session(f"{p_surf}|cpu={int(profile.force_cpu)}")

    # 1B. Generalized 8-class neural semantic part segmentation (anime_part_seg.onnx, ~10-25ms)
    part_seg_probs = None
    if p_partseg.is_file():
        sess_pseg = _get_session(p_partseg, **sess_kw)
        if sess_pseg is not None:
            part_seg_probs = _run_anime_part_seg_onnx(sess_pseg, rgb, sess_path=p_partseg)
            if profile.evict_per_stage:
                release_session(f"{p_partseg}|cpu={int(profile.force_cpu)}")

    # 2. Whole-image magnification (adaptive dynamic pixel & VRAM cap for low-end PCs)
    zoom_rgb = None
    if p_sr.is_file() and (p_line.is_file() or p_vec.is_file() or p_def.is_file()):
        rgb_for_sr = rgb if core_mode else _pre_sr_denoise_1x(rgb, parts_pre, surf_pre)
        px_1x = h * w
        if px_1x >= 10_000_000:
            work_scale = 1.0
        elif max(h, w) <= 2400 and px_1x * 4 <= profile.max_work_px:
            work_scale = 2.0
        else:
            work_scale = min(1.5, float(np.sqrt(float(profile.max_work_px) / max(1, px_1x))))
            work_scale = max(1.0, round(work_scale, 2))
        eff_zoom = req_zoom if (px_1x * req_zoom * req_zoom <= float(profile.max_zoom_px)) else work_scale
        base_esr_scale = min(4.0, max(work_scale, eff_zoom))
        if base_esr_scale <= 1.05:
            esr_hi = rgb_for_sr
        else:
            sess_sr = _get_session(p_sr, **sess_kw)
            esr_hi = _upscale_sr(
                rgb_for_sr, base_esr_scale, sess_sr,
                tile=profile.sr_tile, pad=profile.sr_pad, sess_path=p_sr,
            ) if sess_sr is not None else cv2.resize(rgb_for_sr, (int(round(w * base_esr_scale)), int(round(h * base_esr_scale))), interpolation=cv2.INTER_CUBIC)
            if profile.evict_per_stage and sess_sr is not None:
                release_session(f"{p_sr}|cpu={int(profile.force_cpu)}")
        if eff_zoom > base_esr_scale:
            zh, zw = int(round(h * eff_zoom)), int(round(w * eff_zoom))
            zoom_rgb = cv2.resize(esr_hi, (zw, zh), interpolation=cv2.INTER_LANCZOS4)
        else:
            zoom_rgb = esr_hi
        wh, ww = int(round(h * work_scale)), int(round(w * work_scale))
        sr_hi = zoom_rgb if zoom_rgb.shape[:2] == (wh, ww) else cv2.resize(zoom_rgb, (ww, wh), interpolation=cv2.INTER_AREA)
        sr_clean_1x = sr_hi if sr_hi.shape[:2] == (h, w) else cv2.resize(sr_hi, (w, h), interpolation=cv2.INTER_AREA)
        feat_rgb = sr_clean_1x

    parts = parts_pre.copy()
    surf = surf_pre

    # 3. DefectRepairNet, ElementGradNet & SR flattening (bypassed in core_mode since Ink-Locked Flat input is already solid flat colors!)
    defect_prob = None
    elem_maps = None
    if core_mode:
        grad_core_95 = np.zeros((h, w), bool)
        line_buffer_5 = np.zeros((h, w), bool)
    else:
        sess_def = _get_session(p_def, **sess_kw)
        if sess_def is not None:
            if sr_hi is not None:
                sr_hi, defect_prob = _run_defect_repair_cnn(sess_def, sr_hi, parts, sess_path=p_def)
            else:
                feat_rgb, defect_prob = _run_defect_repair_cnn(sess_def, feat_rgb, parts, sess_path=p_def)
            if profile.evict_per_stage:
                release_session(f"{p_def}|cpu={int(profile.force_cpu)}")

        pre_flat_1x = cv2.resize(sr_hi, (w, h), interpolation=cv2.INTER_AREA) if sr_hi is not None else feat_rgb
        sess_elem = _get_session(p_elem, **sess_kw)
        elem_maps = _run_dense_unet(sess_elem, pre_flat_1x, out_ch=8, hi_weight=0.55, sess_path=p_elem) if sess_elem is not None else None
        if profile.evict_per_stage and sess_elem is not None:
            release_session(f"{p_elem}|cpu={int(profile.force_cpu)}")
        if elem_maps is not None:
            grad_core_95, line_buffer_5 = _compute_grad_core_and_buffer(elem_maps, parts_pre)
            # Prevent Gradient-CNN from crossing crisp/soft color-step edges (e.g. borderless eyes/iris)
            from .color import srgb_to_oklab
            ok1 = srgb_to_oklab(rgb).astype(np.float32)
            g1h = np.zeros((h, w), np.float32)
            g1v = np.zeros((h, w), np.float32)
            dh = np.sqrt(((ok1[:, 1:, :] - ok1[:, :-1, :]) ** 2).sum(axis=2))
            dv = np.sqrt(((ok1[1:, :, :] - ok1[:-1, :, :]) ** 2).sum(axis=2))
            g1h[:, :-1] = np.maximum(g1h[:, :-1], dh); g1h[:, 1:] = np.maximum(g1h[:, 1:], dh)
            g1v[:-1, :] = np.maximum(g1v[:-1, :], dv); g1v[1:, :] = np.maximum(g1v[1:, :], dv)
            g1_edge = np.hypot(g1h, g1v)
            g2_edge = np.zeros((h, w), np.float32)
            for dy_s, dx_s in ((2, 0), (0, 2), (2, 2), (2, -2)):
                p1 = np.roll(np.roll(ok1, -dy_s, axis=0), -dx_s, axis=1)
                p2 = np.roll(np.roll(ok1, dy_s, axis=0), dx_s, axis=1)
                g2_edge = np.maximum(g2_edge, np.sqrt(((p1 - p2) ** 2).sum(axis=2)))
            real_color_edge = ((g2_edge >= 0.028) & (g1_edge >= 0.010)) | (g1_edge >= 0.028)
            if real_color_edge.any():
                edge_guard = cv2.dilate(real_color_edge.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                invaded = grad_core_95 & edge_guard
                if invaded.any():
                    grad_core_95 &= ~invaded
                    line_buffer_5 |= invaded
        else:
            grad_core_95 = np.zeros((h, w), bool)
            line_buffer_5 = np.zeros((h, w), bool)

        if sr_hi is not None:
            sr_hi = _flatten_sr_for_lines(sr_hi, parts, surf)
            sr_clean_1x = cv2.resize(sr_hi, (w, h), interpolation=cv2.INTER_AREA)
            feat_rgb = sr_clean_1x
        else:
            feat_rgb = _flatten_sr_for_lines(feat_rgb, parts, surf)

    # 4. LineVectorNet (skips 512x512 tiles safely enclosed inside 95%-shrunk gradient cores!)
    vec_fields = None
    skipped_vec_tiles, total_vec_tiles = 0, 0
    sess_vec = _get_session(p_vec, **sess_kw)
    if sess_vec is not None:
        if zoom_rgb is not None and eff_zoom > work_scale:
            wh, ww = sr_hi.shape[:2]
            zh, zw = int(round(h * eff_zoom)), int(round(w * eff_zoom))
            zoom_clean = cv2.resize(sr_hi, (zw, zh), interpolation=cv2.INTER_LANCZOS4)
            sess_def_zoom = _get_session(p_def, **sess_kw) if p_def.is_file() else None
            zoom_clean = _reprocess_after_zoom(zoom_clean, sess_def=sess_def_zoom, parts_patch=parts)
            if profile.evict_per_stage and sess_def_zoom is not None:
                release_session(f"{p_def}|cpu={int(profile.force_cpu)}")
            skip_zoom = cv2.resize(grad_core_95.astype(np.uint8), (zw, zh), interpolation=cv2.INTER_NEAREST) > 0
            vf_zoom, skipped_vec_tiles, total_vec_tiles = _run_dense_unet(sess_vec, zoom_clean, out_ch=9, hi_weight=1.0, skip_mask=skip_zoom, sess_path=p_vec)
            vec_fields = _downsample_zoomed_vf(vf_zoom, wh, ww, zoom_ratio=eff_zoom / work_scale)
            grad_core_hi = cv2.resize(grad_core_95.astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0
            vec_fields = _suppress_dense_highlight_plateaus(vec_fields, sr_hi, grad_core_hi=grad_core_hi)
        else:
            vec_in = sr_hi if sr_hi is not None else feat_rgb
            vh, vw = vec_in.shape[:2]
            grad_core_hi = cv2.resize(grad_core_95.astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST) > 0
            vec_fields, skipped_vec_tiles, total_vec_tiles = _run_dense_unet(sess_vec, vec_in, out_ch=9, hi_weight=1.0, skip_mask=grad_core_hi, sess_path=p_vec)
            if not core_mode:
                vec_fields = _suppress_dense_highlight_plateaus(vec_fields, vec_in, grad_core_hi=grad_core_hi)
        if profile.evict_per_stage:
            release_session(f"{p_vec}|cpu={int(profile.force_cpu)}")
    sess_line = _get_session(p_line, **sess_kw)
    if vec_fields is not None:
        if sess_line is not None:
            # Lightweight 1x DedicatedLineNet pass (~0.8s) preserves colored/pastel hair wisps (lm[1])
            # and crisp stained-glass block boundaries (lm[2]) alongside 2x LineVectorNet.
            line_maps = _run_dense_unet(sess_line, feat_rgb, out_ch=4, hi_weight=0.92, sess_path=p_line)
        else:
            if work_scale == 2.0 and vec_fields.shape[1] == h * 2 and vec_fields.shape[2] == w * 2:
                vf0_1x = vec_fields[0].reshape(h, 2, w, 2).max(axis=(1, 3))
                vf1_1x = vec_fields[1].reshape(h, 2, w, 2).max(axis=(1, 3))
            else:
                k2 = np.ones((2, 2), np.uint8)
                vf0_1x = cv2.resize(cv2.dilate(vec_fields[0], k2), (w, h), interpolation=cv2.INTER_AREA)
                vf1_1x = cv2.resize(cv2.dilate(vec_fields[1], k2), (w, h), interpolation=cv2.INTER_AREA)
            line_maps = np.stack([vf0_1x, vf0_1x * 0.92, vf1_1x, vf0_1x * 0.85], axis=0).astype(np.float32)
    elif sess_line is not None:
        if sr_hi is not None:
            line_hi = _run_dense_unet(sess_line, sr_hi, out_ch=4, hi_weight=0.95, sess_path=p_line)
            if work_scale == 2.0 and line_hi.shape[1] == h * 2 and line_hi.shape[2] == w * 2:
                line_maps = line_hi.reshape(4, h, 2, w, 2).max(axis=(2, 4))
            else:
                k2 = np.ones((2, 2), np.uint8)
                line_maps = np.stack([cv2.resize(cv2.dilate(line_hi[c], k2), (w, h), interpolation=cv2.INTER_AREA) for c in range(4)], axis=0)
        else:
            line_maps = _run_dense_unet(sess_line, feat_rgb, out_ch=4, hi_weight=0.92, sess_path=p_line)
    else:
        line_maps = None
    if profile.evict_per_stage and sess_line is not None:
        release_session(f"{p_line}|cpu={int(profile.force_cpu)}")

    # Keep calibrated FeaturePartUNet channels [0..3] stable, while exposing dedicated
    # LineNet [4, H, W] and ElementGradNet [8, H, W] for line-locking and gradient topology.
    if line_maps is not None:
        parts[0] = np.maximum(parts[0], line_maps[0] * 0.85)
        parts[1] = np.maximum(parts[1], np.maximum(line_maps[1] * 0.85, line_maps[3] * 0.85))

    fg = np.any(rgb < 240, axis=2)
    if not fg.any():
        fg = np.ones((h, w), bool)
    summary = {
        "cel_ratio": round(float(surf[0][fg].mean()), 3),
        "grad_ratio": round(float(surf[1][fg].mean()), 3),
        "grain_ratio": round(float(surf[2][fg].mean()), 3),
        "detail_ratio": round(float(surf[3][fg].mean()), 3),
        "delicate_area": round(float((parts[2] > 0.45)[fg].mean()), 4),
        "wash_area": round(float((parts[3] > 0.50)[fg].mean()), 3),
        "grad_core_95": round(float(grad_core_95[fg].mean()), 3),
        "line_buf_5": round(float(line_buffer_5[fg].mean()), 3),
        "vec_tiles_skipped": int(skipped_vec_tiles),
        "vec_tiles_total": int(total_vec_tiles),
        "defect_area": round(float((defect_prob >= 0.35).mean()), 5) if defect_prob is not None else 0.0,
        "line_net": line_maps is not None,
        "line_vec": vec_fields is not None,
        "elem_net": elem_maps is not None,
        "defect_net": defect_prob is not None,
        "part_seg_net": part_seg_probs is not None,
        "core_mode": bool(core_mode),
        "zoom_scale": eff_zoom,
        "vram_budget_mb": profile.vram_budget_mb,
        "sr_tile": profile.sr_tile,
    }
    if not core_mode and sr_hi is not None and vec_fields is not None:
        sr_hi = _restore_tone_1x(sr_hi, rgb)
        sr_clean_1x = cv2.resize(sr_hi, (w, h), interpolation=cv2.INTER_AREA)
    return FeatureMaps(
        ink_main_geom=parts[0],
        fine_strand=parts[1],
        delicate_detail=parts[2],
        mergeable_wash=parts[3],
        surface_probs=surf,
        summary=summary,
        line_net_maps=line_maps,
        elem_grad_maps=elem_maps,
        sr_clean_1x=sr_clean_1x,
        line_vector_fields=vec_fields,
        sr_hi_rgb=sr_hi if vec_fields is not None else None,
        sr_scale=work_scale if vec_fields is not None else 1.0,
        defect_prob=defect_prob,
        grad_core_95=grad_core_95,
        line_buffer_5=line_buffer_5,
        part_seg_probs=part_seg_probs,
    )


def realesr_dejpeg_onnx(
    rgb: np.ndarray,
    fmaps: FeatureMaps | None = None,
    weights_dir: str | Path | None = None,
    pre_down: float | None = None,
) -> np.ndarray | None:
    """Eliminate JPEG ringing and block noise using lightweight realesr_anime.onnx (~2.4 MB)
    guided by CNN delicate_detail protection (reuses pre-computed `fmaps.sr_clean_1x` when present)."""
    h, w = rgb.shape[:2]
    if fmaps is not None and getattr(fmaps, "sr_clean_1x", None) is not None and fmaps.sr_clean_1x.shape[:2] == (h, w):
        clean_full = fmaps.sr_clean_1x
    else:
        wdir = Path(weights_dir) if weights_dir else Path(os.environ.get("LUMINA_WEIGHTS_DIR", str(_WEIGHTS_DEFAULT)))
        profile = compute_runtime_profile()
        cuda_lim = int(profile.vram_budget_mb * 1024 * 1024) if profile.vram_budget_mb > 0 else None
        p_sr = wdir / "realesr_anime.onnx"
        sess = _get_session(p_sr, force_cpu=profile.force_cpu, cuda_mem_limit_bytes=cuda_lim, low_vram=profile.evict_per_stage)
        if sess is None:
            return None

        if pre_down is None:
            pre_down = 1.0 if min(h, w) <= 720 else (1.25 if min(h, w) <= 1100 else 1.6)
        sw = max(64, round(w / pre_down))
        sh = max(64, round(h / pre_down))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)

        clean_u8 = _upscale_sr(small, 4.0, sess, tile=profile.sr_tile, pad=profile.sr_pad, sess_path=p_sr)
        if profile.evict_per_stage:
            release_session(f"{p_sr}|cpu={int(profile.force_cpu)}")
        clean_full = cv2.resize(clean_u8, (w, h), interpolation=cv2.INTER_AREA)
    if fmaps is None:
        return clean_full
    # Pixel-wise CNN routing: de-JPEG strength follows surf_grain while protecting ink, fine hair
    # strands, and delicate facial zones so crisp thin wisps never get blurred.
    protect = np.clip(np.maximum(fmaps.ink_main_geom, np.maximum(fmaps.fine_strand, fmaps.delicate_detail)), 0.0, 1.0)
    grain_w = np.clip(fmaps.surface_probs[2] * 1.15, 0.0, 0.90)
    return _blend_chunked_u8(rgb, clean_full, grain_w * (1.0 - 0.85 * protect))


def repair_image_defects_onnx(
    rgb: np.ndarray,
    weights_dir: str | Path | None = None,
) -> tuple[np.ndarray, float]:
    """Run feature-guided `defect_repair_net.onnx` directly on an RGB image before SVG conversion.

    Evaluates at 2x scale (matching training resolution of `DefectRepairNet`) and composites repaired
    pixels strictly back onto original `rgb` only at active defect masks (`defect_prob >= 0.35`) so
    clean non-defect regions stay bit-exact. Safe no-op returning `(rgb, 0.0)` when weights are absent.
    """
    wdir = Path(weights_dir) if weights_dir else Path(os.environ.get("LUMINA_WEIGHTS_DIR", str(_WEIGHTS_DEFAULT)))
    profile = compute_runtime_profile()
    cuda_lim = int(profile.vram_budget_mb * 1024 * 1024) if profile.vram_budget_mb > 0 else None
    sess_kw = dict(force_cpu=profile.force_cpu, cuda_mem_limit_bytes=cuda_lim, low_vram=profile.evict_per_stage)
    p_def = wdir / "defect_repair_net.onnx"
    p_part = wdir / "feature_part.onnx"
    sess_def = _get_session(p_def, **sess_kw)
    if sess_def is None:
        return rgb, 0.0
    sess_part = _get_session(p_part, **sess_kw)
    h, w = rgb.shape[:2]
    parts = _run_dense_unet(sess_part, rgb, out_ch=4, hi_weight=0.70, sess_path=p_part) if sess_part is not None else np.zeros((4, h, w), np.float32)
    if profile.evict_per_stage and sess_part is not None:
        release_session(f"{p_part}|cpu={int(profile.force_cpu)}")

    px_1x = h * w
    sr_scale = 1.0 if px_1x >= 10_000_000 else (2.0 if (max(h, w) <= 2400 and px_1x * 4 <= profile.max_work_px) else 1.5)
    sh, sw = int(round(h * sr_scale)), int(round(w * sr_scale))
    rgb_hi = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_CUBIC) if (sh, sw) != (h, w) else rgb
    rep_hi, d_prob_hi = _run_defect_repair_cnn(sess_def, rgb_hi, parts, sess_path=p_def)
    if profile.evict_per_stage:
        release_session(f"{p_def}|cpu={int(profile.force_cpu)}")

    d_prob = cv2.resize(d_prob_hi, (w, h), interpolation=cv2.INTER_AREA)
    d_prob *= (1.0 - np.clip((parts[2] - 0.32) / 0.25, 0.0, 1.0))
    rep_1x = cv2.resize(rep_hi, (w, h), interpolation=cv2.INTER_AREA)
    out = _blend_chunked_u8(rgb, rep_1x, np.clip((d_prob - 0.35) / 0.45, 0.0, 1.0))
    return out, float((d_prob >= 0.35).mean())


def render_vector_heatmap_png(
    rgb: np.ndarray,
    out_path: str | Path,
    vec_fields: np.ndarray | None = None,
    weights_dir: str | Path | None = None,
) -> bool:
    """Render an INFERNO probability heatmap of `LineVectorNet` stroke_prob [0] (with subtle
    cyan color-block seam cues from channel [1]) at native 1x resolution to PNG."""
    h, w = rgb.shape[:2]
    if vec_fields is None:
        fmaps = infer_feature_maps(rgb, weights_dir=weights_dir)
        if fmaps is None:
            return False
        vec_fields = fmaps.line_vector_fields
        if vec_fields is None and fmaps.line_net_maps is not None:
            prob = np.max(fmaps.line_net_maps[:2], axis=0)
            cbound = fmaps.line_net_maps[2]
        elif vec_fields is None:
            return False
        else:
            prob = vec_fields[0]
            cbound = vec_fields[1]
    else:
        prob = vec_fields[0]
        cbound = vec_fields[1]

    if prob.shape[:2] != (h, w):
        prob_1x = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        cb_1x = cv2.resize(cbound.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    else:
        prob_1x = prob.astype(np.float32)
        cb_1x = cbound.astype(np.float32)

    u8 = np.rint(np.clip(prob_1x, 0.0, 1.0) * 255.0).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO).astype(np.float32)
    # Highlight un-outlined color-block seam walls in subtle cyan where stroke_prob is low
    seam = np.clip(cb_1x * (1.0 - prob_1x * 1.4), 0.0, 1.0)
    hm_bgr[..., 0] = np.clip(hm_bgr[..., 0] + seam * 155.0, 0.0, 255.0)
    hm_bgr[..., 1] = np.clip(hm_bgr[..., 1] + seam * 115.0, 0.0, 255.0)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_p), np.rint(hm_bgr).astype(np.uint8))
    return bool(ok)
