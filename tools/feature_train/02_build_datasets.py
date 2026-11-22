"""Phase 2: Build curated training datasets for SurfaceStyleCNN (96x96 patches)
and FeaturePartUNet (4-channel dense spatial feature maps) from the 49 dataset images."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lumina_vec.color import srgb_to_oklab

CACHE_DIR = ROOT / "tools" / "feature_train" / "work_cache"
DS_DIR = CACHE_DIR / "datasets"

STAINED_GLASS_STEMS = {
    "test9", "test11", "test15", "test31", "test33", "test34", "test37", "test38", "test39", "test45",
}


def _structure_coherence(lab: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.2)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.2)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.2)
    diff = jxx - jyy
    num = np.sqrt(diff * diff + 4.0 * jxy * jxy)
    den = jxx + jyy + 1e-6
    return np.clip(num / den, 0.0, 1.0)


def detect_facial_delicate(rgb: np.ndarray, edge_hint: np.ndarray, fg: np.ndarray | None = None, stem: str = "") -> np.ndarray:
    """Detect compact facial & micro-detail structures (eyes/iris/highlights, mouths, inner cheek
    blush) cleanly without misclassifying arm shadow rims, white-bg triangles, or kimonos."""
    from scipy.ndimage import binary_fill_holes, distance_transform_edt
    h, w = rgb.shape[:2]
    lab = srgb_to_oklab(rgb).astype(np.float32)
    L, a_c, b_c = lab[..., 0], lab[..., 1], lab[..., 2]
    if fg is None:
        fg = np.any(rgb < 246, axis=2)
    ch2 = np.zeros((h, w), np.float32)
    sob_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(sob_x, sob_y)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ridge_L = np.maximum(cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k5), cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k5))

    if stem not in STAINED_GLASS_STEMS:
        chroma = np.hypot(a_c, b_c)
        # Anime skin spans light neutral-warm ivory through soft shaded peach (avoid pure white paper).
        skin = fg & (L > 0.60) & (L < 0.975) & (a_c > -0.002) & (a_c < 0.095) & (b_c > -0.005) & (b_c < 0.105) & (chroma < 0.115)
        # Any compact non-skin hole bounded predominantly (or on bottom+sides) by skin is a facial feature.
        holes = ~skin
        holes = cv2.morphologyEx(holes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        nh, lh, sth, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
        min_d = max(7, int(0.009 * min(h, w)))
        max_d = int(0.16 * min(h, w))
        for i in range(1, nh):
            bx, by, bw, bh, area = [int(v) for v in sth[i]]
            if bw < min_d or bh < min_d or bw > max_d or bh > max_d:
                continue
            if area > 0.018 * h * w or (area / max(1, bw * bh)) < 0.16:
                continue
            r0, c0 = max(0, by - 4), max(0, bx - 4)
            r1, c1 = min(h, by + bh + 4), min(w, bx + bw + 4)
            ring = np.ones((r1 - r0, c1 - c0), bool)
            ring[by - r0:by + bh - r0, bx - c0:bx + bw - c0] = False
            ring_skin = skin[r0:r1, c0:c1][ring]
            lower_ring = ring.copy()
            lower_ring[:max(1, (by - r0) + bh // 3), :] = False
            lower_skin = skin[r0:r1, c0:c1][lower_ring]
            ok_ring = (ring_skin.size and float(np.mean(ring_skin)) >= 0.42) or (
                lower_skin.size and float(np.mean(lower_skin)) >= 0.52
            )
            if not ok_ring:
                continue
            sub_L = L[by:by+bh, bx:bx+bw]
            sub_ab = lab[by:by+bh, bx:bx+bw, 1:]
            c_span = float(np.max(np.ptp(sub_ab, axis=(0, 1))))
            sub_ln = edge_hint[by:by+bh, bx:bx+bw]
            if (float(np.max(sub_L) - np.min(sub_L)) >= 0.22 or c_span >= 0.035) and float(np.mean(sub_ln > 0.15)) >= 0.05:
                pad = max(4, int(0.14 * max(bw, bh)))
                sub_m = cv2.dilate((lh[by:by+bh, bx:bx+bw] == i).astype(np.uint8), np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
                ch2[by:by+bh, bx:bx+bw] = np.maximum(ch2[by:by+bh, bx:bx+bw], sub_m.astype(np.float32) * 0.95)

        # Cheek blush & glints strictly well INSIDE solid skin (>= 10 px from any non-skin edge!)
        skin_dist = distance_transform_edt(skin)
        skin_bg_a = cv2.GaussianBlur(a_c, (0, 0), max(12.0, min(h, w) * 0.025))
        blush_pink = skin & (skin_dist >= max(10.0, 0.015 * min(h, w))) & ((a_c - skin_bg_a) > 0.016) & (a_c > 0.030) & (L > 0.68)
        if blush_pink.any():
            ch2 = np.maximum(ch2, cv2.GaussianBlur(blush_pink.astype(np.float32), (0, 0), 2.5) * 0.85)

        glint = skin & (skin_dist >= 4.0) & (ridge_L > 0.08) & (grad > 0.08)
        ch2 = np.maximum(ch2, cv2.dilate(glint.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32) * 0.80)

    return np.clip(ch2, 0.0, 1.0)


def build_spatial_targets(rgb: np.ndarray, anime_u8: np.ndarray, manga_u8: np.ndarray,
                          alpha_u8: np.ndarray, stem: str, style: str) -> np.ndarray:
    """Build [4, H, W] float32 targets in [0, 1]:
      ch0: ink_main_geom    (bold outlines, leadlines, strong dark structural lines)
      ch1: fine_strand      (internal hair strands, thin coloured lines, lace/veins)
      ch2: delicate_detail  (eyes/iris/highlights, mouths, cheek blush, small text/gems)
      ch3: mergeable_wash   (smooth gradients & grain washes safe for stained-glass merge)
    """
    h, w = rgb.shape[:2]
    lab = srgb_to_oklab(rgb).astype(np.float32)
    L, a_c, b_c = lab[..., 0], lab[..., 1], lab[..., 2]
    anime = anime_u8.astype(np.float32) / 255.0
    manga = manga_u8.astype(np.float32) / 255.0
    fg = alpha_u8 > 128

    # ch0: main ink & bold structural edges (dark stroke + teacher corroboration)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    local_max_L = cv2.dilate(L, k7)
    dark_drop = np.clip((local_max_L - L) / 0.25, 0.0, 1.0)
    dark_ink = np.clip((0.35 - L) / 0.25, 0.0, 1.0)
    ch0 = np.clip(0.55 * manga + 0.45 * np.maximum(anime * dark_drop, dark_ink * (anime > 0.20)), 0.0, 1.0)
    ch0 = np.where(fg, ch0, 0.0)

    # ch1: fine internal strands & coloured thin ridges
    coh = _structure_coherence(lab)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ridge_L = np.maximum(cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k5), cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k5))
    ridge_C = np.hypot(
        np.maximum(cv2.morphologyEx(a_c, cv2.MORPH_BLACKHAT, k5), cv2.morphologyEx(a_c, cv2.MORPH_TOPHAT, k5)),
        np.maximum(cv2.morphologyEx(b_c, cv2.MORPH_BLACKHAT, k5), cv2.morphologyEx(b_c, cv2.MORPH_TOPHAT, k5)),
    )
    ridge_score = np.clip((ridge_L + 1.8 * ridge_C - 0.015) / 0.08, 0.0, 1.0) * coh
    ch1 = np.clip(np.maximum(anime * (1.0 - 0.5 * ch0), ridge_score * 0.9), 0.0, 1.0)
    ch1 = np.where(fg, ch1, 0.0)

    ch2 = detect_facial_delicate(rgb, np.maximum(anime, manga), fg=fg, stem=stem)

    # ch3: mergeable wash (soft gradients & painterly grain outside linework and delicate details)
    grad_mag = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    grad_c = np.hypot(cv2.Sobel(a_c, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b_c, cv2.CV_32F, 0, 1, ksize=3))
    local_step = grad_mag + 1.5 * grad_c
    line_block = np.clip(np.maximum(ch0, np.maximum(ch1, ch2)), 0.0, 1.0)
    if style == "clean_cel":
        # Clean cel only merges very gentle smooth gradients (if any), NOT flat cel features
        ch3 = np.where(fg & (local_step > 0.008) & (local_step < 0.06) & (line_block < 0.25), 0.35, 0.0).astype(np.float32)
    elif style == "soft_gradient":
        ch3 = np.where(fg & (local_step < 0.14), 1.0 - line_block, 0.15 * (1.0 - line_block)).astype(np.float32)
    else:  # grain_paint or mixed
        ch3 = np.where(fg & (local_step < 0.18), 0.9 * (1.0 - line_block), 0.20 * (1.0 - line_block)).astype(np.float32)
    ch3 = cv2.GaussianBlur(np.clip(ch3, 0.0, 1.0), (0, 0), 2.0) * (1.0 - 0.85 * line_block)

    return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)


def _sample_surface_patches(manifest, rng):
    """Extract curated 96x96 labeled patches [clean_cel, soft_gradient, grain_paint, fine_detail]."""
    patches_by_cls = {0: [], 1: [], 2: [], 3: []}
    PS = 96
    for item in manifest:
        rgb = np.asarray(Image.open(item["rgb_path"]).convert("RGB"))
        npz = np.load(item["teacher_path"])
        anime = npz["anime"].astype(np.float32) / 255.0
        manga = npz["manga"].astype(np.float32) / 255.0
        alpha = npz["alpha"] > 128
        lab = srgb_to_oklab(rgb).astype(np.float32)
        L = lab[..., 0]
        lap = np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3))
        style = item["style"]

        for scale in (1.0, 0.6):
            if scale != 1.0:
                nh, nw = max(PS + 8, round(rgb.shape[0] * scale)), max(PS + 8, round(rgb.shape[1] * scale))
                r_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                a_s = cv2.resize(anime, (nw, nh), interpolation=cv2.INTER_AREA)
                m_s = cv2.resize(manga, (nw, nh), interpolation=cv2.INTER_AREA)
                al_s = cv2.resize(alpha.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
                l_s = cv2.resize(L, (nw, nh), interpolation=cv2.INTER_AREA)
                lp_s = np.abs(cv2.Laplacian(l_s, cv2.CV_32F, ksize=3))
                lab_s = srgb_to_oklab(r_s).astype(np.float32)
            else:
                r_s, a_s, m_s, al_s, l_s, lp_s, lab_s = rgb, anime, manga, alpha, L, lap, lab

            H, W = r_s.shape[:2]
            if H < PS or W < PS:
                continue
            for _ in range(95 if style == "clean_cel" else 65):
                y0 = int(rng.integers(0, H - PS + 1))
                x0 = int(rng.integers(0, W - PS + 1))
                fg_frac = float(np.mean(al_s[y0:y0+PS, x0:x0+PS] & (l_s[y0:y0+PS, x0:x0+PS] < 0.985)))
                if fg_frac < 0.18:
                    continue
                patch = r_s[y0:y0+PS, x0:x0+PS]
                ln_mean = float(np.mean(np.maximum(a_s[y0:y0+PS, x0:x0+PS], m_s[y0:y0+PS, x0:x0+PS]) > 0.25))
                lp_patch = lp_s[y0:y0+PS, x0:x0+PS]
                lp_med = float(np.median(lp_patch))
                lp_q80 = float(np.quantile(lp_patch, 0.80))
                ptp_lab = float(np.max(np.ptp(lab_s[y0:y0+PS, x0:x0+PS], axis=(0, 1))))
                # Distinguish flat hard-edge steps (mostly zero Laplacian interior) vs continuous gradient ramp
                flat_frac = float(np.mean(lp_patch < 0.002))

                if ln_mean > 0.15 or (ln_mean > 0.09 and lp_q80 > 0.045):
                    cls = 3  # fine_detail
                elif (style == "clean_cel" or flat_frac > 0.72) and lp_med < 0.004 and ln_mean < 0.12:
                    cls = 0  # clean_cel
                elif lp_med < 0.007 and flat_frac <= 0.70 and ptp_lab > 0.035 and ln_mean < 0.07:
                    cls = 1  # soft_gradient
                elif (style in ("grain_paint", "mixed") or lp_med >= 0.007) and ln_mean < 0.11:
                    cls = 2  # grain_paint
                elif style == "soft_gradient" and ln_mean < 0.08 and flat_frac <= 0.72:
                    cls = 1
                else:
                    continue

                if len(patches_by_cls[cls]) < 1500:
                    patches_by_cls[cls].append((item["id"], patch))
    return patches_by_cls


def main():
    DS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))

    # Train/Val split by image ID (41 train / 8 val) so val measures unseen generalization
    val_ids = {2, 7, 14, 19, 27, 33, 41, 46}

    # 1. SurfaceStyleCNN patch dataset
    print("Extracting surface patches...", flush=True)
    patches = _sample_surface_patches(manifest, rng)
    print("Patch counts per class:", {k: len(v) for k, v in patches.items()}, flush=True)

    # Save visual gallery for verification (4 classes x 12 patches)
    cls_names = ["clean_cel", "soft_gradient", "grain_paint", "fine_detail"]
    gal = Image.new("RGB", (12 * 100, 4 * 120), (245, 245, 245))
    dr = ImageDraw.Draw(gal)
    X_list, Y_list, V_list = [], [], []
    n_cap = min(len(v) for v in patches.values())
    for c in range(4):
        items_c = patches[c]
        idxs = rng.permutation(len(items_c))[:min(len(items_c), max(n_cap, 900))]
        for k_i, j in enumerate(idxs):
            img_id, patch = items_c[j]
            X_list.append(patch)
            Y_list.append(c)
            V_list.append(1 if img_id in val_ids else 0)
            if k_i < 12:
                gal.paste(Image.fromarray(patch), (k_i * 100 + 2, c * 120 + 20))
        dr.text((6, c * 120 + 3), f"{c}: {cls_names[c]} (n={len(idxs)})", fill=(200, 0, 0))
    gal.save(DS_DIR / "crop_gallery_surface.jpg", quality=90)
    np.savez_compressed(
        DS_DIR / "surface_patches.npz",
        X=np.asarray(X_list, np.uint8),
        y=np.asarray(Y_list, np.int64),
        is_val=np.asarray(V_list, np.uint8),
    )

    # 2. FeaturePartUNet multi-scale spatial targets (512x512 tiles + full resized tiles)
    print("Building FeaturePartUNet 4-ch spatial targets...", flush=True)
    tile_dir = DS_DIR / "part_tiles"
    tile_dir.mkdir(exist_ok=True)
    tile_meta = []
    preview_rows = []
    preview_ids = {0, 2, 5, 10, 14, 18, 24, 30, 35, 42, 46, 48}

    for item in manifest:
        rgb = np.asarray(Image.open(item["rgb_path"]).convert("RGB"))
        npz = np.load(item["teacher_path"])
        targets = build_spatial_targets(rgb, npz["anime"], npz["manga"], npz["alpha"], item["stem"], item["style"])
        h, w = rgb.shape[:2]
        is_val = 1 if item["id"] in val_ids else 0

        if item["id"] in preview_ids and len(preview_rows) < 10:
            tw, th = 160, 160
            row = [Image.fromarray(rgb).resize((tw, th), Image.LANCZOS)]
            for c_i in range(4):
                hm = (np.clip(targets[c_i], 0, 1) * 255).astype(np.uint8)
                col_hm = cv2.applyColorMap(cv2.resize(hm, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLORMAP_INFERNO)
                row.append(Image.fromarray(col_hm[..., ::-1]))
            preview_rows.append((item["stem"], row))

        # Save whole downscaled 512x512 view + 6 random/detail-rich 512x512 crops
        # Full view (padded/resized to 512x512) captures global layout
        r_full = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        t_full = np.stack([cv2.resize(targets[c], (512, 512), interpolation=cv2.INTER_AREA) for c in range(4)], axis=0)
        p_full = tile_dir / f"{item['id']:03d}_full.npz"
        np.savez_compressed(p_full, x=r_full, y=(np.clip(t_full, 0, 1) * 255).astype(np.uint8))
        tile_meta.append({"path": str(p_full), "is_val": is_val, "img_id": item["id"]})

        # Extract 6 native-scale 512x512 crops biased toward detail/edge zones
        if min(h, w) >= 512:
            detail_score = targets[2] + 0.6 * targets[1] + 0.3 * targets[0]
            for k_c in range(6):
                if k_c < 3 and np.max(detail_score) > 0.2:
                    # Pick high-detail center
                    ys, xs = np.where(detail_score > np.quantile(detail_score, 0.90))
                    pick = int(rng.integers(0, len(ys)))
                    cy, cx = int(ys[pick]), int(xs[pick])
                    y0 = int(np.clip(cy - 256, 0, h - 512))
                    x0 = int(np.clip(cx - 256, 0, w - 512))
                else:
                    y0 = int(rng.integers(0, h - 512 + 1))
                    x0 = int(rng.integers(0, w - 512 + 1))
                sub_r = rgb[y0:y0+512, x0:x0+512]
                sub_t = targets[:, y0:y0+512, x0:x0+512]
                p_sub = tile_dir / f"{item['id']:03d}_c{k_c}.npz"
                np.savez_compressed(p_sub, x=sub_r, y=(np.clip(sub_t, 0, 1) * 255).astype(np.uint8))
                tile_meta.append({"path": str(p_sub), "is_val": is_val, "img_id": item["id"]})

    (DS_DIR / "part_tiles.json").write_text(json.dumps(tile_meta, indent=2), encoding="utf-8")
    # Save visual preview grid for FeaturePartUNet targets
    if preview_rows:
        grid = Image.new("RGB", (5 * 164, len(preview_rows) * 182), (255, 255, 255))
        dr2 = ImageDraw.Draw(grid)
        hdrs = ["RGB", "ch0:ink_geom", "ch1:fine_strand", "ch2:delicate_detail", "ch3:merge_wash"]
        for r_i, (stem, ims) in enumerate(preview_rows):
            for c_i, im in enumerate(ims):
                grid.paste(im, (c_i * 164 + 2, r_i * 182 + 18))
                lbl = f"{stem[:10]}" if c_i == 0 else hdrs[c_i]
                dr2.text((c_i * 164 + 4, r_i * 182 + 2), lbl, fill=(0, 0, 0))
        grid.save(DS_DIR / "feature_targets_preview.jpg", quality=90)
    print(f"Done! Total FeaturePart tiles: {len(tile_meta)}", flush=True)


if __name__ == "__main__":
    main()
