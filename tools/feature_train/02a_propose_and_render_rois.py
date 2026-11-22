"""Phase 2a: Extract initial candidate facial/detail ROIs (eyes, mouths, blush, text/emblems)
and sample surface patches across the 49 dataset images, rendering visual verification sheets."""
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


def _skin_mask(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    # Broad anime skin range (fair peach through warm tan)
    return (L > 0.58) & (L < 0.97) & (a > 0.004) & (a < 0.11) & (b > 0.002) & (b < 0.12)


def propose_detail_boxes(rgb: np.ndarray, anime_map: np.ndarray, manga_map: np.ndarray):
    """Propose candidate (x, y, w, h, kind) ROIs for facial features, blush, and text/accents."""
    h, w = rgb.shape[:2]
    lab = srgb_to_oklab(rgb)
    skin = _skin_mask(lab)
    boxes = []

    # 1. Connected skin regions (face / neck / hands); look inside larger skin regions for eyes/mouth/blush
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin_filled = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE, k_close) > 0
    num, lbl, stats, cents = cv2.connectedComponentsWithStats(skin_filled.astype(np.uint8))

    # Candidate eye blobs: compact non-skin regions enclosed or nearly enclosed by skin + high internal edge energy
    non_skin_in_skin = skin_filled & (~skin)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_skin_in_skin = cv2.morphologyEx(non_skin_in_skin.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    n_c, c_lbl, c_stats, _ = cv2.connectedComponentsWithStats(non_skin_in_skin.astype(np.uint8))

    min_dim = max(12, int(0.015 * min(h, w)))
    max_dim = int(0.28 * min(h, w))
    line_comb = np.maximum(anime_map, manga_map).astype(np.float32) / 255.0
    for i in range(1, n_c):
        bx, by, bw, bh, area = [int(v) for v in c_stats[i]]
        if bw < min_dim or bh < min_dim or bw > max_dim or bh > max_dim:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.5 or area < min_dim * min_dim * 0.4:
            continue
        # Require strong anime/manga line presence + specular bright pixel or dark core inside ROI
        roi_line = line_comb[by:by+bh, bx:bx+bw]
        roi_L = lab[by:by+bh, bx:bx+bw, 0]
        if float(np.mean(roi_line > 0.25)) < 0.08:
            continue
        if float(np.max(roi_L) - np.min(roi_L)) < 0.28:
            continue
        pad = max(6, int(0.15 * max(bw, bh)))
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        boxes.append((x0, y0, x1 - x0, y1 - y0, "eye_or_mouth"))

    # 2. Text / high-density isolated graphics regions (white background / bubble strokes)
    txt_mask = (manga_map > 100) & (lab[..., 0] < 0.7)
    k_text = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 9))
    txt_grp = cv2.dilate(txt_mask.astype(np.uint8), k_text)
    nt, tl, ts, _ = cv2.connectedComponentsWithStats(txt_grp)
    # Keep only compact isolated stroke groups (not huge hair/outline components)
    for i in range(1, nt):
        bx, by, bw, bh, area = [int(v) for v in ts[i]]
        if bw < 20 or bh < 14 or bw > w * 0.45 or bh > h * 0.35:
            continue
        # Check border of dilated box is mostly uniform white/light paper
        border_vals = np.concatenate([
            lab[max(0, by-2):by+1, bx:bx+bw, 0].ravel(),
            lab[by+bh:min(h, by+bh+2), bx:bx+bw, 0].ravel(),
        ])
        if len(border_vals) > 0 and float(np.mean(border_vals > 0.90)) > 0.78:
            boxes.append((bx, by, bw, bh, "isolated_accent"))

    # Deduplicate overlapping boxes (keep tighter / higher confidence)
    kept = []
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    for b in boxes:
        bx, by, bw, bh, _ = b
        overlap = False
        for kx, ky, kw, kh, _ in kept:
            ix0, iy0 = max(bx, kx), max(by, ky)
            ix1, iy1 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter > 0.45 * min(bw * bh, kw * kh):
                    overlap = True
                    break
        if not overlap:
            kept.append(b)
    return kept


def main():
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    all_rois = {}
    sheet_imgs = []
    for item in manifest:
        rgb = np.asarray(Image.open(item["rgb_path"]).convert("RGB"))
        npz = np.load(item["teacher_path"])
        boxes = propose_detail_boxes(rgb, npz["anime"], npz["manga"])
        all_rois[item["stem"]] = boxes

        vis = Image.fromarray(rgb)
        dr = ImageDraw.Draw(vis)
        for bx, by, bw, bh, kind in boxes:
            col = (255, 40, 40) if kind == "eye_or_mouth" else (40, 140, 255)
            dr.rectangle([bx, by, bx + bw, by + bh], outline=col, width=max(2, rgb.shape[1] // 250))
        vis.thumbnail((260, 240), Image.LANCZOS)
        sheet_imgs.append((item["stem"], len(boxes), vis))

    (CACHE_DIR / "proposed_rois.json").write_text(json.dumps(all_rois, ensure_ascii=False, indent=2), encoding="utf-8")
    for s_idx, chunk in enumerate([sheet_imgs[:25], sheet_imgs[25:]], 1):
        cols, rows = 5, 5
        tw, th = 270, 270
        canvas = Image.new("RGB", (cols * tw, rows * th), (240, 240, 240))
        dr = ImageDraw.Draw(canvas)
        for i, (stem, n_b, im) in enumerate(chunk):
            r, c = divmod(i, cols)
            canvas.paste(im, (c * tw + (tw - im.width) // 2, r * th + 22 + (th - 26 - im.height) // 2))
            dr.text((c * tw + 6, r * th + 4), f"{stem[:14]} ({n_b})", fill=(200, 0, 0))
        out = CACHE_DIR / f"roi_review_{s_idx}.jpg"
        canvas.save(out, quality=88)
        print("Saved ROI sheet:", out, flush=True)


if __name__ == "__main__":
    main()
