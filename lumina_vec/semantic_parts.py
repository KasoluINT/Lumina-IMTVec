"""Generalized 8-class neural semantic part segmentation & per-subject region budgeting.

Consumes the pure-ONNX distilled `weights/anime_part_seg.onnx` 8-channel probability maps
(`fmaps.part_seg_probs`, trained via color-invariant soft distillation from Apache-2.0
`see-through` / `24yearsold/l2d_sam_iter2`) to segment a planar partition into 8 hard-bounded
semantic visual subjects, so every subject receives its own independent region & color budget
without bleeding across subject boundaries:

  - `PART_BG (0)`          : background / scene wash
  - `PART_HAIR (1)`        : hair masses (dedicated interior hair-strand consolidation budget)
  - `PART_HEADWEAR (2)`    : headwear, flowers, hairpins, ribbons, earwear, pearls (strictly locked)
  - `PART_FACE_SKIN (3)`   : face, ears, nose, neck skin (strictly locked)
  - `PART_EYES_MOUTH (4)`  : eyes, pupils, eyelashes, eyewear, mouth (MAXIMUM lock)
  - `PART_NECK_HAND (5)`   : neckwear, scarves, collars, gloves, handwear (preserved)
  - `PART_CLOTH (6)`       : topwear, bottomwear, legwear, footwear (balanced compaction)
  - `PART_OBJECTS (7)`     : held props, tails, wings, accessories (preserved)
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import srgb_to_oklab

PART_BG = 0
PART_HAIR = 1
PART_HEADWEAR = 2
PART_FACE_SKIN = 3
PART_EYES_MOUTH = 4
PART_NECK_HAND = 5
PART_CLOTH = 6
PART_OBJECTS = 7

# Semantic priority order (later overwrites earlier on multi-label overlaps)
PART_PRIORITY = (
    PART_OBJECTS,
    PART_CLOTH,
    PART_HAIR,
    PART_NECK_HAND,
    PART_FACE_SKIN,
    PART_HEADWEAR,
    PART_EYES_MOUTH,
)

PART_THRESH = {
    PART_BG: 0.50,
    PART_HAIR: 0.44,
    PART_HEADWEAR: 0.42,
    PART_FACE_SKIN: 0.44,
    PART_EYES_MOUTH: 0.38,
    PART_NECK_HAND: 0.44,
    PART_CLOTH: 0.45,
    PART_OBJECTS: 0.45,
}

PART_NAMES = {
    PART_BG: "bg",
    PART_HAIR: "hair",
    PART_HEADWEAR: "headwear",
    PART_FACE_SKIN: "face_skin",
    PART_EYES_MOUTH: "eyes_mouth",
    PART_NECK_HAND: "neck_handwear",
    PART_CLOTH: "clothes",
    PART_OBJECTS: "objects_extra",
}

PART_VIS_RGB = np.array(
    [
        [32, 36, 46],     # 0: bg (dark slate)
        [140, 75, 225],   # 1: hair (vivid purple)
        [245, 55, 85],    # 2: headwear/ornament (crimson red)
        [255, 205, 145],  # 3: face_skin (warm peach)
        [255, 235, 45],   # 4: eyes_mouth (bright yellow)
        [45, 215, 185],   # 5: neck_handwear (teal)
        [60, 145, 240],   # 6: clothes (royal blue)
        [135, 225, 70],   # 7: objects_extra (lime green)
    ],
    dtype=np.uint8,
)


def _resize_map(arr: np.ndarray | None, hs: int, ws: int) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.shape[:2] == (hs, ws):
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr.astype(np.float32), (ws, hs), interpolation=cv2.INTER_LINEAR)


def _build_adjacency(regions: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_pairs = []
    for p1, p2 in ((regions[:-1], regions[1:]), (regions[:, :-1], regions[:, 1:])):
        diff = (p1 != p2) & (p1 >= 0) & (p2 >= 0)
        if not diff.any():
            continue
        va = p1[diff].astype(np.int64)
        vb = p2[diff].astype(np.int64)
        k_pairs.append((va << 32) | vb)
        k_pairs.append((vb << 32) | va)
    if not k_pairs:
        z32 = np.zeros(0, dtype=np.int32)
        return z32, z32, np.zeros(0, dtype=np.int64), np.ones(n, dtype=np.float64)
    uniq_e, edge_w = np.unique(np.concatenate(k_pairs), return_counts=True)
    e_src = (uniq_e >> 32).astype(np.int32)
    e_dst = (uniq_e & 0xFFFFFFFF).astype(np.int32)
    total_border = np.bincount(e_src, weights=edge_w, minlength=n).astype(np.float64)
    return e_src, e_dst, edge_w, total_border


def _neighbor_frac(
    target_mask: np.ndarray,
    e_src: np.ndarray,
    e_dst: np.ndarray,
    edge_w: np.ndarray,
    total_border: np.ndarray,
    n: int,
) -> np.ndarray:
    if len(e_src) == 0:
        return np.zeros(n, dtype=np.float32)
    hit = target_mask[e_dst]
    w_hit = np.bincount(e_src[hit], weights=edge_w[hit], minlength=n)
    return (w_hit / np.maximum(total_border, 1.0)).astype(np.float32)


def _weighted_median_oklab(labs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-channel median OKLab color (robust to ornaments inside hair)."""
    order = np.argsort(labs[:, 0])
    out = np.zeros(3, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0.0:
        return labs.mean(axis=0).astype(np.float32)
    for ch in range(3):
        idx = np.argsort(labs[:, ch])
        cdf = np.cumsum(weights[idx]) / total
        pos = int(np.searchsorted(cdf, 0.5))
        pos = min(pos, len(idx) - 1)
        out[ch] = float(labs[idx[pos], ch])
    return out


def segment_semantic_parts(
    regions: np.ndarray,
    colors: np.ndarray,
    line_mask: np.ndarray,
    fmaps,
    *,
    scale: int = 2,
) -> tuple[np.ndarray, dict]:
    """Classify each planar region `0..N-1` into one of 8 hard-bounded semantic subjects.

    Uses neural `fmaps.part_seg_probs` (`[8, H, W]` from `weights/anime_part_seg.onnx`) as the
    primary color-invariant structural prior, enhanced with:
      1. Region-level probability aggregation so boundaries snap bit-exact to vector contours.
      2. Color-relative embedded ornament rescue inside neural `PART_HAIR` (catches any contrasting
         flower, pearl bead, hairpin, or ribbon embedded in ANY hair color without color rules).
      3. Embedded facial micro-feature protection (catches sclera/pupil/catchlight/fang pockets
         sitting inside neural `PART_FACE_SKIN` or `PART_EYES_MOUTH`).
    """
    n = len(colors)
    if n == 0:
        return np.zeros(0, dtype=np.int8), {"source": "empty"}

    hs, ws = regions.shape
    vis = regions >= 0
    scale2 = float(max(1, scale * scale))
    r_areas = np.bincount(regions[vis].ravel(), minlength=n).astype(np.float64)
    r_area_1x = r_areas / scale2

    r_lm = (
        np.bincount(regions[vis].ravel(), weights=line_mask[vis].astype(np.float32).ravel(), minlength=n)
        / np.maximum(r_areas, 1.0)
    )
    is_line = r_lm >= 0.35

    def _reg_mean(m_1x: np.ndarray | None) -> np.ndarray:
        m_hi = _resize_map(m_1x, hs, ws)
        if m_hi is None:
            return np.zeros(n, dtype=np.float32)
        return (
            np.bincount(regions[vis].ravel(), weights=m_hi[vis].ravel(), minlength=n)
            / np.maximum(r_areas, 1.0)
        ).astype(np.float32)

    r_delicate = _reg_mean(getattr(fmaps, "delicate_detail", None))
    r_wash = _reg_mean(getattr(fmaps, "mergeable_wash", None))

    part_probs_1x = getattr(fmaps, "part_seg_probs", None)
    e_src, e_dst, edge_w, total_border = _build_adjacency(regions, n)
    labs = srgb_to_oklab(colors).astype(np.float32)

    if part_probs_1x is not None and part_probs_1x.ndim == 3 and part_probs_1x.shape[0] == 8:
        source = "anime_part_seg.onnx"
        r_probs = np.zeros((8, n), dtype=np.float32)
        for c in range(8):
            r_probs[c] = _reg_mean(part_probs_1x[c])

        # Priority-resolved region classification snapped cleanly to planar contours
        part = np.full(n, PART_BG, dtype=np.int8)
        for pid in PART_PRIORITY:
            part[r_probs[pid] >= PART_THRESH[pid]] = pid

        # Soft-margin winner rescue: if a region is clearly foreground (`bg < 0.38`) yet no
        # single channel crossed its threshold (e.g. mixed border), assign its argmax fg class.
        ambig_fg = (part == PART_BG) & (r_probs[0] < 0.38)
        if ambig_fg.any():
            best_fg = 1 + np.argmax(r_probs[1:, ambig_fg], axis=0).astype(np.int8)
            part[ambig_fg] = best_fg

        # ------------------------------------------------------------------
        # Refinement A: Color-Relative Embedded Ornament Rescue inside Hair
        # Completely color-agnostic: computes THIS image's own neural hair median
        # OKLab color and elevates compact high-contrast outliers embedded inside
        # hair (e.g. peony flower / pearl beads / gold hairpin in black hair, OR
        # dark clip in blonde hair, OR cyan ribbon in pink hair) -> PART_HEADWEAR!
        # ------------------------------------------------------------------
        hair_mask = part == PART_HAIR
        if hair_mask.any():
            hair_fill = hair_mask & (~is_line)
            if hair_fill.any():
                # Prefer high-confidence hair interior when estimating dominant hair color
                hair_core = hair_fill & (r_probs[PART_HAIR] >= 0.60)
                hair_ref = hair_core if hair_core.any() else hair_fill
                hair_med = _weighted_median_oklab(labs[hair_ref], r_areas[hair_ref])
                hair_c_med = float(np.hypot(hair_med[1], hair_med[2]))
                de_from_hair = np.linalg.norm(labs - hair_med[None, :], axis=1)
                de_ab_from_hair = np.linalg.norm(labs[:, 1:3] - hair_med[None, 1:3], axis=1)
                reg_c = np.hypot(labs[:, 1], labs[:, 2])
                reg_l = labs[:, 0]

                # Boundary Closure 1: Hair Hue/Chroma Bleed Rejection ("边界收束闭合")
                # When hair is distinctly colored (e.g. vivid blue/pink/green/brown hair), neutral
                # dark eyeshadow/eyelashes/sclera or gray background patches splashed by the blurry
                # neural hair fringe (`prob < 0.82`, `de_ab >= 0.09`, `chroma < 0.55 * hair_c_med`)
                # are ejected from PART_HAIR so hair color can NEVER dye eyes or shaded face skin!
                if hair_c_med >= 0.055:
                    skin_eye_nb = _neighbor_frac(
                        (part == PART_FACE_SKIN) | (part == PART_EYES_MOUTH) | (part == PART_NECK_HAND),
                        e_src, e_dst, edge_w, total_border, n,
                    )
                    hair_bleed = (
                        (part == PART_HAIR)
                        & (r_probs[PART_HAIR] < 0.82)
                        & (de_ab_from_hair >= 0.090)
                        & (reg_c < hair_c_med * 0.55)
                    )
                    if hair_bleed.any():
                        bleed_idx = np.flatnonzero(hair_bleed)
                        for ri in bleed_idx.tolist():
                            if skin_eye_nb[ri] >= 0.22:
                                part[ri] = PART_EYES_MOUTH if (reg_l[ri] < 0.48 or r_delicate[ri] >= 0.20) else PART_FACE_SKIN
                            else:
                                # Runner-up non-hair class
                                alt_p = r_probs[:, ri].copy()
                                alt_p[PART_HAIR] = -1.0
                                part[ri] = int(np.argmax(alt_p))
                        hair_mask = part == PART_HAIR
                        hair_fill = hair_mask & (~is_line)

                # Distinct chromatic OR bright/dark jewel/pearl/flower embedded inside hair
                # (excludes subtle hair sheen which stays close in chroma & lightness)
                chroma_outlier = (de_from_hair >= 0.20) & (reg_c >= hair_c_med + 0.080)
                bright_pearl_outlier = (de_from_hair >= 0.22) & (reg_l >= hair_med[0] + 0.20) & (r_area_1x <= 220.0)
                dark_clip_outlier = (de_from_hair >= 0.36) & (reg_l <= hair_med[0] - 0.34) & (r_area_1x <= 260.0)
                orn_in_hair = (
                    hair_fill
                    & (r_area_1x <= 1200.0)
                    & (chroma_outlier | bright_pearl_outlier | dark_clip_outlier)
                )
                if orn_in_hair.any():
                    # Pull in shaded halves of pearls, gold chain links, dark stamen dots, and
                    # outline creases bordering the rescued flowers/beads (2-hop bead-chain rescue!)
                    orn_nb = _neighbor_frac(orn_in_hair, e_src, e_dst, edge_w, total_border, n)
                    orn_inner = hair_mask & (r_area_1x <= 160.0) & (
                        (orn_nb >= 0.40) | ((orn_nb >= 0.18) & (de_from_hair >= 0.13) & (reg_l >= hair_med[0] + 0.10))
                    )
                    orn_total = orn_in_hair | orn_inner
                    # Second hop for bead chains & nested center stamen/links
                    orn_nb2 = _neighbor_frac(orn_total, e_src, e_dst, edge_w, total_border, n)
                    orn_total |= hair_mask & (r_area_1x <= 140.0) & (
                        (orn_nb2 >= 0.50) | ((orn_nb2 >= 0.22) & (de_from_hair >= 0.14) & (reg_l >= hair_med[0] + 0.10))
                    )
                    part[orn_total] = PART_HEADWEAR

                # Prevent neural PART_HAIR mask bleed onto adjacent pale warm neck/ear skin
                # (keeps curling dark neck wisps crisp over skin instead of getting erased as hair bun!)
                non_hair_nb = _neighbor_frac(part != PART_HAIR, e_src, e_dst, edge_w, total_border, n)
                pale_skin_bleed = (
                    (part == PART_HAIR)
                    & (~is_line)
                    & (reg_l >= max(0.54, hair_med[0] + 0.28))
                    & (reg_c <= 0.12)
                    & (labs[:, 1] >= 0.004)
                    & (labs[:, 2] >= -0.015)
                    & ((non_hair_nb >= 0.22) | (r_area_1x >= 45.0))
                )
                if pale_skin_bleed.any():
                    part[pale_skin_bleed] = PART_FACE_SKIN

        # ------------------------------------------------------------------
        # Refinement B: Protect Embedded Eye/Mouth Micro-Pockets inside Face
        # Keeps tiny catchlights, sclera slivers, eyelashes, and fangs locked
        # in PART_EYES_MOUTH even when smaller than the 512x512 neural stride.
        # ------------------------------------------------------------------
        eye_mask = part == PART_EYES_MOUTH
        face_mask = part == PART_FACE_SKIN
        if eye_mask.any():
            eye_nb = _neighbor_frac(eye_mask, e_src, e_dst, edge_w, total_border, n)
            part[(part != PART_EYES_MOUTH) & (r_area_1x <= 140.0) & (eye_nb >= 0.50)] = PART_EYES_MOUTH
        if face_mask.any():
            face_nb = _neighbor_frac(face_mask, e_src, e_dst, edge_w, total_border, n)
            # Tiny high-detail pockets surrounded by face skin (mouth/eyebrow/eye catchlight)
            embedded_face_detail = (
                (part != PART_EYES_MOUTH)
                & (part != PART_HEADWEAR)
                & (r_area_1x <= 160.0)
                & (face_nb >= 0.55)
                & ((r_delicate >= 0.22) | (r_probs[PART_EYES_MOUTH] >= 0.16))
            )
            part[embedded_face_detail] = PART_EYES_MOUTH

        # ------------------------------------------------------------------
        # Refinement C: Color-Guided Region-Graph Boundary Contraction ("边界收束")
        # Snaps uncertain neural fringe regions (`prob < 0.66`) cleanly onto the
        # neighboring semantic subject whose core color is >= 1.8x closer in
        # chroma-weighted OKLab, locking semantic masks tightly to vector edges!
        # ------------------------------------------------------------------
        if len(e_src) > 0:
            labs_w = labs.copy()
            labs_w[:, 1] *= 2.2
            labs_w[:, 2] *= 2.2
            class_med_w = np.zeros((8, 3), dtype=np.float32)
            class_has = np.zeros(8, dtype=bool)
            for c in range(8):
                mc = (part == c) & (~is_line) & (r_probs[c] >= 0.55)
                if not mc.any():
                    mc = (part == c) & (~is_line)
                if mc.any():
                    class_med_w[c] = _weighted_median_oklab(labs_w[mc], r_areas[mc])
                    class_has[c] = True

            for _ in range(2):
                own_prob = r_probs[part, np.arange(n)]
                uncertain = (own_prob < 0.66) & (part != PART_EYES_MOUTH) & (part != PART_HEADWEAR)
                if not uncertain.any():
                    break
                u_ids = np.flatnonzero(uncertain)
                starts = np.searchsorted(e_src, u_ids, side="left")
                ends = np.searchsorted(e_src, u_ids, side="right")
                nb_cls = part[e_dst]
                changed = False
                for ri, s_i, e_i in zip(u_ids.tolist(), starts.tolist(), ends.tolist()):
                    if s_i >= e_i:
                        continue
                    cur_c = int(part[ri])
                    if not class_has[cur_c]:
                        continue
                    d_own = float(np.linalg.norm(labs_w[ri] - class_med_w[cur_c]))
                    if d_own <= 0.06:
                        continue
                    w_cls = np.bincount(nb_cls[s_i:e_i], weights=edge_w[s_i:e_i], minlength=8)
                    w_cls[cur_c] = 0.0
                    best_q = int(np.argmax(w_cls))
                    if w_cls[best_q] / max(float(total_border[ri]), 1.0) < 0.42:
                        continue
                    if not class_has[best_q]:
                        continue
                    d_nb = float(np.linalg.norm(labs_w[ri] - class_med_w[best_q]))
                    if d_nb < d_own * 0.55 and d_nb <= 0.14:
                        part[ri] = best_q
                        changed = True
                if not changed:
                    break
    else:
        # Lightweight fallback when anime_part_seg.onnx is absent (e.g. stripped unit tests)
        source = "feature_part_fallback"
        part = np.full(n, PART_CLOTH, dtype=np.int8)
        part[r_wash >= 0.45] = PART_BG
        part[r_delicate >= 0.30] = PART_HEADWEAR

    # Summarize per-part region counts & area shares
    vis_area = float(r_areas.sum()) or 1.0
    stats: dict = {"source": source}
    for pid, pname in PART_NAMES.items():
        m = part == pid
        stats[pname] = {
            "regions": int(m.sum()),
            "area_pct": round(100.0 * float(r_areas[m].sum()) / vis_area, 1),
        }
    return part, stats


def render_part_map_rgb(regions: np.ndarray, part_ids: np.ndarray) -> np.ndarray:
    """Render a color-coded `(Hs, Ws, 3) uint8` visualization of the 8 semantic subject parts."""
    hs, ws = regions.shape
    out = np.full((hs, ws, 3), 24, dtype=np.uint8)
    vis = regions >= 0
    if vis.any() and len(part_ids) > 0:
        out[vis] = PART_VIS_RGB[np.clip(part_ids[regions[vis]], 0, len(PART_VIS_RGB) - 1)]
    return out
