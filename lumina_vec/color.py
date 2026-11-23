# Copyright 2026 MESFET5198 (Lumina-Vec, Apache License 2.0)
# Modified 2026 by KasoluINT (Lumina-IMTVec): added vectorized OKLab LUT conversion helpers.
# See LICENSE-APACHE-2.0 and NOTICE for full license and attribution notices.
"""Colour helpers."""
import numpy as np


_SRGB_TO_LIN_LUT = np.array(
    [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in (np.arange(256, dtype=np.float64) / 255.0)],
    dtype=np.float32,
)
_M1_OKLAB = np.array(
    [[0.4122214708, 0.5363325363, 0.0514459929],
     [0.2119034982, 0.6806995451, 0.1073969566],
     [0.0883024619, 0.2817188376, 0.6299787005]],
    dtype=np.float32,
).T
_M2_OKLAB = np.array(
    [[0.2104542553, 0.7936177850, -0.0040720468],
     [1.9779984951, -2.4285922050, 0.4505937099],
     [0.0259040371, 0.7827717662, -0.8086757660]],
    dtype=np.float32,
).T


def srgb_to_oklab(rgb):
    if rgb.dtype == np.uint8:
        c = _SRGB_TO_LIN_LUT[rgb]
    else:
        c = rgb.astype(np.float32) / np.float32(255.0)
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(c @ _M1_OKLAB)
    return lms @ _M2_OKLAB


def oklab_to_srgb(lab):
    """Inverse of srgb_to_oklab; returns float sRGB 0..255 (unclipped)."""
    M2i = np.linalg.inv(np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                                  [1.9779984951, -2.4285922050, 0.4505937099],
                                  [0.0259040371, 0.7827717662, -0.8086757660]], np.float64))
    M1i = np.linalg.inv(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                  [0.2119034982, 0.6806995451, 0.1073969566],
                                  [0.0883024619, 0.2817188376, 0.6299787005]], np.float64))
    lms = (lab.astype(np.float64) @ M2i.T) ** 3
    c = np.clip(lms @ M1i.T, 0, 1)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return (c * 255).astype(np.float32)


def flat_sample(rgb_flat, n, power=0.4, seed=0):
    """Sample pixels with probability ~ count(colour bin)^(power-1), so rare but distinct
    hues (an iris, a highlight) are represented instead of drowned by large flat areas."""
    rng = np.random.default_rng(seed)
    q = rgb_flat.astype(np.int32) >> 2
    key = (q[:, 0] << 12) | (q[:, 1] << 6) | q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    p = cnt[inv].astype(np.float64) ** (power - 1.0)
    p /= p.sum()
    return rng.choice(len(rgb_flat), min(n, len(rgb_flat)), replace=False, p=p)
