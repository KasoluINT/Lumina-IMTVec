# Copyright 2026 MESFET5198 (Lumina-Vec, Apache License 2.0)
# Modified 2026 by KasoluINT (Lumina-IMTVec): consolidated presets to print-regions CNN mode.
# See LICENSE-APACHE-2.0 and NOTICE for full license and attribution notices.
"""Preset definitions for Lumina-IMTVec (pure data, zero heavy imports)."""

PRESETS = {
    "print-regions": dict(
        structure="print-regions",
        bands=84,
        gradient_mode="clean",
        smooth_lines=True,
        feature_routing=True,
        color_merge=True,
    ),
}

PRESET_NAMES = tuple(PRESETS)
