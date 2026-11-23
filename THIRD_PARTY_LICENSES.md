# Third-Party Licenses and Attributions

This document lists upstream origins, bundled model weights, and third-party
software components used by **Lumina-IMTVec**.

---

## 1. Upstream Project (`Lumina-Vec` — Apache License 2.0)

- **Upstream**: [MESFET5198/Lumina-Vec](https://github.com/MESFET5198/Lumina-Vec) (v0.9.2 snapshot)
- **Copyright**: `Copyright 2026 MESFET5198`
- **License**: **Apache License, Version 2.0** (SPDX: `Apache-2.0`)
- **Full License Copy (§4(a))**: [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)
- **Preserved Upstream Notice (§4(d))**: [NOTICE](NOTICE)
- **Modified Upstream Files (§4(b), each carrying prominent header modification notices)**:
  - `webui.py` — local Flask/Waitress async Web UI server (adapted for pure-ONNX worker, RGBA preview, AI repaint flag; legacy `vtracer`/`print_mm` routes removed)
  - `index.html` — local Web UI frontend (simplified to CNN two-layer mode, AI flat repaint checkbox, layer viewer, RGBA checkerboard)
  - `lumina_vec/color.py` — colour conversion utilities (extended with vectorized OKLab LUT helpers)
  - `lumina_vec/pipeline.py` — top-level vectorization facade & CLI entrypoint (rewritten around CNN planar engine)
  - `lumina_vec/presets.py` — preset definitions (consolidated to `print-regions`)
  - `scripts/setup_windows.ps1` & `scripts/start_web.bat` — Windows native setup and launcher scripts (switched to pure ONNX DirectML)
- **Notes**: Core CNN routing (`feature_routing.py`), planar disjoint partitioning (`planar_partition.py`, `print_partition.py`), independent Bézier fitting (`bezier_fit.py`), shared-boundary & curve refinement modules, VRAM guard (`vram_guard.py`), and RGBA transparency pipeline were written/rebuilt in Lumina-IMTVec. Zero byte-identical files from v0.9.2 remain in the repository.

---

## 2. Bundled Pretrained Weights (`weights/`)

### Real-ESRGAN (`weights/realesr_anime.onnx`)
- **Upstream**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (`realesr-animevideov3`)
- **Copyright**: `Copyright (c) 2021, Xintao Wang`
- **License**: **BSD-3-Clause** (SPDX: `BSD-3-Clause`)
- **Upstream License Text**: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
- **Redistribution Conditions**:
  1. Source redistributions must retain the upstream copyright notice, conditions, and disclaimer.
  2. Binary / converted weight redistributions (`weights/realesr_anime.onnx`) must reproduce the upstream copyright notice, conditions, and disclaimer in accompanying documentation (this file).
  3. Neither the name of the copyright holder nor the names of contributors may be used to endorse or promote derived products without specific prior written permission.
  4. Provided on an "AS IS" basis without warranty of any kind.

### Lumina-IMTVec Self-Trained ONNX Models
- **Files**: `weights/line_vector_net.onnx`, `weights/line_net.onnx`,
  `weights/element_grad.onnx`, `weights/feature_part.onnx`,
  `weights/surface_style.onnx`, `weights/defect_repair_net.onnx`
- **Copyright**: `Copyright (c) 2026 KasoluINT (Lumina-IMTVec)`
- **License**: MIT (see [LICENSE](LICENSE))

---

## 3. Optional External Models (Not Bundled)

### ControlNet Annotators — `lineart_anime` (`LineartAnimeDetector`)
- **Upstream**: [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) / `controlnet_aux`
- **License**: **Apache-2.0** (SPDX: `Apache-2.0`, full license text in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0))
- **Usage**: Optional runtime hook in `lumina_vec/ai_repaint.py` when `controlnet_aux` is installed locally; weights are **not** bundled in this repository.

---

## 4. Core Python Runtime Dependencies

| Package | License (SPDX) | Purpose |
| --- | --- | --- |
| `onnxruntime` / `onnxruntime-directml` | `MIT` | Pure ONNX CPU / DirectML GPU inference |
| `numpy` | `BSD-3-Clause` | Array and numerical computation |
| `scipy` | `BSD-3-Clause` | Distance transforms, graph & spatial routines |
| `opencv-python-headless` | `Apache-2.0` | Image processing and connected components |
| `scikit-image` | `BSD-3-Clause` | Morphology and segmentation primitives |
| `Pillow` | `MIT-CMU` (HPND) | Image I/O and RGBA alpha compositing |
| `shapely` | `BSD-3-Clause` | Planar polygon / self-intersection geometry guard |
| `Flask` | `BSD-3-Clause` | Local Web UI HTTP server |
| `waitress` | `ZPL-2.1` | Production WSGI server for Windows |

---

## 5. Removed Legacy Components

The following legacy components from earlier v0.9.2 snapshots have been completely
removed from source code, scripts, and dependency manifests:
- `vtracer` (Rust tracer) — removed from `requirements-native.txt`, `pyproject.toml`, and `lumina_vec/`
- `MangaLineExtraction_PyTorch` (`lumina_vec/manga_net.py`) — removed
- `scripts/download_weights.py` (legacy PyTorch `.pth` downloader) — removed
- `docs/LINE_V6.md` & `docs/NATIVE.md` — removed
