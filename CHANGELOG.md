# CHANGELOG · Lumina-Vec-Lumina 架构与改动记录

记录从 `v10` 单网格基准到 `v24` CNN 热力图双图层重构、再到 `v25` 矢量场 CNN（`LineVectorNet`）直出 Cubic Bezier 线稿与严格线稿约束着色主线的完整演进与实测数据。

---

## [v25] · 2026-11-20 · `LineVectorNet` 9 通道矢量场直出矢量线稿 + 严格矢量约束着色 + WebUI 联动

### 核心架构突破
彻底淘汰“热力图 → 1× 二值掩码 → 4× 最近邻放大 → 多边形双边界轮廓描摹”的有损中转链与旧版残屑回灌步骤，改为**超分前置 → 9 通道矢量场 CNN → 直出 Cubic Bezier 流线追踪 + 原生 2× 封闭分区纯色填充**：

- **9 通道神经网络矢量场 `LineVectorNet`（`lumina_vec/feature_nets.py`、`tools/feature_train/09_train_line_vector_cnn.py`）**：
  - 在 2× 超采样合成二次元矢量曲线解析真值（460 train + 64 val）与 `work_cache` 真实插画 2× 超分教师图块（共 `706 train / 80 val`）上以 `base=48` 训练 16 轮（峰值显存 `4.18 GB`，约 `26.6s/epoch`），Epoch 14 验证精度：`off = 0.216 px`、`ang = 3.51°`、`rad = 0.173 px`、`cls = 0.1603`。
  - 导出 `weights/line_vector_net.onnx`（`3316.9 KB`，PyTorch 与 ORT 最大数值差异 `1.88e-06`），预测：`stroke_prob`、`color_bound`、`end_prob`、`cross_junc`、亚像素偏移 `(dx, dy)`、模 $\pi$ 双向切线场 $(\cos 2\theta, \sin 2\theta)$ 与半宽半径 $r$。
- **2× 超分无池化直出 Cubic Bezier 流线追踪（`lumina_vec/vector_tracer.py`）**：
  - `infer_feature_maps`（`lumina_vec/feature_routing.py`）直接保留未经过 2×2 池化的 `2× SR` 原生矢量场。
  - 新增 `trace_vector_fields_to_svg_paths`：沿法向执行严格 1D NMS 脊峰提种、5×5 起笔去重与交叉点动量穿透追踪（`cross_junc` 处维持切线动量直接穿过，不捏合粘连）、`+0.5 / sr_scale` SVG 像元中心对齐，以及深色粗睫毛/粗线分段渐变包络拟合，直出平滑 `<path>` 矢量线条。
- **矢量边界封闭分区纯内采样着色（`lumina_vec/vector_coloring.py`、`lumina_vec/region_merge.py`、`lumina_vec/two_layer_engine.py`、`lumina_vec/print_partition.py`）**：
  - 新增 `build_two_layer_vector_svg` 与 `build_vector_constrained_fill_svg`，跳过会污染填色层的旧版 `pin_all` / `stitch_and_smooth_ink` / `SourceCorridor`；
  - 将实际追踪的 Bezier 中心线与 `stroke_prob / color_bound` 在 2× SR 空间合成为刚性 `wall_hi`，仅在 `dt_wall >= 2.4` 纯内区做双边平滑、同分区 `3×3` 多数投票去椒盐杂色，以及适配 `area_scale = sr_scale ** 2` 的同分区 `merge_transition_patches` 合并，最后经 Voronoi Tuck-Under 令色块接缝精准落于 Bezier 线稿正中心。
- **Web 前端全流程联动（`webui.py`、`lumina_vec/worker.py`、`index.html`）**：
  - 修复双层矢量版 PNG 预览未叠绘矢量线稿的问题，补充 `line_vector_net.onnx` 健康检查，并在前端工具栏新增 **`双层 / 仅线稿 / 仅着色`** 实时图层切换按钮、双层独立路径统计与 `HEAVY_PATHS = 24000` 矢量直接缩放支持。

### 6 图端到端实测数据（`LineVectorNet + vector_coloring`）

| 测试图（尺寸） | 总耗时 | 纯填色层 `Fill-only` (`PSNR / ΔE`) | 双层合成 `Combined` (`PSNR / ΔE`) | 路径数（`fill` / `line`） |
| :--- | :--- | :--- | :--- | :--- |
| `test12` (`800×1200`) | `42.9s` | `19.21 dB / 6.36` | **`23.60 dB / 4.58`** | `6357` / `4604` |
| `A4` (`1760×2208`) | `176.7s` | `17.03 dB / 7.74` | `21.00 dB / 5.78` | `28552` / `21295` |
| `test1` (`502×493`) | **`11.3s`** | `19.82 dB / 4.83` | **`23.42 dB / 3.68`** | `1838` / `1645` |
| `reg1` (`960×1440`) | `45.7s` | `21.67 dB / 4.74` | **`26.07 dB / 3.70`** | `6649` / `1961` |
| `A5` (`1600×1600`) | `54.6s` | `25.21 dB / 2.59` | **`31.46 dB / 1.98`** | `1793` / `1347` |
| `bbb` (`2663×3879`) | `195.8s` | `24.83 dB / 4.18` | **`28.27 dB / 3.67`** | `7947` / `3131` |

---

## [v24] · 2026-11-20 · 3D-Oklab 全彩线条+色块边界 CNN 与 `v10` 边线平滑复用

### 新增与改进
- **前端全彩向量感知（`lumina_vec/feature_nets.py`）**：
  - 将 `_LineRidgeStem` 从 `10 ch` 升级至 `12 ch`，新增全 3 通道 RGB 向量 Sobel 梯度（`c_grad`）与 RGB 拉普拉斯脊（`c_lap`），解决纯灰度亮度边缘对等亮度彩线/黄色花蕊的盲区。
- **4 通道清晰语义分离与重训（`tools/feature_train/08_train_dedicated_line_and_grad_cnns.py`）**：
  - `L0`（`dark_ink_contour`）：深色线稿轮廓，显式过滤 `L < 0.36` 密集暗发内部平坦填充。
  - `L1`（`colored_line_ridge`）：3D Oklab `(L, 1.9·a, 1.9·b)` 对称彩色细线脊。
  - `L2`（`color_block_boundary`）：1D 方向极值抑制（NMS）+ 两侧平坦度约束提取 `1 px` 清晰色块阶跃边界，精准检出 `A5` 黄色花蕊/桃色花环及 `test1` 头发亮影/暗影色块交界。
  - `L3`（`free_wisp_strand`）：`A4` 游离细发丝与睫毛。
  - 在 RTX 4070 SUPER 上以 `base=64`、`edge_focus` 加权损失训练 16 轮并导出 `weights/line_net.onnx`（`Ep 14 val=0.2545`，`5174.8 KB`）。
- **下游语义协同与 `v10` 边线平滑策略迁移**：
  - `lumina_vec/feature_routing.py`：高频分块融合 `hi_weight` 提至 `0.92`，并对齐 `[L0, L1, L2, L3]` 通道路由。
  - `lumina_vec/gradient_topology.py`：将 `L2 > 0.45` 写入 Stage B/C `barrier`，防止赛璐璐花瓣环与色块边界被合并或水平集平滑模糊。
  - `lumina_vec/two_layer_engine.py`：复用 `v10` 边线拟合链路（`compact_fragments(..., scale)` + `secondary_fitter=refit_sketch_chain` + `secondary_network=smooth_outline_junctions` + `tol=0.32*scale, smoothing=0.85*scale`），并在 `chain_scale_fn` 对 `L2 > 0.36` 色块边界保持 `0.90×` 高精度贴合；狭长双轨微孔闭合适配至 `<= 26 px`。

---

## [v20 – v23] · 2026-11-20 · 去发束泛光重训、骨架切向缝合与大圆弧伪影根除

- **根除发束内部热力图泛光**：移除旧版 `5×5 MORPH_BLACKHAT/TOPHAT` 宽核真值构造，改用 1D 对称脊谷局部极大值（NMS）与 1 px 外轮廓阶跃凹槽，消除发束填充内部大面积误高亮及 `fine_strand` 泛光回灌。
- **矢量曲线断线缝合（`lumina_vec/two_layer_engine.py`）**：新增 `_stitch_line_mask_gaps`、` _trace_skel_tangent`、`_bfs_near_skel`，基于 1 px 骨架端点切向在 `±36°`、`2–24 px` 锥区沿最小代价凹槽路径缝合断开的连续发束边线，避免角点短接自环。
- **彻底修复尖端/眼睑圆球伪影（`lumina_vec/layered.py`）**：在 `_cubic_to_arc` 中发现切向接近共线时 `sweep` 翻转会导致 `large-arc-flag=1`（`200°–299°` 大圆弧），将最大圆弧张角限制至 `< 195°` 并增加弧中点与三次曲线中点偏差校验（`<= max(0.8, tol * 1.5)`）。

---

## [v13 – v19] · 2026-11-19 · 双图层架构解耦与保真拓扑净化重构

- **架构解耦（`lumina_vec/two_layer_engine.py`、`lumina_vec/svg_layers.py`、`lumina_vec/print_partition.py`）**：
  - 将 `vectorize_print(..., gradient_mode='clean')` 重构为 **`#region-boundaries` + `#region-colours`（100% 闭合底层） + `#line-art-layer`（独立线稿层）**。
  - `#region-colours` 采用 Voronoi 欠填（并剔除暗色小尖角避免圆球），关闭线稿层后无任何漏白空洞。
  - 放弃 `GaussianBlur` 膨胀二值线稿的方式，改用 0-膨胀 `INTER_NEAREST` + 对角拐点桥接。
- **拓扑净化与防误损修复（`lumina_vec/gradient_topology.py`、`lumina_vec/region_merge.py`）**：
  - Stage A：消除刘海等彩色区域内部捏合的短窄米白色穿帮白线，同时限定跨度与彩度双侧条件以保护服饰斜白条纹与眼部高光。
  - Stage B/C：移除会污染填色层的 Stage D，引入原图 RMSE 误差预算与 `cel_guard` 门控，水平集平滑严格限制在 CNN 渐变区。
  - 修复 `merge_transition_patches` 对低误差赛璐璐环带（`A5` 花朵桃色内环）的误吞并（`crisp_cel` 守护）。
  - 修复 `need_ink` 在 `dists_base <= 5.0` 下的跨色带串色问题（如 `test1` 领口红线），改用 `dists_base <= 1.5` 且 `ΔE <= 0.09` 校验及局部连通分量着色。

---

## 实测基准对比（`v10` 单网格基线 vs `v24` 双图层终版，无头 Edge 渲染）

| 测试图 | 分辨率 | `v10` PSNR | `v24` PSNR | `v24` ΔE | `#region-colours` 路径 | `#line-art-layer` 路径 | 体积 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test12` | `800×1200` | `23.70 dB` | **`24.11 dB`** (`+0.41`) | `4.27` | `4949` | `4544` | `4968.4 KB` |
| `A4` | `1760×2208` | `23.07 dB` | `22.71 dB` (`-0.36`) | `4.94` | `18594` | `20857` | `19722.3 KB` |
| `test1` | `502×493` | `23.15 dB` | `23.09 dB` (`-0.06`) | `2.61` | `914` | `342` | `909.6 KB` |
| `reg1` | `960×1440` | `26.72 dB` | **`26.76 dB`** (`+0.04`) | `3.43` | `2389` | `3337` | `2979.9 KB` |
| `A5` | `1600×1600` | `29.42 dB` | **`30.43 dB`** (`+1.01`) | `0.99` | `2745` | `772` | `3197.5 KB` |
| `bbb` | `2663×3879` | `29.70 dB` | **`30.37 dB`** (`+0.67`) | `3.01` | `5054` | `7356` | `7259.1 KB` |
