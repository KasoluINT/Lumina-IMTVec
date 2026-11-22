"""Compact CNN architectures for Scheme 2 Feature-Routed Vectorization.

Both models take raw RGB float32 tensors in [0, 1] and compute gradient / highpass / OKLab-like
spectral cues internally using fixed Conv2d kernels so PyTorch training and ONNX Runtime deployment
stay 100% identical with zero preprocessing drift.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpectralStem(nn.Module):
    """Differentiable fixed front-end: RGB (3) + Sobel grad (1) + Laplacian highpass (1) + Chroma (1) = 6 ch."""

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32) / 8.0
        gy = gx.t().contiguous()
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32)
        luma = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luma_w", luma)
        self.register_buffer("sobel_x", gx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", gy.view(1, 1, 3, 3))
        self.register_buffer("lap_k", lap.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = (x * self.luma_w).sum(dim=1, keepdim=True)
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        hp = torch.abs(F.conv2d(g, self.lap_k, padding=1))
        c_max = torch.amax(x, dim=1, keepdim=True)
        c_min = torch.amin(x, dim=1, keepdim=True)
        chroma = c_max - c_min
        return torch.cat([x, grad, hp, chroma], dim=1)


class _DSConvBlock(nn.Module):
    """Depthwise-separable residual-ready block (ONNX / DirectML friendly)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=dilation, dilation=dilation, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.skip = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.dw(x)))
        h = self.bn2(self.pw(h))
        return self.act(h + self.skip(x))


class _SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(8, ch // r), 1)
        self.fc2 = nn.Conv2d(max(8, ch // r), ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.mean(x, dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w


class SurfaceStyleCNN(nn.Module):
    """Model 1 (~1.1 MB ONNX FP16): classifies 96x96 patches into 4 surface styles:
    [0: clean_cel, 1: soft_gradient, 2: grain_paint, 3: fine_detail]."""

    CLASSES = ("clean_cel", "soft_gradient", "grain_paint", "fine_detail")

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.stem_prep = _SpectralStem()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            _DSConvBlock(32, 48, stride=1),
            _DSConvBlock(48, 64, stride=2),
            _SEBlock(64),
            _DSConvBlock(64, 96, stride=1),
            _DSConvBlock(96, 128, stride=2),
            _SEBlock(128),
            _DSConvBlock(128, 160, stride=2),
            _DSConvBlock(160, 192, stride=1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(384, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.enc(self.stem_prep(x))
        avg_p = torch.mean(f, dim=(2, 3))
        max_p = torch.amax(f, dim=(2, 3))
        return self.head(torch.cat([avg_p, max_p], dim=1))


class FeaturePartUNet(nn.Module):
    """Model 2 (~4-7 MB ONNX FP16): dense multi-scale feature map predictor.
    Outputs 4 sigmoid channels in [0, 1]:
      ch0: ink_main_geom   (outer black ink, leadlines, crisp geometric edges)
      ch1: fine_strand     (fine internal hair strands, coloured thin strokes, lace)
      ch2: delicate_detail (eyes/iris/highlights, mouth, cheek blush ///, text, jewels)
      ch3: mergeable_wash  (smooth gradients & grain washes safe for stained-glass merging)
    """

    CHANNELS = ("ink_main_geom", "fine_strand", "delicate_detail", "mergeable_wash")

    def __init__(self, base: int = 32, out_ch: int = 4):
        super().__init__()
        self.stem_prep = _SpectralStem()
        self.s1 = nn.Sequential(
            nn.Conv2d(6, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
            _DSConvBlock(base, base),
        )
        self.s2 = nn.Sequential(_DSConvBlock(base, base * 2, stride=2), _DSConvBlock(base * 2, base * 2))
        self.s3 = nn.Sequential(_DSConvBlock(base * 2, base * 4, stride=2), _DSConvBlock(base * 4, base * 4), _SEBlock(base * 4))
        self.s4 = nn.Sequential(
            _DSConvBlock(base * 4, base * 6, stride=2),
            _DSConvBlock(base * 6, base * 6, dilation=2),
            _DSConvBlock(base * 6, base * 6, dilation=4),
            _SEBlock(base * 6),
        )

        self.d3 = nn.Sequential(_DSConvBlock(base * 6 + base * 4, base * 4), _SEBlock(base * 4))
        self.d2 = nn.Sequential(_DSConvBlock(base * 4 + base * 2, base * 2), _SEBlock(base * 2))
        self.d1 = nn.Sequential(_DSConvBlock(base * 2 + base, base), _DSConvBlock(base, base))
        self.out_head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.s1(self.stem_prep(x))
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)

        u3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.d1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out_head(u1))


class _LineRidgeStem(nn.Module):
    """Differentiable front-end for DedicatedLineNet:
    6 spectral channels + 4 luma oriented ridges + 1 full-RGB vector Sobel + 1 full-RGB Laplacian ridge = 12 ch.
    Directly captures BOTH dark thin lines AND iso-luminant colored boundaries (e.g. yellow flower stamens)."""

    def __init__(self):
        super().__init__()
        self.spec = _SpectralStem()
        k0 = torch.tensor([[-1., 2., -1.], [-2., 4., -2.], [-1., 2., -1.]], dtype=torch.float32) / 8.0
        k90 = k0.t().contiguous()
        k45 = torch.tensor([[-2., -1., 2.], [-1., 4., -1.], [2., -1., -2.]], dtype=torch.float32) / 8.0
        k135 = torch.tensor([[2., -1., -2.], [-1., 4., -1.], [-2., -1., 2.]], dtype=torch.float32) / 8.0
        kernels = torch.stack([k0, k45, k90, k135], dim=0).unsqueeze(1)  # [4, 1, 3, 3]
        self.register_buffer("ridge_k", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec(x)
        g = (x * self.spec.luma_w).sum(dim=1, keepdim=True)
        r = torch.abs(F.conv2d(g, self.ridge_k, padding=1))
        # Full 3-channel RGB color-vector Sobel gradient & Laplacian ridge (iso-luminant color edges!)
        gx_c = F.conv2d(x, self.spec.sobel_x.expand(3, 1, 3, 3), padding=1, groups=3)
        gy_c = F.conv2d(x, self.spec.sobel_y.expand(3, 1, 3, 3), padding=1, groups=3)
        c_grad = torch.sqrt((gx_c * gx_c + gy_c * gy_c).sum(dim=1, keepdim=True) + 1e-6)
        c_lap = torch.abs(F.conv2d(x, self.spec.lap_k.expand(3, 1, 3, 3), padding=1, groups=3)).amax(dim=1, keepdim=True)
        return torch.cat([s, r, c_grad, c_lap], dim=1)


class DedicatedLineNet(nn.Module):
    """Dedicated multi-class line & color-boundary recognition CNN.
    Outputs 4 sigmoid channels in [0, 1]:
      L0: dark_ink_contour      (dark primary outlines & thin dark hair edges)
      L1: colored_line_ridge    (chromatic/bright thin lines: yellow/orange/lavender outlines & internal dividers)
      L2: color_block_boundary  (crisp cel/flat color-block step boundaries: hair shadow seams, flower petals/stamen)
      L3: free_wisp_strand      (open-ended thin hair wisps & eyelashes)
    """

    CHANNELS = ("dark_ink_contour", "colored_line_ridge", "color_block_boundary", "free_wisp_strand")

    def __init__(self, base: int = 48, out_ch: int = 4):
        super().__init__()
        self.stem = _LineRidgeStem()
        self.s1 = nn.Sequential(
            nn.Conv2d(12, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
            _DSConvBlock(base, base),
            _DSConvBlock(base, base),
        )
        self.s2 = nn.Sequential(_DSConvBlock(base, base * 2, stride=2), _DSConvBlock(base * 2, base * 2), _SEBlock(base * 2))
        self.s3 = nn.Sequential(_DSConvBlock(base * 2, base * 4, stride=2), _DSConvBlock(base * 4, base * 4), _SEBlock(base * 4))
        self.s4 = nn.Sequential(
            _DSConvBlock(base * 4, base * 6, stride=2),
            _DSConvBlock(base * 6, base * 6, dilation=2),
            _DSConvBlock(base * 6, base * 6, dilation=4),
            _SEBlock(base * 6),
        )
        self.d3 = nn.Sequential(_DSConvBlock(base * 6 + base * 4, base * 4), _SEBlock(base * 4))
        self.d2 = nn.Sequential(_DSConvBlock(base * 4 + base * 2, base * 2), _SEBlock(base * 2))
        self.d1 = nn.Sequential(_DSConvBlock(base * 2 + base, base), _DSConvBlock(base, base))
        self.out_head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.s1(self.stem(x))
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        u3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return torch.sigmoid(self.out_head(u1))


class ElementGradNet(nn.Module):
    """Dedicated multi-element taxonomy & gradient topology CNN.
    Outputs 8 sigmoid channels in [0, 1]:
      E0: flat_cel               (solid flat colour cells, sclera, clean backdrop)
      E1: grad_directional       (Type I directional ribbon gradients: hair locks, skin shading, cloth folds)
      E2: grad_radial            (Type II concentric ring gradients: cheek blush, spherical highlights, orbs)
      E3: eye_iris_core          (eyes: iris shading bands, pupil, crisp specular highlights)
      E4: micro_ornament_feature (mouths, hairpins, jewelry/gems, flowers, lace, text/logos)
      E5: textured_wash_foliage  (watercolor texture, foliage, ice shards, background grain washes)
      G0: grad_dir_cos_norm      (0.5 * (cos(theta) + 1) of local smooth colour gradient flow)
      G1: grad_dir_sin_norm      (0.5 * (sin(theta) + 1) of local smooth colour gradient flow)
    """

    CHANNELS = (
        "flat_cel",
        "grad_directional",
        "grad_radial",
        "eye_iris_core",
        "micro_ornament_feature",
        "textured_wash_foliage",
        "grad_dir_cos_norm",
        "grad_dir_sin_norm",
    )

    def __init__(self, base: int = 48, out_ch: int = 8):
        super().__init__()
        self.stem = _SpectralStem()
        self.s1 = nn.Sequential(
            nn.Conv2d(6, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
            _DSConvBlock(base, base),
        )
        self.s2 = nn.Sequential(_DSConvBlock(base, base * 2, stride=2), _DSConvBlock(base * 2, base * 2))
        self.s3 = nn.Sequential(_DSConvBlock(base * 2, base * 4, stride=2), _DSConvBlock(base * 4, base * 4), _SEBlock(base * 4))
        self.s4 = nn.Sequential(
            _DSConvBlock(base * 4, base * 6, stride=2),
            _DSConvBlock(base * 6, base * 6, dilation=2),
            _DSConvBlock(base * 6, base * 6, dilation=4),
            _DSConvBlock(base * 6, base * 6, dilation=8),
            _SEBlock(base * 6),
        )
        self.d3 = nn.Sequential(_DSConvBlock(base * 6 + base * 4, base * 4), _SEBlock(base * 4))
        self.d2 = nn.Sequential(_DSConvBlock(base * 4 + base * 2, base * 2), _SEBlock(base * 2))
        self.d1 = nn.Sequential(_DSConvBlock(base * 2 + base, base), _DSConvBlock(base, base))
        self.out_head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.s1(self.stem(x))
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        u3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return torch.sigmoid(self.out_head(u1))


class LineVectorNet(nn.Module):
    """Neural Vector-Field CNN predicting geometric vector primitives directly from RGB patches.

    Outputs 9 sigmoid-normalized channels in [0, 1]:
      0: stroke_prob  (continuous stroke centerline heatmap peak)
      1: color_bound  (iso-luminant cel / flat color-block boundary wall)
      2: end_prob     (stroke tip endpoint probability)
      3: cross_junc   (T-junction / X-crossing intersection probability)
      4: offset_dx    (sub-pixel horizontal shift to curve center: (v - 0.5) * 3.0 px)
      5: offset_dy    (sub-pixel vertical shift to curve center: (v - 0.5) * 3.0 px)
      6: dir_cos2t    (unit tangent field cos(2*theta) encoded as 0.5 * (cos2t + 1))
      7: dir_sin2t    (unit tangent field sin(2*theta) encoded as 0.5 * (sin2t + 1))
      8: half_width   (local stroke half-width radius encoded as r / 3.2 px)
    """

    CHANNELS = (
        "stroke_prob",
        "color_bound",
        "end_prob",
        "cross_junc",
        "offset_dx",
        "offset_dy",
        "dir_cos2t",
        "dir_sin2t",
        "half_width",
    )

    def __init__(self, base: int = 64, out_ch: int = 9):
        super().__init__()
        self.stem = _LineRidgeStem()
        self.s1 = nn.Sequential(
            nn.Conv2d(12, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
            _DSConvBlock(base, base),
            _DSConvBlock(base, base),
        )
        self.s2 = nn.Sequential(_DSConvBlock(base, base * 2, stride=2), _DSConvBlock(base * 2, base * 2), _SEBlock(base * 2))
        self.s3 = nn.Sequential(_DSConvBlock(base * 2, base * 4, stride=2), _DSConvBlock(base * 4, base * 4), _SEBlock(base * 4))
        self.s4 = nn.Sequential(
            _DSConvBlock(base * 4, base * 6, stride=2),
            _DSConvBlock(base * 6, base * 6, dilation=2),
            _DSConvBlock(base * 6, base * 6, dilation=4),
            _DSConvBlock(base * 6, base * 6, dilation=6),
            _SEBlock(base * 6),
        )
        self.d3 = nn.Sequential(_DSConvBlock(base * 6 + base * 4, base * 4), _SEBlock(base * 4))
        self.d2 = nn.Sequential(_DSConvBlock(base * 4 + base * 2, base * 2), _SEBlock(base * 2))
        self.d1 = nn.Sequential(_DSConvBlock(base * 2 + base, base), _DSConvBlock(base, base))
        self.head_cls = nn.Sequential(_DSConvBlock(base, base // 2), nn.Conv2d(base // 2, 4, 1))
        self.head_vec = nn.Sequential(_DSConvBlock(base, base // 2), nn.Conv2d(base // 2, out_ch - 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.s1(self.stem(x))
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        u3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        c_out = self.head_cls(u1)
        v_out = self.head_vec(u1)
        return torch.sigmoid(torch.cat([c_out, v_out], dim=1))


class DefectRepairNet(nn.Module):
    """Feature-guided Pre-SVG Image Defect Detection & Restoration CNN.

    Runs on upscaled raster RGB paired with `FeaturePartUNet` semantic maps BEFORE `LineVectorNet`
    and SVG vectorization to detect & repair raster/SR artifacts such as:
      - Hazy, low-contrast semi-transparent pale-pink / off-white slivers ('亮棱') overlaid on dark ink
      - Trapped dim internal scratches inside acute dark wedges / cavities
      - Tiny enclosed bright noise pockets pinched inside dark outlines
    while strictly respecting semantic `delicate_detail` (eyes, pupil catchlights) & `fine_strand`.

    Input:  [B, 7, H, W] = [rgb (3), feature_part (4: ink, strand, delicate, wash)] in [0, 1]
    Output: [B, 4, H, W] sigmoid channels in [0, 1]:
      0: defect_prob  (probability pixel belongs to a hazy sliver / internal scratch defect)
      1..3: clean_rgb (restored clean RGB pixel values underneath the defect)
    """

    CHANNELS = ("defect_prob", "clean_r", "clean_g", "clean_b")

    def __init__(self, base: int = 40):
        super().__init__()
        self.stem = _LineRidgeStem()
        # 12 stem channels (from RGB) + 3 raw RGB + 4 semantic part channels = 19 input channels
        self.s1 = nn.Sequential(
            nn.Conv2d(19, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
            _DSConvBlock(base, base),
            _DSConvBlock(base, base),
        )
        self.s2 = nn.Sequential(_DSConvBlock(base, base * 2, stride=2), _DSConvBlock(base * 2, base * 2), _SEBlock(base * 2))
        self.s3 = nn.Sequential(_DSConvBlock(base * 2, base * 4, stride=2), _DSConvBlock(base * 4, base * 4), _SEBlock(base * 4))
        self.s4 = nn.Sequential(
            _DSConvBlock(base * 4, base * 6, stride=2),
            _DSConvBlock(base * 6, base * 6, dilation=2),
            _DSConvBlock(base * 6, base * 6, dilation=4),
            _SEBlock(base * 6),
        )
        self.d3 = nn.Sequential(_DSConvBlock(base * 6 + base * 4, base * 4), _SEBlock(base * 4))
        self.d2 = nn.Sequential(_DSConvBlock(base * 4 + base * 2, base * 2), _SEBlock(base * 2))
        self.d1 = nn.Sequential(_DSConvBlock(base * 2 + base, base), _DSConvBlock(base, base))
        self.head_mask = nn.Sequential(_DSConvBlock(base, base // 2), nn.Conv2d(base // 2, 1, 1))
        self.head_rgb = nn.Sequential(_DSConvBlock(base, base // 2), nn.Conv2d(base // 2, 3, 1))

    def forward(self, feat_in: torch.Tensor) -> torch.Tensor:
        rgb = feat_in[:, :3]
        parts = feat_in[:, 3:]
        stem_rgb = self.stem(rgb)
        e1 = self.s1(torch.cat([stem_rgb, rgb, parts], dim=1))
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        u3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        mask = torch.sigmoid(self.head_mask(u1))
        clean_rgb = torch.sigmoid(self.head_rgb(u1))
        return torch.cat([mask, clean_rgb], dim=1)
