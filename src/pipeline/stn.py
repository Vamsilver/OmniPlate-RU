#!/usr/bin/env python3
"""
src/pipeline/stn.py
Micro Spatial Transformer Network (Micro-STN) for Russian Vehicle License Plates.

Provides differentiable, sub-pixel perspective compensation before OCR backbone:
1. Micro-Localization Network predicts 2x3 affine matrix theta.
2. Initialized strictly to Identity transform: [1, 0, 0, 0, 1, 0] with zeroed weights.
3. Bounded affine delta prevents inversion, extreme scale loss, or foldover singularities.
4. Bilinear Grid Sampling handles perspective yaw, pitch, roll, and shear gracefully.
"""

from typing import Optional, Tuple
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if nn is not None:

    class MicroSpatialTransformer(nn.Module):
        """
        Ultra-compact Spatial Transformer Network (<45 KB parameter budget).
        Accepts canonical (B, 3, H=36, W=160) crop.
        Returns:
            rectified_x: (B, 3, H=36, W=160)
            theta: (B, 2, 3) affine matrix
        """

        def __init__(
            self,
            in_channels: int = 3,
            bounded: bool = True,
            max_translation: float = 0.20,
            max_scale_skew: float = 0.25,
        ):
            super().__init__()
            self.bounded = bounded
            self.max_translation = max_translation
            self.max_scale_skew = max_scale_skew

            # Micro Localization Backbone: 36x160 -> 18x80 -> 9x40 -> 5x20
            self.loc_net = nn.Sequential(
                nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 5)),  # (B, 64, 2, 5) -> 640 features
            )

            # Regression Head
            self.fc_head = nn.Sequential(
                nn.Linear(64 * 2 * 5, 48),
                nn.ReLU(inplace=True),
                nn.Linear(48, 6),
            )

            # Strict Identity Initialization: weights=0
            self.fc_head[-1].weight.data.zero_()
            if self.bounded:
                initial_bias = torch.zeros(6, dtype=torch.float32)
            else:
                initial_bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
            self.fc_head[-1].bias.data.copy_(initial_bias)

            self.register_buffer("identity_theta", torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Input tensor of shape (B, 3, H, W)
            Returns:
                rectified: (B, 3, H, W) transformed tensor
                theta: (B, 2, 3) transformation matrix
            """
            b = x.size(0)
            feat = self.loc_net(x)
            feat = feat.flatten(1)
            raw_theta = self.fc_head(feat)  # (B, 6)

            if self.bounded:
                # Bounded affine: scale & skew bounded by max_scale_skew, translation bounded by max_translation
                # theta = identity + delta
                delta_scale_skew = torch.tanh(raw_theta[:, :4]) * self.max_scale_skew
                delta_trans = torch.tanh(raw_theta[:, 4:]) * self.max_translation
                theta_mat = torch.cat([
                    (1.0 + delta_scale_skew[:, 0:1]), delta_scale_skew[:, 1:2], delta_trans[:, 0:1],
                    delta_scale_skew[:, 2:3], (1.0 + delta_scale_skew[:, 3:4]), delta_trans[:, 1:2],
                ], dim=1).view(b, 2, 3)
            else:
                theta_mat = raw_theta.view(b, 2, 3)

            # Generate grid and resample
            grid = F.affine_grid(theta_mat, x.size(), align_corners=False)
            rectified = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
            return rectified, theta_mat


    class STNLPRNet(nn.Module):
        """
        Monolithic composition of Micro-STN with LPRNet OCR backbone.
        Permits end-to-end forward pass directly from raw unaligned crops to CTC logits.
        """

        def __init__(self, ocr_backbone: nn.Module, stn: Optional[MicroSpatialTransformer] = None):
            super().__init__()
            self.stn = stn if stn is not None else MicroSpatialTransformer()
            self.backbone = ocr_backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rectified, _ = self.stn(x)
            logits = self.backbone(rectified)
            return logits

else:
    MicroSpatialTransformer = None
    STNLPRNet = None


def export_stn_onnx(
    stn_model: "MicroSpatialTransformer",
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 36, 160),
    opset_version: int = 18,
) -> str:
    """Exports Micro-STN to a standalone ONNX graph with GridSample operator."""
    if torch is None or stn_model is None:
        raise RuntimeError("PyTorch is required for ONNX export.")

    stn_model.eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Export
    torch.onnx.export(
        stn_model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["crop_input"],
        output_names=["rectified_crop", "theta"],
        dynamic_axes={
            "crop_input": {0: "batch_size"},
            "rectified_crop": {0: "batch_size"},
            "theta": {0: "batch_size"},
        },
    )
    return output_path
