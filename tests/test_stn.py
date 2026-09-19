#!/usr/bin/env python3
"""
tests/test_stn.py
Unit tests for Micro Spatial Transformer Network (src/pipeline/stn.py).
Validates:
1. Strict identity transformation at initialization (zero L1 degradation).
2. Gradient flow across differentiable GridSample.
3. Bounded affine stability (preventing spatial foldover / inversion).
4. Monolithic STN-LPRNet composition.
5. ONNX Export and ONNX Runtime execution.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from src.pipeline.stn import MicroSpatialTransformer, STNLPRNet, export_stn_onnx
from src.pipeline.ocr import LPRNet, NUM_CLASSES


class TestMicroSTN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_identity_at_initialization(self):
        """Validates that newly initialized STN produces exact identity transformation."""
        stn = MicroSpatialTransformer(bounded=True).eval()
        x = torch.randn(2, 3, 36, 160)

        with torch.no_grad():
            rectified, theta = stn(x)

        expected_theta = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ])

        self.assertTrue(torch.allclose(theta, expected_theta, atol=1e-5))
        # Bilinear interpolation of identity grid matches input with minimal floating error
        diff = torch.abs(rectified - x).max().item()
        self.assertLess(diff, 1e-4, f"Identity reconstruction error too high: {diff}")

    def test_bounded_deformations(self):
        """Ensures that bounded affine prevents catastrophic image inversion."""
        stn = MicroSpatialTransformer(bounded=True, max_scale_skew=0.25, max_translation=0.20)
        # Force extreme weights in regression head
        stn.fc_head[-1].weight.data.fill_(10.0)
        stn.fc_head[-1].bias.data.fill_(10.0)

        x = torch.randn(1, 3, 36, 160)
        rectified, theta = stn(x)

        self.assertEqual(rectified.shape, (1, 3, 36, 160))
        # Ensure scale parameters remain positive and within bounds (0.75 <= scale <= 1.25)
        s_x = theta[0, 0, 0].item()
        s_y = theta[0, 1, 1].item()
        self.assertGreater(s_x, 0.70)
        self.assertLess(s_x, 1.30)
        self.assertGreater(s_y, 0.70)
        self.assertLess(s_y, 1.30)

    def test_gradient_flow(self):
        """Checks end-to-end differentiability through bilinear grid sampling."""
        stn = MicroSpatialTransformer()
        stn.train()
        x = torch.randn(2, 3, 36, 160, requires_grad=True)

        rectified, theta = stn(x)
        loss = rectified.sum() + theta.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        # Check that localization head receives gradients
        last_layer = stn.fc_head[-1]
        self.assertIsNotNone(last_layer.weight.grad)
        self.assertGreater(last_layer.weight.grad.abs().sum().item(), 0.0)

    def test_stn_lprnet_composition(self):
        """Tests monolithic composition of STN + LPRNet backbone."""
        lprnet = LPRNet(num_classes=NUM_CLASSES)
        composite = STNLPRNet(ocr_backbone=lprnet)
        composite.eval()

        x = torch.randn(2, 3, 36, 160)
        with torch.no_grad():
            logits = composite(x)

        # Output logits shape must be (B, T=40, num_classes=24)
        self.assertEqual(logits.shape, (2, 40, NUM_CLASSES))

    def test_onnx_export_and_runtime(self):
        """Verifies that Micro-STN exports to valid ONNX and executes in ONNX Runtime."""
        import onnxruntime as ort

        stn = MicroSpatialTransformer(bounded=True).eval()
        onnx_path = os.path.join(ROOT_DIR, "models", "test_stn_micro.onnx")

        export_stn_onnx(stn, onnx_path, input_shape=(1, 3, 36, 160), opset_version=18)
        self.assertTrue(os.path.exists(onnx_path))

        # Test ONNX Runtime inference
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_x = np.random.randn(1, 3, 36, 160).astype(np.float32)

        out_rect, out_theta = sess.run(None, {"crop_input": dummy_x})
        self.assertEqual(out_rect.shape, (1, 3, 36, 160))
        self.assertEqual(out_theta.shape, (1, 2, 3))

        # Verify identity output in ONNX
        diff = np.max(np.abs(out_rect - dummy_x))
        self.assertLess(diff, 1e-4)

        if os.path.exists(onnx_path):
            try:
                os.remove(onnx_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
