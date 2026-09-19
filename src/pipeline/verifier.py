#!/usr/bin/env python3
"""
OmniPlate-RU — Plate Verifier Module (Substage 3.3).
Lightweight binary CNN classifier for distinguishing valid Russian license plate crops
from false positive noise (car grilles, headlights, bumper seams, road textures, shadows).

- Canonical input: 160x36 px (W x H) RGB tensor
- Architecture: Compact Depthwise-Separable ConvNet (<300 KB ONNX)
- Latency: <0.4 ms on GPU, <2.0 ms on CPU
- Protects against Fatal Penalties on negative frames (class 'other')
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure PyTorch CUDA DLLs are found by ONNX Runtime on Windows
if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
        if str(torch_lib) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


if torch is not None:
    class ConvBNAct(nn.Module):
        def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.bn(self.conv(x)))

    class DWBlock(nn.Module):
        def __init__(self, in_c: int, out_c: int, s: int = 2):
            super().__init__()
            self.dw = nn.Conv2d(in_c, in_c, 3, stride=s, padding=1, groups=in_c, bias=False)
            self.bn1 = nn.BatchNorm2d(in_c)
            self.act1 = nn.ReLU(inplace=True)
            self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_c)
            self.act2 = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.act1(self.bn1(self.dw(x)))
            x = self.act2(self.bn2(self.pw(x)))
            return x

    class PlateVerifierNet(nn.Module):
        """Ultra-fast binary classifier: Plate (1) vs Non-Plate (0)."""
        def __init__(self, dropout: float = 0.2):
            super().__init__()
            self.stem = ConvBNAct(3, 32, k=3, s=1, p=1)        # (B, 32, 36, 160)
            self.down1 = DWBlock(32, 64, s=2)                  # (B, 64, 18, 80)
            self.down2 = DWBlock(64, 96, s=2)                  # (B, 96, 9, 40)
            self.down3 = DWBlock(96, 128, s=2)                 # (B, 128, 5, 20)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(128, 48),
                nn.ReLU(inplace=True),
                nn.Linear(48, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.pool(x)
            return self.classifier(x)
else:
    PlateVerifierNet = None


class PlateVerifier:
    """Production runtime verifier with ONNX Runtime and PyTorch fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        use_onnx: bool = True,
        threshold: float = 0.50,
    ):
        self.device = device
        self.use_onnx = use_onnx
        self.threshold = threshold
        self.session = None
        self.torch_model = None

        if model_path is None:
            cand_onnx = Path("models/plate_verifier.onnx")
            cand_pt = Path("models/plate_verifier.pt")
            if cand_onnx.exists():
                model_path = str(cand_onnx)
            elif cand_pt.exists():
                model_path = str(cand_pt)

        if model_path and os.path.exists(model_path):
            if model_path.endswith(".onnx") and ort is not None:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.enable_mem_pattern = True
                opts.enable_cpu_mem_arena = True

                cuda_opts = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": "1",
                }
                providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"] if "cuda" in device.lower() else ["CPUExecutionProvider"]
                try:
                    self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
                    self.use_onnx = True
                except Exception:
                    self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
                    self.use_onnx = True
            elif model_path.endswith(".pt") and torch is not None:
                self.torch_model = PlateVerifierNet()
                ckpt = torch.load(model_path, map_location=device)
                state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                self.torch_model.load_state_dict(state)
                self.torch_model.to(device).eval()
                self.use_onnx = False

    def is_valid(self) -> bool:
        return (self.session is not None) or (self.torch_model is not None)

    def verify_single(self, crop: np.ndarray) -> Tuple[bool, float]:
        """
        Verifies whether crop is a valid Russian plate or background noise.
        Returns:
            (is_plate: bool, plate_score: float in [0.0, 1.0])
        """
        if not self.is_valid() or crop is None or crop.size == 0:
            return True, 1.0  # Pass through if model not initialized

        # Prepare 160x36 RGB tensor
        h, w = crop.shape[:2]
        if (h, w) != (36, 160):
            resized = cv2.resize(crop, (160, 36), interpolation=cv2.INTER_LINEAR)
        else:
            resized = crop

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]  # (1, 3, 36, 160)

        if self.use_onnx and self.session is not None:
            inp_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {inp_name: norm})[0]  # (1, 1)
            logit = float(out[0, 0])
        elif self.torch_model is not None and torch is not None:
            with torch.no_grad():
                t = torch.from_numpy(norm).to(self.device)
                out = self.torch_model(t)
                logit = float(out.item())
        else:
            return True, 1.0

        # Sigmoid activation
        score = float(1.0 / (1.0 + np.exp(-logit)))
        return bool(score >= self.threshold), score

    def verify_batch(self, crops: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Batch verification on N crops."""
        if not self.is_valid() or not crops:
            return [(True, 1.0)] * len(crops)

        batch_tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                resized = np.zeros((36, 160, 3), dtype=np.uint8)
            elif crop.shape[:2] != (36, 160):
                resized = cv2.resize(crop, (160, 36), interpolation=cv2.INTER_LINEAR)
            else:
                resized = crop
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            batch_tensors.append(norm)

        batch_arr = np.stack(batch_tensors, axis=0)  # (N, 3, 36, 160)

        if self.use_onnx and self.session is not None:
            inp_name = self.session.get_inputs()[0].name
            outs = self.session.run(None, {inp_name: batch_arr})[0]  # (N, 1)
            logits = outs.squeeze(-1)
        elif self.torch_model is not None and torch is not None:
            with torch.no_grad():
                t = torch.from_numpy(batch_arr).to(self.device)
                logits = self.torch_model(t).squeeze(-1).cpu().numpy()
        else:
            return [(True, 1.0)] * len(crops)

        scores = 1.0 / (1.0 + np.exp(-logits))
        results = []
        for s in scores:
            results.append((bool(float(s) >= self.threshold), float(s)))
        return results
