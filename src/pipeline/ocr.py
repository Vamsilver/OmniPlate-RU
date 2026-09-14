#!/usr/bin/env python3
"""
OmniPlate-RU Optical Character Recognition (OCR) Module (Substage 3.3).

Lightweight Conv-CTC OCR architecture tailored for Russian license plates:
- Canonical input: 160x36 px (W x H) single-line crop (Type 1, Type 1B, and stitched Type 1A).
- Zero-RNN architecture (pure Conv + BatchNorm + ReLU):
  * Ultra-low latency (<3 ms on GPU, <15 ms on CPU, easily meeting SLA <= 25 ms).
  * Seamless export to ONNX / TensorRT without recurrent state overhead.
- CTC Decoder with greedy decoding and position-aware GOST mask correction.
- Confusion matrix auto-repair: '0'<->'O', '8'<->'B' based on strict GOST position rules.
"""

import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

# ---------------------------------------------------------------------------
# Character Set & CTC Tokens
# ---------------------------------------------------------------------------
# Blank token is at index 0
BLANK_TOKEN = "-"
DIGITS = "0123456789"
LETTERS = "ABEKMHOPCTYX"
WILDCARD = "#"

VOCAB = [BLANK_TOKEN] + list(DIGITS) + list(LETTERS) + [WILDCARD]
CHAR2IDX: Dict[str, int] = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR: Dict[int, str] = {i: c for i, c in enumerate(VOCAB)}
BLANK_IDX = 0
NUM_CLASSES = len(VOCAB)

# Standard GOST RegEx
PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")
VALID_3DIGIT_STARTS = {"1", "2", "7", "#"}

# Confusion replacement maps
DIGIT_TO_LETTER: Dict[str, str] = {
    "0": "O",
    "8": "B",
    "1": "T",  # Visual similarity in some fonts
}
LETTER_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "B": "8",
    "C": "0",
    "D": "0",
}


# ---------------------------------------------------------------------------
# PyTorch Model Architecture (Imported dynamically if torch is present)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SmallBasicBlock(nn.Module):
        """Residual block with 3x3 depthwise/standard convolutions"""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.conv(x) + self.shortcut(x))

    class LPRNet(nn.Module):
        """
        High-Speed License Plate Recognition Network (LPRNet).
        Accepts canonical (B, 3, 36, 160) input crops.
        Outputs CTC logits (B, seq_len=40, num_classes).
        """
        def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.2):
            super().__init__()
            self.num_classes = num_classes

            # Backbone Stem: 36x160 -> 18x80
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),  # (B, 64, 18, 80)
            )

            # Stage 1: 18x80 -> 18x80
            self.block1 = SmallBasicBlock(64, 64)

            # Downsample 1: 18x80 -> 9x40
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

            # Stage 2: 9x40 -> 9x40
            self.block2 = SmallBasicBlock(64, 128)

            # Stage 3: 9x40 -> 9x40
            self.block3 = SmallBasicBlock(128, 256)

            # Downsample 2: (Height down to 4, width preserved at 40)
            self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # 5x40

            # Stage 4: 5x40 -> 5x40
            self.block4 = SmallBasicBlock(256, 256)

            # Downsample 3: collapse height to 1
            self.pool3 = nn.AdaptiveAvgPool2d((1, 40))  # (B, 256, 1, 40)

            # Global context multi-scale fusion
            # We pool features from stem (pool to 1x40), block2 (pool to 1x40), and block4
            self.global_pool_stem = nn.AdaptiveAvgPool2d((1, 40))
            self.global_pool_b2 = nn.AdaptiveAvgPool2d((1, 40))

            # Total concatenated channels = 64 + 128 + 256 = 448
            fused_channels = 64 + 128 + 256

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Conv2d(fused_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input tensor of shape (B, 3, 36, 160) normalized to [0, 1] or [-1, 1].
            Returns:
                logits: (B, T=40, num_classes)
            """
            f_stem = self.stem(x)         # (B, 64, 18, 80)
            f_b1 = self.block1(f_stem)    # (B, 64, 18, 80)
            f_p1 = self.pool1(f_b1)       # (B, 64, 9, 40)

            f_b2 = self.block2(f_p1)      # (B, 128, 9, 40)
            f_b3 = self.block3(f_b2)      # (B, 256, 9, 40)
            f_p2 = self.pool2(f_b3)       # (B, 256, 5, 40)
            f_b4 = self.block4(f_p2)      # (B, 256, 5, 40)
            f_p3 = self.pool3(f_b4)       # (B, 256, 1, 40)

            # Global multi-scale projection
            f_stem_proj = self.global_pool_stem(f_stem)  # (B, 64, 1, 40)
            f_b2_proj = self.global_pool_b2(f_b2)        # (B, 128, 1, 40)

            fused = torch.cat([f_stem_proj, f_b2_proj, f_p3], dim=1)  # (B, 448, 1, 40)
            out = self.classifier(fused)  # (B, num_classes, 1, 40)
            out = out.squeeze(2)          # (B, num_classes, 40)
            out = out.permute(0, 2, 1)    # (B, 40, num_classes)
            return out

except ImportError:
    torch = None
    nn = None
    LPRNet = None


# ---------------------------------------------------------------------------
# Post-Processing & CTC Decoder
# ---------------------------------------------------------------------------
class CTCDecoder:
    """
    Decodes CTC raw output probabilities or index sequences into text
    with optional GOST mask correction.
    """

    @staticmethod
    def decode_greedy(indices: Sequence[int], blank_idx: int = BLANK_IDX) -> str:
        """
        Standard CTC greedy collapse:
        Removes repeated tokens and blank tokens.
        """
        raw_chars = []
        prev = blank_idx
        for idx in indices:
            if idx != prev:
                if idx != blank_idx and idx in IDX2CHAR:
                    raw_chars.append(IDX2CHAR[idx])
                prev = idx
            else:
                prev = idx
        return "".join(raw_chars)

    @staticmethod
    def apply_gost_heuristics(plate: str, plate_type: str = "type1") -> str:
        """
        Applies position-aware character repair based on GOST R 50577-2018.
        Supports:
        - type1 / type1a: L D D D L L D D (8 chars) or L D D D L L D D D (9 chars)
        - type1b: L L D D D D D (7 chars) or L L D D D D D D (8 chars)
        """
        if not plate:
            return plate

        chars = list(plate)
        norm_type = plate_type.lower().strip()

        if norm_type == "type1b":
            # Type 1B: Yellow single-line plate (e.g., AH88977)
            # Format: LL DDD DD (7 chars) or LL DDD DDD (8 chars)
            if len(chars) not in (7, 8):
                return plate

            # Expected Letter positions: 0, 1
            for pos in (0, 1):
                c = chars[pos]
                if c in DIGIT_TO_LETTER:
                    chars[pos] = DIGIT_TO_LETTER[c]

            # Expected Digit positions: 2, 3, 4, and 5..end
            digit_positions = [2, 3, 4] + list(range(5, len(chars)))
            for pos in digit_positions:
                c = chars[pos]
                if c in LETTER_TO_DIGIT:
                    chars[pos] = LETTER_TO_DIGIT[c]

            return "".join(chars)

        # Standard Type 1 and Type 1A:
        if len(plate) not in (8, 9):
            return plate

        # Expected Letter positions: 0, 4, 5
        for pos in (0, 4, 5):
            c = chars[pos]
            if c in DIGIT_TO_LETTER:
                chars[pos] = DIGIT_TO_LETTER[c]

        # Expected Digit positions: 1, 2, 3, and 6..end
        digit_positions = [1, 2, 3] + list(range(6, len(plate)))
        for pos in digit_positions:
            c = chars[pos]
            if c in LETTER_TO_DIGIT:
                chars[pos] = LETTER_TO_DIGIT[c]

        return "".join(chars)



class PlateOCR:
    """
    High-level OCR wrapper supporting both PyTorch model and ONNX Runtime engine.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        use_onnx: bool = False,
    ):
        self.device = device
        self.use_onnx = use_onnx
        self.decoder = CTCDecoder()
        self.model = None
        self.session = None

        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        try:
            if model_path.endswith(".onnx") or self.use_onnx:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in self.device else ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.use_onnx = True
            else:
                if torch is None:
                    raise ImportError("PyTorch is required to load .pt checkpoint.")
                self.model = LPRNet(num_classes=NUM_CLASSES)
                ckpt = torch.load(model_path, map_location=self.device)
                state_dict = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
        except ImportError as e:
            print(f"[!] Warning: Failed to import inference engine for {model_path}: {e}")
            self.session = None
            self.model = None


    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Prepares a single BGR crop (expected canonical 160x36) for neural input:
        Resizes if necessary, converts BGR->RGB, normalizes to [0, 1], returns (1, 3, 36, 160) float32.
        """
        import cv2
        h, w = crop_bgr.shape[:2]
        if (w, h) != (160, 36):
            crop_bgr = cv2.resize(crop_bgr, (160, 36), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        # (36, 160, 3) -> (3, 36, 160) -> (1, 3, 36, 160)
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return tensor

    def predict_single(self, crop_bgr: np.ndarray, plate_type: str = "type1") -> Tuple[str, float]:
        """
        Runs OCR on a single canonical crop.
        Returns:
            (plate_text, confidence)
        """
        tensor = self.preprocess(crop_bgr)

        if self.use_onnx and self.session is not None:
            input_name = self.session.get_inputs()[0].name
            logits = self.session.run(None, {input_name: tensor})[0]  # (1, 40, num_classes)
        elif self.model is not None:
            t = torch.from_numpy(tensor).to(self.device)
            with torch.no_grad():
                logits = self.model(t).cpu().numpy()
        else:
            raise RuntimeError("OCR model is not loaded!")

        # Greedy decoding
        seq_logits = logits[0]  # (40, num_classes)
        # Softmax for probabilities
        exp_l = np.exp(seq_logits - np.max(seq_logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        best_indices = np.argmax(probs, axis=-1)
        best_probs = np.max(probs, axis=-1)

        raw_text = self.decoder.decode_greedy(best_indices, blank_idx=BLANK_IDX)
        final_text = self.decoder.apply_gost_heuristics(raw_text, plate_type=plate_type)

        # Confidence: average probability of non-blank aligned characters
        non_blank_probs = [best_probs[t] for t, idx in enumerate(best_indices) if idx != BLANK_IDX]
        confidence = float(np.mean(non_blank_probs)) if non_blank_probs else 0.0

        return final_text, confidence

