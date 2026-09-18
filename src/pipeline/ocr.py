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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
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


from src.pipeline.decoder import FSMBeamSearchDecoder, CTCDecoder

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
PLATE_TYPE2_REGEX = re.compile(r"^[ABEKMHOPCTYX#]{2}[\d#]{4}[\d#]{2,3}$")
PLATE_TYPE1B_REGEX = re.compile(r"^[ABEKMHOPCTYX#]{2}[\d#]{3}[\d#]{2,3}$")
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

# Official Russian GIBDD 3-digit region codes
VALID_3DIGIT_REGIONS = {
    "102", "113", "116", "121", "122", "123", "124", "125", "126",
    "134", "136", "138", "142", "147", "150", "152", "154", "155", "156",
    "159", "161", "163", "164", "169", "172", "173", "174", "177", "178", "180",
    "181", "184", "185", "186", "190", "193", "196", "197", "198", "199",
    "252", "277", "323", "336", "702", "716", "725", "750", "754", "761", "763", "774", "777",
    "790", "797", "799", "977"
}

# Official Russian GIBDD 2-digit region codes (active & historical legal codes 01..99)
VALID_2DIGIT_REGIONS = {f"{i:02d}" for i in range(1, 100)}

# Empirical demographic & traffic frequency prior weights for Russian regions
# Tier 1: Federal capital hubs (Moscow, Moscow Oblast, Saint Petersburg, Leningrad Oblast)
TIER1_REGIONS = {
    "77", "97", "99", "177", "197", "199", "777", "797", "799", "977",
    "50", "90", "150", "190", "750", "790", "550",
    "78", "98", "178", "198",
    "47", "147"
}

# Tier 2: Million-plus population centers & major economic regional hubs
TIER2_REGIONS = {
    "23", "93", "123", "193",  # Krasnodar Krai
    "66", "96", "196",        # Sverdlovsk (Yekaterinburg)
    "16", "116", "716",       # Tatarstan (Kazan)
    "52", "152", "252",       # Nizhny Novgorod
    "63", "163", "763",       # Samara
    "61", "161", "761",       # Rostov
    "02", "102", "702",       # Bashkortostan (Ufa)
    "54", "154", "754",       # Novosibirsk
    "74", "174", "774",       # Chelyabinsk
    "59", "159",              # Perm Krai
    "24", "124",              # Krasnoyarsk Krai
    "36", "136",              # Voronezh
    "34", "134",              # Volgograd
    "25", "125",              # Primorsky (Vladivostok)
    "64", "164",              # Saratov
    "72", "172",              # Tyumen
    "55", "155",              # Omsk
    "38", "138",              # Irkutsk
    "26", "126",              # Stavropol Krai
    "42", "142",              # Kemerovo
    "35", "39", "31", "71", "69", "33", "62", "40", "48", "58", "43", "73", "173",
    "21", "121", "18", "82", "92", "56", "156", "11", "169", "86", "186", "89",
}

REGION_FREQUENCY_WEIGHTS: Dict[str, float] = {}
for _r in TIER1_REGIONS:
    REGION_FREQUENCY_WEIGHTS[_r] = 0.06
for _r in TIER2_REGIONS:
    if _r not in REGION_FREQUENCY_WEIGHTS:
        REGION_FREQUENCY_WEIGHTS[_r] = 0.02



OPTICAL_DIGIT_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("8", "0"): 0.5, ("0", "8"): 0.5,
    ("5", "0"): 0.5, ("0", "5"): 0.5,
    ("6", "5"): 0.7, ("5", "6"): 0.7,
    ("4", "7"): 0.5, ("7", "4"): 0.5,
    ("2", "7"): 0.5, ("7", "2"): 0.5,
    ("2", "9"): 0.8, ("9", "2"): 0.8,
    ("1", "7"): 0.6, ("7", "1"): 0.6,
    ("5", "9"): 0.7, ("9", "5"): 0.7,
    ("3", "8"): 0.7, ("8", "3"): 0.7,
    ("6", "8"): 0.7, ("8", "6"): 0.7,
    ("9", "0"): 0.8, ("0", "9"): 0.8,
    ("9", "7"): 0.8, ("7", "9"): 0.8,
}

OPTICAL_LETTER_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("K", "H"): 0.5, ("H", "K"): 0.5,
    ("Y", "T"): 0.5, ("T", "Y"): 0.5,
    ("M", "T"): 0.6, ("T", "M"): 0.6,
    ("C", "M"): 0.6, ("M", "C"): 0.6,
    ("C", "O"): 0.6, ("O", "C"): 0.6,
    ("B", "O"): 0.6, ("O", "B"): 0.6,
    ("E", "P"): 0.6, ("P", "E"): 0.6,
    ("B", "P"): 0.7, ("P", "B"): 0.7,
    ("Y", "E"): 0.6, ("E", "Y"): 0.6,
    ("C", "A"): 0.7, ("A", "C"): 0.7,
    ("H", "M"): 0.6, ("M", "H"): 0.6,
    ("X", "K"): 0.7, ("K", "X"): 0.7,
    ("X", "H"): 0.7, ("H", "X"): 0.7,
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
# Post-Processing & CTC Decoder (SSOT: src.pipeline.decoder)
# ---------------------------------------------------------------------------
# CTCDecoder and FSMBeamSearchDecoder are imported from src.pipeline.decoder




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
            is_onnx = model_path.endswith(".onnx")
            if is_onnx:
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
                self.use_onnx = False
        except Exception as e:
            print(f"[!] Warning: Failed to import/load inference engine for {model_path}: {e}")
            self.session = None
            self.model = None


    @staticmethod
    def apply_adaptive_clahe(crop_bgr: np.ndarray) -> np.ndarray:
        """
        Applies adaptive Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the Luminance (L) channel in LAB color space for dark or low-contrast crops.
        Preserves natural color fidelity in A and B channels (critical for yellow Type 1B).
        """
        import cv2
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        h, w = crop_bgr.shape[:2]
        if (w, h) != (160, 36):
            crop_bgr = cv2.resize(crop_bgr, (160, 36), interpolation=cv2.INTER_LINEAR)

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = float(np.mean(l))
        std_l = float(np.std(l))

        # Check if crop is dark or low contrast
        if mean_l < 90:
            clip = 2.5
        elif mean_l < 120:
            clip = 2.0
        elif mean_l < 140 and std_l < 38:
            clip = 1.5
        else:
            return crop_bgr

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(2, 8))
        cl = clahe.apply(l)

        if mean_l >= 90:
            alpha = min(1.0, max(0.4, (135.0 - mean_l) / 45.0))
            cl = cv2.addWeighted(cl, alpha, l, 1.0 - alpha, 0)

        lab_enh = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    def preprocess(self, crop_bgr: np.ndarray, apply_clahe: bool = False) -> np.ndarray:
        """
        Prepares a single BGR crop (expected canonical 160x36) for neural input:
        Resizes if necessary, optionally applies adaptive CLAHE, converts BGR->RGB,
        normalizes to [0, 1], returns (1, 3, 36, 160) float32.
        """
        import cv2
        h, w = crop_bgr.shape[:2]
        if (w, h) != (160, 36):
            crop_bgr = cv2.resize(crop_bgr, (160, 36), interpolation=cv2.INTER_LINEAR)
        if apply_clahe:
            crop_bgr = self.apply_adaptive_clahe(crop_bgr)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        # (36, 160, 3) -> (3, 36, 160) -> (1, 3, 36, 160)
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return tensor

    def predict_single(
        self,
        crop_bgr: np.ndarray,
        plate_type: str = "type1",
        return_type: bool = False,
        use_beam_search: bool = True,
        beam_width: int = 10,
        use_tta: bool = True,
    ) -> Union[Tuple[str, float], Tuple[str, float, str]]:
        """
        Runs OCR on a single canonical crop using multi-mask CTC hypothesis scoring.
        Supports Test-Time Augmentation (TTA) with adaptive CLAHE fusion on dark/low-contrast crops.
        Returns:
            (plate_text, confidence) if return_type=False
            (plate_text, confidence, detected_plate_type) if return_type=True
        """
        import cv2
        h, w = crop_bgr.shape[:2]
        if (w, h) != (160, 36):
            crop_bgr = cv2.resize(crop_bgr, (160, 36), interpolation=cv2.INTER_LINEAR)

        # Evaluate darkness / contrast for adaptive CLAHE
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        mean_l = float(np.mean(l_chan))
        std_l = float(np.std(l_chan))
        # Activate adaptive CLAHE on dark or low-contrast plates
        is_dark_or_low_contrast = (mean_l < 75) or (mean_l < 100 and std_l < 22)

        if use_tta and is_dark_or_low_contrast:
            crop_clahe = self.apply_adaptive_clahe(crop_bgr)
            tensor_orig = self.preprocess(crop_bgr, apply_clahe=False)
            tensor_clahe = self.preprocess(crop_clahe, apply_clahe=False)
            tensor = np.concatenate([tensor_orig, tensor_clahe], axis=0)  # (2, 3, 36, 160)

            if self.use_onnx and self.session is not None:
                input_name = self.session.get_inputs()[0].name
                logits_batch = self.session.run(None, {input_name: tensor})[0]  # (2, 40, num_classes)
            elif self.model is not None:
                t = torch.from_numpy(tensor).to(self.device)
                with torch.no_grad():
                    logits_batch = self.model(t).cpu().numpy()
            else:
                raise RuntimeError("OCR model is not loaded!")

            # Safe Hypothesis Selection: Score both branches independently (no destructive logit blurring)
            text_orig, type_orig, conf_orig = self.decoder.score_hypotheses(
                logits_batch[0],
                plate_type_prior=plate_type,
                blank_idx=BLANK_IDX,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
            )
            text_clahe, type_clahe, conf_clahe = self.decoder.score_hypotheses(
                logits_batch[1],
                plate_type_prior=plate_type,
                blank_idx=BLANK_IDX,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
            )

            # Choose CLAHE only if it rescues wildcards (#) or achieves substantially higher confidence
            wc_orig = text_orig.count("#")
            wc_clahe = text_clahe.count("#")
            if wc_clahe < wc_orig or (wc_clahe == wc_orig and conf_clahe > conf_orig + 0.08):
                final_text, detected_type, confidence = text_clahe, type_clahe, conf_clahe
            else:
                final_text, detected_type, confidence = text_orig, type_orig, conf_orig
        else:
            tensor = self.preprocess(crop_bgr, apply_clahe=False)

            if self.use_onnx and self.session is not None:
                input_name = self.session.get_inputs()[0].name
                logits = self.session.run(None, {input_name: tensor})[0]  # (1, 40, num_classes)
            elif self.model is not None:
                t = torch.from_numpy(tensor).to(self.device)
                with torch.no_grad():
                    logits = self.model(t).cpu().numpy()
            else:
                raise RuntimeError("OCR model is not loaded!")
            seq_logits = logits[0]  # (40, num_classes)

            final_text, detected_type, confidence = self.decoder.score_hypotheses(
                seq_logits,
                plate_type_prior=plate_type,
                blank_idx=BLANK_IDX,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
            )

        if return_type:
            return final_text, confidence, detected_type
        return final_text, confidence

    def predict_batch(
        self,
        crops_bgr: Sequence[np.ndarray],
        plate_types: Optional[Sequence[str]] = None,
        return_type: bool = True,
        use_beam_search: bool = True,
        beam_width: int = 10,
    ) -> List[Union[Tuple[str, float], Tuple[str, float, str]]]:
        """
        Runs batch OCR inference on N canonical crops simultaneously in a single forward pass [N, 3, 36, 160].
        Greatly accelerates multi-plate images and batch benchmarks without looping.
        """
        if not crops_bgr:
            return []

        n = len(crops_bgr)
        p_types = list(plate_types) if plate_types is not None else ["type1"] * n
        if len(p_types) < n:
            p_types.extend(["type1"] * (n - len(p_types)))

        # Preprocess all crops into canonical (N, 3, 36, 160)
        tensors = []
        for crop in crops_bgr:
            tensors.append(self.preprocess(crop, apply_clahe=False))
        batch_tensor = np.concatenate(tensors, axis=0)  # (N, 3, 36, 160)

        if self.use_onnx and self.session is not None:
            input_name = self.session.get_inputs()[0].name
            logits_batch = self.session.run(None, {input_name: batch_tensor})[0]  # (N, 40, num_classes)
        elif self.model is not None:
            t = torch.from_numpy(batch_tensor).to(self.device)
            with torch.no_grad():
                logits_batch = self.model(t).cpu().numpy()
        else:
            raise RuntimeError("OCR model is not loaded!")

        results: List[Union[Tuple[str, float], Tuple[str, float, str]]] = []
        for i in range(n):
            seq_logits = logits_batch[i]
            final_text, detected_type, confidence = self.decoder.score_hypotheses(
                seq_logits,
                plate_type_prior=p_types[i],
                blank_idx=BLANK_IDX,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
            )
            if return_type:
                results.append((final_text, round(confidence, 4), detected_type))
            else:
                results.append((final_text, round(confidence, 4)))

        return results

    def predict_type1a_dual(
        self,
        top_crop: np.ndarray,
        bot_crop: np.ndarray,
        beam_width: int = 10,
    ) -> Tuple[str, float]:
        """
        Dual-Line Pass OCR for Russian Type 1A square plates using already trained LPRNet:
        1. Resizes top line crop to (160, 36) and bot line crop to (160, 36).
        2. Inferences both in a single neural forward pass (batch size 2).
        3. Decodes top line with mask '^[ABEKMHOPCTYX]\d{3}$' (length 4).
        4. Decodes bottom line with mask '^[ABEKMHOPCTYX]{2}\d{2,3}$' (length 4 or 5).
        5. Concatenates: text = top_text + bot_text.
        Returns:
            (plate_text, confidence)
        """
        import cv2
        if top_crop is None or bot_crop is None:
            return "", 0.0

        h_t, w_t = top_crop.shape[:2]
        if (w_t, h_t) != (160, 36):
            top_crop = cv2.resize(top_crop, (160, 36), interpolation=cv2.INTER_LINEAR)

        h_b, w_b = bot_crop.shape[:2]
        if (w_b, h_b) != (160, 36):
            bot_crop = cv2.resize(bot_crop, (160, 36), interpolation=cv2.INTER_LINEAR)

        t_top = self.preprocess(top_crop, apply_clahe=False)
        t_bot = self.preprocess(bot_crop, apply_clahe=False)
        batch_tensor = np.concatenate([t_top, t_bot], axis=0)  # (2, 3, 36, 160)

        if self.use_onnx and self.session is not None:
            input_name = self.session.get_inputs()[0].name
            logits_batch = self.session.run(None, {input_name: batch_tensor})[0]  # (2, 40, num_classes)
        elif self.model is not None:
            t = torch.from_numpy(batch_tensor).to(self.device)
            with torch.no_grad():
                logits_batch = self.model(t).cpu().numpy()
        else:
            raise RuntimeError("OCR model is not loaded!")

        dual_results = FSMBeamSearchDecoder.decode_fsm_type1a_dual(
            logits_batch[0],
            logits_batch[1],
            beam_width=beam_width,
            blank_idx=BLANK_IDX,
        )

        if dual_results:
            best_text, best_score = dual_results[0]
            conf = float(np.clip(np.exp(min(0.0, best_score)), 0.05, 0.99))
            return best_text, round(conf, 4)

        # Fallback to greedy if beam search returned empty
        top_greedy = CTCDecoder.decode_greedy(np.argmax(logits_batch[0], axis=-1))
        bot_greedy = CTCDecoder.decode_greedy(np.argmax(logits_batch[1], axis=-1))
        clean_top = CTCDecoder.apply_gost_heuristics(top_greedy, plate_type="type1")[:4]
        clean_bot = CTCDecoder.apply_gost_heuristics(bot_greedy, plate_type="type1")[:5]
        text = clean_top + clean_bot
        return text, 0.5000
