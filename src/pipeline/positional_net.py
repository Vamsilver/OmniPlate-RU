"""
PositionalPlateNet — Vector 3: Positional Non-CTC Multi-Head Classifier
=======================================================================
Альтернативная парадигма OCR для номерных знаков РФ без CTC Loss и blank-токена.

Архитектура:
  Input:      (B, 3, 36, 160)  — стандартный кроп номера ГОСТ (Type 1, 1B)
  Backbone:   Shared LPRNet-style stem → multi-scale fusion → (B, 448, 1, 9)
  9 Heads:    Независимые nn.Linear на каждую позицию символа:
    head[0]   → 14 классов  (буква серии A,B,E,K,M,H,O,P,C,T,У,Х + pad2 = 16)
    head[1-3] → 10 классов  (цифры 0-9)
    head[4-5] → 14 классов  (буквы серии)
    head[6]   →  9 классов  (первая цифра региона 1-9)
    head[7-8] → 10 классов  (цифры региона 0-9)
  Loss:       sum(CrossEntropyLoss per position)
  Inference:  argmax per head → concat → string

Ключевая гипотеза: устранение CTC blank-collapse ошибок на 2/3-значных регионах.

Формат номера ГОСТ Type 1 (9 символов):
  Позиции: [L] [D] [D] [D] [L] [L] [R] [R] [R?]
  L = буква серии (14 допустимых букв кириллицы)
  D = цифра 0-9
  R = цифра региона (pos 6: 1-9, pos 7-8: 0-9)
  pos 8 может быть пустым для 2-значного региона → класс PAD

Поддерживаемые форматы:
  - Type 1  (9 символов): А123ВС77    [len=9]
  - Type 1B (9 символов): А123ВС777   [len=9]
  - 8-символьный вариант: А123ВС7 → pos8 = PAD
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

# ─── Алфавит ──────────────────────────────────────────────────────────────────
# 12 кириллических букв, используемых в российских номерах (ГОСТ Р 50577-93):
PLATE_LETTERS = list("АВЕКМНОРСТУХ")   # 12 букв
PLATE_DIGITS  = list("0123456789")      # 10 цифр

# Словари кодирования
PAD_IDX = 0
LETTER_VOCAB: List[str] = ["<PAD>"] + PLATE_LETTERS   # 13 классов (0=pad)
DIGIT_VOCAB:  List[str] = ["<PAD>"] + PLATE_DIGITS    # 11 классов (0=pad)
REGION_FIRST_VOCAB: List[str] = ["<PAD>"] + list("123456789")  # 10 классов (0=pad)
REGION_VOCAB: List[str]       = ["<PAD>"] + PLATE_DIGITS        # 11 классов (0=pad)

NUM_LETTER_CLASSES      = len(LETTER_VOCAB)       # 13
NUM_DIGIT_CLASSES       = len(DIGIT_VOCAB)        # 11
NUM_REGION_FIRST_CLASSES = len(REGION_FIRST_VOCAB) # 10
NUM_REGION_CLASSES      = len(REGION_VOCAB)       # 11

# Карта: позиция → (тип, кол-во классов)
#  pos: 0=L, 1=D, 2=D, 3=D, 4=L, 5=L, 6=R1, 7=R, 8=R(opt)
POSITION_SPEC = [
    ("letter",       NUM_LETTER_CLASSES),        # pos 0
    ("digit",        NUM_DIGIT_CLASSES),          # pos 1
    ("digit",        NUM_DIGIT_CLASSES),          # pos 2
    ("digit",        NUM_DIGIT_CLASSES),          # pos 3
    ("letter",       NUM_LETTER_CLASSES),         # pos 4
    ("letter",       NUM_LETTER_CLASSES),         # pos 5
    ("region_first", NUM_REGION_FIRST_CLASSES),   # pos 6
    ("region",       NUM_REGION_CLASSES),         # pos 7
    ("region",       NUM_REGION_CLASSES),         # pos 8 (PAD for 2-digit region)
]

NUM_POSITIONS = len(POSITION_SPEC)  # 9


# ─── Вспомогательные блоки (минимальный backbone) ─────────────────────────────

class SmallBlock(nn.Module):
    """Облегчённый базовый блок (Conv-BN-ReLU × 2, с shortcut)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# ─── PositionalPlateNet ────────────────────────────────────────────────────────

class PositionalPlateNet(nn.Module):
    """
    Positional Non-CTC Multi-Head Classifier for Russian License Plates.

    Backbone: LPRNet-style multi-scale feature extractor → (B, C, 1, 9)
    Heads:    9 × nn.Linear(C → num_classes[pos])

    Args:
        fused_channels: Number of channels after multi-scale fusion (default: 448).
        dropout_rate:   Dropout before each classification head (default: 0.3).
        num_positions:  Number of positional heads (default: 9).
    """

    def __init__(
        self,
        fused_channels: int = 448,
        dropout_rate: float = 0.3,
        num_positions: int = NUM_POSITIONS,
    ):
        super().__init__()
        self.fused_channels = fused_channels
        self.num_positions  = num_positions

        # ── Backbone stem: (B, 3, 36, 160) → (B, 64, 18, 80)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),
        )

        # ── Stage 1-4 (shared)
        self.block1 = SmallBlock(64, 64)
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)   # 9×40
        self.block2 = SmallBlock(64, 128)
        self.block3 = SmallBlock(128, 256)
        self.pool2  = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # 5×40
        self.block4 = SmallBlock(256, 256)

        # ── Multi-scale projectors → (B, *, 1, 9)
        #    Collapse height to 1 and width to exactly 9 (= num_positions)
        self.pool_main  = nn.AdaptiveAvgPool2d((1, num_positions))   # main path (256ch)
        self.pool_stem  = nn.AdaptiveAvgPool2d((1, num_positions))   # stem path   (64ch)
        self.pool_b2    = nn.AdaptiveAvgPool2d((1, num_positions))   # block2 path (128ch)
        # Total fused: 64 + 128 + 256 = 448

        # ── Bottleneck before heads: (B, 448, 1, 9) → (B, 256, 1, 9)
        self.bottleneck = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(fused_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        head_in = 256

        # ── 9 positional heads (nn.Linear each)
        self.heads = nn.ModuleList([
            nn.Linear(head_in, num_cls)
            for (_, num_cls) in POSITION_SPEC
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, 36, 160) — normalized plate crop.
        Returns:
            logits_per_pos: List of 9 tensors, each (B, num_classes[pos]).
        """
        f_stem = self.stem(x)          # (B, 64, 18, 80)
        f_b1   = self.block1(f_stem)   # (B, 64, 18, 80)
        f_p1   = self.pool1(f_b1)      # (B, 64,  9, 40)
        f_b2   = self.block2(f_p1)     # (B, 128,  9, 40)
        f_b3   = self.block3(f_b2)     # (B, 256,  9, 40)
        f_p2   = self.pool2(f_b3)      # (B, 256,  5, 40)
        f_b4   = self.block4(f_p2)     # (B, 256,  5, 40)

        # Multi-scale fusion → (B, 448, 1, 9)
        f_main = self.pool_main(f_b4)           # (B, 256, 1, 9)
        f_s    = self.pool_stem(f_stem)         # (B, 64,  1, 9)
        f_m    = self.pool_b2(f_b2)             # (B, 128, 1, 9)
        fused  = torch.cat([f_s, f_m, f_main], dim=1)  # (B, 448, 1, 9)

        feat = self.bottleneck(fused)  # (B, 256, 1, 9)
        feat = feat.squeeze(2)         # (B, 256, 9)
        feat = feat.permute(0, 2, 1)   # (B, 9, 256)

        # Apply each head to its positional slice
        logits = [self.heads[i](feat[:, i, :]) for i in range(self.num_positions)]
        return logits  # List[9] of (B, num_cls_i)


# ─── Loss ─────────────────────────────────────────────────────────────────────

class PositionalCELoss(nn.Module):
    """
    Sum of CrossEntropyLoss over all 9 positions.
    Supports per-position weighting and padding ignore.

    Args:
        position_weights: Optional tensor of shape (9,) to weight each position's loss.
        ignore_index:     Class index to ignore (PAD_IDX by default).
        reduction:        'mean' (mean over batch) or 'sum'.
    """

    def __init__(
        self,
        position_weights: Optional[torch.Tensor] = None,
        ignore_index: int = PAD_IDX,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index      = ignore_index
        self.reduction         = reduction
        if position_weights is not None:
            self.register_buffer("position_weights", position_weights)
        else:
            self.position_weights = None

    def forward(
        self,
        logits_per_pos: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_per_pos: List of 9 tensors each (B, num_cls_i).
            targets:        (B, 9) integer targets. PAD_IDX is ignored.
        Returns:
            Scalar loss.
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        n = len(logits_per_pos)
        for i, logits_i in enumerate(logits_per_pos):
            w = self.position_weights[i].item() if self.position_weights is not None else 1.0
            loss_i = F.cross_entropy(
                logits_i,
                targets[:, i],
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
            total_loss = total_loss + w * loss_i
        return total_loss / n


# ─── Decode ───────────────────────────────────────────────────────────────────

def decode_positional(logits_per_pos: List[torch.Tensor]) -> List[str]:
    """
    Greedy decode: argmax at each position → character string.

    Returns:
        List of decoded plate strings, one per batch element.
        PAD positions are skipped (empty string contribution).
    """
    preds = []
    for i, logits_i in enumerate(logits_per_pos):
        idx = logits_i.argmax(dim=-1)  # (B,)
        preds.append(idx)
    preds = torch.stack(preds, dim=1)  # (B, 9)

    batch_strs = []
    for b in range(preds.size(0)):
        chars = []
        for pos in range(preds.size(1)):
            cls_idx = preds[b, pos].item()
            p_type, _ = POSITION_SPEC[pos]
            if p_type == "letter":
                vocab = LETTER_VOCAB
            elif p_type == "digit":
                vocab = DIGIT_VOCAB
            elif p_type == "region_first":
                vocab = REGION_FIRST_VOCAB
            else:
                vocab = REGION_VOCAB
            ch = vocab[cls_idx] if cls_idx < len(vocab) else "<UNK>"
            if ch != "<PAD>":
                chars.append(ch)
        batch_strs.append("".join(chars))
    return batch_strs


def decode_positional_constrained(logits_per_pos: List[torch.Tensor]) -> List[str]:
    """
    ГОСТ-constrained greedy decode: маскирует недопустимые классы per-position.

    Применяет структурный приор ГОСТ Р 50577-93 как жёсткие ограничения:
      pos 0, 4, 5 (letter):       argmax среди [1..12] (12 кириллических букв, без PAD)
      pos 1, 2, 3 (digit):        argmax среди [1..10] (цифры 0-9, без PAD)
      pos 6       (region_first): argmax среди [1..9]  (только 1-9, без 0 и PAD)
      pos 7       (region):       argmax среди [1..10] (цифры 0-9, без PAD)
      pos 8       (region, opt):  unconstrained (PAD → 2-значный регион)

    Returns:
        List of decoded plate strings, one per batch element.
    """
    pred_indices = []
    for i, logits_i in enumerate(logits_per_pos):
        p_type, _ = POSITION_SPEC[i]

        if i == 8:
            # pos 8: PAD допустим (2-значный регион)
            idx = logits_i.argmax(dim=-1)
        else:
            # Все позиции 0-7: маскируем PAD (index 0)
            masked = logits_i.clone()
            masked[:, 0] = float("-inf")
            idx = masked.argmax(dim=-1)

        pred_indices.append(idx)

    pred_indices = torch.stack(pred_indices, dim=1)  # (B, 9)

    batch_strs = []
    for b in range(pred_indices.size(0)):
        chars = []
        for pos in range(9):
            cls_idx = pred_indices[b, pos].item()
            p_type, _ = POSITION_SPEC[pos]
            if p_type == "letter":
                vocab = LETTER_VOCAB
            elif p_type == "digit":
                vocab = DIGIT_VOCAB
            elif p_type == "region_first":
                vocab = REGION_FIRST_VOCAB
            else:
                vocab = REGION_VOCAB
            ch = vocab[cls_idx] if cls_idx < len(vocab) else "<UNK>"
            if ch != "<PAD>":
                chars.append(ch)
        batch_strs.append("".join(chars))
    return batch_strs




# ─── ONNX Export ──────────────────────────────────────────────────────────────

def export_positional_onnx(
    model: PositionalPlateNet,
    save_path: str,
    opset_version: int = 17,
    batch_size: int = 1,
) -> str:
    """
    Export PositionalPlateNet to ONNX.

    The ONNX model outputs 9 tensors (one per position).
    Dynamic batch dimension is supported.

    Args:
        model:          Trained PositionalPlateNet (eval mode recommended).
        save_path:      Output .onnx file path.
        opset_version:  ONNX opset (default: 17).
        batch_size:     Dummy batch size for tracing (default: 1).

    Returns:
        Absolute path to saved ONNX file.
    """
    model.eval()
    dummy = torch.zeros(batch_size, 3, 36, 160)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            save_path,
            input_names=["input"],
            output_names=[f"logits_pos{i}" for i in range(model.num_positions)],
            dynamic_axes={"input": {0: "batch"}, **{f"logits_pos{i}": {0: "batch"} for i in range(model.num_positions)}},
            opset_version=opset_version,
        )
    return os.path.abspath(save_path)


# ─── Target Encoding Helper ───────────────────────────────────────────────────

def encode_plate_positional(plate_str: str) -> Optional[torch.Tensor]:
    """
    Encode a Russian plate string to a (9,) integer target tensor.

    Format accepted:
      - Type 1  (9-char):  А123ВС777
      - Type 1B (9-char):  А123ВС 77 (same)
      - 8-char:            А123ВС77  → pos8 = PAD

    Returns:
        (9,) tensor of class indices, or None if plate cannot be parsed.
    """
    # Normalise: strip spaces, uppercase
    s = plate_str.strip().upper()
    # Replace Latin look-alikes with Cyrillic
    transliterate = {"A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
                     "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
                     "Y": "У", "X": "Х"}
    s = "".join(transliterate.get(c, c) for c in s)

    if len(s) not in (8, 9):
        return None

    targets = []

    def encode_letter(ch: str) -> Optional[int]:
        return LETTER_VOCAB.index(ch) if ch in LETTER_VOCAB else None

    def encode_digit(ch: str) -> Optional[int]:
        return DIGIT_VOCAB.index(ch) if ch in DIGIT_VOCAB else None

    def encode_region_first(ch: str) -> Optional[int]:
        return REGION_FIRST_VOCAB.index(ch) if ch in REGION_FIRST_VOCAB else None

    def encode_region(ch: str) -> Optional[int]:
        return REGION_VOCAB.index(ch) if ch in REGION_VOCAB else None

    encoders = [
        encode_letter,         # pos 0
        encode_digit,          # pos 1
        encode_digit,          # pos 2
        encode_digit,          # pos 3
        encode_letter,         # pos 4
        encode_letter,         # pos 5
        encode_region_first,   # pos 6
        encode_region,         # pos 7
    ]

    for i, enc_fn in enumerate(encoders):
        idx = enc_fn(s[i])
        if idx is None:
            return None
        targets.append(idx)

    # pos 8 (optional 3rd region digit)
    if len(s) == 9:
        idx = encode_region(s[8])
        if idx is None:
            return None
        targets.append(idx)
    else:
        targets.append(PAD_IDX)  # 2-digit region → PAD

    return torch.tensor(targets, dtype=torch.long)


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== PositionalPlateNet Sanity Check ===")
    model = PositionalPlateNet()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    dummy = torch.randn(4, 3, 36, 160)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(dummy)
    dt = (time.perf_counter() - t0) * 1000
    print(f"Forward pass (B=4): {dt:.2f} ms")
    print(f"Output shapes: {[l.shape for l in logits]}")

    decoded = decode_positional(logits)
    print(f"Decoded (random weights): {decoded}")

    # Test target encoding
    test_plates = ["А123ВС777", "В456МН78", "К789ТУ1234"]
    for p in test_plates:
        t = encode_plate_positional(p)
        print(f"  '{p}' → {t if t is not None else 'INVALID'}")

    # ONNX export
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    export_positional_onnx(model, onnx_path)
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"ONNX exported: {onnx_path} ({size_kb:.1f} KB)")
    print("=== DONE ===")
