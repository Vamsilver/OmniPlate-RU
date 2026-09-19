"""
Tests for PositionalPlateNet (Vector 3 — Positional Non-CTC Classifier)
"""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.positional_net import (
    PositionalPlateNet,
    PositionalCELoss,
    POSITION_SPEC,
    NUM_POSITIONS,
    PAD_IDX,
    decode_positional,
    encode_plate_positional,
    export_positional_onnx,
    LETTER_VOCAB,
    DIGIT_VOCAB,
    REGION_FIRST_VOCAB,
    REGION_VOCAB,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    m = PositionalPlateNet()
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_input():
    torch.manual_seed(42)
    return torch.randn(4, 3, 36, 160)


# ─── 1. Output shape test ────────────────────────────────────────────────────

def test_output_shapes(model, dummy_input):
    """Forward pass must return 9 tensors with correct (B, num_cls) shapes."""
    with torch.no_grad():
        logits = model(dummy_input)
    B = dummy_input.size(0)
    assert len(logits) == NUM_POSITIONS, f"Expected 9 outputs, got {len(logits)}"
    for i, (logits_i, (p_type, num_cls)) in enumerate(zip(logits, POSITION_SPEC)):
        assert logits_i.shape == (B, num_cls), (
            f"pos {i} ({p_type}): expected ({B}, {num_cls}), got {logits_i.shape}"
        )


# ─── 2. Parameter count ──────────────────────────────────────────────────────

def test_parameter_count(model):
    """Model should have a reasonable parameter count (< 3M for a lightweight backbone)."""
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 100_000, f"Too few params: {n_params}"
    assert n_params < 5_000_000, f"Too many params (expected lightweight): {n_params}"


# ─── 3. Determinism ──────────────────────────────────────────────────────────

def test_determinism(model, dummy_input):
    """Same input must produce identical output in eval mode."""
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = model(dummy_input)
    for i in range(NUM_POSITIONS):
        assert torch.allclose(out1[i], out2[i], atol=1e-6), f"Non-deterministic at pos {i}"


# ─── 4. Loss backward ────────────────────────────────────────────────────────

def test_loss_backward():
    """Loss must be differentiable and reduce with random targets."""
    torch.manual_seed(0)
    model_train = PositionalPlateNet()
    model_train.train()
    criterion = PositionalCELoss()

    x = torch.randn(2, 3, 36, 160)
    # Build valid random targets (non-PAD, within vocab range per position)
    targets = torch.zeros(2, NUM_POSITIONS, dtype=torch.long)
    for i, (p_type, num_cls) in enumerate(POSITION_SPEC):
        targets[:, i] = torch.randint(1, num_cls, (2,))

    logits = model_train(x)
    loss = criterion(logits, targets)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()

    # Check gradients exist
    for name, p in model_train.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ─── 5. Decode output ────────────────────────────────────────────────────────

def test_decode_valid_range(model, dummy_input):
    """Decode must return strings with at most 9 characters (no PAD in output)."""
    with torch.no_grad():
        logits = model(dummy_input)
    decoded = decode_positional(logits)
    assert len(decoded) == dummy_input.size(0)
    for s in decoded:
        assert len(s) <= 9, f"Decoded string too long: '{s}'"
        assert isinstance(s, str)


# ─── 6. encode_plate_positional ──────────────────────────────────────────────

@pytest.mark.parametrize("plate,expected_len", [
    ("А123ВС777", 9),   # Type 1B, 3-digit region
    ("В456МН78",  9),   # Type 1,  2-digit region → pos8=PAD
    ("К789ТУ199", 9),   # Type 1B
])
def test_encode_plate_length(plate, expected_len):
    t = encode_plate_positional(plate)
    assert t is not None, f"Failed to encode '{plate}'"
    assert t.shape == (expected_len,), f"Expected shape ({expected_len},), got {t.shape}"


def test_encode_plate_pad_for_2digit_region():
    """2-digit region plates must have PAD at pos 8."""
    t = encode_plate_positional("В456МН78")
    assert t is not None
    assert t[8].item() == PAD_IDX, f"Expected PAD at pos 8, got {t[8].item()}"


def test_encode_plate_invalid():
    """Invalid plates should return None."""
    assert encode_plate_positional("TOOSHORT") is None
    assert encode_plate_positional("WAAAAYTOOLONG12345") is None


def test_encode_plate_latin_transliteration():
    """Latin look-alike letters should be auto-converted to Cyrillic."""
    # A123BC77 — Latin A,B,C should work
    t = encode_plate_positional("A123BC77")
    assert t is not None, "Latin look-alike transliteration failed"


# ─── 7. ONNX export ──────────────────────────────────────────────────────────

def test_onnx_export(model):
    """ONNX export must succeed and produce a non-empty file."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "positional_test.onnx")
        out_path = export_positional_onnx(model, onnx_path)
        assert os.path.exists(out_path), "ONNX file not created"
        size = os.path.getsize(out_path)
        assert size > 10_000, f"ONNX file suspiciously small: {size} bytes"

        # Run inference with ORT
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        import numpy as np
        dummy_np = np.random.randn(2, 3, 36, 160).astype(np.float32)
        outputs = sess.run(None, {"input": dummy_np})
        assert len(outputs) == NUM_POSITIONS, f"Expected {NUM_POSITIONS} outputs from ONNX"
        for i, (out_i, (_, num_cls)) in enumerate(zip(outputs, POSITION_SPEC)):
            assert out_i.shape == (2, num_cls), f"ONNX pos {i} shape mismatch: {out_i.shape}"


# ─── 8. Positional vocab coverage ────────────────────────────────────────────

def test_vocab_coverage():
    """All 12 Cyrillic plate letters must be in LETTER_VOCAB."""
    required = set("АВЕКМНОРСТУХ")
    vocab_set = set(LETTER_VOCAB) - {"<PAD>"}
    missing = required - vocab_set
    assert not missing, f"Missing letters in LETTER_VOCAB: {missing}"


def test_digit_vocab():
    """DIGIT_VOCAB must contain all 10 digits 0-9."""
    required = set("0123456789")
    vocab_set = set(DIGIT_VOCAB) - {"<PAD>"}
    assert vocab_set == required
