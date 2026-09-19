#!/usr/bin/env python3
"""
Unit tests for Dual-Line Pass OCR on square Type 1A plates (tests/test_ocr_1a_dual.py).
Tests FSM masks (type1a_top, type1a_bot), decode_fsm_type1a_dual, predict_type1a_dual,
and pipeline ocr_1a_mode options.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import (
    BLANK_IDX,
    CHAR2IDX,
    IDX2CHAR,
    NUM_CLASSES,
    PlateOCR,
)
from src.pipeline.decoder import FSMBeamSearchDecoder, is_valid_gost_plate
from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection


class TestType1ADualPass(unittest.TestCase):
    def setUp(self):
        mpath = "models/ocr_lprnet_best.onnx" if os.path.exists("models/ocr_lprnet_best.onnx") else "models/ocr_lprnet_best.pt"
        self.ocr = PlateOCR(model_path=mpath, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")

    def test_fsm_masks_type1a(self):
        """Tests that FSM table strictly enforces length and letter/digit positions for Type 1A lines."""
        # Top line: 4 chars (pos 0 letter, pos 1..3 digits)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(0, "type1a_top")), 12)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(1, "type1a_top")), 10)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(2, "type1a_top")), 10)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(3, "type1a_top")), 10)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(4, "type1a_top")), 0)

        # Bottom line: 4 or 5 chars (pos 0..1 letters, pos 2..4 digits)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(0, "type1a_bot")), 12)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(1, "type1a_bot")), 12)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(2, "type1a_bot")), 10)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(3, "type1a_bot")), 10)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(4, "type1a_bot")), 10)
        self.assertEqual(len(FSMBeamSearchDecoder.get_allowed_tokens(5, "type1a_bot")), 0)

    def test_decode_fsm_synthetic_logits(self):
        """Tests that synthesized logits for top (A123) and bot (BC77) are correctly decoded."""
        top_logits = np.full((40, NUM_CLASSES), -5.0, dtype=np.float32)
        top_logits[:, BLANK_IDX] = 0.0
        top_logits[5, CHAR2IDX["A"]] = 8.0
        top_logits[12, CHAR2IDX["1"]] = 8.0
        top_logits[20, CHAR2IDX["2"]] = 8.0
        top_logits[28, CHAR2IDX["3"]] = 8.0

        top_cands = FSMBeamSearchDecoder.decode_fsm_beam_search(
            top_logits, plate_type="type1a_top", beam_width=5
        )
        self.assertTrue(len(top_cands) > 0)
        self.assertEqual(top_cands[0][0], "A123")

        bot_logits = np.full((40, NUM_CLASSES), -5.0, dtype=np.float32)
        bot_logits[:, BLANK_IDX] = 0.0
        bot_logits[5, CHAR2IDX["B"]] = 8.0
        bot_logits[12, CHAR2IDX["C"]] = 8.0
        bot_logits[20, CHAR2IDX["7"]] = 8.0
        bot_logits[28, CHAR2IDX["7"]] = 8.0

        bot_cands = FSMBeamSearchDecoder.decode_fsm_beam_search(
            bot_logits, plate_type="type1a_bot", beam_width=5
        )
        self.assertTrue(len(bot_cands) > 0)
        self.assertEqual(bot_cands[0][0], "BC77")

        dual_cands = FSMBeamSearchDecoder.decode_fsm_type1a_dual(top_logits, bot_logits)
        self.assertTrue(len(dual_cands) > 0)
        self.assertEqual(dual_cands[0][0], "A123BC77")
        self.assertTrue(is_valid_gost_plate(dual_cands[0][0], "type1a"))

    def test_predict_type1a_dual_interface(self):
        """Tests predict_type1a_dual with arbitrary crops to ensure no crashes and valid outputs."""
        dummy_top = np.full((36, 160, 3), 220, dtype=np.uint8)
        dummy_bot = np.full((36, 160, 3), 220, dtype=np.uint8)

        text, conf = self.ocr.predict_type1a_dual(dummy_top, dummy_bot)
        self.assertIsInstance(text, str)
        self.assertIsInstance(conf, float)
        self.assertTrue(0.0 <= conf <= 1.0)
        if text:
            self.assertIn(len(text), (8, 9))

    def test_predict_type1a_native_interface(self):
        """Tests predict_type1a_native with canonical 160x96 crop."""
        dummy_crop = np.full((96, 160, 3), 220, dtype=np.uint8)
        text, conf = self.ocr.predict_type1a_native(dummy_crop)
        self.assertIsInstance(text, str)
        self.assertIsInstance(conf, float)
        self.assertTrue(0.0 <= conf <= 1.0)
        if text:
            self.assertIn(len(text), (8, 9))

    def test_pipeline_modes(self):
        """Tests that OmniPlatePipeline initializes and executes with all ocr_1a_mode options."""
        for mode in ("native", "dual", "stitch", "ensemble"):
            p = OmniPlatePipeline(ocr_1a_mode=mode)
            self.assertEqual(p.ocr_1a_mode, mode)
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            dets = p.predict(dummy_img)
            self.assertIsInstance(dets, list)


if __name__ == "__main__":
    unittest.main()

