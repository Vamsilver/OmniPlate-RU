#!/usr/bin/env python3
"""
Unit tests for OCR module (src/pipeline/ocr.py).
Tests CTCDecoder, vocabulary mappings, GOST heuristic repair, and preprocessing.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import (
    BLANK_IDX,
    BLANK_TOKEN,
    CHAR2IDX,
    IDX2CHAR,
    NUM_CLASSES,
    VOCAB,
    CTCDecoder,
    PlateOCR,
)


class TestPlateOCR(unittest.TestCase):
    def setUp(self):
        self.decoder = CTCDecoder()

    def test_vocab_integrity(self):
        self.assertEqual(VOCAB[0], BLANK_TOKEN)
        self.assertEqual(BLANK_IDX, 0)
        self.assertEqual(len(VOCAB), NUM_CLASSES)
        for i, ch in enumerate(VOCAB):
            self.assertEqual(CHAR2IDX[ch], i)
            self.assertEqual(IDX2CHAR[i], ch)

    def test_ctc_greedy_decoder(self):
        # Target: "A123BC77"
        # Simulate index sequence with repeats and blanks:
        # [blank, 'A', 'A', blank, '1', blank, '2', '2', '3', blank, 'B', 'C', blank, '7', '7']
        indices = [
            BLANK_IDX,
            CHAR2IDX["A"], CHAR2IDX["A"],
            BLANK_IDX,
            CHAR2IDX["1"],
            BLANK_IDX,
            CHAR2IDX["2"], CHAR2IDX["2"],
            CHAR2IDX["3"],
            BLANK_IDX,
            CHAR2IDX["B"],
            CHAR2IDX["C"],
            BLANK_IDX,
            CHAR2IDX["7"],
            BLANK_IDX,
            CHAR2IDX["7"],
        ]
        decoded = self.decoder.decode_greedy(indices)
        self.assertEqual(decoded, "A123BC77")

    def test_gost_heuristics_letter_fixes(self):
        # 0 (zero) in pos 0 should become 'O'
        raw = "0123BC77"
        repaired = self.decoder.apply_gost_heuristics(raw)
        self.assertEqual(repaired, "O123BC77")

        # 8 in pos 4 should become 'B'
        raw2 = "A1238C77"
        repaired2 = self.decoder.apply_gost_heuristics(raw2)
        self.assertEqual(repaired2, "A123BC77")

    def test_gost_heuristics_digit_fixes(self):
        # 'O' in pos 1 should become '0'
        raw = "AO23BC77"
        repaired = self.decoder.apply_gost_heuristics(raw)
        self.assertEqual(repaired, "A023BC77")

        # 'B' in pos 3 should become '8'
        raw2 = "A12BBC77"
        repaired2 = self.decoder.apply_gost_heuristics(raw2)
        self.assertEqual(repaired2, "A128BC77")

        # 'B' in region code pos 6 should become '8'
        raw3 = "A123BCB7"
        repaired3 = self.decoder.apply_gost_heuristics(raw3)
        self.assertEqual(repaired3, "A123BC87")

    def test_preprocess_tensor_shape(self):
        ocr = PlateOCR()
        dummy_crop = np.zeros((36, 160, 3), dtype=np.uint8)
        tensor = ocr.preprocess(dummy_crop)
        self.assertEqual(tensor.shape, (1, 3, 36, 160))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(0.0 <= tensor.min() <= tensor.max() <= 1.0)


if __name__ == "__main__":
    unittest.main()
