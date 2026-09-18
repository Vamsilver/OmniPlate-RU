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
from src.pipeline.decoder import is_valid_region, is_valid_gost_plate


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

    def test_gost_heuristics_type1b_bus(self):
        # Classic 7-char Russian bus plate: LL DDD DD
        # Bolt artifact 'H' at end of 7-char plate should be truncated
        raw = "BX18750H"
        repaired = self.decoder.apply_gost_heuristics(raw, plate_type="type1b")
        self.assertEqual(repaired, "BX18750")

        # Digit confusion in pos 0 should become letter
        raw2 = "0X18750"
        repaired2 = self.decoder.apply_gost_heuristics(raw2, plate_type="type1b")
        self.assertEqual(repaired2, "OX18750")

        # Letter confusion in region should become digit
        raw3 = "AH8897B"
        repaired3 = self.decoder.apply_gost_heuristics(raw3, plate_type="type1b")
        self.assertEqual(repaired3, "AH88978")

    def test_gost_heuristics_type2_trailer(self):
        # Type 2 GOST trailer format: LL DDDD RR (8 chars) or LL DDDD RRR (9 chars)
        raw = "AE123477"
        repaired = self.decoder.apply_gost_heuristics(raw, plate_type="type2")
        self.assertEqual(repaired, "AE123477")

        # 0 in pos 0 should become letter 'O'
        raw2 = "0E123477"
        repaired2 = self.decoder.apply_gost_heuristics(raw2, plate_type="type2")
        self.assertEqual(repaired2, "OE123477")

        # Letter 'B' in digit pos 5 should become digit '8'
        raw3 = "AE123B77"
        repaired3 = self.decoder.apply_gost_heuristics(raw3, plate_type="type2")
        self.assertEqual(repaired3, "AE123877")

        # Trailing border noise character should be trimmed
        raw4 = "AE123477T"
        repaired4 = self.decoder.apply_gost_heuristics(raw4, plate_type="type2")
        self.assertEqual(repaired4, "AE123477")

    def test_ctc_hypothesis_scoring_trailer(self):
        # Simulate logits for trailer AE123477
        target = "AE123477"
        positions = [2, 6, 11, 16, 21, 26, 31, 36]
        logits = np.random.randn(40, NUM_CLASSES) * 0.1
        logits[:, BLANK_IDX] = 4.0
        for pos, ch in zip(positions, target):
            logits[pos, BLANK_IDX] = -2.0
            logits[pos, CHAR2IDX[ch]] = 8.0
            logits[pos + 1, BLANK_IDX] = -2.0
            logits[pos + 1, CHAR2IDX[ch]] = 5.0

        best_text, best_type, conf = self.decoder.score_hypotheses(logits, plate_type_prior="type1")
        self.assertEqual(best_text, "AE123477")
        self.assertEqual(best_type, "type2")
        self.assertGreater(conf, 0.5)

    def test_ctc_hypothesis_scoring_car_type1(self):
        # Simulate logits for standard car A123BC77
        target = "A123BC77"
        positions = [2, 6, 11, 16, 21, 26, 31, 36]
        np.random.seed(42)
        logits = np.random.randn(40, NUM_CLASSES) * 0.1
        logits[:, BLANK_IDX] = 4.0
        for pos, ch in zip(positions, target):
            logits[pos, BLANK_IDX] = -2.0
            logits[pos, CHAR2IDX[ch]] = 8.0
            logits[pos + 1, BLANK_IDX] = -2.0
            logits[pos + 1, CHAR2IDX[ch]] = 5.0

        best_text, best_type, conf = self.decoder.score_hypotheses(logits, plate_type_prior="type1")
        self.assertEqual(best_text, "A123BC77")
        self.assertEqual(best_type, "type1")
        self.assertGreater(conf, 0.5)

    def test_preprocess_tensor_shape(self):
        ocr = PlateOCR()
        dummy_crop = np.zeros((36, 160, 3), dtype=np.uint8)
        tensor = ocr.preprocess(dummy_crop)
        self.assertEqual(tensor.shape, (1, 3, 36, 160))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(0.0 <= tensor.min() <= tensor.max() <= 1.0)

    def test_strip_parasitic_edge(self):
        # Parasitic 'H' before Type 1 M235PX77
        self.assertEqual(self.decoder.strip_parasitic_edge("HM235PX77"), "M235PX77")
        self.assertEqual(self.decoder.apply_gost_heuristics("HM235PX77", plate_type="type1"), "M235PX77")
        # Ensure genuine Type 1 plates starting with letters are not corrupted
        self.assertEqual(self.decoder.strip_parasitic_edge("B045KE152"), "B045KE152")
        self.assertEqual(self.decoder.strip_parasitic_edge("H074KP799"), "H074KP799")
        self.assertEqual(self.decoder.strip_parasitic_edge("H937AP155"), "H937AP155")
        # Ensure region 155 is valid and preserved
        self.assertEqual(self.decoder.repair_3digit_region("155"), "155")

    def test_noisy_crop_prior_preserves_type1(self):
        # Case real_type1a_0205.jpg: noisy digits '#9#447#' should produce Type 1 format with digit run 447
        t1_cand = self.decoder.apply_gost_heuristics("#9#447#", plate_type="type1")
        self.assertEqual(t1_cand, "#447####")

        # Simulate noisy logits with wildcards
        logits = np.random.randn(40, NUM_CLASSES) * 0.1
        logits[:, BLANK_IDX] = 1.0
        # When confidence is low (<0.35) or wildcards >= 3, score_hypotheses must prioritize Type 1
        best_text, best_type, conf = self.decoder.score_hypotheses(logits, plate_type_prior="type1")
        self.assertEqual(best_type, "type1")
        self.assertLess(conf, 0.35)

    def test_decode_beam_search(self):
        # Construct clear sequence logits for target text 'M235PX77'
        target = "M235PX77"
        logits = np.full((40, NUM_CLASSES), -5.0)
        logits[:, BLANK_IDX] = 2.0  # Dominant blank

        positions = [2, 6, 10, 14, 18, 22, 26, 30]
        for pos, ch in zip(positions, target):
            logits[pos, BLANK_IDX] = -2.0
            logits[pos, CHAR2IDX[ch]] = 8.0
            logits[pos + 1, BLANK_IDX] = -2.0
            logits[pos + 1, CHAR2IDX[ch]] = 6.0

        max_l = np.max(logits, axis=-1, keepdims=True)
        log_probs = logits - max_l - np.log(np.sum(np.exp(logits - max_l), axis=-1, keepdims=True))

        beam_results = self.decoder.decode_beam_search(log_probs, beam_width=5)
        self.assertGreater(len(beam_results), 0)
        top_text, top_score = beam_results[0]
        self.assertEqual(top_text, target)

    def test_adaptive_clahe(self):
        dark_crop = np.full((36, 160, 3), 40, dtype=np.uint8)
        enh = PlateOCR.apply_adaptive_clahe(dark_crop)
        self.assertEqual(enh.shape, (36, 160, 3))
        self.assertGreater(float(np.mean(enh)), float(np.mean(dark_crop)))

    def test_predict_batch_empty(self):
        ocr = PlateOCR()
        res = ocr.predict_batch([])
        self.assertEqual(res, [])

    def test_is_valid_region(self):
        self.assertTrue(is_valid_region("77"))
        self.assertTrue(is_valid_region("177"))
        self.assertTrue(is_valid_region("777"))
        self.assertTrue(is_valid_region("799"))
        self.assertTrue(is_valid_region("16"))
        self.assertTrue(is_valid_region("116"))
        self.assertTrue(is_valid_region("550"))
        # Invalid regions
        self.assertFalse(is_valid_region("00"))
        self.assertFalse(is_valid_region("999"))
        self.assertFalse(is_valid_region("456"))
        self.assertFalse(is_valid_region("abc"))
        self.assertFalse(is_valid_region(""))

    def test_is_valid_gost_plate(self):
        self.assertTrue(is_valid_gost_plate("A123BC77", "type1"))
        self.assertTrue(is_valid_gost_plate("M235PX777", "type1"))
        self.assertTrue(is_valid_gost_plate("M235PX777", "type1a"))
        self.assertTrue(is_valid_gost_plate("BX18750", "type1b"))
        self.assertTrue(is_valid_gost_plate("AE123477", "type2"))
        # Invalid plates without allow_wildcards
        self.assertFalse(is_valid_gost_plate("A123BC00", "type1"))  # Invalid region 00
        self.assertFalse(is_valid_gost_plate("A123BC999", "type1"))  # Invalid region 999
        self.assertFalse(is_valid_gost_plate("A123##77", "type1"))  # Contains #
        self.assertFalse(is_valid_gost_plate("########", "type1"))  # Contains #
        self.assertFalse(is_valid_gost_plate("INVALID", "type1"))
        self.assertFalse(is_valid_gost_plate("", "type1"))

        # Valid plates with allow_wildcards=True
        self.assertTrue(is_valid_gost_plate("A123##77", "type1", allow_wildcards=True))
        self.assertTrue(is_valid_gost_plate("########", "type1", allow_wildcards=True))
        self.assertTrue(is_valid_gost_plate("#########", "type1", allow_wildcards=True))
        self.assertTrue(is_valid_gost_plate("#246#C73", "type1", allow_wildcards=True))
        self.assertTrue(is_valid_gost_plate("########", "type1a", allow_wildcards=True))
        self.assertTrue(is_valid_gost_plate("#######", "type1b", allow_wildcards=True))
        self.assertTrue(is_valid_gost_plate("########", "type2", allow_wildcards=True))

        # Invalid plates even with allow_wildcards=True
        self.assertFalse(is_valid_gost_plate("####", "type1", allow_wildcards=True))  # Too short
        self.assertFalse(is_valid_gost_plate("##########", "type1", allow_wildcards=True))  # Too long
        self.assertFalse(is_valid_gost_plate("A123BC00", "type1", allow_wildcards=True))  # Invalid region 00
        self.assertFalse(is_valid_gost_plate("A123BC888", "type1", allow_wildcards=True))  # Invalid 3-digit region 888


if __name__ == "__main__":
    unittest.main()


