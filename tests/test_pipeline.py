#!/usr/bin/env python3
"""
Unit tests for OmniPlatePipeline and PlateDetection (src/pipeline/pipeline.py).
Tests quad extraction, bbox fallback, 1A split/stitch integration, and mock pipeline runs.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection
from src.pipeline.ocr import CTCDecoder, PlateOCR


class DummyOCR:
    """Mock OCR for deterministic pipeline integration testing."""
    def __init__(self):
        self.session = "mock_session"
        self.model = None
        self.decoder = CTCDecoder()

    def predict_single(self, crop: np.ndarray, *args, **kwargs):
        # Assert canonical crop size (36, 160, 3)
        assert crop.shape == (36, 160, 3), f"Expected (36, 160, 3), got {crop.shape}"
        p_type = kwargs.get("plate_type")
        if p_type == "type1b" or (args and args[0] == "type1b"):
            return "AA12377", 0.95
        return "A123BC77", 0.95

    def predict_batch(self, crops, plate_types=None, *args, **kwargs):
        res = []
        p_types = list(plate_types) if plate_types is not None else ["type1"] * len(crops)
        for crop, pt in zip(crops, p_types):
            r = self.predict_single(crop, plate_type=pt, *args, **kwargs)
            res.append((r[0], r[1], pt))
        return res



class TestOmniPlatePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = OmniPlatePipeline(
            detector_path="",
            ocr_path="",
            device="cpu",
        )
        self.pipeline.ocr = DummyOCR()


    def test_plate_detection_dataclass(self):
        det = PlateDetection(
            bbox=(100, 200, 150, 40),
            quad=[100.0, 200.0, 250.0, 200.0, 250.0, 240.0, 100.0, 240.0],
            plate_type="type1",
            confidence=0.92,
            text="A123BC77",
            ocr_confidence=0.95,
        )
        d = det.to_dict()
        self.assertEqual(d["bbox"], "100,200,150,40")
        self.assertEqual(d["plate_type"], "type1")
        self.assertEqual(d["plate_num"], "A123BC77")
        self.assertEqual(d["confidence"], 0.92)
        self.assertEqual(d["ocr_confidence"], 0.95)

    def test_fallback_quad_from_bbox(self):
        bbox = (50, 100, 200, 60)
        quad = self.pipeline._fallback_quad_from_bbox(bbox)
        expected = [50.0, 100.0, 250.0, 100.0, 250.0, 160.0, 50.0, 160.0]
        self.assertEqual(quad, expected)

    def test_extract_quad_with_valid_keypoints(self):
        kpts = np.array([
            [105.0, 205.0],
            [245.0, 200.0],
            [250.0, 238.0],
            [100.0, 240.0],
        ], dtype=np.float32)
        bbox = (100, 200, 150, 40)
        quad = self.pipeline._extract_quad(kpts, bbox, img_w=1920, img_h=1080)
        self.assertEqual(len(quad), 8)
        # Verify coordinates are preserved
        self.assertAlmostEqual(quad[0], 105.0, delta=1.0)
        self.assertAlmostEqual(quad[1], 205.0, delta=1.0)

    def test_extract_quad_fallback_on_zeros(self):
        kpts = np.zeros((4, 2), dtype=np.float32)
        bbox = (100, 200, 150, 40)
        quad = self.pipeline._extract_quad(kpts, bbox, img_w=1920, img_h=1080)
        expected = [100.0, 200.0, 250.0, 200.0, 250.0, 240.0, 100.0, 240.0]
        self.assertEqual(quad, expected)

    def test_recognize_single_type1(self):
        # Synthetic scene image
        scene = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        det = PlateDetection(
            bbox=(300, 200, 160, 36),
            quad=[300.0, 200.0, 460.0, 200.0, 460.0, 236.0, 300.0, 236.0],
            plate_type="type1",
            confidence=0.88,
        )
        processed = self.pipeline.recognize_single(scene, det)
        self.assertEqual(processed.text, "A123BC77")
        self.assertEqual(processed.ocr_confidence, 0.95)
        self.assertIsNotNone(processed.rectified_crop)
        self.assertEqual(processed.rectified_crop.shape, (36, 160, 3))

    def test_recognize_single_type1a_split_stitch(self):
        # Type 1A canonical is 160x96
        scene = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        det = PlateDetection(
            bbox=(300, 200, 160, 96),
            quad=[300.0, 200.0, 460.0, 200.0, 460.0, 296.0, 300.0, 296.0],
            plate_type="type1a",
            confidence=0.89,
        )
        processed = self.pipeline.recognize_single(scene, det)
        self.assertEqual(processed.text, "A123BC77")
        self.assertEqual(processed.ocr_confidence, 0.95)
        self.assertIsNotNone(processed.rectified_crop)
        # Rectified crop for 1a should be (96, 160, 3)
        self.assertEqual(processed.rectified_crop.shape, (96, 160, 3))

    def test_recognize_single_other_no_false_positive(self):
        # Category 'other' must NOT be read as GOST
        scene = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        det = PlateDetection(
            bbox=(300, 200, 160, 36),
            quad=[300.0, 200.0, 460.0, 200.0, 460.0, 236.0, 300.0, 236.0],
            plate_type="other",
            confidence=0.95,
        )
        processed = self.pipeline.recognize_single(scene, det)
        self.assertEqual(processed.text, "")
        self.assertEqual(processed.ocr_confidence, 0.0)

    def test_is_gost_yellow_plate_positive(self):
        # Pantone 116C yellow #FFCC00 -> BGR (0, 204, 255)
        yellow_crop = np.zeros((36, 160, 3), dtype=np.uint8)
        yellow_crop[:, :] = (0, 204, 255)
        is_yellow, bg_s, y_ratio = self.pipeline.is_gost_yellow_plate(yellow_crop, s_threshold=60.0)
        self.assertTrue(is_yellow)
        self.assertGreaterEqual(bg_s, 150.0)
        self.assertGreaterEqual(y_ratio, 0.90)

    def test_is_gost_yellow_plate_negative(self):
        # White/gray plate crop -> BGR (230, 230, 230)
        white_crop = np.zeros((36, 160, 3), dtype=np.uint8)
        white_crop[:, :] = (230, 230, 230)
        is_yellow, bg_s, y_ratio = self.pipeline.is_gost_yellow_plate(white_crop, s_threshold=60.0)
        self.assertFalse(is_yellow)
        self.assertLess(bg_s, 60.0)

    def test_recognize_single_type1b_color_demotion(self):
        # White plate wrongly classified as type1b by detector
        scene = np.ones((720, 1280, 3), dtype=np.uint8) * 230
        det = PlateDetection(
            bbox=(300, 200, 160, 36),
            quad=[300.0, 200.0, 460.0, 200.0, 460.0, 236.0, 300.0, 236.0],
            plate_type="type1b",
            confidence=0.85,
        )
        processed = self.pipeline.recognize_single(scene, det)
        # Should be demoted to type1 due to white background (S < 60)
        self.assertEqual(processed.plate_type, "type1")
        self.assertEqual(processed.text, "A123BC77")

    def test_recognize_single_type1b_genuine_yellow(self):
        # Genuine yellow plate: Pantone 116C
        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene[:, :] = (0, 204, 255)
        det = PlateDetection(
            bbox=(300, 200, 160, 36),
            quad=[300.0, 200.0, 460.0, 200.0, 460.0, 236.0, 300.0, 236.0],
            plate_type="type1b",
            confidence=0.92,
        )
        processed = self.pipeline.recognize_single(scene, det)
        # Should retain type1b due to saturated yellow background
        self.assertEqual(processed.plate_type, "type1b")

    def test_recognize_batch(self):
        scene = np.full((480, 640, 3), 200, dtype=np.uint8)
        det1 = PlateDetection(
            bbox=(100, 100, 160, 36),
            quad=[100.0, 100.0, 260.0, 100.0, 260.0, 136.0, 100.0, 136.0],
            plate_type="type1",
            confidence=0.90,
        )
        det2 = PlateDetection(
            bbox=(300, 100, 160, 36),
            quad=[300.0, 100.0, 460.0, 100.0, 460.0, 136.0, 300.0, 136.0],
            plate_type="other",
            confidence=0.85,
        )
        results = self.pipeline.recognize_batch(scene, [det1, det2])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].plate_type, "other")
        self.assertEqual(results[1].text, "")

    def test_recognize_batch_verifier_annulment(self):
        # When PlateVerifier marks crop as non-plate or OCR has '#', detection is annulled to other
        class MockVerifier:
            def is_valid(self): return True
            def verify_batch(self, crops):
                # First crop is non-plate (p_score=0.10), second is valid plate (p_score=0.95)
                return [(False, 0.10), (True, 0.95)]

        self.pipeline.verifier = MockVerifier()
        scene = np.full((480, 640, 3), 200, dtype=np.uint8)
        det1 = PlateDetection(
            bbox=(100, 100, 160, 36),
            quad=[100.0, 100.0, 260.0, 100.0, 260.0, 136.0, 100.0, 136.0],
            plate_type="type1",
            confidence=0.90,
        )
        det2 = PlateDetection(
            bbox=(300, 100, 160, 36),
            quad=[300.0, 100.0, 460.0, 100.0, 460.0, 136.0, 300.0, 136.0],
            plate_type="type1",
            confidence=0.90,
        )
        results = self.pipeline.recognize_batch(scene, [det1, det2])
        # det1 should be annulled to other with text=""
        self.assertEqual(results[0].plate_type, "other")
        self.assertEqual(results[0].text, "")
        self.assertEqual(results[0].ocr_confidence, 0.0)
        # det2 should be accepted as type1 with text="A123BC77"
        self.assertEqual(results[1].plate_type, "type1")
        self.assertEqual(results[1].text, "A123BC77")

    def test_wildcard_mask_retention_when_verified(self):
        class MockVerifier:
            def is_valid(self): return True
            def verify_single(self, crop):
                return True, 0.85

        self.pipeline.verifier = MockVerifier()
        # Mock OCR that outputs unreadable mask with wildcards
        class MockWildcardOCR:
            def predict_single(self, crop, *args, **kwargs):
                return "########", 0.0, "type1"

        self.pipeline.ocr = MockWildcardOCR()
        scene = np.full((480, 640, 3), 200, dtype=np.uint8)
        det = PlateDetection(
            bbox=(100, 100, 160, 36),
            quad=[100.0, 100.0, 260.0, 100.0, 260.0, 136.0, 100.0, 136.0],
            plate_type="type1",
            confidence=0.88,
        )
        processed = self.pipeline.recognize_single(scene, det)
        # Should retain type1 and the wildcard mask
        self.assertEqual(processed.plate_type, "type1")
        self.assertEqual(processed.text, "########")

        # Now test when PlateVerifier gives low score (unconfirmed noise)
        class MockFailingVerifier:
            def is_valid(self): return True
            def verify_single(self, crop):
                return False, 0.40

        self.pipeline.verifier = MockFailingVerifier()
        det_fail = PlateDetection(
            bbox=(100, 100, 160, 36),
            quad=[100.0, 100.0, 260.0, 100.0, 260.0, 136.0, 100.0, 136.0],
            plate_type="type1",
            confidence=0.88,
        )
        processed_fail = self.pipeline.recognize_single(scene, det_fail)
        # Should be annulled to other with empty text
        self.assertEqual(processed_fail.plate_type, "other")
        self.assertEqual(processed_fail.text, "")

    def test_predict_batch(self):
        img1 = np.full((100, 100, 3), 5, dtype=np.uint8)  # Black dummy image
        img2 = np.full((100, 100, 3), 5, dtype=np.uint8)
        batch_res = self.pipeline.predict_batch([img1, img2])
        self.assertEqual(len(batch_res), 2)
        self.assertEqual(len(batch_res[0]), 0)
        self.assertEqual(len(batch_res[1]), 0)

    def test_clean_plate_accuracy_smoke(self):
        """Smoke test verifying that clean real plates recognize accurately and pass GOST verification."""
        import cv2
        img_path = os.path.join(ROOT_DIR, "dataset", "images", "real", "real_type1b_0001.jpg")
        if os.path.exists(img_path):
            real_pipe = OmniPlatePipeline(device="cpu")
            img = cv2.imread(img_path)
            dets = real_pipe.predict(img)
            self.assertGreaterEqual(len(dets), 1)
            best_det = dets[0]
            self.assertEqual(best_det.plate_type, "type1b")
            self.assertEqual(best_det.text, "AH889777")
            self.assertGreaterEqual(best_det.confidence, 0.70)


if __name__ == "__main__":
    unittest.main()
