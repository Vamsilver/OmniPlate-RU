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
from src.pipeline.ocr import PlateOCR


class DummyOCR:
    """Mock OCR for deterministic pipeline integration testing."""
    def __init__(self):
        self.session = "mock_session"
        self.model = None

    def predict_single(self, crop: np.ndarray):
        # Assert canonical crop size (36, 160, 3)
        assert crop.shape == (36, 160, 3), f"Expected (36, 160, 3), got {crop.shape}"
        return "A123BC77", 0.95


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


if __name__ == "__main__":
    unittest.main()
