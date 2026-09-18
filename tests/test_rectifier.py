#!/usr/bin/env python3
"""
Unit tests for PlateRectifier (src/pipeline/rectifier.py).
"""

import os
import sys
import unittest
import numpy as np
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.rectifier import PlateRectifier


class TestPlateRectifier(unittest.TestCase):
    def setUp(self):
        self.rectifier = PlateRectifier()

    def test_parse_quad_formats(self):
        # 1. String format
        q_str = "100, 200, 300, 210, 290, 280, 95, 270"
        pts1 = self.rectifier.parse_quad(q_str)
        self.assertEqual(pts1.shape, (4, 2))
        self.assertAlmostEqual(pts1[0, 0], 100.0)
        self.assertAlmostEqual(pts1[3, 1], 270.0)

        # 2. Flat sequence
        q_list = [100, 200, 300, 210, 290, 280, 95, 270]
        pts2 = self.rectifier.parse_quad(q_list)
        self.assertEqual(pts2.shape, (4, 2))
        np.testing.assert_allclose(pts1, pts2)

        # 3. 2D array
        q_arr = np.array([[100, 200], [300, 210], [290, 280], [95, 270]], dtype=np.float32)
        pts3 = self.rectifier.parse_quad(q_arr)
        self.assertEqual(pts3.shape, (4, 2))
        np.testing.assert_allclose(pts1, pts3)

    def test_order_quad_points(self):
        # Points permuted: [BR, TL, BL, TR]
        permuted = np.array([
            [300.0, 250.0],  # BR
            [100.0, 100.0],  # TL
            [100.0, 250.0],  # BL
            [300.0, 100.0],  # TR
        ], dtype=np.float32)

        ordered = self.rectifier.order_quad_points(permuted)
        expected = np.array([
            [100.0, 100.0],  # TL
            [300.0, 100.0],  # TR
            [300.0, 250.0],  # BR
            [100.0, 250.0],  # BL
        ], dtype=np.float32)
        np.testing.assert_allclose(ordered, expected)

    def test_rectify_dimensions(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        quad = [100, 150, 300, 155, 295, 210, 105, 205]

        # Type 1 & Type 1B -> 160x36 (HxW: 36, 160)
        w1 = self.rectifier.rectify(img, quad, plate_type="type1")
        self.assertEqual(w1.shape, (36, 160, 3))

        w1b = self.rectifier.rectify(img, quad, plate_type="type1b")
        self.assertEqual(w1b.shape, (36, 160, 3))

        # Type 1A -> 160x96 (HxW: 96, 160)
        w1a = self.rectifier.rectify(img, quad, plate_type="type1a")
        self.assertEqual(w1a.shape, (96, 160, 3))

    def test_split_and_stitch_type1a(self):
        img_1a = np.zeros((96, 160, 3), dtype=np.uint8)
        img_1a[:48, :, :] = 100  # Top line
        img_1a[48:, :, :] = 200  # Bottom line

        top, bot = self.rectifier.split_type1a(img_1a)
        self.assertEqual(top.shape, (48, 160, 3))
        self.assertEqual(bot.shape, (48, 160, 3))
        self.assertTrue(np.all(top == 100))
        self.assertTrue(np.all(bot == 200))

        # Natural stitch
        stitched_raw = self.rectifier.stitch_type1a_horizontal(top, bot)
        self.assertEqual(stitched_raw.shape, (48, 320, 3))

        # Stitched resized to canonical single-line (160x36)
        stitched_canonical = self.rectifier.stitch_type1a_horizontal(top, bot, target_size=(160, 36))
        self.assertEqual(stitched_canonical.shape, (36, 160, 3))

    def test_bbox_fallback(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        bbox = "50, 60, 120, 40"
        crop = self.rectifier.rectify_bbox_fallback(img, bbox, plate_type="type1")
        self.assertEqual(crop.shape, (36, 160, 3))

    def test_rectify_margin(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        quad = [100, 150, 300, 155, 295, 210, 105, 205]
        # Margin tuple
        w_margin = self.rectifier.rectify(img, quad, plate_type="type1", margin=(0.04, 0.03))
        self.assertEqual(w_margin.shape, (36, 160, 3))
        # Margin scalar float
        w_scalar = self.rectifier.rectify(img, quad, plate_type="type1a", margin=0.03)
        self.assertEqual(w_scalar.shape, (96, 160, 3))

    def test_adaptive_split_seam(self):
        # Create synthetic plate with text lines in upper and lower halves, gap at row 52
        img = np.full((96, 160, 3), 240, dtype=np.uint8)  # White plate background
        # Top text: black pixels with variance
        img[15:48:2, 20:140:2] = 20
        # Bottom text: black pixels with variance
        img[56:88:2, 20:140:2] = 20
        # Gap is around y=49..55
        seam_y = self.rectifier.find_adaptive_split_seam(img)
        self.assertTrue(48 <= seam_y <= 55, f"Expected seam between 48 and 55, got {seam_y}")

        top, bot = self.rectifier.split_type1a(img, adaptive_seam=True)
        self.assertEqual(top.shape, (48, 160, 3))
        self.assertEqual(bot.shape, (48, 160, 3))

    def test_refine_corners_subpixel(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        quad = np.array([[21.0, 21.0], [79.0, 21.0], [79.0, 79.0], [21.0, 79.0]], dtype=np.float32)
        refined = self.rectifier.refine_corners_subpixel(img, quad, window_size=(5, 5), max_drift_px=3.0)
        self.assertEqual(refined.shape, (4, 2))
        # Ensure refinement does not diverge
        for i in range(4):
            self.assertLessEqual(np.linalg.norm(refined[i] - quad[i]), 3.01)

        # Test rectify with refine_corners=True
        warped = self.rectifier.rectify(img, quad, plate_type="type1", refine_corners=True)
        self.assertEqual(warped.shape, (36, 160, 3))


if __name__ == "__main__":
    unittest.main()
