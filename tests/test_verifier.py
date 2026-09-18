import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.verifier import PlateVerifier

class TestPlateVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = PlateVerifier(device="cpu", use_onnx=True)

    def test_initialization(self):
        self.assertTrue(self.verifier.is_valid())

    def test_verify_single_shape(self):
        dummy_crop = np.zeros((36, 160, 3), dtype=np.uint8)
        is_plate, score = self.verifier.verify_single(dummy_crop)
        self.assertIsInstance(is_plate, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_verify_batch(self):
        crops = [np.zeros((36, 160, 3), dtype=np.uint8) for _ in range(4)]
        res = self.verifier.verify_batch(crops)
        self.assertEqual(len(res), 4)
        for is_p, sc in res:
            self.assertIsInstance(is_p, bool)
            self.assertIsInstance(sc, float)

if __name__ == "__main__":
    unittest.main()
