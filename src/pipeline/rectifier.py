#!/usr/bin/env python3
"""
OmniPlate-RU Perspective Rectification Module (Substage 3.2).

Performs high-speed 4-point homography and perspective warping:
- Type 1 / Type 1B (Single-line): Canonical 160x36 px (W x H)
- Type 1A (Two-line square): Canonical 160x96 px (W x H)
- Split & Stitch mechanisms for Type 1A 2-line OCR processing.
- Strict latency constraint: <= 4 ms per crop on GPU/CPU.
"""

import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class PlateRectifier:
    """
    Performs perspective warping of detected license plate quads
    into canonical normalized crops for downstream OCR models.
    """

    # Canonical (Width, Height) in pixels according to docs/specs/01_gost_geometry.md
    CANONICAL_SIZES: Dict[str, Tuple[int, int]] = {
        "type1": (160, 36),
        "type1b": (160, 36),
        "type1a": (160, 96),
        "other": (160, 36),
    }

    # Type 1A two-line canonical height splits (48 px each for 96 px total height)
    TYPE1A_LINE_HEIGHT: int = 48

    def __init__(self, default_interpolation: int = cv2.INTER_LINEAR) -> None:
        self.interpolation = default_interpolation

    @staticmethod
    def parse_quad(quad: Union[str, Sequence[Union[int, float]], np.ndarray]) -> np.ndarray:
        """
        Parses various quad representations into a float32 numpy array of shape (4, 2).

        Supported inputs:
            - str: "x1,y1,x2,y2,x3,y3,x4,y4"
            - list/tuple: [x1, y1, x2, y2, x3, y3, x4, y4] or [[x1, y1], ...]
            - np.ndarray: shape (8,) or (4, 2)
        """
        if isinstance(quad, str):
            vals = [float(v.strip()) for v in quad.split(",") if v.strip()]
            if len(vals) != 8:
                raise ValueError(f"Expected 8 coordinates in quad string, got {len(vals)}: {quad}")
            pts = np.array(vals, dtype=np.float32).reshape(4, 2)
        elif isinstance(quad, np.ndarray):
            if quad.shape == (4, 2):
                pts = quad.astype(np.float32)
            elif quad.shape == (8,):
                pts = quad.reshape(4, 2).astype(np.float32)
            else:
                raise ValueError(f"Unexpected numpy array shape for quad: {quad.shape}, expected (4, 2) or (8,)")
        elif isinstance(quad, (list, tuple)):
            arr = np.array(quad, dtype=np.float32)
            if arr.size == 8:
                pts = arr.reshape(4, 2)
            else:
                raise ValueError(f"Expected sequence of 8 numbers or 4 pairs, got size {arr.size}")
        else:
            raise TypeError(f"Unsupported quad type: {type(quad)}")

        return pts

    @staticmethod
    def order_quad_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 quad vertices clockwise starting from Top-Left:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

        Uses coordinate sums and differences:
        - Top-Left has smallest (x + y)
        - Bottom-Right has largest (x + y)
        - Top-Right has smallest (y - x)
        - Bottom-Left has largest (y - x)
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-Left
        rect[2] = pts[np.argmax(s)]  # Bottom-Right

        diff = pts[:, 1] - pts[:, 0]  # (y - x)
        rect[1] = pts[np.argmin(diff)]  # Top-Right
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left

        return rect

    def get_canonical_size(self, plate_type: str = "type1") -> Tuple[int, int]:
        """Returns (width, height) for the given plate_type."""
        norm_type = plate_type.lower().strip()
        return self.CANONICAL_SIZES.get(norm_type, (160, 36))

    def rectify(
        self,
        image: np.ndarray,
        quad: Union[str, Sequence[Union[int, float]], np.ndarray],
        plate_type: str = "type1",
        target_size: Optional[Tuple[int, int]] = None,
        auto_order: bool = True,
    ) -> np.ndarray:
        """
        Warps the quadrilateral region defined by `quad` into a canonical rectangular crop.

        Args:
            image: Source image (BGR or RGB, HxWxC or HxW).
            quad: 4 corner coordinates (str, list, or ndarray).
            plate_type: One of 'type1', 'type1a', 'type1b', 'other'.
            target_size: Optional (width, height) override. Defaults to CANONICAL_SIZES.
            auto_order: If True, re-orders vertices to [TL, TR, BR, BL].

        Returns:
            Warped numpy array of shape (target_h, target_w, C).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty.")

        src_pts = self.parse_quad(quad)

        if auto_order:
            src_pts = self.order_quad_points(src_pts)

        if target_size is None:
            target_w, target_h = self.get_canonical_size(plate_type)
        else:
            target_w, target_h = target_size

        dst_pts = np.array(
            [
                [0.0, 0.0],
                [float(target_w - 1), 0.0],
                [float(target_w - 1), float(target_h - 1)],
                [0.0, float(target_h - 1)],
            ],
            dtype=np.float32,
        )

        # Compute homography transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Perform fast perspective warp
        warped = cv2.warpPerspective(
            image,
            M,
            (target_w, target_h),
            flags=self.interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return warped

    def rectify_bbox_fallback(
        self,
        image: np.ndarray,
        bbox: Union[str, Sequence[Union[int, float]], np.ndarray],
        plate_type: str = "type1",
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Fallback rectification when only axis-aligned BBox (x, y, w, h) is available.
        Crops and resizes to the canonical dimension.
        """
        if isinstance(bbox, str):
            bx, by, bw, bh = [int(float(v.strip())) for v in bbox.split(",") if v.strip()]
        else:
            bx, by, bw, bh = [int(float(v)) for v in bbox[:4]]

        ih, iw = image.shape[:2]
        x1 = max(0, min(iw - 1, bx))
        y1 = max(0, min(ih - 1, by))
        x2 = max(0, min(iw, bx + bw))
        y2 = max(0, min(ih, by + bh))

        if x2 <= x1 or y2 <= y1:
            # Degenerate bbox fallback: return black patch
            tw, th = target_size or self.get_canonical_size(plate_type)
            channels = image.shape[2] if image.ndim == 3 else 1
            return np.zeros((th, tw, channels), dtype=image.dtype)

        crop = image[y1:y2, x1:x2]
        tw, th = target_size or self.get_canonical_size(plate_type)
        return cv2.resize(crop, (tw, th), interpolation=self.interpolation)

    @classmethod
    def split_type1a(cls, rectified_1a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits a canonical 160x96 Type 1A crop into its two distinct text lines:
        - Top line (H=48): Series Letter + 3 Digits (e.g., 'A123')
        - Bottom line (H=48): 2 Series Letters + Region (e.g., 'BC716')

        Args:
            rectified_1a: Warped image of shape (96, 160, C).

        Returns:
            (top_line, bottom_line) tuple, each of shape (48, 160, C).
        """
        h = rectified_1a.shape[0]
        mid = h // 2
        top_line = rectified_1a[:mid, :]
        bottom_line = rectified_1a[mid:, :]
        return top_line, bottom_line

    @classmethod
    def stitch_type1a_horizontal(
        cls,
        top_line: np.ndarray,
        bottom_line: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Stitches two lines of Type 1A horizontally side-by-side into a single continuous strip.
        Resulting natural dimensions: (48, 320, C).
        Optionally resizes to `target_size` (e.g., canonical 160x36 for shared 1-line OCR).

        Args:
            top_line: Image of shape (48, 160, C).
            bottom_line: Image of shape (48, 160, C).
            target_size: Optional (width, height) to resize after stitching.

        Returns:
            Stitched image.
        """
        stitched = np.hstack([top_line, bottom_line])
        if target_size is not None:
            stitched = cv2.resize(stitched, target_size, interpolation=cv2.INTER_LINEAR)
        return stitched


def main():
    """Quick verification and benchmark of PlateRectifier."""
    import time

    print("=" * 60)
    print("🚀 Benchmarking PlateRectifier (OpenCV Perspective Warp)")
    print("=" * 60)

    rectifier = PlateRectifier()

    # Synthetic test image
    h_src, w_src = 1080, 1920
    test_img = np.random.randint(0, 255, (h_src, w_src, 3), dtype=np.uint8)

    # Quad corners simulating angled plate
    quad = [500.0, 420.0, 720.0, 410.0, 730.0, 470.0, 495.0, 485.0]

    # Warmup
    for _ in range(10):
        _ = rectifier.rectify(test_img, quad, plate_type="type1")

    # Benchmark 1000 warps
    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = rectifier.rectify(test_img, quad, plate_type="type1")
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_iters) * 1000.0
    print(f"  • Single-line (160x36) Average Latency: {avg_ms:.3f} ms (Budget: <= 4.0 ms) -> {'✅ PASS' if avg_ms <= 4.0 else '❌ FAIL'}")

    # Type 1A split & stitch test
    quad_1a = [400.0, 300.0, 580.0, 310.0, 570.0, 420.0, 395.0, 410.0]
    warped_1a = rectifier.rectify(test_img, quad_1a, plate_type="type1a")
    assert warped_1a.shape[:2] == (96, 160), f"Unexpected shape {warped_1a.shape}"

    top_line, bot_line = rectifier.split_type1a(warped_1a)
    assert top_line.shape[:2] == (48, 160)
    assert bot_line.shape[:2] == (48, 160)

    stitched = rectifier.stitch_type1a_horizontal(top_line, bot_line, target_size=(160, 36))
    assert stitched.shape[:2] == (36, 160)

    print("  • Type 1A (160x96) Split & Stitch: Verified ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
