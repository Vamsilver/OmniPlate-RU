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

    @staticmethod
    def refine_corners_subpixel(
        image: np.ndarray,
        pts: np.ndarray,
        window_size: Tuple[int, int] = (5, 5),
        max_drift_px: float = 3.5,
    ) -> np.ndarray:
        """
        Applies subpixel corner refinement (cv2.cornerSubPix) to quadrilateral keypoints
        using local image gradients.
        Safely bounds drift to max_drift_px to prevent divergence on noisy borders.
        """
        if image is None or image.size == 0 or pts is None or len(pts) != 4:
            return pts

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            h, w = gray.shape[:2]
            corners = pts.astype(np.float32).reshape(4, 1, 2).copy()

            # Ensure coordinates are safely within image boundaries
            for i in range(4):
                corners[i, 0, 0] = np.clip(corners[i, 0, 0], 2.0, w - 3.0)
                corners[i, 0, 1] = np.clip(corners[i, 0, 1], 2.0, h - 3.0)

            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01)
            refined = cv2.cornerSubPix(gray, corners, window_size, (-1, -1), crit)
            refined_pts = refined.reshape(4, 2)

            # Verification: enforce max_drift_px and convexity
            res = pts.copy()
            for i in range(4):
                drift = np.linalg.norm(refined_pts[i] - pts[i])
                if drift <= max_drift_px:
                    res[i] = refined_pts[i]

            # Verify polygon remains convex and non-degenerate
            if cv2.isContourConvex(res.astype(np.int32)):
                return res
            return pts
        except Exception:
            return pts

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
        margin: Union[float, Tuple[float, float]] = 0.0,
        refine_corners: bool = False,
    ) -> np.ndarray:
        """
        Warps the quadrilateral region defined by `quad` into a canonical rectangular crop.

        Args:
            image: Source image (BGR or RGB, HxWxC or HxW).
            quad: 4 corner coordinates (str, list, or ndarray).
            plate_type: One of 'type1', 'type1a', 'type1b', 'other'.
            target_size: Optional (width, height) override. Defaults to CANONICAL_SIZES.
            auto_order: If True, re-orders vertices to [TL, TR, BR, BL].
            margin: Safety margin padding ratio (e.g. 0.035 for 3.5% padding, or (0.035, 0.025)).
                   Expands the captured region outward to prevent clipping boundary strokes of symbols.
            refine_corners: If True, applies subpixel corner refinement via local gradients.

        Returns:
            Warped numpy array of shape (target_h, target_w, C).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty.")

        src_pts = self.parse_quad(quad)

        if auto_order:
            src_pts = self.order_quad_points(src_pts)

        if refine_corners:
            src_pts = self.refine_corners_subpixel(image, src_pts)

        if target_size is None:
            target_w, target_h = self.get_canonical_size(plate_type)
        else:
            target_w, target_h = target_size

        if isinstance(margin, (tuple, list)):
            margin_x, margin_y = float(margin[0]), float(margin[1])
        else:
            margin_x = margin_y = float(margin)

        # Inset destination coordinates by safety margin so warp captures surrounding image context
        pad_w = float(target_w) * max(0.0, min(0.15, margin_x))
        pad_h = float(target_h) * max(0.0, min(0.15, margin_y))

        dst_pts = np.array(
            [
                [pad_w, pad_h],
                [float(target_w - 1) - pad_w, pad_h],
                [float(target_w - 1) - pad_w, float(target_h - 1) - pad_h],
                [pad_w, float(target_h - 1) - pad_h],
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
    def find_adaptive_split_seam(
        cls,
        rectified_1a: np.ndarray,
        search_range: Tuple[float, float] = (0.44, 0.54),
    ) -> int:
        """
        Dynamically finds the optimal horizontal split line between line 1 and line 2
        for Type 1A square plates by evaluating row brightness, intra-row variance, and edge energy.
        Prevents vertical cutting through character strokes.

        Args:
            rectified_1a: Warped crop of shape (H, W, C) or (H, W).
            search_range: (min_ratio, max_ratio) of total height H to search for the seam.

        Returns:
            Optimal split row index y (int).
        """
        h = rectified_1a.shape[0]
        if h < 30:
            return h // 2

        if rectified_1a.ndim == 3:
            gray = cv2.cvtColor(rectified_1a, cv2.COLOR_BGR2GRAY)
        else:
            gray = rectified_1a

        row_stds = np.array([float(np.std(gray[y, :])) for y in range(h)], dtype=np.float32)
        # Flat image guard: If image lacks intra-row text variance, use geometric midpoint
        if float(np.mean(row_stds)) < 8.0:
            return h // 2

        y_min = max(1, int(h * search_range[0]))
        y_max = min(h - 2, int(h * search_range[1]))
        if y_min >= y_max:
            return h // 2

        # Vertical Sobel gradient to detect character strokes crossing the row
        sobel_y = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))

        best_y = h // 2
        min_energy = 1e9

        # Seam between lines has:
        # 1. Low darkness (bright background reflective plate)
        # 2. Low edge intensity (no character strokes crossing)
        # 3. Low intra-row standard deviation (pure background row)
        # 4. Strict center anchoring to prevent slicing character rows
        for y in range(y_min, y_max + 1):
            darkness = 255.0 - float(np.mean(gray[y, :]))
            edge = float(np.mean(sobel_y[y, :]))
            std = float(row_stds[y])
            center_dist = abs(y - (h / 2.0)) / float(h)
            energy = darkness + (2.0 * edge) + (1.5 * std) + (60.0 * center_dist) + (150.0 * (center_dist ** 2))
            if energy < min_energy:
                min_energy = energy
                best_y = y

        return best_y

    @classmethod
    def split_type1a(
        cls,
        rectified_1a: np.ndarray,
        adaptive_seam: bool = True,
        vertical_margin: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits a canonical 160x96 Type 1A crop into its two distinct text lines:
        - Top line (H=48): Series Letter + 3 Digits (e.g., 'A123')
        - Bottom line (H=48): 2 Series Letters + Region (e.g., 'BC716')

        Args:
            rectified_1a: Warped image of shape (96, 160, C).
            adaptive_seam: If True, uses dynamic seam finding to avoid cutting letter strokes.
            vertical_margin: Protective vertical margin padding (in px, e.g. 0 to 2) to preserve
                            character descenders (У, Д, Ц, Р) and ascenders/serifs.

        Returns:
            (top_line, bottom_line) tuple, each normalized to canonical (48, 160, C).
        """
        h, w = rectified_1a.shape[:2]

        if adaptive_seam and h >= 30:
            mid = cls.find_adaptive_split_seam(rectified_1a)
        else:
            mid = h // 2

        pad_y = max(0, int(vertical_margin))
        raw_top = rectified_1a[:min(h, mid + pad_y), :]
        raw_bottom = rectified_1a[max(0, mid - pad_y):, :]

        target_h = cls.TYPE1A_LINE_HEIGHT  # 48 px

        # Normalize line heights to canonical 48 px
        if raw_top.shape[0] != target_h:
            top_line = cv2.resize(raw_top, (w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            top_line = raw_top

        if raw_bottom.shape[0] != target_h:
            bottom_line = cv2.resize(raw_bottom, (w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            bottom_line = raw_bottom

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
            top_line: Image of shape (H1, W1, C).
            bottom_line: Image of shape (H2, W2, C).
            target_size: Optional (width, height) to resize after stitching.

        Returns:
            Stitched image.
        """
        h1, w1 = top_line.shape[:2]
        h2, w2 = bottom_line.shape[:2]

        # Ensure matching heights before horizontal concatenation
        if h1 != h2:
            target_h = max(h1, h2, cls.TYPE1A_LINE_HEIGHT)
            top_line = cv2.resize(top_line, (w1, target_h), interpolation=cv2.INTER_LINEAR)
            bottom_line = cv2.resize(bottom_line, (w2, target_h), interpolation=cv2.INTER_LINEAR)

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
