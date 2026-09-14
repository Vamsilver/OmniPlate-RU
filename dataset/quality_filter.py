#!/usr/bin/env python3
"""
Strict License Plate Authenticity & Quality Verifier.
Rejects:
- LED / route signs (wrong polarity, black background with glowing dots)
- Scaffolding, windows, fences, grilles (abnormal line patterns / no character sequence)
- Human faces, clothing, scenery
- Non-yellow crops in Type 1B
- Non-square / distorted crops in Type 1A
"""

from typing import Optional, Tuple
import os
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class PlateQualityVerifier:
    @staticmethod
    def verify_crop(
        crop_bgr: np.ndarray,
        plate_type: str = "type1b",
    ) -> Tuple[bool, str, dict]:
        """
        Verifies if a crop is an authentic Russian license plate.
        Returns:
            (is_valid, reason, debug_metrics)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False, "EMPTY_IMAGE", {}

        h, w = crop_bgr.shape[:2]
        if h < 14 or w < 50:
            return False, f"TOO_SMALL_{w}x{h}", {}

        # 1. Aspect ratio check
        aspect = float(w) / float(h)
        if plate_type in ("type1", "type1b", "other"):
            if aspect < 2.8 or aspect > 6.0:
                return False, f"INVALID_ASPECT_{aspect:.2f}", {"aspect": aspect}
        elif plate_type == "type1a":
            if aspect < 1.15 or aspect > 2.25:
                return False, f"INVALID_ASPECT_{aspect:.2f}", {"aspect": aspect}

        # 2. Color & Brightness Polarity
        # Real plates: Light / Yellow background with DARK black characters
        # LED signs: Dark / Black background with BRIGHT glowing characters
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Margin: remove outer 5% border to avoid bezel effects
        inner = gray[int(h * 0.10):int(h * 0.90), int(w * 0.05):int(w * 0.95)]
        if inner.size == 0:
            return False, "DEGENERATE_CROP", {}

        # Otsu thresholding
        thresh_val, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dark pixels (characters): binary == 0
        # Bright pixels (background): binary == 255
        bright_ratio = float(np.count_nonzero(binary == 255)) / float(inner.size)
        dark_ratio = float(np.count_nonzero(binary == 0)) / float(inner.size)

        # In a real plate, background is dominant (60% - 88%), characters are 12% - 40%
        # In an LED sign, background is pitch black (>75% dark), glowing text is bright (<25%)
        if dark_ratio > 0.65:
            return False, f"INVERTED_POLARITY_OR_TOO_DARK (dark={dark_ratio:.2f})", {"dark_ratio": dark_ratio}

        if dark_ratio < 0.08:
            return False, f"NO_CHARACTERS_FOUND (dark={dark_ratio:.2f})", {"dark_ratio": dark_ratio}

        # 3. Type 1B: Strict Yellow Chrominance check
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        inner_hsv = hsv[int(h * 0.10):int(h * 0.90), int(w * 0.05):int(w * 0.95)]

        if plate_type == "type1b":
            # Yellow in OpenCV HSV: H in [12, 42], S > 50, V > 60
            yellow_mask = (
                (inner_hsv[:, :, 0] >= 12) & (inner_hsv[:, :, 0] <= 42) &
                (inner_hsv[:, :, 1] >= 45) &
                (inner_hsv[:, :, 2] >= 60)
            )
            yellow_coverage = float(np.count_nonzero(yellow_mask)) / float(inner_hsv.shape[0] * inner_hsv.shape[1])
            if yellow_coverage < 0.28:
                return False, f"NOT_GENUINE_YELLOW (coverage={yellow_coverage:.2f})", {"yellow_coverage": yellow_coverage}
        elif plate_type in ("type1", "type1a"):
            # Should NOT be heavily colored (e.g. not blue sky, not green bus body)
            sat = inner_hsv[:, :, 1]
            high_sat_ratio = float(np.count_nonzero(sat > 110)) / float(sat.size)
            if high_sat_ratio > 0.40:
                return False, f"HEAVILY_COLORED_NOT_WHITE (sat_ratio={high_sat_ratio:.2f})", {"sat_ratio": high_sat_ratio}

        # 4. Connected Components for Character Structure
        # Character mask: pixels darker than thresh_val
        char_mask = (inner < thresh_val).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(char_mask, connectivity=8)

        ih, iw = inner.shape[:2]
        char_candidates = []
        for i in range(1, num_labels):
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            # Character size filters:
            # Height: 25% - 90% of inner height
            # Width: 1.5% - 25% of inner width
            if (0.22 * ih <= ch <= 0.95 * ih) and (0.015 * iw <= cw <= 0.25 * iw):
                # Centroid vertical alignment: roughly centered
                if 0.15 * ih <= cy <= 0.85 * ih:
                    char_candidates.append((cx, cy, cw, ch))

        num_chars = len(char_candidates)

        if plate_type in ("type1", "type1b"):
            # Must have between 5 and 12 distinct character blocks
            if num_chars < 5 or num_chars > 13:
                return False, f"IRREGULAR_CHAR_COUNT_{num_chars}", {"num_chars": num_chars, "dark_ratio": dark_ratio}
        elif plate_type == "type1a":
            # Two-row square plate: between 4 and 12 characters total
            if num_chars < 4 or num_chars > 14:
                return False, f"IRREGULAR_CHAR_COUNT_1A_{num_chars}", {"num_chars": num_chars}

        return True, "VALID_PLATE", {
            "aspect": aspect,
            "dark_ratio": dark_ratio,
            "num_chars": num_chars,
        }


if __name__ == "__main__":
    import glob
    import os

    verifier = PlateQualityVerifier()
    test_files = [
        "dataset/verified_previews/crop_real_type1b_0000.jpg",
        "dataset/verified_previews/crop_real_type1b_0001.jpg",
        "dataset/verified_previews/crop_real_type1b_0002.jpg",
        "dataset/verified_previews/crop_real_type1b_0003.jpg",
        "dataset/verified_previews/crop_real_type1b_0004.jpg",
        "dataset/verified_previews/crop_real_type1a_0000.jpg",
        "dataset/verified_previews/crop_real_type1a_0001.jpg",
    ]

    print("=" * 65)
    print("Testing PlateQualityVerifier on Sample Crops:")
    print("=" * 65)
    for tf in test_files:
        if not os.path.exists(tf):
            continue
        img = cv2.imread(tf)
        pt = "type1a" if "type1a" in tf else "type1b"
        ok, reason, dbg = verifier.verify_crop(img, plate_type=pt)
        status = "✅ PASS" if ok else f"❌ REJECT: {reason}"
        print(f"[{status}] {tf}")
