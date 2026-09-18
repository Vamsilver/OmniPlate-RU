#!/usr/bin/env python3
"""
Build contact sheets for visual inspection of Turbo JDM Type 1A candidates.
"""

import math
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CROPS_DIR = PROJECT_ROOT / "test_output" / "turbo_1a_verified" / "crops"
SHEETS_DIR = PROJECT_ROOT / "test_output" / "turbo_1a_verified" / "sheets"
SHEETS_DIR.mkdir(parents=True, exist_ok=True)

def build_sheets():
    crop_files = sorted(list(CROPS_DIR.glob("*.jpg")))
    print(f"Total crops to review: {len(crop_files)}")
    if not crop_files:
        return

    COLS, ROWS = 5, 5
    PER_SHEET = COLS * ROWS
    CELL_W, CELL_H = 220, 170

    num_sheets = math.ceil(len(crop_files) / PER_SHEET)
    for s_idx in range(num_sheets):
        batch = crop_files[s_idx * PER_SHEET : (s_idx + 1) * PER_SHEET]
        sheet = np.zeros((ROWS * CELL_H, COLS * CELL_W, 3), dtype=np.uint8)
        sheet[:] = (35, 35, 35)

        for i, c_p in enumerate(batch):
            r_idx = i // COLS
            c_idx = i % COLS
            x0 = c_idx * CELL_W
            y0 = r_idx * CELL_H

            im = cv2.imread(str(c_p))
            if im is None:
                continue

            max_w, max_h = CELL_W - 12, CELL_H - 32
            scale = min(max_w / im.shape[1], max_h / im.shape[0])
            nw, nh = max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))
            resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

            dx = x0 + (CELL_W - nw) // 2
            dy = y0 + 6 + (max_h - nh) // 2
            sheet[dy:dy+nh, dx:dx+nw] = resized

            # Text label
            label = c_p.stem.replace("_crop", "").replace("cand_1a_", "")
            cv2.putText(sheet, label, (x0 + 4, y0 + CELL_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        out_path = SHEETS_DIR / f"turbo_sheet_{s_idx:02d}.jpg"
        cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Saved {out_path.name} ({len(batch)} crops)")

if __name__ == "__main__":
    build_sheets()
