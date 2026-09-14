#!/usr/bin/env python3
"""
Real Dataset Audit and Quality Filter for Volga IT 2026.
Evaluates all harvested real images using:
1. YOLO-Pose detector (models/detector_yolo_pose_best.pt)
2. Strict geometric aspect ratio validation (Type 1B ~4.5:1, Type 1A ~1.5:1)
3. Strict HSV color filtering (Type 1B must be authentically yellow!)
4. Minimum resolution and boundary sanity checks.
"""

import os
import sys
import glob
import csv
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

def is_yellow_plate(crop_bgr: np.ndarray):
    """Verifies if the plate crop has authentic yellow background (Type 1B)"""
    if crop_bgr is None or crop_bgr.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Yellow in HSV: Hue 14 to 42, Saturation > 45, Value > 60
    lower_yellow = np.array([14, 45, 60], dtype=np.uint8)
    upper_yellow = np.array([42, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    ratio = float(np.count_nonzero(mask)) / float(crop_bgr.shape[0] * crop_bgr.shape[1])
    # Yellow background should cover at least 18% of the crop area
    return ratio >= 0.18, ratio

def is_square_plate(w: int, h: int) -> bool:
    """Verifies if plate dimensions match Type 1A (GOST 290x170 mm -> ~1.7:1)"""
    if h <= 0:
        return False
    aspect = float(w) / float(h)
    return 1.15 <= aspect <= 2.25

def is_rect_plate(w: int, h: int) -> bool:
    """Verifies if plate dimensions match Type 1 / 1B (GOST 520x112 mm -> ~4.6:1)"""
    if h <= 0:
        return False
    aspect = float(w) / float(h)
    return 2.8 <= aspect <= 6.0

def main():
    print("=" * 60)
    print(" Volga IT: Real Dataset Quality & Validity Audit")
    print("=" * 60)

    from ultralytics import YOLO

    model_path = os.path.join(ROOT_DIR, "models", "detector_yolo_pose_best.pt")
    if not os.path.exists(model_path):
        print(f"[-] Detector model not found: {model_path}")
        return

    model = YOLO(model_path)
    real_dir = os.path.join(ROOT_DIR, "dataset", "images", "real")
    preview_dir = os.path.join(ROOT_DIR, "dataset", "audit_previews")
    os.makedirs(preview_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(real_dir, "real_*.jpg")))
    print(f"Loaded detector. Auditing {len(all_files)} real images...\n")

    results = {
        "type1b": {"total": 0, "pass": 0, "fail_no_det": 0, "fail_geom": 0, "fail_not_yellow": 0, "kept": []},
        "type1a": {"total": 0, "pass": 0, "fail_no_det": 0, "fail_not_square": 0, "kept": []},
        "other":  {"total": 0, "pass": 0, "fail_no_det": 0, "kept": []}
    }

    audit_csv_path = os.path.join(ROOT_DIR, "dataset", "audit_results.csv")
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as af:
        writer = csv.writer(af, delimiter=";")
        writer.writerow(["filename", "category", "status", "reason", "bbox", "quad", "w", "h", "aspect", "extra"])

        for idx, fpath in enumerate(all_files):
            fname = os.path.basename(fpath)
            if "type1b" in fname:
                cat = "type1b"
            elif "type1a" in fname:
                cat = "type1a"
            else:
                cat = "other"

            results[cat]["total"] += 1
            img = cv2.imread(fpath)
            if img is None:
                writer.writerow([fname, cat, "FAIL", "CORRUPT_IMAGE", "", "", 0, 0, 0, ""])
                continue

            ih, iw = img.shape[:2]

            # Run YOLO-Pose with moderate confidence
            preds = model(fpath, conf=0.35, verbose=False)
            if len(preds) == 0 or len(preds[0].boxes) == 0:
                results[cat]["fail_no_det"] += 1
                writer.writerow([fname, cat, "FAIL", "NO_DETECTION", "", "", 0, 0, 0, ""])
                continue

            # Pick best box
            box = preds[0].boxes[0]
            bx, by, bw, bh = box.xywh[0].cpu().numpy()
            x = int(max(0, bx - bw / 2))
            y = int(max(0, by - bh / 2))
            w = int(min(iw - x, bw))
            h = int(min(ih - y, bh))

            if w < 50 or h < 15:
                results[cat]["fail_no_det"] += 1
                writer.writerow([fname, cat, "FAIL", "TOO_SMALL", f"{x},{y},{w},{h}", "", w, h, 0, ""])
                continue

            aspect = float(w) / float(h)
            crop = img[y:y+h, x:x+w]

            # Quad points if available
            quad_str = ""
            if preds[0].keypoints is not None and len(preds[0].keypoints.xy) > 0:
                kpts = preds[0].keypoints.xy[0].cpu().numpy()
                quad_str = ",".join(str(int(v)) for pt in kpts for v in pt)

            # Category-specific validation
            if cat == "type1b":
                if not is_rect_plate(w, h):
                    results[cat]["fail_geom"] += 1
                    writer.writerow([fname, cat, "FAIL", "BAD_ASPECT_RATIO", f"{x},{y},{w},{h}", quad_str, w, h, f"{aspect:.2f}", ""])
                    continue
                is_yel, yel_ratio = is_yellow_plate(crop)
                if not is_yel:
                    results[cat]["fail_not_yellow"] += 1
                    writer.writerow([fname, cat, "FAIL", "NOT_YELLOW", f"{x},{y},{w},{h}", quad_str, w, h, f"{aspect:.2f}", f"yellow={yel_ratio:.2f}"])
                    continue
                # Passed Type 1B
                results[cat]["pass"] += 1
                results[cat]["kept"].append((fname, f"{x},{y},{w},{h}", quad_str))
                writer.writerow([fname, cat, "PASS", "VALID_TYPE1B", f"{x},{y},{w},{h}", quad_str, w, h, f"{aspect:.2f}", f"yellow={yel_ratio:.2f}"])
                if len(results[cat]["kept"]) <= 20:
                    cv2.imwrite(os.path.join(preview_dir, f"valid_{fname}"), crop)

            elif cat == "type1a":
                if not is_square_plate(w, h):
                    results[cat]["fail_not_square"] += 1
                    writer.writerow([fname, cat, "FAIL", "NOT_SQUARE_TYPE1A", f"{x},{y},{w},{h}", quad_str, w, h, f"{aspect:.2f}", ""])
                    continue
                # Passed Type 1A
                results[cat]["pass"] += 1
                results[cat]["kept"].append((fname, f"{x},{y},{w},{h}", quad_str))
                writer.writerow([fname, cat, "PASS", "VALID_TYPE1A", f"{x},{y},{w},{h}", quad_str, w, h, f"{aspect:.2f}", ""])
                if len(results[cat]["kept"]) <= 20:
                    cv2.imwrite(os.path.join(preview_dir, f"valid_{fname}"), crop)

            else: # other
                results[cat]["pass"] += 1
                results[cat]["kept"].append((fname, f"{x},{y},{w},{h}", quad_str))
                writer.writerow([fname, cat, "PASS", "VALID_OTHER", f"{x},{y},{w},{h}", quad_str, w, h, f"{aspect:.2f}", ""])
                if len(results[cat]["kept"]) <= 10:
                    cv2.imwrite(os.path.join(preview_dir, f"valid_{fname}"), crop)

    print("\n================ AUDIT SUMMARY ================")
    for c, data in results.items():
        pass_pct = (data['pass'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"[{c.upper()}]: Total={data['total']} | Valid PASS={data['pass']} ({pass_pct:.1f}%) | Rejected={data['total'] - data['pass']}")
        for k, v in data.items():
            if k.startswith("fail_"):
                print(f"   - {k}: {v}")
    print(f"\nDetailed audit log written to: {audit_csv_path}")

if __name__ == "__main__":
    main()
