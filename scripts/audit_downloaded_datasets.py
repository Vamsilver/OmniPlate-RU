#!/usr/bin/env python3
"""
OmniPlate-RU — Audit of Downloaded Datasets (Roboflow, Nomeroff, Parquet).
Evaluates 50 random samples (seed=42) from each source against OmniPlatePipeline.
Generates metrics (Exact Match, CER) and visual review images.
"""

import json
import os
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection

CYR_TO_LAT: Dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "а": "A", "в": "B", "е": "E", "к": "K", "м": "M", "н": "H",
    "о": "O", "р": "P", "с": "C", "т": "T", "у": "Y", "х": "X",
}

ROBOFLOW_CLASSES: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"
]


def clean_plate_text(text: str) -> str:
    """Normalizes plate text: removes spaces/hyphens, maps Cyrillic to Latin, uppercases."""
    if not text:
        return ""
    res = []
    for ch in text.strip().upper():
        if ch in CYR_TO_LAT:
            res.append(CYR_TO_LAT[ch])
        elif ch.isalnum():
            res.append(ch)
    return "".join(res)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def calculate_cer(gt: str, pred: str) -> float:
    """Computes Character Error Rate (CER)."""
    if not gt and not pred:
        return 0.0
    if not gt:
        return 1.0
    dist = levenshtein_distance(gt, pred)
    return dist / max(len(gt), 1)


@dataclass
class AuditSample:
    dataset: str
    sample_id: str
    image: np.ndarray
    gt_text: str
    bbox: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class RoboflowZipAdapter:
    """Adapter for Roboflow YOLOv8 character-level annotation zip."""

    def __init__(self, zip_path: str):
        self.zip_path = zip_path

    def load_samples(self, n_samples: int = 50, seed: int = 42) -> List[AuditSample]:
        with zipfile.ZipFile(self.zip_path) as z:
            all_files = z.namelist()
            label_files = [f for f in all_files if f.endswith(".txt") and "/labels/" in f]
            label_files.sort()

            rng = random.Random(seed)
            rng.shuffle(label_files)

            samples: List[AuditSample] = []
            for lbl_path in label_files:
                if len(samples) >= n_samples:
                    break

                content = z.read(lbl_path).decode("utf-8").strip()
                if not content:
                    continue

                lines = [l.strip().split() for l in content.split("\n") if l.strip()]
                parsed_boxes = []
                for parts in lines:
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if 0 <= cid < len(ROBOFLOW_CLASSES):
                            char = ROBOFLOW_CLASSES[cid]
                            parsed_boxes.append((xc, yc, w, h, char))

                if len(parsed_boxes) < 4:
                    continue

                # Check if two-line (Type 1A) or single-line
                ys = [b[1] for b in parsed_boxes]
                y_span = max(ys) - min(ys)
                if y_span > 0.30:
                    mean_y = (max(ys) + min(ys)) / 2.0
                    line1 = sorted([b for b in parsed_boxes if b[1] < mean_y], key=lambda b: b[0])
                    line2 = sorted([b for b in parsed_boxes if b[1] >= mean_y], key=lambda b: b[0])
                    reconstructed_text = "".join(b[4] for b in line1) + "".join(b[4] for b in line2)
                else:
                    line_sorted = sorted(parsed_boxes, key=lambda b: b[0])
                    reconstructed_text = "".join(b[4] for b in line_sorted)

                img_path = lbl_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
                if img_path not in all_files:
                    img_path = lbl_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".png"
                if img_path not in all_files:
                    continue

                img_bytes = z.read(img_path)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                base_id = Path(lbl_path).stem
                samples.append(AuditSample(
                    dataset="Roboflow",
                    sample_id=base_id,
                    image=img,
                    gt_text=clean_plate_text(reconstructed_text),
                    meta={"lbl_path": lbl_path, "img_path": img_path, "num_chars": len(parsed_boxes)}
                ))

            return samples


class NomeroffZipAdapter:
    """Adapter for Nomeroff Net Russian Plates zip."""

    def __init__(self, zip_path: str):
        self.zip_path = zip_path

    def load_samples(self, n_samples: int = 50, seed: int = 42) -> List[AuditSample]:
        with zipfile.ZipFile(self.zip_path) as z:
            all_files = z.namelist()
            json_files = [f for f in all_files if f.endswith(".json") and "/ann/" in f]
            json_files.sort()

            rng = random.Random(seed)
            rng.shuffle(json_files)

            samples: List[AuditSample] = []
            for j_path in json_files:
                if len(samples) >= n_samples:
                    break

                try:
                    data = json.loads(z.read(j_path).decode("utf-8"))
                except Exception:
                    continue

                gt_raw = data.get("description") or data.get("name") or ""
                gt_text = clean_plate_text(gt_raw)
                if not gt_text or len(gt_text) < 4:
                    continue

                img_path_png = j_path.replace("/ann/", "/img/").rsplit(".", 1)[0] + ".png"
                img_path_jpg = j_path.replace("/ann/", "/img/").rsplit(".", 1)[0] + ".jpg"

                target_img = None
                if img_path_png in all_files:
                    target_img = img_path_png
                elif img_path_jpg in all_files:
                    target_img = img_path_jpg

                if not target_img:
                    continue

                img_bytes = z.read(target_img)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                samples.append(AuditSample(
                    dataset="Nomeroff",
                    sample_id=Path(j_path).stem,
                    image=img,
                    gt_text=gt_text,
                    meta={"ann_path": j_path, "img_path": target_img, "raw_data": data}
                ))

            return samples


class HuggingFaceParquetAdapter:
    """Adapter for HuggingFace Parquet dataset."""

    def __init__(self, parquet_paths: List[str]):
        self.parquet_paths = parquet_paths

    def load_samples(self, n_samples: int = 50, seed: int = 42) -> List[AuditSample]:
        all_rows = []
        for p in self.parquet_paths:
            if not os.path.exists(p):
                continue
            table = pq.read_table(p)
            for row in table.to_pylist():
                all_rows.append(row)

        rng = random.Random(seed)
        rng.shuffle(all_rows)

        samples: List[AuditSample] = []
        for row in all_rows:
            if len(samples) >= n_samples:
                break

            img_dict = row.get("image", {})
            img_bytes = img_dict.get("bytes")
            img_path = img_dict.get("path", "")
            if not img_bytes:
                continue

            stem = Path(img_path).name
            prefix = stem.split("_")[0]
            gt_text = clean_plate_text(prefix)

            objs = row.get("objects", {})
            bboxes = objs.get("bbox", []) if objs else []
            primary_bbox = bboxes[0] if len(bboxes) > 0 else None

            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            samples.append(AuditSample(
                dataset="Parquet",
                sample_id=stem[:30],
                image=img,
                gt_text=gt_text,
                bbox=primary_bbox,
                meta={"full_path": img_path, "bboxes": bboxes}
            ))

        return samples


# ---------------------------------------------------------------------------
# Pipeline Inference & Audit Execution
# ---------------------------------------------------------------------------

def run_sample_inference(
    pipeline: OmniPlatePipeline,
    sample: AuditSample,
) -> Tuple[str, str, float, Optional[np.ndarray]]:
    """
    Runs pipeline on sample. Returns (pred_text, pred_type, conf, crop_to_visualize).
    """
    h, w = sample.image.shape[:2]
    dets = pipeline.predict(sample.image)

    # 1. Pipeline found detection(s)
    if dets:
        best_det = dets[0]
        if sample.bbox and len(dets) > 1:
            gx, gy, gw, gh = sample.bbox
            best_iou = -1.0
            for d in dets:
                bx, by, bw, bh = d.bbox
                ix1, iy1 = max(bx, gx), max(by, gy)
                ix2, iy2 = min(bx + bw, gx + gw), min(by + gy, gh + gy)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                union = (bw * bh) + (gw * gh) - inter
                iou = inter / max(1.0, union)
                if iou > best_iou:
                    best_iou = iou
                    best_det = d

        vis_crop = best_det.rectified_crop
        if vis_crop is None:
            bx, by, bw, bh = best_det.bbox
            vis_crop = sample.image[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)]

        return best_det.text, best_det.plate_type, best_det.confidence, vis_crop

    # 2. No detection on image: tight crop fallback
    if sample.dataset in ("Roboflow", "Nomeroff") or (w < 450 and h < 200):
        det = PlateDetection(
            bbox=(0, 0, w, h),
            quad=[0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)],
            plate_type="type1",
            confidence=0.5,
        )
        det = pipeline.recognize_single(sample.image, det)
        vis_crop = det.rectified_crop if det.rectified_crop is not None else sample.image
        return det.text, det.plate_type, det.ocr_confidence, vis_crop

    # 3. If full scene and bbox is given, try OCR directly on GT crop
    if sample.bbox:
        gx, gy, gw, gh = [int(round(v)) for v in sample.bbox]
        gx1, gy1 = max(0, gx), max(0, gy)
        gx2, gy2 = min(w, gx + gw), min(h, gy + gh)
        if gx2 > gx1 and gy2 > gy1:
            crop = sample.image[gy1:gy2, gx1:gx2]
            ch, cw = crop.shape[:2]
            det = PlateDetection(
                bbox=(0, 0, cw, ch),
                quad=[0.0, 0.0, float(cw), 0.0, float(cw), float(ch), 0.0, float(ch)],
                plate_type="type1",
                confidence=0.5,
            )
            det = pipeline.recognize_single(crop, det)
            return det.text, det.plate_type, det.ocr_confidence, crop

    return "", "none", 0.0, None


def create_visual_review_card(
    sample: AuditSample,
    pred_text: str,
    pred_type: str,
    conf: float,
    cer: float,
    is_match: bool,
    crop: Optional[np.ndarray],
) -> np.ndarray:
    """Generates an annotated visual review card for inspection."""
    src_img = sample.image.copy()

    if sample.bbox:
        bx, by, bw, bh = [int(round(v)) for v in sample.bbox]
        cv2.rectangle(src_img, (bx, by), (bx + bw, by + bh), (255, 100, 0), 2)

    sh, sw = src_img.shape[:2]
    target_h = 240
    new_w = int(sw * (target_h / float(sh)))
    src_resized = cv2.resize(src_img, (new_w, target_h))

    if crop is not None and crop.size > 0:
        ch, cw = crop.shape[:2]
        crop_target_h = 240
        crop_new_w = int(cw * (crop_target_h / float(ch)))
        crop_resized = cv2.resize(crop, (crop_new_w, crop_target_h))
    else:
        crop_resized = np.zeros((240, 160, 3), dtype=np.uint8)

    canvas_w = new_w + crop_resized.shape[1] + 20
    canvas_h = target_h + 100
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    canvas[80:80 + target_h, 10:10 + new_w] = src_resized
    canvas[80:80 + target_h, 10 + new_w + 10:10 + new_w + 10 + crop_resized.shape[1]] = crop_resized

    banner_color = (35, 140, 35) if is_match else (35, 35, 180)
    cv2.rectangle(canvas, (0, 0), (canvas_w, 75), banner_color, -1)

    status_str = "MATCH" if is_match else "MISMATCH"
    cv2.putText(canvas, f"[{sample.dataset}] {status_str} | CER: {cer:.2f}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    info_str = f"GT: '{sample.gt_text}'  vs  PRED: '{pred_text}' ({pred_type}, conf={conf:.2f})"
    cv2.putText(canvas, info_str, (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0) if not is_match else (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def main():
    print("=" * 70)
    print("  OmniPlate-RU — Comprehensive Dataset Quality & Label Audit")
    print("  Datasets: Roboflow ZIP, Nomeroff ZIP, HuggingFace Parquet")
    print("  Sample Size: 50 per source (Seed=42)")
    print("=" * 70)

    downloads_dir = Path(r"C:\Users\Vamsi\Downloads")
    roboflow_zip = downloads_dir / "russian_car_plates.v2i.yolov8.zip"
    nomeroff_zip = downloads_dir / "archive.zip"
    parquet_files = [
        str(downloads_dir / "train-00000-of-00001.parquet"),
        str(downloads_dir / "validation-00000-of-00001.parquet"),
        str(downloads_dir / "test-00000-of-00001.parquet"),
    ]

    output_dir = PROJECT_ROOT / "test_output" / "audit_samples_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Initializing OmniPlatePipeline on CUDA...")
    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(iterations=2)

    print("\n[+] Loading 50 samples from Roboflow ZIP...")
    roboflow_adapter = RoboflowZipAdapter(str(roboflow_zip))
    roboflow_samples = roboflow_adapter.load_samples(n_samples=50, seed=42)
    print(f"    Loaded {len(roboflow_samples)} Roboflow samples.")

    print("\n[+] Loading 50 samples from Nomeroff ZIP...")
    nomeroff_adapter = NomeroffZipAdapter(str(nomeroff_zip))
    nomeroff_samples = nomeroff_adapter.load_samples(n_samples=50, seed=42)
    print(f"    Loaded {len(nomeroff_samples)} Nomeroff samples.")

    print("\n[+] Loading 50 samples from HF Parquet...")
    parquet_adapter = HuggingFaceParquetAdapter(parquet_files)
    parquet_samples = parquet_adapter.load_samples(n_samples=50, seed=42)
    print(f"    Loaded {len(parquet_samples)} Parquet samples.")

    all_dataset_samples = {
        "Roboflow": roboflow_samples,
        "Nomeroff": nomeroff_samples,
        "Parquet": parquet_samples,
    }

    summary_stats: Dict[str, Dict[str, Any]] = {}

    for ds_name, samples in all_dataset_samples.items():
        print(f"\n{'=' * 30} Auditing: {ds_name} ({len(samples)} items) {'=' * 30}")
        matches = 0
        total_cer = 0.0
        mismatches_list = []

        for idx, sample in enumerate(samples, 1):
            pred_text, pred_type, conf, crop = run_sample_inference(pipeline, sample)
            pred_clean = clean_plate_text(pred_text)
            gt_clean = clean_plate_text(sample.gt_text)

            is_match = (gt_clean == pred_clean) and len(gt_clean) > 0
            cer = calculate_cer(gt_clean, pred_clean)
            total_cer += cer
            if is_match:
                matches += 1
            else:
                mismatches_list.append({
                    "id": sample.sample_id,
                    "gt": gt_clean,
                    "pred": pred_clean,
                    "type": pred_type,
                    "cer": cer,
                    "conf": conf,
                })

            status_tag = "MATCH" if is_match else "MISMATCH"
            if not is_match or idx <= 5:
                card = create_visual_review_card(sample, pred_clean, pred_type, conf, cer, is_match, crop)
                out_name = f"{ds_name}_{idx:02d}_{status_tag}_{sample.sample_id[:16]}.jpg"
                cv2.imwrite(str(output_dir / out_name), card)

            status_icon = "✓" if is_match else "✗"
            print(f"  [{idx:02d}/50] {status_icon} GT: {gt_clean:<12} | PRED: {pred_clean:<12} | CER: {cer:.2f} | Conf: {conf:.2f} ({sample.sample_id[:18]})")

        acc = (matches / max(1, len(samples))) * 100.0
        mean_cer = (total_cer / max(1, len(samples))) * 100.0
        summary_stats[ds_name] = {
            "total": len(samples),
            "matches": matches,
            "accuracy": acc,
            "mean_cer": mean_cer,
            "mismatches": mismatches_list,
        }

    print("\n" + "=" * 75)
    print("                      AUDIT SUMMARY RESULTS TABLE")
    print("=" * 75)
    print(f"{'Dataset':<15} | {'Samples':<8} | {'Exact Match':<12} | {'Accuracy %':<12} | {'Mean CER %':<12}")
    print("-" * 75)
    for ds_name, stats in summary_stats.items():
        print(f"{ds_name:<15} | {stats['total']:<8} | {stats['matches']:<12} | {stats['accuracy']:<11.1f}% | {stats['mean_cer']:<11.1f}%")
    print("=" * 75)

    summary_path = output_dir / "audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Audit summary and details saved to: {summary_path}")
    print(f"[+] Visual review images saved to: {output_dir}")


if __name__ == "__main__":
    main()
