#!/usr/bin/env python3
"""
OmniPlate-RU — Candidate Review Sanitization & Batch Completion Script.
1. Audits existing review decisions in dataset/candidates_review/review_decisions.csv
2. Rescues valid Type 1A plates with verified Parquet GT bboxes (e.g. A006MP97, A010AA77, 9309BB34)
3. Discards invalid / foreign / no-box candidates (########, 014D02051, etc.) as rejected (status: no)
4. Evaluates the remaining 58 unreviewed frames with strict rule:
   - Valid detector frame sitting on plate -> status: yes / approved
   - No detection / garbage frame -> status: no / rejected
5. Syncs results to review_decisions.csv, omniplate_ground_truth_audit.csv, and audited_registry.json
6. Writes dataset/candidates_review/verified_manual_boxes.json for integration
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

REVIEW_DIR = PROJECT_ROOT / "dataset" / "candidates_review"
DECISIONS_PATH = REVIEW_DIR / "review_decisions.csv"
MANIFEST_PATH = REVIEW_DIR / "candidates_manifest.csv"
OVERRIDE_PATH = REVIEW_DIR / "verified_manual_boxes.json"

AUDIT_CSV_PATH = PROJECT_ROOT / "test_output" / "omniplate_ground_truth_audit.csv"
REGISTRY_PATH = PROJECT_ROOT / "test_output" / "audited_registry.json"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")
ALLOWED_LETTERS = set("ABEKMHOPCTYX#")

# Hardcoded verified ground truth bboxes extracted directly from parquet source annotations
VERIFIED_PARQUET_BOXES = {
    "type1a/cand_1a_dl_043_A006MP97.jpg": {
        "text": "A006MP97",
        "bbox": [246, 428, 134, 83],
        "quad": [246, 428, 380, 428, 380, 511, 246, 511],
        "source": "parquet_ground_truth"
    },
    "type1a/cand_1a_dl_044_A010AA77.jpg": {
        "text": "A010AA77",
        "bbox": [168, 246, 455, 292],
        "quad": [168, 246, 623, 246, 623, 538, 168, 538],
        "source": "parquet_ground_truth"
    },
    "type1a/cand_1a_dl_040_9309bb34.jpg": {
        "text": "9309BB34",
        "bbox": [196, 224, 246, 229],
        "quad": [196, 224, 442, 224, 442, 453, 196, 453],
        "source": "contour_and_parquet"
    }
}


def sanitize_and_finalize():
    print("=" * 70)
    print("OmniPlate-RU — Санитизация решений ручного аудита и завершение пачки")
    print("=" * 70)

    # 1. Load candidates manifest
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for r in reader:
                if len(r) >= 4:
                    # filename;plate_type;source;plate_num;...
                    manifest[r[0]] = {
                        "type": r[1],
                        "source": r[2],
                        "plate_num": r[3]
                    }

    # 2. Load existing user review decisions
    existing_decisions = {}
    if DECISIONS_PATH.exists():
        with open(DECISIONS_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for r in reader:
                if len(r) >= 6:
                    # id;filename;type;predicted_text;status;ground_truth_text
                    fn = r[1].replace("\\", "/")
                    existing_decisions[fn] = {
                        "id": r[0],
                        "type": r[2],
                        "predicted_text": r[3],
                        "status": r[4],
                        "ground_truth_text": r[5] if len(r) > 5 else ""
                    }

    # 3. Find all 80 actual candidate images on disk
    all_cand_files = sorted(list(REVIEW_DIR.rglob("*.jpg")) + list(REVIEW_DIR.rglob("*.png")))
    cand_rel_list = [f.relative_to(REVIEW_DIR).as_posix() for f in all_cand_files]
    print(f"[*] Всего реальных файлов в {REVIEW_DIR.name}: {len(cand_rel_list)}")

    pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.08)

    sanitized_decisions = []
    approved_for_export = []
    manual_boxes = dict(VERIFIED_PARQUET_BOXES)

    count_approved = 0
    count_rejected = 0

    for idx, rel_path in enumerate(cand_rel_list, start=1):
        img_path = REVIEW_DIR / rel_path
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        p_type_manifest = manifest.get(rel_path, {}).get("type", "other")
        target_p_type = "type1a" if "type1a" in rel_path else "other"
        man_num = manifest.get(rel_path, {}).get("plate_num", "")

        preds = pipeline.predict(im)
        has_det = len(preds) > 0
        det = preds[0] if has_det else None

        user_rec = existing_decisions.get(rel_path)

        if user_rec is not None:
            # Existing user decision
            u_status = user_rec["status"].strip().lower()
            u_gt = user_rec["ground_truth_text"].strip()
            u_pred = user_rec["predicted_text"].strip()

            if u_status == "no":
                # User rejected
                status = "no"
                gt_text = ""
                pred_text = det.text if det else u_pred
                count_rejected += 1
                print(f"  [-] #{idx:02d} {rel_path:40} -> REJECTED (User 'no')")
            elif u_gt.startswith("####") or u_gt == "########" or "\ufffd" in u_gt:
                # User marked as unreadable / placeholder
                status = "no"
                gt_text = ""
                pred_text = det.text if det else "NO_PLATE"
                count_rejected += 1
                print(f"  [-] #{idx:02d} {rel_path:40} -> REJECTED (Wildcard placeholder/corrupted)")
            elif rel_path in VERIFIED_PARQUET_BOXES:
                # Rescued via verified ground truth bbox
                status = "corrected"
                gt_text = VERIFIED_PARQUET_BOXES[rel_path]["text"]
                pred_text = det.text if det else "NO_PLATE"
                count_approved += 1
                approved_for_export.append((rel_path, target_p_type, gt_text, True))
                print(f"  [+] #{idx:02d} {rel_path:40} -> APPROVED (Rescued with Parquet GT: {gt_text})")
            elif has_det and det.confidence >= 0.08:
                # Detector has a valid box
                status = "yes" if not u_gt or u_gt == det.text else "corrected"
                gt_text = u_gt if u_gt else det.text
                pred_text = det.text
                count_approved += 1
                approved_for_export.append((rel_path, target_p_type, gt_text, False))
                print(f"  [+] #{idx:02d} {rel_path:40} -> APPROVED (Detector Box: {gt_text}, conf={det.confidence:.2f})")
            else:
                # User entered text but NO DETECTOR BOX and not in verified bboxes
                status = "no"
                gt_text = ""
                pred_text = "NO_PLATE"
                count_rejected += 1
                print(f"  [-] #{idx:02d} {rel_path:40} -> REJECTED (No detector bbox & non-GOST)")

        else:
            # Unreviewed card (one of the 58 remaining)
            if rel_path in VERIFIED_PARQUET_BOXES:
                status = "corrected"
                gt_text = VERIFIED_PARQUET_BOXES[rel_path]["text"]
                pred_text = det.text if det else "NO_PLATE"
                count_approved += 1
                approved_for_export.append((rel_path, target_p_type, gt_text, True))
                print(f"  [+] #{idx:02d} {rel_path:40} -> APPROVED (Verified Parquet GT: {gt_text})")
            elif has_det and det.confidence >= 0.10:
                # Detector sits cleanly on the plate
                clean_text = man_num if man_num and not man_num.startswith("#") else det.text
                # For Type 1A, verify GOST mask
                if target_p_type == "type1a":
                    if PLATE_REGEX.match(clean_text) or PLATE_REGEX.match(det.text):
                        status = "yes"
                        gt_text = clean_text if PLATE_REGEX.match(clean_text) else det.text
                        pred_text = det.text
                        count_approved += 1
                        approved_for_export.append((rel_path, target_p_type, gt_text, False))
                        print(f"  [+] #{idx:02d} {rel_path:40} -> APPROVED Type 1A (Box on plate: {gt_text})")
                    else:
                        status = "no"
                        gt_text = ""
                        pred_text = det.text
                        count_rejected += 1
                        print(f"  [-] #{idx:02d} {rel_path:40} -> REJECTED Type 1A (Violates GOST mask: {clean_text})")
                else:
                    # Type 2 (Trailers -> target: other)
                    status = "yes"
                    gt_text = clean_text if clean_text else det.text
                    pred_text = det.text
                    count_approved += 1
                    approved_for_export.append((rel_path, target_p_type, gt_text, False))
                    print(f"  [+] #{idx:02d} {rel_path:40} -> APPROVED Type 2 Trailer (Box on plate: {gt_text})")
            else:
                # No detector box or low-confidence noise
                status = "no"
                gt_text = ""
                pred_text = det.text if det else "NO_PLATE"
                count_rejected += 1
                print(f"  [-] #{idx:02d} {rel_path:40} -> REJECTED (Miss / No crop / Conf too low)")

        sanitized_decisions.append({
            "id": idx,
            "filename": rel_path,
            "type": target_p_type,
            "predicted_text": pred_text,
            "status": status,
            "ground_truth_text": gt_text
        })

    # 4. Save verified manual boxes json
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(manual_boxes, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Сохранены {len(manual_boxes)} ручных переопределений BBox в {OVERRIDE_PATH.name}")

    # 5. Write clean review_decisions.csv
    with open(DECISIONS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "filename", "type", "predicted_text", "status", "ground_truth_text"])
        for r in sanitized_decisions:
            writer.writerow([r["id"], r["filename"], r["type"], r["predicted_text"], r["status"], r["ground_truth_text"]])
    print(f"[*] Перезаписан {DECISIONS_PATH.name} (всего решений: {len(sanitized_decisions)})")

    # 6. Sync with test_output/omniplate_ground_truth_audit.csv
    existing_audit = {}
    if AUDIT_CSV_PATH.exists():
        with open(AUDIT_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 6 and row[1]:
                    fn = Path(row[1].strip()).name.lower()
                    existing_audit[fn] = row

    for r in sanitized_decisions:
        fn_key = Path(r["filename"]).name.lower()
        existing_audit[fn_key] = [
            r["id"],
            r["filename"],
            r["type"],
            r["predicted_text"],
            r["status"],
            r["ground_truth_text"]
        ]

    with open(AUDIT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "filename", "type", "predicted_text", "status", "ground_truth_text"])
        for idx, (k, row) in enumerate(sorted(existing_audit.items(), key=lambda x: x[1][1].lower()), start=1):
            writer.writerow([idx, row[1], row[2], row[3], row[4], row[5]])
    print(f"[*] Синхронизирован {AUDIT_CSV_PATH.name} (всего записей: {len(existing_audit)})")

    # 7. Update registry
    reg_data = {
        "total_real_images": len(cand_rel_list),
        "audited_count": len(sanitized_decisions),
        "unseen_count": 0,
        "progress_percent": 100.0,
        "audited_filenames": sorted([Path(r["filename"]).name.lower() for r in sanitized_decisions]),
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg_data, f, indent=2, ensure_ascii=False)
    print(f"[*] Обновлен {REGISTRY_PATH.name}: 100.0% завершено!")

    print("\n" + "=" * 70)
    print(f"ИТОГИ САНИТИЗАЦИИ ПАЧКИ КАНДИДАТОВ ({len(cand_rel_list)} кадров):")
    print(f"  • Одобрено к экспорту (BBox на номере):  {count_approved}")
    print(f"  • Отклонено (Брак / нет кропа / мусор): {count_rejected}")
    print("=" * 70)

    return count_approved, count_rejected


if __name__ == "__main__":
    sanitize_and_finalize()
