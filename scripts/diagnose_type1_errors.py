import csv, os, sys, time
from collections import Counter, defaultdict
import cv2, numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark_full_dataset import bbox_iou, is_seq_match, levenshtein_dist, normalize_plate
from src.pipeline.pipeline import OmniPlatePipeline

def main():
    print("=" * 65)
    print("DIAGNOSING TYPE 1 ERRORS (Real Images)")
    print("=" * 65)

    meta_path = os.path.join(PROJECT_ROOT, "dataset", "meta.csv")
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter=";"))
        data = rows[1:]

    t1_rows = [r for r in data if r[2] == "type1" and r[6] == "0"]
    print(f"[*] Total real Type 1 images to analyze: {len(t1_rows)}")

    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(1)

    stats = {
        "total": len(t1_rows),
        "detected": 0,
        "exact_matches": 0,
        "missed_detection": 0,
        "verifier_rejected": 0,
        "char_confusions": Counter(),
        "region_mismatches": 0,
        "wildcard_mismatches": 0,
        "length_mismatches": 0,
    }

    detailed_errors = []

    t0 = time.time()
    for idx, row in enumerate(t1_rows):
        img_rel, gt_num, p_type, bbox_str, quad_str = row[:5]
        gt_norm = normalize_plate(gt_num)
        img_full = os.path.join(PROJECT_ROOT, "dataset", img_rel)
        fname = os.path.basename(img_rel)

        img = cv2.imread(img_full)
        if img is None:
            continue

        dets = pipeline.detect(img)
        if not dets:
            stats["missed_detection"] += 1
            detailed_errors.append((fname, gt_norm, "<NO_DET>", "missed_detection"))
            continue

        stats["detected"] += 1

        gt_bbox = None
        if bbox_str:
            parts = [int(float(x)) for x in bbox_str.split(",")]
            if len(parts) == 4:
                gt_bbox = tuple(parts)

        best_det = None
        best_iou = -1.0
        for d in dets:
            iou = bbox_iou(d.bbox, gt_bbox) if gt_bbox else 0.5
            if iou > best_iou:
                best_iou = iou
                best_det = d

        det = pipeline.recognize_single(img, best_det)
        pred_norm = normalize_plate(det.text)

        if is_seq_match(pred_norm, gt_norm):
            stats["exact_matches"] += 1
        else:
            rect = pipeline.rectifier.rectify(img, best_det.quad, plate_type="type1", margin=(0.010, 0.010), refine_corners=False)
            is_p, p_sc = pipeline.verifier.verify_single(rect)

            err_reason = "ocr_mismatch"
            if not is_p or p_sc < 0.40:
                stats["verifier_rejected"] += 1
                err_reason = f"verifier_rejected(p_sc={p_sc:.2f})"

            if len(pred_norm) != len(gt_norm):
                stats["length_mismatches"] += 1
            else:
                for cp, ct in zip(pred_norm, gt_norm):
                    if cp != ct and ct != "#":
                        stats["char_confusions"][(ct, cp)] += 1

            if len(gt_norm) >= 8 and len(pred_norm) >= 8:
                gt_reg = gt_norm[6:]
                pred_reg = pred_norm[6:]
                if gt_reg != pred_reg and "#" not in gt_reg:
                    stats["region_mismatches"] += 1

            if "#" in gt_norm or "#" in pred_norm:
                stats["wildcard_mismatches"] += 1

            detailed_errors.append((fname, gt_norm, pred_norm, err_reason))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(t1_rows):
            acc = stats["exact_matches"] / (idx + 1) * 100.0
            print(f"  [{idx + 1}/{len(t1_rows)}] Exact matches so far: {stats['exact_matches']} ({acc:.2f}%) in {time.time() - t0:.1f}s")

    print("\n" + "=" * 65)
    print("📊 TYPE 1 ERROR DIAGNOSIS SUMMARY")
    print("=" * 65)
    print(f"Total images:         {stats['total']}")
    print(f"Detected:             {stats['detected']} ({stats['detected']/stats['total']*100:.2f}%)")
    print(f"Exact Matches:        {stats['exact_matches']} ({stats['exact_matches']/stats['total']*100:.2f}%)")
    print(f"Missed Detections:    {stats['missed_detection']}")
    print(f"Verifier Rejected:    {stats['verifier_rejected']}")
    print(f"Length Mismatches:    {stats['length_mismatches']}")
    print(f"Region Mismatches:    {stats['region_mismatches']}")
    print(f"Wildcard Mismatches:  {stats['wildcard_mismatches']}")

    print("\nTop 15 Character Confusions (GT -> Pred):")
    for pair, cnt in stats["char_confusions"].most_common(15):
        print(f"  {pair[0]} -> {pair[1]}: {cnt}")

    print("\nSample 25 Mismatches:")
    for err in detailed_errors[:25]:
        print(f"  {err[0]:<22} | GT: {err[1]:<10} | PRED: {err[2]:<10} | Reason: {err[3]}")

if __name__ == "__main__":
    main()
