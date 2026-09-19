#!/usr/bin/env python3
"""
scripts/research/evaluate_blind_test_50.py
Computes detailed comparison metrics between Vision Ground Truth and Model Predictions.
Generates test_output/blind_test_50/evaluation_report.json and evaluation_table.csv.
"""

import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT_DIR / "test_output" / "blind_test_50"


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


GT_TABLE = {
    1: ("type1", "B152MM142"),
    2: ("type1", "Y078XA57"),
    3: ("type1", "H312YH57"),
    4: ("type1", "A620AC57"),
    5: ("type1", "O165HC58"),
    6: ("type1", "A119AO777"),
    7: ("type1", "A793EO197"),
    8: ("type1", "K226OX68"),
    9: ("type1", "A004MA37"),
    10: ("type1", "A922AK150"),
    11: ("type1", "C400PO25"),
    12: ("type1", "P425BB68"),
    13: ("type1", "E585XY68"),
    14: ("type1", "E534TH197"),
    15: ("type1", "B993OX199"),
    16: ("type1a", "X728HP152"),
    17: ("type1a", "O475BT797"),
    18: ("type1a", "A676BH777"),
    19: ("type1a", "X604BO193"),
    20: ("type1a", "H100AO61"),
    21: ("type1a", "M247PP28"),
    22: ("type1a", "O402HC790"),
    23: ("type1a", "E313YO79"),
    24: ("type1a", "K147HM70"),
    25: ("type1a", "Y452AT198"),
    26: ("type1a", "P333OX24"),
    27: ("type1a", "X745PK152"),
    28: ("type1a", "P093OC24"),
    29: ("type1a", "Y015TX193"),
    30: ("type1a", "O432XA38"),
    31: ("type1b", ""),
    32: ("type1b", ""),
    33: ("type1b", ""),
    34: ("type1b", ""),
    35: ("type1b", ""),
    36: ("type1b", ""),
    37: ("type1b", ""),
    38: ("type1b", ""),
    39: ("type1b", ""),
    40: ("type1b", "EM96177"),
    41: ("other", "XX584457"),
    42: ("type1", "E053BH48"),
    43: ("other", ""),
    44: ("other", ""),
    45: ("other", ""),
    46: ("other", ""),
    47: ("other", ""),
    48: ("other", ""),
    49: ("other", ""),
    50: ("other", ""),
}


def main():
    manifest_path = OUT_DIR / "manifest_50.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    rows = []
    total_samples = len(preds)
    total_exact = 0
    total_dist = 0
    total_gt_chars = 0

    by_type = {}

    for p in preds:
        idx = p["idx"]
        gt_type, gt_text = GT_TABLE[idx]
        pred_text = p["pred_text"].strip()
        pred_type = p["pred_type"].strip()

        if gt_text == "":
            em = (pred_text == "")
            dist = 0 if em else len(pred_text)
            cer = 0.0 if em else 1.0
            ned = 1.0 if em else 0.0
        else:
            type_match = (pred_type == gt_type) or (gt_type in ("type1", "type2") and pred_type in ("type1", "type2"))
            em = (pred_text == gt_text) and type_match
            dist = levenshtein_distance(pred_text, gt_text)
            cer = dist / max(len(gt_text), 1)
            ned = 1.0 - (dist / max(len(gt_text), len(pred_text), 1))
            total_dist += dist
            total_gt_chars += len(gt_text)

        if em:
            total_exact += 1

        group = gt_type if gt_text else "negative_other"
        if group not in by_type:
            by_type[group] = {"count": 0, "exact": 0, "dist": 0, "chars": 0}
        by_type[group]["count"] += 1
        if em:
            by_type[group]["exact"] += 1
        if gt_text:
            by_type[group]["dist"] += dist
            by_type[group]["chars"] += len(gt_text)

        rows.append({
            "idx": idx,
            "filename": p["filename"],
            "target_type": p["target_type"],
            "gt_type": gt_type,
            "gt_text": gt_text,
            "pred_type": pred_type,
            "pred_text": pred_text,
            "conf": p["pred_conf"],
            "ocr_conf": p["ocr_conf"],
            "exact_match": em,
            "cer": round(cer, 4),
            "ned": round(ned, 4),
            "latency_ms": p["latency_ms"],
            "source": p["source"]
        })

    # Save CSV
    csv_path = OUT_DIR / "evaluation_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "filename", "target_type", "gt_type", "gt_text",
            "pred_type", "pred_text", "conf", "ocr_conf", "exact_match", "cer", "ned", "latency_ms"
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    overall_seq_acc = total_exact / total_samples
    overall_cer = (total_dist / total_gt_chars) if total_gt_chars > 0 else 0.0

    report = {
        "total_images": total_samples,
        "exact_match_total": total_exact,
        "sequence_accuracy": round(overall_seq_acc, 4),
        "total_edit_distance": total_dist,
        "total_gt_chars": total_gt_chars,
        "overall_cer": round(overall_cer, 4),
        "by_class": {}
    }

    print("=" * 70)
    print("  OmniPlate-RU Blind Test 50 Evaluation Summary")
    print("=" * 70)
    print(f"Total Evaluated Images : {total_samples}")
    print(f"Exact Matches (Overall): {total_exact} / {total_samples} ({overall_seq_acc * 100:.2f}%)")
    print(f"Total Character Errors : {total_dist} / {total_gt_chars} (CER: {overall_cer * 100:.2f}%)")
    print("\nClass Breakdown:")
    for grp, s in sorted(by_type.items()):
        acc = s["exact"] / s["count"] * 100
        cer_val = (s["dist"] / s["chars"] * 100) if s["chars"] > 0 else 0.0
        report["by_class"][grp] = {
            "count": s["count"],
            "exact": s["exact"],
            "accuracy": round(acc, 2),
            "cer": round(cer_val, 2)
        }
        print(f"  • {grp:<16}: Acc={acc:6.2f}% ({s['exact']:>2}/{s['count']:<2}) | CER={cer_val:5.2f}%")

    json_path = OUT_DIR / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved detailed evaluation CSV to {csv_path}")
    print(f"[+] Saved evaluation report JSON to {json_path}")


if __name__ == "__main__":
    main()
