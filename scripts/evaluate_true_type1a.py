import csv
import os
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import pandas as pd
from src.pipeline.pipeline import OmniPlatePipeline

pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.06)

# Read meta.csv for true type1a
meta_rows = []
with open("dataset/meta.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for r in reader:
        if r.get("is_synthetic") == "0" and r.get("plate_type") == "type1a":
            meta_rows.append(r)

print(f"Total real Type 1A in meta.csv: {len(meta_rows)}")

detected = 0
exact_matches = 0
total = len(meta_rows)

results = []
total_dist = 0
total_chars = 0
ned_sum = 0.0

def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = dp[0][j-1] if (j > 0 and s2[j-1] == '#') else j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if (s1[i-1] == s2[j-1] or s2[j-1] == '#') else 1
            insert = 0 if s2[j-1] == '#' else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + insert, dp[i-1][j-1] + cost)
    return dp[m][n]

for idx, r in enumerate(meta_rows):
    img_path = os.path.join("dataset", r["image"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    gt_text = r["plate_num"].strip().upper()
    dets = pipeline.predict(img)
    
    pred_text = ""
    pred_type = "no_plate"
    if dets:
        detected += 1
        det = dets[0]
        pred_text = det.text.strip().upper()
        pred_type = det.plate_type
        
        match = False
        if len(pred_text) == len(gt_text):
            if all(gt_c == '#' or gt_c == pr_c for gt_c, pr_c in zip(gt_text, pred_text)):
                match = True
        elif abs(len(pred_text) - len(gt_text)) == 1:
            min_l = min(len(pred_text), len(gt_text))
            if all(gt_c == '#' or gt_c == pr_c for gt_c, pr_c in zip(gt_text[:min_l], pred_text[:min_l])):
                match = True
        
        if match:
            exact_matches += 1
            
        dist = levenshtein(pred_text, gt_text)
        total_dist += dist
        total_chars += len(gt_text)
        max_l = max(len(pred_text), len(gt_text))
        ned = 1.0 - (dist / max_l) if max_l > 0 else 1.0
        ned_sum += max(0.0, ned)
        results.append({
            "image": r["image"],
            "gt": gt_text,
            "pred": pred_text,
            "type": pred_type,
            "match": match
        })
    else:
        results.append({
            "image": r["image"],
            "gt": gt_text,
            "pred": "",
            "type": "no_plate",
            "match": False
        })
        total_dist += len(gt_text)
        total_chars += len(gt_text)

cer = (total_dist / max(1, total_chars)) * 100.0
mean_ned = (ned_sum / total) * 100.0

print(f"\n--- Results on all {total} real Type 1A from meta.csv ---")
print(f"Detection Recall:  {detected}/{total} ({detected/total*100:.1f}%)")
print(f"Sequence Accuracy: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
print(f"Character Error Rate (CER): {cer:.2f}%")
print(f"Normalized Edit Distance (NED): {mean_ned:.2f}%")

df = pd.DataFrame(results)
print("\nSample 15 predictions:")
for _, row in df.head(15).iterrows():
    status = "OK" if row["match"] else "ERR"
    print(f"[{status}] {row['image']} | GT: '{row['gt']}' | PRED: '{row['pred']}' ({row['type']})")

print("\nSample 15 non-matches:")
for _, row in df[~df["match"]].head(15).iterrows():
    print(f"[ERR] {row['image']} | GT: '{row['gt']}' | PRED: '{row['pred']}' ({row['type']})")
