import csv
from pathlib import Path

with open('dataset/meta.csv', 'r', encoding='utf-8') as f:
    rows = {r['image']: r for r in csv.DictReader(f, delimiter=';')}

with open('test_output/visual_audit/audit_meta_report.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in audit_meta_report: {len(lines)}")
