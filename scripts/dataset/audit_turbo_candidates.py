import csv
import json
import re
from pathlib import Path
import cv2

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
TURBO_DIR = PROJECT_ROOT / "test_output" / "turbo_1a_verified"
JSON_PATH = TURBO_DIR / "candidates_metadata.json"
META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")

with open(META_PATH, "r", encoding="utf-8") as f:
    existing_plates = {r["plate_num"] for r in csv.DictReader(f, delimiter=";")}

with open(JSON_PATH, "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Total in turbo_1a_verified: {len(candidates)}")
print(f"Existing plates in meta: {len(existing_plates)}")

valid_fresh = []
seen_texts = set()

for c in candidates:
    txt = c["text"]
    if txt in existing_plates or txt in seen_texts:
        continue
    if not PLATE_REGEX.match(txt):
        continue
    if c.get("conf_1a", 0) < 0.90:
        continue
    ar = c.get("ar", 1.0)
    if not (1.12 <= ar <= 1.85):
        continue
    img_file = TURBO_DIR / c["full_fn"]
    if not img_file.exists():
        continue
    
    seen_texts.add(txt)
    valid_fresh.append(c)

print(f"Pristine, novel candidates with conf_1a >= 0.90: {len(valid_fresh)}")
for v in valid_fresh[:15]:
    print(f"  {v['full_fn']} | {v['text']} | conf_1a={v.get('conf_1a', 0):.3f} | AR={v.get('ar', 0):.2f}")
