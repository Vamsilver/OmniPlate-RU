import csv
from pathlib import Path

META_PATH = Path(r"D:\AIProjects\VolgaIT\dataset\meta.csv")

rows = []
with open(META_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    fieldnames = reader.fieldnames
    for r in reader:
        # Check if row has bracket formatting
        bbox = r["bbox"]
        if bbox.startswith("[") and bbox.endswith("]"):
            nums = [int(float(x.strip())) for x in bbox[1:-1].split(",")]
            r["bbox"] = ",".join(str(x) for x in nums)

        quad = r["quad"]
        if quad.startswith("[") and quad.endswith("]"):
            nums = [int(round(float(x.strip()))) for x in quad[1:-1].split(",")]
            r["quad"] = ",".join(str(x) for x in nums)

        conds = r["conditions"].split(",")
        valid_conds = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}
        cleaned_conds = [c.strip() for c in conds if c.strip() in valid_conds]
        if not cleaned_conds:
            cleaned_conds = ["day", "angle"]
        r["conditions"] = ",".join(cleaned_conds)

        rows.append(r)

with open(META_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, delimiter=";", fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Fixed formatting for {len(rows)} rows in {META_PATH}")
