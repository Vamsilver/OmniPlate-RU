import csv
from pathlib import Path

META_PATH = Path(r"D:\AIProjects\VolgaIT\dataset\meta.csv")

# Parse user ranges and list of suspects
suspect_files = set()

# Single files
singles = [
    "real_type1_0184", "real_type1_0190", "real_type1_0191", "real_type1_0192",
    "real_type1_0226", "real_type1_0227", "real_type1_0228", "real_type1_0310",
    "real_type1_0328", "real_type1_0329", "real_type1_0330", "real_type1_0332",
    "real_type1_0351", "real_type1_0354", "real_type1_0355", "real_type1_0360",
    "real_type1_0369", "real_type1_0370", "real_type1_0371", "real_type1_0379",
    "real_type1_0381", "real_type1_0388", "real_type1_0395", "real_type1_0396",
    "real_type1_0397", "real_type1_0398", "real_type1_0399", "real_type1_0400",
    "real_type1_0401", "real_type1_0402", "real_type1_0403", "real_type1_0404",
    "real_type1_0406", "real_type1_0407", "real_type1_0408", "real_type1_0409",
    "real_type1_0412", "real_type1_0419", "real_type1_0420", "real_type1_0421",
    "real_type1_0422", "real_type1_0423", "real_type1_0424", "real_type1_0425",
    "real_type1_0426", "real_type1_0427", "real_type1_0428", "real_type1_0429",
    "real_type1_0430", "real_type1_0432", "real_type1_0433", "real_type1_0434",
    "real_type1_0435", "real_type1_0436", "real_type1_0437", "real_type1_0438",
    "real_type1_0439", "real_type1_0441", "real_type1_0442", "real_type1_0443",
    "real_type1_0444", "real_type1_0445", "real_type1_0446", "real_type1_0447",
    "real_type1_0529", "real_type1_0530", "real_type1_0531", "real_type1_0534",
    "real_type1_0553", "real_type1_0556", "real_type1_0557", "real_type1_0573",
    "real_type1_0575", "real_type1_0576", "real_type1_0577", "real_type1_0862",
    "real_type1_0864"
]
for s in singles:
    suspect_files.add(f"images/real/{s}.jpg")

# Ranges
def add_range(start, end):
    for i in range(start, end + 1):
        suspect_files.add(f"images/real/real_type1_{i:04d}.jpg")

add_range(448, 506)
add_range(509, 519)
add_range(538, 543)
add_range(588, 593)

print(f"Total suspects in list: {len(suspect_files)}")

# Audit from meta.csv
found_records = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter=";"):
        if r["image"] in suspect_files:
            found_records.append(r)

print(f"Matched in meta.csv: {len(found_records)}")
sources = {}
for r in found_records:
    src_prefix = r["source"][:35]
    sources[src_prefix] = sources.get(src_prefix, 0) + 1

for s, cnt in sorted(sources.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cnt} шт. <- {s}")
