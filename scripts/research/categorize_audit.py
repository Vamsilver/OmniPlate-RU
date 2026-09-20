import csv
from pathlib import Path

img_dir = Path("dataset/images/real")
label_dir = Path("dataset/labels")
meta_path = Path("dataset/meta.csv")

with open(meta_path, "r", encoding="utf-8") as f:
    meta_rows = {r["image"]: r for r in csv.DictReader(f, delimiter=";")}

other_imgs = sorted(list(img_dir.glob("real_other_*.jpg")))

print(f"Total other images: {len(other_imgs)}")

# Let's inspect sources and patterns
categories = {
    "nazi_fascism": [],
    "violence_war_shooting": [],
    "desktop_screenshots": [],
    "synthetic_glyphs": [],
    "non_vehicle_graphics": [],
    "people_dominant": [],
    "russian_plates_to_reclassify": [],
    "trailers_type2": [],
    "diplomatic_red": [],
    "military_black": [],
    "foreign_plates": [],
    "motorcycles_no_plate": [],
    "tractors_special_machinery": [],
    "urban_backgrounds_objects": []
}

# Explicit classifications based on our visual audit of sheets 1-10:
# Nazi/Fascist:
nazi_list = [
    "real_other_0075.jpg", # Wehrmacht WH-489311 with swastika
    "real_other_0155.jpg", # German army uniforms
    "real_other_0156.jpg", # German army uniforms
    "real_other_0157.jpg", # German army uniforms
    "real_other_0158.jpg", # German army insignia
    "real_other_0159.jpg", # German army insignia
    "real_other_0160.jpg", # German army colors
    "real_other_0161.jpg", # Luftwaffe uniforms
    "real_other_0162.jpg", # Luftwaffe uniforms
    "real_other_0163.jpg", # Luftwaffe uniforms
    "real_other_0164.jpg", # Luftwaffe insignia
    "real_other_0165.jpg", # Luftwaffe insignia
    "real_other_0166.jpg", # Luftwaffe colors
    "real_other_0176.jpg", # DDAC swastika badge
    "real_other_0181.jpg", # Reichskommissar swastika flag
    "real_other_0183.jpg", # Hitler Youth flags
    "real_other_0184.jpg", # NSKK insignia
]

# Violence/War/Terror:
violence_list = [
    "real_other_0141.jpg", # Soldier shooting down FPV drone with shotgun
    "real_other_0146.jpg", # Columbine school shooting CCTV
    "real_other_0149.jpg", # Patty Hearst SLA bank raid with guns
    "real_other_0180.jpg", # Kyiv missile strike destruction
]

# People only / political / non-vehicle graphics:
non_vehicle_list = [
    "real_other_0096.jpg", # Mrs Waller in Siberia (group of women, box on coats)
    "real_other_0111.jpg", # Belarus flag graphic clipart
    "real_other_0121.jpg", # Armenia flag graphic clipart
    "real_other_0185.jpg", # Zelensky speech portrait
    "real_other_0196.jpg", # NASA San Andreas fault radar map
    "real_other_0197.jpg", # Alaska BMEWS radar antenna
]

# Synthetic glyphs / roboflow desktop screenshots:
desktop_and_glyphs = [
    "real_other_0211.jpg", # KKKKK
    "real_other_0214.jpg", # XXXXX
    "real_other_0215.jpg", # AAAAA
    "real_other_0217.jpg", # AAAAA
    "real_other_0256.jpg", # Windows desktop taskbar
    "real_other_0257.jpg", # Windows desktop taskbar
    "real_other_0258.jpg", # Windows desktop taskbar
    "real_other_0259.jpg", # Windows desktop Start menu
    "real_other_0264.jpg", # Windows desktop taskbar
    "real_other_0269.jpg", # Windows desktop taskbar
    "real_other_0270.jpg", # Windows desktop taskbar
    "real_other_0271.jpg", # Windows desktop taskbar
]

print("Defined deletion candidate lists.")
