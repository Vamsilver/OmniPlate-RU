#!/usr/bin/env python3
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']

def main():
    zip_path = r"C:\Users\vamsi\Downloads\archive (1).zip"
    with zipfile.ZipFile(zip_path, "r") as z:
        txts = [n for n in z.namelist() if n.endswith(".txt") and "train" in n]
        print(f"Total label files in train: {len(txts)}")
        for t in txts[:12]:
            lines = z.read(t).decode("utf-8").strip().split("\n")
            chars = []
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cls_idx = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    chars.append((xc, yc, w, h, CLASSES[cls_idx]))
            chars.sort(key=lambda x: x[0])
            plate_str = "".join([c[4] for c in chars])
            if chars:
                min_x = min(c[0] - c[2] / 2 for c in chars)
                max_x = max(c[0] + c[2] / 2 for c in chars)
                min_y = min(c[1] - c[3] / 2 for c in chars)
                max_y = max(c[1] + c[3] / 2 for c in chars)
                print(f"Plate: '{plate_str:<10}' | {t.split('/')[-1]}")

if __name__ == "__main__":
    main()
