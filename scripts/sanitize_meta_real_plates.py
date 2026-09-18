#!/usr/bin/env python3
import csv
import os
import re
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
META_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'meta.csv')
BACKUP_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'meta.csv.bak_iter2')
AUDIT_PATH = os.path.join(PROJECT_ROOT, 'test_output', 'omniplate_ground_truth_audit.csv')
OCR_RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results_new_ocr.csv')

CYR_TO_LAT = {
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
    'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X',
}

PLATE_REGEX_TYPE1 = re.compile(r'^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$')
PLATE_REGEX_TYPE1B = re.compile(r'^[ABEKMHOPCTYX#]{2}[\d#]{3}[\d#]{2,3}$')


def normalize_plate(raw: str) -> str:
    if not raw:
        return ''
    s = raw.strip().upper()
    out = []
    for c in s:
        if c in CYR_TO_LAT:
            out.append(CYR_TO_LAT[c])
        elif c.isalnum() or c == '#':
            out.append(c)
    return ''.join(out)


def main():
    print('=' * 65)
    print('OmniPlate-RU - Sanitizing Real Plate Markup in meta.csv')
    print('=' * 65)

    if not os.path.exists(BACKUP_PATH):
        print(f'[*] Creating backup: {BACKUP_PATH}')
        shutil.copyfile(META_PATH, BACKUP_PATH)
    else:
        print(f'[*] Backup already exists: {BACKUP_PATH}')

    audit_gt = {}
    if os.path.exists(AUDIT_PATH):
        with open(AUDIT_PATH, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader, None)
            for row in reader:
                if len(row) >= 6:
                    fn = os.path.basename(row[1].strip())
                    gt = normalize_plate(row[5].strip())
                    if gt and 'НОМЕР' not in gt and len(gt) >= 6:
                        audit_gt[fn] = gt

    print(f'[*] Loaded {len(audit_gt)} verified plates from audit CSV')

    ocr_preds = {}
    if os.path.exists(OCR_RESULTS_PATH):
        with open(OCR_RESULTS_PATH, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    fn = os.path.basename(row[0].strip())
                    text = normalize_plate(row[1].strip())
                    conf = float(row[3]) if row[3] else 0.0
                    if fn not in ocr_preds or conf > ocr_preds[fn][1]:
                        ocr_preds[fn] = (text, conf)

    with open(META_PATH, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        rows = list(reader)

    print(f'[*] Total rows in meta.csv: {len(rows)}')

    updated_t1b = 0
    updated_t1 = 0
    kept_as_is = 0

    new_rows = []
    for row in rows:
        if len(row) < 10:
            new_rows.append(row)
            continue

        img_rel, plate_num, p_type, bbox_str, quad_str, is_veh_str, is_syn_str, source, lic, conds_str = row
        fn = os.path.basename(img_rel)
        is_real = (is_syn_str == '0')

        if is_real and p_type == 'type1b':
            target_num = None
            if fn in audit_gt:
                cand = audit_gt[fn]
                if PLATE_REGEX_TYPE1B.match(cand) or PLATE_REGEX_TYPE1.match(cand):
                    target_num = cand
            if target_num is None and fn in ocr_preds:
                cand = ocr_preds[fn][0]
                if PLATE_REGEX_TYPE1B.match(cand) or PLATE_REGEX_TYPE1.match(cand):
                    target_num = cand

            if target_num and target_num != plate_num:
                row[1] = target_num
                updated_t1b += 1
            else:
                kept_as_is += 1

        elif is_real and p_type == 'type1':
            target_num = None
            if fn in audit_gt:
                cand = audit_gt[fn]
                if PLATE_REGEX_TYPE1.match(cand):
                    target_num = cand
            if target_num and target_num != plate_num:
                row[1] = target_num
                updated_t1 += 1
            else:
                kept_as_is += 1

        new_rows.append(row)

    print(f'[*] Updated Type 1B rows: {updated_t1b}')
    print(f'[*] Updated Type 1 rows:  {updated_t1}')

    with open(META_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        writer.writerows(new_rows)

    print(f'[*] Successfully saved sanitized meta.csv with {len(new_rows)} rows.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
