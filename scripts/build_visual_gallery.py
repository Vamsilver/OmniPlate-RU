#!/usr/bin/env python3
"""
OmniPlate-RU — Interactive Visual Audit Gallery with Yes/No Feedback & Ground Truth Correction.
Generates an interactive HTML audit suite (test_output/gallery.html) featuring:
- Grid mode & Smooth full-screen scrolling audit mode
- Yes / No buttons ("ДА, ВЕРНО" / "НЕТ, ОШИБКА")
- In-place Ground Truth correction field ("Истинный номер: E686XH199")
- Real-time statistics & progress counter
- LocalStorage auto-save & CSV export for fine-tuning
"""

import argparse
import base64
import html
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline


def image_to_base64_thumbnail(img_bgr: np.ndarray, max_dim: int = 540) -> str:
    h, w = img_bgr.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_audited_registry(project_root: Path):
    """
    Loads all audited images from test_output/omniplate_ground_truth_audit.csv
    and syncs with test_output/audited_registry.json.
    Normalizes filenames to lower-case base filenames to strictly prevent duplicates.
    """
    import csv
    import json

    csv_path = project_root / "test_output" / "omniplate_ground_truth_audit.csv"
    reg_path = project_root / "test_output" / "audited_registry.json"
    img_dir = project_root / "dataset" / "images" / "real"

    all_real_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.exists() else []
    total_real = len(all_real_files) if all_real_files else 531

    audited_map = {}
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=";")
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 6 and row[1]:
                        fn = Path(row[1].strip()).name.lower()
                        status = row[4].strip()
                        if status != "unreviewed":
                            audited_map[fn] = {
                                "filename": Path(row[1].strip()).name,
                                "type": row[2].strip() if len(row) > 2 else "",
                                "pred": row[3].strip() if len(row) > 3 else "",
                                "status": status,
                                "gt": row[5].strip() if len(row) > 5 else "",
                            }
        except Exception as e:
            print(f"[!] Warning reading {csv_path.name}: {e}")

    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                r_data = json.load(f)
                for fn in r_data.get("audited_filenames", []):
                    audited_map.setdefault(Path(fn).name.lower(), {"filename": Path(fn).name, "status": "yes"})
        except Exception:
            pass

    audited_count = len(audited_map)
    unseen_count = max(0, total_real - audited_count)
    progress_pct = round((audited_count / max(1, total_real)) * 100.0, 1)

    reg_info = {
        "total_real_images": total_real,
        "audited_count": audited_count,
        "unseen_count": unseen_count,
        "progress_percent": progress_pct,
        "audited_filenames": sorted(list(audited_map.keys())),
    }

    try:
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg_info, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return set(audited_map.keys()), total_real, audited_count, reg_info


def sample_candidates(unseen_files: list, limit: int = 45, do_shuffle: bool = True) -> list:
    """
    Performs balanced category sampling among unseen real images.
    Guarantees no duplicates and dynamically redistributes deficits across pools.
    """
    import random

    p_1b = [f for f in unseen_files if "type1b" in f.name.lower()]
    p_1a = [f for f in unseen_files if "type1a" in f.name.lower() or "1a" in f.name.lower() or "type1a" in str(f.parent).lower()]
    p_t2 = [f for f in unseen_files if "type2" in f.name.lower() or "trailer" in f.name.lower() or "cand_t2" in f.name.lower() or "type2" in str(f.parent).lower()]
    p_other = [f for f in unseen_files if "other" in f.name.lower()]
    p_ref = [f for f in unseen_files if "ref_real" in f.name.lower()]
    categorized = set(p_1b + p_1a + p_t2 + p_other + p_ref)
    p_rest = [f for f in unseen_files if f not in categorized]

    if do_shuffle:
        for p in (p_1b, p_1a, p_t2, p_other, p_ref, p_rest):
            random.shuffle(p)

    target_quotas = [
        ("1a", p_1a, 25),
        ("t2", p_t2, 25),
        ("1b", p_1b, 15),
        ("other", p_other, 10),
        ("ref", p_ref, 5),
    ]

    selected = []
    selected_set = set()

    for _, pool, quota in target_quotas:
        take = min(quota, len(pool))
        for item in pool[:take]:
            selected.append(item)
            selected_set.add(item)

    needed = min(limit, len(unseen_files)) - len(selected)
    if needed > 0:
        remaining = [f for f in unseen_files if f not in selected_set]
        if do_shuffle:
            random.shuffle(remaining)
        selected.extend(remaining[:needed])

    return selected[:limit]


def main():
    parser = argparse.ArgumentParser(description="OmniPlate Interactive Audit Gallery")
    parser.add_argument("--input_dir", type=str, default="dataset/images/real", help="Directory with input images")
    parser.add_argument("--output_html", type=str, default="test_output/gallery.html", help="Path to output HTML")
    parser.add_argument("--limit", type=int, default=45, help="Max images to include in gallery")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device ('cuda' or 'cpu')")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--filter_type", type=str, default=None, help="Filter input by filename substring")
    parser.add_argument("--include_audited", action="store_true", help="Include already audited images")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=True, help="Randomly shuffle candidates (default: True)")
    parser.add_argument("--no_shuffle", dest="shuffle", action="store_false", help="Disable random shuffle")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling")
    parser.add_argument("--ocr_model", type=str, default=None, help="Explicit path to OCR model (.pt or .onnx)")
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    output_html = PROJECT_ROOT / args.output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        import random
        random.seed(args.seed)

    # 1. Load Audited Registry and strict exclusion list
    audited_files, total_real, audited_count, _ = load_audited_registry(PROJECT_ROOT)

    all_real_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))) if input_dir.exists() else []
    if not all_real_files and input_dir.exists():
        all_real_files = sorted(list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png")))
    all_real_files = [f for f in all_real_files if not f.name.startswith(".")]
    if args.include_audited:
        candidate_pool = all_real_files
    else:
        candidate_pool = [f for f in all_real_files if f.name.lower() not in audited_files]

    progress_pct = (audited_count / max(1, total_real)) * 100.0

    print("=" * 70)
    print("OmniPlate-RU — Генерация интерактивной галереи аудита")
    print("=" * 70)
    print(f"[*] Всего реальных кадров на диске: {total_real}")
    print(f"[*] Уже размечено / отсмотрено:     {audited_count} ({progress_pct:.1f}%)")
    print(f"[*] Доступно новых (Unseen):        {len(candidate_pool)}")

    if not candidate_pool:
        print("[!] Все реальные кадры уже отсмотрены! Загрузка полного пула для ревизии.")
        candidate_pool = all_real_files

    if args.filter_type:
        filtered = [f for f in candidate_pool if args.filter_type.lower() in f.name.lower()]
        if args.shuffle:
            import random
            random.shuffle(filtered)
        sample_files = filtered[:args.limit]
    else:
        sample_files = sample_candidates(candidate_pool, limit=args.limit, do_shuffle=args.shuffle)

    print(f"[*] Выбрано для текущей пачки:      {len(sample_files)} уникальных кадров")
    print("=" * 70)

    pipeline = OmniPlatePipeline(
        device=args.device,
        conf_threshold=args.conf,
        ocr_path=args.ocr_model,
    )
    pipeline.warmup(iterations=2)

    ocr_path_obj = Path(pipeline.ocr_path) if pipeline.ocr_path else None
    ocr_name = ocr_path_obj.name if ocr_path_obj else "Default"
    if ocr_path_obj and ocr_path_obj.exists():
        ocr_mtime_str = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(os.path.getmtime(ocr_path_obj)))
    else:
        ocr_mtime_str = "N/A"

    det_name = Path(pipeline.detector_path).name if pipeline.detector_path else "Default"
    batch_time_str = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime())
    batch_time_short = time.strftime("%H:%M:%S", time.localtime())

    print(f"[*] Активная модель OCR:      {ocr_name} (обновлена: {ocr_mtime_str})")
    print(f"[*] Детектор YOLO-Pose:       {det_name}")
    print(f"[*] Таймстемп пачки:          {batch_time_str}")
    print("=" * 70)

    cards_html = []
    total_time = 0.0

    for idx, fpath in enumerate(sample_files, start=1):
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        t0 = time.perf_counter()
        dets = pipeline.predict(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_time += dt_ms

        vis_scene = img.copy()
        crop_b64 = ""
        p_type = "no_plate"
        pred_text = "НОМЕР НЕ НАЙДЕН"
        conf_val = 0.0

        if dets:
            det = dets[0]
            p_type = det.plate_type
            pred_text = det.text if det.text else ("(ПЛАСТИНА БЕЗ ТЕКСТА)" if p_type != "other" else "NEGATIVE (OTHER)")
            conf_val = det.confidence

            color_map = {
                "type1": (0, 255, 0),
                "type1a": (255, 200, 0),
                "type1b": (0, 215, 255),
                "type2": (255, 100, 200),
                "other": (0, 0, 255),
            }
            color = color_map.get(p_type, (0, 255, 0))
            bx, by, bw, bh = det.bbox
            cv2.rectangle(vis_scene, (bx, by), (bx + bw, by + bh), color, 3)

            if det.quad and len(det.quad) == 8:
                pts = np.array(det.quad, dtype=np.int32).reshape(4, 2)
                cv2.polylines(vis_scene, [pts], True, (255, 255, 255), 2)

            if det.rectified_crop is not None and det.rectified_crop.size > 0:
                crop_b64 = image_to_base64_thumbnail(det.rectified_crop, max_dim=260)

        scene_b64 = image_to_base64_thumbnail(vis_scene, max_dim=540)
        badge_map = {
            "type1": "#10b981",
            "type1a": "#3b82f6",
            "type1b": "#f59e0b",
            "type2": "#a855f7",
            "other": "#a855f7",
            "no_plate": "#ef4444",
        }
        badge_color = badge_map.get(p_type, "#6b7280")

        init_input_val = pred_text if p_type != "no_plate" and not pred_text.startswith("(") else ""

        type_options = [
            ("type1", "Type 1", "Белый легковой 1-строчный (A000AA77)", "#10b981", "🚗"),
            ("type1a", "Type 1A", "Квадрат 2-строчный (JDM/USDM)", "#3b82f6", "🟦"),
            ("type1b", "Type 1B", "Желтый автобус/такси (AA00077)", "#f59e0b", "🚌"),
            ("other", "Other", "Прицеп (Тип 2) / Мото / Дипломат / Спецномер", "#a855f7", "🏷️"),
            ("no_plate", "Брак", "Ложная детекция / Мусор / Нет номера", "#ef4444", "🚫"),
        ]
        options_html = "".join(
            f'<option value="{opt_val}" {"selected" if opt_val == p_type else ""}>{opt_icon} {opt_title} ({opt_desc})</option>'
            for opt_val, opt_title, opt_desc, opt_col, opt_icon in type_options
        )

        chips_html = "".join(
            f'''<button type="button" class="type-chip {'active' if opt_val == p_type else ''}" id="chip-{idx}-{opt_val}" onclick="selectTypeDirect({idx}, '{opt_val}')" title="{opt_desc}" style="--chip-col:{opt_col};">
                <span class="chip-icon">{opt_icon}</span>
                <span class="chip-name">{opt_title}</span>
            </button>'''
            for opt_val, opt_title, opt_desc, opt_col, opt_icon in type_options
        )

        curr_opt = next((x for x in type_options if x[0] == p_type), type_options[0])

        crop_block = f'''
        <div class="crop-box-wrap">
            <img class="crop-img" src="{crop_b64}" alt="Warped Crop" />
        </div>
        ''' if crop_b64 else '''
        <div class="crop-box-wrap no-crop-box">
            <span style="color:#64748b; font-size:12px;">(Нет кропа)</span>
        </div>
        '''

        fpath_disp = f"{fpath.parent.name}/{fpath.name}" if fpath.parent.name in ("type1a", "type2") else fpath.name
        card = f"""
        <div class="card" id="card-{idx}" data-idx="{idx}" data-type="{html.escape(p_type)}" data-filename="{html.escape(fpath_disp)}" data-pred="{html.escape(pred_text)}">
            <div class="card-header">
                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                    <span class="card-num">#{idx} / {len(sample_files)}</span>
                    <span class="badge" style="background-color: {badge_color};">{curr_opt[4]} {html.escape(p_type.upper())}</span>
                    <span class="fname" title="Точное имя файла">📄 {html.escape(fpath_disp)}</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span class="card-time-tag">🕒 {batch_time_short}</span>
                    <span class="time">⚡ {dt_ms:.1f} ms</span>
                </div>
            </div>

            <div class="card-body">
                <div class="scene-col">
                    <div class="scene-wrap">
                        <img class="thumb" src="{scene_b64}" alt="Scene" loading="lazy" />
                    </div>
                </div>

                <div class="cockpit-col">
                    <div class="crop-section">
                        <div class="cockpit-label-row">
                            <span class="cockpit-sublabel">ВАРПНУТЫЙ КРОП:</span>
                            <span class="crop-type-hint" id="type-badge-hint-{idx}" style="color: {badge_color}; font-weight: 700; font-size: 11px;">{curr_opt[4]} {curr_opt[1]}: {curr_opt[2]}</span>
                        </div>
                        {crop_block}
                    </div>

                    <div class="pred-section">
                        <div>
                            <span class="cockpit-sublabel">РАСПОЗНАННЫЙ ТЕКСТ:</span>
                            <div class="pred-text" id="pred-display-{idx}">{html.escape(pred_text)}</div>
                        </div>
                        <div class="conf-box">
                            <span class="conf-label">Conf: <b>{conf_val:.2f}</b></span>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar" style="width: {min(100, int(conf_val * 100))}%;"></div>
                            </div>
                        </div>
                    </div>

                    <div class="type-selector-box">
                        <div class="type-selector-header">
                            <span class="cockpit-sublabel">ТИП НОМЕРА (КЛИК ДЛЯ СМЕНЫ):</span>
                            <span class="type-curr-label" id="type-curr-label-{idx}" style="color:{badge_color}; font-weight:bold; font-size:11px;">{curr_opt[4]} {curr_opt[1]}</span>
                        </div>
                        <div class="type-chips-grid" id="type-chips-{idx}">
                            {chips_html}
                        </div>
                    </div>

                    <div class="vote-section">
                        <button class="btn btn-yes" onclick="vote({idx}, 'yes')" title="Горячая клавиша: Y">✅ ОДОБРИТЬ GT (Y)</button>
                        <button class="btn btn-no" onclick="vote({idx}, 'no')" title="Горячая клавиша: N">🗑️ БРАК / НЕТ НОМЕРА (N)</button>
                    </div>

                    <div class="audit-status-bar">
                        <div class="audit-status" id="status-{idx}">Не проверено</div>
                    </div>

                    <div class="edit-section">
                        <div class="edit-row-header">
                            <span style="font-size:11px; color:#38bdf8; font-weight:700;">✏️ УТОЧНИТЬ ТЕКСТ GT:</span>
                            <div style="display:flex; gap:4px;">
                                <button type="button" class="btn-quick-wildcard" onclick="quickFillWildcard({idx}, 8)" title="Вставить ######## (номер есть, но 8 знаков не читаются)">8x #</button>
                                <button type="button" class="btn-quick-wildcard" onclick="quickFillWildcard({idx}, 9)" title="Вставить ######### (номер есть, но 9 знаков не читаются)">9x #</button>
                            </div>
                            <select class="type-select" id="type-select-{idx}" style="display:none;" onchange="changeCardType({idx})">
                                {options_html}
                            </select>
                        </div>
                        <div class="edit-input-row">
                            <input type="text" class="edit-input" id="edit-{idx}" value="{html.escape(init_input_val)}" placeholder="Номер (E686XH199)..." onkeydown="if(event.key==='Enter') saveCorrection({idx})" />
                            <button class="btn btn-save" onclick="saveCorrection({idx})" title="Enter">💾 Сохранить GT</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

        if idx % 10 == 0 or idx == len(sample_files):
            print(f"[{idx}/{len(sample_files)}] Processed {fpath.name} ({dt_ms:.1f} ms)")

    avg_ms = total_time / max(1, len(sample_files))
    session_id = f"batch_{int(time.time())}"
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniPlate-RU — Аудит номеров и разметка Ground Truth</title>
    <style>
        :root {{
            --bg: #0b1329;
            --surface: #1e293b;
            --surface-hover: #273549;
            --border: #334155;
            --text: #f8fafc;
            --subtext: #94a3b8;
            --primary: #38bdf8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            padding-bottom: 80px;
        }}
        .header {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(11, 19, 41, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 14px 20px;
            margin: -20px -20px 20px -20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .stats-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .stat-pill {{
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
        }}
        .stat-progress {{
            background: rgba(56, 189, 248, 0.15);
            border-color: rgba(56, 189, 248, 0.4);
            color: #e0f2fe;
            font-weight: 600;
        }}
        .stat-model {{
            background: rgba(139, 92, 246, 0.15);
            border-color: rgba(139, 92, 246, 0.4);
            color: #ede9fe;
        }}
        .card-time-tag {{
            font-size: 11px;
            color: var(--subtext);
            font-family: monospace;
            background: #0f172a;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid var(--border);
        }}
        .stat-yes {{ color: var(--success); font-weight: 700; }}
        .stat-no {{ color: var(--danger); font-weight: 700; }}
        .stat-cor {{ color: var(--primary); font-weight: 700; }}
        .controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .filter-btn, .action-btn {{
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 7px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .action-btn:hover {{ background: var(--surface-hover); }}
        .filter-btn.active {{
            background: var(--primary);
            color: #0f172a;
            font-weight: 600;
        }}
        .action-btn.export {{
            background: #0284c7;
            color: #fff;
            border: none;
            font-weight: 700;
        }}

        /* Viewing modes */
        .container.grid-mode {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(640px, 1fr));
            gap: 20px;
        }}
        .container.scroll-mode {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 24px;
            max-width: 960px;
            margin: 0 auto;
        }}
        .container.scroll-mode .card {{
            width: 100%;
            border-width: 2px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}

        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
        }}
        .card:hover {{
            border-color: rgba(56, 189, 248, 0.45);
        }}
        .card.voted-yes {{
            border-color: var(--success);
            background: linear-gradient(180deg, rgba(16, 185, 129, 0.10) 0%, var(--surface) 100%);
        }}
        .card.voted-no {{
            border-color: var(--danger);
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.10) 0%, var(--surface) 100%);
        }}
        .card.voted-corrected {{
            border-color: var(--primary);
            background: linear-gradient(180deg, rgba(56, 189, 248, 0.13) 0%, var(--surface) 100%);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }}
        .card-num {{
            color: var(--primary);
            font-weight: 700;
            font-family: monospace;
            font-size: 14px;
        }}
        .badge {{
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            color: #fff;
            letter-spacing: 0.5px;
        }}
        .fname {{
            color: var(--subtext);
            font-family: monospace;
            font-size: 12px;
        }}
        .time {{
            color: var(--primary);
            font-family: monospace;
            font-size: 13px;
        }}

        /* 2-Column Cockpit Body Layout */
        .card-body {{
            display: flex;
            gap: 16px;
            align-items: stretch;
        }}

        /* Left column: Scene overview */
        .scene-col {{
            flex: 1 1 50%;
            min-width: 0;
            display: flex;
            flex-direction: column;
        }}
        .scene-wrap {{
            width: 100%;
            height: 100%;
            min-height: 250px;
            background: #090d16;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .thumb {{
            width: 100%;
            height: 100%;
            max-height: 320px;
            object-fit: contain;
            display: block;
        }}

        /* Right column: Inspection Cockpit (Zero eye horizontal drift) */
        .cockpit-col {{
            flex: 0 0 350px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            justify-content: space-between;
        }}

        @media (max-width: 768px) {{
            .card-body {{
                flex-direction: column;
            }}
            .cockpit-col {{
                flex: 1 1 auto;
                width: 100%;
            }}
            .scene-wrap {{
                min-height: 200px;
            }}
        }}

        .cockpit-label-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .cockpit-sublabel {{
            font-size: 10px;
            color: var(--subtext);
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        /* Crop Section */
        .crop-section {{
            display: flex;
            flex-direction: column;
        }}
        .crop-box-wrap {{
            background: #050811;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 72px;
            max-height: 85px;
            overflow: hidden;
            transition: border-color 0.2s;
        }}
        .crop-box-wrap:hover {{
            border-color: #38bdf8;
        }}
        .crop-box-wrap.no-crop-box {{
            background: #0b1329;
            border-style: dashed;
        }}
        .crop-img {{
            max-width: 100%;
            max-height: 68px;
            object-fit: contain;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.6);
            transition: transform 0.15s ease;
        }}
        .crop-box-wrap:hover .crop-img {{
            transform: scale(1.15);
        }}

        /* Pred Section (Directly under Crop) */
        .pred-section {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: #090e1d;
            border-radius: 8px;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }}
        .pred-text {{
            font-size: 24px;
            font-weight: 800;
            font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
            letter-spacing: 2px;
            color: #38bdf8;
            line-height: 1.1;
        }}
        .conf-box {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 3px;
        }}
        .conf-label {{
            font-size: 11px;
            color: var(--subtext);
            font-family: monospace;
        }}
        .conf-bar-wrap {{
            width: 75px;
            background: #1e293b;
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
        }}
        .conf-bar {{
            background: var(--success);
            height: 100%;
        }}

        /* Primary Vote Buttons */
        .vote-section {{
            display: flex;
            gap: 8px;
        }}
        .vote-section .btn {{
            flex: 1;
            padding: 10px 14px;
            font-size: 13px;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        .btn {{
            border: none;
            cursor: pointer;
            font-weight: 700;
            transition: transform 0.1s, opacity 0.2s, background 0.2s;
        }}
        .btn:active {{ transform: scale(0.96); }}
        .btn-yes {{
            background: #059669;
            color: #fff;
        }}
        .btn-yes:hover {{
            background: #10b981;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
        }}
        .btn-no {{
            background: #dc2626;
            color: #fff;
        }}
        .btn-no:hover {{
            background: #ef4444;
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
        }}

        /* Status Bar */
        .audit-status-bar {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 22px;
        }}
        .audit-status {{
            font-size: 11px;
            color: var(--subtext);
            font-weight: 600;
            text-align: center;
        }}

        /* Type Selector Box & Chips */
        .type-selector-box {{
            background: #090e1d;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .type-selector-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .type-chips-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
        }}
        .type-chip {{
            background: #0f172a;
            border: 1px solid #334155;
            color: #cbd5e1;
            padding: 5px 6px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            transition: all 0.15s ease;
        }}
        .type-chip:hover {{
            background: #1e293b;
            border-color: var(--chip-col, #38bdf8);
            color: #fff;
        }}
        .type-chip.active {{
            background: var(--chip-col, #38bdf8);
            border-color: var(--chip-col, #38bdf8);
            color: #0f172a;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
        }}

        /* Quick Edit Section */
        .edit-section {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px dashed var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .edit-row-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .type-select {{
            background: #0b1329;
            color: #f8fafc;
            border: 1px solid var(--border);
            padding: 3px 6px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }}
        .type-select:focus {{
            border-color: var(--primary);
        }}
        .edit-input-row {{
            display: flex;
            gap: 6px;
        }}
        .edit-input {{
            flex: 1;
            background: #090e1d;
            border: 1px solid var(--border);
            color: #f8fafc;
            padding: 6px 10px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
        .edit-input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
        }}
        .btn-save {{
            background: #0284c7;
            color: #fff;
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 6px;
            white-space: nowrap;
        }}
        .btn-save:hover {{
            background: #38bdf8;
            color: #0f172a;
        }}
        .btn-quick-wildcard {{
            background: #1e293b;
            border: 1px solid #475569;
            color: #38bdf8;
            padding: 2px 7px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn-quick-wildcard:hover {{
            background: #0284c7;
            color: #fff;
            border-color: #38bdf8;
        }}

        .hotkeys-hint {{
            position: fixed;
            bottom: 15px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.96);
            border: 1px solid var(--border);
            padding: 8px 20px;
            border-radius: 9999px;
            font-size: 12px;
            color: var(--subtext);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            pointer-events: none;
        }}
        .hotkeys-hint kbd {{
            background: #334155;
            color: #fff;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size: 19px;">OmniPlate-RU — Аудит номеров и разметка Ground Truth</h1>
            <div style="color: var(--subtext); font-size: 12px; margin-top: 4px;">Volga IT 2026 | Оценка точности и ручное уточнение Ground Truth</div>
        </div>

        <div class="stats-bar">
            <div class="stat-pill stat-progress">📁 Размечено: <strong id="globalAuditedCount">{audited_count}</strong> / <strong>{total_real}</strong> ({progress_pct:.1f}%)</div>
            <div class="stat-pill">📦 Пачка: <strong>{len(sample_files)}</strong> кадров</div>
            <div class="stat-pill stat-model" title="Модель: {ocr_name}">🧠 OCR: <strong>{ocr_name}</strong> <span style="font-size:11px; opacity:0.8;">({ocr_mtime_str})</span></div>
            <div class="stat-pill">🕒 Сгенерировано: <strong>{batch_time_str}</strong></div>
            <div class="stat-pill">Одобрено: <span class="stat-yes" id="yesCount">0</span></div>
            <div class="stat-pill">Скорректировано GT: <span class="stat-cor" id="corCount">0</span></div>
            <div class="stat-pill">Отклонено: <span class="stat-no" id="noCount">0</span></div>
            <div class="stat-pill">Осталось: <strong id="remainCount">{len(sample_files)}</strong></div>
            <div class="stat-pill">Задержка: <strong>{avg_ms:.1f} мс</strong></div>
        </div>

        <div class="controls">
            <span id="saveIndicator" style="transition:opacity 0.3s; font-size:12px; color:var(--success); font-weight:bold;"></span>
            <button class="action-btn" id="newBatchBtn" onclick="fetchNewBatch()" style="background:#0284c7; color:#fff; border-color:#38bdf8; font-weight:bold;">🔀 Новая пачка (Unseen)</button>
            <button class="action-btn" id="modeBtn" onclick="toggleMode()">📜 Режим скроллинга</button>
            <button class="action-btn export" onclick="exportAuditCSV()">📥 Экспорт CSV (с GT)</button>
            <button class="action-btn" onclick="resetVotes()" style="color:#ef4444;">🧹 Очистить отметки</button>
        </div>
    </div>

    <div style="display:flex; gap:8px; margin-bottom:20px; overflow-x:auto;">
        <button class="filter-btn active" onclick="filterCards('all')">Все типы</button>
        <button class="filter-btn" onclick="filterCards('type1')">🚗 Type 1 (Белые 1-строчные)</button>
        <button class="filter-btn" onclick="filterCards('type1a')">🟦 Type 1A (Квадрат JDM/США)</button>
        <button class="filter-btn" onclick="filterCards('type1b')">🚌 Type 1B (Желтые автобусы)</button>
        <button class="filter-btn" onclick="filterCards('other')">🏷️ Other (Прицепы / Мото / Спец)</button>
        <button class="filter-btn" onclick="filterCards('no_plate')">🚫 Брак / Мусор</button>
    </div>

    <div class="container grid-mode" id="cardContainer">
        {"".join(cards_html)}
    </div>

    <div class="hotkeys-hint">
        Горячие клавиши: <kbd>Y</kbd> — Одобрить GT &bull; <kbd>N</kbd> — Брак / Нет номера &bull; <kbd>Enter</kbd> в поле — сохранить GT &bull; <kbd>&uarr;</kbd> <kbd>&darr;</kbd> — Навигация
    </div>

    <script>
        const STORAGE_KEY = 'omniplate_audit_{session_id}';
        let auditData = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
        let currentFocusIdx = 1;
        const totalCards = {len(sample_files)};

        async function fetchNewBatch() {{
            if (!confirm('Сгенерировать новую выборку из ещё не размеченных реальных изображений?')) return;
            const btn = document.getElementById('newBatchBtn');
            if (btn) {{
                btn.textContent = '⏳ Сохранение и генерация...';
                btn.disabled = true;
            }}

            try {{
                // 1. Flush any pending votes to disk before sampling fresh batch
                if (location.protocol.startsWith('http')) {{
                    const rows = buildAuditRows();
                    await fetch('/save', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(rows)
                    }});
                }}

                // 2. Request new unseen batch
                const resp = await fetch('/api/next_batch', {{ method: 'POST' }});
                const data = await resp.json();
                if (data.status === 'ok') {{
                    // Force cache-busting reload
                    window.location.href = '/?t=' + Date.now();
                }} else {{
                    throw new Error(data.message || 'Ошибка сервера при генерации пачки');
                }}
            }} catch (err) {{
                alert('Ошибка генерации новой пачки: ' + err.message);
                if (btn) {{
                    btn.textContent = '🔀 Новая пачка (Unseen)';
                    btn.disabled = false;
                }}
            }}
        }}

        function selectTypeDirect(idx, newType) {{
            const sel = document.getElementById('type-select-' + idx);
            if (sel) {{
                sel.value = newType;
            }}
            changeCardType(idx);
        }}

        function changeCardType(idx) {{
            const sel = document.getElementById('type-select-' + idx);
            if (!sel) return;
            const newType = sel.value;
            const card = document.getElementById('card-' + idx);
            if (!card) return;
            card.dataset.type = newType;

            const typeMap = {{
                'type1': {{ col: '#10b981', icon: '🚗', name: 'TYPE 1', desc: 'Белый легковой 1-строчный (A000AA77)' }},
                'type1a': {{ col: '#3b82f6', icon: '🟦', name: 'TYPE 1A', desc: 'Квадрат 2-строчный (JDM/USDM)' }},
                'type1b': {{ col: '#f59e0b', icon: '🚌', name: 'TYPE 1B', desc: 'Желтый автобус (AA00077)' }},
                'type2': {{ col: '#a855f7', icon: '🏷️', name: 'OTHER', desc: 'Прицеп (ГОСТ Тип 2 -> Other по ТЗ)' }},
                'other': {{ col: '#a855f7', icon: '🏷️', name: 'OTHER', desc: 'Прицеп / Мото / Дипломат / Спецномер' }},
                'no_plate': {{ col: '#ef4444', icon: '🚫', name: 'БРАК', desc: 'Ложная детекция / Мусор / Нет номера' }}
            }};
            const info = typeMap[newType] || {{ col: '#6b7280', icon: '🏷️', name: newType.toUpperCase(), desc: '' }};

            // 1. Update top badge
            const badge = card.querySelector('.badge');
            if (badge) {{
                badge.textContent = `${{info.icon}} ${{info.name}}`;
                badge.style.backgroundColor = info.col;
            }}

            // 2. Update crop hint and current label
            const hintEl = document.getElementById('type-badge-hint-' + idx);
            if (hintEl) {{
                hintEl.textContent = `${{info.icon}} ${{info.name}}: ${{info.desc}}`;
                hintEl.style.color = info.col;
            }}
            const currEl = document.getElementById('type-curr-label-' + idx);
            if (currEl) {{
                currEl.textContent = `${{info.icon}} ${{info.name}}`;
                currEl.style.color = info.col;
            }}

            // 3. Update active chips
            const chips = document.querySelectorAll('#type-chips-' + idx + ' .type-chip');
            chips.forEach(ch => {{
                if (ch.id === `chip-${{idx}}-${{newType}}`) {{
                    ch.classList.add('active');
                }} else {{
                    ch.classList.remove('active');
                }}
            }});

            if (!auditData[idx]) auditData[idx] = {{}};
            auditData[idx].corrected_type = newType;
            if (!auditData[idx].status || auditData[idx].status === 'unreviewed') {{
                auditData[idx].status = 'corrected';
                card.classList.add('voted-corrected');
                const statusEl = document.getElementById('status-' + idx);
                if (statusEl) statusEl.innerHTML = `<span style="color:var(--primary); font-weight:bold;">✏️ ТИП: ${{info.icon}} ${{info.name}}</span>`;
            }}
            saveData();
        }}

        function buildAuditRows() {{
            const rows = [['id', 'filename', 'type', 'predicted_text', 'status', 'ground_truth_text']];
            document.querySelectorAll('.card').forEach(c => {{
                const idx = c.dataset.idx;
                const fn = c.dataset.filename;
                const sel = document.getElementById('type-select-' + idx);
                const item = auditData[idx] || {{ status: 'unreviewed' }};
                let tp = item.corrected_type || (sel ? sel.value : c.dataset.type);
                if (tp === 'type2') {{
                    tp = 'other';
                }}
                const pred = c.dataset.pred;

                let status = item.status;
                let gt = pred;
                if (status === 'corrected') {{
                    gt = item.corrected_text !== undefined ? item.corrected_text : pred;
                }} else if (status === 'no') {{
                    gt = '';
                }}
                rows.push([idx, fn, tp, pred, status, gt]);
            }});
            return rows;
        }}

        function saveData() {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(auditData));
            updateStats();

            // Auto-sync with local Python server (http://localhost:8080)
            if (location.protocol.startsWith('http')) {{
                const rows = buildAuditRows();
                fetch('/save', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(rows)
                }})
                .then(r => r.json())
                .then(data => {{
                    const ind = document.getElementById('saveIndicator');
                    if (ind) {{
                        ind.textContent = '⚡ Сохранено на диск';
                        ind.style.opacity = '1';
                        setTimeout(() => {{ ind.style.opacity = '0'; }}, 1200);
                    }}
                    if (data && data.audited_count !== undefined) {{
                        const globalEl = document.getElementById('globalAuditedCount');
                        if (globalEl) globalEl.textContent = data.audited_count;
                    }}
                }})
                .catch(() => {{}});
            }}
        }}

        function vote(idx, choice) {{
            if (!auditData[idx]) auditData[idx] = {{}};
            auditData[idx].status = choice; // 'yes' or 'no'
            const sel = document.getElementById('type-select-' + idx);
            if (sel) {{
                auditData[idx].corrected_type = sel.value;
            }}

            const card = document.getElementById('card-' + idx);
            const status = document.getElementById('status-' + idx);
            if (!card || !status) return;

            card.classList.remove('voted-yes', 'voted-no', 'voted-corrected');
            if (choice === 'yes') {{
                card.classList.add('voted-yes');
                status.innerHTML = '<span style="color:var(--success); font-weight:bold;">✅ ОДОБРЕНО (ТИП И ТЕКСТ ВЕРНЫ)</span>';
            }} else {{
                card.classList.add('voted-no');
                selectTypeDirect(idx, 'no_plate');
                status.innerHTML = '<span style="color:var(--danger); font-weight:bold;">🗑️ БРАК ДЕТЕКТОРА (НЕТ НОМЕРА)</span>';
            }}
            saveData();

            // Auto-advance in scroll mode
            const container = document.getElementById('cardContainer');
            if (container.classList.contains('scroll-mode') && idx < totalCards) {{
                focusCard(idx + 1);
            }}
        }}

        function quickFillWildcard(idx, n) {{
            const input = document.getElementById('edit-' + idx);
            if (input) {{
                input.value = '#'.repeat(n);
                saveCorrection(idx);
            }}
        }}

        function saveCorrection(idx) {{
            const input = document.getElementById('edit-' + idx);
            if (!input) return;
            const text = input.value.trim().toUpperCase();
            if (!text) return;

            if (!auditData[idx]) auditData[idx] = {{}};
            auditData[idx].status = 'corrected';
            auditData[idx].corrected_text = text;
            const sel = document.getElementById('type-select-' + idx);
            if (sel) {{
                auditData[idx].corrected_type = sel.value;
            }}

            const card = document.getElementById('card-' + idx);
            const status = document.getElementById('status-' + idx);
            const display = document.getElementById('pred-display-' + idx);

            if (card && status) {{
                card.classList.remove('voted-yes', 'voted-no', 'voted-corrected');
                card.classList.add('voted-corrected');
                status.innerHTML = `<span style="color:var(--primary); font-weight:bold;">✏️ УТОЧНЁН GT: ${{text}}</span>`;
            }}
            saveData();

            // Auto-advance in scroll mode
            const container = document.getElementById('cardContainer');
            if (container.classList.contains('scroll-mode') && idx < totalCards) {{
                focusCard(idx + 1);
            }}
        }}

        function updateStats() {{
            let yes = 0, no = 0, cor = 0;
            for (const k in auditData) {{
                const item = auditData[k];
                if (item.status === 'yes') yes++;
                else if (item.status === 'no') no++;
                else if (item.status === 'corrected') cor++;
            }}
            document.getElementById('yesCount').textContent = yes;
            document.getElementById('noCount').textContent = no;
            document.getElementById('corCount').textContent = cor;
            document.getElementById('remainCount').textContent = Math.max(0, totalCards - (yes + no + cor));
        }}

        function applySavedData() {{
            for (const idx in auditData) {{
                const item = auditData[idx];
                const card = document.getElementById('card-' + idx);
                const status = document.getElementById('status-' + idx);
                const input = document.getElementById('edit-' + idx);

                if (!card || !status) continue;
                card.classList.remove('voted-yes', 'voted-no', 'voted-corrected');

                if (item.status === 'yes') {{
                    card.classList.add('voted-yes');
                    status.innerHTML = '<span style="color:var(--success); font-weight:bold;">✅ ОДОБРЕНО</span>';
                }} else if (item.status === 'no') {{
                    card.classList.add('voted-no');
                    status.innerHTML = '<span style="color:var(--danger); font-weight:bold;">🗑️ БРАК / НЕТ НОМЕРА</span>';
                }} else if (item.status === 'corrected') {{
                    card.classList.add('voted-corrected');
                    status.innerHTML = `<span style="color:var(--primary); font-weight:bold;">✏️ УТОЧНЁН GT: ${{item.corrected_text}}</span>`;
                    if (input && item.corrected_text) input.value = item.corrected_text;
                }}
            }}
            updateStats();
        }}

        function toggleMode() {{
            const container = document.getElementById('cardContainer');
            const btn = document.getElementById('modeBtn');
            if (container.classList.contains('grid-mode')) {{
                container.classList.remove('grid-mode');
                container.classList.add('scroll-mode');
                btn.textContent = '▦ Режим сетки';
                focusCard(currentFocusIdx);
            }} else {{
                container.classList.remove('scroll-mode');
                container.classList.add('grid-mode');
                btn.textContent = '📜 Режим скроллинга';
            }}
        }}

        function focusCard(idx) {{
            const card = document.getElementById('card-' + idx);
            if (card) {{
                currentFocusIdx = idx;
                card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        function resetVotes() {{
            if (confirm('Сбросить все оценки и исправления аудита?')) {{
                auditData = {{}};
                localStorage.removeItem(STORAGE_KEY);
                document.querySelectorAll('.card').forEach(c => {{
                    c.classList.remove('voted-yes', 'voted-no', 'voted-corrected');
                }});
                document.querySelectorAll('.audit-status').forEach(s => {{
                    s.textContent = 'Не проверено';
                }});
                updateStats();
            }}
        }}

        function exportAuditCSV() {{
            const rows = [['id', 'filename', 'type', 'predicted_text', 'status', 'ground_truth_text']];
            document.querySelectorAll('.card').forEach(c => {{
                const idx = c.dataset.idx;
                const fn = c.dataset.filename;
                const tp = c.dataset.type;
                const pred = c.dataset.pred;

                const item = auditData[idx] || {{ status: 'unreviewed' }};
                let status = item.status;
                let gt = pred;
                if (status === 'corrected') {{
                    gt = item.corrected_text;
                }} else if (status === 'no') {{
                    gt = '';
                }}

                rows.push([idx, fn, tp, pred, status, gt]);
            }});

            const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(";")).join("\\n");
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "omniplate_ground_truth_audit.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        function filterCards(type) {{
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {{
                if (type === 'all' || card.dataset.type === type) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        // Keyboard hotkeys
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return; // Don't trigger hotkeys when typing in input
            const key = e.key.toLowerCase();
            if (key === 'y' || key === 'н' || key === '1') {{
                vote(currentFocusIdx, 'yes');
            }} else if (key === 'n' || key === 'т' || key === '2') {{
                vote(currentFocusIdx, 'no');
            }} else if (key === 'arrowdown' || key === 'arrowright') {{
                if (currentFocusIdx < totalCards) focusCard(currentFocusIdx + 1);
            }} else if (key === 'arrowup' || key === 'arrowleft') {{
                if (currentFocusIdx > 1) focusCard(currentFocusIdx - 1);
            }}
        }});

        function fetchLiveStats() {{
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {{
                    if (data && data.stats && data.stats.audited_count !== undefined) {{
                        const count = data.stats.audited_count;
                        const total = data.stats.total_real_images || 531;
                        const pct = data.stats.progress_percent || ((count / total) * 100).toFixed(1);
                        const el = document.getElementById('globalAuditedCount');
                        if (el) el.textContent = count;
                        const progEl = document.querySelector('.stat-progress');
                        if (progEl) {{
                            progEl.innerHTML = `📁 Размечено: <strong id="globalAuditedCount">${{count}}</strong> / <strong>${{total}}</strong> (${{pct}}%)`;
                        }}
                    }}
                }})
                .catch(() => {{}});
        }}

        applySavedData();
        fetchLiveStats();
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[SUCCESS] Interactive Audit & GT Correction Gallery generated at: {output_html}")
    print(f"          Open in browser: file:///{output_html}")


if __name__ == "__main__":
    main()
