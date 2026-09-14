#!/usr/bin/env python3
"""
OmniPlate-RU — Interactive Visual Gallery Generator for User Audit.
Runs inference across real or synthetic images, generates annotated scenes and crops,
and outputs a standalone interactive HTML gallery (test_output/gallery.html).
"""

import argparse
import base64
import html
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline


def image_to_base64_thumbnail(img_bgr: np.ndarray, max_dim: int = 360) -> str:
    """Resizes and encodes an image to base64 JPEG data URL for standalone HTML."""
    h, w = img_bgr.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def main():
    parser = argparse.ArgumentParser(description="OmniPlate HTML Audit Gallery Generator")
    parser.add_argument("--input_dir", type=str, default="dataset/images/real", help="Directory with input images")
    parser.add_argument("--output_html", type=str, default="test_output/gallery.html", help="Path to output HTML")
    parser.add_argument("--limit", type=int, default=60, help="Max images to include in gallery (default: 60)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device ('cuda' or 'cpu')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--filter_type", type=str, default=None, help="Filter input by filename substring (e.g. 'type1b')")
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    output_html = PROJECT_ROOT / args.output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading images from {input_dir}...")
    all_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if args.filter_type:
        all_files = [f for f in all_files if args.filter_type in f.name]

    sample_files = all_files[:args.limit]
    print(f"[*] Processing {len(sample_files)} images (limit={args.limit})...")

    pipeline = OmniPlatePipeline(device=args.device, conf_threshold=args.conf)
    pipeline.warmup(iterations=2)

    cards_html = []
    total_time = 0.0

    for idx, fpath in enumerate(sample_files, start=1):
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        t0 = time.perf_counter()
        dets, timings = pipeline.predict_with_timing(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_time += dt_ms

        # Prepare visuals
        vis_scene = img.copy()
        crop_b64 = ""
        pred_text = "—"
        p_type = "no_plate"
        conf_val = 0.0

        if dets:
            d = dets[0]
            pred_text = d.text or "N/A"
            p_type = d.plate_type
            conf_val = d.confidence

            # Draw bbox and quad
            bx, by, bw, bh = d.bbox
            cv2.rectangle(vis_scene, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            if d.quad and len(d.quad) == 8:
                pts = np.array(d.quad, dtype=np.int32).reshape(4, 2)
                cv2.polylines(vis_scene, [pts], True, (255, 255, 255), 2)

            if d.rectified_crop is not None and d.rectified_crop.size > 0:
                crop_b64 = image_to_base64_thumbnail(d.rectified_crop, max_dim=240)

        scene_b64 = image_to_base64_thumbnail(vis_scene, max_dim=480)

        badge_color = "#10b981" if p_type == "type1" else ("#3b82f6" if p_type == "type1a" else ("#f59e0b" if p_type == "type1b" else "#ef4444"))
        if p_type == "no_plate":
            badge_color = "#6b7280"

        card = f"""
        <div class="card" data-type="{html.escape(p_type)}">
            <div class="card-header">
                <span class="badge" style="background-color: {badge_color};">{html.escape(p_type.upper())}</span>
                <span class="fname">{html.escape(fpath.name)}</span>
                <span class="time">{dt_ms:.1f} ms</span>
            </div>
            <div class="media-row">
                <div class="scene-col">
                    <img class="thumb" src="{scene_b64}" alt="Scene" loading="lazy" />
                </div>
                {f'<div class="crop-col"><img class="crop-img" src="{crop_b64}" alt="Crop" /></div>' if crop_b64 else ''}
            </div>
            <div class="meta-row">
                <div class="pred-text">{html.escape(pred_text)}</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width: {min(100, int(conf_val * 100))}%;"></div>
                    <span class="conf-label">{conf_val:.2f}</span>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

        if idx % 10 == 0 or idx == len(sample_files):
            print(f"[{idx}/{len(sample_files)}] Processed {fpath.name} ({dt_ms:.1f} ms)")

    avg_ms = total_time / max(1, len(sample_files))
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniPlate-RU — Аудит предсказаний на реальной выборке</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --subtext: #94a3b8;
            --primary: #38bdf8;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .stats {{
            display: flex;
            gap: 16px;
        }}
        .stat-pill {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
        }}
        .stat-pill strong {{
            color: var(--primary);
        }}
        .filters {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .filter-btn {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .filter-btn.active {{
            background: var(--primary);
            color: #0f172a;
            font-weight: 600;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
        }}
        .badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            color: #fff;
        }}
        .fname {{
            color: var(--subtext);
            font-family: monospace;
            font-size: 12px;
        }}
        .time {{
            color: var(--primary);
            font-family: monospace;
        }}
        .media-row {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .scene-col {{
            flex: 2;
        }}
        .crop-col {{
            flex: 1;
            display: flex;
            justify-content: center;
        }}
        .thumb {{
            width: 100%;
            height: 140px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        .crop-img {{
            max-width: 100%;
            max-height: 48px;
            object-fit: contain;
            border-radius: 4px;
            border: 1px solid #64748b;
            background: #000;
        }}
        .meta-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 4px;
            padding-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.05);
        }}
        .pred-text {{
            font-size: 18px;
            font-weight: 700;
            font-family: monospace;
            letter-spacing: 1px;
            color: #38bdf8;
        }}
        .conf-bar-wrap {{
            width: 100px;
            background: #334155;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}
        .conf-bar {{
            background: #10b981;
            height: 100%;
        }}
        .conf-label {{
            position: absolute;
            right: 4px;
            top: -14px;
            font-size: 10px;
            color: var(--subtext);
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size: 20px;">OmniPlate-RU — Аудит предсказаний (Volga IT 2026)</h1>
            <div style="color: var(--subtext); font-size: 13px; margin-top: 4px;">Интерактивная галерея инференса на реальной дорожной выборке</div>
        </div>
        <div class="stats">
            <div class="stat-pill">Кадров: <strong>{len(sample_files)}</strong></div>
            <div class="stat-pill">Средняя задержка: <strong>{avg_ms:.1f} мс</strong></div>
            <div class="stat-pill">SLA лимит: <strong>&le; 100 мс (&times;{100.0/max(1e-3, avg_ms):.1f} быстрее)</strong></div>
        </div>
    </div>

    <div class="filters">
        <button class="filter-btn active" onclick="filterCards('all')">Все кадры</button>
        <button class="filter-btn" onclick="filterCards('type1')">Type 1 (Белые)</button>
        <button class="filter-btn" onclick="filterCards('type1a')">Type 1A (Квадрат)</button>
        <button class="filter-btn" onclick="filterCards('type1b')">Type 1B (Желтые)</button>
        <button class="filter-btn" onclick="filterCards('no_plate')">Без номера / Negatives</button>
    </div>

    <div class="grid" id="cardGrid">
        {"".join(cards_html)}
    </div>

    <script>
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
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[SUCCESS] Gallery generated at: {output_html}")
    print(f"          Open in browser: file:///{output_html.resolve()}")


if __name__ == "__main__":
    main()
