#!/usr/bin/env python3
"""Генератор синтетических ГРЗ типов 1, 1А (квадратные) и 1Б (жёлтые) — Волга-IT 2026.

Рендерит знак в геометрии ГОСТ Р 50577-2018, вклеивает его в кадр (сгенерированный
фон или случайный кроп из ваших фотографий), применяет «камерные» аугментации
и пишет результат сразу в структуре датасета из раздела 6 задания:

    <out>/images/synthetic/syn_000001.jpg
    <out>/labels/syn_000001.txt        # YOLO: class cx cy w h [+ quad x1 y1 ... x4 y4], нормированные
    <out>/meta.csv                     # строки с bbox / quad / plate_type / is_synthetic=1

Классы в labels: 0 = type1, 1 = type1a, 2 = type1b, 3 = other.

Запуск:
    python generate_synthetic_plates.py --count 5000 --out dataset --seed 42
    python generate_synthetic_plates.py --count 200 --out dataset --seed 1 --types type1a,type1b
    python generate_synthetic_plates.py --count 200 --out dataset --seed 1 --backgrounds my_photos/

Зависимости: numpy, opencv-python (или -headless), Pillow. Шрифт: по умолчанию
ищется системный жирный (arialbd / DejaVuSans-Bold); для сходства с ГОСТ укажите
--font RoadNumbers.ttf. Один и тот же --seed даёт побайтово тот же набор.

Основано на генераторе однострочных номеров из рабочего LPR-пайплайна;
добавлены типы 1А/1Б, вклейка в кадр и вывод разметки.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Алфавит, регионы
# ---------------------------------------------------------------------------

LATIN_LETTERS = "ABEKMHOPCTYX"
CYR_LETTERS = "АВЕКМНОРСТУХ"
_LAT_TO_CYR = str.maketrans(LATIN_LETTERS, CYR_LETTERS)

REGION_CODES = (
    "02", "16", "23", "24", "36", "38", "50", "52", "54", "61",
    "63", "66", "72", "73", "74", "77", "78", "86", "96", "99",
    "102", "116", "123", "124", "136", "150", "152", "154", "161",
    "163", "173", "174", "177", "178", "186", "196", "197", "199",
    "702", "716", "725", "750", "761", "763", "777", "790", "797", "799",
)

PLATE_TYPES = ("type1", "type1a", "type1b")
LABEL_CLASS = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

# Геометрия ГОСТ (мм) → пиксели, 1 мм = 2 px.
SCALE = 2
PLATE_SIZE_MM = {"type1": (520, 112), "type1a": (290, 170), "type1b": (520, 112)}
DIGIT_H_MM, LETTER_H_MM = 76, 58
CHAR_GAP_MM, GROUP_GAP_MM = 5, 13

INK = (15, 15, 15)
WHITE_BG = (245, 245, 240)
YELLOW_BG = (250, 205, 30)

_DEFAULT_FONTS = (
    "arialbd.ttf",
    "arial.ttf",
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def random_plate_text(rng: np.random.Generator, plate_type: str) -> str:
    """Случайный номер латиницей: Л ЦЦЦ ЛЛ + регион (тип 1/1А) или ЛЛ ЦЦЦ + регион (тип 1Б)."""
    region = REGION_CODES[int(rng.integers(0, len(REGION_CODES)))]
    digits = "".join(str(d) for d in rng.integers(0, 10, size=3))
    if plate_type == "type1b":
        letters = "".join(rng.choice(list(LATIN_LETTERS), size=2))
        return f"{letters}{digits}{region}"
    letters = rng.choice(list(LATIN_LETTERS), size=3)
    return f"{letters[0]}{digits}{letters[1]}{letters[2]}{region}"


def split_plate(text: str, plate_type: str) -> tuple[str, str]:
    """(тело, регион): тело — 6 символов для типов 1/1А, 5 — для 1Б."""
    n = 5 if plate_type == "type1b" else 6
    return text[:n], text[n:]


# ---------------------------------------------------------------------------
# Шрифты
# ---------------------------------------------------------------------------

def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    candidates = (font_path,) if font_path else _DEFAULT_FONTS
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise FileNotFoundError("Не найден TTF-шрифт: укажите --font (RoadNumbers.ttf, arialbd.ttf, ...)")


def glyph_supported(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    notdef = font.getmask("͸")
    mask = font.getmask(ch)
    return mask.size != notdef.size or bytes(mask) != bytes(notdef)


def to_render_charset(font: ImageFont.FreeTypeFont, latin_text: str) -> str:
    """Кириллица, если шрифт её умеет (глифы одинаковы), иначе латиница."""
    if glyph_supported(font, "А"):
        return latin_text.translate(_LAT_TO_CYR)
    if glyph_supported(font, "A"):
        return latin_text
    raise ValueError("Шрифт не содержит букв номера")


def font_for_glyph_height(font_path: str | None, target_h: int, sample: str = "0") -> ImageFont.FreeTypeFont:
    font = load_font(font_path, target_h)
    left, top, right, bottom = font.getbbox(sample)
    measured = bottom - top
    if measured > 0 and measured != target_h:
        font = load_font(font_path, max(8, round(target_h * target_h / measured)))
    return font


@dataclass
class FontSet:
    digit: ImageFont.FreeTypeFont
    letter: ImageFont.FreeTypeFont
    region: ImageFont.FreeTypeFont
    rus: ImageFont.FreeTypeFont
    rus_small: ImageFont.FreeTypeFont
    small_digit: ImageFont.FreeTypeFont
    small_letter: ImageFont.FreeTypeFont


_FONT_CACHE: dict[str | None, FontSet] = {}


def get_fonts(font_path: str | None) -> FontSet:
    if font_path in _FONT_CACHE:
        return _FONT_CACHE[font_path]
    probe = load_font(font_path, 50)
    sample_letter = "А" if glyph_supported(probe, "А") else "A"
    rus_font = load_font(font_path, 20 * SCALE)
    rus_path: str | None = font_path
    if not glyph_supported(rus_font, "R"):
        rus_font = load_font(None, 20 * SCALE)
        rus_path = None
    fs = FontSet(
        digit=font_for_glyph_height(font_path, DIGIT_H_MM * SCALE),
        letter=font_for_glyph_height(font_path, LETTER_H_MM * SCALE, sample_letter),
        region=font_for_glyph_height(font_path, 58 * SCALE),
        rus=rus_font,
        rus_small=load_font(rus_path, 11 * SCALE),
        small_digit=font_for_glyph_height(font_path, 58 * SCALE),
        small_letter=font_for_glyph_height(font_path, 58 * SCALE, sample_letter),
    )
    _FONT_CACHE[font_path] = fs
    return fs


# ---------------------------------------------------------------------------
# Отрисовка чистого знака
# ---------------------------------------------------------------------------

def _shrink_fonts(fonts: list[ImageFont.FreeTypeFont], factor: float) -> list[ImageFont.FreeTypeFont]:
    return [ImageFont.truetype(f.path, max(8, int(f.size * factor))) for f in fonts]


def _draw_row(draw: ImageDraw.ImageDraw, chars: str, fonts: list[ImageFont.FreeTypeFont],
              gaps_mm: list[int], *, x0: int, x1: int, baseline: int, align: str = "center") -> None:
    """Рисует строку символов посимвольно (свои шрифты и зазоры), выравнивая по базовой линии.

    Если строка не влезает в зону даже с минимальными зазорами (широкие системные
    шрифты вместо номерного), кегль всей строки пропорционально уменьшается.
    """
    zone = x1 - x0
    min_gaps = sum(2 * SCALE for _ in gaps_mm)
    for _ in range(6):
        widths_now = sum(r - l for l, _t, r, _b in (f.getbbox(ch) for ch, f in zip(chars, fonts)))
        if widths_now + min_gaps <= zone:
            break
        fonts = _shrink_fonts(fonts, 0.92 * zone / (widths_now + min_gaps))
    boxes = [f.getbbox(ch) for ch, f in zip(chars, fonts)]
    widths = [r - l for l, _t, r, _b in boxes]
    gaps_px = [g * SCALE for g in gaps_mm]
    total = sum(widths) + sum(gaps_px)
    zone = x1 - x0
    if total > zone and sum(widths) < zone:
        shrink = (zone - sum(widths)) / max(1, sum(gaps_px))
        gaps_px = [int(g * shrink) for g in gaps_px]
        total = sum(widths) + sum(gaps_px)
    if align == "center":
        x = x0 + max(0, (zone - total) // 2)
    elif align == "left":
        x = x0
    else:
        x = x1 - total
    for i, (ch, font, (l, _t, _r, b)) in enumerate(zip(chars, fonts, boxes)):
        draw.text((x - l, baseline - b), ch, font=font, fill=INK)
        x += widths[i] + (gaps_px[i] if i < len(gaps_px) else 0)


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                   *, x0: int, x1: int, y_center: int) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top
    draw.text((x0 + ((x1 - x0) - w) // 2 - left, y_center - h // 2 - top), text, font=font, fill=INK)


def _draw_flag(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    stripe = h // 3
    draw.rectangle([x, y, x + w, y + stripe], fill=(245, 245, 245))
    draw.rectangle([x, y + stripe, x + w, y + 2 * stripe], fill=(40, 70, 180))
    draw.rectangle([x, y + 2 * stripe, x + w, y + h], fill=(200, 40, 45))


def render_single_row(latin_text: str, plate_type: str, fonts: FontSet) -> Image.Image:
    """Однострочный знак 520×112: тип 1 (белый) или 1Б (жёлтый)."""
    w_mm, h_mm = PLATE_SIZE_MM[plate_type]
    W, H = w_mm * SCALE, h_mm * SCALE
    img = Image.new("RGB", (W, H), YELLOW_BG if plate_type == "type1b" else WHITE_BG)
    draw = ImageDraw.Draw(img)
    border = 3 * SCALE
    draw.rectangle([0, 0, W - 1, H - 1], outline=(20, 20, 20), width=border)
    region_x = int(388 * SCALE)
    draw.line([(region_x, border), (region_x, H - border)], fill=(20, 20, 20), width=SCALE)

    body, region = split_plate(latin_text, plate_type)
    body_r = to_render_charset(fonts.letter, body)
    if plate_type == "type1b":
        row_fonts = [fonts.letter] * 2 + [fonts.digit] * 3
        gaps = [CHAR_GAP_MM, GROUP_GAP_MM, CHAR_GAP_MM, CHAR_GAP_MM]
    else:
        row_fonts = [fonts.letter] + [fonts.digit] * 3 + [fonts.letter] * 2
        gaps = [GROUP_GAP_MM, CHAR_GAP_MM, CHAR_GAP_MM, GROUP_GAP_MM, CHAR_GAP_MM]
    baseline = (H + DIGIT_H_MM * SCALE) // 2
    _draw_row(draw, body_r, row_fonts, gaps, x0=border * 3, x1=region_x - SCALE * 5, baseline=baseline)

    _draw_centered(draw, region, fonts.region, x0=region_x, x1=W - border * 2, y_center=int(H * 0.36))
    flag_w, flag_h = 28 * SCALE, 16 * SCALE
    rus_x1 = W - border * 2 - flag_w - 4 * SCALE
    _draw_centered(draw, "RUS", fonts.rus, x0=region_x, x1=rus_x1, y_center=int(H * 0.78))
    _draw_flag(draw, rus_x1 + 2 * SCALE, int(H * 0.78) - flag_h // 2, flag_w, flag_h)
    return img


def render_two_row(latin_text: str, fonts: FontSet) -> Image.Image:
    """Двухстрочный знак 1А, 290×170: сверху Л ЦЦЦ, снизу ЛЛ + регион, RUS и флаг."""
    w_mm, h_mm = PLATE_SIZE_MM["type1a"]
    W, H = w_mm * SCALE, h_mm * SCALE
    img = Image.new("RGB", (W, H), WHITE_BG)
    draw = ImageDraw.Draw(img)
    border = 3 * SCALE
    draw.rectangle([0, 0, W - 1, H - 1], outline=(20, 20, 20), width=border)

    body, region = split_plate(latin_text, "type1a")
    body_r = to_render_charset(fonts.letter, body)
    top_chars, bottom_chars = body_r[:4], body_r[4:]

    # Верхняя строка: буква + три цифры, базовая линия на ~48% высоты.
    top_baseline = int(H * 0.50)
    _draw_row(draw, top_chars, [fonts.letter] + [fonts.digit] * 3,
              [GROUP_GAP_MM, CHAR_GAP_MM, CHAR_GAP_MM], x0=border * 3, x1=W - border * 3, baseline=top_baseline)

    # Нижняя строка: две буквы слева, регион, справа узкая колонка флаг/RUS.
    flag_w, flag_h = 22 * SCALE, 12 * SCALE
    col_w = flag_w + 6 * SCALE
    col_x0 = W - border * 3 - col_w
    bottom_baseline = int(H * 0.91)
    bottom_chars_all = bottom_chars + region
    bfonts = [fonts.small_letter] * 2 + [fonts.small_digit] * len(region)
    gaps = [CHAR_GAP_MM, GROUP_GAP_MM + 4] + [CHAR_GAP_MM] * (len(region) - 1)
    _draw_row(draw, bottom_chars_all, bfonts, gaps, x0=border * 3, x1=col_x0 - 4 * SCALE, baseline=bottom_baseline)

    _draw_flag(draw, col_x0 + 3 * SCALE, int(H * 0.62), flag_w, flag_h)
    _draw_centered(draw, "RUS", fonts.rus_small, x0=col_x0, x1=W - border * 3, y_center=int(H * 0.84))
    return img


def render_plate(latin_text: str, plate_type: str, font_path: str | None = None) -> np.ndarray:
    """Чистый знак в BGR."""
    fonts = get_fonts(font_path)
    img = render_two_row(latin_text, fonts) if plate_type == "type1a" else render_single_row(latin_text, plate_type, fonts)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Сцена: фон, вклейка, аугментации
# ---------------------------------------------------------------------------

def make_background(rng: np.random.Generator, size: tuple[int, int], backgrounds: list[str]) -> np.ndarray:
    """Кадр-подложка: случайный кроп из пользовательских фото или градиент с шумом."""
    w, h = size
    if backgrounds:
        for _ in range(5):
            path = backgrounds[int(rng.integers(0, len(backgrounds)))]
            bg = cv2.imread(path)
            if bg is None or bg.shape[0] < 64 or bg.shape[1] < 64:
                continue
            scale = max(w / bg.shape[1], h / bg.shape[0]) * float(rng.uniform(1.0, 1.6))
            bg = cv2.resize(bg, (max(w, int(bg.shape[1] * scale)), max(h, int(bg.shape[0] * scale))))
            y0 = int(rng.integers(0, bg.shape[0] - h + 1))
            x0 = int(rng.integers(0, bg.shape[1] - w + 1))
            return bg[y0:y0 + h, x0:x0 + w].copy()
    # Синтетический фон: вертикальный градиент «асфальт/небо» + шум + «кузов» вокруг знака.
    base_top, base_bottom = float(rng.integers(50, 170)), float(rng.integers(20, 110))
    top = np.clip(base_top + rng.integers(-20, 20, size=3), 0, 255).astype(np.float32)
    bottom = np.clip(base_bottom + rng.integers(-20, 20, size=3), 0, 255).astype(np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    canvas = (top * (1 - t) + bottom * t)
    canvas = np.repeat(canvas, w, axis=1)
    canvas += rng.normal(0, 8, canvas.shape).astype(np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def paste_plate(frame: np.ndarray, plate: np.ndarray, rng: np.random.Generator,
                occupied: list[tuple[float, float, float, float]] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Вклеивает знак в кадр с масштабом, перспективой и поворотом.

    Возвращает кадр, quad (4×2, по часовой от левого верхнего) и теги условий.
    occupied — bbox уже вклеенных знаков (x0, y0, x1, y1), с ними не пересекаемся.
    """
    fh, fw = frame.shape[:2]
    ph, pw = plate.shape[:2]
    tags: dict[str, bool] = {}
    occupied = occupied or []

    # Целевая высота знака в кадре: логравномерно от мелкого (далеко) до крупного.
    frac = float(np.exp(rng.uniform(np.log(0.03), np.log(0.25))))
    scale = frac * fh / ph
    scale = min(scale, 0.8 * fw / pw)  # широкий однострочный знак не должен вылезать за кадр
    sw, sh = max(8, int(pw * scale)), max(8, int(ph * scale))

    # Перспектива: углы знака смещаются до ~12% (сильный угол → тег angle).
    jitter = float(rng.uniform(0.0, 0.12))
    if jitter > 0.07:
        tags["angle"] = True
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = dst + rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32) * np.float32([sw, sh])
    angle = float(rng.uniform(-8, 8))
    rot = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
    dst = cv2.transform(dst.reshape(-1, 1, 2), rot).reshape(-1, 2)
    dst -= dst.min(axis=0)
    qw, qh = float(dst[:, 0].max()), float(dst[:, 1].max())

    # Положение: целиком внутри кадра и без пересечения с уже вклеенными знаками.
    x0 = y0 = 0.0
    for _ in range(20):
        x0 = float(rng.uniform(2, max(3, fw - qw - 2)))
        y0 = float(rng.uniform(2, max(3, fh - qh - 2)))
        if not any(x0 < ox1 and x0 + qw > ox0 and y0 < oy1 and y0 + qh > oy0 for ox0, oy0, ox1, oy1 in occupied):
            break
    dst += np.float32([x0, y0])
    dst[:, 0] = np.clip(dst[:, 0], 0, fw - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, fh - 1)
    occupied.append((x0, y0, x0 + qw, y0 + qh))

    # «Кузов»: малонасыщенный прямоугольник с мягким краем вокруг знака, чтобы он не висел в воздухе.
    cx, cy = x0 + qw / 2, y0 + qh / 2
    base = int(rng.integers(15, 190))
    body_color = [int(np.clip(base + rng.integers(-25, 25), 0, 255)) for _ in range(3)]
    bw, bh = qw * float(rng.uniform(1.6, 3.5)), qh * float(rng.uniform(2.0, 5.0))
    bx0, by0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    bx1, by1 = int(min(fw, cx + bw / 2)), int(min(fh, cy + bh / 2))
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx0, by0), (bx1, by1), body_color, -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, M, (fw, fh), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (fw, fh), flags=cv2.INTER_LINEAR)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask_f = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0)[..., None]
    frame = (warped.astype(np.float32) * mask_f + frame.astype(np.float32) * (1 - mask_f)).astype(np.uint8)
    return frame, dst, tags


def augment_frame(frame: np.ndarray, quad: np.ndarray, rng: np.random.Generator, tags: dict) -> np.ndarray:
    """Фотометрические аугментации всего кадра; quad не меняется."""
    img = frame
    xs, ys = quad[:, 0], quad[:, 1]
    px0, py0, px1, py1 = int(xs.min()), int(ys.min()), int(np.ceil(xs.max())), int(np.ceil(ys.max()))

    # Освещение: ночь / день / пересвет.
    roll = rng.random()
    if roll < 0.25:
        img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.35, 0.65)), beta=float(rng.uniform(-50, -10)))
        tags["night"] = True
    elif roll < 0.35:
        img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(1.1, 1.4)), beta=float(rng.uniform(10, 50)))
        tags["day"] = True
    else:
        img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.8, 1.15)), beta=float(rng.uniform(-20, 20)))
        tags["day"] = True

    # Блик: яркое эллиптическое пятно на знаке.
    if rng.random() < 0.15 and px1 > px0 and py1 > py0:
        overlay = img.copy()
        center = (int(rng.integers(px0, px1 + 1)), int(rng.integers(py0, py1 + 1)))
        axes = (max(2, (px1 - px0) // int(rng.integers(3, 7))), max(2, (py1 - py0) // 2))
        cv2.ellipse(overlay, center, axes, float(rng.uniform(-30, 30)), 0, 360, (255, 255, 255), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        tags["glare"] = True

    # Грязь: тёмные полупрозрачные пятна на знаке.
    if rng.random() < 0.2 and px1 > px0 and py1 > py0:
        overlay = img.copy()
        for _ in range(int(rng.integers(1, 5))):
            center = (int(rng.integers(px0, px1 + 1)), int(rng.integers(py0, py1 + 1)))
            radius = max(2, int((py1 - py0) * float(rng.uniform(0.1, 0.4))))
            cv2.circle(overlay, center, radius, (25, 22, 18), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        tags["dirt"] = True

    # Дождь/снег: штрихи или белые точки по кадру.
    roll = rng.random()
    if roll < 0.08:
        overlay = img.copy()
        for _ in range(int(rng.integers(100, 400))):
            x, y = int(rng.integers(0, img.shape[1])), int(rng.integers(0, img.shape[0]))
            cv2.line(overlay, (x, y), (x + 2, y + int(rng.integers(6, 16))), (200, 200, 210), 1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        tags["rain"] = True
    elif roll < 0.14:
        pts = rng.integers(0, [img.shape[1], img.shape[0]], size=(int(rng.integers(200, 700)), 2))
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), (235, 235, 240), -1)
        tags["snow"] = True

    # Смаз.
    roll = rng.random()
    if roll < 0.35:
        pass
    elif roll < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.2)))
    else:
        k = int(rng.choice([5, 7, 9]))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)
        tags["motion_blur"] = True

    noise = rng.normal(0, float(rng.uniform(1.0, 6.0)), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Запись датасета
# ---------------------------------------------------------------------------

META_COLUMNS = ("image", "plate_num", "plate_type", "bbox", "quad",
                "is_vehicle", "is_synthetic", "source", "license", "conditions")
CONDITION_ORDER = ("day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle")


def generate(count: int, out_dir: str, *, seed: int, types: tuple[str, ...] = PLATE_TYPES,
             font: str | None = None, backgrounds_dir: str | None = None,
             frame_size: tuple[int, int] = (1280, 720), prefix: str = "syn",
             jpeg_quality: tuple[int, int] = (45, 92), start_index: int = 1) -> list[dict]:
    """Генерирует count кадров и пишет images/synthetic, labels, meta.csv. Возвращает строки meta."""
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "images", "synthetic")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    backgrounds: list[str] = []
    if backgrounds_dir:
        for dirpath, _d, files in os.walk(backgrounds_dir):
            backgrounds += [os.path.join(dirpath, f) for f in sorted(files) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not backgrounds:
            raise FileNotFoundError(f"В {backgrounds_dir} нет изображений")

    meta_path = os.path.join(out_dir, "meta.csv")
    write_header = not os.path.isfile(meta_path) or os.path.getsize(meta_path) == 0
    rows: list[dict] = []
    seen: set[str] = set()
    with open(meta_path, "a", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=META_COLUMNS, delimiter=";", lineterminator="\n")
        if write_header:
            writer.writeheader()
        idx = start_index
        n_frames = 0
        while n_frames < count:
            plate_type = types[int(rng.integers(0, len(types)))]
            text = random_plate_text(rng, plate_type)
            if text in seen:
                continue
            seen.add(text)

            # Небольшая часть кадров — с двумя знаками, чтобы решение умело выдавать несколько строк.
            n_plates = 2 if rng.random() < 0.1 else 1
            fw, fh = frame_size
            # Разброс размера кадра под разные камеры.
            if rng.random() < 0.3:
                fw, fh = (640, 480) if rng.random() < 0.5 else (1920, 1080)
            frame = make_background(rng, (fw, fh), backgrounds)
            plates: list[tuple[str, str, np.ndarray]] = []
            texts = [text]
            ptypes = [plate_type]
            for _ in range(n_plates - 1):
                t2 = types[int(rng.integers(0, len(types)))]
                x2 = random_plate_text(rng, t2)
                if x2 in seen:
                    continue
                seen.add(x2)
                texts.append(x2)
                ptypes.append(t2)
            tags: dict[str, bool] = {}
            occupied: list[tuple[float, float, float, float]] = []
            for txt, pt in zip(texts, ptypes):
                clean = render_plate(txt, pt, font_path=font)
                frame, quad, t = paste_plate(frame, clean, rng, occupied)
                tags.update(t)
                plates.append((txt, pt, quad))
            frame = augment_frame(frame, plates[0][2], rng, tags)
            quality = int(rng.integers(jpeg_quality[0], jpeg_quality[1] + 1))
            name = f"{prefix}_{idx:06d}.jpg"
            idx += 1
            n_frames += 1
            ok = cv2.imwrite(os.path.join(img_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise OSError(f"Не удалось записать {name}")

            conditions = ",".join(c for c in CONDITION_ORDER if tags.get(c))
            label_lines: list[str] = []
            for txt, pt, quad in plates:
                q = np.round(quad).astype(int)
                x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
                x1, y1 = int(q[:, 0].max()), int(q[:, 1].max())
                bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
                row = {
                    "image": f"images/synthetic/{name}",
                    "plate_num": txt,
                    "plate_type": pt,
                    "bbox": ",".join(map(str, bbox)),
                    "quad": ",".join(str(int(v)) for v in q.flatten()),
                    "is_vehicle": "1",
                    "is_synthetic": "1",
                    "source": f"generate_synthetic_plates.py seed={seed}",
                    "license": "CC-BY-4.0",
                    "conditions": conditions,
                }
                writer.writerow(row)
                rows.append(row)
                cx, cy = (x0 + bbox[2] / 2) / fw, (y0 + bbox[3] / 2) / fh
                quad_norm = " ".join(f"{q[i, 0] / fw:.6f} {q[i, 1] / fh:.6f}" for i in range(4))
                label_lines.append(f"{LABEL_CLASS[pt]} {cx:.6f} {cy:.6f} {bbox[2] / fw:.6f} {bbox[3] / fh:.6f} {quad_norm}")
            with open(os.path.join(lbl_dir, name[:-4] + ".txt"), "w", encoding="utf-8") as lf:
                lf.write("\n".join(label_lines) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация синтетических ГРЗ типов 1 / 1А / 1Б (Волга-IT 2026)")
    parser.add_argument("--count", type=int, default=100, help="сколько кадров сгенерировать")
    parser.add_argument("--out", default="dataset", help="корень датасета (создаются images/synthetic, labels, meta.csv)")
    parser.add_argument("--seed", type=int, default=42, help="random seed — фиксируется для воспроизводимости")
    parser.add_argument("--types", default="type1,type1a,type1b", help="какие типы генерировать, через запятую")
    parser.add_argument("--font", default=None, help="путь к TTF (RoadNumbers.ttf и т.п.)")
    parser.add_argument("--backgrounds", default=None, help="каталог с фотографиями для фона (иначе синтетический фон)")
    parser.add_argument("--frame", default="1280x720", help="базовый размер кадра WxH")
    parser.add_argument("--prefix", default="syn", help="префикс имён файлов")
    parser.add_argument("--start-index", type=int, default=1, help="с какого номера нумеровать файлы")
    args = parser.parse_args()

    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    bad = [t for t in types if t not in PLATE_TYPES]
    if bad:
        parser.error(f"неизвестные типы {bad}; допустимо {PLATE_TYPES}")
    fw, fh = (int(v) for v in args.frame.lower().split("x"))
    rows = generate(args.count, args.out, seed=args.seed, types=types, font=args.font,
                    backgrounds_dir=args.backgrounds, frame_size=(fw, fh), prefix=args.prefix,
                    start_index=args.start_index)
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["plate_type"]] = by_type.get(r["plate_type"], 0) + 1
    print(f"Сгенерировано {args.count} кадров, {len(rows)} знаков {by_type} → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
