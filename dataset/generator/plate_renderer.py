"""
Refined Plate Renderer with photorealistic base templates and exact character coordinates.
Supports:
  - type1: Single-line white (1040x224 px)
  - type1a: Two-line square white (580x340 px)
  - type1b: Single-line yellow (1040x224 px)
All three conform to competition mask: [Letter][3 Digits][2 Letters][Region 2-3 Digits]
"""

import os
import random
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ALLOWED_LETTERS = list("ABEKMHOPCTYX")
DIGITS = list("0123456789")

POPULAR_REGIONS = [
    "77", "99", "97", "177", "199", "197", "777", "799", "797",
    "78", "98", "178", "198",
    "16", "116", "716",
    "23", "93", "123", "193",
    "63", "163", "763",
    "66", "96", "196",
    "52", "152", "54", "154",
    "61", "161", "761", "74", "174", "774",
    "25", "125", "38", "138", "02", "102", "702",
    "50", "90", "150", "190", "750", "790"
]

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def get_square_1a_template() -> Image.Image:
    """Generates clean high-res 580x340 square plate template for Type 1A"""
    w, h = 580, 340
    im = Image.new("RGB", (w, h), (248, 248, 248))
    d = ImageDraw.Draw(im)

    # Outer border & inner bevel
    d.rounded_rectangle([6, 6, w - 7, h - 7], radius=24, outline=(15, 15, 15), width=8)
    d.rounded_rectangle([18, 18, w - 19, h - 19], radius=16, outline=(230, 230, 230), width=3)

    # RUS & Russian Flag at bottom-right
    font_rus = ImageFont.truetype(os.path.join(FONTS_DIR, "arialbd.ttf"), 30)
    d.text((473, 210), "RUS", font=font_rus, fill=(15, 15, 15))

    # Flag 54x34
    fx, fy, fw, fh = 472, 244, 54, 34
    sh = fh // 3
    d.rectangle([fx, fy, fx + fw, fy + sh], fill=(255, 255, 255))
    d.rectangle([fx, fy + sh, fx + fw, fy + 2 * sh], fill=(0, 57, 166))
    d.rectangle([fx, fy + 2 * sh, fx + fw, fy + fh], fill=(213, 43, 30))
    d.rectangle([fx, fy, fx + fw, fy + fh], outline=(15, 15, 15), width=1)

    # Mounting bolt holes
    for bx, by in [(45, 170), (535, 170)]:
        d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(40, 40, 40), outline=(180, 180, 180), width=2)

    return im


def get_yellow_1b_template() -> Image.Image:
    """Generates procedural Type 1B yellow template (1040x224 px) matching competition mask"""
    w, h = 1040, 224
    # Pantone 116C: RGB (255, 204, 0)
    im = Image.new("RGB", (w, h), (255, 204, 0))
    d = ImageDraw.Draw(im)

    # Outer rounded border
    d.rounded_rectangle([6, 6, w - 7, h - 7], radius=24, outline=(15, 15, 15), width=8)
    d.rounded_rectangle([18, 18, w - 19, h - 19], radius=16, outline=(245, 195, 0), width=3)

    # Vertical separator line matching Type 1 geometry
    d.line([(675, 14), (675, h - 14)], fill=(15, 15, 15), width=6)

    # RUS text
    font_rus = ImageFont.truetype(os.path.join(FONTS_DIR, "arialbd.ttf"), 38)
    d.text((765, 154), "RUS", font=font_rus, fill=(15, 15, 15))

    # Mounting holes
    for bx, by in [(40, 112), (1000, 112)]:
        d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(50, 40, 0), outline=(160, 130, 0), width=2)

    return im


class PlateRenderer:
    def __init__(self):
        font_path = os.path.join(FONTS_DIR, "RoadNumbers2.0.ttf")
        self.font_path = font_path

        # Load / construct templates
        self.tpl_white_long = Image.open(os.path.join(ASSETS_DIR, "rus_white_long_lp.png")).convert("RGB").resize((1040, 224))
        self.tpl_yellow_long = get_yellow_1b_template().convert("RGB")
        self.tpl_square_1a = get_square_1a_template().convert("RGB")

        # Fonts calibrated for 1040x224 long plates
        self.font_long_digits = ImageFont.truetype(font_path, 135)
        self.font_long_letters = ImageFont.truetype(font_path, 105)
        self.font_long_region = ImageFont.truetype(font_path, 105)

        # Fonts calibrated for 580x340 square 1A plates
        self.font_1a_top_digits = ImageFont.truetype(font_path, 115)
        self.font_1a_top_letter = ImageFont.truetype(font_path, 95)
        self.font_1a_bot_letters = ImageFont.truetype(font_path, 95)
        self.font_1a_region = ImageFont.truetype(font_path, 95)

    @staticmethod
    def generate_random_plate_text() -> Tuple[str, str, str, str, str]:
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
        region = random.choice(POPULAR_REGIONS)
        full_str = f"{l1}{d3}{l2}{region}"
        return full_str, l1, d3, l2, region

    def render_type1(self, plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_num is None:
            full_str, l1, d3, l2, reg = self.generate_random_plate_text()
        else:
            full_str = plate_num
            l1, d3, l2, reg = plate_num[0], plate_num[1:4], plate_num[4:6], plate_num[6:]

        im = self.tpl_white_long.copy()
        d = ImageDraw.Draw(im)

        # Letter 1
        d.text((65, 58), l1, font=self.font_long_letters, fill=(20, 20, 20))

        # Digits (3)
        dx_start = 160
        dx_spacing = 95
        for i, digit in enumerate(d3):
            d.text((dx_start + i * dx_spacing, 42), digit, font=self.font_long_digits, fill=(20, 20, 20))

        # Letters 2 & 3
        lx_start = 470
        d.text((lx_start, 58), l2[0], font=self.font_long_letters, fill=(20, 20, 20))
        d.text((lx_start + 90, 58), l2[1], font=self.font_long_letters, fill=(20, 20, 20))

        # Region Code
        if len(reg) == 2:
            d.text((745, 38), reg[0], font=self.font_long_region, fill=(20, 20, 20))
            d.text((830, 38), reg[1], font=self.font_long_region, fill=(20, 20, 20))
        else:
            d.text((705, 38), reg[0], font=self.font_long_region, fill=(20, 20, 20))
            d.text((780, 38), reg[1], font=self.font_long_region, fill=(20, 20, 20))
            d.text((855, 38), reg[2], font=self.font_long_region, fill=(20, 20, 20))

        return im, full_str

    def render_type1b(self, plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_num is None:
            full_str, l1, d3, l2, reg = self.generate_random_plate_text()
        else:
            full_str = plate_num
            l1, d3, l2, reg = plate_num[0], plate_num[1:4], plate_num[4:6], plate_num[6:]

        im = self.tpl_yellow_long.copy()
        d = ImageDraw.Draw(im)

        # Letter 1
        d.text((65, 58), l1, font=self.font_long_letters, fill=(20, 20, 20))

        # Digits (3)
        dx_start = 160
        dx_spacing = 95
        for i, digit in enumerate(d3):
            d.text((dx_start + i * dx_spacing, 42), digit, font=self.font_long_digits, fill=(20, 20, 20))

        # Letters 2 & 3
        lx_start = 470
        d.text((lx_start, 58), l2[0], font=self.font_long_letters, fill=(20, 20, 20))
        d.text((lx_start + 90, 58), l2[1], font=self.font_long_letters, fill=(20, 20, 20))

        # Region Code
        if len(reg) == 2:
            d.text((745, 38), reg[0], font=self.font_long_region, fill=(20, 20, 20))
            d.text((830, 38), reg[1], font=self.font_long_region, fill=(20, 20, 20))
        else:
            d.text((705, 38), reg[0], font=self.font_long_region, fill=(20, 20, 20))
            d.text((780, 38), reg[1], font=self.font_long_region, fill=(20, 20, 20))
            d.text((855, 38), reg[2], font=self.font_long_region, fill=(20, 20, 20))

        return im, full_str

    def render_type1a(self, plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_num is None:
            full_str, l1, d3, l2, reg = self.generate_random_plate_text()
        else:
            full_str = plate_num
            l1, d3, l2, reg = plate_num[0], plate_num[1:4], plate_num[4:6], plate_num[6:]

        im = self.tpl_square_1a.copy()
        d = ImageDraw.Draw(im)

        # TOP ROW: Letter 1 + 3 Digits
        d.text((55, 52), l1, font=self.font_1a_top_letter, fill=(20, 20, 20))

        d1a_start = 160
        d1a_spacing = 90
        for i, digit in enumerate(d3):
            d.text((d1a_start + i * d1a_spacing, 42), digit, font=self.font_1a_top_digits, fill=(20, 20, 20))

        # BOTTOM ROW: 2 Letters + Region
        d.text((55, 195), l2[0], font=self.font_1a_bot_letters, fill=(20, 20, 20))
        d.text((145, 195), l2[1], font=self.font_1a_bot_letters, fill=(20, 20, 20))

        # Region
        if len(reg) == 2:
            d.text((275, 185), reg[0], font=self.font_1a_region, fill=(20, 20, 20))
            d.text((355, 185), reg[1], font=self.font_1a_region, fill=(20, 20, 20))
        else:
            d.text((250, 185), reg[0], font=self.font_1a_region, fill=(20, 20, 20))
            d.text((320, 185), reg[1], font=self.font_1a_region, fill=(20, 20, 20))
            d.text((390, 185), reg[2], font=self.font_1a_region, fill=(20, 20, 20))

        return im, full_str

    def render(self, plate_type: str = "type1", plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_type == "type1":
            return self.render_type1(plate_num)
        elif plate_type == "type1a":
            return self.render_type1a(plate_num)
        elif plate_type == "type1b":
            return self.render_type1b(plate_num)
        else:
            raise ValueError(f"Unknown plate_type: {plate_type}")


if __name__ == "__main__":
    renderer = PlateRenderer()
    for p_type in ["type1", "type1a", "type1b"]:
        img, num = renderer.render(p_type)
        out_p = f"perfect_{p_type}_{num}.png"
        img.save(out_p)
        print(f"Generated {p_type}: {num} -> {out_p}")
