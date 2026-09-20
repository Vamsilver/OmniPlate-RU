"""
Refined Plate Renderer with photorealistic base templates and exact character coordinates.
Calibrated strictly to GOST R 50577-2018 dimensions and real physical references.
Supports:
  - type1: Single-line white (1040x224 px)
  - type1a: Two-line square white (558x331 px) with authentic region compartment box
  - type1b: Single-line yellow (1040x224 px)
All three conform to competition mask: [Letter][3 Digits][2 Letters][Region 2-3 Digits]
"""

import os
import random
from typing import Tuple
import cv2
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

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def draw_char_exact(draw: ImageDraw.ImageDraw, char: str, x: int, y: int, font: ImageFont.FreeTypeFont, fill=(15, 15, 15)):
    """Draws character aligned exactly to top-left of ink bounding box (x, y)."""
    bx, by, _, _ = font.getbbox(char)
    draw.text((x - bx, y - by), char, font=font, fill=fill)


class PlateRenderer:
    def __init__(self):
        font_path = os.path.join(FONTS_DIR, "RoadNumbers2.0.ttf")
        font_arial = os.path.join(FONTS_DIR, "arialbd.ttf")

        # Type 1 & Type 1B fonts (1040x224)
        self.font_main = ImageFont.truetype(font_path, 235)
        self.font_reg_2 = ImageFont.truetype(font_path, 170)
        self.font_reg_3 = ImageFont.truetype(font_path, 135)
        self.font_rus_long = ImageFont.truetype(font_arial, 38)

        # Type 1A fonts (558x331) - calibrated exactly to reference
        self.font_1a_letter = ImageFont.truetype(font_path, 218)
        self.font_1a_digit = ImageFont.truetype(font_path, 168)
        self.font_1a_reg_2 = ImageFont.truetype(font_path, 128)
        self.font_1a_reg_3 = ImageFont.truetype(font_path, 102)
        self.font_1a_rus = ImageFont.truetype(font_arial, 25)

    @staticmethod
    def generate_random_plate_text() -> Tuple[str, str, str, str, str]:
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = f"{random.randint(1, 999):03d}"
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

        w, h = 1040, 224
        im = Image.new("RGB", (w, h), (248, 248, 248))
        d = ImageDraw.Draw(im)

        # Borders
        d.rounded_rectangle([6, 6, w - 7, h - 7], radius=24, outline=(15, 15, 15), width=8)
        d.rounded_rectangle([18, 18, w - 19, h - 19], radius=16, outline=(230, 230, 230), width=3)
        # Separator line
        d.line([(770, 14), (770, h - 14)], fill=(15, 15, 15), width=6)

        # Bolt holes
        for bx, by in [(45, 112), (1000, 112)]:
            d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(40, 40, 40), outline=(180, 180, 180), width=2)

        # Characters
        draw_char_exact(d, l1, 75, 68, self.font_main)
        draw_char_exact(d, d3[0], 190, 34, self.font_main)
        draw_char_exact(d, d3[1], 295, 34, self.font_main)
        draw_char_exact(d, d3[2], 400, 34, self.font_main)
        draw_char_exact(d, l2[0], 525, 68, self.font_main)
        draw_char_exact(d, l2[1], 635, 68, self.font_main)

        # Region
        if len(reg) == 2:
            draw_char_exact(d, reg[0], 810, 25, self.font_reg_2)
            draw_char_exact(d, reg[1], 900, 25, self.font_reg_2)
        else:
            draw_char_exact(d, reg[0], 780, 38, self.font_reg_3)
            draw_char_exact(d, reg[1], 845, 38, self.font_reg_3)
            draw_char_exact(d, reg[2], 910, 38, self.font_reg_3)

        # RUS text
        d.text((805, 155), "RUS", font=self.font_rus_long, fill=(15, 15, 15))

        # Russian flag
        fx, fy, fw, fh = 900, 155, 70, 36
        sh = fh // 3
        d.rectangle([fx, fy, fx + fw, fy + sh], fill=(255, 255, 255))
        d.rectangle([fx, fy + sh, fx + fw, fy + 2 * sh], fill=(0, 57, 166))
        d.rectangle([fx, fy + 2 * sh, fx + fw, fy + fh], fill=(213, 43, 30))
        d.rectangle([fx, fy, fx + fw, fy + fh], outline=(15, 15, 15), width=1)

        return im, full_str

    @staticmethod
    def generate_random_bus_plate_text() -> Tuple[str, str, str, str]:
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
        d3 = f"{random.randint(1, 999):03d}"
        region = random.choice(POPULAR_REGIONS)
        full_str = f"{l2}{d3}{region}"
        return full_str, l2, d3, region

    def render_type1b(self, plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_num is None:
            full_str, l2, d3, reg = self.generate_random_bus_plate_text()
        else:
            full_str = plate_num
            l2 = plate_num[:2]
            d3 = plate_num[2:5]
            reg = plate_num[5:]

        w, h = 1040, 224
        # Pantone 116C: RGB (255, 204, 0)
        im = Image.new("RGB", (w, h), (255, 204, 0))
        d = ImageDraw.Draw(im)

        # Borders
        d.rounded_rectangle([6, 6, w - 7, h - 7], radius=24, outline=(15, 15, 15), width=8)
        d.rounded_rectangle([18, 18, w - 19, h - 19], radius=16, outline=(240, 190, 0), width=3)
        d.line([(770, 14), (770, h - 14)], fill=(15, 15, 15), width=6)

        for bx, by in [(45, 112), (1000, 112)]:
            d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(50, 40, 0), outline=(180, 140, 0), width=2)

        # Characters: GOST Type 1B is strictly 2 letters then 3 digits (LL DDD RR)
        draw_char_exact(d, l2[0], 90, 68, self.font_main)
        draw_char_exact(d, l2[1], 200, 68, self.font_main)
        draw_char_exact(d, d3[0], 345, 34, self.font_main)
        draw_char_exact(d, d3[1], 465, 34, self.font_main)
        draw_char_exact(d, d3[2], 585, 34, self.font_main)

        # Region
        if len(reg) == 2:
            draw_char_exact(d, reg[0], 810, 25, self.font_reg_2)
            draw_char_exact(d, reg[1], 900, 25, self.font_reg_2)
        else:
            draw_char_exact(d, reg[0], 780, 38, self.font_reg_3)
            draw_char_exact(d, reg[1], 845, 38, self.font_reg_3)
            draw_char_exact(d, reg[2], 910, 38, self.font_reg_3)

        # RUS text (No flag for 1B as per GOST)
        d.text((835, 155), "RUS", font=self.font_rus_long, fill=(15, 15, 15))

        return im, full_str

    def render_type1a(self, plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_num is None:
            full_str, l1, d3, l2, reg = self.generate_random_plate_text()
        else:
            full_str = plate_num
            l1, d3, l2, reg = plate_num[0], plate_num[1:4], plate_num[4:6], plate_num[6:]

        w, h = 558, 331
        color = (15, 15, 15)

        # Base image
        pil_im = Image.new("RGB", (w, h), (252, 252, 252))
        d_temp = ImageDraw.Draw(pil_im)
        d_temp.rounded_rectangle([7, 7, w - 8, h - 8], radius=22, outline=color, width=7)

        # OpenCV overlay for antialiased region compartment curves
        img = np.array(pil_im)
        x_corner = 328
        y_line = 166
        R = 20
        y_bot = h - 8

        # 1. Top horizontal line of region box
        cv2.line(img, (x_corner + R, y_line), (w - 11, y_line), color, 6, cv2.LINE_AA)
        # 2. Rounded corner arc
        cv2.ellipse(img, (x_corner + R, y_line + R), (R, R), 0, 180, 270, color, 6, cv2.LINE_AA)
        # 3. Vertical line down
        cv2.line(img, (x_corner, y_line + R), (x_corner, y_bot - 10), color, 6, cv2.LINE_AA)
        # 4. Smooth fillet joining bottom border
        fillet_pts = np.array([
            [x_corner - 8, y_bot - 2],
            [x_corner + 8, y_bot - 2],
            [x_corner + 3, y_bot - 12],
            [x_corner - 3, y_bot - 12]
        ], dtype=np.int32)
        cv2.fillPoly(img, [fillet_pts], color, cv2.LINE_AA)

        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)

        # TOP ROW: Letter 1 + 3 Digits
        draw_char_exact(d, l1, 99, 36, self.font_1a_letter)
        draw_char_exact(d, d3[0], 232, 37, self.font_1a_digit)
        draw_char_exact(d, d3[1], 312, 37, self.font_1a_digit)
        draw_char_exact(d, d3[2], 392, 36, self.font_1a_digit)

        # BOTTOM ROW: 2 Letters
        draw_char_exact(d, l2[0], 84, 186, self.font_1a_letter)
        draw_char_exact(d, l2[1], 182, 188, self.font_1a_letter)

        # Region Code (Inside compartment)
        if len(reg) == 2:
            draw_char_exact(d, reg[0], 382, 180, self.font_1a_reg_2)
            draw_char_exact(d, reg[1], 444, 180, self.font_1a_reg_2)
        else:
            draw_char_exact(d, reg[0], 350, 182, self.font_1a_reg_3)
            draw_char_exact(d, reg[1], 405, 182, self.font_1a_reg_3)
            draw_char_exact(d, reg[2], 460, 182, self.font_1a_reg_3)

        # RUS + Flag
        fx, fy, fw, fh = 460, 277, 52, 25
        sh = fh // 3
        d.rectangle([fx, fy, fx + fw, fy + sh], fill=(255, 255, 255))
        d.rectangle([fx, fy + sh, fx + fw, fy + 2 * sh], fill=(0, 57, 166))
        d.rectangle([fx, fy + 2 * sh, fx + fw, fy + fh], fill=(213, 43, 30))
        d.rectangle([fx, fy, fx + fw, fy + fh], outline=(15, 15, 15), width=1)

        d.text((380, 274), "RUS", font=self.font_1a_rus, fill=(15, 15, 15))

        return im, full_str

    def render_type2(self, plate_num: str = None) -> Tuple[Image.Image, str]:
        """
        Renders Type 2 Russian trailer plate (GOST R 50577-2018):
        Format: LL DDDD RR (or RRR) - 2 letters, 4 digits, region code.
        """
        if plate_num is None:
            l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
            d4 = "".join(random.choices(DIGITS, k=4))
            reg = random.choice(POPULAR_REGIONS)
            full_str = f"{l2}{d4}{reg}"
        else:
            full_str = plate_num
            l2, d4, reg = plate_num[:2], plate_num[2:6], plate_num[6:]

        w, h = 1040, 224
        im = Image.new("RGB", (w, h), (248, 248, 248))
        d = ImageDraw.Draw(im)

        # Borders
        d.rounded_rectangle([6, 6, w - 7, h - 7], radius=24, outline=(15, 15, 15), width=8)
        d.rounded_rectangle([18, 18, w - 19, h - 19], radius=16, outline=(230, 230, 230), width=3)
        # Separator line
        d.line([(770, 14), (770, h - 14)], fill=(15, 15, 15), width=6)

        # Bolt holes
        for bx, by in [(45, 112), (1000, 112)]:
            d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(40, 40, 40), outline=(180, 180, 180), width=2)

        # Characters: 2 letters, 4 digits
        draw_char_exact(d, l2[0], 75, 68, self.font_main)
        draw_char_exact(d, l2[1], 180, 68, self.font_main)
        draw_char_exact(d, d4[0], 300, 34, self.font_main)
        draw_char_exact(d, d4[1], 405, 34, self.font_main)
        draw_char_exact(d, d4[2], 510, 34, self.font_main)
        draw_char_exact(d, d4[3], 615, 34, self.font_main)

        # Region
        if len(reg) == 2:
            draw_char_exact(d, reg[0], 810, 25, self.font_reg_2)
            draw_char_exact(d, reg[1], 900, 25, self.font_reg_2)
        else:
            draw_char_exact(d, reg[0], 780, 38, self.font_reg_3)
            draw_char_exact(d, reg[1], 845, 38, self.font_reg_3)
            draw_char_exact(d, reg[2], 910, 38, self.font_reg_3)

        # RUS text
        d.text((805, 155), "RUS", font=self.font_rus_long, fill=(15, 15, 15))

        # Russian flag
        fx, fy, fw, fh = 900, 155, 70, 36
        sh = fh // 3
        d.rectangle([fx, fy, fx + fw, fy + sh], fill=(255, 255, 255))
        d.rectangle([fx, fy + sh, fx + fw, fy + 2 * sh], fill=(0, 57, 166))
        d.rectangle([fx, fy + 2 * sh, fx + fw, fy + fh], fill=(213, 43, 30))
        d.rectangle([fx, fy, fx + fw, fy + fh], outline=(15, 15, 15), width=1)

        return im, full_str

    def render(self, plate_type: str = "type1", plate_num: str = None) -> Tuple[Image.Image, str]:
        if plate_type == "type1":
            return self.render_type1(plate_num)
        elif plate_type == "type1a":
            return self.render_type1a(plate_num)
        elif plate_type == "type1b":
            return self.render_type1b(plate_num)
        elif plate_type in ("type2", "trailer"):
            return self.render_type2(plate_num)
        else:
            raise ValueError(f"Unknown plate_type: {plate_type}")


if __name__ == "__main__":
    renderer = PlateRenderer()
    for p_type in ["type1", "type1a", "type1b"]:
        img, num = renderer.render(p_type)
        out_p = f"perfect_{p_type}_{num}.png"
        img.save(out_p)
        print(f"Generated {p_type}: {num} -> {out_p}")
