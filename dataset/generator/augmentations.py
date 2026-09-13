"""
Augmentation module for synthetic license plates.
Implements photorealistic physical defects:
  - Stamped metal embossing (3D light & shadow)
  - Mud, dust, and water splatter
  - Sun & headlight glare (radial and gradient lighting)
  - Sensor noise and optical blur
"""

import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class PlateAugmentor:
    def __init__(self):
        pass

    @staticmethod
    def apply_embossing(img: Image.Image) -> Image.Image:
        """Simulates 3D stamped metal relief with subtle highlight and shadow"""
        arr = np.array(img).astype(np.float32)

        # Kernel for directional lighting (top-left light source)
        kernel = np.array([
            [-1, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  1]
        ], dtype=np.float32) * 0.15

        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        relief = cv2.filter2D(gray, -1, kernel)

        # Blend relief with original
        relief_3c = np.stack([relief] * 3, axis=-1).astype(np.float32)
        blended = np.clip(arr + relief_3c - 128 * 0.15, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    @staticmethod
    def apply_dirt_and_scratches(img: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Adds procedural mud, dust splatters, and road grime"""
        if random.random() > intensity:
            return img

        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]

        # Generate low-frequency procedural dirt mask
        dirt_mask = np.zeros((h, w), dtype=np.float32)

        # Random dirt patches
        num_patches = random.randint(3, 10)
        for _ in range(num_patches):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(15, w // 4)
            ry = random.randint(10, h // 3)
            patch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rx * 2 + 1, ry * 2 + 1))
            
            # Place patch safely
            x1, x2 = max(0, cx - rx), min(w, cx + rx + 1)
            y1, y2 = max(0, cy - ry), min(h, cy + ry + 1)
            px1 = rx - (cx - x1)
            px2 = px1 + (x2 - x1)
            py1 = ry - (cy - y1)
            py2 = py1 + (y2 - y1)

            val = random.uniform(0.15, 0.45)
            dirt_mask[y1:y2, x1:x2] += patch[py1:py2, px1:px2].astype(np.float32) * val

        # Blur the dirt mask for natural organic spread
        dirt_mask = cv2.GaussianBlur(dirt_mask, (25, 25), 0)
        dirt_mask = np.clip(dirt_mask, 0.0, 0.75)[:, :, np.newaxis]

        # Brownish / grayish road dust color
        dirt_color = np.array([random.randint(60, 110), random.randint(55, 95), random.randint(45, 80)], dtype=np.float32)

        arr = arr * (1.0 - dirt_mask) + dirt_color * dirt_mask
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def apply_lighting_gradient(img: Image.Image) -> Image.Image:
        """Adds natural uneven lighting, headlight flare or vignette"""
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]

        # Linear or radial gradient
        mode = random.choice(["linear", "radial", "none"])
        if mode == "linear":
            grad_1d = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.8, 1.25), w)
            grad = np.tile(grad_1d, (h, 1))[:, :, np.newaxis]
            arr = np.clip(arr * grad, 0, 255)
        elif mode == "radial":
            # Radial highlight (sun or headlight spot)
            cx, cy = random.randint(0, w), random.randint(0, h)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_dist = np.sqrt(w ** 2 + h ** 2)
            grad = 1.0 + random.uniform(-0.35, 0.4) * (1.0 - dist / max_dist)
            arr = np.clip(arr * grad[:, :, np.newaxis], 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def apply_sensor_noise_and_blur(img: Image.Image) -> Image.Image:
        """Adds camera sensor noise and slight motion/optical blur"""
        # Slight motion blur or Gaussian blur
        if random.random() < 0.35:
            radius = random.choice([0.5, 1.0, 1.5])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Sensor noise
        if random.random() < 0.4:
            arr = np.array(img).astype(np.float32)
            noise = np.empty(arr.shape, dtype=np.float32)
            cv2.randn(noise, 0, random.uniform(3, 10))
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def augment_plate(self, img: Image.Image) -> Image.Image:
        """Full augmentation pipeline on clean plate"""
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.apply_embossing(img)
        img = self.apply_dirt_and_scratches(img, intensity=0.55)
        img = self.apply_lighting_gradient(img)
        img = self.apply_sensor_noise_and_blur(img)
        return img
