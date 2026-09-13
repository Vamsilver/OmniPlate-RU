"""
3D Perspective Projection & Quad Corner Transformation.
Mathematically projects rendered license plates onto scene backgrounds
while computing exact ground-truth quadrilateral corners (quad) and BBox.
"""

import math
import random
from typing import Tuple, List
import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(self):
        pass

    @staticmethod
    def project_plate_onto_background(
        plate_img: np.ndarray,
        bg_img: np.ndarray,
        max_yaw: float = 38.0,
        max_pitch: float = 22.0,
        max_roll: float = 12.0
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Projects plate onto background using 3D homography.
        Returns:
          - composited_image: uint8 RGB numpy array
          - bbox: [x, y, w, h] in absolute integer pixels
          - quad: [x1, y1, x2, y2, x3, y3, x4, y4] clockwise from top-left
        """
        ph, pw = plate_img.shape[:2]
        bh, bw = bg_img.shape[:2]

        # 1. Target scale: plate width in scene between 160 and 420 px
        target_w = random.randint(180, min(500, bw // 2))
        scale = target_w / pw
        target_h = int(ph * scale)

        # 2. Random 3D Euler angles (degrees to radians)
        yaw = math.radians(random.uniform(-max_yaw, max_yaw))
        pitch = math.radians(random.uniform(-max_pitch, max_pitch))
        roll = math.radians(random.uniform(-max_roll, max_roll))

        # 3. Source coordinates centered at origin
        src_pts = np.array([
            [-target_w / 2, -target_h / 2, 0],
            [ target_w / 2, -target_h / 2, 0],
            [ target_w / 2,  target_h / 2, 0],
            [-target_w / 2,  target_h / 2, 0]
        ], dtype=np.float32)

        # 4. Rotation matrices
        # Rx (Pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch),  math.cos(pitch)]
        ])
        # Ry (Yaw)
        Ry = np.array([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ])
        # Rz (Roll)
        Rz = np.array([
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll),  math.cos(roll), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx

        # Rotate points
        rot_pts = (R @ src_pts.T).T

        # 5. Perspective projection (focal distance)
        focal_dist = 1200.0
        z_offset = focal_dist

        proj_pts = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            z = rot_pts[i, 2] + z_offset
            proj_pts[i, 0] = (rot_pts[i, 0] * focal_dist) / z
            proj_pts[i, 1] = (rot_pts[i, 1] * focal_dist) / z

        # 6. Target center location in background
        # Keep comfortably within background bounds with margin
        margin_x = int(target_w * 0.7)
        margin_y = int(target_h * 0.7)
        min_cx = margin_x
        max_cx = max(min_cx + 10, bw - margin_x)
        min_cy = margin_y
        max_cy = max(min_cy + 10, bh - margin_y)

        cx = random.randint(min_cx, max_cx)
        cy = random.randint(min_cy, max_cy)

        # Final destination 4 corner points
        dst_pts = proj_pts + np.array([cx, cy], dtype=np.float32)

        # Original plate 4 corners
        orig_corners = np.array([
            [0, 0],
            [pw - 1, 0],
            [pw - 1, ph - 1],
            [0, ph - 1]
        ], dtype=np.float32)

        # 7. Compute Homography matrix
        H = cv2.getPerspectiveTransform(orig_corners, dst_pts)

        # 8. Warp plate and alpha mask
        warped_plate = cv2.warpPerspective(
            plate_img, H, (bw, bh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        plate_mask = np.ones((ph, pw), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            plate_mask, H, (bw, bh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Slight anti-aliasing edge softening for realism
        soft_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0).astype(np.float32) / 255.0
        soft_mask_3c = np.stack([soft_mask] * 3, axis=-1)

        # Composite onto background
        composited = (warped_plate.astype(np.float32) * soft_mask_3c +
                      bg_img.astype(np.float32) * (1.0 - soft_mask_3c))
        composited = np.clip(composited, 0, 255).astype(np.uint8)

        # 9. Extract Quad and BBox coordinates
        # Quad: x1,y1,x2,y2,x3,y3,x4,y4 (clockwise from top-left)
        quad = []
        for i in range(4):
            qx = int(round(dst_pts[i, 0]))
            qy = int(round(dst_pts[i, 1]))
            # Clamp to image boundaries
            qx = max(0, min(bw - 1, qx))
            qy = max(0, min(bh - 1, qy))
            quad.extend([qx, qy])

        # BBox: x, y, w, h
        all_x = [quad[0], quad[2], quad[4], quad[6]]
        all_y = [quad[1], quad[3], quad[5], quad[7]]
        bx = min(all_x)
        by = min(all_y)
        bw_box = max(all_x) - bx
        bh_box = max(all_y) - by
        bbox = [bx, by, bw_box, bh_box]

        return composited, bbox, quad
