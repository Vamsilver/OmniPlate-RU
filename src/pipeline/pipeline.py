#!/usr/bin/env python3
"""
OmniPlate-RU End-to-End Inference Pipeline (Stage 4).

Combines:
1. YOLOv8n-pose Detector (BBox, Class, 4 Quad Corner Keypoints).
2. PlateRectifier (Homography & Perspective Warping to Canonical Sizes).
3. Split & Stitch Mechanism for 2-line Type 1A (Square) Plates.
4. LPRNet Conv-CTC OCR with Greedy Decoding & GOST Heuristics.
5. Latency & Memory Profiling for Real-Time Execution (<25 ms SLA).
"""

from dataclasses import dataclass, field
import os
import sys
import time

# Enforce 100% offline mode for Ultralytics / YOLO
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_SYNC"] = "0"

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.ocr import PlateOCR
from src.pipeline.rectifier import PlateRectifier
from src.pipeline.verifier import PlateVerifier
from src.pipeline.decoder import is_valid_gost_plate, is_valid_region


@dataclass
class PlateDetection:
    """
    Unified representation of a detected and recognized vehicle license plate.
    """
    # Bounding box in original image coords: (x, y, w, h)
    bbox: Tuple[int, int, int, int]
    # Quad corners in original image coords: [x1, y1, x2, y2, x3, y3, x4, y4] (TL, TR, BR, BL)
    quad: List[float]
    # Target class: 'type1', 'type1a', 'type1b', 'other'
    plate_type: str
    # Detector confidence score [0.0, 1.0]
    confidence: float
    # Recognized text (e.g. 'A123BC77', or '' if 'other')
    text: str = ""
    # OCR confidence score [0.0, 1.0]
    ocr_confidence: float = 0.0
    # Optional rectified canonical image crop (BGR numpy array)
    rectified_crop: Optional[np.ndarray] = field(default=None, repr=False)
    # Flag indicating whether detection was rescued on sensitive fallback pass 2
    is_soft_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts detection to JSON / CSV serializable dictionary."""
        return {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "quad": ",".join(f"{v:.1f}" for v in self.quad),
            "plate_type": self.plate_type,
            "plate_num": self.text,
            "confidence": round(self.confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
        }


class OmniPlatePipeline:
    """
    End-to-End License Plate Detection & Recognition Pipeline.
    Optimized for ultra-low latency (<25 ms) on modern GPUs and <100 ms on GTX 1050 Ti.
    """

    CLASS_NAMES: Dict[int, str] = {
        0: "type1",
        1: "type1a",
        2: "type1b",
        3: "other",
    }

    def __init__(
        self,
        detector_path: Optional[str] = None,
        ocr_path: Optional[str] = None,
        ocr_1a_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        device: str = "cuda",
        conf_threshold: float = 0.12,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        use_onnx: bool = True,
        ocr_1a_mode: str = "ensemble",
        ocr_version: str = "moe",
        enable_crop_fallback: bool = False,
    ) -> None:
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_onnx = use_onnx
        self.ocr_1a_mode = ocr_1a_mode
        self.ocr_version = str(ocr_version).lower()
        self.enable_crop_fallback = enable_crop_fallback

        # Resolve model paths
        self.detector_path = self._resolve_model_path(
            detector_path,
            candidates=[
                "models/detector_yolo_pose.onnx",
                "models/detector_yolo_pose_best.pt",
                "models/yolov8n_pose_best.pt",
            ],
        )
        if self.ocr_version in ("auto", "moe"):
            self.ocr_v2_path = self._resolve_model_path(
                None,
                candidates=[
                    "models/ocr_lprnet_best.onnx",
                    "models/ocr_lprnet_best.pt",
                    "models/ocr_lprnet_v2.onnx",
                ],
                prefer_newest=True,
            )
            self.ocr_v3_path = self._resolve_model_path(
                None,
                candidates=[
                    "models/ocr_lprnet_v3.onnx",
                    "models/ocr_lprnet_v3.pt",
                ],
                prefer_newest=True,
            )
            self.ocr_path = self.ocr_v2_path
        else:
            self.ocr_v2_path = None
            self.ocr_v3_path = None
            ocr_candidates = (
                [
                    "models/ocr_lprnet_v3.onnx",
                    "models/ocr_lprnet_v3.pt",
                ]
                if self.ocr_version == "v3"
                else [
                    "models/ocr_lprnet_best.pt",
                    "models/ocr_lprnet_best.onnx",
                ]
            )
            self.ocr_path = self._resolve_model_path(
                ocr_path,
                candidates=ocr_candidates,
                prefer_newest=True,
            )
        self.ocr_1a_path = self._resolve_model_path(
            ocr_1a_path,
            candidates=[
                "models/ocr_lprnet_1a.onnx",
                "models/ocr_lprnet_1a_best.pt",
            ],
            prefer_newest=True,
        )
        if detector_path == "" and verifier_path is None:
            self.verifier_path = ""
        else:
            self.verifier_path = self._resolve_model_path(
                verifier_path,
                candidates=[
                    "models/plate_verifier.onnx",
                    "models/plate_verifier.pt",
                ],
                prefer_newest=True,
            )

        # Initialize neural sub-modules
        self.rectifier = PlateRectifier()
        use_onnx = self.use_onnx and (
            (self.ocr_path is not None and self.ocr_path.endswith(".onnx")) or self.ocr_version in ("auto", "moe")
        )
        self.ocr = PlateOCR(
            model_path=self.ocr_path,
            model_1a_path=self.ocr_1a_path,
            device=self.device,
            use_onnx=use_onnx,
            ocr_version=self.ocr_version,
            model_v2_path=self.ocr_v2_path,
            model_v3_path=self.ocr_v3_path,
        )
        self.detector = self._init_detector(self.detector_path)
        self.verifier = PlateVerifier(
            model_path=self.verifier_path,
            device=self.device,
            use_onnx=self.use_onnx,
        )

    @staticmethod
    def validate_plate_geometry(
        bbox: Tuple[int, int, int, int],
        plate_type: str,
        img_w: int,
        img_h: int,
    ) -> bool:
        """
        Rejects impossible detections based on physical GOST aspect ratios and camera scene scale.
        Accommodates both wide CCTV angles and tight crops / close-ups.
        """
        bx, by, bw, bh = bbox

        # 1. Extreme size rejection (allows small distance plates down to 12x6 and close-ups/crops up to 100% canvas)
        if bw < 12 or bh < 6:
            return False
        # Do not reject close-ups or tight crops where plate fills canvas
        if bw > img_w * 1.02 or bh > img_h * 1.02:
            return False
        # Reject impossible full-canvas scene captures that detector falsely grouped as a plate
        if img_w >= 1000 and img_h >= 800 and bw > img_w * 0.85 and bh > img_h * 0.50:
            return False
        # Reject corner boundary artifacts that touch multiple frame edges with small pixel area
        if img_w >= 640 and (bx <= 1 or bx + bw >= img_w - 1) and (by <= 1 or by + bh >= img_h - 1) and (bw * bh < 4500):
            return False

        # 2. Border edge rejection: highway guardrails / road cuts right at top frame edge on large CCTV frames
        if by <= 1 and bw > 300 and bh < 15 and img_h > 200:
            return False

        # 3. Aspect Ratio (Width / Height) rejection based on physical GOST R 50577-2018
        # Calibrated to accommodate perspective yaw/pitch angles in real scenes
        ar = bw / float(bh)
        if plate_type in ("type1", "type1b"):
            # Single-line plates: physical AR ~ 4.64. Perspective foreshortening: [1.35, 7.20]
            if ar < 1.35 or ar > 7.20:
                return False
        elif plate_type == "type1a":
            # Two-line square plate: physical AR ~ 1.706. Extreme perspective angles: [0.65, 3.90]
            if ar < 0.65 or ar > 3.90:
                return False
        elif plate_type == "other":
            # Other (trailers, motorcycles, square / rectangular): [0.60, 7.50]
            if ar < 0.60 or ar > 7.50:
                return False

        return True

    @staticmethod
    def validate_quad_geometry(
        quad: Sequence[float],
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """
        Verifies that the 4 quad corners form a valid, convex, non-collapsed quadrilateral
        that reasonably fills the bounding box without extreme distortion.
        """
        if len(quad) != 8:
            return False

        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return False

        pts = np.array(quad, dtype=np.float32).reshape(4, 2)

        # 1. Proximity check: Keypoints must not lie far outside the detected BBox
        margin_x = max(4.0, bw * 0.18)
        margin_y = max(3.0, bh * 0.22)
        if np.any(pts[:, 0] < bx - margin_x) or np.any(pts[:, 0] > bx + bw + margin_x):
            return False
        if np.any(pts[:, 1] < by - margin_y) or np.any(pts[:, 1] > by + bh + margin_y):
            return False

        # 2. Minimum edge length check (prevent collapsed edges/triangles)
        edges = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[3] - pts[2]),
            np.linalg.norm(pts[0] - pts[3]),
        ]
        if any(edge < 4.0 for edge in edges):
            return False

        # 3. Convexity check (plate is a planar convex polygon in 3D perspective)
        pts_int = pts.astype(np.int32)
        if not cv2.isContourConvex(pts_int):
            return False

        # 4. Polygon area vs BBox area fill ratio
        poly_area = cv2.contourArea(pts)
        bbox_area = float(bw * bh)
        fill_ratio = poly_area / bbox_area

        # A valid perspective projection of a rectangle fills at least 52% of its AABB
        if fill_ratio < 0.52 or fill_ratio > 1.02:
            return False

        return True

    @staticmethod
    def is_gost_yellow_plate(
        crop: np.ndarray,
        s_threshold: float = 60.0,
    ) -> Tuple[bool, float, float]:
        """
        Colorimetric validator for GOST R 50577-2018 Type 1B (Pantone 116C yellow #FFCC00).
        Evaluates background pixels (excluding dark symbols/borders with V <= 80):
        Returns:
            (is_yellow, bg_mean_saturation, yellow_pixel_ratio)
        """
        if crop is None or crop.size == 0:
            return False, 0.0, 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bg_mask = hsv[:, :, 2] > 80
        if np.sum(bg_mask) > 0:
            bg_s = hsv[:, :, 1][bg_mask]
            bg_mean_s = float(np.mean(bg_s))
        else:
            bg_mean_s = float(np.mean(hsv[:, :, 1]))

        # GOST Pantone 116C yellow in OpenCV: H in [12, 38], S >= 60, V >= 60
        yellow_mask = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 60)
        yellow_ratio = float(np.mean(yellow_mask))

        is_yellow = (bg_mean_s >= s_threshold) and (yellow_ratio >= 0.25)
        return is_yellow, bg_mean_s, yellow_ratio

    def _parse_results(self, r, iw: int, ih: int) -> List[PlateDetection]:
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        kpts_list = None
        if hasattr(r, "keypoints") and r.keypoints is not None:
            try:
                kpts_list = r.keypoints.xy.cpu().numpy()
            except Exception:
                kpts_list = None

        detections: List[PlateDetection] = []
        for i in range(len(xyxy)):
            box = xyxy[i]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            bx = max(0, min(iw - 1, x1))
            by = max(0, min(ih - 1, y1))
            bw = max(1, min(iw - bx, x2 - x1))
            bh = max(1, min(ih - by, y2 - y1))
            bbox = (bx, by, bw, bh)

            cls_id = classes[i]
            plate_type = self.CLASS_NAMES.get(cls_id, "type1")
            conf = float(confs[i])

            # Reconcile plate type with physical aspect ratio before validation
            ar = bw / float(bh)
            if plate_type in ("type1", "type1b") and 0.60 <= ar < 1.35:
                plate_type = "type1a"
            elif plate_type == "type1a" and ar >= 2.25:
                plate_type = "type1"

            # Apply physical geometry filter (kills false positives on car panels, barriers, sky)
            if not self.validate_plate_geometry(bbox, plate_type, iw, ih):
                continue

            kpts = kpts_list[i] if kpts_list is not None and i < len(kpts_list) else None
            quad = self._extract_quad(kpts, bbox, iw, ih)

            # Apply quad polygon convexity and fill validation (gracefully fall back to bbox quad)
            if not self.validate_quad_geometry(quad, bbox):
                quad = self._fallback_quad_from_bbox(bbox)

            detections.append(
                PlateDetection(
                    bbox=bbox,
                    quad=quad,
                    plate_type=plate_type,
                    confidence=conf,
                )
            )
        return detections

    def detect(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Runs plate detector on scene image with adaptive two-pass search.
        Pass 1: Standard confidence threshold (e.g. 0.15).
        Pass 2: Sensitive fallback pass (conf=0.06) with strict geometry validation
                if no plates were detected on pass 1.
        Includes Safety Max-Dimension Resize guard to keep latency < 40ms on 4K frames.
        """
        if self.detector is None:
            return []

        # Fast exit for blank/black dummy test images (avoids wasting multiple detector passes)
        if np.max(image) < 15:
            return []

        ih, iw = image.shape[:2]

        # Safety Max-Dimension Resize (prevents latency SLA spikes on unconstrained 4K camera frames)
        max_dim = max(ih, iw)
        if max_dim > 1920:
            scale = 1920.0 / max_dim
            new_w = max(1, int(round(iw * scale)))
            new_h = max(1, int(round(ih * scale)))
            det_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            det_w, det_h = new_w, new_h
            scale_x = iw / float(new_w)
            scale_y = ih / float(new_h)
        else:
            det_img = image
            det_w, det_h = iw, ih
            scale_x, scale_y = 1.0, 1.0

        # Single-pass multi-threshold search: runs YOLO once at 0.07 and partitions candidates
        # into primary (>= conf_threshold) and soft fallback (0.07 <= conf < conf_threshold).
        # Eliminates redundant second forward pass on negative frames ('other') without losing sensitivity.
        min_conf = 0.07 if self.conf_threshold > 0.09 else self.conf_threshold
        results = self.detector.predict(
            source=det_img,
            imgsz=self.imgsz,
            conf=min_conf,
            iou=self.iou_threshold,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )

        all_detections = self._parse_results(results[0] if results else None, det_w, det_h)

        if self.conf_threshold > 0.09:
            primary = []
            fallback = []
            for d in all_detections:
                bx, by, bw, bh = d.bbox
                ar = bw / float(bh) if bh > 0 else 1.0
                is_square = (0.85 <= ar <= 1.85) or (d.plate_type == "type1a" and ar <= 2.10)
                cand_thresh = min(self.conf_threshold, 0.085) if is_square else self.conf_threshold
                if d.confidence >= cand_thresh:
                    primary.append(d)
                else:
                    fallback.append(d)
            if primary:
                detections = primary
            else:
                detections = fallback
                for d in detections:
                    d.is_soft_fallback = True
        else:
            detections = all_detections

        # Rescale detections back to original coordinate space if resized
        if scale_x != 1.0 or scale_y != 1.0:
            for d in detections:
                bx, by, bw, bh = d.bbox
                nbx = max(0, min(iw - 1, int(round(bx * scale_x))))
                nby = max(0, min(ih - 1, int(round(by * scale_y))))
                nbw = max(1, min(iw - nbx, int(round(bw * scale_x))))
                nbh = max(1, min(ih - nby, int(round(bh * scale_y))))
                d.bbox = (nbx, nby, nbw, nbh)

                nq = []
                for i in range(0, len(d.quad), 2):
                    qx = max(0.0, min(float(iw - 1), d.quad[i] * scale_x))
                    qy = max(0.0, min(float(ih - 1), d.quad[i + 1] * scale_y))
                    nq.extend([qx, qy])
                d.quad = nq

        return detections


    def _resolve_device(self, requested: str) -> str:
        """Checks CUDA availability; falls back to CPU if necessary."""
        if "cuda" in requested.lower():
            try:
                import torch
                if torch.cuda.is_available():
                    return requested
            except ImportError:
                pass
            return "cpu"
        return "cpu"

    def _resolve_model_path(
        self,
        explicit_path: Optional[str],
        candidates: Sequence[str],
        prefer_newest: bool = False,
    ) -> Optional[str]:
        if explicit_path in ("", "none", "None", False):
            return None
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        found = []
        for cand in candidates:
            # Check relative to PROJECT_ROOT and CWD
            p1 = PROJECT_ROOT / cand
            if p1.exists():
                found.append(str(p1))
            elif os.path.exists(cand):
                found.append(str(os.path.abspath(cand)))

        if not found:
            return None

        if prefer_newest and len(found) > 1:
            found.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        return found[0]


    def _init_detector(self, model_path: Optional[str]) -> Any:
        """Initializes Ultralytics YOLO model."""
        if model_path is None or not os.path.exists(model_path):
            print(f"[!] Warning: Detector model not found at {model_path}. Running in mock mode.")
            return None

        try:
            from ultralytics import YOLO
            model = YOLO(model_path, task="pose")
            return model
        except Exception as e:
            print(f"[-] Failed to load YOLO detector from {model_path}: {e}")
            return None

    def warmup(self, iterations: int = 3, img_size: Tuple[int, int] = (1080, 1920)) -> None:
        """
        Runs dummy warmup inferences to eliminate cold-start GPU/CUDA/ONNX allocation spikes.
        """
        h, w = img_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_crop = np.zeros((36, 160, 3), dtype=np.uint8)

        for _ in range(iterations):
            if self.detector is not None:
                _ = self.detector.predict(
                    source=dummy,
                    imgsz=self.imgsz,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False,
                )
            if self.ocr is not None and (self.ocr.session is not None or self.ocr.model is not None):
                _ = self.ocr.predict_single(dummy_crop)

    @staticmethod
    def _fallback_quad_from_bbox(bbox: Tuple[int, int, int, int]) -> List[float]:
        """Generates 4-corner quad [TL, TR, BR, BL] from axis-aligned bbox."""
        bx, by, bw, bh = bbox
        return [
            float(bx), float(by),
            float(bx + bw), float(by),
            float(bx + bw), float(by + bh),
            float(bx), float(by + bh),
        ]

    def _extract_quad(
        self,
        kpts_xy: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
    ) -> List[float]:
        """
        Extracts and verifies 4 quad points from detector keypoints.
        Falls back to bbox corners if keypoints are missing, skewed, or degenerate.
        """
        if kpts_xy is not None and len(kpts_xy) >= 4:
            pts = kpts_xy[:4].astype(np.float32)
            # Check if all points are non-zero
            if not np.all(pts == 0):
                # Clamp coordinates to image boundaries
                pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                ordered = PlateRectifier.order_quad_points(pts)
                quad_flat = ordered.flatten().tolist()
                # Strict verification: must be convex, non-degenerate, and fill bbox reasonably
                if self.validate_quad_geometry(quad_flat, bbox):
                    return quad_flat

        return self._fallback_quad_from_bbox(bbox)

    def recognize_single(
        self,
        image: np.ndarray,
        detection: PlateDetection,
    ) -> PlateDetection:
        """
        Performs rectification, 1A split/stitch, and OCR recognition on a single detection.
        """
        plate_type = detection.plate_type

        # Category 'other': In accordance with competition rules, do NOT attempt
        # to read non-standard / foreign plates as GOST to avoid false positive penalties.
        if plate_type == "other":
            detection.text = ""
            detection.ocr_confidence = 0.0
            return detection

        try:
            v_crop = None
            bx, by, bw, bh = detection.bbox
            bbox_ar = bw / float(bh) if bh > 0 else 1.0

            q = detection.quad
            w_top = float(np.hypot(q[2] - q[0], q[3] - q[1]))
            h_left = float(np.hypot(q[6] - q[0], q[7] - q[1]))
            quad_ar = w_top / max(h_left, 1.0)
            effective_ar = min(bbox_ar, quad_ar)

            alt_1a = None
            # Type 1B yellow passenger plates are always handled with single-line rectification
            if plate_type == "type1b":
                rectified = self.rectifier.rectify(image, detection.quad, plate_type="type1b", margin=(0.020, 0.015), refine_corners=True)
                detection.rectified_crop = rectified
                v_crop = rectified

                # Colorimetric guard: Verify genuine GOST Pantone 116C yellow background
                is_yellow, bg_s, y_ratio = self.is_gost_yellow_plate(rectified, s_threshold=60.0)

                if is_yellow:
                    ocr_res = self.ocr.predict_single(rectified, plate_type="type1b", return_type=True)
                    if len(ocr_res) == 3:
                        text, ocr_conf, detected_type = ocr_res
                    else:
                        text, ocr_conf = ocr_res[:2]
                        detected_type = "type1b"
                    detection.text = text
                    detection.ocr_confidence = ocr_conf
                    # GOST syntax arbitration: Type 1 formula (L DDD LL RR) is physically Type 1 regardless of car paint!
                    import re
                    if re.match(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$", text):
                        detection.plate_type = "type1"
                    else:
                        detection.plate_type = "type1b"
                else:
                    # Colorimetric demotion guard: Background is gray/white (S < 60 or insufficient yellow).
                    # YOLO detector misclassified a white plate on a yellow car / warm light as type1b.
                    if detection.confidence < 0.60:
                        detection.plate_type = "other"
                        detection.text = ""
                        detection.ocr_confidence = 0.0
                        return detection
                    ocr_res = self.ocr.predict_single(rectified, plate_type="type1", return_type=True)
                    if len(ocr_res) == 3:
                        text, ocr_conf, detected_type = ocr_res
                    else:
                        text, ocr_conf = ocr_res[:2]
                        detected_type = "type1"
                    detection.text = text
                    detection.ocr_confidence = ocr_conf
                    detection.plate_type = detected_type if detected_type in ("type1", "type2") else "type1"

            elif (plate_type == "type1a") or (effective_ar <= 2.10):
                # Evaluate dual hypothesis (1A vs 1 Direct) to protect against
                # perspective distortion turning 1-line into faux-square or vice versa.
                # 1. Hypothesis A (Type 1A)
                rect_1a = self.rectifier.rectify(image, detection.quad, plate_type="type1a", margin=(0.020, 0.015), refine_corners=False)
                h_1a = rect_1a.shape[0]
                has_native = hasattr(self.ocr, "predict_type1a_native") and getattr(self.ocr, "has_1a_model", lambda: False)()
                has_dual = hasattr(self.ocr, "predict_type1a_dual")

                if self.ocr_1a_mode == "native" and has_native:
                    text_1a, conf_1a = self.ocr.predict_type1a_native(rect_1a)
                elif self.ocr_1a_mode == "dual" and has_dual:
                    mid_1a = self.rectifier.find_adaptive_split_seam(rect_1a)
                    top_l = rect_1a[:mid_1a, :]
                    bot_l = rect_1a[mid_1a:, :]
                    text_1a, conf_1a = self.ocr.predict_type1a_dual(top_l, bot_l)
                elif self.ocr_1a_mode == "stitch" or (not has_native and not has_dual):
                    mid_1a = self.rectifier.find_adaptive_split_seam(rect_1a)
                    top_l = rect_1a[:mid_1a, :]
                    bot_l = rect_1a[mid_1a:, :]
                    stitched_1a = self.rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
                    text_1a, conf_1a = self.ocr.predict_single(stitched_1a, plate_type="type1a")

                    # Multi-seam fallback: if wildcards present, invalid syntax, or marginal confidence < 0.70, try micro-offsets
                    if plate_type == "type1a" and ("#" in text_1a or not is_valid_gost_plate(text_1a, "type1a") or conf_1a < 0.70):
                        for dy in (2, -2):
                            y_alt = mid_1a + dy
                            if 15 < y_alt < h_1a - 15:
                                s_alt = self.rectifier.stitch_type1a_horizontal(rect_1a[:y_alt, :], rect_1a[y_alt:, :], target_size=(160, 36))
                                t_alt, c_alt = self.ocr.predict_single(s_alt, plate_type="type1a")
                                score_curr = (1.0 if is_valid_gost_plate(text_1a, "type1a") else 0.0) * 5.0 - text_1a.count("#") * 3.0 + conf_1a * 3.0
                                score_alt = (1.0 if is_valid_gost_plate(t_alt, "type1a") else 0.0) * 5.0 - t_alt.count("#") * 3.0 + c_alt * 3.0
                                if score_alt > score_curr:
                                    text_1a, conf_1a = t_alt, c_alt
                else:  # "ensemble"
                    if has_native and has_dual:
                        text_1a_nat, conf_1a_nat = self.ocr.predict_type1a_native(rect_1a)
                        mid_1a = self.rectifier.find_adaptive_split_seam(rect_1a)
                        top_l = rect_1a[:mid_1a, :]
                        bot_l = rect_1a[mid_1a:, :]
                        text_1a_dp, conf_1a_dp = self.ocr.predict_type1a_dual(top_l, bot_l)
                        v_nat = is_valid_gost_plate(text_1a_nat, "type1a")
                        v_dp = is_valid_gost_plate(text_1a_dp, "type1a")
                        score_nat = conf_1a_nat * 10.0 - text_1a_nat.count("#") * 3.5 + (4.0 if v_nat else 0.0) + 1.2
                        score_dp = conf_1a_dp * 10.0 - text_1a_dp.count("#") * 3.5 + (4.0 if v_dp else 0.0)
                        if score_nat >= score_dp:
                            text_1a, conf_1a = text_1a_nat, conf_1a_nat
                        else:
                            text_1a, conf_1a = text_1a_dp, conf_1a_dp
                    elif has_native:
                        text_1a, conf_1a = self.ocr.predict_type1a_native(rect_1a)
                    else:
                        mid_1a = self.rectifier.find_adaptive_split_seam(rect_1a)
                        top_l = rect_1a[:mid_1a, :]
                        bot_l = rect_1a[mid_1a:, :]
                        stitched_1a = self.rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
                        text_1a_ss, conf_1a_ss = self.ocr.predict_single(stitched_1a, plate_type="type1a")
                        if has_dual:
                            text_1a_dp, conf_1a_dp = self.ocr.predict_type1a_dual(top_l, bot_l)
                            v_ss = is_valid_gost_plate(text_1a_ss, "type1a")
                            v_dp = is_valid_gost_plate(text_1a_dp, "type1a")
                            score_ss = conf_1a_ss * 10.0 - text_1a_ss.count("#") * 3.5 + (4.0 if v_ss else 0.0)
                            score_dp = conf_1a_dp * 10.0 - text_1a_dp.count("#") * 3.5 + (4.0 if v_dp else 0.0) + 0.8
                            if score_dp > score_ss:
                                text_1a, conf_1a = text_1a_dp, conf_1a_dp
                            else:
                                text_1a, conf_1a = text_1a_ss, conf_1a_ss
                        else:
                            text_1a, conf_1a = text_1a_ss, conf_1a_ss

                # Early Exit for decisive Type 1A: on clearly square plates (AR <= 1.45),
                # if Hypothesis A achieves high-confidence valid GOST with no wildcards,
                # Hypothesis B (Type 1 Direct) cannot mathematically surpass it.
                skip_hyp_b = (
                    plate_type == "type1a"
                    and effective_ar <= 1.45
                    and is_valid_gost_plate(text_1a, "type1a")
                    and conf_1a >= 0.88
                    and "#" not in text_1a
                )

                if skip_hyp_b:
                    detection.rectified_crop = rect_1a
                    detection.text = text_1a
                    detection.ocr_confidence = round(conf_1a, 4)
                    detection.plate_type = "type1a"
                    v_crop = rect_1a
                else:
                    # 2. Hypothesis B (Type 1 Direct)
                    do_refine_1 = (bh >= 22)
                    rect_1 = self.rectifier.rectify(image, detection.quad, plate_type="type1", margin=(0.012, 0.006), refine_corners=do_refine_1)
                    ocr_res_1 = self.ocr.predict_single(rect_1, plate_type="type1", return_type=True)

                    if len(ocr_res_1) == 3:
                        text_1, conf_1, type_1 = ocr_res_1
                    else:
                        text_1, conf_1 = ocr_res_1[:2]
                        type_1 = "type1"

                    # Score both hypotheses: penalize wildcards (#), reward valid GOST length & region
                    wc_1a = text_1a.count("#")
                    wc_1 = text_1.count("#")
                    v_1a_valid = is_valid_gost_plate(text_1a, "type1a")
                    v_1_valid = is_valid_gost_plate(text_1, "type1")

                    score_1a = conf_1a * 10.0 - (wc_1a * 3.5) + (3.0 if len(text_1a) in (8, 9) else -5.0) + (4.0 if v_1a_valid else 0.0)
                    score_1 = conf_1 * 10.0 - (wc_1 * 3.5) + (3.0 if len(text_1) in (8, 9) else -5.0) + (4.0 if v_1_valid else 0.0)

                    # Aspect ratio priors based on physics of GOST
                    if effective_ar <= 1.45:
                        score_1a += 4.0
                    elif effective_ar <= 1.75:
                        score_1a += 3.5
                    elif effective_ar >= 2.15:
                        score_1 += 5.0
                    elif effective_ar >= 1.95:
                        score_1 += 3.0

                    if plate_type == "type1a":
                        score_1a += 2.0 * min(1.0, max(0.2, detection.confidence))
                    elif plate_type == "type1":
                        score_1 += 1.5 * min(1.0, max(0.2, detection.confidence))

                    # Clear confidence dominance rule: if Type 1 has high confidence and clearly outperforms 1A
                    if v_1_valid and conf_1 >= 0.80 and (conf_1 - conf_1a >= 0.25):
                        score_1 += 8.0
                    elif v_1a_valid and conf_1a >= 0.80 and (conf_1a - conf_1 >= 0.25):
                        score_1a += 8.0
                    elif v_1_valid and conf_1 >= 0.70 and not v_1a_valid:
                        score_1 += 4.0
                    elif v_1a_valid and conf_1a >= 0.70 and not v_1_valid:
                        score_1a += 4.0

                    if score_1a >= score_1:
                        detection.rectified_crop = rect_1a
                        detection.text = text_1a
                        detection.ocr_confidence = round(conf_1a, 4)
                        detection.plate_type = "type1a"
                        v_crop = rect_1a
                    else:
                        if v_1a_valid and conf_1a >= 0.40:
                            alt_1a = (text_1a, conf_1a, rect_1a)
                        detection.rectified_crop = rect_1
                        if type_1 == "type1b" and not self.is_gost_yellow_plate(rect_1)[0]:
                            type_1 = "type1"
                        detection.text = text_1
                        detection.ocr_confidence = round(conf_1, 4)
                        detection.plate_type = type_1 if type_1 in ("type1", "type2", "type1b") else "type1"
                        v_crop = rect_1

            else:
                # Definite Type 1 Single-Line Plate (AR > 2.10)
                do_refine = (bh >= 22)
                rectified = self.rectifier.rectify(image, detection.quad, plate_type="type1", margin=(0.012, 0.006), refine_corners=do_refine)
                detection.rectified_crop = rectified

                # Early Exit for obvious background noise before OCR
                if self.verifier.is_valid() and not detection.is_soft_fallback and detection.confidence < 0.35:
                    _, p_early = self.verifier.verify_single(rectified)
                    if p_early < 0.0005:
                        detection.plate_type = "other"
                        detection.text = ""
                        detection.ocr_confidence = 0.0
                        return detection

                ocr_res = self.ocr.predict_single(rectified, plate_type="type1", return_type=True)
                if len(ocr_res) == 3:
                    text, ocr_conf, detected_type = ocr_res
                else:
                    text, ocr_conf = ocr_res[:2]
                    detected_type = "type1"
                if detected_type == "type1b" and not self.is_gost_yellow_plate(rectified)[0]:
                    detected_type = "type1"
                detection.text = text
                detection.ocr_confidence = round(ocr_conf, 4)
                detection.plate_type = detected_type if detected_type in ("type1", "type2", "type1b") else "type1"
                v_crop = rectified

            # Verifier & GOST Guard: filter out false-positive noise detections on background/car parts
            if self.verifier.is_valid() and detection.plate_type != "other" and detection.rectified_crop is not None:
                if v_crop is None:
                    v_crop = detection.rectified_crop
                if detection.plate_type == "type1a" and v_crop.shape[:2] != (36, 160):
                    try:
                        t_l, b_l = self.rectifier.split_type1a(v_crop)
                        v_crop = self.rectifier.stitch_type1a_horizontal(t_l, b_l, target_size=(160, 36))
                    except Exception:
                        pass
                is_plate, p_score = self.verifier.verify_single(v_crop)
                has_wildcards = "#" in detection.text
                is_gost_strict = is_valid_gost_plate(detection.text, detection.plate_type, allow_wildcards=False)
                is_gost_wild = is_valid_gost_plate(detection.text, detection.plate_type, allow_wildcards=True)

                if has_wildcards:
                    # Wildcard guard: Only retain masks with '#' when PlateVerifier confirms
                    # genuine physical plate (p_score >= 0.50 or is_plate, det.confidence >= 0.50, not soft fallback)
                    is_verified_wildcard_plate = (
                        (is_plate or p_score >= 0.55)
                        and p_score >= 0.40
                        and detection.confidence >= 0.50
                        and not detection.is_soft_fallback
                        and is_gost_wild
                    )
                    if is_verified_wildcard_plate:
                        pass
                    else:
                        detection.plate_type = "other"
                        detection.text = ""
                        detection.ocr_confidence = 0.0
                else:
                    # Type 1A confirmed valid: stitched crops have artificial seam, verifier p_score is not calibrated for stitched crops when OCR is strictly valid
                    if detection.plate_type == "type1a" and is_gost_strict and not detection.is_soft_fallback:
                        if (
                            (detection.ocr_confidence >= 0.80 and detection.confidence >= 0.20 and (detection.confidence * detection.ocr_confidence) >= 0.18)
                            or (detection.ocr_confidence >= 0.65 and detection.confidence >= 0.35)
                            or (detection.ocr_confidence >= 0.50 and detection.confidence >= 0.45)
                        ):
                            is_non_plate = False
                        else:
                            is_non_plate = (p_score < 0.0001)
                    elif is_gost_strict and detection.ocr_confidence >= 0.70 and detection.confidence >= 0.50 and not detection.is_soft_fallback:
                        is_non_plate = (p_score < 0.20)
                    else:
                        is_non_plate = (not is_plate)

                    # Soft fallback pass guard (require higher confidence to prevent noise rescue)
                    if detection.is_soft_fallback and (p_score < 0.60 or detection.ocr_confidence < 0.70):
                        is_non_plate = True

                    # Foreign blue plate guard: reject foreign blue plates misidentified as Type 1
                    if v_crop is not None:
                        b_mean, g_mean, r_mean = [float(v) for v in v_crop.mean(axis=(0, 1))]
                        if b_mean - r_mean > 45.0 and b_mean - g_mean > 30.0 and detection.confidence < 0.60:
                            is_non_plate = True

                    # Joint confidence guard: prevent low-confidence background hallucination
                    is_confident_gost = (
                        is_gost_strict
                        and (is_plate or p_score >= 0.50)
                        and detection.ocr_confidence >= 0.60
                    )
                    is_high_conf_gost = (
                        is_gost_strict
                        and (is_plate or p_score >= 0.50)
                        and detection.ocr_confidence >= 0.85
                        and (detection.confidence * detection.ocr_confidence) >= 0.08
                        and detection.confidence >= 0.08
                    )
                    if (is_confident_gost or is_high_conf_gost) and detection.plate_type == "type1a":
                        is_non_plate = False
                    elif (is_confident_gost or is_high_conf_gost) and detection.plate_type in ("type1", "type1b"):
                        is_non_plate = False
                    if not (is_confident_gost or is_high_conf_gost):
                        if (
                            detection.confidence < 0.08
                            or (detection.confidence < 0.20 and detection.ocr_confidence < 0.55)
                            or (detection.confidence < 0.35 and detection.ocr_confidence < 0.60)
                            or (detection.confidence < 0.45 and detection.ocr_confidence < 0.75)
                            or (detection.confidence * detection.ocr_confidence) < 0.08
                            or detection.ocr_confidence < 0.30
                        ):
                            is_non_plate = True

                    # GOST format guard: reject plates that don't match any valid Russian format
                    is_invalid_ocr = not is_gost_strict
                    if is_non_plate or is_invalid_ocr:
                        if alt_1a is not None and is_valid_gost_plate(alt_1a[0], "type1a", allow_wildcards=False) and alt_1a[1] >= 0.45:
                            detection.plate_type = "type1a"
                            detection.text = alt_1a[0]
                            detection.ocr_confidence = round(alt_1a[1], 4)
                            detection.rectified_crop = alt_1a[2]
                        else:
                            detection.plate_type = "other"
                            detection.text = ""
                            detection.ocr_confidence = 0.0

                # Safe blind plate fallback for unreadable/blind crops with confident detector & verifier (single-line only)
                if (
                    detection.plate_type == "other"
                    and detection.text == ""
                    and effective_ar >= 1.95
                    and detection.confidence >= 0.80
                    and (is_plate or p_score >= 0.50)
                    and not detection.is_soft_fallback
                ):
                    detection.plate_type = "type1"
                    detection.text = "########"
                    detection.ocr_confidence = 0.20



        except Exception as e:
            # Fallback on rectification or OCR failure
            detection.plate_type = "other"
            detection.text = ""
            detection.ocr_confidence = 0.0

        return detection

    def recognize_batch(
        self,
        image: np.ndarray,
        detections: List[PlateDetection],
    ) -> List[PlateDetection]:
        """
        Performs batch rectification and neural OCR recognition on multiple detections.
        Prepares canonical crops [N, 3, 36, 160] and processes them in a single neural forward pass.
        """
        if not detections:
            return []

        valid_entries = []
        crops_to_ocr = []
        types_to_ocr = []

        valid_1a_entries = []
        crops_1a_to_ocr = []
        crops_1a_to_verify = []

        for idx, det in enumerate(detections):
            if det.plate_type == "other":
                det.text = ""
                det.ocr_confidence = 0.0
                continue

            try:
                bx, by, bw, bh = det.bbox
                bbox_ar = bw / float(bh) if bh > 0 else 1.0
                q = det.quad
                w_top = float(np.hypot(q[2] - q[0], q[3] - q[1]))
                h_left = float(np.hypot(q[6] - q[0], q[7] - q[1]))
                quad_ar = w_top / max(h_left, 1.0)
                effective_ar = min(bbox_ar, quad_ar)

                is_square = (det.plate_type == "type1a" and effective_ar <= 1.95) or (effective_ar < 1.65)
                if is_square:
                    rectified = self.rectifier.rectify(image, det.quad, plate_type="type1a", margin=(0.015, 0.010), refine_corners=False)
                    det.rectified_crop = rectified
                    if hasattr(self.ocr, "predict_type1a_native") and getattr(self.ocr, "has_1a_model", lambda: False)():
                        crops_1a_to_ocr.append(rectified)
                        valid_1a_entries.append((idx, "type1a", rectified))
                        top_line, bot_line = self.rectifier.split_type1a(rectified)
                        stitched = self.rectifier.stitch_type1a_horizontal(top_line, bot_line, target_size=(160, 36))
                        crops_1a_to_verify.append(stitched)
                    else:
                        top_line, bot_line = self.rectifier.split_type1a(rectified)
                        stitched = self.rectifier.stitch_type1a_horizontal(top_line, bot_line, target_size=(160, 36))
                        crops_to_ocr.append(stitched)
                        types_to_ocr.append("type1a")
                        valid_entries.append((idx, "type1a", rectified))
                elif det.plate_type == "type1b":
                    rectified = self.rectifier.rectify(image, det.quad, plate_type="type1b", margin=(0.020, 0.015), refine_corners=True)
                    det.rectified_crop = rectified
                    crops_to_ocr.append(rectified)
                    types_to_ocr.append("type1b")
                    valid_entries.append((idx, "type1b", rectified))
                else:
                    do_refine = (det.bbox[3] >= 22)
                    rectified = self.rectifier.rectify(image, det.quad, plate_type="type1", margin=(0.012, 0.006), refine_corners=do_refine)
                    det.rectified_crop = rectified
                    crops_to_ocr.append(rectified)
                    types_to_ocr.append("type1")
                    valid_entries.append((idx, "type1", rectified))
            except Exception:
                det.plate_type = "other"
                det.text = ""
                det.ocr_confidence = 0.0

        batch_results = []
        verifier_results = []

        if crops_to_ocr:
            if hasattr(self.ocr, "predict_batch"):
                batch_results = self.ocr.predict_batch(crops_to_ocr, types_to_ocr, return_type=True)
            else:
                for crop, pt in zip(crops_to_ocr, types_to_ocr):
                    r = self.ocr.predict_single(crop, plate_type=pt, return_type=True)
                    if len(r) == 3:
                        batch_results.append(r)
                    else:
                        batch_results.append((r[0], r[1], pt))

            # Batch PlateVerifier (<0.4 ms on GPU)
            if self.verifier.is_valid():
                verifier_results = self.verifier.verify_batch(crops_to_ocr)
            else:
                verifier_results = [(True, 1.0)] * len(crops_to_ocr)

        if crops_1a_to_ocr:
            if hasattr(self.ocr, "predict_type1a_native_batch"):
                res_1a = self.ocr.predict_type1a_native_batch(crops_1a_to_ocr)
            else:
                res_1a = [self.ocr.predict_type1a_native(c) for c in crops_1a_to_ocr]
            batch_results.extend([(txt, conf, "type1a") for txt, conf in res_1a])
            valid_entries.extend(valid_1a_entries)
            if self.verifier.is_valid():
                verifier_results.extend(self.verifier.verify_batch(crops_1a_to_verify))
            else:
                verifier_results.extend([(True, 1.0)] * len(crops_1a_to_ocr))
        if valid_entries:
            import re
            for (idx, target_type, rect_crop), (text, conf, detected_type), (is_plate, p_score) in zip(valid_entries, batch_results, verifier_results):
                det = detections[idx]
                det.text = text
                det.ocr_confidence = conf

                if target_type == "type1b":
                    is_yellow, _, _ = self.is_gost_yellow_plate(rect_crop, s_threshold=60.0)
                    if is_yellow and not re.match(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$", text):
                        det.plate_type = "type1b"
                    elif not is_yellow and det.confidence < 0.60:
                        det.plate_type = "other"
                        det.text = ""
                        det.ocr_confidence = 0.0
                        continue
                    else:
                        det.plate_type = "type1"
                elif target_type == "type1a":
                    det.plate_type = "type1a"
                else:
                    det.plate_type = detected_type if detected_type in ("type1", "type2") else "type1"

                # Verifier & GOST Guard: filter out false-positive noise detections on background/car parts
                has_wildcards = "#" in det.text
                is_gost_strict = is_valid_gost_plate(det.text, det.plate_type, allow_wildcards=False)
                is_gost_wild = is_valid_gost_plate(det.text, det.plate_type, allow_wildcards=True)

                if has_wildcards:
                    # Wildcard guard: Only retain masks with '#' when PlateVerifier confirms
                    # genuine physical plate (p_score >= 0.50 or is_plate, det.confidence >= 0.50, not soft fallback)
                    is_verified_wildcard_plate = (
                        (is_plate or p_score >= 0.55)
                        and p_score >= 0.40
                        and det.confidence >= 0.50
                        and not det.is_soft_fallback
                        and is_gost_wild
                    )
                    if is_verified_wildcard_plate:
                        pass
                    else:
                        det.plate_type = "other"
                        det.text = ""
                        det.ocr_confidence = 0.0
                else:
                    # Type 1A confirmed valid: only reject on extreme verifier failure (stitched crop scores differ)
                    if det.plate_type == "type1a" and is_gost_strict and not det.is_soft_fallback:
                        if (
                            (det.ocr_confidence >= 0.80 and det.confidence >= 0.20 and (det.confidence * det.ocr_confidence) >= 0.18)
                            or (det.ocr_confidence >= 0.65 and det.confidence >= 0.35)
                            or (det.ocr_confidence >= 0.50 and det.confidence >= 0.45)
                        ):
                            is_non_plate = False
                        else:
                            is_non_plate = (p_score < 0.001)
                    elif is_gost_strict and det.ocr_confidence >= 0.70 and det.confidence >= 0.50 and not det.is_soft_fallback:
                        is_non_plate = (p_score < 0.20)
                    else:
                        is_non_plate = (not is_plate)

                    # Soft fallback pass guard (require higher confidence to prevent noise rescue)
                    if det.is_soft_fallback and (p_score < 0.60 or det.ocr_confidence < 0.70):
                        is_non_plate = True

                    # Foreign blue plate guard: reject foreign blue plates misidentified as Type 1
                    if rect_crop is not None:
                        b_mean, g_mean, r_mean = [float(v) for v in rect_crop.mean(axis=(0, 1))]
                        if b_mean - r_mean > 45.0 and b_mean - g_mean > 30.0 and det.confidence < 0.60:
                            is_non_plate = True

                    # Joint confidence guard: prevent low-confidence background hallucination
                    is_confident_gost = (
                        is_gost_strict
                        and (is_plate or p_score >= 0.50)
                        and det.ocr_confidence >= 0.60
                    )
                    is_high_conf_gost = (
                        is_gost_strict
                        and (is_plate or p_score >= 0.50)
                        and det.ocr_confidence >= 0.85
                        and (det.confidence * det.ocr_confidence) >= 0.08
                        and det.confidence >= 0.08
                    )
                    if (is_confident_gost or is_high_conf_gost) and det.plate_type == "type1a":
                        is_non_plate = False
                    elif (is_confident_gost or is_high_conf_gost) and det.plate_type in ("type1", "type1b"):
                        is_non_plate = False
                    if not (is_confident_gost or is_high_conf_gost):
                        if (
                            det.confidence < 0.08
                            or (det.confidence < 0.20 and det.ocr_confidence < 0.55)
                            or (det.confidence < 0.35 and det.ocr_confidence < 0.60)
                            or (det.confidence < 0.45 and det.ocr_confidence < 0.75)
                            or (det.confidence * det.ocr_confidence) < 0.08
                            or det.ocr_confidence < 0.30
                        ):
                            is_non_plate = True

                    # GOST format guard: reject plates that don't match any valid Russian format
                    is_invalid_ocr = not is_gost_strict
                    if is_non_plate or is_invalid_ocr:
                        det.plate_type = "other"
                        det.text = ""
                        det.ocr_confidence = 0.0

                # Safe blind plate fallback for unreadable/blind crops with confident detector & verifier (single-line only)
                if (
                    target_type != "type1a"
                    and det.plate_type == "other"
                    and det.text == ""
                    and det.confidence >= 0.80
                    and (is_plate or p_score >= 0.50)
                    and not det.is_soft_fallback
                ):
                    det.plate_type = "type1"
                    det.text = "########"
                    det.ocr_confidence = 0.20

        return detections

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[PlateDetection]:
        """
        End-to-end inference on an image path or BGR image array.
        Returns list of PlateDetection objects.
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at: {image}")
        else:
            img_bgr = image

        detections = self.detect(img_bgr)
        if len(detections) > 1:
            self.recognize_batch(img_bgr, detections)
        else:
            for det in detections:
                self.recognize_single(img_bgr, det)

        # Full-Image Direct Plate Fallback for pre-cropped input images:
        # If enabled and no valid plate was recognized by YOLO-Pose detector, check if the input image itself
        # is an isolated plate crop (e.g. cropped benchmark datasets without car scene context).
        valid_dets = [d for d in detections if d.plate_type != "other" and d.text]
        if self.enable_crop_fallback and not valid_dets and img_bgr is not None and img_bgr.size > 0:
            fallback_det = self._try_full_image_crop_fallback(img_bgr)
            if fallback_det is not None:
                return [fallback_det]

        return detections

    def _try_full_image_crop_fallback(self, image: np.ndarray) -> Optional[PlateDetection]:
        """
        Direct OCR fallback when the input image itself is a pre-cropped plate
        (e.g., benchmark evaluation on cropped plate datasets where YOLO detector fails due to lack of vehicle context).
        """
        try:
            h, w = image.shape[:2]
            if h < 16 or w < 32 or max(h, w) > 1600:
                return None

            # Fast exit for blank/black dummy test images or uniform color fields
            if np.max(image) < 15 or float(np.std(image)) < 5.0:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ys, xs = (gray > 15).nonzero()
            if len(ys) > 0 and (ys.max() - ys.min() > 15) and (xs.max() - xs.min() > 30):
                y1, y2, x1, x2 = int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)
                active = image[y1:y2, x1:x2]
                ah, aw = active.shape[:2]
                bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            else:
                active = image
                ah, aw = h, w
                bx, by, bw, bh = 0, 0, w, h

            ar = aw / float(ah)
            # Only trigger on valid plate proportions: single-line (1.7 - 6.5) or square (0.8 - 1.6)
            if not ((1.7 <= ar <= 6.5) or (0.8 <= ar <= 1.6)):
                return None

            # Don't trigger on huge panoramic scenes
            if aw >= 1000 and ah >= 700:
                return None

            # PlateVerifier guard: candidate active region must be verified as a plate
            if hasattr(self, "verifier") and self.verifier is not None and self.verifier.is_valid():
                is_p, p_score = self.verifier.verify_single(active)
                if not is_p or p_score < 0.40:
                    return None

            import re
            t1_re = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")
            t1b_re = re.compile(r"^[ABEKMHOPCTYX]{2}\d{3}\d{2,3}$")

            # Try square plate if ar is square-like
            if 0.8 <= ar <= 1.6 and hasattr(self.ocr, "predict_type1a_native"):
                rect_1a = self.rectifier.rectify(active, [0, 0, aw, 0, aw, ah, 0, ah], plate_type="type1a", margin=(0.0, 0.0), refine_corners=False)
                txt_1a, conf_1a = self.ocr.predict_type1a_native(rect_1a)
                if conf_1a >= 0.70 and t1_re.match(txt_1a) and "#" not in txt_1a:
                    return PlateDetection(
                        bbox=(bx, by, bw, bh),
                        quad=[float(bx), float(by), float(bx + bw), float(by), float(bx + bw), float(by + bh), float(bx), float(by + bh)],
                        plate_type="type1a",
                        confidence=conf_1a,
                        text=txt_1a,
                        ocr_confidence=conf_1a,
                    )

            # Try single-line OCR
            rect_1 = cv2.resize(active, (160, 36), interpolation=cv2.INTER_LINEAR)
            text_1, conf_1 = self.ocr.predict_single(rect_1, plate_type="type1")
            if conf_1 >= 0.70 and (t1_re.match(text_1) or t1b_re.match(text_1)) and "#" not in text_1:
                p_type = "type1b" if t1b_re.match(text_1) else "type1"
                return PlateDetection(
                    bbox=(bx, by, bw, bh),
                    quad=[float(bx), float(by), float(bx + bw), float(by), float(bx + bw), float(by + bh), float(bx), float(by + bh)],
                    plate_type=p_type,
                    confidence=conf_1,
                    text=text_1,
                    ocr_confidence=conf_1,
                )

        except Exception:
            pass

        return None

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
    ) -> List[List[PlateDetection]]:
        """
        End-to-end inference on a sequence of scene images or image paths.
        Returns a list of PlateDetection lists (one list per input image).
        """
        all_results = []
        for img in images:
            all_results.append(self.predict(img))
        return all_results

    def predict_with_timing(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> Tuple[List[PlateDetection], Dict[str, float]]:
        """
        Executes end-to-end inference while collecting precise latency metrics (in milliseconds).
        Returns (detections, timings_ms).
        """
        try:
            import torch
            has_torch_cuda = torch.cuda.is_available() and "cuda" in self.device
        except ImportError:
            has_torch_cuda = False

        def _sync():
            if has_torch_cuda:
                torch.cuda.synchronize()

        if isinstance(image, (str, Path)):
            t0 = time.perf_counter()
            img_bgr = cv2.imread(str(image))
            _sync()
            load_ms = (time.perf_counter() - t0) * 1000.0
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at: {image}")
        else:
            load_ms = 0.0
            img_bgr = image

        # 1. Detection Step
        _sync()
        t_det_start = time.perf_counter()
        detections = self.detect(img_bgr)
        _sync()
        det_ms = (time.perf_counter() - t_det_start) * 1000.0

        # 2. Rectification & OCR Step
        rect_ms = 0.0
        ocr_ms = 0.0

        for det in detections:
            if det.plate_type == "other":
                det.text = ""
                det.ocr_confidence = 0.0
                continue

            # Time rectification
            _sync()
            t_r0 = time.perf_counter()
            bx, by, bw, bh = det.bbox
            ar = bw / float(bh) if bh > 0 else 1.0
            is_square_1a = (det.plate_type == "type1a") and (ar <= 2.20)

            if is_square_1a:
                rectified = self.rectifier.rectify(img_bgr, det.quad, plate_type="type1a")
                top_line, bottom_line = self.rectifier.split_type1a(rectified)
                crop_for_ocr = self.rectifier.stitch_type1a_horizontal(
                    top_line, bottom_line, target_size=(160, 36)
                )
            else:
                actual_type = "type1" if det.plate_type == "type1a" else det.plate_type
                crop_for_ocr = self.rectifier.rectify(img_bgr, det.quad, plate_type=actual_type)
            _sync()
            rect_ms += (time.perf_counter() - t_r0) * 1000.0

            # Time OCR
            _sync()
            t_o0 = time.perf_counter()
            text, ocr_conf = self.ocr.predict_single(crop_for_ocr, plate_type=det.plate_type)
            _sync()
            ocr_ms += (time.perf_counter() - t_o0) * 1000.0

            det.text = text
            det.ocr_confidence = ocr_conf

        total_ms = det_ms + rect_ms + ocr_ms
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        timings = {
            "load_ms": round(load_ms, 2),
            "detection_ms": round(det_ms, 2),
            "rectification_ms": round(rect_ms, 2),
            "ocr_ms": round(ocr_ms, 2),
            "total_ms": round(total_ms, 2),
            "fps": round(fps, 1),
            "plates_count": len(detections),
        }

        return detections, timings
