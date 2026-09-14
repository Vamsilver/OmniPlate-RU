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
        device: str = "cuda",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        use_onnx: bool = True,
    ) -> None:
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_onnx = use_onnx

        # Resolve model paths
        self.detector_path = self._resolve_model_path(
            detector_path,
            candidates=[
                "models/detector_yolo_pose.onnx",
                "models/detector_yolo_pose_best.pt",
                "models/yolov8n_pose_best.pt",
            ],
        )
        self.ocr_path = self._resolve_model_path(
            ocr_path,
            candidates=[
                "models/ocr_lprnet_best.onnx",
                "models/ocr_lprnet_best.pt",
            ],
        )

        # Initialize submodules
        self.rectifier = PlateRectifier()
        self.ocr = PlateOCR(
            model_path=self.ocr_path,
            device=self.device,
            use_onnx=self.use_onnx,
        )
        self.detector = self._init_detector(self.detector_path)

    @staticmethod
    def validate_plate_geometry(
        bbox: Tuple[int, int, int, int],
        plate_type: str,
        img_w: int,
        img_h: int,
    ) -> bool:
        """
        Rejects impossible detections based on physical GOST aspect ratios and camera scene scale.
        """
        bx, by, bw, bh = bbox

        # 1. Extreme size rejection (plate cannot be smaller than 20x10 or larger than 85% width / 70% height)
        if bw < 20 or bh < 10:
            return False
        if bw > img_w * 0.85 or bh > img_h * 0.70:
            return False

        # 2. Border edge rejection: highway guardrails / road cuts right at frame top edge
        if by <= 2 and bw > 250:
            return False

        # 3. Aspect Ratio (Width / Height) rejection based on physical GOST R 50577-2018
        # Calibrated to accommodate perspective yaw/pitch angles in real scenes
        ar = bw / float(bh)
        if plate_type in ("type1", "type1b"):
            # Single-line plates: physical AR ~ 4.64. Perspective foreshortening: [1.80, 6.50]
            if ar < 1.80 or ar > 6.50:
                return False
        elif plate_type == "type1a":
            # Two-line square plate: physical AR ~ 1.706. Extreme perspective angles: [0.75, 3.80]
            if ar < 0.75 or ar > 3.80:
                return False
        elif plate_type == "other":
            # Other (trailers, motorcycles, square / rectangular): [0.70, 6.50]
            if ar < 0.70 or ar > 6.50:
                return False

        return True

    @staticmethod
    def validate_quad_geometry(
        quad: Sequence[float],
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """
        Verifies that the 4 quad corners form a valid, convex, non-collapsed quadrilateral
        that reasonably fills the bounding box.
        """
        if len(quad) != 8:
            return False

        bx, by, bw, bh = bbox
        pts = np.array(quad, dtype=np.float32).reshape(4, 2)

        # 1. Convexity check (plate is a planar convex polygon in 3D perspective)
        pts_int = pts.astype(np.int32)
        if not cv2.isContourConvex(pts_int):
            return False

        # 2. Polygon area vs BBox area fill ratio
        poly_area = cv2.contourArea(pts)
        bbox_area = float(bw * bh)
        if bbox_area <= 0:
            return False

        fill_ratio = poly_area / bbox_area
        # A valid perspective projection of a rectangle fills at least 50% of its AABB
        if fill_ratio < 0.50 or fill_ratio > 1.05:
            return False

        return True

    def detect(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Runs plate detector on scene image.
        Returns list of PlateDetection with bounding boxes and quads.
        """
        if self.detector is None:
            return []

        ih, iw = image.shape[:2]
        results = self.detector.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[PlateDetection] = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        kpts_list = None
        if hasattr(r, "keypoints") and r.keypoints is not None:
            try:
                kpts_list = r.keypoints.xy.cpu().numpy()
            except Exception:
                kpts_list = None

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

            # Apply physical geometry filter (kills false positives on car panels, barriers, sky)
            if not self.validate_plate_geometry(bbox, plate_type, iw, ih):
                continue

            kpts = kpts_list[i] if kpts_list is not None and i < len(kpts_list) else None
            quad = self._extract_quad(kpts, bbox, iw, ih)

            # Apply quad polygon convexity and fill validation (kills collapsed/skewed background noise)
            if not self.validate_quad_geometry(quad, bbox):
                continue

            detections.append(
                PlateDetection(
                    bbox=bbox,
                    quad=quad,
                    plate_type=plate_type,
                    confidence=conf,
                )
            )

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
    ) -> Optional[str]:
        if explicit_path in ("", "none", "None", False):
            return None
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        for cand in candidates:
            # Check relative to PROJECT_ROOT and CWD
            p1 = PROJECT_ROOT / cand
            if p1.exists():
                return str(p1)
            if os.path.exists(cand):
                return str(os.path.abspath(cand))
        return None


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
        Falls back to bbox corners if keypoints are missing or degenerate.
        """
        if kpts_xy is not None and len(kpts_xy) >= 4:
            pts = kpts_xy[:4].astype(np.float32)
            # Check if all points are non-zero
            if not np.all(pts == 0):
                # Clamp coordinates to image boundaries
                pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                ordered = PlateRectifier.order_quad_points(pts)
                return ordered.flatten().tolist()

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
            bx, by, bw, bh = detection.bbox
            ar = bw / float(bh) if bh > 0 else 1.0

            # Guard: Physical Type 1A is a 2-line square plate (aspect ratio ~1.706).
            # In perspective perspective angles, AR decreases. If AR > 2.20, it is physically
            # an elongated single-line plate (Type 1), so we must NOT split it horizontally.
            is_square_1a = (plate_type == "type1a") and (ar <= 2.20)

            if is_square_1a:
                # Type 1A Two-Line Square Plate
                rectified = self.rectifier.rectify(
                    image,
                    detection.quad,
                    plate_type="type1a",
                )
                detection.rectified_crop = rectified

                # Split top and bottom lines
                top_line, bottom_line = self.rectifier.split_type1a(rectified)

                # Canonical Stitched horizontal strip (format LPRNet was trained on)
                stitched = self.rectifier.stitch_type1a_horizontal(
                    top_line,
                    bottom_line,
                    target_size=(160, 36),
                )
                text_stitched, conf_stitched = self.ocr.predict_single(stitched, plate_type="type1a")
                detection.text = text_stitched
                detection.ocr_confidence = round(conf_stitched, 4)

            else:
                # Type 1 and Type 1B Single-Line Plates (or single-line fallback if misclassified as 1a)
                actual_type = "type1" if plate_type == "type1a" else plate_type
                rectified = self.rectifier.rectify(
                    image,
                    detection.quad,
                    plate_type=actual_type,
                )
                detection.rectified_crop = rectified

                text, ocr_conf = self.ocr.predict_single(rectified, plate_type=actual_type)
                detection.text = text
                detection.ocr_confidence = ocr_conf

        except Exception as e:
            # Fallback on rectification or OCR failure
            detection.text = "#" * 8
            detection.ocr_confidence = 0.0

        return detection


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
        for det in detections:
            self.recognize_single(img_bgr, det)

        return detections

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
