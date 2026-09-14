"""
OmniPlate-RU Pipeline Modules
"""

from src.pipeline.ocr import CTCDecoder, PlateOCR
from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection
from src.pipeline.rectifier import PlateRectifier

__all__ = [
    "CTCDecoder",
    "OmniPlatePipeline",
    "PlateDetection",
    "PlateOCR",
    "PlateRectifier",
]

