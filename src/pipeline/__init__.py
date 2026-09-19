"""
OmniPlate-RU Pipeline Modules
"""

import os
import sys
from pathlib import Path

# Ensure PyTorch CUDA DLLs are found by ONNX Runtime on Windows
if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
        if str(torch_lib) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")

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

