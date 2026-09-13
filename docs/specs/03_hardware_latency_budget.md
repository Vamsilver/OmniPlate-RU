# ⏱️ 03. Hardware SLA & Latency Budget Specification

```yaml
# QUICK_CONSTANTS
MAX_LATENCY_TOTAL: 100 ms          # Hard deadline per image
TARGET_FPS: 10.0                   # Minimum throughput
GPU_NAME: "NVIDIA GeForce GTX 1050 Ti"
GPU_VRAM_MB: 4096                  # 4 GB Pascal architecture
GPU_COMPUTE_CAPABILITY: 6.1
CPU_NAME: "Intel Core i5-7600"     # 4 cores, 4 threads @ 3.50 GHz
SYSTEM_RAM_MB: 16384
MAX_MODEL_VRAM_FOOTPRINT: 2048 MB  # Keep under 2GB to prevent OOM
```

---

## 1. Бюджет задержки по этапам (Latency Budget Breakdown)

Для кадра разрешением $1920 \times 1080$ (или $1280 \times 720$):

| Модуль | Архитектура / Метод | Макс. время (GPU) | Примечание |
|---|---|---|---|
| **Препроцессинг** | `cv2.resize` + Letterbox $640 \times 640$ | $\le 5$ мс | CPU/GPU |
| **Детекция + 4 Quad** | YOLOv8n-pose (ONNX FP16) | $\le 35$ мс | BBox + 4 угла + класс |
| **Perspective Warp** | OpenCV `getPerspectiveTransform` + `warpPerspective` | $\le 4$ мс | На каждый найденный знак |
| **OCR Распознавание** | LPRNet / CRNN MobileNetV3 (ONNX) | $\le 25$ мс | Для 1–2 знаков на кадр |
| **Маска & Post-process** | Regex + CTC greedy decode | $\le 2$ мс | CPU |
| **Запись CSV** | Прямой append в буфер | $\le 1$ мс | Дисковый I/O |
| **ИТОГО** | | $\mathbf{\approx 72\text{ мс}}$ | **Запас 28 мс до лимита 100 мс!** |

---

## 2. Требования к среде выполнения инференса

1. **Библиотеки инференса**:
   - `onnxruntime-gpu` (с `CUDAExecutionProvider`) или прямой `TensorRT`.
   - Исключить тяжелые зависимости вроде `torchvision.transforms` в цикле инференса — использовать быстрый `cv2` и `numpy`.
2. **Память VRAM**:
   - Память GTX 1050 Ti составляет **4 ГБ**. Модели в процессе инференса не должны выделять более $2.0$ ГБ суммарно.
   - Освобождение временных тензоров после каждого кадра (без накапливания истории в списках).
