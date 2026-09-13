# 🌐 06. Architecture Extensibility Specification (Section 6 Requirement)

```yaml
# QUICK_SUMMARY
PURPOSE: "Architectural justification for expanding to other plate shapes, vehicle types, and countries (CIS & international)"
MODULARITY_AXES:
  - Detection: "Class-agnostic Quad Keypoints (works for any 4-corner polygon)"
  - Geometry: "Canonical size registry by plate aspect ratio"
  - OCR: "Language-agnostic CTC-loss with dynamic charset & regex grammar masks"
```

---

## 1. Концепция модульного расширения

Решение OmniPlate-RU построено по принципу **слабосвязанных модулей (Decoupled Micro-Pipeline)**, что позволяет масштабировать его на новые форматы без переписывания кодовой базы:

```mermaid
flowchart LR
    A[YOLO-Pose: 4 Quad Corners] --> B{Aspect Ratio & Geometry Registry}
    B -->|Ratio ~4.6:1| C[Single-Line OCR Branch]
    B -->|Ratio ~1.7:1| D[Split & Stitch 2-Line Branch]
    B -->|Ratio ~1.3:1 (Мото/Трактор)| E[Square 2-Line Branch]
    C --> F[Country Grammars: RU / BY / KZ / AM]
    D --> F
    E --> F
    F --> G[Validated Output]
```

---

## 2. Поддержка других стран (СНГ и мир)

1. **Беларусь (BY)**:
   - Формат: `1234 AB-7` (белый) или `TAX 1234` (желтый).
   - Расширение: добавление дефиса в алфавит и подключение грамматики `^\d{4}\s?[A-Z]{2}-[1-7]$`.
2. **Казахстан (KZ)**:
   - Формат: `123 ABC 01` (флаг KZ слева).
   - Расширение: соотношение сторон идентично (520×112 мм), требуется только переключение маски регулярного выражения.
3. **Армения (AM)**:
   - Формат: `12 AB 345` или `123 AB 45`.
   - Расширение: идентичный двухэтапный пайплайн с настройкой алфавита.

---

## 3. Поддержка специальных видов знаков РФ

- **Мотоциклы и спецтехника (Типы 3, 4)**:
  - Имеют двухстрочный квадратный формат $190 \times 145$ мм (соотношение $\approx 1.31:1$).
  - Подключаются через канонический размер гомографии $140 \times 105$ px и вертикальный сплит строк.
- **Дипломатические (красные) и Военные (черные)**:
  - Цветовая дифференциация в HSV-пространстве + добавление соответствующих классов в классификатор детектора.
