# 🧭 OmniPlate-RU: Навигатор по документации и архитектурный индекс

> **Справочник проекта**: Сводный индекс базы знаний, технических инвариантов и архитектурных решений OmniPlate-RU. Используйте приведенную ниже карту для быстрого доступа к спецификациям, стандартам ГОСТ и отчетам.

---

## ⚡ Ключевые проектные инварианты (Global Invariants)
```yaml
TASK: "Recognition of Non-Standard Russian License Plates (Volga IT 2026 / AIS Gorod)"
TARGET_CLASSES: [type1, type1a, type1b, other]
MAX_LATENCY_PER_IMAGE: 100 ms
REFERENCE_HARDWARE: "Intel Core i5-7600, NVIDIA GTX 1050 Ti (4GB VRAM), 16GB RAM"
NETWORK_ACCESS: "STRICTLY FORBIDDEN AT RUNTIME (100% Offline, all weights local in models/)"
DATASET_LICENSE: "CC BY 4.0"
CODE_LICENSE: "MIT"
ALLOWED_LETTERS: "ABEKMHOPCTYX" # 12 Latin letters matching Cyrillic
REGEX_PLATE: "^[ABEKMHOPCTYX]\\d{3}[ABEKMHOPCTYX]{2}\\d{2,3}$"
CSV_DELIMITER: ";"
CSV_HEADER: "image;plate_num;plate_type;confidence"
PROJECT_STATUS: "ALL STAGES 1-5 COMPLETED (100% Verified, 72/72 Pytest PASS, Dataset 6518 PASS)"
OCR_ARCHITECTURE: "MoE (Mixture of Experts): LPRNet-v3 for Type 1, LPRNet-v2 for Type 1B, 2D Dual-Line for Type 1A"
```

---

## 🗺️ Карта документации и спецификаций

| Документ | Раздел и ключевые параметры | Назначение документа |
|---|---|---|
| [MASTER_PLAN.md](../MASTER_PLAN.md) | **Генеральный план разработки**: 5 этапов, чек-листы, результаты тестирования. | Полный журнал выполнения задач и контрольные метрики. |
| [CONSTITUTION.md](../CONSTITUTION.md) | **Технические стандарты и гарантии**: 100% оффлайн, CC BY 4.0, деидентификация лиц. | Гарантии регламента соревнований и протоколы контроля качества. |
| [specs/01_gost_geometry.md](specs/01_gost_geometry.md) | **Геометрия ГОСТ**: <br>• `type1`: 520×112 мм (4.64:1)<br>• `type1a`: 290×170 мм (1.70:1, 2 строки)<br>• `type1b`: 520×112 мм, Желтый `#FFCC00`<br>• Шрифт: ГОСТ Р 50577-2018 | Параметры синтетического рендеринга и канонические размеры гомографии. |
| [specs/02_plate_mask_regex.md](specs/02_plate_mask_regex.md) | **Маска пластин и синтаксис**: <br>• Маска: `[L][D][D][D][L][L][REGION]`<br>• Регион: 2 или 3 цифры (трехзначный начинается с 1, 2, 7)<br>• Нераспознанный символ: `#`<br>• Регистр: верхний латинский | Спецификация FSM Beam Search декодера и regex-валидатора. |
| [specs/03_hardware_latency_budget.md](specs/03_hardware_latency_budget.md) | **Бюджет задержки и SLA**: <br>• Лимит: $\le 100$ мс на GTX 1050 Ti<br>• Детекция: $\le 40$ мс<br>• Выравнивание: $\le 5$ мс<br>• OCR: $\le 20$ мс<br>• Память: $<3.5$ ГБ VRAM | Спецификация аппаратных замеров скорости и экспорта ONNX FP16. |
| [specs/04_dataset_quotas_schema.md](specs/04_dataset_quotas_schema.md) | **Схема данных и квоты**: <br>• Реальные: 1а $\ge 150$ (50 уник.), 1б $\ge 300$ (100 уник.), other $\ge 50$<br>• Синтетика: $\ge 5000$<br>• `meta.csv`: 10 столбцов с разделителем `;` | Спецификация структуры датасета и валидатора `validate_dataset.py`. |
| [specs/05_evaluation_metrics.md](specs/05_evaluation_metrics.md) | **Метрики оценки**: <br>• Sequence Accuracy (точное совпадение)<br>• Character Error Rate (CER)<br>• Недопустимость ошибки на классе `other` (Fatal Penalty) | Методика расчета точности и функции штрафов. |
| [specs/06_extensibility_design.md](specs/06_extensibility_design.md) | **Расширяемость архитектуры**: <br>• Модульность для знаков СНГ (BY, KZ, AM)<br>• Поддержка мотоциклов, военных и дипломатических знаков | Архитектурное обоснование масштабирования пайплайна. |
| [report/EXPLANATORY_NOTE.md](report/EXPLANATORY_NOTE.md) | **Пояснительная записка**: 5-страничный академический отчет для жюри. | Сводный отчет с описанием архитектуры, экспериментов и замеров скорости. |
| [workflow/SUBMISSION_GUIDE.md](workflow/SUBMISSION_GUIDE.md) | **Руководство по сдаче**: инструкции по упаковке, контрольные суммы SHA-256. | Финальный регламент упаковки и шаблон сопроводительного письма. |
