# 🎯 05. Evaluation Metrics & Matching Logic Specification

```yaml
# QUICK_FORMULAS
METRICS:
  - Sequence Accuracy (Exact Match): "pred == target (100% characters)"
  - Normalized Levenshtein / CER: "edit_distance(pred, target) / len(target)"
  - Detection IoU Threshold: 0.50
  - Type Classification: Macro F1-score across [type1, type1a, type1b, other]
FATAL_PENALTY: "Predicting type1/type1a/type1b on an 'other' plate"
```

---

## 1. Автоматическое сопоставление предсказаний с эталоном

Жюри сопоставляет предсказанный CSV с эталонным скрытым CSV:

1. **Матчинг объектов на кадре**:
   - Если на изображении $N$ знаков, предсказанные строки сопоставляются с эталонными.
   - Критерий совпадения локализации: IoU предсказанного знака и эталонного $> 0.5$ (или венгерский алгоритм сопоставления по наибольшему сходству строк).

---

## 2. Формулы метрик

### 1. Sequence Accuracy (Точное совпадение номера)
$$Acc_{seq} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{pred}_i == \text{target}_i)$$
- Позиции с `#` в таргете: считаются совпавшими, если предсказан либо точный символ, либо `#`.
- **Запрет угадывания**: попытка наугад вписать нечитаемый символ вместо `#` в эталон или предсказание при ошибке обнуляет точность всей последовательности ($Acc_{seq}=0$) и ухудшает CER. См. детальный регламент в `docs/specs/02_plate_mask_regex.md` (п. 4).

### 2. Character Error Rate (CER) с учётом нечитаемых символов
$$CER = \frac{\sum_{i=1}^N \text{Levenshtein}(\text{pred}_i, \text{target}_i)}{\sum_{i=1}^N \text{len}(\text{target}_i)}$$

### 3. F1-Score по типам знаков (`plate_type`)
$$F1_{macro} = \frac{1}{4} \sum_{c \in \{\text{type1, type1a, type1b, other}\}} F1_c$$

---

## 3. Штрафные санкции за класс `other`

| Ситуация | Оценка жюри |
|---|---|
| Настоящий класс `other` предсказан как `other` | ✅ Правильно (True Negative) |
| Настоящий класс `other` не выдан вовсе | ✅ Допустимо (Filtered Out) |
| Настоящий класс `other` прочитан как `type1` / `type1a` / `type1b` | ❌ **Грубая ошибка (False Positive)** со штрафом! |
