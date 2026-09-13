# 🔤 02. Plate Mask, Character Set & RegEx Specification

```yaml
# QUICK_CONSTANTS
ALLOWED_LETTERS: "ABEKMHOPCTYX"       # 12 Latin equivalents of Cyrillic
ALLOWED_DIGITS: "0123456789"
UNREADABLE_CHAR: "#"
MIN_PLATE_LEN: 8                      # 1 letter + 3 digits + 2 letters + 2-digit region
MAX_PLATE_LEN: 9                      # 1 letter + 3 digits + 2 letters + 3-digit region
REGEX_STRICT: "^[ABEKMHOPCTYX]\\d{3}[ABEKMHOPCTYX]{2}\\d{2,3}$"
REGEX_WITH_WILDCARDS: "^[ABEKMHOPCTYX#][\\d#]{3}[ABEKMHOPCTYX#]{2}[\\d#]{2,3}$"
THREE_DIGIT_REGION_PREFIXES: ["1", "2", "7"]
```

---

## 1. Позиционная структура номера

Номерной знак состоит строго из 8 или 9 позиций:

$$\begin{array}{|c|c|c|c|c|c|c|}
\hline
\text{Pos 0} & \text{Pos 1} & \text{Pos 2} & \text{Pos 3} & \text{Pos 4} & \text{Pos 5} & \text{Pos 6..7 (или 6..8)} \\
\hline
\text{Буква} & \text{Цифра} & \text{Цифра} & \text{Цифра} & \text{Буква} & \text{Буква} & \text{Код региона (2 или 3 цифры)} \\
\hline
\end{array}$$

- **Регионы из 3 цифр**: первая цифра всегда $1, 2$ или $7$ (например: `777`, `116`, `799`, `196`).
- **Символ `#`**: подставляется **ровно в ту позицию**, где символ нечитаем человеком (например, `A123##77`, `#246#C73`). Нельзя смещать остальные символы.

---

## 2. Матрица путаницы OCR (Visual Confusion Fixes)

При постобработке вероятностных выходов CTC-декодера применяется эвристика исправления похожих символов по позиции:

| Ожидаемый тип позиции | Ошибочно предсказанный символ | Автозамена на |
|---|---|---|
| **Буква** (Pos 0, 4, 5) | `0` (ноль) | `O` (буква) |
| **Буква** (Pos 0, 4, 5) | `8` (восемь) | `B` (буква) |
| **Буква** (Pos 0, 4, 5) | `1` (один) | `T` или `#` |
| **Цифра** (Pos 1..3, 6..8) | `O` (буква) | `0` (ноль) |
| **Цифра** (Pos 1..3, 6..8) | `B` (буква) | `8` (восемь) |

---

## 3. Python-валидатор номера (Snippets)

```python
import re

ALLOWED_LETTERS = set("ABEKMHOPCTYX")
VALID_3DIGIT_STARTS = {"1", "2", "7"}
PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")

def is_valid_plate_format(plate: str) -> bool:
    if not PLATE_REGEX.match(plate):
        return False
    if len(plate) == 9 and plate[6] not in VALID_3DIGIT_STARTS and plate[6] != '#':
        return False
    return True
```
