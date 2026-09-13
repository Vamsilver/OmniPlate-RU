# 📊 04. Dataset Quotas, Meta.csv Schema & Quality Rules

```yaml
# QUICK_QUOTAS
REAL_TYPE1A_MIN_IMAGES: 150
REAL_TYPE1A_MIN_UNIQUE_PLATES: 50
REAL_TYPE1B_MIN_IMAGES: 300
REAL_TYPE1B_MIN_UNIQUE_PLATES: 100
REAL_OTHER_MIN_IMAGES: 50
SYNTHETIC_MIN_TOTAL: 5000
META_COLUMNS_COUNT: 10
DELIMITER: ";"
ENCODING: "UTF-8"
LICENSE_DATASET: "CC BY 4.0"
FACE_BLUR_REQUIRED: true
```

---

## 1. Спецификация колонок `meta.csv`

Файл содержит заголовок и ровно 10 столбцов (разделитель `;`):

$$\text{image;plate\_num;plate\_type;bbox;quad;is\_vehicle;is\_synthetic;source;license;conditions}$$

| Индекс | Имя колонки | Тип данных | Пример | Валидационное правило |
|---|---|---|---|---|
| 0 | `image` | `str` | `images/real/img_001.jpg` | Файл обязан существовать по пути относительно корня датасета |
| 1 | `plate_num` | `str` | `A123BC716` | Латиница верхний регистр (`ABEKMHOPCTYX`), цифры, символ `#` |
| 2 | `plate_type` | `enum` | `type1a` | Строго из `['type1', 'type1a', 'type1b', 'other']` |
| 3 | `bbox` | `str` | `120,340,180,95` | Формат `x,y,w,h` целые числа пикселей |
| 4 | `quad` | `str` | `120,345,295,340,300,435,125,440` | `x1,y1,x2,y2,x3,y3,x4,y4` (по часовой стрелке от верхнего левого) |
| 5 | `is_vehicle` | `int` | `1` | `1` — на ТС; `0` — на щите, витрине, экране, одежде |
| 6 | `is_synthetic` | `int` | `0` | `0` — реальное фото; `1` — генератор синтетики |
| 7 | `source` | `str` | `own_photo` | URL или название открытого источника |
| 8 | `license` | `str` | `CC-BY-4.0` | Свободная лицензия источника |
| 9 | `conditions` | `str` | `day,angle` | Теги через запятую из: `day,night,rain,snow,dirt,glare,motion_blur,angle` |

---

## 2. Формат файлов разметки `dataset/labels/<name>.txt`
По одному `.txt` файлу на каждое изображение. Каждая строка содержит:
```text
<class_id> <x_center> <y_center> <width> <height> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <plate_num>
```
где координаты нормализованы к $[0, 1]$ относительно ширины и высоты кадра.  
`class_id`: 0 = `type1`, 1 = `type1a`, 2 = `type1b`, 3 = `other`.

---

## 3. Политика анонимизации лиц (Face Privacy Rule)
- Все лица людей на изображениях должны быть размыты (`cv2.GaussianBlur` с ядром $\ge 25 \times 25$) или закрашены.
- Если человек является смысловым центром кадра (площадь bounding box лица $> 15\%$ кадра) — изображение **запрещено** включать в датасет.
- Запрещено включать любые изображения из 30 картинок отладочного набора организаторов!
