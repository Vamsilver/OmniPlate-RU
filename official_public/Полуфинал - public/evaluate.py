#!/usr/bin/env python3
"""Проверка решения: сопоставление выходного CSV с эталоном и подсчёт метрик.
Волга-IT 2026, полуфинал, номинация «ИИ и анализ данных» (раздел 8, п. 3).

Эталон (раздел 4.2):   image;plate_num;plate_type;is_vehicle
Выход решения (раздел 5): image;plate_num;plate_type;confidence

Только стандартная библиотека Python 3.9+.

Запуск:
    python evaluate.py --gt reference.csv --pred result.csv
    python evaluate.py --gt reference.csv --pred result.csv --images test_images/ --json score.json --details details.csv

--images нужен, чтобы знать полный список изображений скрытого набора (в том числе
кадры без знаков): без него список берётся из эталона, и ложные срабатывания на
«пустых» кадрах видны только если такие кадры перечислены в эталоне пустой строкой
(image;;;) или переданы через --images.

Метрики (предложение для раздела 9 задания, веса настраиваются флагами):

  Для каждого целевого знака эталона (plate_type ∈ {type1, type1a, type1b}, is_vehicle=1)
  ищется строка решения на том же изображении с наибольшим посимвольным сходством
  (≥ 0.5, иначе знак считается пропущенным). Знак засчитан (full match), если все
  читаемые человеком позиции (не «#») совпали И тип знака определён верно.

    A_t   = засчитанных знаков типа t / всего знаков типа t          (t = type1, type1a, type1b)
    P     = засчитанных знаков / всех строк решения с plate_type ≠ other  (precision)
    Score = 0.15·A_type1 + 0.35·A_type1a + 0.35·A_type1b + 0.15·P       ∈ [0, 1]

  В precision «штрафуются»: лишние строки, строки для знаков вне области задания
  (эталонный plate_type=other) с типом ≠ other, строки для знаков не на ТС
  (is_vehicle=0) с типом ≠ other, дубли. Строки решения с plate_type=other
  никогда не штрафуются и не засчитываются — это корректный способ «отсеять».

  Дополнительно считаются (информационно, в Score не входят): доля обнаруженных
  знаков, точность типа среди обнаруженных, посимвольная точность, калибровка
  confidence. Время обработки измеряется отдельно (раздел 3, п. 2).

Код возврата: 0 — посчитано; 2 — не удалось прочитать входные файлы.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

VERSION = "1.0"

PLATE_TYPES = ("type1", "type1a", "type1b", "other")
TARGET_TYPES = ("type1", "type1a", "type1b")
GT_COLUMNS = ("image", "plate_num", "plate_type", "is_vehicle")
PRED_COLUMNS = ("image", "plate_num", "plate_type", "confidence")

LATIN_LETTERS = "ABEKMHOPCTYX"
CYR_LETTERS = "АВЕКМНОРСТУХ"
CYR_TO_LAT = str.maketrans(CYR_LETTERS + CYR_LETTERS.lower(), LATIN_LETTERS + LATIN_LETTERS)

_L = f"[{LATIN_LETTERS}#]"
_D = "[0-9#]"
PLATE_MASKS = {
    "type1": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_D}{{2,3}}$"),
    "type1a": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_D}{{2,3}}$"),
    "type1b": re.compile(f"^{_L}{{2}}{_D}{{3}}{_D}{{2,3}}$"),
    "other": re.compile(r"^[A-Z0-9#]{0,12}$"),
}

MATCH_THRESHOLD = 0.5
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

@dataclass
class GtPlate:
    line: int
    image: str
    plate: str
    ptype: str
    is_vehicle: bool

    @property
    def is_target(self) -> bool:
        return self.ptype in TARGET_TYPES and self.is_vehicle


@dataclass
class Pred:
    line: int
    image: str
    plate: str
    ptype: str
    confidence: float
    format_issues: list[str] = field(default_factory=list)


@dataclass
class Match:
    gt: GtPlate
    pred: Pred | None
    similarity: float
    plate_ok: bool
    type_ok: bool
    char_hits: int
    char_total: int

    @property
    def full_ok(self) -> bool:
        return self.pred is not None and self.plate_ok and self.type_ok


# ---------------------------------------------------------------------------
# Сравнение номеров
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compare_plates(gt: str, pred: str) -> tuple[float, bool, int, int]:
    """(сходство 0..1, полное совпадение, совпавших читаемых позиций, читаемых позиций).

    Позиции эталона с «#» не оцениваются. «#» в ответе решения — это ошибка в
    соответствующей позиции. Полное совпадение требует равной длины.
    """
    readable = [i for i, ch in enumerate(gt) if ch != "#"]
    total = len(readable)
    if len(gt) == len(pred):
        hits = sum(1 for i in readable if pred[i] == gt[i])
        sim = hits / total if total else (1.0 if pred == gt else 0.0)
        return sim, hits == total and total > 0, hits, total
    # Разная длина: сходство через расстояние Левенштейна по читаемым символам,
    # позиционные попадания — по выравниванию с начала.
    gt_read = "".join(gt[i] for i in readable)
    dist = levenshtein(gt_read, pred)
    sim = max(0.0, 1.0 - dist / max(len(gt_read), len(pred), 1))
    hits = sum(1 for i in readable if i < len(pred) and pred[i] == gt[i])
    return sim, False, hits, total


# ---------------------------------------------------------------------------
# Чтение CSV
# ---------------------------------------------------------------------------

def read_csv(path: str, expected: tuple[str, ...], problems: list[str]) -> list[dict[str, str]]:
    with open(path, "rb") as fb:
        raw = fb.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        problems.append(f"{os.path.basename(path)}: не UTF-8, читаю с заменой символов")
        text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        problems.append(f"{os.path.basename(path)}: пустой файл")
        return []
    if ";" not in lines[0]:
        problems.append(f"{os.path.basename(path)}: в заголовке нет «;» — ожидается разделитель «;»")
    reader = csv.DictReader(lines, delimiter=";")
    header = tuple((h or "").strip() for h in (reader.fieldnames or ()))
    if header != expected:
        problems.append(f"{os.path.basename(path)}: заголовок {list(header)} ≠ {list(expected)}")
    rows = []
    for i, row in enumerate(reader, start=2):
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None}
        if row.get(None):
            problems.append(f"{os.path.basename(path)}:{i}: лишние поля {row[None]}")
        clean["_line"] = str(i)
        rows.append(clean)
    return rows


def load_gt(path: str, problems: list[str]) -> tuple[list[GtPlate], set[str]]:
    plates: list[GtPlate] = []
    images: set[str] = set()
    for r in read_csv(path, GT_COLUMNS, problems):
        image = os.path.basename(r.get("image", ""))
        if not image:
            continue
        images.add(image)
        plate = r.get("plate_num", "").upper().translate(CYR_TO_LAT)
        ptype = r.get("plate_type", "")
        if not plate and not ptype:
            continue  # «пустой» кадр: image;;; — только регистрирует изображение
        if ptype not in PLATE_TYPES:
            problems.append(f"эталон:{r['_line']}: неизвестный plate_type «{ptype}»")
            continue
        plates.append(GtPlate(int(r["_line"]), image, plate, ptype, r.get("is_vehicle", "1") != "0"))
    return plates, images


def load_pred(path: str, known_images: set[str], problems: list[str]) -> list[Pred]:
    preds: list[Pred] = []
    for r in read_csv(path, PRED_COLUMNS, problems):
        line = int(r["_line"])
        issues: list[str] = []
        raw_image = r.get("image", "")
        image = os.path.basename(raw_image)
        if image != raw_image:
            issues.append("image должен быть именем файла без пути")
        if not image:
            problems.append(f"решение:{line}: пустое image — строка пропущена")
            continue
        if known_images and image not in known_images:
            problems.append(f"решение:{line}: изображение «{image}» не входит в тестовый набор — строка пропущена")
            continue
        plate = r.get("plate_num", "")
        if any(ch in CYR_LETTERS + CYR_LETTERS.lower() for ch in plate):
            issues.append("кириллица в plate_num (нужна латиница)")
            plate = plate.translate(CYR_TO_LAT)
        if plate != plate.upper():
            issues.append("plate_num не в верхнем регистре")
            plate = plate.upper()
        if " " in plate:
            issues.append("пробелы в plate_num")
            plate = plate.replace(" ", "")
        ptype = r.get("plate_type", "")
        if ptype not in PLATE_TYPES:
            issues.append(f"неизвестный plate_type «{ptype}» — считается как other")
            ptype = "other"
        mask = PLATE_MASKS[ptype]
        if not mask.fullmatch(plate):
            issues.append(f"plate_num «{plate}» не соответствует маске {ptype}")
        conf_s = r.get("confidence", "")
        try:
            conf = float(conf_s)
            if not 0.0 <= conf <= 1.0:
                issues.append(f"confidence {conf} вне [0, 1]")
                conf = min(1.0, max(0.0, conf))
        except ValueError:
            issues.append(f"confidence «{conf_s}» не число — принято 0")
            conf = 0.0
        preds.append(Pred(line, image, plate, ptype, conf, issues))
    return preds


# ---------------------------------------------------------------------------
# Сопоставление
# ---------------------------------------------------------------------------

def match_image(gts: list[GtPlate], preds: list[Pred]) -> tuple[list[Match], list[Pred]]:
    """Жадное сопоставление по убыванию сходства. Возвращает матчи для всех gts и несопоставленные preds."""
    cands: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            sim, _ok, _h, _t = compare_plates(g.plate, p.plate)
            if sim >= MATCH_THRESHOLD:
                cands.append((sim, gi, pi))
    # Тай-брейк: при равном сходстве — совпадение типа, затем большая уверенность.
    cands.sort(key=lambda c: (-c[0], preds[c[2]].ptype != gts[c[1]].ptype, -preds[c[2]].confidence))
    used_g: set[int] = set()
    used_p: set[int] = set()
    assigned: dict[int, tuple[int, float]] = {}
    for sim, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        assigned[gi] = (pi, sim)
    matches: list[Match] = []
    for gi, g in enumerate(gts):
        if gi in assigned:
            pi, sim = assigned[gi]
            p = preds[pi]
            _s, ok, hits, total = compare_plates(g.plate, p.plate)
            matches.append(Match(g, p, sim, ok, p.ptype == g.ptype, hits, total))
        else:
            total = sum(1 for ch in g.plate if ch != "#")
            matches.append(Match(g, None, 0.0, False, False, 0, total))
    unmatched = [p for pi, p in enumerate(preds) if pi not in used_p]
    return matches, unmatched


def evaluate(gt_plates: list[GtPlate], images: set[str], preds: list[Pred], weights: dict[str, float]) -> dict:
    gt_by_image: dict[str, list[GtPlate]] = defaultdict(list)
    for g in gt_plates:
        gt_by_image[g.image].append(g)
    pred_by_image: dict[str, list[Pred]] = defaultdict(list)
    for p in preds:
        pred_by_image[p.image].append(p)

    all_matches: list[Match] = []
    false_positives: list[tuple[Pred, str]] = []   # (строка, причина)
    for image in sorted(images | set(gt_by_image) | set(pred_by_image)):
        matches, unmatched = match_image(gt_by_image.get(image, []), pred_by_image.get(image, []))
        all_matches.extend(matches)
        for p in unmatched:
            if p.ptype != "other":
                reason = "лишняя строка: на изображении нет такого знака" if gt_by_image.get(image) else "знак на кадре без ГРЗ"
                false_positives.append((p, reason))
        for m in matches:
            if m.pred is not None and not m.gt.is_target and m.pred.ptype != "other":
                if m.gt.ptype == "other":
                    false_positives.append((m.pred, f"знак вне области задания прочитан как {m.pred.ptype} (эталон other)"))
                else:
                    false_positives.append((m.pred, f"знак не на ТС (is_vehicle=0) выдан как {m.pred.ptype}"))

    # ---- агрегаты по типам ----
    per_type: dict[str, dict] = {}
    for t in TARGET_TYPES:
        ms = [m for m in all_matches if m.gt.is_target and m.gt.ptype == t]
        n = len(ms)
        detected = [m for m in ms if m.pred is not None]
        full = [m for m in ms if m.full_ok]
        char_hits = sum(m.char_hits for m in ms)
        char_total = sum(m.char_total for m in ms)
        per_type[t] = {
            "gt_plates": n,
            "detected": len(detected),
            "plate_text_correct": sum(1 for m in ms if m.plate_ok),
            "type_correct_among_detected": sum(1 for m in detected if m.type_ok),
            "full_match": len(full),
            "recall_detect": len(detected) / n if n else None,
            "type_accuracy": (sum(1 for m in detected if m.type_ok) / len(detected)) if detected else None,
            "char_accuracy": char_hits / char_total if char_total else None,
            "accuracy": len(full) / n if n else None,
        }

    n_target = sum(v["gt_plates"] for v in per_type.values())
    n_full = sum(v["full_match"] for v in per_type.values())
    n_pred_nonother = sum(1 for p in preds if p.ptype != "other")
    precision = n_full / n_pred_nonother if n_pred_nonother else 0.0

    # ---- итоговый балл ----
    score = 0.0
    score_terms = {}
    for t in TARGET_TYPES:
        acc = per_type[t]["accuracy"]
        w = weights[t]
        if acc is None:  # типа нет в тестовом наборе — вес перераспределяется на precision
            score_terms[t] = None
            continue
        score += w * acc
        score_terms[t] = w * acc
    w_used = sum(weights[t] for t in TARGET_TYPES if per_type[t]["accuracy"] is not None)
    w_prec = 1.0 - w_used if w_used < 1.0 - 1e-9 else weights["precision"]
    score += w_prec * precision
    score_terms["precision"] = w_prec * precision
    total_w = w_used + w_prec
    score = score / total_w if total_w > 0 else 0.0

    # ---- калибровка confidence ----
    conf_ok = [m.pred.confidence for m in all_matches if m.full_ok]
    conf_bad = [p.confidence for p, _r in false_positives] + \
               [m.pred.confidence for m in all_matches if m.pred is not None and m.gt.is_target and not m.full_ok]
    calibration = {
        "mean_confidence_correct": sum(conf_ok) / len(conf_ok) if conf_ok else None,
        "mean_confidence_wrong": sum(conf_bad) / len(conf_bad) if conf_bad else None,
    }

    format_issue_rows = [p for p in preds if p.format_issues]
    return {
        "evaluator_version": VERSION,
        "images": len(images | set(gt_by_image) | set(pred_by_image)),
        "gt_target_plates": n_target,
        "gt_other_plates": sum(1 for g in gt_plates if not g.is_target),
        "pred_rows": len(preds),
        "pred_rows_non_other": n_pred_nonother,
        "pred_rows_with_format_issues": len(format_issue_rows),
        "per_type": per_type,
        "full_match_total": n_full,
        "false_positives": len(false_positives),
        "precision": precision,
        "score": score,
        "score_terms": score_terms,
        "weights": {**weights, "precision_effective": w_prec},
        "calibration": calibration,
        "_matches": all_matches,
        "_false_positives": false_positives,
    }


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def fmt_pct(v: float | None) -> str:
    return "  n/a " if v is None else f"{100 * v:6.2f}"


def render(res: dict, problems: list[str], max_score: float, max_list: int) -> str:
    out: list[str] = []
    out.append(f"evaluate.py v{VERSION} — Волга-IT 2026, полуфинал, номинация «ИИ и анализ данных»")
    out.append(f"Изображений: {res['images']}; целевых знаков в эталоне: {res['gt_target_plates']}; "
               f"нецелевых (other / не на ТС): {res['gt_other_plates']}; строк в ответе: {res['pred_rows']} "
               f"(с типом ≠ other: {res['pred_rows_non_other']})")
    out.append("")
    out.append("== По типам знаков ==")
    out.append(f"{'тип':8} {'знаков':>7} {'найдено':>8} {'текст ок':>9} {'тип ок':>7} {'засчитано':>10} | "
               f"{'recall%':>8} {'type%':>7} {'char%':>7} {'ACC%':>7}")
    for t in TARGET_TYPES:
        v = res["per_type"][t]
        out.append(f"{t:8} {v['gt_plates']:7d} {v['detected']:8d} {v['plate_text_correct']:9d} "
                   f"{v['type_correct_among_detected']:7d} {v['full_match']:10d} | "
                   f"{fmt_pct(v['recall_detect']):>8} {fmt_pct(v['type_accuracy']):>7} "
                   f"{fmt_pct(v['char_accuracy']):>7} {fmt_pct(v['accuracy']):>7}")
    out.append("")
    out.append(f"Засчитано всего: {res['full_match_total']} из {res['gt_target_plates']}; "
               f"ложных/лишних строк: {res['false_positives']}; precision = {100 * res['precision']:.2f}%")
    cal = res["calibration"]
    if cal["mean_confidence_correct"] is not None or cal["mean_confidence_wrong"] is not None:
        c_ok = cal["mean_confidence_correct"]
        c_bad = cal["mean_confidence_wrong"]
        out.append(f"Средняя confidence: у засчитанных {c_ok if c_ok is None else round(c_ok, 3)}, "
                   f"у ошибочных {c_bad if c_bad is None else round(c_bad, 3)}")
    out.append("")
    w = res["weights"]
    terms = res["score_terms"]
    out.append("== Балл ==")
    parts = []
    for t in TARGET_TYPES:
        if terms[t] is None:
            parts.append(f"{t}: нет в наборе")
        else:
            parts.append(f"{w[t]:.2f}·A_{t}={terms[t]:.4f}")
    parts.append(f"{w['precision_effective']:.2f}·P={terms['precision']:.4f}")
    out.append("Score = " + " + ".join(parts))
    out.append(f"Score = {res['score']:.4f}  →  {res['score'] * max_score:.2f} из {max_score:g} первичных баллов")
    out.append("")

    fps = res["_false_positives"]
    if fps:
        out.append(f"== Ложные / лишние строки ({len(fps)}) ==")
        for p, reason in fps[:max_list]:
            out.append(f"  решение:{p.line} {p.image};{p.plate};{p.ptype};{p.confidence:.2f} — {reason}")
        if len(fps) > max_list:
            out.append(f"  ... ещё {len(fps) - max_list} (см. --details)")
        out.append("")

    misses = [m for m in res["_matches"] if m.gt.is_target and not m.full_ok]
    if misses:
        out.append(f"== Не засчитанные целевые знаки ({len(misses)}) ==")
        for m in misses[:max_list]:
            if m.pred is None:
                out.append(f"  эталон:{m.gt.line} {m.gt.image} {m.gt.plate} {m.gt.ptype} — не найден")
            else:
                why = []
                if not m.plate_ok:
                    if len(m.pred.plate) != len(m.gt.plate):
                        why.append(f"текст «{m.pred.plate}» (длина {len(m.pred.plate)} ≠ {len(m.gt.plate)})")
                    else:
                        why.append(f"текст «{m.pred.plate}» ({m.char_hits}/{m.char_total} читаемых символов)")
                if not m.type_ok:
                    why.append(f"тип {m.pred.ptype}")
                out.append(f"  эталон:{m.gt.line} {m.gt.image} {m.gt.plate} {m.gt.ptype} — {', '.join(why)}")
        if len(misses) > max_list:
            out.append(f"  ... ещё {len(misses) - max_list} (см. --details)")
        out.append("")

    if problems or res["pred_rows_with_format_issues"]:
        out.append(f"== Замечания к формату ({len(problems)} + строк с проблемами: {res['pred_rows_with_format_issues']}) ==")
        for pr in problems[:max_list]:
            out.append("  " + pr)
        for p in [p for p in res["_all_preds"] if p.format_issues][:max_list]:
            out.append(f"  решение:{p.line}: " + "; ".join(p.format_issues))
        out.append("")
    return "\n".join(out)


def write_details(path: str, res: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["image", "gt_plate", "gt_type", "gt_is_vehicle", "pred_plate", "pred_type",
                    "confidence", "similarity", "plate_ok", "type_ok", "counted", "status"])
        for m in res["_matches"]:
            if m.pred is None:
                status = "missed" if m.gt.is_target else "ignored-not-target"
                w.writerow([m.gt.image, m.gt.plate, m.gt.ptype, int(m.gt.is_vehicle), "", "", "", "0",
                            "", "", "", status])
            else:
                if m.gt.is_target:
                    status = "ok" if m.full_ok else "wrong"
                else:
                    status = "ok-rejected" if m.pred.ptype == "other" else "false-positive"
                w.writerow([m.gt.image, m.gt.plate, m.gt.ptype, int(m.gt.is_vehicle), m.pred.plate, m.pred.ptype,
                            f"{m.pred.confidence:.3f}", f"{m.similarity:.3f}", int(m.plate_ok), int(m.type_ok),
                            int(m.full_ok and m.gt.is_target), status])
        for p, reason in res["_false_positives"]:
            if not any(m.pred is p for m in res["_matches"]):
                w.writerow([p.image, "", "", "", p.plate, p.ptype, f"{p.confidence:.3f}", "0", "", "", "",
                            f"false-positive: {reason}"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подсчёт метрик решения (Волга-IT 2026)")
    parser.add_argument("--gt", required=True, help="эталонный CSV (image;plate_num;plate_type;is_vehicle)")
    parser.add_argument("--pred", required=True, help="CSV решения (image;plate_num;plate_type;confidence)")
    parser.add_argument("--images", default=None, help="каталог тестовых изображений (полный список кадров)")
    parser.add_argument("--json", default=None, help="сохранить метрики в JSON")
    parser.add_argument("--details", default=None, help="сохранить построчное сопоставление в CSV")
    parser.add_argument("--max-score", type=float, default=100.0, help="максимум первичных баллов")
    parser.add_argument("--w-type1", type=float, default=0.15)
    parser.add_argument("--w-type1a", type=float, default=0.35)
    parser.add_argument("--w-type1b", type=float, default=0.35)
    parser.add_argument("--w-precision", type=float, default=0.15)
    parser.add_argument("--max-list", type=int, default=40, help="сколько строк печатать в списках ошибок")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    weights = {"type1": args.w_type1, "type1a": args.w_type1a, "type1b": args.w_type1b, "precision": args.w_precision}
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        weights = {k: v / total_w for k, v in weights.items()}

    problems: list[str] = []
    try:
        gt_plates, images = load_gt(args.gt, problems)
    except OSError as exc:
        print(f"Не удалось прочитать эталон: {exc}", file=sys.stderr)
        return 2
    if args.images:
        if not os.path.isdir(args.images):
            print(f"Каталог не найден: {args.images}", file=sys.stderr)
            return 2
        images |= {f for f in os.listdir(args.images) if f.lower().endswith(IMAGE_EXTS)}
    try:
        preds = load_pred(args.pred, images, problems)
    except OSError as exc:
        print(f"Не удалось прочитать ответ решения: {exc}", file=sys.stderr)
        return 2

    res = evaluate(gt_plates, images, preds, weights)
    res["_all_preds"] = preds
    print(render(res, problems, args.max_score, args.max_list))

    if args.details:
        write_details(args.details, res)
        print(f"Построчное сопоставление: {os.path.abspath(args.details)}")
    if args.json:
        public = {k: v for k, v in res.items() if not k.startswith("_")}
        public["format_problems"] = problems
        public["primary_score"] = res["score"] * args.max_score
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(public, f, ensure_ascii=False, indent=2)
        print(f"JSON: {os.path.abspath(args.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
