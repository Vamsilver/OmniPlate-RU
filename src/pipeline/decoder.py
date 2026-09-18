#!/usr/bin/env python3
r"""
OmniPlate-RU Syntactic FSM Beam Search Decoder Module (Stage 2).

Implements finite-state machine (FSM) constrained Beam Search (K=10)
and hypothesis scoring tailored strictly to GOST R 50577-2018:
- Type 1:  [ABEKMHOPCTYX] \d{3} [ABEKMHOPCTYX]{2} \d{2,3} (8 or 9 chars)
- Type 1B: [ABEKMHOPCTYX]{2} \d{3} \d{2,3}                (7 or 8 chars)
- Type 2:  [ABEKMHOPCTYX]{2} \d{4} \d{2,3}                (8 or 9 chars)

Eliminates mutual character-digit confusions (0/O, 8/B, 1/T) via positional masking.
"""

import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

# Character Set & CTC Tokens
# ---------------------------------------------------------------------------
# Blank token is at index 0
BLANK_TOKEN = "-"
DIGITS = "0123456789"
LETTERS = "ABEKMHOPCTYX"
WILDCARD = "#"

VOCAB = [BLANK_TOKEN] + list(DIGITS) + list(LETTERS) + [WILDCARD]
CHAR2IDX: Dict[str, int] = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR: Dict[int, str] = {i: c for i, c in enumerate(VOCAB)}
BLANK_IDX = 0
NUM_CLASSES = len(VOCAB)

# Standard GOST RegEx
PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")
PLATE_TYPE2_REGEX = re.compile(r"^[ABEKMHOPCTYX#]{2}[\d#]{4}[\d#]{2,3}$")
PLATE_TYPE1B_REGEX = re.compile(r"^[ABEKMHOPCTYX#]{2}[\d#]{3}[\d#]{2,3}$")
VALID_3DIGIT_STARTS = {"1", "2", "7", "#"}

# Confusion replacement maps
DIGIT_TO_LETTER: Dict[str, str] = {
    "0": "O",
    "8": "B",
    "1": "T",  # Visual similarity in some fonts
}
LETTER_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "B": "8",
    "C": "0",
    "D": "0",
}

# Official Russian GIBDD 3-digit region codes
VALID_3DIGIT_REGIONS = {
    "102", "113", "116", "121", "122", "123", "124", "125", "126",
    "134", "136", "138", "142", "147", "150", "152", "154", "155", "156",
    "159", "161", "163", "164", "169", "172", "173", "174", "177", "178", "180",
    "181", "184", "185", "186", "190", "193", "196", "197", "198", "199",
    "252", "277", "323", "336", "702", "716", "725", "750", "754", "761", "763", "774", "777",
    "790", "797", "799", "977", "550"
}

# Official Russian GIBDD 2-digit region codes (active & historical legal codes 01..99)
VALID_2DIGIT_REGIONS = {f"{i:02d}" for i in range(1, 100)}


def is_valid_region(reg: str, allow_wildcards: bool = False) -> bool:
    """
    Validates whether 2-digit or 3-digit code corresponds to a real Russian administrative region.
    If allow_wildcards is True, matches candidate regions containing '#' wildcards.
    """
    if not reg:
        return False
    if not allow_wildcards:
        if not reg.isdigit():
            return False
        if len(reg) == 2:
            return reg in VALID_2DIGIT_REGIONS
        elif len(reg) == 3:
            return reg in VALID_3DIGIT_REGIONS
        return False

    # Wildcard support
    if len(reg) == 2:
        return any(all(c1 == "#" or c1 == c2 for c1, c2 in zip(reg, valid_r)) for valid_r in VALID_2DIGIT_REGIONS)
    elif len(reg) == 3:
        return any(all(c1 == "#" or c1 == c2 for c1, c2 in zip(reg, valid_r)) for valid_r in VALID_3DIGIT_REGIONS)
    return False


def is_valid_gost_plate(text: str, plate_type: str = "type1", allow_wildcards: bool = False) -> bool:
    """
    Strictly validates Russian license plate format by position and legal region code.
    If allow_wildcards is False:
        Returns True ONLY if text has 0 wildcards and matches official GOST R 50577-2018 mask.
    If allow_wildcards is True:
        Accepts unreadable positions marked with '#' adhering to REGEX_WITH_WILDCARDS
        from docs/specs/02_plate_mask_regex.md.
    """
    if not text:
        return False
    if not allow_wildcards and "#" in text:
        return False

    norm_type = plate_type.lower().strip()
    if norm_type in ("type1", "type1a"):
        pattern = (
            r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$"
            if allow_wildcards
            else r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$"
        )
        if not re.match(pattern, text):
            return False
        reg = text[6:]
    elif norm_type == "type1b":
        pattern_bus = (
            r"^[ABEKMHOPCTYX#]{2}[\d#]{3}[\d#]{2,3}$"
            if allow_wildcards
            else r"^[ABEKMHOPCTYX]{2}\d{3}\d{2,3}$"
        )
        pattern_car = (
            r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$"
            if allow_wildcards
            else r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$"
        )
        if re.match(pattern_bus, text):
            reg = text[5:]
        elif re.match(pattern_car, text):
            reg = text[6:]
        else:
            return False
    elif norm_type in ("type2", "trailer"):
        pattern = (
            r"^[ABEKMHOPCTYX#]{2}[\d#]{4}[\d#]{2,3}$"
            if allow_wildcards
            else r"^[ABEKMHOPCTYX]{2}\d{4}\d{2,3}$"
        )
        if not re.match(pattern, text):
            return False
        reg = text[6:]
    else:
        return False

    return is_valid_region(reg, allow_wildcards=allow_wildcards)


# Empirical demographic & traffic frequency prior weights for Russian regions
# Tier 1: Federal capital hubs (Moscow, Moscow Oblast, Saint Petersburg, Leningrad Oblast)
TIER1_REGIONS = {
    "77", "97", "99", "177", "197", "199", "777", "797", "799", "977",
    "50", "90", "150", "190", "750", "790", "550",
    "78", "98", "178", "198",
    "47", "147"
}

# Tier 2: Million-plus population centers & major economic regional hubs
TIER2_REGIONS = {
    "23", "93", "123", "193",  # Krasnodar Krai
    "66", "96", "196",        # Sverdlovsk (Yekaterinburg)
    "16", "116", "716",       # Tatarstan (Kazan)
    "52", "152", "252",       # Nizhny Novgorod
    "63", "163", "763",       # Samara
    "61", "161", "761",       # Rostov
    "02", "102", "702",       # Bashkortostan (Ufa)
    "54", "154", "754",       # Novosibirsk
    "74", "174", "774",       # Chelyabinsk
    "59", "159",              # Perm Krai
    "24", "124",              # Krasnoyarsk Krai
    "36", "136",              # Voronezh
    "34", "134",              # Volgograd
    "25", "125",              # Primorsky (Vladivostok)
    "64", "164",              # Saratov
    "72", "172",              # Tyumen
    "55", "155",              # Omsk
    "38", "138",              # Irkutsk
    "26", "126",              # Stavropol Krai
    "42", "142",              # Kemerovo
    "35", "39", "31", "71", "69", "33", "62", "40", "48", "58", "43", "73", "173",
    "21", "121", "18", "82", "92", "56", "156", "11", "169", "86", "186", "89",
}

REGION_FREQUENCY_WEIGHTS: Dict[str, float] = {}
for _r in TIER1_REGIONS:
    REGION_FREQUENCY_WEIGHTS[_r] = 0.06
for _r in TIER2_REGIONS:
    if _r not in REGION_FREQUENCY_WEIGHTS:
        REGION_FREQUENCY_WEIGHTS[_r] = 0.02

# Demographically prioritized regional economic hubs & highway corridors
REGION_FREQUENCY_WEIGHTS["36"] = 0.035
REGION_FREQUENCY_WEIGHTS["136"] = 0.035
REGION_FREQUENCY_WEIGHTS["102"] = 0.035
REGION_FREQUENCY_WEIGHTS["702"] = 0.035
REGION_FREQUENCY_WEIGHTS["152"] = 0.030
REGION_FREQUENCY_WEIGHTS["252"] = 0.030
REGION_FREQUENCY_WEIGHTS["174"] = 0.035
REGION_FREQUENCY_WEIGHTS["774"] = 0.035
REGION_FREQUENCY_WEIGHTS["56"] = 0.015
REGION_FREQUENCY_WEIGHTS["156"] = 0.015



OPTICAL_DIGIT_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("8", "0"): 0.5, ("0", "8"): 0.5,
    ("5", "0"): 0.5, ("0", "5"): 0.5,
    ("6", "5"): 0.7, ("5", "6"): 0.7,
    ("4", "7"): 0.5, ("7", "4"): 0.5,
    ("2", "7"): 0.5, ("7", "2"): 0.5,
    ("2", "9"): 0.8, ("9", "2"): 0.8,
    ("1", "7"): 0.6, ("7", "1"): 0.6,
    ("5", "9"): 0.7, ("9", "5"): 0.7,
    ("3", "8"): 0.7, ("8", "3"): 0.7,
    ("6", "8"): 0.7, ("8", "6"): 0.7,
    ("9", "0"): 0.8, ("0", "9"): 0.8,
    ("9", "7"): 0.8, ("7", "9"): 0.8,
}

OPTICAL_LETTER_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("K", "H"): 0.5, ("H", "K"): 0.5,
    ("Y", "T"): 0.5, ("T", "Y"): 0.5,
    ("M", "T"): 0.6, ("T", "M"): 0.6,
    ("C", "M"): 0.6, ("M", "C"): 0.6,
    ("C", "O"): 0.6, ("O", "C"): 0.6,
    ("B", "O"): 0.6, ("O", "B"): 0.6,
    ("E", "P"): 0.6, ("P", "E"): 0.6,
    ("B", "P"): 0.7, ("P", "B"): 0.7,
    ("Y", "E"): 0.6, ("E", "Y"): 0.6,
    ("C", "A"): 0.7, ("A", "C"): 0.7,
    ("H", "M"): 0.6, ("M", "H"): 0.6,
    ("X", "K"): 0.7, ("K", "X"): 0.7,
    ("X", "H"): 0.7, ("H", "X"): 0.7,
}


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FSM Beam Search Decoder (Stage 2)
# ---------------------------------------------------------------------------
class FSMBeamSearchDecoder:
    """
    Finite State Machine (FSM) Constrained Beam Search Decoder (K=10).
    Guarantees strict compliance with Russian license plate masks (GOST R 50577-2018).
    """

    DIGIT_INDICES: Tuple[int, ...] = tuple(CHAR2IDX[d] for d in DIGITS)
    LETTER_INDICES: Tuple[int, ...] = tuple(CHAR2IDX[l] for l in LETTERS)

    ALLOWED_TABLE: Dict[str, Dict[int, Tuple[int, ...]]] = {}
    for _pt in ("type1", "type1a", "type1b", "type2", "trailer"):
        ALLOWED_TABLE[_pt] = {}
        for _llen in range(10):
            if _pt in ("type1", "type1a"):
                ALLOWED_TABLE[_pt][_llen] = () if _llen >= 9 else (LETTER_INDICES if _llen in (0, 4, 5) else DIGIT_INDICES)
            elif _pt == "type1b":
                ALLOWED_TABLE[_pt][_llen] = () if _llen >= 8 else (LETTER_INDICES if _llen in (0, 1) else DIGIT_INDICES)
            else:
                ALLOWED_TABLE[_pt][_llen] = () if _llen >= 9 else (LETTER_INDICES if _llen in (0, 1) else DIGIT_INDICES)

    @classmethod
    def get_allowed_tokens(cls, prefix_len: int, plate_type: str = "type1") -> Tuple[int, ...]:
        norm_type = plate_type.lower().strip()
        table = cls.ALLOWED_TABLE.get(norm_type, cls.ALLOWED_TABLE["type1"])
        return table.get(prefix_len, ())

    @classmethod
    def decode_fsm_beam_search(
        cls,
        logits: np.ndarray,
        plate_type: str = "type1",
        beam_width: int = 10,
        blank_idx: int = BLANK_IDX,
    ) -> List[Tuple[str, float]]:
        import math
        if logits.ndim == 3:
            logits = logits[0]

        max_l = np.max(logits, axis=-1, keepdims=True)
        log_probs = logits - max_l - np.log(np.sum(np.exp(logits - max_l), axis=-1, keepdims=True))

        T, C = log_probs.shape
        NEG_INF = -1e9

        def log_sum_exp(a: float, b: float) -> float:
            if a <= -1e8: return b
            if b <= -1e8: return a
            if a > b:
                diff = b - a
                return a + (math.log1p(math.exp(diff)) if diff > -37.0 else 0.0)
            else:
                diff = a - b
                return b + (math.log1p(math.exp(diff)) if diff > -37.0 else 0.0)

        norm_type = plate_type.lower().strip()
        max_target_len = 8 if norm_type == "type1b" else 9
        min_target_len = 7 if norm_type == "type1b" else 8
        table = cls.ALLOWED_TABLE.get(norm_type, cls.ALLOWED_TABLE["type1"])

        beams_by_len: Dict[int, Dict[Tuple[int, ...], Tuple[float, float]]] = {
            0: {(): (0.0, NEG_INF)}
        }

        for t in range(T):
            curr_lp = log_probs[t]
            lp_blank = float(curr_lp[blank_idx])

            # High-speed early blank skip: if blank probability is >99%, only blank emissions occur
            if lp_blank > -0.01:
                for l_len, beams in beams_by_len.items():
                    for p, (pb, pnb) in beams.items():
                        beams[p] = (log_sum_exp(pb, pnb) + lp_blank, NEG_INF)
                continue

            new_beams_by_len: Dict[int, Dict[Tuple[int, ...], Tuple[float, float]]] = {
                l: {} for l in range(max_target_len + 1)
            }

            for l_len, beams in beams_by_len.items():
                allowed = table.get(l_len, ())
                for prefix, (p_b, p_nb) in beams.items():
                    p_total = log_sum_exp(p_b, p_nb)

                    # 1. Blank emission maintains length
                    cur_pb, cur_pnb = new_beams_by_len[l_len].get(prefix, (NEG_INF, NEG_INF))
                    new_beams_by_len[l_len][prefix] = (log_sum_exp(cur_pb, p_total + lp_blank), cur_pnb)

                    # 2. Self-loop maintains length
                    if l_len > 0:
                        last_c = prefix[-1]
                        cur_pb, cur_pnb = new_beams_by_len[l_len].get(prefix, (NEG_INF, NEG_INF))
                        new_beams_by_len[l_len][prefix] = (cur_pb, log_sum_exp(cur_pnb, p_nb + float(curr_lp[last_c])))

                    # 3. Transition to next position
                    if l_len < max_target_len and allowed:
                        next_l = l_len + 1
                        for c in allowed:
                            lp_c = float(curr_lp[c])
                            if lp_c < -4.5:
                                continue
                            new_pref = prefix + (c,)
                            cur_pb, cur_pnb = new_beams_by_len[next_l].get(new_pref, (NEG_INF, NEG_INF))
                            if l_len > 0 and c == prefix[-1]:
                                new_beams_by_len[next_l][new_pref] = (cur_pb, log_sum_exp(cur_pnb, p_b + lp_c))
                            else:
                                new_beams_by_len[next_l][new_pref] = (cur_pb, log_sum_exp(cur_pnb, p_total + lp_c))

            beams_by_len = {}
            for l_len in range(max_target_len + 1):
                beams = new_beams_by_len[l_len]
                if beams:
                    scored = [(p, log_sum_exp(pb, pnb), pb, pnb) for p, (pb, pnb) in beams.items()]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    beams_by_len[l_len] = {p: (pb, pnb) for p, s, pb, pnb in scored[:beam_width]}

        candidates = []
        for l_len in range(min_target_len, max_target_len + 1):
            for pref, (pb, pnb) in beams_by_len.get(l_len, {}).items():
                text = "".join(IDX2CHAR[idx] for idx in pref)
                score = log_sum_exp(pb, pnb)
                norm_score = score / float(l_len)
                reg_start = 5 if norm_type == "type1b" else 6
                reg = text[reg_start:]
                if is_valid_region(reg):
                    norm_score += 0.25 + REGION_FREQUENCY_WEIGHTS.get(reg, 0.0)
                else:
                    norm_score -= 3.0
                candidates.append((text, norm_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:beam_width]


class CTCDecoder:
    """
    Decodes CTC raw output probabilities or index sequences into text
    with optional GOST mask correction.
    """

    @staticmethod
    def decode_greedy(indices: Sequence[int], blank_idx: int = BLANK_IDX) -> str:
        """
        Standard CTC greedy collapse:
        Removes repeated tokens and blank tokens.
        """
        raw_chars = []
        prev = blank_idx
        for idx in indices:
            if idx != prev:
                if idx != blank_idx and idx in IDX2CHAR:
                    raw_chars.append(IDX2CHAR[idx])
                prev = idx
            else:
                prev = idx
        return "".join(raw_chars)

    @staticmethod
    def decode_beam_search(
        log_probs: np.ndarray,
        beam_width: int = 5,
        prune_top_k: int = 8,
        blank_threshold: float = -0.01,
        blank_idx: int = BLANK_IDX,
    ) -> List[Tuple[str, float]]:
        """
        Performs high-speed CTC Prefix Beam Search.
        Merges duplicate character alignments, models blank transitions,
        and returns Top-K candidate decoded text sequences.

        Args:
            log_probs: Normalized log-probabilities of shape (T, num_classes).
            beam_width: Number of active candidate beams to maintain (default 5).
            prune_top_k: Number of highest-probability tokens considered per frame.
            blank_threshold: Log probability above which frame is treated as pure blank skip.
            blank_idx: Index of the blank token (default 0).

        Returns:
            List of (decoded_text, log_probability) tuples sorted descending by score.
        """
        T, C = log_probs.shape
        NEG_INF = -1e9

        def log_sum(a: float, b: float) -> float:
            if a <= NEG_INF:
                return b
            if b <= NEG_INF:
                return a
            m = max(a, b)
            return m + np.log(np.exp(a - m) + np.exp(b - m))

        # Beam dictionary: prefix tuple of token indices -> (p_blank, p_non_blank)
        beams = {(): (0.0, NEG_INF)}

        for t in range(T):
            curr_lp = log_probs[t]
            lp_blank = float(curr_lp[blank_idx])

            # Early skip: if frame is overwhelmingly blank (>99% probability), fast-forward
            if lp_blank > blank_threshold:
                new_beams = {}
                for prefix, (p_b, p_nb) in beams.items():
                    p_tot = log_sum(p_b, p_nb)
                    new_beams[prefix] = (p_tot + lp_blank, NEG_INF)
                beams = new_beams
                continue

            top_tokens = np.argsort(curr_lp)[-prune_top_k:]
            if blank_idx not in top_tokens:
                top_tokens = np.append(top_tokens, blank_idx)

            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_total = log_sum(p_b, p_nb)
                for c in top_tokens:
                    lp_c = float(curr_lp[c])
                    if c == blank_idx:
                        cur_pb, cur_pnb = new_beams.get(prefix, (NEG_INF, NEG_INF))
                        new_beams[prefix] = (log_sum(cur_pb, p_total + lp_c), cur_pnb)
                    else:
                        if len(prefix) > 0 and c == prefix[-1]:
                            cur_pb, cur_pnb = new_beams.get(prefix + (c,), (NEG_INF, NEG_INF))
                            new_beams[prefix + (c,)] = (cur_pb, log_sum(cur_pnb, p_b + lp_c))
                            cur_pb2, cur_pnb2 = new_beams.get(prefix, (NEG_INF, NEG_INF))
                            new_beams[prefix] = (cur_pb2, log_sum(cur_pnb2, p_nb + lp_c))
                        else:
                            new_pref = prefix + (c,)
                            cur_pb, cur_pnb = new_beams.get(new_pref, (NEG_INF, NEG_INF))
                            new_beams[new_pref] = (cur_pb, log_sum(cur_pnb, p_total + lp_c))

            # Prune to beam_width
            scored = [(p, log_sum(pb, pnb), pb, pnb) for p, (pb, pnb) in new_beams.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            beams = {p: (pb, pnb) for p, s, pb, pnb in scored[:beam_width]}

        results = []
        for pref, (pb, pnb) in beams.items():
            text = "".join(IDX2CHAR[idx] for idx in pref)
            score = log_sum(pb, pnb)
            results.append((text, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def strip_parasitic_edge(text: str) -> str:
        """
        Detects and trims parasitic edge / frame strokes at the beginning of license plates.
        For example: 'HM235PX77' -> 'M235PX77', '#M235PX777' -> 'M235PX777', 'KM235PX77' -> 'M235PX77'.
        Trigger: When text has >= 9 characters, and characters starting from index 1 conform
        to the standard Type 1 structure (1 letter, 3 digits, 2 letters, 2-3 digits),
        while index 0 does NOT form a valid Type 1 plate.
        """
        if not text or len(text) < 9:
            return text

        chars = list(text.strip().upper())
        # Index 1 MUST be a genuine letter (not digit)
        c1_is_real_letter = chars[1] in LETTERS
        # Index 2..4 MUST be digits
        c24_are_real_digits = all(c in DIGITS for c in chars[2:5])
        # Index 5..6 MUST be genuine letters
        c56_are_real_letters = all(c in LETTERS for c in chars[5:7])
        # Index 7..end MUST be digits
        c_region_digits = all(c in DIGITS for c in chars[7:])

        # If prefix HM..., #M..., KM... transitions into standard DDD LL RR, strip the parasitic frame stroke!
        if c1_is_real_letter and c24_are_real_digits and c56_are_real_letters and c_region_digits:
            return "".join(chars[1:])

        return text

    @staticmethod
    def repair_3digit_region(reg: str) -> str:
        """
        Validates and auto-repairs 3-digit region codes against official GIBDD database.
        Maps optical digit confusions (e.g. 172->177, 192->197, 798->790, 165->125) to valid codes.
        """
        if len(reg) != 3:
            return reg
        if reg in VALID_3DIGIT_REGIONS:
            return reg
        if "#" in reg:
            matches = [v for v in VALID_3DIGIT_REGIONS if all(c1 == "#" or c1 == c2 for c1, c2 in zip(reg, v))]
            if len(matches) == 1:
                return matches[0]
            return reg

        # Find closest valid region using weighted optical distance and frequency prior
        best_cand = reg
        min_dist = 999.0
        for v in VALID_3DIGIT_REGIONS:
            diffs = sum(1 for c1, c2 in zip(reg, v) if c1 != c2)
            if diffs == 1:
                w_dist = sum(OPTICAL_DIGIT_WEIGHTS.get((c1, c2), 1.0) for c1, c2 in zip(reg, v) if c1 != c2)
                w_dist -= REGION_FREQUENCY_WEIGHTS.get(v, 0.0)
                if w_dist < min_dist:
                    min_dist = w_dist
                    best_cand = v

        return best_cand


    @staticmethod
    def apply_gost_heuristics(plate: str, plate_type: str = "type1") -> str:
        """
        Applies strict position-aware character repair based on GOST R 50577-2018.
        Guarantees that the output ALWAYS strictly adheres to the Russian license plate mask:
        - Type 1 / 1A: L DDD LL RR (8 chars) or L DDD LL RRR (9 chars).
        - Type 2 (Trailers): LL DDDD RR (8 chars) or LL DDDD RRR (9 chars).
        - Type 1B (Buses): LL DDD DD (7 chars) or unified L DDD LL RR (8/9 chars).
        - Resolves over-long sequences (>9 chars) by optimal template-matching window selection.
        - Replaces unresolvable / corrupt positions with '#' wildcard.
        """
        if not plate:
            return plate

        chars = list(plate.strip().upper())
        norm_type = plate_type.lower().strip()

        # Early check: if plate starts with a parasitic edge stroke that masks Type 1 into Type 2
        # (e.g. HM235PX77 or #M235PX77), strip the edge stroke and restore Type 1
        stripped = CTCDecoder.strip_parasitic_edge("".join(chars))
        if len(stripped) < len(chars):
            chars = list(stripped)
            if norm_type in ("type2", "trailer", "type_2"):
                norm_type = "type1"

        # -------------------------------------------------------------------
        # 1. Russian Trailer format (Type 2: LL DDDD RR / LL DDDD RRR)
        # -------------------------------------------------------------------
        if norm_type in ("type2", "trailer", "type_2"):
            if len(chars) > 9:
                best_sub = chars[:9]
                best_score = -999
                for w_len in (8, 9):
                    for start in range(len(chars) - w_len + 1):
                        sub = chars[start : start + w_len]
                        score = 0
                        # Letter positions: 0, 1
                        for lp in (0, 1):
                            if sub[lp] in LETTERS:
                                score += 3
                            elif sub[lp] in DIGIT_TO_LETTER and DIGIT_TO_LETTER[sub[lp]] in LETTERS:
                                score += 1
                            else:
                                score -= 2
                        # Digit positions: 2..end
                        for dp in range(2, w_len):
                            if sub[dp] in DIGITS:
                                score += 3
                            elif sub[dp] in LETTER_TO_DIGIT:
                                score += 1
                            else:
                                score -= 2
                        if w_len == 9:
                            if sub[6] in VALID_3DIGIT_STARTS:
                                score += 2
                            else:
                                score -= 5
                        if score > best_score:
                            best_score = score
                            best_sub = sub
                chars = best_sub

            # Check 3-digit region validity for 9-char
            if len(chars) == 9:
                if chars[8] not in DIGITS and chars[8] not in LETTER_TO_DIGIT:
                    chars = chars[:8]
                else:
                    c6 = chars[6]
                    if c6 in LETTER_TO_DIGIT:
                        c6 = LETTER_TO_DIGIT[c6]
                        chars[6] = c6
                    if c6 not in VALID_3DIGIT_STARTS:
                        chars = chars[:8]

            # If shorter than 8, pad with #
            if len(chars) < 8:
                padded = ["#"] * 8
                padded[:len(chars)] = chars
                chars = padded

            # Enforce Letter positions: 0, 1
            for pos in (0, 1):
                if pos < len(chars):
                    c = chars[pos]
                    if c in LETTERS:
                        pass
                    elif c in DIGIT_TO_LETTER and DIGIT_TO_LETTER[c] in LETTERS:
                        chars[pos] = DIGIT_TO_LETTER[c]
                    else:
                        chars[pos] = "#"

            # Enforce Digit positions: 2..end
            for pos in range(2, len(chars)):
                c = chars[pos]
                if c in DIGITS:
                    pass
                elif c in LETTER_TO_DIGIT:
                    chars[pos] = LETTER_TO_DIGIT[c]
                else:
                    chars[pos] = "#"

            # Auto-repair 3-digit trailer region code against official GIBDD database
            if len(chars) == 9 and all(c in DIGITS or c == "#" for c in chars[6:]):
                raw_reg = "".join(chars[6:])
                chars[6:] = list(CTCDecoder.repair_3digit_region(raw_reg))

            return "".join(chars)

        # -------------------------------------------------------------------
        # 2. Classic Russian bus format (Type 1B: LL DDD DD - 7 chars, or LL DDD DDD - 8 chars)
        # -------------------------------------------------------------------
        is_classic_bus = False
        if norm_type == "type1b":
            has_letters_at_45 = len(chars) >= 6 and (chars[4] in LETTERS or chars[5] in LETTERS)
            if not has_letters_at_45:
                c0_is_let = chars[0] in LETTERS or chars[0] in DIGIT_TO_LETTER
                c1_is_let = len(chars) > 1 and (chars[1] in LETTERS or (chars[1] in DIGIT_TO_LETTER and len(chars) <= 7))
                if c0_is_let and c1_is_let and len(chars) in (7, 8):
                    if len(chars) > 3 and (chars[2] in DIGITS or chars[2] in LETTER_TO_DIGIT):
                        is_classic_bus = True

        if is_classic_bus:
            bus_chars = chars[:8] if len(chars) == 8 and (chars[7] in DIGITS or chars[7] in LETTER_TO_DIGIT) else chars[:7]
            while len(bus_chars) < 7:
                bus_chars.append("#")
            # Pos 0, 1 -> Letters
            for pos in (0, 1):
                c = bus_chars[pos]
                if c in LETTERS:
                    pass
                elif c in DIGIT_TO_LETTER and DIGIT_TO_LETTER[c] in LETTERS:
                    bus_chars[pos] = DIGIT_TO_LETTER[c]
                else:
                    bus_chars[pos] = "#"
            # Pos 2..end -> Digits
            for pos in range(2, len(bus_chars)):
                c = bus_chars[pos]
                if c in DIGITS:
                    pass
                elif c in LETTER_TO_DIGIT:
                    bus_chars[pos] = LETTER_TO_DIGIT[c]
                else:
                    bus_chars[pos] = "#"
            # Auto-repair 3-digit region code against official GIBDD database
            if len(bus_chars) == 8 and all(c in DIGITS or c == "#" for c in bus_chars[5:]):
                raw_reg = "".join(bus_chars[5:])
                bus_chars[5:] = list(CTCDecoder.repair_3digit_region(raw_reg))

            return "".join(bus_chars)

        # -------------------------------------------------------------------
        # 3. Standard Unified GOST (Type 1, Type 1A, and unified Type 1B):
        # Target formats: L DDD LL RR (8 chars) or L DDD LL RRR (9 chars)
        # -------------------------------------------------------------------
        def _score_type1_window(sub_chars, w_len):
            sc = 0
            for lp in (0, 4, 5):
                c = sub_chars[lp]
                if c in LETTERS:
                    sc += 3
                elif c in DIGIT_TO_LETTER and DIGIT_TO_LETTER[c] in LETTERS:
                    sc += 1
                elif c == "#":
                    sc += 0
                else:
                    sc -= 2
            for dp in [1, 2, 3] + list(range(6, w_len)):
                c = sub_chars[dp]
                if c in DIGITS:
                    sc += 3
                elif c in LETTER_TO_DIGIT:
                    sc += 1
                elif c == "#":
                    sc += 0
                else:
                    sc -= 2
            if w_len == 9:
                if sub_chars[6] in VALID_3DIGIT_STARTS:
                    sc += 2
                elif sub_chars[6] == "#":
                    sc += 0
                else:
                    sc -= 5
            return sc

        best_sub = list(chars[:8]) if len(chars) >= 8 else list(chars) + ["#"] * (8 - len(chars))
        best_score = -9999

        for w_len in (8, 9):
            if len(chars) >= w_len:
                for start in range(len(chars) - w_len + 1):
                    sub = chars[start : start + w_len]
                    sc = _score_type1_window(sub, w_len)
                    if sc > best_score:
                        best_score = sc
                        best_sub = sub
            else:
                # When len(chars) < w_len, evaluate candidate placements
                sub_candidates = []
                for s in range(w_len - len(chars) + 1):
                    sub_candidates.append((chars, s))

                chars_str = "".join(chars)
                for m in re.finditer(r"\d{2,3}", chars_str):
                    d_idx = m.start()
                    sub_slice = chars[d_idx:]
                    sub_candidates.append((sub_slice, 1))
                    if 1 - d_idx >= 0:
                        sub_candidates.append((chars, 1 - d_idx))

                for sub, s in sub_candidates:
                    cand = ["#"] * w_len
                    for idx, c in enumerate(sub):
                        t_pos = s + idx
                        if 0 <= t_pos < w_len:
                            cand[t_pos] = c
                    sc = _score_type1_window(cand, w_len)
                    if sc > best_score:
                        best_score = sc
                        best_sub = cand

        chars = best_sub

        # If exactly 9 chars, check if 3-digit region is valid
        if len(chars) == 9:
            if chars[8] not in DIGITS and chars[8] not in LETTER_TO_DIGIT:
                chars = chars[:8]
            else:
                c6 = chars[6]
                if c6 in LETTER_TO_DIGIT:
                    c6 = LETTER_TO_DIGIT[c6]
                    chars[6] = c6
                if c6 not in VALID_3DIGIT_STARTS:
                    chars = chars[:8]

        # If shorter than 8, pad with # to valid length
        while len(chars) < 8:
            chars.append("#")

        # Strictly enforce GOST symbol types by position:
        # Expected Letter positions: 0, 4, 5 (MUST BE LETTERS)
        for pos in (0, 4, 5):
            if pos < len(chars):
                c = chars[pos]
                if c in LETTERS:
                    pass
                elif c in DIGIT_TO_LETTER and DIGIT_TO_LETTER[c] in LETTERS:
                    chars[pos] = DIGIT_TO_LETTER[c]
                else:
                    chars[pos] = "#"

        # Expected Digit positions: 1, 2, 3, and 6..end (MUST BE DIGITS)
        digit_positions = [1, 2, 3] + list(range(6, len(chars)))
        for pos in digit_positions:
            if pos < len(chars):
                c = chars[pos]
                if c in DIGITS:
                    pass
                elif c in LETTER_TO_DIGIT:
                    chars[pos] = LETTER_TO_DIGIT[c]
                else:
                    chars[pos] = "#"

        # Auto-repair 3-digit region code against official GIBDD database
        if len(chars) == 9 and all(c in DIGITS or c == "#" for c in chars[6:]):
            raw_reg = "".join(chars[6:])
            chars[6:] = list(CTCDecoder.repair_3digit_region(raw_reg))

        return "".join(chars)

    @staticmethod
    def compute_ctc_log_prob(
        log_probs: np.ndarray,
        target_str: str,
        blank_idx: int = BLANK_IDX,
    ) -> float:
        """
        Computes exact CTC forward log-likelihood log P(target_str | log_probs).
        Uses PyTorch C++ acceleration if torch is available, otherwise fast NumPy forward variable.

        Args:
            log_probs: Normalized log-probabilities of shape (T, num_classes).
            target_str: Candidate text sequence (e.g. 'A123BC77').
            blank_idx: CTC blank index (default 0).

        Returns:
            log_prob: Scalar log-likelihood in range (-inf, 0.0].
        """
        if not target_str:
            return float(np.sum(log_probs[:, blank_idx]))

        target_indices = [CHAR2IDX.get(c, CHAR2IDX[WILDCARD]) for c in target_str]
        L = len(target_indices)

        if torch is not None:
            try:
                # Shape log_probs: (T, C) -> (T, 1, C) for PyTorch CTCLoss
                t_log_probs = torch.from_numpy(log_probs).unsqueeze(1).float()
                t_targets = torch.tensor(target_indices, dtype=torch.long)
                input_len = torch.tensor([log_probs.shape[0]], dtype=torch.long)
                target_len = torch.tensor([L], dtype=torch.long)
                loss = F.ctc_loss(
                    t_log_probs,
                    t_targets,
                    input_len,
                    target_len,
                    blank=blank_idx,
                    reduction="none",
                    zero_infinity=True,
                )
                return float(-loss.item())
            except Exception:
                pass

        # NumPy fallback
        T, C = log_probs.shape
        L_prime = 2 * L + 1
        l_prime = np.zeros(L_prime, dtype=np.int32)
        for i, idx in enumerate(target_indices):
            l_prime[2 * i + 1] = idx

        NEG_INF = -1e9
        alpha = np.full(L_prime, NEG_INF, dtype=np.float64)
        alpha[0] = log_probs[0, blank_idx]
        alpha[1] = log_probs[0, l_prime[1]]

        for t in range(1, T):
            new_alpha = np.full(L_prime, NEG_INF, dtype=np.float64)
            for s in range(L_prime):
                tok = l_prime[s]
                p_emit = log_probs[t, tok]
                preds = [alpha[s]]
                if s > 0:
                    preds.append(alpha[s - 1])
                if s > 1 and tok != blank_idx and tok != l_prime[s - 2]:
                    preds.append(alpha[s - 2])
                m = max(preds)
                if m > -1e8:
                    log_sum = m + np.log(sum(np.exp(p - m) for p in preds))
                    new_alpha[s] = log_sum + p_emit
            alpha = new_alpha

        term = [alpha[L_prime - 1], alpha[L_prime - 2]]
        m = max(term)
        if m > -1e8:
            return float(m + np.log(sum(np.exp(p - m) for p in term)))
        return float(NEG_INF)

    @classmethod
    def score_hypotheses(
        cls,
        logits: np.ndarray,
        plate_type_prior: str = "type1",
        blank_idx: int = BLANK_IDX,
        use_beam_search: bool = True,
        beam_width: int = 5,
    ) -> Tuple[str, str, float]:
        """
        Multi-mask hypothesis scorer based on CTC log-likelihood:
        Evaluates Type 1 (L DDD LL RR), Type 2 (LL DDDD RR), and Type 1B (LL DDD RR).
        Explores top hypotheses from CTC Prefix Beam Search and greedy decoding,
        penalizes wildcard tokens, and rewards valid GOST sequences.

        Args:
            logits: Output logits of shape (T, num_classes) or (1, T, num_classes).
            plate_type_prior: Detector classification prior ('type1', 'type1a', 'type1b', 'other', 'auto').
            blank_idx: CTC blank index.
            use_beam_search: Whether to run Top-K CTC prefix beam search.
            beam_width: Number of active beams to maintain during search.

        Returns:
            (best_text, best_type, best_confidence)
        """
        if logits.ndim == 3:
            logits = logits[0]

        # Calculate normalized log-probabilities
        max_l = np.max(logits, axis=-1, keepdims=True)
        log_probs = logits - max_l - np.log(np.sum(np.exp(logits - max_l), axis=-1, keepdims=True))
        probs = np.exp(log_probs)

        best_indices = np.argmax(probs, axis=-1)
        best_probs = np.max(probs, axis=-1)
        raw_text = cls.decode_greedy(best_indices, blank_idx=blank_idx)

        # Baseline greedy confidence
        non_blank_probs = [best_probs[t] for t, idx in enumerate(best_indices) if idx != blank_idx]
        greedy_conf = float(np.mean(non_blank_probs)) if non_blank_probs else 0.0

        norm_prior = plate_type_prior.lower().strip()

        # Gather base candidate texts from greedy decoding + CTC Beam Search
        raw_texts = [raw_text]
        if use_beam_search:
            try:
                beam_results = cls.decode_beam_search(
                    log_probs,
                    beam_width=beam_width,
                    blank_idx=blank_idx,
                )
                for b_text, _ in beam_results:
                    if b_text and b_text not in raw_texts:
                        raw_texts.append(b_text)
            except Exception:
                pass

        # Syntactic FSM Beam Search candidates (Stage 2)
        try:
            fsm_results = FSMBeamSearchDecoder.decode_fsm_beam_search(
                logits,
                plate_type=norm_prior,
                beam_width=beam_width,
                blank_idx=blank_idx,
            )
            for f_text, _ in fsm_results:
                if f_text and f_text not in raw_texts:
                    raw_texts.append(f_text)
        except Exception:
            pass

        # Build candidate hypotheses across all decoded beams
        candidates = []
        for r_txt in raw_texts:
            if norm_prior == "type1a":
                cand_1a = cls.apply_gost_heuristics(r_txt, plate_type="type1")
                candidates.append(("type1a", cand_1a))
            elif norm_prior == "type1b":
                cand_1b = cls.apply_gost_heuristics(r_txt, plate_type="type1b")
                cand_1 = cls.apply_gost_heuristics(r_txt, plate_type="type1")
                candidates.append(("type1b", cand_1b))
                if cand_1 != cand_1b:
                    candidates.append(("type1", cand_1))
            elif norm_prior in ("type2", "trailer"):
                cand_2 = cls.apply_gost_heuristics(r_txt, plate_type="type2")
                candidates.append(("type2", cand_2))
                cand_1 = cls.apply_gost_heuristics(r_txt, plate_type="type1")
                if cand_1 != cand_2:
                    candidates.append(("type1", cand_1))
            else:
                # Single-line plate ('type1', 'other', 'auto')
                cand_1 = cls.apply_gost_heuristics(r_txt, plate_type="type1")
                cand_2 = cls.apply_gost_heuristics(r_txt, plate_type="type2")
                cand_1b = cls.apply_gost_heuristics(r_txt, plate_type="type1b")
                candidates.append(("type1", cand_1))
                if cand_2 != cand_1:
                    candidates.append(("type2", cand_2))
                if cand_1b not in (cand_1, cand_2):
                    candidates.append(("type1b", cand_1b))

            # Direct regex match candidate
            if PLATE_REGEX.match(r_txt):
                candidates.append(("type1", r_txt))
                if len(r_txt) == 9 and all(c in DIGITS for c in r_txt[6:]):
                    rep_reg = cls.repair_3digit_region(r_txt[6:])
                    if rep_reg != r_txt[6:]:
                        candidates.append(("type1", r_txt[:6] + rep_reg))
            if PLATE_TYPE1B_REGEX.match(r_txt):
                candidates.append(("type1b", r_txt))
                if len(r_txt) == 8 and all(c in DIGITS for c in r_txt[5:]):
                    rep_reg = cls.repair_3digit_region(r_txt[5:])
                    if rep_reg != r_txt[5:]:
                        candidates.append(("type1b", r_txt[:5] + rep_reg))
            if PLATE_TYPE2_REGEX.match(r_txt):
                candidates.append(("type2", r_txt))
                if len(r_txt) == 9 and all(c in DIGITS for c in r_txt[6:]):
                    rep_reg = cls.repair_3digit_region(r_txt[6:])
                    if rep_reg != r_txt[6:]:
                        candidates.append(("type2", r_txt[:6] + rep_reg))

            # Check for parasitic frame edge stroke (e.g. HM... or #M... -> Type 1)
            raw_stripped = cls.strip_parasitic_edge(r_txt)
            if raw_stripped != r_txt:
                cand_clean = cls.apply_gost_heuristics(raw_stripped, plate_type="type1")
                if ("type1", cand_clean) not in candidates:
                    candidates.append(("type1", cand_clean))

        # Optical confusion alternative candidate generation (K<->H, 5<->0, 6<->5, 7<->4, C<->M, etc.)
        candidate_dict: Dict[Tuple[str, str], float] = {}
        for p_type, text in candidates:
            if (p_type, text) not in candidate_dict or candidate_dict[(p_type, text)] < 0.0:
                candidate_dict[(p_type, text)] = 0.0

        optical_letter_map: Dict[str, List[Tuple[str, float]]] = {
            "T": [("M", 0.28), ("Y", 0.22)],
            "M": [("H", 0.25), ("T", 0.25), ("A", 0.15), ("C", 0.0)],
            "H": [("M", 0.25), ("B", 0.15), ("K", 0.0)],
            "Y": [("X", 0.20), ("T", 0.20), ("E", 0.0)],
            "X": [("Y", 0.20), ("K", 0.0), ("H", 0.0)],
            "C": [("O", 0.20), ("M", 0.0), ("A", 0.0)],
            "O": [("C", 0.20), ("B", 0.20), ("P", 0.15), ("A", 0.15)],
            "B": [("H", 0.20), ("P", 0.28), ("O", 0.20)],
            "P": [("B", 0.28), ("O", 0.10), ("E", 0.0)],
            "E": [("P", 0.0), ("Y", 0.0), ("B", 0.15)],
            "K": [("H", 0.0), ("X", 0.0)],
            "A": [("O", 0.15), ("M", 0.15), ("C", 0.0)],
            "#": [("B", 0.0), ("O", 0.0), ("P", 0.0), ("E", 0.0), ("T", 0.0), ("C", 0.0), ("A", 0.0), ("M", 0.0), ("H", 0.0), ("K", 0.0)],
        }

        optical_digit_map: Dict[str, List[Tuple[str, float]]] = {
            "1": [("7", 0.22), ("0", 0.15)],
            "7": [("2", 0.22), ("1", 0.20), ("3", 0.20), ("4", 0.25), ("9", 0.15), ("0", 0.15)],
            "3": [("7", 0.20), ("5", 0.25), ("8", 0.0), ("2", 0.10)],
            "5": [("7", 0.18), ("0", 0.28), ("3", 0.32), ("6", 0.0), ("9", 0.0)],
            "8": [("0", 0.28), ("7", 0.15), ("6", 0.0), ("3", 0.0)],
            "0": [("8", 0.15), ("5", 0.22), ("7", 0.15), ("9", 0.0)],
            "9": [("7", 0.20), ("0", 0.28), ("4", 0.20), ("2", 0.0), ("5", 0.0)],
            "2": [("7", 0.22), ("4", 0.18), ("9", 0.0), ("3", 0.10)],
            "4": [("2", 0.18), ("7", 0.25), ("9", 0.20)],
            "6": [("5", 0.0), ("8", 0.0)],
        }

        for p_type, text in list(candidate_dict.keys()):
            # 1. Letter optical confusion
            let_positions = (0, 4, 5) if p_type in ("type1", "type1a") else (0, 1)
            for pos in let_positions:
                if pos < len(text):
                    c = text[pos]
                    for alt_c, b_val in optical_letter_map.get(c, []):
                        b_actual = b_val if pos == 0 else (b_val * 0.85)
                        alt_cand = text[:pos] + alt_c + text[pos + 1 :]
                        if alt_cand != text:
                            key = (p_type, alt_cand)
                            if key not in candidate_dict or b_actual > candidate_dict[key]:
                                candidate_dict[key] = b_actual

            # 2. Region optical confusion
            if len(text) >= 8:
                reg_start = 6 if p_type in ("type1", "type1a", "type2") else 5
                for rpos in range(reg_start, len(text)):
                    rc = text[rpos]
                    for alt_rc, b_val in optical_digit_map.get(rc, []):
                        alt_text = text[:rpos] + alt_rc + text[rpos + 1 :]
                        if len(alt_text) == 9 and p_type != "type1b":
                            raw_reg = alt_text[6:]
                            repaired_reg = cls.repair_3digit_region(raw_reg)
                            alt_text = alt_text[:6] + repaired_reg
                        if alt_text != text:
                            key = (p_type, alt_text)
                            if key not in candidate_dict or b_val > candidate_dict[key]:
                                candidate_dict[key] = b_val

        best_score = -1e9
        best_cand_text = raw_text
        best_cand_type = norm_prior if norm_prior in ("type1", "type1a", "type1b", "type2") else "type1"

        for (p_type, text), bonus in candidate_dict.items():
            lp = cls.compute_ctc_log_prob(log_probs, text, blank_idx=blank_idx)
            norm_lp = lp / max(1, len(text))
            wildcards = text.count("#")
            score = norm_lp - (1.0 * wildcards) + bonus

            # Prior preference for detector class if not contradicted
            if p_type == norm_prior:
                score += 0.5
            elif norm_prior in ("type1", "type1a") and p_type in ("type2", "type1b"):
                # Strengthen penalty for selecting non-Type 1 over detector prior Type 1/1A
                score -= 0.8

            # Bonus for complete valid Russian plate regex without wildcard
            if wildcards == 0:
                is_valid_reg = False
                reg_code = ""
                if len(text) == 9 and p_type in ("type1", "type1a", "type2"):
                    reg_code = text[6:]
                    is_valid_reg = reg_code in VALID_3DIGIT_REGIONS
                elif len(text) == 8 and p_type in ("type1", "type1a", "type2"):
                    reg_code = text[6:]
                    is_valid_reg = reg_code in VALID_2DIGIT_REGIONS
                elif len(text) == 8 and p_type == "type1b":
                    reg_code = text[5:]
                    is_valid_reg = reg_code in VALID_3DIGIT_REGIONS
                elif len(text) == 7 and p_type == "type1b":
                    reg_code = text[5:]
                    is_valid_reg = reg_code in VALID_2DIGIT_REGIONS

                if is_valid_reg:
                    reg_prior = REGION_FREQUENCY_WEIGHTS.get(reg_code, 0.0)
                    if p_type in ("type1", "type1a") and PLATE_REGEX.match(text):
                        score += 0.5 + reg_prior
                    elif p_type == "type1b" and PLATE_TYPE1B_REGEX.match(text):
                        score += 0.5 + reg_prior
                    elif p_type == "type2" and PLATE_TYPE2_REGEX.match(text):
                        score += 0.5 + reg_prior
                else:
                    score -= 5.0

            if score > best_score:
                best_score = score
                best_cand_text = text
                best_cand_type = p_type

        # Confidence: combine greedy confidence with wildcard penalty
        wildcards = best_cand_text.count("#")
        confidence = float(greedy_conf * max(0.0, 1.0 - 0.15 * wildcards))
        confidence = max(0.0, min(1.0, confidence))

        # Fallback Guard: Prevent false Type 2 / Type 1B switching on noisy, blurred or dark crops
        # If best candidate is not Type 1, but wildcards dominate (>=3) or confidence is low (< 0.35),
        # forcibly prioritize the standard passenger car format (Type 1).
        if best_cand_type != "type1" and norm_prior in ("type1", "type1a", "other", "auto") and (wildcards >= 3 or confidence < 0.35):
            type1_cands = [c for c in candidates if c[0] == "type1"]
            if type1_cands:
                best_t1_text = type1_cands[0][1]
                best_t1_score = -1e9
                for _, t1_text in type1_cands:
                    lp = cls.compute_ctc_log_prob(log_probs, t1_text, blank_idx=blank_idx)
                    norm_lp = lp / max(1, len(t1_text))
                    wc = t1_text.count("#")
                    t1_sc = norm_lp - (1.0 * wc)
                    if t1_sc > best_t1_score:
                        best_t1_score = t1_sc
                        best_t1_text = t1_text
                best_cand_text = best_t1_text
                best_cand_type = "type1"
            else:
                best_cand_text = cls.apply_gost_heuristics(raw_text, plate_type="type1")
                best_cand_type = "type1"

            wildcards = best_cand_text.count("#")
            confidence = float(greedy_conf * max(0.0, 1.0 - 0.15 * wildcards))
            confidence = max(0.0, min(1.0, confidence))

        return best_cand_text, best_cand_type, confidence


