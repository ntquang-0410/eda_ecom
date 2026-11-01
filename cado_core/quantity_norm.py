#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn hoá đại lượng (số + đơn vị) cho cặp VI ↔ ZH.

Mục tiêu: so sánh **cùng một đại lượng vật lý** giữa hai ngôn ngữ, thay vì so
chuỗi số thô. Ví dụ:

    144,5 cm  ↔  144.5 厘米      -> khớp (dấu thập phân khác nhau)
    1.234 kg  ↔  1234 克         -> khớp (dấu ngăn nghìn + đổi đơn vị)
    10 gói x 16g ↔ 10包×16克     -> khớp (2 đại lượng: count + mass)
    3-5 cm    ↔  3-5厘米         -> khớp (range, đơn vị chỉ ghi 1 lần)

API:
    extract_quantities(text, lang) -> list[Quantity]
    signature(quantities)          -> list[(dim, base_value)]
    compare(src_text, tgt_text)    -> dict(match, missing, extra, ...)

Đơn vị được quy về hệ SI cơ sở: m, kg, l, m², W, V, A, Ah, s, byte, count...
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Iterable

# ---------------------------------------------------------------------------
# 1. Bảng đơn vị: (tên, thứ nguyên, hệ số về đơn vị cơ sở)
# ---------------------------------------------------------------------------

UNITS: dict[str, tuple[str, float]] = {}


def _add(dim: str, factor: float, *names: str) -> None:
    for name in names:
        UNITS[name.lower()] = (dim, factor)


# độ dài -> m
_add("length", 1.0, "m", "met", "mét", "meter", "米")
_add("length", 0.01, "cm", "centimet", "厘米", "公分")
_add("length", 0.001, "mm", "milimet", "毫米")
_add("length", 1000.0, "km", "kilomet", "公里", "千米")
_add("length", 0.0254, "inch", "in", "英寸", "寸")
_add("length", 0.3048, "ft", "foot", "feet", "英尺")
_add("length", 0.9144, "yard", "yd", "码")
_add("length", 0.333333, "尺")
_add("length", 500.0, "里")

# khối lượng -> kg
_add("mass", 1.0, "kg", "kilogram", "kilogam", "公斤", "千克")
_add("mass", 0.001, "g", "gram", "gam", "gr", "克")
_add("mass", 1e-6, "mg", "miligram", "毫克")
_add("mass", 1000.0, "t", "tan", "tấn", "吨")
_add("mass", 0.5, "斤")
_add("mass", 0.05, "两", "lạng")
_add("mass", 0.453592, "lb", "lbs", "pound", "磅")
_add("mass", 0.0283495, "oz", "ounce", "盎司")

# thể tích -> l
_add("volume", 1.0, "l", "lit", "lít", "liter", "升")
_add("volume", 0.001, "ml", "mililit", "毫升", "cc")
_add("volume", 1000.0, "m3", "m³", "立方米")
_add("volume", 3.78541, "gallon", "gal", "加仑")

# diện tích -> m²
_add("area", 1.0, "m2", "m²", "平方米")
_add("area", 1e-4, "cm2", "cm²", "平方厘米")
_add("area", 666.667, "亩")

# công suất -> W
_add("power", 1.0, "w", "watt", "瓦", "瓦特")
_add("power", 1000.0, "kw", "千瓦")
_add("power", 745.7, "hp", "horsepower", "马力")

# điện áp -> V
_add("voltage", 1.0, "v", "volt", "伏", "伏特")
_add("voltage", 1000.0, "kv", "千伏")

# dòng điện -> A
_add("current", 1.0, "a", "amp", "ampere", "安", "安培")
_add("current", 0.001, "ma", "毫安")

# dung lượng pin -> Ah
_add("capacity", 1.0, "ah", "安时")
_add("capacity", 0.001, "mah", "毫安时")

# nhiệt độ -> °C
_add("temperature", 1.0, "°c", "c", "độ c", "摄氏度", "℃")

# thời gian -> s
_add("time", 1.0, "s", "giây", "秒")
_add("time", 60.0, "min", "phút", "分钟")
_add("time", 3600.0, "h", "giờ", "小时", "时")
_add("time", 86400.0, "ngày", "天", "日")
_add("time", 2592000.0, "tháng", "个月", "月")
_add("time", 31536000.0, "năm", "年")

# dữ liệu -> byte
_add("data", 1.0, "b", "byte")
_add("data", 1e3, "kb")
_add("data", 1e6, "mb")
_add("data", 1e9, "gb")
_add("data", 1e12, "tb")

# tỉ lệ
_add("percent", 1.0, "%", "％", "phần trăm", "百分之")

# đếm (counter) — giữ thứ nguyên riêng, hệ số 1
_add("count", 1.0,
     "cái", "chiếc", "con", "bộ", "đôi", "cuộn", "chai", "lọ", "viên", "gói",
     "hộp", "túi", "miếng", "thanh", "ống", "nắp", "ổ", "ly", "cốc", "bát",
     "đĩa", "đôi chiếc", "bịch", "thùng", "combo",
     "个", "件", "只", "双", "套", "包", "盒", "袋", "片", "支", "台", "张",
     "条", "瓶", "罐", "箱", "组", "枚", "粒", "块", "本", "副", "对")

# tiền tệ: không quy đổi, chỉ so khớp khi cùng đơn vị
_add("currency", 1.0, "đ", "vnd", "vnđ", "元", "¥", "rmb", "人民币", "美元", "usd")

_UNIT_NAMES = sorted(UNITS, key=len, reverse=True)

VI_NUMBERS = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "sáu": 6,
              "bảy": 7, "tám": 8, "chín": 9, "mười": 10}
VI_NUM_RE = re.compile(r"\b(" + "|".join(VI_NUMBERS) + r")\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 2. Số: Ả Rập (locale) + chữ Hán
# ---------------------------------------------------------------------------

ARABIC_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
CN_NUM_RE = re.compile(r"[零〇一二两三四五六七八九十百千万亿半]+")


def cn_to_number(text: str) -> float | None:
    """Chuyển số Hán đơn giản: 三百 -> 300, 十五 -> 15, 两 -> 2, 半 -> 0.5."""
    if not text:
        return None
    if text == "半":
        return 0.5
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
        elif char in CN_UNITS:
            unit = CN_UNITS[char]
            if unit >= 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
            number = 0
        elif char == "半":
            section += 0.5
        else:
            return None
    return float(total + section + number)


def parse_number(token: str, lang: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    if lang == "zh" and CN_NUM_RE.fullmatch(token):
        return cn_to_number(token)
    # chuẩn hoá khoảng trắng/NBSP trong số
    token = token.replace("\u00a0", "").replace(" ", "")
    has_comma = "," in token
    has_dot = "." in token
    if has_comma and has_dot:
        # dạng nào đứng sau cùng là dấu thập phân
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif has_comma:
        parts = token.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            # 1,234 -> nghìn (trừ khi có đơn vị nhỏ, xử lý ở compare)
            token = token.replace(",", "")
        else:
            token = token.replace(",", ".")
    elif has_dot:
        parts = token.split(".")
        if len(parts) > 2:
            token = token.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            # 1.234 -> có thể là nghìn hoặc 1,234 thập phân
            token = token  # giữ nguyên, compare sẽ thử cả 2
    try:
        return float(token)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 3. Trích đại lượng
# ---------------------------------------------------------------------------

@dataclass
class Quantity:
    raw: str
    value: float
    unit: str
    dimension: str
    base_value: float
    start: int
    end: int


def _match_unit(text: str, pos: int) -> tuple[str, int]:
    """Khớp đơn vị dài nhất tại vị trí pos (bỏ qua khoảng trắng đầu)."""
    while pos < len(text) and text[pos] in " \t\n\r\u00a0":
        pos += 1
    lowered = text[pos:].lower()
    for name in _UNIT_NAMES:
        if lowered.startswith(name):
            # tránh khớp "a" trong "apple": đơn vị 1 ký tự phải đứng riêng
            after = pos + len(name)
            if len(name) == 1 and after < len(text):
                nxt = text[after]
                if nxt.isalpha() and not ("\u4e00" <= nxt <= "\u9fff"):
                    continue
            # "尺寸" là từ, không phải đơn vị "尺"
            if name == "尺" and text[pos + len(name):pos + len(name) + 1] == "寸":
                continue
            return name, after
    return "", pos


FRACTIONS = {"½": "0.5", "¼": "0.25", "¾": "0.75", "⅓": "0.3333", "⅔": "0.6667"}
RANGE_SEPARATORS = {"-", "–", "—", "~", "～", "至", "到", "x", "X", "×", "*"}


def _normalize_fractions(text: str) -> str:
    text = text.replace("⁄", "/").replace("／", "/")
    for symbol, value in FRACTIONS.items():
        text = text.replace(symbol, value)
    return re.sub(r"(\d+)\s*/\s*(\d+)",
                  lambda m: f"{float(m.group(1)) / float(m.group(2)):.4f}".rstrip("0").rstrip(".")
                  if float(m.group(2)) else m.group(0),
                  text)


def _propagate_range_units(text: str, quantities: list[Quantity]) -> None:
    """Đơn vị ghi một lần cho cả dải: '3-5cm', '3x5cm', '3cm-5cm'."""
    for left, right in zip(quantities, quantities[1:]):
        separator = text[left.end:right.start].strip()
        if separator not in RANGE_SEPARATORS:
            continue
        if left.dimension and not right.dimension:
            right.unit = left.unit
            right.dimension = left.dimension
            right.base_value = right.value * UNITS.get(left.unit, ("", 1.0))[1]
        elif right.dimension and not left.dimension:
            left.unit = right.unit
            left.dimension = right.dimension
            left.base_value = left.value * UNITS.get(right.unit, ("", 1.0))[1]


CODE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9./+\-]*$")
_UNIT_ALT = "(?:" + "|".join(re.escape(name) for name in _UNIT_NAMES) + ")"
_NUM = r"\d+(?:[.,]\d+)?"
_PAIR = rf"{_NUM}{_UNIT_ALT}?"
_NUMBER_UNIT = re.compile(
    rf"^{_PAIR}$|^(?:{_PAIR})(?:[-~/x*×,](?:{_PAIR}))+$",
    re.IGNORECASE,
)


def _in_code_token(text: str, start: int, end: int) -> bool:
    """Số nằm trong token chữ+số (mã sản phẩm) nhưng không phải '20W'."""
    left = start
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace():
        right += 1
    token = text[left:right]
    if not CODE_TOKEN.match(token):
        return False
    if _NUMBER_UNIT.match(token):
        return False
    return any(ch.isalpha() for ch in token)


CURRENCY_VI = re.compile(r"nhân\s+dân\s+tệ|\btệ\b|\bđồng\b", re.IGNORECASE)


def extract_quantities(text: str, lang: str = "vi") -> list[Quantity]:
    text = unicodedata.normalize("NFKC", str(text or ""))
    if lang == "vi":
        text = CURRENCY_VI.sub("元", text)
    text = _normalize_fractions(text)
    found: list[Quantity] = []

    # số Ả Rập
    for match in ARABIC_RE.finditer(text):
        # "E300", "ABF916", "HAP739": số nằm trong mã sản phẩm -> bỏ
        value = parse_number(match.group(), lang)
        if value is None:
            continue
        unit, end = _match_unit(text, match.end())
        if _in_code_token(text, match.start(), end):
            continue
        if unit == "件":
            tail = text[end:end + 2]
            if tail.startswith("套") or tail.startswith("装") or tail.startswith("代发"):
                continue
        found.append(Quantity(match.group(), value, unit,
                              UNITS.get(unit, ("", 1.0))[0],
                              value * UNITS.get(unit, ("", 1.0))[1],
                              match.start(), end))

    # số Hán (chỉ khi lang=zh)
    if lang == "zh":
        for match in CN_NUM_RE.finditer(text):
            # bỏ qua nếu nằm trong số Ả Rập đã bắt
            if any(q.start <= match.start() < q.end for q in found):
                continue
            # "千克"/"千卡" là đơn vị, không phải số 1000
            head = text[match.start():match.start() + 2].lower()
            if head in UNITS:
                continue
            value = cn_to_number(match.group())
            if value is None:
                continue
            unit, end = _match_unit(text, match.end())
            if not unit:
                # số Hán đứng một mình thường là từ ("一体", "一款") -> bỏ
                continue
            if unit == "件" and text[end:end + 1] in ("套", "装"):
                continue
            found.append(Quantity(match.group(), value, unit,
                                  UNITS.get(unit, ("", 1.0))[0],
                                  value * UNITS.get(unit, ("", 1.0))[1],
                                  match.start(), end))

    if False:  # tắt: "ba lô", "một" gây false positive
        for match in VI_NUM_RE.finditer(text):
            if any(q.start <= match.start() < q.end for q in found):
                continue
            value = VI_NUMBERS.get(match.group().lower())
            if value is None:
                continue
            unit, end = _match_unit(text, match.end())
            found.append(Quantity(match.group(), float(value), unit,
                                  UNITS.get(unit, ("", 1.0))[0],
                                  value * UNITS.get(unit, ("", 1.0))[1],
                                  match.start(), end))

    found.sort(key=lambda q: q.start)
    _propagate_range_units(text, found)
    return found


def signature(quantities: Iterable[Quantity], round_digits: int = 4) -> list[tuple[str, float]]:
    """Chuỗi đại lượng đã quy đổi; bỏ đơn vị đếm khỏi so khớp vật lý."""
    result = []
    for q in quantities:
        if not q.dimension:
            result.append(("bare", float(round(q.value, round_digits))))
            continue
        if q.dimension == "count":
            result.append(("count", float(round(q.value))))
            continue
        if q.unit in ("年", "năm") and 1900 <= q.value <= 2100:
            result.append(("year", float(round(q.value))))
            continue
        result.append((q.dimension, round(q.base_value, round_digits)))
    return sorted(result)


# ---------------------------------------------------------------------------
# 4. So khớp
# ---------------------------------------------------------------------------

def _close(a: float, b: float, rel: float = 0.02) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= rel


WEAK_DIMS = {"bare", "count", "year"}


def compare(src_text: str, tgt_text: str,
            src_lang: str = "vi", tgt_lang: str = "zh",
            strict: bool = True) -> dict:
    src = signature(extract_quantities(src_text, src_lang))
    tgt = signature(extract_quantities(tgt_text, tgt_lang))
    if strict:
        # chỉ so đại lượng vật lý; bỏ số đếm/năm/bare vì dễ nhiễu
        src = [item for item in src if item[0] not in WEAK_DIMS]
        tgt = [item for item in tgt if item[0] not in WEAK_DIMS]

    missing = list(src)
    extra = []
    for item in tgt:
        dim, value = item
        for index, candidate in enumerate(missing):
            if candidate[0] == dim and _close(candidate[1], value):
                missing.pop(index)
                break
        else:
            extra.append(item)

    # vòng 2: cho phép "bare" khớp với đơn vị khác cùng giá trị
    still_missing = []
    for item in missing:
        dim, value = item
        if dim not in ("bare", "count", "year"):
            still_missing.append(item)
            continue
        for index, candidate in enumerate(extra):
            if candidate[0] in ("bare", "count", "year") and _close(candidate[1], value):
                extra.pop(index)
                break
        else:
            still_missing.append(item)
    missing = still_missing

    src_dims = {dim for dim, _ in src}
    tgt_dims = {dim for dim, _ in tgt}
    # strong: cùng thứ nguyên xuất hiện cả hai bên nhưng giá trị không khớp
    strong = any(dim in tgt_dims for dim, _ in missing) or \
             any(dim in src_dims for dim, _ in extra)

    return {
        "src": src,
        "tgt": tgt,
        "missing": missing,
        "extra": extra,
        "match": not missing and not extra,
        "strong_mismatch": bool(strong and (missing or extra)),
    }
