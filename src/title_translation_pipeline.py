"""Legacy TranslateGemma/Qwen3 runner; use title_translation_hybrid.py for current runs.

The hybrid runner imports this module for its dataset validation, deterministic
sampling, technical QA, and comparison-key helpers. Its legacy model runner is
kept for provenance and should not be used for Gemma 4 + Qwen3.5 execution.
"""

from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[1]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DATASET_ID = "ntquang0410/zh-vie_ecom"
DATASET_REVISION = "e098c4dbd40a5ab1b4f2670f7d4b028aa4e09f88"
DATASET_FILE = "data/cleaned/bilingual_zh_vi_cado.parquet"
DATASET_SHA256 = "d5095f50f09d539140822b5bd41d2fba517a199ea5ba4f0204ccf50ac2cd0559"
EXPECTED_ROWS = 16_348
EXPECTED_UNIQUE_TITLES = 15_863
SEED = 42
SAMPLE_VERSION = "cado-title-sampling-v1"
QA_VERSION = "title-technical-qa-v13-top-ten-and-combined-one-equivalences"
SELECTOR_VERSION = "baseline-first-agreeing-models-v1"
PROMPT_VERSION = "zh-vi-product-title-v1"
SHARD_SIZE = 500
FULL_TITLES_PER_SESSION = 1000
GENERATION_BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
TRUNCATION_RETRY_TOKENS = 512
MAX_SINGLETON_RETRIES = 3
PILOT_MIN_COMPLETION_RATE = 0.99
PILOT_MIN_REPRESENTATIVE_QA_RATE = 0.95
FULL_WINDOW_SIZE = 200
FULL_MAX_INCOMPLETE_RATE = 0.01
FULL_MAX_QA_FAILURE_RATE = 0.05

MODEL_SPECS = {
    "translategemma": {
        "model_id": "google/translategemma-12b-it",
        "template_version": "translategemma-official-structured-content-v1",
        "device_preference": 0,
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-14B",
        "template_version": "qwen3-chat-template-no-thinking-v1",
        "device_preference": 1,
    },
}
BGE_MODEL_ID = "BAAI/bge-m3"
BGE_TEMPLATE_VERSION = "bge-m3-cls-normalized-v1"

ORIGINAL_COLUMNS = [
    "product_id",
    "title_zh",
    "title_vi",
    "description_zh",
    "description_vi",
    "category",
    "source_site",
    "url",
    "crawl_time",
    "worker",
    "keyword",
    "page",
    "search_url",
    "price",
    "sales",
    "province",
    "city",
    "biz_type",
    "shop",
    "is_ad",
    "category_1688",
    "category_id_1688",
    "has_vi",
    "attributes_zh",
    "attributes_vi",
    "n_attributes_zh",
    "n_attributes_vi",
    "n_attributes_aligned",
    "description_extra_zh",
    "description_images",
    "title_vi_cleaned",
    "desc_vi_cleaned",
]
OUTPUT_COLUMNS = [
    "title_vi_candidate_translategemma",
    "title_vi_candidate_qwen3",
    "title_vi_selected",
    "title_vi_selection_source",
    "title_vi_selection_decision",
    "title_vi_qa_status",
    "title_vi_qa_flags",
    "title_vi_candidate_agreement",
    "title_vi_chrf",
    "title_vi_bge_m3_cosine",
    "title_vi_semantic_verified",
    "title_vi_pipeline_config_id",
]

NUMBER_RE = re.compile(r"(?<!\d)\d+(?:[.,]\d+)*(?:%|％)?")
CHINESE_NUMERAL_PATTERN = "零〇一二两兩三四五六七八九十百千万萬亿億半双雙"
CHINESE_NUMERAL_RE = re.compile(rf"[{CHINESE_NUMERAL_PATTERN}]+")
CODE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*(?![A-Za-z0-9])")
CURRENCY_RE = re.compile(
    r"(?:[¥￥]\s*|(?i:RMB)\s*)(\d+(?:[.,]\d+)*)"
    # Chinese units may be adjacent to Han text; Latin currency names and đ
    # must have a word boundary so Vietnamese words like 'đồ' are not prices.
    r"|(\d+(?:[.,]\d+)*)\s*(?:元|人民币)"
    r"|(\d+(?:[.,]\d+)*)\s*(?i:RMB|VND|yuan|tệ)(?!\w)"
    r"|(\d+(?:[.,]\d+)*)\s*(?:đ|₫)(?!\w)"
)
JIN_RE = re.compile(
    rf"(?:(?P<amount>\d+(?:[.,]\d+)?|[{CHINESE_NUMERAL_PATTERN}]+)\s*斤(?P<extra_half>半)?|(?P<half_only>半)\s*斤)"
)
MASS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|公斤|千克|克|grams|gram|g|cân)(?=$|[^A-Za-z0-9_]|\d)", re.IGNORECASE)
CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)
CYRILLIC_RANGES = ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))
CHINESE_COUNT_UNITS = "个件款只台套支双枚片袋盒瓶杯斤克公斤千克米厘米毫米寸英寸号小时日天年月季层座合色档位字镜路录錄次代对份根辆级声趟轮星面肩头人排鞋袜筷节節岁元子格针宫本用方芯类型"
VIETNAMESE_NUMBER_WORDS = {
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
}
VIETNAMESE_NUMBER_RE = re.compile(
    r"\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\b",
    re.IGNORECASE,
)
VIETNAMESE_DOUBLE_RE = re.compile(r"\b(?:kép|double)\b", re.IGNORECASE)
VIETNAMESE_COUNT_MARKERS = (
    "chỗ", "trong", "mảnh", "miếng", "chiếc", "cái", "bộ", "màu", "ngăn",
    "lớp", "tầng", "sao", "số", "đầu", "ống kính", "camera", "lần", "bánh",
    "kênh", "mức", "nấc", "cổng", "phần", "mục", "loại", "kiểu", "cấp",
    "kg", "gram", "ml", "cm", "mm", "inch", "giờ", "ngày", "tháng", "mùa",
    "đôi", "cặp", "hướng", "chiều", "chức năng", "tác dụng", "dây", "thanh",
    "vòng", "điểm", "cánh", "cửa", "sợi", "tấm", "viên", "hộp", "set", "quai",
    "mặt", "vai", "món", "cân", "tuổi", "thế hệ", "công dụng", "lõi", "thứ",
)
EXPLANATION_RE = re.compile(
    r"(?i)^\s*(?:translation\s*:|here(?:'s| is) the translation\s*:|"
    r"bản dịch\s*:|tiêu đề dịch\s*:|dịch sang tiếng việt\s*:)"
)
LATIN_WORD_RE = re.compile(
    r"(?<![A-Za-zÀ-ỹ0-9])[A-Za-z][A-Za-z0-9+._/-]*(?![A-Za-zÀ-ỹ0-9])"
)
LATIN_ADVISORY_STOPWORDS = {
    "ao", "be", "bo", "cho", "co", "da", "day", "de", "den", "do",
    "dung", "gia", "hai", "kem", "khong", "la", "loai", "ma", "mau",
    "nam", "nu", "set", "tac", "thanh", "tre", "tui", "tu", "va", "xe",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
        temp_path = Path(stream.name)
    os.replace(temp_path, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temp_path = Path(stream.name)
    os.replace(temp_path, path)


def has_codepoint_in_ranges(value: Any, ranges: tuple[tuple[int, int], ...]) -> bool:
    text = "" if value is None else str(value)
    return any(start <= ord(char) <= end for char in text for start, end in ranges)


def contains_cjk(value: Any) -> bool:
    return has_codepoint_in_ranges(value, CJK_RANGES)


def contains_cyrillic(value: Any) -> bool:
    return has_codepoint_in_ranges(value, CYRILLIC_RANGES)


def normalize_title_key(value: Any) -> str:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return re.sub(r"""[\s.!?,，。！？；;:：、]+$""", "", text).strip()


def _number_value(value: str) -> float:
    vietnamese = VIETNAMESE_NUMBER_WORDS.get(value.casefold())
    if vietnamese is not None:
        return float(vietnamese)
    chinese_values = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10000,
        "萬": 10000,
        "半": 0.5,
        "双": 2,
        "雙": 2,
    }
    chinese_values.update({"亿": 100_000_000, "億": 100_000_000})
    if value and all(char in chinese_values for char in value):
        if value == "半":
            return 0.5
        # 双/雙 is a classifier meaning a pair in phrases such as 一双鞋;
        # it contributes no extra digit to the count (一双 = one pair).
        if len(value) > 1 and value[-1] in "双雙" and all(
            char in "零〇一二两兩三四五六七八九十百千万萬" for char in value[:-1]
        ):
            return _number_value(value[:-1])
        if not any(char in value for char in "十百千万萬亿億"):
            return float("".join(str(chinese_values[char]) for char in value))
        total = 0.0
        section = 0.0
        digit = 0.0
        for char in value:
            number = chinese_values[char]
            if char in "零〇":
                digit = 0.0
            elif char in "一二两兩三四五六七八九双雙":
                digit = number
            elif char in "十百千":
                section += (digit if digit else 1.0) * number
                digit = 0.0
            elif char in "万萬亿億":
                section += digit
                multiplier = 10_000 if char in "万萬" else 100_000_000
                total += (section if section else 1.0) * multiplier
                section = 0.0
                digit = 0.0
            elif char == "半":
                digit += 0.5
        return total + section + digit
    # Titles in this source use decimal comma or decimal point; when both occur,
    # commas are interpreted as thousands separators.
    normalized = value.replace("％", "%").rstrip("%")
    if normalized.count('.') >= 2 and re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}(?:,\d{1,2})?", normalized):
        # Unambiguous repeated thousands groups in Vietnamese prices.
        # Keep the existing interpretation for single decimal points.
        normalized = normalized.replace('.', '').replace(',', '.')
    elif "," in normalized and "." in normalized:
        normalized = normalized.replace(",", "")
    elif "," in normalized:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", normalized):
            normalized = normalized.replace(",", "")
        else:
            normalized = normalized.replace(",", ".")
    return float(normalized)


def _number_tokens(text: str) -> list[tuple[str, int, int]]:
    code_spans = [
        (match.start(), match.end())
        for match in CODE_RE.finditer(text)
        if _is_contextual_model_code(text, match)
    ]
    tokens = []
    storage_spans: list[tuple[int, int]] = []
    voltage_spans: list[tuple[int, int]] = []
    resolution_spans: list[tuple[int, int]] = []
    if re.search(r"优盘|U\s*盘|USB|内存|存储|容量", text, re.IGNORECASE):
        # Compact supplier notation such as 64128g means 64 GB + 128 GB.
        # Restrict the split to common storage sizes and an explicit storage context.
        for match in re.finditer(
            r"(?<!\d)(?P<first>64|128|256|512)(?P<second>64|128|256|512)\s*(?:gb|g)(?![A-Za-z])",
            text,
            re.IGNORECASE,
        ):
            tokens.extend(
                (
                    (match.group("first"), match.start("first"), match.end("first")),
                    (match.group("second"), match.start("second"), match.end("second")),
                )
            )
            storage_spans.append(match.span())
    # Voltage values are numeric specifications even when written compactly
    # as model-like tokens such as 12V/24V or with the Chinese unit 伏.
    for match in re.finditer(
        r"(?<![A-Za-z0-9])(?P<number>\d+(?:[.,]\d+)?)\s*(?:V(?![A-Za-z0-9])|伏(?:特)?(?![A-Za-z]))",
        text,
        re.IGNORECASE,
    ):
        tokens.append((match.group("number"), match.start("number"), match.end("number")))
        voltage_spans.append(match.span())
    # Camera/display resolutions ending in P are numbers, not alphanumeric
    # product codes; keep them comparable across 1296 vs. 1296P.
    for match in re.finditer(
        # A Unicode word-boundary after P does not match when the next
        # character is Han (both are treated as word characters by Python).
        r"(?<!\d)(?P<number>480|720|1080|1296|1440|2160)\s*[pP](?![A-Za-z0-9])",
        text,
        re.IGNORECASE,
    ):
        tokens.append((match.group("number"), match.start("number"), match.end("number")))
        resolution_spans.append(match.span())
    # Arabic digits followed by Chinese large units represent one scaled amount;
    # do not compare the digit and multiplier as separate numbers.
    scaled_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?P<unit>万|萬|亿|億)", text):
        try:
            multiplier = 10_000 if match.group("unit") in {"万", "萬"} else 100_000_000
            scaled = _number_value(match.group(1)) * multiplier
        except ValueError:
            continue
        normalized = str(int(scaled)) if float(scaled).is_integer() else str(scaled)
        tokens.append((normalized, match.start(), match.end()))
        scaled_spans.append((match.start(), match.end()))
    # Vietnamese frequently spells out large number units. Normalize them to
    # the same absolute value as Chinese 亿, e.g. 九亿 == 900 triệu.
    for match in re.finditer(
        r"(?<!\d)(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>triệu|tỷ|tỉ)(?!\w)",
        text,
        re.IGNORECASE,
    ):
        try:
            multiplier = 1_000_000 if match.group("unit").casefold() == "triệu" else 1_000_000_000
            scaled = _number_value(match.group("number")) * multiplier
        except ValueError:
            continue
        normalized = str(int(scaled)) if float(scaled).is_integer() else str(scaled)
        tokens.append((normalized, match.start(), match.end()))
        scaled_spans.append((match.start(), match.end()))
    for match in NUMBER_RE.finditer(text):
        start, end = match.start(), match.end()
        if any(span_start <= start and end <= span_end for span_start, span_end in storage_spans):
            continue
        if any(span_start <= start and end <= span_end for span_start, span_end in voltage_spans):
            continue
        if any(span_start <= start and end <= span_end for span_start, span_end in resolution_spans):
            continue
        if any(span_start <= start and end <= span_end for span_start, span_end in scaled_spans):
            continue
        after = text[end : end + 16]
        quantity_suffix = re.match(
            r"(?i)\s*(?:mg|kg|gb|g(?=\s*/)|ml|l(?=\s*/)|mah|克|公斤|千克|斤|毫安|厘米|毫米|cm|mm|寸|英寸|inch|in|m(?=$|[\s.,;:!?，。；：！？])|分钟|小时|hr|hours?)(?=$|[^A-Za-z0-9])",
            after,
        )
        range_quantity_suffix = re.match(
            r"(?i)\s*[-~至]\s*\d+(?:[.,]\d+)?\s*(?:mg|kg|g(?=\s*/)|ml|l(?=\s*/)|mah|克|公斤|千克|斤|毫安|厘米|毫米|cm|mm|寸|英寸|inch|in|m(?=$|[\s.,;:!?，。；：！？])|分钟)(?=$|[^A-Za-z0-9])",
            after,
        )
        covering_codes = [
            text[code_start:code_end]
            for code_start, code_end in code_spans
            if code_start <= start and end <= code_end
        ]
        slash_measurement = any(
            "/" in code
            and re.fullmatch(
                r"(?i)\d+(?:[.,]\d+)?(?:mg|kg|g|ml|l|mah|克|公斤|千克|斤|毫安|cm|mm|寸|英寸)(?:/\d+(?:[.,]\d+)?(?:mg|kg|g|ml|l|mah|克|公斤|千克|斤|毫安|cm|mm|寸|英寸))+",
                code,
            )
            for code in covering_codes
        )
        if covering_codes and not (quantity_suffix or range_quantity_suffix or slash_measurement):
            continue
        tokens.append((match.group(0), start, end))
    for match in CHINESE_NUMERAL_RE.finditer(text):
        start, end = match.start(), match.end()
        if any(span_start <= start and end <= span_end for span_start, span_end in scaled_spans):
            continue
        before = text[max(0, start - 2) : start]
        after = text[end : end + 3]
        if match.group(0) == "百" and after.startswith("合"):
            continue  # 百合 is the flower name, not a numeral plus 合.
        if match.group(0) == "零" and after.startswith("件"):
            continue  # 零件 means spare parts/components.
        if match.group(0) == "一" and after.startswith("件代发"):
            continue  # 一件代发 is the dropshipping term, not a one-item quantity.
        if match.group(0) == "三" and text[max(0, start - 2) : end + 3].find("三只松鼠") >= 0:
            continue  # 三只松鼠 is the Three Squirrels brand name.
        raw = match.group(0)
        if raw == "一" and after.startswith("字领"):
            continue  # 一字领 is a neckline term, not a quantity.
        if (
            (after and after[0] in CHINESE_COUNT_UNITS)
            or (raw == "十" and after.startswith("大"))
            or before.endswith(("第", "合"))
            or raw.endswith(("双", "雙"))
            or "亿" in raw
            or "億" in raw
            or (raw == "四" and after.startswith("拉杆") and "套三套四" in text)
        ):
            if len(raw) > 1 and all(char in "零〇一二两兩三四五六七八九" for char in raw):
                # Compact alternatives such as 二三代 and 四五六代 mean
                # generations 2/3 and 4/5/6, not 23 and 456.
                tokens.extend(
                    (char, start + offset, start + offset + 1)
                    for offset, char in enumerate(raw)
                )
            else:
                tokens.append((raw, start, end))
    for match in re.finditer(
        r"单(?=\s*(?:镜头|鏡頭|摄像头|攝像頭|摄|攝|录|錄|[双雙]\s*录|錄))",
        text,
    ):
        tokens.append(("一", match.start(), match.end()))
    for match in VIETNAMESE_NUMBER_RE.finditer(text):
        word = match.group(0).casefold()
        before = text[: match.start()].rstrip().casefold()
        after = text[match.end() :].lstrip().casefold()
        if word == "ba" and re.match(r"lô\b", after):
            continue  # “ba lô” is a backpack, not the number three.
        if word == "năm" and re.search(
            r"(?:\d+(?:[.,]\d+)?|một|hai|ba|bốn|sáu|bảy|tám|chín|mười)\s*$",
            before,
            re.IGNORECASE,
        ):
            # In '10 năm tuổi', năm is the unit year, not the number five.
            continue
        followed_by_count = any(after.startswith(marker) for marker in VIETNAMESE_COUNT_MARKERS)
        preceded_by_count = any(before.endswith(marker) for marker in VIETNAMESE_COUNT_MARKERS)
        if followed_by_count or preceded_by_count:
            tokens.append((match.group(0), match.start(), match.end()))
    for match in VIETNAMESE_DOUBLE_RE.finditer(text):
        tokens.append(("2", match.start(), match.end()))
    roman_values = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
    for match in re.finditer(r"(?i)\btype\s+(?P<roman>VIII|VII|VI|IV|IX|III|II|X|V|I)\b", text):
        tokens.append((str(roman_values[match.group("roman").casefold()]), match.start("roman"), match.end("roman")))
    tokens.sort(key=lambda item: (item[1], item[2]))
    return tokens


def _currency_amounts(text: str) -> list[float]:
    amounts = []
    for match in CURRENCY_RE.finditer(text):
        raw = next((value for value in match.groups() if value is not None), None)
        if raw:
            try:
                amounts.append(_number_value(raw))
            except ValueError:
                continue
    chinese_currency_re = re.compile(rf"([{CHINESE_NUMERAL_PATTERN}]+)\s*(?:元|人民币)")
    for match in chinese_currency_re.finditer(text):
        try:
            amounts.append(_number_value(match.group(1)))
        except ValueError:
            continue
    return sorted(amounts)


def _mass_in_grams(amount: float, unit: str) -> float:
    lower = unit.casefold()
    if lower in {"kg", "公斤", "千克"}:
        return amount * 1000.0
    if lower in {"克", "g", "gram", "grams"}:
        return amount
    if lower == "cân":
        return amount * 1000.0
    return amount


def _close_number_lists(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(sorted(left), sorted(right)))


def _number_specs_match(source: list[float], candidate: list[float]) -> bool:
    """Require all distinct source values without allowing invented/repeated values.

    Repeated wording of the same spec (for example, a pack count repeated twice)
    need not be repeated in Vietnamese. Distinct numbers and candidate additions
    remain strict so a missing size or an invented quantity still fails QA.
    """
    source_counts = Counter(round(value, 6) for value in source)
    candidate_counts = Counter(round(value, 6) for value in candidate)
    return set(source_counts) == set(candidate_counts) and all(
        count <= source_counts[value] for value, count in candidate_counts.items()
    )


def _number_check(source: str, candidate: str) -> tuple[bool, bool, bool]:
    """Return (all_numbers_match, currency_matches, valid_jin_conversion)."""
    source_currency = _currency_amounts(source)
    candidate_currency = _currency_amounts(candidate)
    currency_matches = _close_number_lists(source_currency, candidate_currency)

    source_tokens = _number_tokens(source)
    candidate_tokens = _number_tokens(candidate)
    candidate_has_backpack = bool(
        re.search(r"\b(?:ba\s*lô|balo|backpack|rucksack)\b", candidate, re.IGNORECASE)
    )
    candidate_explicit_two_straps = bool(
        re.search(
            r"\b(?:hai|2)\s*(?:quai|dây(?:\s+đeo|\s+vai)?)\b|"
            r"\b(?:two|2)\s+shoulder\s+straps?\b",
            candidate,
            re.IGNORECASE,
        )
    )
    # “Ba lô/backpack” already denotes a two-strap bag in Vietnamese/English.
    # Do not require a literal numeral unless the candidate explicitly describes
    # the straps; keep the check strict if the output did not translate it as a bag.
    if candidate_has_backpack and not candidate_explicit_two_straps:
        source_tokens = [
            token
            for token in source_tokens
            if not (
                token[0] in {"双", "雙"}
                and re.match(r"肩(?:包|背|带|帶)", source[token[2] :])
            )
        ]

    school_levels_in_source: set[int] = set()
    if "小学" in source:
        school_levels_in_source.add(1)
    if "初中" in source:
        school_levels_in_source.add(2)
    if "高中" in source:
        school_levels_in_source.add(3)
    dual_recording_spans = {
        (start, end)
        for raw, start, end in source_tokens
        if raw in {"双", "雙"} and source[end : end + 1] in {"录", "錄", "摄", "攝", "镜", "鏡"}
    }
    has_front_and_rear = bool(
        re.search(r"\btrước\b", candidate, re.IGNORECASE)
        and re.search(r"\bsau\b", candidate, re.IGNORECASE)
    )
    candidate_mentions_pair = bool(
        re.search(r"\b(?:đôi|kép|double)\b", candidate, re.IGNORECASE)
    )
    candidate_has_explicit_two = any(
        math.isclose(_number_value(raw), 2.0) for raw, _, _ in candidate_tokens
    )
    if dual_recording_spans and has_front_and_rear and not (candidate_has_explicit_two or candidate_mentions_pair):
        source_tokens = [
            token for token in source_tokens if (token[1], token[2]) not in dual_recording_spans
        ]
    source_jin = []
    source_jin_range_spans: set[tuple[int, int]] = set()
    for match in re.finditer(
        r"(?P<lower>\d+(?:[.,]\d+)?)\s*[-~至]\s*(?P<upper>\d+(?:[.,]\d+)?)\s*斤",
        source,
    ):
        for group in ("lower", "upper"):
            span = match.span(group)
            source_jin.append((match.group(group), span[0], span[1], 0.0))
            source_jin_range_spans.add(span)
    for match in JIN_RE.finditer(source):
        raw = match.group("amount") or match.group("half_only")
        if raw:
            start_group = "amount" if match.group("amount") else "half_only"
            if (match.start(start_group), match.end(start_group)) in source_jin_range_spans:
                continue
            extra_half = 0.5 if match.group("extra_half") else 0.0
            source_jin.append((raw, match.start(start_group), match.end(start_group), extra_half))
    source_jin_spans = {(start, end) for _, start, end, _ in source_jin}

    # A 斤 conversion is accepted only when the candidate explicitly gives an
    # equivalent mass unit. The source price itself is always compared exactly.
    valid_jin_conversion = False
    converted_source_spans: set[tuple[int, int]] = set()
    converted_candidate_spans: set[tuple[int, int]] = set()
    if source_jin and not source_currency:
        source_masses: list[tuple[float, tuple[int, int], str]] = []
        candidate_masses: list[tuple[float, tuple[int, int], str]] = []
        for match, collection in (
            *((match, source_masses) for match in MASS_RE.finditer(source)),
            *((match, candidate_masses) for match in MASS_RE.finditer(candidate)),
        ):
            try:
                grams = _mass_in_grams(_number_value(match.group(1)), match.group(2))
            except ValueError:
                continue
            collection.append((grams, (match.start(1), match.end(1)), match.group(2).casefold()))

        # Vietnamese commonly writes a shared unit once after a range, e.g.
        # 50-150 kg. Treat both endpoints as masses for 斤 conversion checks.
        for match in re.finditer(
            r"(?P<lower>\d+(?:[.,]\d+)?)\s*[-~]\s*(?P<upper>\d+(?:[.,]\d+)?)\s*(?P<unit>kg|公斤|千克|克|grams|gram|g|cân)(?=$|[^A-Za-z0-9_])",
            candidate,
            re.IGNORECASE,
        ):
            unit = match.group("unit").casefold()
            for group in ("lower", "upper"):
                try:
                    grams = _mass_in_grams(_number_value(match.group(group)), unit)
                except ValueError:
                    continue
                candidate_masses.append((grams, match.span(group), unit))

        for raw, start, end, extra_half in source_jin:
            try:
                expected_grams = (_number_value(raw) + extra_half) * 500.0
            except ValueError:
                continue
            matching_candidates = [
                item for item in candidate_masses
                if math.isclose(expected_grams, item[0], rel_tol=0.01, abs_tol=1.0)
                and item[1] not in converted_candidate_spans
            ]
            if not matching_candidates:
                continue

            # If the source already repeats the same quantity in metric units,
            # prefer a converted candidate token in a different unit so the
            # original metric amount remains available for its own source token.
            source_same_mass = [
                item for item in source_masses
                if math.isclose(expected_grams, item[0], rel_tol=0.01, abs_tol=1.0)
            ]
            source_signatures = {(round(item[0], 6), item[2]) for item in source_same_mass}
            selected = next(
                (
                    item for item in matching_candidates
                    if (round(item[0], 6), item[2]) not in source_signatures
                ),
                matching_candidates[0],
            )
            valid_jin_conversion = True
            converted_source_spans.add((start, end))
            converted_candidate_spans.add(selected[1])
            # A single equivalent metric quantity can faithfully represent a
            # duplicated source form such as 1斤/500g without duplicating it.
            if source_same_mass and (round(selected[0], 6), selected[2]) in source_signatures:
                converted_source_spans.add(source_same_mass[0][1])

    source_numbers = []
    for raw, start, end in source_tokens:
        if (start, end) in converted_source_spans:
            continue
        if (
            raw == "七"
            and source[end : end + 2] in {"子饼", "子粉"}
            and re.search(r"\bthất\s+tử\b", candidate, re.IGNORECASE)
        ):
            continue  # 七子饼 / 七子粉 are named product terms rendered as Thất Tử.
        if raw == "双" and source[end : end + 1] == "抗" and len(
            re.findall(r"\bchống\b", candidate, re.IGNORECASE)
        ) >= 2 and not re.search(
            r"\b(?:hai|2)\s+(?:tác dụng|công dụng|chức năng)\b", candidate, re.IGNORECASE
        ):
            continue  # 双抗 can be expressed by two explicit Vietnamese anti-action phrases.
        if raw == "双" and source[end : end + 2] == "下巴" and re.search(
            r"\bnọng\s+cằm\b", candidate, re.IGNORECASE
        ):
            continue  # 双下巴 is the idiom “double chin”, commonly translated as nọng cằm.
        if raw == "0" and source[end : end + 2] == "添加" and re.search(
            r"\bkhông\b.{0,30}\b(?:phụ\s+gia|chất\s+phụ\s+gia)\b",
            candidate,
            re.IGNORECASE,
        ):
            continue  # 0添加 -> không phụ gia is a semantic zero-additive equivalent.
        if raw == "一" and source[max(0, start - 1) : end] == "合一" and re.search(
            r"\bkết\s+hợp\b|\ball[- ]in[- ]one\b|\bcombined\b",
            candidate,
            re.IGNORECASE,
        ):
            continue  # 合一 describes combined functions in one product; “kết hợp” preserves that idiom.
        if (
            "一片式" in source
            and raw == "一"
            and re.search(r"liền\s+thân|one[- ]piece", candidate, re.IGNORECASE)
        ):
            continue  # 一片式 is expressed by the Vietnamese clothing term liền thân.
        if (
            raw == "四"
            and source[end : end + 1] == "季"
            and re.search(r"(?:mọi|mỗi|tất cả các)\s+mùa|quanh năm|suốt năm", candidate, re.IGNORECASE)
        ):
            continue
        if (
            raw == "70"
            and source[end : end + 1] == "迈"
            and re.search(r"\b70mai\b", candidate, re.IGNORECASE)
        ):
            continue  # 70迈 is the 70mai brand name, not a standalone quantity.
        if re.fullmatch(r"\d{2}", raw) and source[end : end + 1] in {"年", "款"}:
            shorthand_year = float(f"20{raw}")
            if any(
                math.isclose(_number_value(candidate_raw), shorthand_year)
                for candidate_raw, _, _ in candidate_tokens
                if candidate_raw.isdigit()
            ):
                source_numbers.append(shorthand_year)
                continue  # 26年 / 25款 are commonly rendered as 2026 / 2025.
        if raw == "一" and source[end : end + 1] == "对":
            source_numbers.append(2.0)  # 一对 denotes one pair, i.e. two items.
            continue
        try:
            source_numbers.append(_number_value(raw))
        except ValueError:
            continue
    candidate_numbers = []
    for raw, start, end in candidate_tokens:
        if (start, end) in converted_candidate_spans:
            continue
        if "次抛" in source and raw.casefold() == "một" and re.match(
            r"\s+lần\b", candidate[end:], re.IGNORECASE
        ):
            continue  # 次抛 denotes a single-use portion, not an added numeric spec.
        if "单肩" in source and raw.casefold() == "một" and re.match(
            r"\s+vai\b", candidate[end:], re.IGNORECASE
        ):
            continue  # 单肩 -> một vai is a faithful grammatical equivalent.
        if "吊带" in source and raw.casefold() == "hai" and re.match(
            r"\s*dây\b", candidate[end:], re.IGNORECASE
        ):
            # 'váy hai dây' is the conventional Vietnamese rendering of 吊带,
            # not a separately added numeric specification.
            continue
        if (
            "初高中" in source
            and raw.casefold() in {"2", "3"}
            and re.search(r"\bcấp\s*2\b", candidate, re.IGNORECASE)
            and re.search(r"\bcấp\s*3\b", candidate, re.IGNORECASE)
        ):
            continue  # 初高中 maps to Vietnamese schooling levels cấp 2/cấp 3.
        if (
            _number_value(raw).is_integer()
            and int(_number_value(raw)) in school_levels_in_source
            and re.search(r"\bcấp\s*$", candidate[max(0, start - 12) : start], re.IGNORECASE)
        ):
            continue  # Chinese school-stage terms are conventionally localized as Vietnamese cấp N.
        if (
            "一体" in source
            and raw.casefold() == "một"
            and re.search(r"\btất cả trong\s*$", candidate[max(0, start - 24) : start], re.IGNORECASE)
        ):
            continue  # 一体机 -> tất cả trong một is an idiom, not a quantity.
        if (
            "瑞虎九" in source
            and raw == "9"
            and re.search(r"\btiggo\s*$", candidate[max(0, start - 12) : start], re.IGNORECASE)
        ):
            continue  # 瑞虎九 is conventionally rendered Tiggo 9 (a model alias).
        try:
            candidate_numbers.append(_number_value(raw))
        except ValueError:
            continue
    # These are faithful, conventional renderings rather than invented specs:
    # 一折 means pay 10% of the original price (or receive a 90% discount), and
    # 纯棉 means 100% cotton. Keep the exception narrowly tied to explicit text.
    if "一折" in source:
        if re.search(r"\bgiảm(?:\s+giá)?\s*90\s*%|90\s*%\s*(?:giảm|chiết khấu)", candidate, re.IGNORECASE):
            for index, number in enumerate(candidate_numbers):
                if math.isclose(number, 90.0):
                    candidate_numbers.pop(index)
                    break
        elif re.search(r"\b(?:còn|giá còn)\s*10\s*%", candidate, re.IGNORECASE):
            for index, number in enumerate(candidate_numbers):
                if math.isclose(number, 10.0):
                    candidate_numbers.pop(index)
                    break
    source_has_explicit_cotton_percent = bool(
        re.search(r"100\s*[%％]?\s*(?:棉|纯棉|全棉)", source)
    )
    if (
        "纯棉" in source
        and not source_has_explicit_cotton_percent
        and re.search(r"\b100\s*%", candidate, re.IGNORECASE)
    ):
        for index, number in enumerate(candidate_numbers):
            if math.isclose(number, 100.0):
                candidate_numbers.pop(index)
                break
    if (
        re.search(r"100\s*[%％]?\s*(?:纯棉|全棉)", source)
        and any(term in source for term in ("纯棉", "全棉"))
        and re.search(
            r"\bcotton\b.{0,24}(?:nguyên chất|pure)|(?:nguyên chất|pure).{0,24}\bcotton\b",
            candidate,
            re.IGNORECASE,
        )
    ):
        for index, number in enumerate(source_numbers):
            if math.isclose(number, 100.0):
                source_numbers.pop(index)
                break
    # "đôi" is a classifier in "một đôi" (one pair), but it can also express
    # a double feature (e.g. giường đôi). Use it as 2 only to match an explicit
    # source 双/雙 token, and only for source-side 2s not already represented.
    pair_source_count = sum(
        raw in {"双", "雙"} or (raw == "一" and source[end : end + 1] == "对")
        for raw, _, end in source_tokens
    )
    if pair_source_count:
        pair_positions = [
            match.start() for match in re.finditer(r"\bđôi\b", candidate, re.IGNORECASE)
        ]
        missing_twos = max(0, source_numbers.count(2.0) - candidate_numbers.count(2.0))
        for _ in pair_positions[: min(pair_source_count, missing_twos)]:
            candidate_numbers.append(2.0)
    all_numbers_match = _number_specs_match(source_numbers, candidate_numbers)
    if (
        not all_numbers_match
        and len(source_numbers) == 1
        and len(candidate_numbers) == 2
        and math.isclose(candidate_numbers[0], candidate_numbers[1], rel_tol=1e-6, abs_tol=1e-6)
        and _close_number_lists(source_numbers, [candidate_numbers[0]])
        and re.search(r"\d+\s*/\s*\d+", candidate)
        and re.search(r"\d+(?:[.,]\d+)?\s*(?:小时|小時|hour|hours|hr|h)(?!\w)", source, re.IGNORECASE)
    ):
        all_numbers_match = True
    all_numbers_match = all_numbers_match and currency_matches
    return all_numbers_match, currency_matches, valid_jin_conversion


def _is_model_code(token: str) -> bool:
    if not (any(char.isalpha() for char in token) and any(char.isdigit() for char in token)):
        return False
    # Quantity abbreviations such as 43PCS are counts, not product/model codes.
    if re.fullmatch(r"\d+(?:pcs?|pieces?|sets?|pairs?|units?)", token, re.IGNORECASE):
        return False
    if re.fullmatch(r"\d+(?:[.,]\d+)?(?:x\d+(?:[.,]\d+)?)+(?:cm|mm|m)?", token, re.IGNORECASE):
        return False
    if re.fullmatch(
        r"\d+(?:[.,]\d+)?(?:kg|mg|g|ml|l|mah|cm|mm|inch|in|h|hr|hrs|hour|hours|camera|cameras|lens|lenses|channel|channels)",
        token,
        re.IGNORECASE,
    ):
        if token.upper() not in {"3G", "4G", "5G"}:
            return False
    # Measurements can be concatenated with another number in noisy titles,
    # e.g. 2500g5斤, or split at an ASCII letter inside a Vietnamese word,
    # e.g. 800g/hộp. They are quantities, not opaque model identifiers.
    if re.fullmatch(r"\d+(?:[.,]\d+)?(?:kg|mg|g|ml|l|cm|mm)\d+", token, re.IGNORECASE):
        return False
    if re.fullmatch(
        r"\d+(?:[.,]\d+)?(?:mg|kg|g|ml|l|mah|克|公斤|千克|斤|毫安|cm|mm|寸|英寸)"
        r"(?:/\d+(?:[.,]\d+)?(?:mg|kg|g|ml|l|mah|克|公斤|千克|斤|毫安|cm|mm|寸|英寸))+",
        token,
        re.IGNORECASE,
    ):
        return False  # Slash-separated mass options (e.g. 50g/180g) are quantities.
    if re.fullmatch(r"\d+(?:[.,]\d+)?(?:kg|mg|g|ml|l)/h", token, re.IGNORECASE):
        return False
    return True


def _is_contextual_model_code(text: str, match: re.Match[str]) -> bool:
    token = match.group(0)
    if not _is_model_code(token):
        return False
    if re.fullmatch(r"\d+(?:[.,]\d+)?s", token, re.IGNORECASE) and re.search(
        r"\b(?:cotton|supima|textile|mercerized)\b|支|纯棉|全棉|棉", text, re.IGNORECASE
    ):
        return False  # Textile thread-count notation such as 60s is a quantity, not a model code.
    suffix = text[match.end() :]
    if re.fullmatch(r"\d+(?:[.,]\d+)?(?:kg|mg|g|ml|l|cm|mm)\d+", token, re.IGNORECASE) and suffix.startswith(
        ("斤", "半斤", "公斤", "千克", "克")
    ):
        return False
    if re.search(r"20\d{2}$", token) and suffix.startswith(("年", "年度")):
        return False
    return True


def _model_codes(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in CODE_RE.finditer(text)
        if _is_contextual_model_code(text, match)
    }


def _missing_model_codes(source: str, candidate: str) -> list[str]:
    candidate_compact = re.sub(r"[^A-Za-z0-9]+", "", candidate).casefold()
    return sorted(
        code
        for code in _model_codes(source)
        if re.sub(r"[^A-Za-z0-9]+", "", code).casefold() not in candidate_compact
    )


def _repeated_phrase(text: str) -> bool:
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if len(words) < 12:
        return False
    for width in (4, 5, 6):
        grams = Counter(tuple(words[index : index + width]) for index in range(len(words) - width + 1))
        if any(count >= 3 for count in grams.values()):
            return True
    return False


def check_translation(
    source: Any,
    candidate: Any,
    *,
    generation_truncated: bool = False,
) -> dict[str, Any]:
    source_text = unicodedata.normalize("NFC", "" if source is None else str(source)).strip()
    text = unicodedata.normalize("NFC", "" if candidate is None else str(candidate)).strip()
    hard_flags: list[str] = []
    advisory_flags: list[str] = []
    if not text:
        hard_flags.append("empty")
        return {"status": "fail", "hard_flags": hard_flags, "advisory_flags": advisory_flags}

    if generation_truncated:
        hard_flags.append("generation_limit_reached")
    if contains_cjk(text):
        hard_flags.append("cjk_remaining")
    if EXPLANATION_RE.search(text) or "\n" in text.strip():
        hard_flags.append("explanation_or_multiple_lines")
    if _repeated_phrase(text):
        hard_flags.append("repeated_phrase")
    if contains_cjk(source_text) and normalize_title_key(text) == normalize_title_key(source_text):
        hard_flags.append("copied_chinese_source")

    number_match, currency_match, jin_converted = _number_check(source_text, text)
    if not number_match:
        hard_flags.append("number_mismatch")
    if not currency_match:
        hard_flags.append("currency_amount_mismatch")

    source_codes = _model_codes(source_text)
    missing_codes = _missing_model_codes(source_text, text)
    if missing_codes:
        hard_flags.append("model_code_missing:" + ",".join(missing_codes))

    if "斤" in source_text:
        advisory_flags.append("source_unit_jin")
        if jin_converted:
            advisory_flags.append("jin_mass_conversion_detected")
        elif not any(unit in text.casefold() for unit in ("斤", "jin")):
            hard_flags.append("jin_unit_unresolved")

    if contains_cyrillic(source_text):
        advisory_flags.append("source_contains_cyrillic")
    source_latin = {word.casefold() for word in LATIN_WORD_RE.findall(source_text)}
    candidate_latin = {word.casefold() for word in LATIN_WORD_RE.findall(text)}
    extra_latin = sorted(candidate_latin - source_latin - LATIN_ADVISORY_STOPWORDS)
    if extra_latin:
        advisory_flags.append("unmatched_latin_tokens:" + ",".join(extra_latin[:8]))
    ratio = len(text) / max(1, len(source_text))
    if ratio > 3.5 or ratio < 0.12:
        advisory_flags.append("length_ratio_outlier")

    status = "fail" if hard_flags else "pass"
    return {
        "status": status,
        "hard_flags": sorted(set(hard_flags)),
        "advisory_flags": sorted(set(advisory_flags)),
        "number_match": number_match,
        "currency_match": currency_match,
        "jin_conversion_detected": jin_converted,
        "source_model_codes": sorted(source_codes),
        "missing_model_codes": missing_codes,
        "length_ratio": ratio,
    }


def select_translation(
    source: Any,
    title_vi: Any,
    title_vi_cleaned: Any,
    translategemma: Any,
    qwen3: Any,
    *,
    translategemma_truncated: bool = False,
    qwen3_truncated: bool = False,
) -> dict[str, Any]:
    checks = {
        "baseline_cleaned": check_translation(source, title_vi_cleaned),
        "baseline_raw": check_translation(source, title_vi),
        "translategemma": check_translation(
            source, translategemma, generation_truncated=translategemma_truncated
        ),
        "qwen3": check_translation(source, qwen3, generation_truncated=qwen3_truncated),
    }
    cleaned = "" if title_vi_cleaned is None else str(title_vi_cleaned).strip()
    raw = "" if title_vi is None else str(title_vi).strip()
    gemma = "" if translategemma is None else str(translategemma).strip()
    qwen = "" if qwen3 is None else str(qwen3).strip()
    if checks["baseline_cleaned"]["status"] == "pass":
        selected, source_name, decision = cleaned, "title_vi_cleaned", "keep_cleaned"
    elif checks["baseline_raw"]["status"] == "pass":
        selected, source_name, decision = raw, "title_vi", "fallback_to_raw_vi"
    elif (
        checks["translategemma"]["status"] == "pass"
        and checks["qwen3"]["status"] == "pass"
        and normalize_title_key(gemma)
        and normalize_title_key(gemma) == normalize_title_key(qwen)
    ):
        selected, source_name, decision = gemma, "translategemma_agreement", "replace_existing_by_agreement"
    else:
        selected, source_name, decision = None, "hold", "hold_no_safe_automatic_choice"
    return {
        "selected": selected,
        "source": source_name,
        "decision": decision,
        "qa_status": "pass" if selected is not None else "hold",
        "checks": checks,
        "candidate_agreement": bool(normalize_title_key(gemma) and normalize_title_key(gemma) == normalize_title_key(qwen)),
        "semantic_verified": False,
    }


def allocate_proportionally(counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    population = sum(counts.values())
    if population == 0 or total == 0:
        return {key: 0 for key in counts}
    exact = {key: total * value / population for key, value in counts.items()}
    allocated = {key: int(math.floor(value)) for key, value in exact.items()}
    left = total - sum(allocated.values())
    order = sorted(counts, key=lambda key: (exact[key] - allocated[key], counts[key], key), reverse=True)
    for key in order[:left]:
        allocated[key] += 1
    return allocated


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    category_counts = df.groupby("title_zh", sort=False)["category"].nunique(dropna=False)
    if (category_counts > 1).any():
        raise ValueError("A source title appears in more than one category; sampling assumptions changed.")
    title_categories = df.drop_duplicates("title_zh").set_index("title_zh")["category"].to_dict()
    has_cjk = df["title_vi"].map(contains_cjk) | df["title_vi_cleaned"].map(contains_cjk)
    title_has_cjk = (
        pd.DataFrame({"title_zh": df["title_zh"], "has_cjk": has_cjk})
        .groupby("title_zh", sort=False)["has_cjk"]
        .any()
        .to_dict()
    )
    titles = sorted(df["title_zh"].astype(str).unique().tolist())
    records = []
    for source in titles:
        records.append(
            {
                "title_zh": source,
                "category": str(title_categories.get(source, "unknown")),
                "source_cyrillic": contains_cyrillic(source),
                "source_latin": any(char.isascii() and char.isalpha() for char in source),
                "source_digits": any(char.isdigit() for char in source),
                "source_jin": "斤" in source,
                "source_brackets": any(char in source for char in "()[]{}【】（）"),
                "existing_vi_cjk": bool(title_has_cjk.get(source, False)),
            }
        )
    return pd.DataFrame.from_records(records)


def _sample_challenge(groups: pd.DataFrame, excluded: set[str], count: int, rng: random.Random) -> list[str]:
    eligible = groups[~groups["title_zh"].isin(excluded)].copy()
    if eligible.empty or count <= 0:
        return []
    selected: list[str] = []
    remaining = eligible.copy()
    # Include every known Cyrillic source and every existing VI title with CJK
    # in the first pilot's challenge set whenever the planned size permits it.
    required = remaining[remaining["source_cyrillic"] | remaining["existing_vi_cjk"]]
    required_titles = required["title_zh"].tolist()
    if len(required_titles) <= count:
        selected.extend(required_titles)
        remaining = remaining[~remaining["title_zh"].isin(selected)]
    elif count:
        order = list(required_titles)
        rng.shuffle(order)
        selected.extend(order[:count])
        return selected

    feature_columns = ["source_jin", "source_digits", "source_latin", "source_brackets"]
    scored = []
    for _, row in remaining.iterrows():
        feature_count = sum(bool(row[column]) for column in feature_columns)
        scored.append((feature_count, rng.random(), str(row["title_zh"])))
    scored.sort(reverse=True)
    selected.extend(title for _, _, title in scored[: max(0, count - len(selected))])
    if len(selected) < count:
        leftovers = [title for title in eligible["title_zh"].tolist() if title not in set(selected)]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])
    return selected[:count]


def _sample_representative(
    groups: pd.DataFrame,
    excluded: set[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    eligible = groups[~groups["title_zh"].isin(excluded)]
    category_counts = eligible.groupby("category")["title_zh"].size().astype(int).to_dict()
    allocation = allocate_proportionally(category_counts, min(count, len(eligible)))
    selected: list[str] = []
    seed = rng.randrange(0, 2**32 - 1)
    for index, (category, take) in enumerate(sorted(allocation.items())):
        subset = eligible[eligible["category"] == category]
        if take:
            sample = subset.sample(n=take, random_state=(seed + index) % (2**32 - 1))
            selected.extend(sample["title_zh"].tolist())
    return selected


def build_sample_manifest(df: pd.DataFrame) -> dict[str, Any]:
    groups = _group_table(df)
    if groups["title_zh"].nunique() != EXPECTED_UNIQUE_TITLES:
        raise ValueError("Unexpected number of unique source titles; review the pinned dataset.")
    rng = random.Random(SEED)
    pilot_challenge = _sample_challenge(groups, set(), 50, rng)
    pilot_rep = _sample_representative(groups, set(pilot_challenge), 100, rng)
    if len(pilot_challenge) != 50 or len(pilot_rep) != 100:
        raise ValueError("Could not construct the planned 150-title pilot sample.")
    pilot_titles = pilot_rep + pilot_challenge
    validation_challenge = _sample_challenge(groups, set(pilot_titles), 40, rng)
    validation_rep = _sample_representative(groups, set(pilot_titles) | set(validation_challenge), 160, rng)
    validation_titles = validation_rep + validation_challenge
    if len(validation_titles) != 200 or set(validation_titles) & set(pilot_titles):
        raise ValueError("Validation sample overlaps pilot or does not contain 200 unique titles.")
    return {
        "sample_version": SAMPLE_VERSION,
        "seed": SEED,
        "dataset_sha256": DATASET_SHA256,
        "pilot": {
            "representative": pilot_rep,
            "challenge": pilot_challenge,
            "titles": pilot_titles,
        },
        "validate": {
            "representative": validation_rep,
            "challenge": validation_challenge,
            "titles": validation_titles,
        },
        "smoke": pilot_titles[:20],
        "counts": {
            "unique_source_titles": int(len(groups)),
            "pilot_representative": len(pilot_rep),
            "pilot_challenge": len(pilot_challenge),
            "validation_representative": len(validation_rep),
            "validation_challenge": len(validation_challenge),
        },
        "challenge_coverage": {
            "pilot_cyrillic_source_titles": sum(
                contains_cyrillic(title) for title in pilot_challenge
            ),
            "pilot_existing_vi_cjk_source_titles": int(
                groups[groups["title_zh"].isin(pilot_challenge)]["existing_vi_cjk"].sum()
            ),
            "pilot_titles_with_斤": sum("斤" in title for title in pilot_challenge),
        },
        "created_at_utc": utc_now(),
    }


def validate_input_frame(df: pd.DataFrame) -> None:
    missing = sorted(set(ORIGINAL_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError("Input parquet is missing required columns: " + ", ".join(missing))
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source rows; found {len(df)}.")
    if df["product_id"].isna().any() or not df["product_id"].is_unique:
        raise ValueError("product_id must be present and unique for all source rows.")
    if df["title_zh"].isna().any() or df["title_zh"].astype(str).str.strip().eq("").any():
        raise ValueError("title_zh has missing or blank values.")
    if df["title_zh"].nunique() != EXPECTED_UNIQUE_TITLES:
        raise ValueError("Unique title_zh count does not match the pinned source.")
    cross_categories = df.groupby("title_zh")["category"].nunique(dropna=False)
    if (cross_categories > 1).any():
        raise ValueError("At least one repeated source title spans multiple categories.")


def _runtime_package_versions() -> dict[str, str]:
    names = ("transformers", "accelerate", "bitsandbytes", "sacrebleu", "huggingface_hub", "torch", "pyarrow")
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _model_revision_lock(token: str | None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before starting smoke.") from exc
    api = HfApi(token=token)
    models = {}
    for key, spec in MODEL_SPECS.items():
        info = api.model_info(spec["model_id"], revision="main")
        models[key] = {
            "model_id": spec["model_id"],
            "revision": info.sha,
            "template_version": spec["template_version"],
        }
    bge_info = api.model_info(BGE_MODEL_ID, revision="main")
    models["bge_m3"] = {
        "model_id": BGE_MODEL_ID,
        "revision": bge_info.sha,
        "template_version": BGE_TEMPLATE_VERSION,
    }
    return models


def _config_payload(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "models": lock["models"],
        "prompt_version": PROMPT_VERSION,
        "qa_version": QA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "sample_version": SAMPLE_VERSION,
        "generation": {
            "initial_max_new_tokens": MAX_NEW_TOKENS,
            "truncation_retry_tokens": TRUNCATION_RETRY_TOKENS,
            "do_sample": False,
            "qwen_enable_thinking": False,
            "quantization": "NF4-4bit",
            "compute_dtype": "float16",
        },
        "quality_gates": {
            "completion_rate": PILOT_MIN_COMPLETION_RATE,
            "representative_qa_rate": PILOT_MIN_REPRESENTATIVE_QA_RATE,
            "full_window_size": FULL_WINDOW_SIZE,
            "full_max_incomplete_rate": FULL_MAX_INCOMPLETE_RATE,
            "full_max_qa_failure_rate": FULL_MAX_QA_FAILURE_RATE,
        },
        "package_versions": lock["package_versions"],
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def load_or_create_environment_lock(
    workdir: Path,
    *,
    mode: str,
    token: str | None,
) -> dict[str, Any]:
    lock_path = workdir / "environment_lock.json"
    if not lock_path.exists():
        if mode != "smoke":
            raise RuntimeError("Run smoke first; it resolves and pins model revisions and environment.")
        models = _model_revision_lock(token)
        package_versions = _runtime_package_versions()
        required = {
            "transformers": "5.20.2",
            "accelerate": "1.15.0",
            "bitsandbytes": "0.50.2",
            "sacrebleu": "2.6.0",
        }
        mismatches = [
            f"{name} expected {wanted}, found {package_versions.get(name)}"
            for name, wanted in required.items()
            if package_versions.get(name) != wanted
        ]
        if mismatches:
            raise RuntimeError("Runtime packages are not locked as planned: " + "; ".join(mismatches))
        lock = {
            "created_at_utc": utc_now(),
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": DATASET_SHA256,
            "models": models,
            "package_versions": package_versions,
            "python": platform.python_version(),
            "torch_cuda_runtime": None,
            "gpu_names": [],
        }
        try:
            import torch

            lock["torch_cuda_runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                lock["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except ImportError:
            pass
        lock["config"] = _config_payload(lock)
        lock["config_id"] = _fingerprint(lock["config"])
        atomic_write_json(lock_path, lock)
        return lock

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("dataset_revision") != DATASET_REVISION or lock.get("dataset_sha256") != DATASET_SHA256:
        raise RuntimeError("Saved environment lock belongs to another dataset revision.")
    current = _runtime_package_versions()
    locked = lock.get("package_versions", {})
    for name in ("transformers", "accelerate", "bitsandbytes", "sacrebleu", "huggingface_hub", "torch"):
        if locked.get(name) != current.get(name):
            raise RuntimeError(
                f"Environment changed for {name}: locked {locked.get(name)}, current {current.get(name)}. "
                "Create a new smoke run and validation instead of mixing checkpoints."
            )
    expected_config = _config_payload(lock)
    if lock.get("config_id") != _fingerprint(expected_config):
        raise RuntimeError("Environment lock fingerprint is invalid or settings were edited.")
    return lock


def _resolve_input_path(data_path: str | Path | None, token: str | None) -> Path:
    if data_path:
        candidate = Path(data_path)
        if candidate.exists():
            return candidate
    local = Path.cwd() / Path(DATASET_FILE).name
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Input parquet not found; install huggingface_hub for the pinned HF download.") from exc
    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
            token=token,
        )
    )


def _copy_checkpoint_inputs(source: Path | None, workdir: Path) -> None:
    if source is None or not source.exists():
        return
    workdir.mkdir(parents=True, exist_ok=True)
    for name in (
        "environment_lock.json",
        "sample_manifest.json",
        "run_state.json",
        "cache_translategemma.sqlite3",
        "cache_qwen3.sqlite3",
        "title_translation_report.md",
        "pilot_report.json",
        "validate_report.json",
        "full_report.json",
        "pilot_audit.parquet",
        "validate_audit.parquet",
        "disagreement_metrics.parquet",
        "bilingual_zh_vi_cado_title_v2.parquet",
    ):
        origin = source / name
        destination = workdir / name
        if origin.exists() and not destination.exists():
            shutil.copy2(origin, destination)


def prepare_pipeline(
    *,
    mode: str,
    data_path: str | Path | None,
    workdir: str | Path,
    checkpoint_input: str | Path | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    mode = mode.lower().strip()
    if mode not in {"smoke", "pilot", "validate", "full"}:
        raise ValueError("mode must be smoke, pilot, validate, or full")
    output_dir = Path(workdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_input) if checkpoint_input else None
    _copy_checkpoint_inputs(checkpoint_path, output_dir)

    input_path = _resolve_input_path(data_path, token)
    actual_sha = sha256_file(input_path)
    if actual_sha != DATASET_SHA256:
        raise ValueError(
            f"Input parquet SHA-256 mismatch. Expected {DATASET_SHA256}; got {actual_sha}."
        )
    frame = pd.read_parquet(input_path)
    validate_input_frame(frame)
    lock = load_or_create_environment_lock(output_dir, mode=mode, token=token)

    sample_path = output_dir / "sample_manifest.json"
    if sample_path.exists():
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        if samples.get("dataset_sha256") != DATASET_SHA256 or samples.get("sample_version") != SAMPLE_VERSION:
            raise RuntimeError("Saved sample manifest does not match the pinned dataset and sampler.")
    else:
        if mode != "smoke":
            raise RuntimeError("Run smoke first to persist the sample manifest before other stages.")
        samples = build_sample_manifest(frame)
        atomic_write_json(sample_path, samples)
        _write_sample_csv(samples, output_dir / "sample_manifest.csv")
    state_path = output_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "config_id": lock["config_id"],
        "dataset_sha256": DATASET_SHA256,
        "stages": {},
        "created_at_utc": utc_now(),
    }
    if state.get("config_id") != lock["config_id"]:
        raise RuntimeError("Run state has another config id; do not mix model or QA settings.")
    return {
        "mode": mode,
        "data_path": str(input_path),
        "workdir": output_dir,
        "frame": frame,
        "lock": lock,
        "samples": samples,
        "state": state,
    }


def _write_sample_csv(samples: dict[str, Any], path: Path) -> None:
    records = []
    for title in samples["smoke"]:
        records.append({"stage": "smoke", "sample_type": "pilot_subset", "title_zh": title})
    for stage in ("pilot", "validate"):
        for sample_type in ("representative", "challenge"):
            for title in samples[stage][sample_type]:
                records.append({"stage": stage, "sample_type": sample_type, "title_zh": title})
    pd.DataFrame.from_records(records).drop_duplicates(["stage", "title_zh"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


class TranslationCache:
    def __init__(self, path: Path, model_key: str, model_id: str, revision: str, config_id: str):
        self.path = path
        self.model_key = model_key
        self.model_id = model_id
        self.revision = revision
        self.config_id = config_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=60)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    source_title TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    output_text TEXT,
                    status TEXT NOT NULL,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    elapsed_seconds REAL NOT NULL DEFAULT 0,
                    error_text TEXT,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def key(self, title: str) -> str:
        payload = "\n".join(
            (
                title,
                self.model_key,
                self.model_id,
                self.revision,
                self.config_id,
                PROMPT_VERSION,
                str(MAX_NEW_TOKENS),
                str(TRUNCATION_RETRY_TOKENS),
                "greedy",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, title: str) -> dict[str, Any] | None:
        return self.get_many([title]).get(title)

    def get_many(self, titles: Iterable[str]) -> dict[str, dict[str, Any]]:
        titles = list(dict.fromkeys(titles))
        by_key = {self.key(title): title for title in titles}
        rows = []
        keys = list(by_key)
        with self._connect() as connection:
            for offset in range(0, len(keys), 500):
                part = keys[offset : offset + 500]
                placeholders = ",".join("?" for _ in part)
                rows.extend(
                    connection.execute(
                        "SELECT cache_key,output_text,status,truncated,output_tokens,attempts,elapsed_seconds,error_text "
                        f"FROM translations WHERE cache_key IN ({placeholders})",
                        part,
                    ).fetchall()
                )
        result = {}
        for row in rows:
            title = by_key[row[0]]
            result[title] = {
                "output_text": row[1],
                "status": row[2],
                "truncated": bool(row[3]),
                "output_tokens": row[4],
                "attempts": row[5],
                "elapsed_seconds": row[6],
                "error_text": row[7],
            }
        return result

    def get_success(self, title: str) -> dict[str, Any] | None:
        result = self.get(title)
        return result if result and result["status"] == "ok" else None

    def get_complete(self, title: str) -> dict[str, Any] | None:
        result = self.get_success(title)
        return result if result and not result["truncated"] and result["output_text"] else None

    def put(self, title: str, result: dict[str, Any]) -> None:
        self.put_many([(title, result)])

    def put_many(self, items: Iterable[tuple[str, dict[str, Any]]]) -> None:
        records = []
        for title, result in items:
            records.append(
                (
                    self.key(title),
                    title,
                    self.model_key,
                    self.model_id,
                    self.revision,
                    self.config_id,
                    result.get("output_text"),
                    result["status"],
                    int(bool(result.get("truncated", False))),
                    result.get("output_tokens"),
                    int(result.get("attempts", 0)),
                    float(result.get("elapsed_seconds", 0.0)),
                    result.get("error_text"),
                    utc_now(),
                )
            )
        if not records:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO translations (
                    cache_key,source_title,model_key,model_id,model_revision,config_id,
                    output_text,status,truncated,output_tokens,attempts,elapsed_seconds,
                    error_text,updated_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    output_text=excluded.output_text,
                    status=excluded.status,
                    truncated=excluded.truncated,
                    output_tokens=excluded.output_tokens,
                    attempts=excluded.attempts,
                    elapsed_seconds=excluded.elapsed_seconds,
                    error_text=excluded.error_text,
                    updated_at_utc=excluded.updated_at_utc
                """,
                records,
            )
            connection.commit()

    def success_count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM translations WHERE status='ok' AND config_id=?", (self.config_id,)).fetchone()[0]
            )


def _cached_complete(result: dict[str, Any] | None) -> bool:
    return bool(
        result
        and result.get("status") == "ok"
        and not result.get("truncated")
        and result.get("output_text")
    )


def _canonical_value(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None
    if value is None:
        return ("__missing__",)
    if np is not None and isinstance(value, np.ndarray):
        return tuple(_canonical_value(item) for item in value.tolist())
    if isinstance(value, dict):
        return tuple(
            (str(key), _canonical_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item) for item in value)
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return ("__missing__",)
        if np is not None and isinstance(missing, np.bool_) and bool(missing):
            return ("__missing__",)
    except (TypeError, ValueError):
        pass
    return value


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    if left.dtype == right.dtype and left.equals(right):
        return True
    return all(
        _canonical_value(first) == _canonical_value(second)
        for first, second in zip(left.tolist(), right.tolist())
    )


def _model_messages(model_key: str, title: str) -> list[dict[str, Any]]:
    if model_key == "translategemma":
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "zh",
                        "target_lang_code": "vi",
                        "text": title,
                    }
                ],
            }
        ]
    prompt = (
        "Translate this Chinese e-commerce product title into Vietnamese. "
        "Return only the Vietnamese title. Preserve brands, model codes, numbers, and units.\n\n"
        + title
    )
    return [{"role": "user", "content": prompt}]


def _load_generation_model(
    model_key: str,
    revision: str,
    device_index: int,
    token: str | None,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    spec = MODEL_SPECS[model_key]
    model_id = spec["model_id"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    if model_key == "translategemma":
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, token=token)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            token=token,
            quantization_config=quantization,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": device_index},
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            token=token,
            quantization_config=quantization,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": device_index},
        )
        processor = None
    model.eval()
    return model, processor, tokenizer


def _tokenize_batch(
    model_key: str,
    titles: list[str],
    processor: Any,
    tokenizer: Any,
    device: str,
) -> Any:
    conversations = [[*_model_messages(model_key, title)] for title in titles]
    if model_key == "translategemma":
        encoded = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
    return encoded.to(device)


def _decode_generated(
    sequences: Any,
    input_width: int,
    max_new_tokens: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_set: set[int] = set()
    elif isinstance(eos_ids, (list, tuple)):
        eos_set = {int(item) for item in eos_ids}
    else:
        eos_set = {int(eos_ids)}
    results = []
    for sequence in sequences:
        new_tokens = sequence[input_width:].tolist()
        token_count = len(new_tokens)
        found_eos = False
        for index, token_id in enumerate(new_tokens):
            if int(token_id) in eos_set:
                token_count = index + 1
                found_eos = True
                break
        text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        results.append(
            {
                "output_text": text,
                "output_tokens": token_count,
                "truncated": (not found_eos and token_count >= max_new_tokens),
            }
        )
    return results


def _generate_batch(
    model_key: str,
    titles: list[str],
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    encoded = _tokenize_batch(model_key, titles, processor, tokenizer, device)
    input_width = int(encoded["input_ids"].shape[-1])
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_kwargs)
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    return _decode_generated(generated, input_width, max_new_tokens, tokenizer)


def _run_model_titles(
    *,
    model_key: str,
    titles: list[str],
    lock: dict[str, Any],
    workdir: Path,
    token: str | None,
    device_index: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    model_info = lock["models"][model_key]
    cache = TranslationCache(
        workdir / f"cache_{model_key}.sqlite3",
        model_key,
        model_info["model_id"],
        model_info["revision"],
        lock["config_id"],
    )
    old_results = cache.get_many(titles)
    pending = [
        title
        for title in titles
        if title not in old_results
        or old_results[title]["status"] != "ok"
        or old_results[title]["truncated"]
        or not old_results[title]["output_text"]
    ]
    if not pending:
        return {
            "model_key": model_key,
            "requested": len(titles),
            "already_cached": len(titles),
            "completed": len(titles),
            "failed": 0,
            "load_seconds": 0.0,
            "translation_seconds": 0.0,
            "peak_vram_gb": 0.0,
            "device": f"cuda:{device_index}",
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Run this model pipeline on Kaggle GPU.")
    if device_index >= torch.cuda.device_count():
        device_index = 0
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    torch.cuda.reset_peak_memory_stats(device_index)
    load_started = time.perf_counter()
    model, processor, tokenizer = _load_generation_model(
        model_key, model_info["revision"], device_index, token
    )
    load_seconds = time.perf_counter() - load_started
    translate_started = time.perf_counter()
    cursor = 0
    current_batch_size = min(batch_size, len(pending))
    singleton_failures: Counter[str] = Counter()
    while cursor < len(pending):
        batch = pending[cursor : cursor + current_batch_size]
        try:
            batch_started = time.perf_counter()
            outputs = _generate_batch(
                model_key, batch, model, processor, tokenizer, device, MAX_NEW_TOKENS
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Model returned {len(outputs)} outputs for a batch of {len(batch)} titles."
                )
            batch_results = []
            for title, output in zip(batch, outputs):
                attempts = 1
                if output["truncated"]:
                    attempts += 1
                    try:
                        retry_output = _generate_batch(
                            model_key,
                            [title],
                            model,
                            processor,
                            tokenizer,
                            device,
                            TRUNCATION_RETRY_TOKENS,
                        )[0]
                        output = retry_output
                    except Exception as exc:
                        output = {
                            "output_text": output["output_text"],
                            "output_tokens": output["output_tokens"],
                            "truncated": True,
                            "error_text": f"512-token retry failed: {type(exc).__name__}: {exc}",
                        }
                batch_results.append(
                    (
                        title,
                    {
                        "output_text": output.get("output_text"),
                        "status": "ok" if output.get("output_text") else "error",
                        "truncated": output["truncated"],
                        "output_tokens": output.get("output_tokens"),
                        "attempts": attempts,
                        "elapsed_seconds": time.perf_counter() - batch_started,
                        "error_text": output.get("error_text"),
                    },
                    )
                )
            cache.put_many(batch_results)
            cursor += len(batch)
            singleton_failures.clear()
            if current_batch_size < batch_size and cursor < len(pending):
                current_batch_size = min(batch_size, current_batch_size * 2)
            print(
                f"[{model_key}] {cursor}/{len(pending)} uncached titles; batch={current_batch_size}; "
                f"device={device}",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                print(f"[{model_key}] CUDA OOM; reducing batch to {current_batch_size}.", flush=True)
                continue
            title = batch[0]
            singleton_failures[title] += 1
            if singleton_failures[title] >= MAX_SINGLETON_RETRIES:
                cache.put(
                    title,
                    {
                        "status": "error",
                        "attempts": singleton_failures[title],
                        "error_text": f"CUDA OOM after singleton retries: {exc}",
                    },
                )
                cursor += 1
                singleton_failures.pop(title, None)
            continue
        except Exception as exc:
            if current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"[{model_key}] batch failed ({type(exc).__name__}); retrying with batch={current_batch_size}.",
                    flush=True,
                )
                continue
            title = batch[0]
            singleton_failures[title] += 1
            if singleton_failures[title] >= MAX_SINGLETON_RETRIES:
                cache.put(
                    title,
                    {
                        "status": "error",
                        "attempts": singleton_failures[title],
                        "error_text": f"{type(exc).__name__}: {exc}",
                    },
                )
                cursor += 1
                singleton_failures.pop(title, None)
            else:
                time.sleep(1.0)
    translation_seconds = time.perf_counter() - translate_started
    completed_records = cache.get_many(titles)
    completed = sum(
        title in completed_records
        and completed_records[title]["status"] == "ok"
        and not completed_records[title]["truncated"]
        and bool(completed_records[title]["output_text"])
        for title in titles
    )
    peak_vram = torch.cuda.max_memory_allocated(device_index) / (1024**3)
    del model, processor, tokenizer
    torch.cuda.empty_cache()
    return {
        "model_key": model_key,
        "requested": len(titles),
        "already_cached": len(titles) - len(pending),
        "completed": completed,
        "failed": len(titles) - completed,
        "load_seconds": load_seconds,
        "translation_seconds": translation_seconds,
        "titles_per_hour": (len(pending) / translation_seconds * 3600.0) if translation_seconds else None,
        "peak_vram_gb": peak_vram,
        "device": device,
    }


def _cache_for(model_key: str, context: dict[str, Any]) -> TranslationCache:
    info = context["lock"]["models"][model_key]
    return TranslationCache(
        context["workdir"] / f"cache_{model_key}.sqlite3",
        model_key,
        info["model_id"],
        info["revision"],
        context["lock"]["config_id"],
    )


def _candidate_for(cache: TranslationCache, title: str) -> dict[str, Any] | None:
    return cache.get_success(title)


def _evaluate_titles(
    context: dict[str, Any],
    titles: list[str],
    *,
    include_rows: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    frame = context["frame"]
    gemma_cache = _cache_for("translategemma", context)
    qwen_cache = _cache_for("qwen3", context)
    gemma_results = gemma_cache.get_many(titles)
    qwen_results = qwen_cache.get_many(titles)
    groups = frame[frame["title_zh"].isin(titles)].groupby("title_zh", sort=False)
    records = []
    for title in titles:
        gemma = gemma_results.get(title)
        qwen = qwen_results.get(title)
        if gemma and gemma["status"] != "ok":
            gemma = None
        if qwen and qwen["status"] != "ok":
            qwen = None
        if gemma:
            check_g = check_translation(title, gemma["output_text"], generation_truncated=gemma["truncated"])
        else:
            check_g = {"status": "fail", "hard_flags": ["generation_missing"], "advisory_flags": []}
        if qwen:
            check_q = check_translation(title, qwen["output_text"], generation_truncated=qwen["truncated"])
        else:
            check_q = {"status": "fail", "hard_flags": ["generation_missing"], "advisory_flags": []}
        if title not in groups.indices:
            raise ValueError("Sample manifest contains a title absent from the source parquet.")
        group = groups.get_group(title)
        selected = select_translation(
            title,
            group.iloc[0]["title_vi"],
            group.iloc[0]["title_vi_cleaned"],
            gemma["output_text"] if gemma else None,
            qwen["output_text"] if qwen else None,
            translategemma_truncated=bool(gemma and gemma["truncated"]),
            qwen3_truncated=bool(qwen and qwen["truncated"]),
        )
        records.append(
            {
                "title_zh": title,
                "category": str(group.iloc[0]["category"]),
                "translategemma": gemma["output_text"] if gemma else None,
                "qwen3": qwen["output_text"] if qwen else None,
                "translategemma_complete": bool(gemma and not gemma["truncated"] and gemma["output_text"]),
                "qwen3_complete": bool(qwen and not qwen["truncated"] and qwen["output_text"]),
                "translategemma_truncated": bool(gemma and gemma["truncated"]),
                "qwen3_truncated": bool(qwen and qwen["truncated"]),
                "translategemma_status": check_g["status"],
                "qwen3_status": check_q["status"],
                "translategemma_hard_flags": check_g["hard_flags"],
                "qwen3_hard_flags": check_q["hard_flags"],
                "translategemma_advisory_flags": check_g.get("advisory_flags", []),
                "qwen3_advisory_flags": check_q.get("advisory_flags", []),
                "candidate_agreement": selected["candidate_agreement"],
                "selected": selected["selected"],
                "selection_source": selected["source"],
                "selection_decision": selected["decision"],
                "selection_status": selected["qa_status"],
                "selection_checks": selected["checks"],
                "semantic_verified": False,
            }
        )
    audit = pd.DataFrame.from_records(records)
    report: dict[str, Any] = {
        "unique_titles": len(titles),
        "models": {},
        "selected_hold_count": int((audit["selection_status"] == "hold").sum()),
        "selected_source_counts": {str(key): int(value) for key, value in audit["selection_source"].value_counts().items()},
    }
    for model_key in MODEL_SPECS:
        status_column = f"{model_key}_status"
        flags_column = f"{model_key}_hard_flags"
        passed = audit[status_column].eq("pass")
        report["models"][model_key] = {
            "complete_outputs": int(audit[f"{model_key}_complete"].sum()),
            "complete_rate": float(audit[f"{model_key}_complete"].mean()) if len(audit) else 0.0,
            "technical_pass": int(passed.sum()),
            "technical_pass_rate": float(passed.mean()) if len(audit) else 0.0,
            "hard_flag_counts": {
                key: int(sum(key in flags for flags in audit[flags_column].tolist()))
                for key in sorted({flag for flags in audit[flags_column].tolist() for flag in flags})
            },
        }
    if include_rows:
        return report, audit
    return report, None


def _representative_titles(samples: dict[str, Any], mode: str) -> list[str]:
    if mode == "pilot":
        return samples["pilot"]["representative"]
    if mode == "validate":
        return samples["validate"]["representative"]
    return []


def _write_audit(audit: pd.DataFrame, path: Path) -> None:
    output = audit.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (list, dict))).any():
            output[column] = output[column].map(stable_json)
    output.to_parquet(path, index=False, compression="zstd")


def _compute_disagreement_metrics(
    audit: pd.DataFrame,
    lock: dict[str, Any],
    *,
    device_index: int = 0,
    batch_size: int = 16,
) -> pd.DataFrame:
    valid = audit.dropna(subset=["translategemma", "qwen3"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["title_zh", "chrf_gemma_qwen", "bge_m3_cosine"])
    try:
        from sacrebleu.metrics import CHRF
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install sacrebleu and transformers to compute disagreement metrics.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("BGE-M3 disagreement metrics require CUDA in this notebook.")
    device_index = min(device_index, torch.cuda.device_count() - 1)
    device = f"cuda:{device_index}"
    model_info = lock["models"]["bge_m3"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["model_id"], revision=model_info["revision"]
    )
    model = AutoModel.from_pretrained(
        model_info["model_id"],
        revision=model_info["revision"],
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    chrf_metric = CHRF()
    result_rows = []
    texts = []
    pairs = []
    for row in valid.itertuples(index=False):
        pairs.append((row.title_zh, str(row.translategemma), str(row.qwen3), len(texts), len(texts) + 1))
        texts.extend([str(row.translategemma), str(row.qwen3)])
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]
            embedding = functional.normalize(embedding.float(), p=2, dim=1)
        vectors.append(embedding.cpu())
    matrix = torch.cat(vectors, dim=0)
    for title, left, right, left_index, right_index in pairs:
        chrf_score = chrf_metric.sentence_score(left, [right]).score
        cosine = float(torch.dot(matrix[left_index], matrix[right_index]).item())
        result_rows.append(
            {
                "title_zh": title,
                "chrf_gemma_qwen": float(chrf_score),
                "bge_m3_cosine": cosine,
            }
        )
    del model, tokenizer
    torch.cuda.empty_cache()
    return pd.DataFrame.from_records(result_rows)


def _merge_metrics(audit: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        result = audit.copy()
        result["chrf_gemma_qwen"] = pd.NA
        result["bge_m3_cosine"] = pd.NA
        return result
    return audit.merge(metrics, on="title_zh", how="left", validate="one_to_one")


def _gate_report(
    mode: str,
    report: dict[str, Any],
    audit: pd.DataFrame,
    samples: dict[str, Any],
) -> dict[str, Any]:
    result = dict(report)
    if mode == "smoke":
        completion = min(
            report["models"]["translategemma"]["complete_rate"],
            report["models"]["qwen3"]["complete_rate"],
        )
        result["passed"] = completion >= PILOT_MIN_COMPLETION_RATE and audit["selection_status"].notna().all()
        result["gate_reason"] = "smoke completion and cache path"
        return result
    representative = _representative_titles(samples, mode)
    challenge = (
        samples["pilot"]["challenge"] if mode == "pilot" else samples["validate"]["challenge"]
    )
    representative_audit = audit[audit["title_zh"].isin(representative)]
    challenge_audit = audit[audit["title_zh"].isin(challenge)]
    representative_rates = {}
    challenge_rates = {}
    for model_key in MODEL_SPECS:
        status = representative_audit[f"{model_key}_status"].eq("pass")
        representative_rates[model_key] = float(status.mean()) if len(status) else 0.0
        challenge_status = challenge_audit[f"{model_key}_status"].eq("pass")
        challenge_rates[model_key] = float(challenge_status.mean()) if len(challenge_status) else 0.0
    complete_rates = {key: value["complete_rate"] for key, value in report["models"].items()}
    output_ok = bool(
        ((audit["selection_status"] == "pass") == audit["selected"].notna()).all()
    )
    selected_defects = 0
    for record in audit.to_dict("records"):
        if record["selected"] is None:
            continue
        checks = record["selection_checks"]
        check_key = {
            "title_vi_cleaned": "baseline_cleaned",
            "title_vi": "baseline_raw",
            "translategemma_agreement": "translategemma",
        }.get(record["selection_source"])
        selected_check = checks.get(check_key, {}) if check_key else {}
        if selected_check.get("status") != "pass":
            selected_defects += 1
    result["representative_technical_pass_rate"] = representative_rates
    result["challenge_technical_pass_rate"] = challenge_rates
    result["selected_detected_defects"] = selected_defects
    result["passed"] = (
        min(complete_rates.values(), default=0.0) >= PILOT_MIN_COMPLETION_RATE
        and min(representative_rates.values(), default=0.0) >= PILOT_MIN_REPRESENTATIVE_QA_RATE
        and selected_defects == 0
        and output_ok
    )
    result["gate_reason"] = (
        "99% complete/model; 95% technical pass on representative titles/model; "
        "no selected candidate has a detected technical defect"
    )
    return result


def _write_stage_report(
    mode: str,
    result: dict[str, Any],
    audit: pd.DataFrame,
    context: dict[str, Any],
) -> None:
    stage_path = context["workdir"] / f"{mode}_report.json"
    atomic_write_json(stage_path, result)
    if mode in {"pilot", "validate"}:
        _write_audit(audit, context["workdir"] / f"{mode}_audit.parquet")
    lines = [
        "# Báo cáo pipeline dịch title Cado",
        "",
        f"- Giai đoạn: {mode}",
        f"- Thời điểm UTC: {utc_now()}",
        f"- Kết quả cổng: {'PASS' if result.get('passed') else 'NOT PASS'}",
        f"- Config ID: {context['lock']['config_id']}",
        f"- Số title nguồn: {result.get('unique_titles')}",
        f"- Dữ liệu: {DATASET_ID} / revision {DATASET_REVISION}",
        "",
        "## Kết quả theo model",
        "",
        "| Model | Đầu ra hoàn chỉnh | Tỷ lệ hoàn chỉnh | QA kỹ thuật đạt | QA đạt trên mẫu đại diện |",
        "|---|---:|---:|---:|---:|",
    ]
    representative = result.get("representative_technical_pass_rate", {})
    for key, stats in result.get("models", {}).items():
        lines.append(
            f"| {key} | {stats.get('complete_outputs')} | {stats.get('complete_rate', 0):.1%} | "
            f"{stats.get('technical_pass')} | {representative.get(key, float('nan')):.1%} |"
        )
    lines += ["", "## Ví dụ title để kiểm tra", ""]
    source_frame = context["frame"].drop_duplicates("title_zh").set_index("title_zh")
    example_specs = [
        ("Có số", lambda text: any(char.isdigit() for char in text)),
        ("Có mã model", lambda text: bool(_model_codes(text))),
        ("Có đơn vị 斤", lambda text: "斤" in text),
        ("Có tiếng Anh", lambda text: any(char.isascii() and char.isalpha() for char in text)),
    ]
    used_titles: set[str] = set()
    examples_written = 0
    for label, predicate in example_specs:
        matches = audit[audit["title_zh"].map(predicate)]
        if matches.empty:
            lines.append(f"- {label}: không có title phù hợp trong mẫu của giai đoạn này.")
            continue
        row = next(
            (item for item in matches.to_dict("records") if item["title_zh"] not in used_titles),
            matches.iloc[0].to_dict(),
        )
        title = str(row["title_zh"])
        used_titles.add(title)
        original = source_frame.loc[title]
        selected = row.get("selected")
        selected_text = "HOLD (chưa tự chọn)" if selected is None or pd.isna(selected) else str(selected)
        lines += [
            f"### {label}",
            f"- Nguồn ZH: {title[:240]}",
            f"- VI hiện có: {str(original.get('title_vi_cleaned', ''))[:240]}",
            f"- TranslateGemma: {str(row.get('translategemma') or '')[:240]}",
            f"- Qwen3: {str(row.get('qwen3') or '')[:240]}",
            f"- Quyết định: {row.get('selection_decision', 'chưa có')} — {selected_text[:240]}",
            f"- Hai model đồng ý: {row.get('candidate_agreement', False)}",
            "",
        ]
        examples_written += 1
    lines += [
        "Ví dụ chỉ cho thấy dữ liệu và quyết định tự động; không phải xác nhận ngữ nghĩa.",
        "",
    ]
    lines += [
        "",
        "ChrF và BGE-M3 chỉ phản ánh mức khác nhau giữa các đầu ra; không dùng để xác nhận dịch đúng nghĩa.",
        "semantic_verified luôn là false vì đợt này chỉ có kiểm tra tự động.",
        "",
    ]
    atomic_write_text(context["workdir"] / "title_translation_report.md", "\n".join(lines))


def _persist_state(context: dict[str, Any]) -> None:
    atomic_write_json(context["workdir"] / "run_state.json", context["state"])


def _stage_titles(mode: str, samples: dict[str, Any]) -> list[str]:
    if mode == "smoke":
        return list(samples["smoke"])
    if mode in {"pilot", "validate"}:
        return list(samples[mode]["titles"])
    return []


def _verify_stage_prerequisites(mode: str, state: dict[str, Any], config_id: str) -> None:
    if state.get("config_id") != config_id:
        raise RuntimeError("Stage config fingerprint mismatch.")
    if mode == "pilot" and not state.get("stages", {}).get("smoke", {}).get("passed"):
        raise RuntimeError("Pilot is blocked until smoke passes under this config.")
    if mode == "validate" and not state.get("stages", {}).get("pilot", {}).get("passed"):
        raise RuntimeError("Validation is blocked until pilot passes under this config.")
    if mode == "full":
        validation = state.get("stages", {}).get("validate", {})
        if not validation.get("passed") or validation.get("config_id") != config_id:
            raise RuntimeError("Full run is blocked until validation passes with the exact current config.")
        prior_full = state.get("stages", {}).get("full", {})
        if prior_full.get("status") == "stopped_quality_gate":
            raise RuntimeError(
                "Full previously stopped on a quality gate. Review the cause, change the approved config, "
                "and rerun validation before continuing."
            )


def _run_model_pair_for_titles(
    titles: list[str],
    context: dict[str, Any],
    token: str | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Select Kaggle GPU T4 x2.")
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        errors = []
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    _run_model_titles,
                    model_key=key,
                    titles=titles,
                    lock=context["lock"],
                    workdir=context["workdir"],
                    token=token,
                    device_index=MODEL_SPECS[key]["device_preference"],
                    batch_size=batch_size,
                )
                for key in MODEL_SPECS
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(exc)
        if not errors:
            return results
        print(
            "Parallel model loading failed; retrying uncached work sequentially: "
            + "; ".join(f"{type(error).__name__}: {error}" for error in errors),
            flush=True,
        )
        return [
            _run_model_titles(
                model_key=key,
                titles=titles,
                lock=context["lock"],
                workdir=context["workdir"],
                token=token,
                device_index=0,
                batch_size=batch_size,
            )
            for key in MODEL_SPECS
        ]
    # A single GPU is still supported for small pilots, with models loaded in
    # sequence so the two 4-bit weights never compete for the same VRAM.
    return [
        _run_model_titles(
            model_key=key,
            titles=titles,
            lock=context["lock"],
            workdir=context["workdir"],
            token=token,
            device_index=0,
            batch_size=batch_size,
        )
        for key in MODEL_SPECS
    ]


def _full_window_gate(audit: pd.DataFrame, start: int, end: int) -> dict[str, Any]:
    window = audit.iloc[start:end]
    checks = {}
    for model_key in MODEL_SPECS:
        complete = window[f"{model_key}_complete"]
        qa_pass = window[f"{model_key}_status"].eq("pass")
        incomplete_count = int((~complete).sum())
        failure_count = int((~qa_pass).sum())
        incomplete_rate = incomplete_count / len(window) if len(window) else 0.0
        failure_rate = failure_count / len(window) if len(window) else 0.0
        max_incomplete = math.floor(len(window) * FULL_MAX_INCOMPLETE_RATE + 1e-9)
        max_qa_failures = math.floor(len(window) * FULL_MAX_QA_FAILURE_RATE + 1e-9)
        checks[model_key] = {
            "incomplete_count": incomplete_count,
            "incomplete_rate": incomplete_rate,
            "technical_qa_failure_count": failure_count,
            "technical_qa_failure_rate": failure_rate,
            "passed": incomplete_count <= max_incomplete and failure_count <= max_qa_failures,
        }
    return {"start": start, "end": end, "models": checks, "passed": all(item["passed"] for item in checks.values())}


def _build_full_output(context: dict[str, Any], metrics: pd.DataFrame | None) -> pd.DataFrame:
    frame = context["frame"].copy()
    if list(frame.columns) != ORIGINAL_COLUMNS:
        # Some parquet readers may expose additional fields; retain them, and
        # validate that the original schema remains an exact prefix.
        if not set(ORIGINAL_COLUMNS).issubset(frame.columns):
            raise ValueError("Input columns changed after validation.")
    gemma_cache = _cache_for("translategemma", context)
    qwen_cache = _cache_for("qwen3", context)
    titles = frame["title_zh"].drop_duplicates().tolist()
    gemma_results = gemma_cache.get_many(titles)
    qwen_results = qwen_cache.get_many(titles)
    title_rows: dict[str, dict[str, Any]] = {}
    for title in titles:
        gemma = gemma_results.get(title)
        qwen = qwen_results.get(title)
        if gemma and gemma["status"] != "ok":
            gemma = None
        if qwen and qwen["status"] != "ok":
            qwen = None
        title_rows[title] = {
            "translategemma": gemma,
            "qwen3": qwen,
        }

    gemma_values = []
    qwen_values = []
    selected_values = []
    source_values = []
    decision_values = []
    status_values = []
    flags_values = []
    agreement_values = []
    semantic_values = []
    config_values = []
    for row in frame.itertuples(index=False):
        title = str(getattr(row, "title_zh"))
        entries = title_rows[title]
        gemma = entries["translategemma"]
        qwen = entries["qwen3"]
        gtext = gemma["output_text"] if gemma else None
        qtext = qwen["output_text"] if qwen else None
        choice = select_translation(
            title,
            getattr(row, "title_vi"),
            getattr(row, "title_vi_cleaned"),
            gtext,
            qtext,
            translategemma_truncated=bool(gemma and gemma["truncated"]),
            qwen3_truncated=bool(qwen and qwen["truncated"]),
        )
        checks = choice["checks"]
        flags = {
            key: {
                "hard": value.get("hard_flags", []),
                "advisory": value.get("advisory_flags", []),
            }
            for key, value in checks.items()
        }
        gemma_values.append(gtext)
        qwen_values.append(qtext)
        selected_values.append(choice["selected"])
        source_values.append(choice["source"])
        decision_values.append(choice["decision"])
        status_values.append(choice["qa_status"])
        flags_values.append(stable_json(flags))
        agreement_values.append(choice["candidate_agreement"])
        semantic_values.append(False)
        config_values.append(context["lock"]["config_id"])
    frame["title_vi_candidate_translategemma"] = gemma_values
    frame["title_vi_candidate_qwen3"] = qwen_values
    frame["title_vi_selected"] = selected_values
    frame["title_vi_selection_source"] = source_values
    frame["title_vi_selection_decision"] = decision_values
    frame["title_vi_qa_status"] = status_values
    frame["title_vi_qa_flags"] = flags_values
    frame["title_vi_candidate_agreement"] = agreement_values
    frame["title_vi_semantic_verified"] = semantic_values
    frame["title_vi_pipeline_config_id"] = config_values

    if metrics is None or metrics.empty:
        frame["title_vi_chrf"] = pd.NA
        frame["title_vi_bge_m3_cosine"] = pd.NA
    else:
        metric_by_title = metrics.set_index("title_zh")
        frame["title_vi_chrf"] = frame["title_zh"].map(metric_by_title["chrf_gemma_qwen"])
        frame["title_vi_bge_m3_cosine"] = frame["title_zh"].map(metric_by_title["bge_m3_cosine"])
    expected_output_columns = ORIGINAL_COLUMNS + OUTPUT_COLUMNS
    frame = frame[expected_output_columns]
    if list(frame.columns) != expected_output_columns:
        raise AssertionError("Output columns do not match originals plus planned title fields.")
    if len(frame) != EXPECTED_ROWS or not frame["product_id"].is_unique:
        raise AssertionError("Output row count or product_id uniqueness changed.")
    if int(frame["title_vi_semantic_verified"].sum()) != 0:
        raise AssertionError("This automatic-only stage must never mark semantic verification true.")
    return frame


def _write_full_output(context: dict[str, Any], metrics: pd.DataFrame | None) -> dict[str, Any]:
    output = _build_full_output(context, metrics)
    source = context["frame"]
    for column in ORIGINAL_COLUMNS:
        if not _series_values_equal(
            source[column].reset_index(drop=True), output[column].reset_index(drop=True)
        ):
            raise AssertionError(f"Original column changed during join: {column}")
    path = context["workdir"] / "bilingual_zh_vi_cado_title_v2.parquet"
    temp_path = path.with_suffix(".parquet.tmp")
    output.to_parquet(temp_path, index=False, compression="zstd")
    check = pd.read_parquet(temp_path)
    if len(check) != len(source) or check["product_id"].tolist() != source["product_id"].tolist():
        raise AssertionError("Parquet round-trip changed row count, product ids, or row order.")
    for column in ORIGINAL_COLUMNS:
        if not _series_values_equal(
            source[column].reset_index(drop=True), check[column].reset_index(drop=True)
        ):
            raise AssertionError(f"Parquet round-trip changed original column: {column}")
    os.replace(temp_path, path)
    return {
        "path": str(path),
        "rows": len(output),
        "unique_title_sources": int(output["title_zh"].nunique()),
        "selected": int(output["title_vi_selected"].notna().sum()),
        "held": int(output["title_vi_selection_source"].eq("hold").sum()),
        "source_counts": {str(key): int(value) for key, value in output["title_vi_selection_source"].value_counts().items()},
        "sha256": sha256_file(path),
    }


def _save_metrics(metrics: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(".parquet.tmp")
    metrics.to_parquet(temp_path, index=False, compression="zstd")
    os.replace(temp_path, path)


def run_stage(
    context: dict[str, Any],
    *,
    token: str | None = None,
    batch_size: int = GENERATION_BATCH_SIZE,
    full_titles_per_session: int = FULL_TITLES_PER_SESSION,
) -> dict[str, Any]:
    mode = context["mode"]
    lock = context["lock"]
    workdir = context["workdir"]
    state = context["state"]
    _verify_stage_prerequisites(mode, state, lock["config_id"])
    start_time = time.perf_counter()
    started_at = utc_now()
    state["active_stage"] = {"mode": mode, "started_at_utc": started_at}
    _persist_state(context)

    if mode == "full":
        all_titles = context["frame"]["title_zh"].drop_duplicates().tolist()
        window_reports = []
        model_runtime = []
        audit_parts = []
        completed_before = 0
        gemma_cache = _cache_for("translategemma", context)
        qwen_cache = _cache_for("qwen3", context)
        gemma_cache_rows = gemma_cache.get_many(all_titles)
        qwen_cache_rows = qwen_cache.get_many(all_titles)
        pending_titles = [
            title
            for title in all_titles
            if not _cached_complete(gemma_cache_rows.get(title))
            or not _cached_complete(qwen_cache_rows.get(title))
        ]
        session_titles = pending_titles[: max(1, int(full_titles_per_session))]
        window_start = 0
        for offset in range(0, len(session_titles), FULL_WINDOW_SIZE):
            window_titles = session_titles[offset : offset + FULL_WINDOW_SIZE]
            runtimes = _run_model_pair_for_titles(window_titles, context, token, batch_size)
            model_runtime.extend(runtimes)
            _, window_audit = _evaluate_titles(context, window_titles, include_rows=True)
            audit_parts.append(window_audit)
            completed_before += len(window_titles)
            window_report = _full_window_gate(window_audit, 0, len(window_audit))
            window_report["title_start"] = window_start + offset
            window_report["title_end"] = window_start + offset + len(window_titles)
            window_reports.append(window_report)
            if completed_before % SHARD_SIZE < FULL_WINDOW_SIZE or offset + len(window_titles) == len(session_titles):
                print(
                    f"[full] checkpointed {completed_before}/{len(session_titles)} titles this session; "
                    f"last window {'PASS' if window_report['passed'] else 'STOP'}",
                    flush=True,
                )
            _persist_state(context)
            if not window_report["passed"]:
                result = {
                    "mode": mode,
                    "passed": False,
                    "status": "stopped_quality_gate",
                    "config_id": lock["config_id"],
                    "unique_titles": completed_before,
                    "model_runtime": model_runtime,
                    "window_gates": window_reports,
                    "reason": "A 200-title window exceeded the agreed technical error or QA-failure threshold.",
                    "checkpoint_preserved": True,
                }
                state["stages"]["full"] = result
                state["active_stage"] = None
                _persist_state(context)
                atomic_write_json(workdir / "full_report.json", result)
                return result
        remaining = len(pending_titles) - len(session_titles)
        if remaining > 0:
            partial_audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
            _, overall_partial = _evaluate_titles(context, session_titles, include_rows=True)
            result = {
                "mode": mode,
                "passed": False,
                "status": "checkpoint_saved_continue",
                "config_id": lock["config_id"],
                "unique_titles": len(session_titles),
                "processed_this_session": len(session_titles),
                "remaining_unique_titles": remaining,
                "completed_unique_titles": len(all_titles) - remaining,
                "total_unique_titles": len(all_titles),
                "models": overall_partial["models"],
                "selected_hold_count": overall_partial["selected_hold_count"],
                "model_runtime": model_runtime,
                "window_gates": window_reports,
                "session_report": overall_partial,
                "checkpoint_preserved": True,
                "next_step": "Save this run as a Kaggle version, attach its output as input, then rerun full.",
            }
            state["full_progress"] = result
            state["active_stage"] = None
            _persist_state(context)
            _write_stage_report(mode, result, partial_audit, context)
            return result
        _, audit_by_title = _evaluate_titles(context, all_titles, include_rows=True)
        audit = audit_by_title
    else:
        titles = _stage_titles(mode, context["samples"])
        model_runtime = _run_model_pair_for_titles(titles, context, token, batch_size)
        evaluation, audit_by_title = _evaluate_titles(context, titles, include_rows=True)
        audit = audit_by_title
        result = _gate_report(mode, evaluation, audit, context["samples"])
        result["mode"] = mode
        result["config_id"] = lock["config_id"]
        result["model_runtime"] = model_runtime
        result["started_at_utc"] = started_at
        result["elapsed_seconds"] = time.perf_counter() - start_time
        if mode == "smoke":
            # Confirm a newly reopened cache reads the finished smoke entries,
            # which is the restart/resume check without generating them again.
            resume_counts = {
                key: sum(
                    _cached_complete(value)
                    for value in _cache_for(key, context).get_many(titles).values()
                )
                for key in MODEL_SPECS
            }
            result["checkpoint_resume_counts"] = resume_counts
            result["passed"] = bool(
                result["passed"] and all(count == len(titles) for count in resume_counts.values())
            )
        if mode in {"pilot", "validate"}:
            representative = set(_representative_titles(context["samples"], mode))
            representative_audit = audit[audit["title_zh"].isin(representative)].copy()
            try:
                metrics = _compute_disagreement_metrics(representative_audit, lock)
                _save_metrics(metrics, workdir / f"{mode}_disagreement_metrics.parquet")
                audit = _merge_metrics(audit, metrics)
                result["disagreement_metrics"] = {
                    "computed_titles": len(metrics),
                    "chrf_mean": float(metrics["chrf_gemma_qwen"].mean()) if len(metrics) else None,
                    "bge_m3_cosine_mean": float(metrics["bge_m3_cosine"].mean()) if len(metrics) else None,
                    "use": "disagreement signal only; not a correctness label",
                }
            except Exception as exc:
                result["disagreement_metrics"] = {
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                    "required_for_selection": False,
                }
        result["started_at_utc"] = started_at
        result["elapsed_seconds"] = time.perf_counter() - start_time
        _write_stage_report(mode, result, audit, context)
        state["stages"][mode] = result
        state["active_stage"] = None
        _persist_state(context)
        return result

    # Full stage quality gates passed for every 200-title window.
    evaluation, audit = _evaluate_titles(context, all_titles, include_rows=True)
    result = {
        "mode": mode,
        "passed": True,
        "status": "all_windows_passed",
        "config_id": lock["config_id"],
        "unique_titles": len(all_titles),
        "model_runtime": model_runtime,
        "window_gates": window_reports,
        "started_at_utc": started_at,
        "elapsed_seconds": time.perf_counter() - start_time,
        "candidate_qa": evaluation["models"],
    }
    metrics = None
    try:
        metrics = _compute_disagreement_metrics(audit, lock)
        _save_metrics(metrics, workdir / "disagreement_metrics.parquet")
        audit = _merge_metrics(audit, metrics)
        result["disagreement_metrics"] = {
            "computed_titles": len(metrics),
            "chrf_mean": float(metrics["chrf_gemma_qwen"].mean()) if len(metrics) else None,
            "bge_m3_cosine_mean": float(metrics["bge_m3_cosine"].mean()) if len(metrics) else None,
            "use": "disagreement signal only; not a correctness label",
        }
    except Exception as exc:
        result["disagreement_metrics"] = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "required_for_selection": False,
        }
    output_report = _write_full_output(context, metrics)
    result["output"] = output_report
    result["semantic_verified"] = False
    result["started_at_utc"] = started_at
    atomic_write_json(workdir / "full_report.json", result)
    _write_stage_report(mode, result, audit, context)
    state["stages"]["full"] = result
    state["active_stage"] = None
    _persist_state(context)
    return result


def environment_summary() -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "packages": _runtime_package_versions(),
    }
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu_count"] = int(torch.cuda.device_count())
        result["gpus"] = (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        result["cuda_runtime"] = torch.version.cuda
    except ImportError:
        result["cuda_available"] = False
        result["gpu_count"] = 0
        result["gpus"] = []
    return result
