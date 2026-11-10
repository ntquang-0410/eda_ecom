# -*- coding: utf-8 -*-
"""Tiki Phase 1: profile ``title_vi`` + description key-value, build BT units.

Module này **không gọi model**. Nó chỉ:

* parse ``attributes_vi`` (fallback ``description_vi``) thành các cặp KV;
* chuẩn hoá key, phân loại key/value thành ``admin | model | numeric | text``;
* tạo pair-level units cho ``title`` và từng ``attribute value``;
* tạo product-level table chỉ chứa title + description KV (không có prose).

Các bước dịch/back-translation, key map VI→ZH và QC nằm ở Phase 2.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from .config import TIKI_KEY_MAP_CSV
from .text import normalize_text
from .tiki import clean_text, parse_attributes

TIKI_SOURCE = "tiki"
TIKI_ZH_ORIGIN = "synthetic_tiki"

# ═══════════════════════════════════════════════════════════════════
# KEY NORMALIZATION / CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

_KEY_ALIASES = {
    "màu": "màu sắc",
    "mau": "màu sắc",
    "hạn dùng": "hạn sử dụng",
    "han dung": "hạn sử dụng",
}

_ADMIN_MARKERS = (
    "thương hiệu",
    "xuất xứ",
    "bảo hành",
    "hạn sử dụng",
    "hạn dùng",
    "ngày sản xuất",
    "ngày hết hạn",
    "tổ chức chịu trách nhiệm",
    "địa chỉ",
    "made in",
    "nhà sản xuất",
    "nơi sản xuất",
    "quốc gia",
    "chứng nhận",
    "tiêu chuẩn",
    "mã vạch",
    "barcode",
    "số lô",
    "lô sản xuất",
    "khuyến cáo",
    "cảnh báo",
)

_MODEL_MARKERS = (
    "model",
    "sku",
    "mã sản phẩm",
    "mã hàng",
    "mã số",
    "số mặt hàng",
    "part number",
    "ký hiệu",
    "kí hiệu",
)

_NUMERIC_MARKERS = (
    "trọng lượng",
    "khối lượng",
    "kích thước",
    "kích cỡ",
    "dung tích",
    "công suất",
    "điện áp",
    "chiều dài",
    "chiều rộng",
    "chiều cao",
    "độ dày",
    "số lượng",
    "quy cách",
    "thể tích",
    "nhiệt độ",
    "tần số",
    "định lượng",
    "hàm lượng",
    "nồng độ",
    "đường kính",
    "diện tích",
    "độ tuổi",
    "tuổi thọ",
    "số trang",
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:%|cm|mm|km|kg|mg|ml|m2|m3|mah|hz|kw|w|v|g|l|m|inch|ft|pcs)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+")
_NUMERIC_UNIT_RE = re.compile(
    r"(?<![A-Za-zÀ-ỹĐđ])"
    r"(?:%|cm|mm|km|kg|mg|ml|m2|m3|mah|hz|kw|w|v|g|l|m|inch|ft|pcs|h)"
    r"(?![A-Za-zÀ-ỹĐđ])",
    re.IGNORECASE,
)
_NUMERIC_ALLOWED_WORDS = {
    "size", "inch", "cm", "mm", "km", "kg", "mg", "ml", "m2", "m3",
    "mah", "hz", "kw", "w", "v", "g", "l", "m", "ft", "pcs", "h",
    "gr", "gram", "gam", "lít", "kilogam", "x", "set", "bộ", "chiếc", "cái",
    "đôi", "viên", "gói", "chai", "hũ", "lon", "miếng", "cuộn", "tờ",
    "quyển", "cặp", "túi", "hộp",
}


def normalize_key(name: object) -> str:
    """Chuẩn hoá key về dạng so khớp: NFC, lowercase, gộp khoảng trắng."""
    text = normalize_text(name, collapse_whitespace=True).strip().lower()
    text = text.rstrip(":：").strip()
    text = re.sub(r"\s+", " ", text)
    return _KEY_ALIASES.get(text, text)


def classify_key(name: object) -> str:
    """Phân loại key theo cách xử lý dự kiến.

    Thứ tự ưu tiên: ``admin`` → ``model`` → ``numeric`` → ``text``.
    Đây là heuristic theo tên key, không phải phân loại ngữ nghĩa hoàn hảo.
    """
    key = normalize_key(name)
    if not key:
        return "text"
    if any(marker in key for marker in _ADMIN_MARKERS):
        return "admin"
    if any(marker in key for marker in _MODEL_MARKERS):
        return "model"
    if any(marker in key for marker in _NUMERIC_MARKERS):
        return "numeric"
    return "text"


def bt_method_for_class(value_class: str) -> str:
    """Cách xử lý dự kiến ở Phase 2."""
    return {
        "admin": "dictionary",
        "model": "preserve",
        "numeric": "rule",
        "text": "mt",
    }.get(value_class, "mt")


def _looks_numeric_value(value: str) -> bool:
    """True nếu value chủ yếu là số/đơn vị, không phải câu cần dịch."""
    text = value.strip()
    if not text or not _NUMBER_RE.search(text):
        return False

    stripped = _NUMBER_RE.sub(" ", text)
    stripped = _NUMERIC_UNIT_RE.sub(" ", stripped)
    stripped = re.sub(r"[\s.,;:/x×*+\-–—<>~()\[\]]+", " ", stripped, flags=re.IGNORECASE)
    tokens = [token.lower() for token in _LATIN_LETTER_RE.findall(stripped)]
    if not tokens:
        return True
    return all(token in _NUMERIC_ALLOWED_WORDS for token in tokens)


def classify_value(value: str, key_class: str) -> str:
    """Phân loại value theo cách xử lý thực tế.

    Key ``numeric`` nhưng value là câu mô tả (``Nhiều size``, ``Dùng cho...``)
    được chuyển về ``text`` để không bỏ sót MT. Ngược lại, value chỉ gồm
    số/đơn vị dưới key ``text`` được coi là ``numeric``.
    """
    if key_class in {"admin", "model"}:
        return key_class
    numeric = _looks_numeric_value(value)
    if key_class == "numeric":
        return "numeric" if numeric else "text"
    if numeric:
        return "numeric"
    return "text"


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _as_list(value: Any) -> list[Any]:
    """Chuyển list/tuple/ndarray/dict/scalar về list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return []


def _structured_has_values(value: Any) -> bool:
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        if str(item.get("name", "")).strip() and _as_list(item.get("values")):
            return True
    return False


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    raw = value if isinstance(value, str) else str(value)
    cleaned, _ = clean_text(raw)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_attributes(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Trả về ``(attributes, parse_source)``.

    ``attributes_vi`` là nguồn chính. Nếu không có structured attributes,
    fallback parse ``description_vi``. Không đọc prose.
    """
    structured = row.get("attributes_vi")
    source = "attributes_vi" if _structured_has_values(structured) else "description_vi"

    attributes = parse_attributes(structured, row.get("description_vi"))
    if not attributes:
        return [], "none"

    cleaned: list[dict[str, Any]] = []
    for item in attributes:
        name = normalize_text(item.get("name", ""), collapse_whitespace=True).strip()
        values = [
            _clean_value(value)
            for value in _as_list(item.get("values"))
        ]
        values = [value for value in values if value]
        if name and values:
            cleaned.append({"name": name, "values": values})

    return cleaned, source if cleaned else "none"


def serialize_attributes(attributes: Iterable[Mapping[str, Any]]) -> str:
    """Serialize attributes thành ``key: value; ...`` theo format Tiki/1688."""
    pairs: list[str] = []
    for attribute in attributes:
        name = str(attribute.get("name", "")).strip()
        values = [
            str(value).replace(";", ",").strip()
            for value in _as_list(attribute.get("values"))
            if str(value).strip()
        ]
        if name and values:
            pairs.append(f"{name}: {', '.join(values)}")
    return "; ".join(pairs)


def _text_key(text: object) -> str:
    normalized = normalize_text(text, collapse_whitespace=True)
    return re.sub(r"[^a-zà-ỹđ0-9]+", " ", normalized.lower()).strip()


def split_for_product(product_id: str, title: str) -> str:
    """Chia train/dev/test 80/10/10 ổn định theo product_id + title."""
    payload = f"{product_id}|{title}".encode("utf-8")
    bucket = int(hashlib.md5(payload).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "dev"
    return "test"


def load_key_map(path: str | Path = TIKI_KEY_MAP_CSV) -> dict[str, str]:
    """Nạp glossary VI→ZH nếu đã có; Phase 1 chưa yêu cầu file này."""
    key_path = Path(path)
    if not key_path.exists():
        return {}

    frame = pd.read_csv(key_path, dtype=str, encoding="utf-8-sig").fillna("")
    key_column = next(
        (column for column in ("key_vi", "key_norm", "canonical_vi") if column in frame.columns),
        None,
    )
    if key_column is None or "key_zh" not in frame.columns:
        raise ValueError(
            f"{key_path} cần có cột key_vi/key_norm/canonical_vi và key_zh."
        )

    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        key = normalize_key(row.get(key_column, ""))
        value = str(row.get("key_zh", "")).strip()
        if key and value and key not in mapping:
            mapping[key] = value
    return mapping


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _describe(values: Iterable[int | float]) -> dict[str, Any]:
    data = list(values)
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0, "max": 0}
    series = pd.Series(data, dtype="float64")
    return {
        "count": len(data),
        "mean": round(float(series.mean()), 2),
        "p50": round(float(series.median()), 2),
        "p95": round(float(series.quantile(0.95)), 2),
        "min": int(series.min()),
        "max": int(series.max()),
    }


def _jsonable(value: Any) -> Any:
    """Chuyển object numpy/pandas sang kiểu JSON thuần."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


# ═══════════════════════════════════════════════════════════════════
# PROFILE
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TikiProfileResult:
    stats: dict[str, Any]
    key_inventory: pd.DataFrame
    parse_errors: pd.DataFrame


def profile_tiki(
    df: pd.DataFrame,
    *,
    progress: Callable[[str], None] | None = None,
) -> TikiProfileResult:
    """Profile title + description KV trên toàn bộ Tiki.

    Không trả về product-level table; chỉ số liệu tổng hợp, key inventory và
    danh sách phần description không parse được.
    """
    required = ("product_id", "title_vi", "description_vi")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột Tiki: {missing}")

    rows = df.to_dict("records")
    title_keys: list[str] = []
    title_words: list[int] = []
    title_chars: list[int] = []
    n_title_numbers = 0
    n_title_with_numbers = 0
    n_title_with_units = 0
    n_title_with_brand = 0
    n_titles_empty = 0

    n_desc_empty = 0
    n_desc_parts = 0
    n_desc_pairs = 0
    parse_errors: list[dict[str, Any]] = []

    n_attr_products = 0
    n_attr_missing = 0
    n_attr_pairs = 0
    n_attr_values = 0
    n_attr_value_words = 0
    n_attr_mismatch = 0
    attr_counts_per_product: list[int] = []
    value_counts_per_product: list[int] = []
    parse_sources: Counter[str] = Counter()
    class_values: Counter[str] = Counter()
    class_words: Counter[str] = Counter()
    key_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n_pairs": 0,
            "n_values": 0,
            "n_products": 0,
            "value_words": 0,
            "max_value_words": 0,
            "sample_values": [],
            "raw_names": Counter(),
        }
    )

    for index, row in enumerate(rows):
        title_raw = normalize_text(row.get("title_vi"))
        title = clean_text(title_raw, title=True)[0]
        words = title.split()
        title_keys.append(_text_key(title))
        title_words.append(len(words))
        title_chars.append(len(title))
        if not title:
            n_titles_empty += 1

        numbers = _NUMBER_RE.findall(title)
        n_title_numbers += len(numbers)
        if numbers:
            n_title_with_numbers += 1
        if _UNIT_RE.search(title):
            n_title_with_units += 1
        brand = normalize_text(row.get("brand")).strip()
        if brand and brand.lower() in title.lower():
            n_title_with_brand += 1

        description_raw = normalize_text(row.get("description_vi"))
        if not description_raw.strip():
            n_desc_empty += 1
        parts = [
            part.strip()
            for part in re.split(r";\s*", description_raw)
            if part.strip()
        ]
        n_desc_parts += len(parts)
        malformed = [part for part in parts if ":" not in part]
        n_desc_pairs += len(parts) - len(malformed)
        for part_index, part in enumerate(parts):
            if ":" not in part:
                parse_errors.append({
                    "product_id": str(row.get("product_id", "")),
                    "part_index": part_index,
                    "part": part,
                    "reason": "missing_colon",
                })

        attributes, source = extract_attributes(row)
        parse_sources[source] += 1
        attr_counts_per_product.append(len(attributes))
        value_count = sum(len(attribute["values"]) for attribute in attributes)
        value_counts_per_product.append(value_count)
        if attributes:
            n_attr_products += 1
        else:
            n_attr_missing += 1
        if source == "description_vi":
            pass  # đã có trong parse_sources

        if "n_attributes_vi" in row and not pd.isna(row.get("n_attributes_vi")):
            try:
                expected = int(row.get("n_attributes_vi"))
            except (TypeError, ValueError):
                expected = None
            if expected is not None and expected != len(attributes):
                n_attr_mismatch += 1

        seen_keys: set[str] = set()
        for attribute in attributes:
            key_norm = normalize_key(attribute["name"])
            key_class = classify_key(key_norm)
            values = attribute["values"]
            words_in_values = sum(len(value.split()) for value in values)
            n_attr_pairs += 1
            n_attr_values += len(values)
            n_attr_value_words += words_in_values
            for value in values:
                value_class = classify_value(value, key_class)
                class_values[value_class] += 1
                class_words[value_class] += len(value.split())
            seen_keys.add(key_norm)

            item = key_stats[key_norm]
            item["n_pairs"] += 1
            item["n_values"] += len(values)
            item["value_words"] += words_in_values
            item["max_value_words"] = max(item["max_value_words"], *(len(value.split()) for value in values))
            item["raw_names"][attribute["name"]] += 1
            for value in values:
                if len(item["sample_values"]) < 5 and value not in item["sample_values"]:
                    item["sample_values"].append(_truncate(value))

        for key_norm in seen_keys:
            key_stats[key_norm]["n_products"] += 1

        if progress and (index + 1) % 2000 == 0:
            progress(f"  profile: {index + 1:,}/{len(rows):,}")

    title_counter = Counter(key for key in title_keys if key)
    duplicate_title_groups = sum(1 for count in title_counter.values() if count > 1)
    duplicate_title_rows = sum(count for count in title_counter.values() if count > 1)
    duplicate_title_extra = duplicate_title_rows - duplicate_title_groups

    key_rows: list[dict[str, Any]] = []
    for key_norm, item in key_stats.items():
        n_values = int(item["n_values"])
        key_rows.append({
            "key_norm": key_norm,
            "key_vi_examples": " | ".join(
                name for name, _ in item["raw_names"].most_common(3)
            ),
            "key_class": classify_key(key_norm),
            "n_pairs": int(item["n_pairs"]),
            "n_values": n_values,
            "n_products": int(item["n_products"]),
            "value_words": int(item["value_words"]),
            "mean_value_words": round(item["value_words"] / n_values, 2) if n_values else 0.0,
            "max_value_words": int(item["max_value_words"]),
            "sample_values": " | ".join(item["sample_values"]),
            "key_zh": "",
            "key_map_status": "pending_phase2",
        })
    key_inventory = pd.DataFrame(
        key_rows,
        columns=[
            "key_norm",
            "key_vi_examples",
            "key_class",
            "n_pairs",
            "n_values",
            "n_products",
            "value_words",
            "mean_value_words",
            "max_value_words",
            "sample_values",
            "key_zh",
            "key_map_status",
        ],
    ).sort_values(
        ["n_pairs", "n_values", "key_norm"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    parse_errors_df = pd.DataFrame(
        parse_errors,
        columns=["product_id", "part_index", "part", "reason"],
    )

    top_keys = sorted(
        key_rows,
        key=lambda item: (-item["n_pairs"], -item["n_values"], item["key_norm"]),
    )[:50]

    class_stats = {
        key_class: {
            "n_values": int(class_values[key_class]),
            "value_words": int(class_words[key_class]),
        }
        for key_class in ("admin", "model", "numeric", "text")
    }

    stats = {
        "rows": len(df),
        "categories": dict(Counter(
            str(value) for value in df.get("category", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        )),
        "title": {
            "n_titles": len(rows),
            "n_empty": n_titles_empty,
            "n_words": _describe(title_words),
            "n_chars": _describe(title_chars),
            "n_short_lt3": sum(1 for count in title_words if count < 3),
            "n_3_40": sum(1 for count in title_words if 3 <= count <= 40),
            "n_long_gt40": sum(1 for count in title_words if count > 40),
            "n_with_numbers": n_title_with_numbers,
            "n_numbers": n_title_numbers,
            "n_with_units": n_title_with_units,
            "n_with_brand": n_title_with_brand,
            "duplicate_groups": duplicate_title_groups,
            "duplicate_rows": duplicate_title_rows,
            "duplicate_extra": duplicate_title_extra,
        },
        "description": {
            "n_empty": n_desc_empty,
            "n_parts": n_desc_parts,
            "n_pairs": n_desc_pairs,
            "n_malformed_parts": len(parse_errors),
            "n_products_with_malformed": len({
                error["product_id"] for error in parse_errors
            }),
        },
        "attributes": {
            "parse_sources": dict(parse_sources),
            "n_products_with_attributes": n_attr_products,
            "n_products_without_attributes": n_attr_missing,
            "n_pairs": n_attr_pairs,
            "n_values": n_attr_values,
            "value_words": n_attr_value_words,
            "n_products_n_attributes_mismatch": n_attr_mismatch,
            "pairs_per_product": _describe(attr_counts_per_product),
            "values_per_product": _describe(value_counts_per_product),
        },
        "value_classes": class_stats,
        "top_keys": top_keys,
        "key_inventory_rows": len(key_inventory),
        "parse_error_rows": len(parse_errors_df),
    }
    return TikiProfileResult(
        stats=_jsonable(stats),
        key_inventory=key_inventory,
        parse_errors=parse_errors_df,
    )


# ═══════════════════════════════════════════════════════════════════
# BT UNITS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TikiUnitsResult:
    products: pd.DataFrame
    units: pd.DataFrame
    stats: dict[str, Any]


def build_tiki_units(
    df: pd.DataFrame,
    *,
    key_map: Mapping[str, str] | None = None,
    key_map_path: str | Path = TIKI_KEY_MAP_CSV,
    min_title_words: int = 3,
    max_title_words: int = 40,
    progress: Callable[[str], None] | None = None,
) -> TikiUnitsResult:
    """Tạo product-level table + pair-level BT units.

    Không gọi MT. ``text_zh``/``value_zh``/``key_zh`` để trống nếu chưa có
    key map hoặc chưa dịch; ``bt_status`` là ``pending``.
    """
    required = ("product_id", "title_vi", "description_vi")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột Tiki: {missing}")

    mapping = dict(key_map) if key_map is not None else load_key_map(key_map_path)
    rows = df.to_dict("records")
    product_ids = [str(row.get("product_id", "")).strip() for row in rows]
    if any(not product_id for product_id in product_ids):
        raise ValueError("Có product_id rỗng; cần xử lý trước khi tạo units.")
    duplicated_ids = len(product_ids) - len(set(product_ids))
    if duplicated_ids:
        raise ValueError(
            f"product_id bị trùng {duplicated_ids:,} dòng; cần xử lý trước khi tạo units."
        )

    # Chuẩn bị title + đếm trùng title.
    prepared: list[dict[str, Any]] = []
    title_keys: list[str] = []
    for row in rows:
        title_raw = normalize_text(row.get("title_vi"))
        title = clean_text(title_raw, title=True)[0]
        attributes, source = extract_attributes(row)
        description_raw = normalize_text(row.get("description_vi"))
        parts = [
            part.strip()
            for part in re.split(r";\s*", description_raw)
            if part.strip()
        ]
        malformed_parts = [part for part in parts if ":" not in part]
        prepared.append({
            "row": row,
            "title_raw": title_raw,
            "title": title,
            "title_key": _text_key(title),
            "attributes": attributes,
            "parse_source": source,
            "description_raw": description_raw,
            "description_parts": len(parts),
            "malformed_parts": len(malformed_parts),
        })
        title_keys.append(_text_key(title))

    title_counts = Counter(key for key in title_keys if key)
    products: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    parse_sources: Counter[str] = Counter()
    value_classes: Counter[str] = Counter()
    bt_methods: Counter[str] = Counter()
    n_malformed_parts = 0
    n_empty_descriptions = 0
    n_title_short = 0
    n_title_long = 0
    n_empty_products = 0

    for index, item in enumerate(prepared):
        row = item["row"]
        product_id = str(row.get("product_id", "")).strip()
        category = str(row.get("category", "")).strip()
        title = item["title"]
        title_words = len(title.split())
        attributes = item["attributes"]
        parse_source = item["parse_source"]
        description_raw = item["description_raw"]
        malformed_parts = item["malformed_parts"]
        n_malformed_parts += malformed_parts
        if not description_raw.strip():
            n_empty_descriptions += 1

        split = split_for_product(product_id, title)
        product_flags: list[str] = []
        if not title:
            product_flags.append("title_empty")
        elif title_words < min_title_words:
            product_flags.append("title_short")
            n_title_short += 1
        elif title_words > max_title_words:
            product_flags.append("title_long")
            n_title_long += 1
        if title_counts.get(item["title_key"], 0) > 1:
            product_flags.append("title_duplicate")
        if not attributes:
            product_flags.append("attributes_missing")
        if not description_raw.strip():
            product_flags.append("description_empty")
        if malformed_parts:
            product_flags.append("description_malformed")
        if _CJK_RE.search(title):
            product_flags.append("title_cjk_present")

        description_vi = serialize_attributes(attributes)
        n_attribute_records = len(attributes)
        n_attribute_values = sum(len(attribute["values"]) for attribute in attributes)
        parse_sources[parse_source] += 1

        product_unit_count = 0
        product_train_eligible = 0

        if title:
            title_class = "text"
            title_method = "mt"
            title_eligible = min_title_words <= title_words <= max_title_words
            title_flags: list[str] = []
            if title_words < min_title_words:
                title_flags.append("title_short")
            if title_words > max_title_words:
                title_flags.append("title_long")
            if _CJK_RE.search(title):
                title_flags.append("cjk_present")
            unit_rows.append({
                "pair_id": f"tiki:{product_id}:title",
                "product_id": product_id,
                "source": TIKI_SOURCE,
                "category": category,
                "split": split,
                "unit_type": "title",
                "unit_index": 0,
                "value_index": -1,
                "key_vi": None,
                "key_norm": None,
                "key_class": None,
                "key_zh": None,
                "value_vi": None,
                "value_zh": None,
                "text_vi": title,
                "text_zh": None,
                "value_class": title_class,
                "bt_method": title_method,
                "bt_status": "pending",
                "key_map_status": "not_applicable",
                "parse_source": parse_source,
                "n_words": title_words,
                "value_n_words": title_words,
                "n_chars": len(title),
                "train_eligible": bool(title_eligible),
                "qc_flags": title_flags,
                "qc_flags_text": "|".join(title_flags),
            })
            product_unit_count += 1
            product_train_eligible += int(title_eligible)

        for attribute_index, attribute in enumerate(attributes):
            key_vi = attribute["name"]
            key_norm = normalize_key(key_vi)
            key_class = classify_key(key_norm)
            key_zh = mapping.get(key_norm)
            key_map_status = "mapped" if key_zh else "pending_phase2"

            for value_index, value_vi in enumerate(attribute["values"]):
                if not value_vi:
                    continue
                value_class = classify_value(value_vi, key_class)
                method = bt_method_for_class(value_class)
                text_vi = f"{key_vi}: {value_vi}"
                value_words = len(value_vi.split())
                unit_flags: list[str] = []
                if _CJK_RE.search(value_vi):
                    unit_flags.append("cjk_present")
                if value_class == "text" and value_words == 0:
                    unit_flags.append("empty_value")
                if not key_zh and value_class == "text":
                    unit_flags.append("key_unmapped")

                train_eligible = value_class == "text" and value_words > 0
                unit_rows.append({
                    "pair_id": f"tiki:{product_id}:attr:{attribute_index}:{value_index}",
                    "product_id": product_id,
                    "source": TIKI_SOURCE,
                    "category": category,
                    "split": split,
                    "unit_type": "attribute",
                    "unit_index": attribute_index,
                    "value_index": value_index,
                    "key_vi": key_vi,
                    "key_norm": key_norm,
                    "key_class": key_class,
                    "key_zh": key_zh,
                    "value_vi": value_vi,
                    "value_zh": None,
                    "text_vi": text_vi,
                    "text_zh": None,
                    "value_class": value_class,
                    "bt_method": method,
                    "bt_status": "pending",
                    "key_map_status": key_map_status,
                    "parse_source": parse_source,
                    "n_words": len(text_vi.split()),
                    "value_n_words": value_words,
                    "n_chars": len(text_vi),
                    "train_eligible": bool(train_eligible),
                    "qc_flags": unit_flags,
                    "qc_flags_text": "|".join(unit_flags),
                })
                value_classes[value_class] += 1
                bt_methods[method] += 1
                product_unit_count += 1
                product_train_eligible += int(train_eligible)

        if product_unit_count == 0:
            n_empty_products += 1
            product_flags.append("nothing_to_translate")

        products.append({
            "product_id": product_id,
            "source": TIKI_SOURCE,
            "category": category,
            "title_vi_raw": item["title_raw"],
            "title_vi": title,
            "title_zh": None,
            "description_vi_raw": description_raw,
            "description_vi": description_vi,
            "description_zh": None,
            "attributes_vi": attributes,
            "attributes_vi_json": json.dumps(attributes, ensure_ascii=False),
            "attributes_zh": [],
            "n_attributes_vi": n_attribute_records,
            "n_attributes_zh": 0,
            "n_attributes_aligned": 0,
            "n_values_vi": n_attribute_values,
            "n_units": product_unit_count,
            "n_train_eligible_units": product_train_eligible,
            "description_parse_source": parse_source,
            "description_n_parts": item["description_parts"],
            "description_n_malformed_parts": malformed_parts,
            "title_duplicate_count": int(title_counts.get(item["title_key"], 0)),
            "split": split,
            "bt_status": "pending" if product_unit_count else "empty",
            "zh_origin": TIKI_ZH_ORIGIN,
            "qc_flags": product_flags,
            "qc_flags_text": "|".join(product_flags),
        })

        if progress and (index + 1) % 2000 == 0:
            progress(f"  units: {index + 1:,}/{len(rows):,}")

    products_df = pd.DataFrame(products)
    units_df = pd.DataFrame(unit_rows)

    if not units_df.empty:
        text_counts = Counter(units_df["text_vi"].str.lower())
        units_df["duplicate_count"] = (
            units_df["text_vi"].str.lower().map(text_counts).fillna(0).astype(int)
        )
        units_df["qc_flags"] = [
            list(flags) + (["duplicate_text"] if duplicate_count > 1 else [])
            for flags, duplicate_count in zip(
                units_df["qc_flags"], units_df["duplicate_count"]
            )
        ]
        units_df["qc_flags_text"] = units_df["qc_flags"].map(
            lambda flags: "|".join(flags)
        )
    else:
        units_df = pd.DataFrame(columns=[
            "pair_id", "product_id", "source", "category", "split",
            "unit_type", "unit_index", "value_index", "key_vi", "key_norm",
            "key_class", "key_zh", "value_vi", "value_zh", "text_vi", "text_zh",
            "value_class", "bt_method", "bt_status", "key_map_status",
            "parse_source", "n_words", "value_n_words", "n_chars",
            "train_eligible", "qc_flags", "qc_flags_text", "duplicate_count",
        ])

    stats = {
        "rows": len(df),
        "products": len(products_df),
        "title_units": int((units_df["unit_type"] == "title").sum()) if not units_df.empty else 0,
        "attribute_units": int((units_df["unit_type"] == "attribute").sum()) if not units_df.empty else 0,
        "total_units": len(units_df),
        "train_eligible_units": int(units_df["train_eligible"].sum()) if not units_df.empty else 0,
        "value_classes": dict(value_classes),
        "bt_methods": dict(bt_methods),
        "parse_sources": dict(parse_sources),
        "products_with_attributes": int((products_df["n_attributes_vi"] > 0).sum()) if not products_df.empty else 0,
        "empty_products": n_empty_products,
        "empty_descriptions": n_empty_descriptions,
        "malformed_description_parts": n_malformed_parts,
        "title_short": n_title_short,
        "title_long": n_title_long,
        "duplicate_title_rows": int((products_df["title_duplicate_count"] > 1).sum()) if not products_df.empty else 0,
        "key_map_entries": len(mapping),
        "mapped_keys": int(units_df["key_map_status"].eq("mapped").sum()) if not units_df.empty else 0,
        "pending_key_map": int(units_df["key_map_status"].eq("pending_phase2").sum()) if not units_df.empty else 0,
        "duplicate_units": int((units_df["duplicate_count"] > 1).sum()) if not units_df.empty else 0,
        "mean_units_per_product": round(len(units_df) / len(products_df), 2) if len(products_df) else 0.0,
    }
    return TikiUnitsResult(
        products=products_df,
        units=units_df,
        stats=_jsonable(stats),
    )
