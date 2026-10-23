# -*- coding: utf-8 -*-
"""Học prior độ dài VI → ZH từ cặp human 1688, rồi áp dụng cho Tiki.

Mục tiêu không phải dịch, mà là **hiệu chỉnh độ dài kỳ vọng**:

* Từ 16.348 cặp 1688 human, tạo các cặp ``title``, ``description`` và
  ``attribute`` (parse KV description theo đúng thứ tự ZH/VI).
* Học phân bố tỷ lệ ``vi_chars / zh_chars`` theo ``unit_type`` và bucket
  độ dài VI. Các percentile cho biết khoảng độ dài ZH kỳ vọng.
* Áp dụng gián tiếp cho Tiki: mỗi unit VI nhận ``expected_zh_chars`` và
  khoảng ``zh_chars_p05..zh_chars_p95``. Phase 2 dùng khoảng này để QC
  hoặc constrained decoding, không cần dịch trước.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .text import normalize_text
from .tiki import parse_attributes

SOURCE_1688 = "1688_human"
DEFAULT_PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

# Bucket theo số ký tự VI (không tính khoảng trắng). Đây là biến ổn định hơn
# số từ vì title Tiki ngắn hơn title 1688 khá nhiều.
DEFAULT_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (0, 10, "0-10"),
    (11, 20, "11-20"),
    (21, 30, "21-30"),
    (31, 40, "31-40"),
    (41, 50, "41-50"),
    (51, 60, "51-60"),
    (61, 70, "61-70"),
    (71, 80, "71-80"),
    (81, 100, "81-100"),
    (101, 120, "101-120"),
    (121, 150, "121-150"),
    (151, 200, "151-200"),
    (201, 300, "201-300"),
    (301, None, "301+"),
)


@dataclass(frozen=True)
class LengthPriorConfig:
    """Cấu hình học và áp dụng prior."""

    min_group_samples: int = 20
    min_category_samples: int = 100
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES
    buckets: tuple[tuple[int, int | None, str], ...] = DEFAULT_BUCKETS


@dataclass
class LengthPrior:
    """Bảng prior đã học và thống kê đi kèm."""

    table: pd.DataFrame
    stats: dict[str, Any]
    config: LengthPriorConfig


@dataclass
class LengthPriorApplyResult:
    """Kết quả gắn prior độ dài vào unit VI."""

    units: pd.DataFrame
    stats: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# LENGTH HELPERS
# ═══════════════════════════════════════════════════════════════════


def measure_text(text: object) -> tuple[int, int]:
    """Trả về ``(vi_chars, vi_words)``.

    ``vi_chars`` bỏ toàn bộ khoảng trắng để so sánh trực tiếp với ZH; ``vi_words``
    tách theo khoảng trắng (đủ ổn định cho VI và chỉ dùng để audit).
    """
    normalized = normalize_text(text, collapse_whitespace=True).strip()
    if not normalized:
        return 0, 0
    return len("".join(normalized.split())), len(normalized.split())


def length_bucket(
    vi_chars: int,
    buckets: Sequence[tuple[int, int | None, str]] = DEFAULT_BUCKETS,
) -> str:
    """Xếp ``vi_chars`` vào bucket độ dài VI."""
    value = int(vi_chars)
    if value <= 0:
        return "0"
    for low, high, label in buckets:
        if value >= low and (high is None or value <= high):
            return label
    return buckets[-1][2]


def _as_list(value: Any) -> list[Any]:
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
        return converted if isinstance(converted, list) else [converted]
    return []


def _attribute_text(attribute: dict[str, Any]) -> str:
    name = normalize_text(attribute.get("name", ""), collapse_whitespace=True).strip()
    values = [
        normalize_text(value, collapse_whitespace=True).strip()
        for value in _as_list(attribute.get("values"))
        if str(value).strip()
    ]
    if not name or not values:
        return ""
    return f"{name}: {', '.join(values)}"


def _describe(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"count": 0, "mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(data.size),
        "mean": round(float(data.mean()), 2),
        "p05": round(float(np.percentile(data, 5)), 2),
        "p50": round(float(np.percentile(data, 50)), 2),
        "p95": round(float(np.percentile(data, 95)), 2),
        "min": round(float(data.min()), 2),
        "max": round(float(data.max()), 2),
    }


# ═══════════════════════════════════════════════════════════════════
# FIT FROM 1688 HUMAN PAIRS
# ═══════════════════════════════════════════════════════════════════


def extract_human_length_pairs(
    df: pd.DataFrame,
    *,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Tạo các cặp human 1688 ở grain title / description / attribute.

    Attribute pairs được lấy từ ``description_zh`` và ``description_vi`` khi
    hai bên parse ra cùng số phần; thứ tự KV trong description của 1688 khớp
    nhau (khác với thứ tự trong ``attributes_zh`` / ``attributes_vi``).
    """
    required = (
        "title_zh",
        "title_vi",
        "description_zh",
        "description_vi",
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột human 1688: {missing}")

    records: list[dict[str, Any]] = []
    rows = df.to_dict("records")

    def add_pair(
        *,
        unit_type: str,
        text_vi: object,
        text_zh: object,
        product_id: str,
        category: str,
    ) -> None:
        vi = normalize_text(text_vi, collapse_whitespace=True).strip()
        zh = normalize_text(text_zh, collapse_whitespace=True).strip()
        if not vi or not zh:
            return
        records.append({
            "product_id": product_id,
            "source": SOURCE_1688,
            "category": category,
            "unit_type": unit_type,
            "text_vi": vi,
            "text_zh": zh,
        })

    for index, row in enumerate(rows):
        product_id = str(row.get("product_id", "")).strip()
        category = str(row.get("category", "")).strip()
        description_vi = normalize_text(row.get("description_vi"), collapse_whitespace=True).strip()
        description_zh = normalize_text(row.get("description_zh"), collapse_whitespace=True).strip()

        add_pair(
            unit_type="title",
            text_vi=row.get("title_vi"),
            text_zh=row.get("title_zh"),
            product_id=product_id,
            category=category,
        )
        add_pair(
            unit_type="description",
            text_vi=description_vi,
            text_zh=description_zh,
            product_id=product_id,
            category=category,
        )

        attributes_vi = parse_attributes(None, description_vi)
        attributes_zh = parse_attributes(None, description_zh)
        if len(attributes_vi) == len(attributes_zh):
            for attribute_vi, attribute_zh in zip(attributes_vi, attributes_zh):
                add_pair(
                    unit_type="attribute",
                    text_vi=_attribute_text(attribute_vi),
                    text_zh=_attribute_text(attribute_zh),
                    product_id=product_id,
                    category=category,
                )

        if progress and (index + 1) % 2000 == 0:
            progress(f"  pairs: {index + 1:,}/{len(rows):,}")

    pairs = pd.DataFrame(records)
    if pairs.empty:
        raise ValueError("Không tạo được cặp human 1688 nào.")

    measured_vi = pairs["text_vi"].map(measure_text)
    measured_zh = pairs["text_zh"].map(measure_text)
    pairs["vi_chars"] = measured_vi.map(lambda value: value[0])
    pairs["vi_words"] = measured_vi.map(lambda value: value[1])
    pairs["zh_chars"] = measured_zh.map(lambda value: value[0])
    pairs["zh_words"] = measured_zh.map(lambda value: value[1])
    pairs = pairs[(pairs["vi_chars"] > 0) & (pairs["zh_chars"] > 0)].copy()
    pairs["ratio_vi_zh"] = pairs["vi_chars"] / pairs["zh_chars"]
    pairs["zh_per_vi_word"] = pairs["zh_chars"] / pairs["vi_words"].clip(lower=1)
    pairs["vi_char_bucket"] = pairs["vi_chars"].map(length_bucket)
    return pairs.reset_index(drop=True)


def _summarize_group(
    group: pd.DataFrame,
    percentiles: tuple[float, ...],
) -> dict[str, Any]:
    ratios = group["ratio_vi_zh"].to_numpy(dtype=float)
    zh_per_word = group["zh_per_vi_word"].to_numpy(dtype=float)
    vi_chars = group["vi_chars"].to_numpy(dtype=float)
    zh_chars = group["zh_chars"].to_numpy(dtype=float)
    vi_words = group["vi_words"].to_numpy(dtype=float)

    def _pct(values: np.ndarray, percentile: float) -> float:
        return round(float(np.percentile(values, percentile * 100)), 4)

    result: dict[str, Any] = {
        "n": int(len(group)),
        "vi_chars_p05": round(float(np.percentile(vi_chars, 5)), 2),
        "vi_chars_p50": round(float(np.percentile(vi_chars, 50)), 2),
        "vi_chars_p95": round(float(np.percentile(vi_chars, 95)), 2),
        "vi_words_p50": round(float(np.percentile(vi_words, 50)), 2),
        "zh_chars_p05": round(float(np.percentile(zh_chars, 5)), 2),
        "zh_chars_p50": round(float(np.percentile(zh_chars, 50)), 2),
        "zh_chars_p95": round(float(np.percentile(zh_chars, 95)), 2),
    }
    for percentile in percentiles:
        suffix = f"p{int(round(percentile * 100)):02d}"
        result[f"ratio_vi_zh_{suffix}"] = _pct(ratios, percentile)
        result[f"zh_per_vi_word_{suffix}"] = _pct(zh_per_word, percentile)
    return result


def fit_length_prior(
    pairs: pd.DataFrame,
    *,
    config: LengthPriorConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> LengthPrior:
    """Học bảng tỷ lệ độ dài từ các cặp human 1688."""
    config = config or LengthPriorConfig()
    required = ("unit_type", "category", "vi_chars", "zh_chars", "ratio_vi_zh", "vi_char_bucket")
    missing = [column for column in required if column not in pairs.columns]
    if missing:
        raise KeyError(f"Thiếu cột pairs: {missing}")
    if pairs.empty:
        raise ValueError("Không có cặp human nào để fit length prior.")

    records: list[dict[str, Any]] = []

    def add_group(
        *,
        scope: str,
        group: pd.DataFrame,
        unit_type: str = "",
        category: str = "",
        bucket: str = "",
    ) -> None:
        if group.empty:
            return
        record = {
            "scope": scope,
            "unit_type": unit_type,
            "category": category,
            "vi_char_bucket": bucket,
        }
        record.update(_summarize_group(group, config.percentiles))
        records.append(record)

    add_group(scope="global", group=pairs)

    for unit_type, unit_group in pairs.groupby("unit_type", dropna=False):
        unit_type = str(unit_type)
        add_group(scope="unit_type", group=unit_group, unit_type=unit_type)

        for bucket, bucket_group in unit_group.groupby("vi_char_bucket", dropna=False):
            add_group(
                scope="unit_type+bucket",
                group=bucket_group,
                unit_type=unit_type,
                bucket=str(bucket),
            )

        for category, category_group in unit_group.groupby("category", dropna=False):
            category = str(category)
            add_group(
                scope="unit_type+category",
                group=category_group,
                unit_type=unit_type,
                category=category,
            )
            for bucket, category_bucket_group in category_group.groupby("vi_char_bucket", dropna=False):
                add_group(
                    scope="unit_type+category+bucket",
                    group=category_bucket_group,
                    unit_type=unit_type,
                    category=category,
                    bucket=str(bucket),
                )

        if progress:
            progress(f"  fit: {unit_type} ({len(unit_group):,} cặp)")

    table = pd.DataFrame(records)
    scope_order = {
        "global": 0,
        "unit_type": 1,
        "unit_type+bucket": 2,
        "unit_type+category": 3,
        "unit_type+category+bucket": 4,
    }
    table["_scope_order"] = table["scope"].map(scope_order).fillna(99).astype(int)
    table = table.sort_values(
        ["_scope_order", "unit_type", "category", "vi_char_bucket"],
        kind="stable",
    ).drop(columns="_scope_order").reset_index(drop=True)

    global_row = table[table["scope"] == "global"].iloc[0].to_dict()
    by_unit_type = {
        row["unit_type"]: row
        for row in table[table["scope"] == "unit_type"].to_dict("records")
    }
    pair_counts = {
        str(key): int(value)
        for key, value in pairs["unit_type"].value_counts().items()
    }
    category_counts = {
        str(key): int(value)
        for key, value in pairs["category"].value_counts().items()
    }
    stats = {
        "source": SOURCE_1688,
        "n_pairs": int(len(pairs)),
        "n_products": int(pairs["product_id"].nunique()) if "product_id" in pairs.columns else 0,
        "pair_counts": pair_counts,
        "category_counts": category_counts,
        "overall": global_row,
        "by_unit_type": by_unit_type,
        "n_groups": int(len(table)),
        "vi_chars": _describe(pairs["vi_chars"]),
        "zh_chars": _describe(pairs["zh_chars"]),
        "ratio_vi_zh": _describe(pairs["ratio_vi_zh"]),
        "config": asdict(config),
    }
    return LengthPrior(table=table, stats=stats, config=config)


# ═══════════════════════════════════════════════════════════════════
# APPLY TO VI UNITS
# ═══════════════════════════════════════════════════════════════════


def _prior_lookup(table: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row["scope"]),
            str(row["unit_type"]),
            str(row["category"]),
            str(row["vi_char_bucket"]),
        ): row
        for row in table.to_dict("records")
    }


def _select_prior(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    unit_type: str,
    category: str,
    bucket: str,
    config: LengthPriorConfig,
) -> tuple[dict[str, Any] | None, str]:
    candidates: list[tuple[str, str, str, str, int]] = []
    if unit_type in {"title", "attribute"}:
        if category:
            candidates.append((
                "unit_type+category+bucket",
                unit_type,
                category,
                bucket,
                max(config.min_group_samples, config.min_category_samples),
            ))
        candidates.append((
            "unit_type+bucket",
            unit_type,
            "",
            bucket,
            config.min_group_samples,
        ))
        # Bucket là điều kiện quan trọng hơn category: title 1688 dài ~30 ký tự
        # ZH bất kể độ dài VI, nên fallback category (không có bucket) sẽ dự
        # đoán quá ngắn cho title Tiki ngắn. Nếu bucket có ít mẫu, vẫn dùng
        # bucket trước category; ``length_prior_n`` cho biết độ tin cậy.
        candidates.append((
            "unit_type+bucket",
            unit_type,
            "",
            bucket,
            1,
        ))
        if category:
            candidates.append((
                "unit_type+category",
                unit_type,
                category,
                "",
                config.min_category_samples,
            ))
        candidates.append((
            "unit_type",
            unit_type,
            "",
            "",
            config.min_group_samples,
        ))
    candidates.append(("global", "", "", "", 1))

    for scope, candidate_unit, candidate_category, candidate_bucket, min_samples in candidates:
        record = lookup.get((scope, candidate_unit, candidate_category, candidate_bucket))
        if record is not None and int(record["n"]) >= min_samples:
            return record, scope
    return None, "missing"


def apply_length_prior(
    units: pd.DataFrame,
    prior: LengthPrior,
    *,
    config: LengthPriorConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> LengthPriorApplyResult:
    """Gắn khoảng độ dài ZH kỳ vọng vào từng unit VI."""
    config = config or prior.config or LengthPriorConfig()
    if "text_vi" not in units.columns:
        raise KeyError("Thiếu cột text_vi trong units.")
    if "unit_type" not in units.columns:
        raise KeyError("Thiếu cột unit_type trong units.")

    if units.empty:
        result = units.copy()
        for column, default in (
            ("length_prior_source", SOURCE_1688),
            ("length_prior_scope", "missing"),
            ("length_prior_n", 0),
            ("vi_char_bucket", "0"),
            ("vi_chars", 0),
            ("vi_words", 0),
            ("ratio_vi_zh_p05", 0.0),
            ("ratio_vi_zh_p50", 0.0),
            ("ratio_vi_zh_p95", 0.0),
            ("zh_per_vi_word_p05", 0.0),
            ("zh_per_vi_word_p50", 0.0),
            ("zh_per_vi_word_p95", 0.0),
            ("expected_zh_chars", 0),
            ("zh_chars_p05", 0),
            ("zh_chars_p95", 0),
        ):
            result[column] = default
        return LengthPriorApplyResult(
            units=result,
            stats={"n_units": 0, "source": SOURCE_1688, "scope_counts": {}},
        )

    lookup = _prior_lookup(prior.table)
    records: list[dict[str, Any]] = []
    rows = units.to_dict("records")

    for index, row in enumerate(rows):
        vi_chars, vi_words = measure_text(row.get("text_vi"))
        bucket = length_bucket(vi_chars, config.buckets)
        unit_type = str(row.get("unit_type", "")).strip()
        category = str(row.get("category", "")).strip()
        stat, scope = _select_prior(
            lookup,
            unit_type=unit_type,
            category=category,
            bucket=bucket,
            config=config,
        )

        if stat is None or vi_chars <= 0:
            expected = low = high = 0
            ratio_p05 = ratio_p50 = ratio_p95 = 0.0
            word_p05 = word_p50 = word_p95 = 0.0
            prior_n = 0
        else:
            ratio_p05 = float(stat["ratio_vi_zh_p05"])
            ratio_p50 = float(stat["ratio_vi_zh_p50"])
            ratio_p95 = float(stat["ratio_vi_zh_p95"])
            word_p05 = float(stat["zh_per_vi_word_p05"])
            word_p50 = float(stat["zh_per_vi_word_p50"])
            word_p95 = float(stat["zh_per_vi_word_p95"])
            prior_n = int(stat["n"])

            if ratio_p50 > 0:
                expected = max(1, int(round(vi_chars / ratio_p50)))
                low = max(1, int(math.floor(vi_chars / ratio_p95))) if ratio_p95 > 0 else expected
                high = max(1, int(math.ceil(vi_chars / ratio_p05))) if ratio_p05 > 0 else expected
                low = min(low, expected)
                high = max(high, expected)
            else:
                expected = low = high = 0

        records.append({
            "length_prior_source": SOURCE_1688,
            "length_prior_scope": scope,
            "length_prior_n": prior_n,
            "vi_char_bucket": bucket,
            "vi_chars": int(vi_chars),
            "vi_words": int(vi_words),
            "ratio_vi_zh_p05": round(ratio_p05, 4),
            "ratio_vi_zh_p50": round(ratio_p50, 4),
            "ratio_vi_zh_p95": round(ratio_p95, 4),
            "zh_per_vi_word_p05": round(word_p05, 4),
            "zh_per_vi_word_p50": round(word_p50, 4),
            "zh_per_vi_word_p95": round(word_p95, 4),
            "expected_zh_chars": int(expected),
            "zh_chars_p05": int(low),
            "zh_chars_p95": int(high),
        })

        if progress and (index + 1) % 20000 == 0:
            progress(f"  apply: {index + 1:,}/{len(rows):,}")

    prior_frame = pd.DataFrame(records)
    result = units.copy().reset_index(drop=True)
    for column in prior_frame.columns:
        result[column] = prior_frame[column].to_numpy()

    scope_counts = {
        str(key): int(value)
        for key, value in prior_frame["length_prior_scope"].value_counts().items()
    }
    bucket_counts = {
        str(key): int(value)
        for key, value in prior_frame["vi_char_bucket"].value_counts().items()
    }
    stats = {
        "n_units": int(len(result)),
        "source": SOURCE_1688,
        "scope_counts": scope_counts,
        "bucket_counts": bucket_counts,
        "n_zero_vi_chars": int((prior_frame["vi_chars"] <= 0).sum()),
        "expected_zh_chars": _describe(prior_frame["expected_zh_chars"]),
        "zh_chars_p05": _describe(prior_frame["zh_chars_p05"]),
        "zh_chars_p95": _describe(prior_frame["zh_chars_p95"]),
        "ratio_vi_zh_p50": _describe(prior_frame["ratio_vi_zh_p50"]),
    }
    return LengthPriorApplyResult(units=result, stats=stats)


def length_qc(
    text_zh: object,
    row: pd.Series | dict[str, Any],
    *,
    margin: int = 0,
) -> dict[str, Any]:
    """QC độ dài một output ZH so với prior của unit VI.

    Hàm này dùng ở Phase 2, sau khi có ``text_zh``. Nó không gọi model.
    """
    zh_chars, _ = measure_text(text_zh)
    vi_chars = int(row.get("vi_chars", 0) or 0)
    low = int(row.get("zh_chars_p05", 0) or 0) - int(margin)
    high = int(row.get("zh_chars_p95", 0) or 0) + int(margin)
    expected = int(row.get("expected_zh_chars", 0) or 0)

    flags: list[str] = []
    if vi_chars > 0:
        if low > 0 and zh_chars < low:
            flags.append("zh_too_short")
        elif high > 0 and zh_chars > high:
            flags.append("zh_too_long")

    ratio = round(vi_chars / zh_chars, 4) if zh_chars else None
    if not flags:
        fit = "ok"
    elif "zh_too_short" in flags:
        fit = "short"
    else:
        fit = "long"

    return {
        "zh_chars": int(zh_chars),
        "expected_zh_chars": int(expected),
        "zh_chars_p05": int(low),
        "zh_chars_p95": int(high),
        "length_ratio_vi_zh": ratio,
        "length_fit": fit,
        "length_qc_flags": flags,
    }


def write_prior_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Ghi bảng prior ra CSV audit."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    return output
