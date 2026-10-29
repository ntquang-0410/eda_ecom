# -*- coding: utf-8 -*-
"""Length prior v3 — Phase B alignment fallback + Phase C title smoothing.

V3 dùng lại prior value-only của v2, nhưng:

* thêm key-alignment metadata/fallback trước khi fit;
* smooth prior title theo hierarchical partial pooling + monotonic bucket;
* title chỉ dùng soft QC, không ép độ dài khi decode.

V1/v2 vẫn được giữ nguyên để audit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .length_prior_v2 import (
    LengthPriorV2,
    LengthPriorV2ApplyResult,
    LengthPriorV2Config,
    _metric_block,
    apply_length_prior_v2,
    fit_length_prior_v2,
)
from .length_prior_v2 import PRIMARY_GRAINS


@dataclass(frozen=True)
class LengthPriorV3Config:
    """Cấu hình v3; bọc config v2 và thêm title smoothing."""

    v2: LengthPriorV2Config = LengthPriorV2Config()
    title_smoothing_k: int = 20
    title_interval_inflation: float = 0.5
    title_soft_qc: bool = True


@dataclass
class LengthPriorV3:
    table: pd.DataFrame
    stats: dict[str, Any]
    config: LengthPriorV3Config
    v2_prior: LengthPriorV2


@dataclass
class LengthPriorV3ApplyResult:
    units: pd.DataFrame
    stats: dict[str, Any]


def _bucket_order(buckets: tuple[tuple[int, int | None, str], ...]) -> dict[str, int]:
    return {label: index for index, (_, _, label) in enumerate(buckets)}


def _smooth_title_table(
    table: pd.DataFrame,
    config: LengthPriorV3Config,
) -> pd.DataFrame:
    """Thêm expected/range title đã hierarchical-smooth vào table."""
    result = table.copy()
    result["expected_zh_chars_smooth"] = np.nan
    result["zh_chars_p05_smooth"] = np.nan
    result["zh_chars_p95_smooth"] = np.nan
    result["title_smoothing_method"] = ""

    title_mask = result["grain"] == "title"
    parent_mask = (
        title_mask
        & (result["scope"] == "unit_type")
        & (result["unit_type"] == "title")
    )
    if not parent_mask.any():
        return result

    parent = result[parent_mask].iloc[0]
    parent_mu = float(parent["zh_chars_p50"])
    parent_low = float(parent["zh_chars_p05"])
    parent_high = float(parent["zh_chars_p95"])
    k = max(1, int(config.title_smoothing_k))

    for index, row in result[title_mask].iterrows():
        n = int(row["n"])
        if n <= 0:
            continue

        if row["scope"] == "unit_type":
            expected = float(row["zh_chars_p50"])
            low = float(row["zh_chars_p05"])
            high = float(row["zh_chars_p95"])
            method = "parent"
        else:
            mu = float(row["zh_chars_p50"])
            raw_low = float(row["zh_chars_p05"])
            raw_high = float(row["zh_chars_p95"])
            expected = (n * mu + k * parent_mu) / (n + k)
            low = (n * raw_low + k * parent_low) / (n + k)
            high = (n * raw_high + k * parent_high) / (n + k)

            # Bucket ít mẫu: mở rộng interval thay vì giả vờ chắc chắn.
            center = (low + high) / 2.0
            half = max(0.0, (high - low) / 2.0)
            half *= 1.0 + config.title_interval_inflation * (k / (n + k))
            low = center - half
            high = center + half
            method = "hierarchical_parent"

        expected = max(1.0, expected)
        low = max(1.0, min(low, expected))
        high = max(expected, high)
        result.at[index, "expected_zh_chars_smooth"] = expected
        result.at[index, "zh_chars_p05_smooth"] = low
        result.at[index, "zh_chars_p95_smooth"] = high
        result.at[index, "title_smoothing_method"] = method

    # Monotonic expected length theo bucket title.
    bucket_rows = result[
        title_mask
        & (result["scope"] == "unit_type+bucket")
        & result["expected_zh_chars_smooth"].notna()
    ].copy()
    order = _bucket_order(config.v2.buckets)
    bucket_rows["_bucket_order"] = bucket_rows["vi_char_bucket"].map(order)
    bucket_rows = bucket_rows.sort_values(
        ["_bucket_order", "category"], kind="stable"
    )
    if not bucket_rows.empty:
        expected = bucket_rows["expected_zh_chars_smooth"].to_numpy(dtype=float)
        expected = np.maximum.accumulate(expected)
        low = bucket_rows["zh_chars_p05_smooth"].to_numpy(dtype=float)
        high = bucket_rows["zh_chars_p95_smooth"].to_numpy(dtype=float)
        low = np.minimum(low, expected)
        high = np.maximum(high, expected)
        for position, index in enumerate(bucket_rows.index):
            result.at[index, "expected_zh_chars_smooth"] = float(expected[position])
            result.at[index, "zh_chars_p05_smooth"] = float(low[position])
            result.at[index, "zh_chars_p95_smooth"] = float(high[position])

    return result


def fit_length_prior_v3(
    pairs: pd.DataFrame,
    *,
    config: LengthPriorV3Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> LengthPriorV3:
    """Fit v2 value-only rồi smooth riêng prior title."""
    config = config or LengthPriorV3Config()
    v2_prior = fit_length_prior_v2(
        pairs,
        config=config.v2,
        progress=progress,
    )
    table = _smooth_title_table(v2_prior.table, config)
    stats = dict(v2_prior.stats)
    stats["version"] = 3
    stats["title_smoothing"] = {
        "method": "hierarchical_parent+monotonic_bucket",
        "k": int(config.title_smoothing_k),
        "interval_inflation": float(config.title_interval_inflation),
        "soft_qc": bool(config.title_soft_qc),
        "n_title_bucket_rows": int(
            (
                (table["grain"] == "title")
                & (table["scope"] == "unit_type+bucket")
            ).sum()
        ),
    }
    stats["config"] = asdict(config)
    return LengthPriorV3(
        table=table,
        stats=stats,
        config=config,
        v2_prior=v2_prior,
    )


def _title_lookup(
    table: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    title = table[table["grain"] == "title"]
    bucket_rows = {
        str(row["vi_char_bucket"]): row
        for row in title[
            (title["scope"] == "unit_type+bucket")
            & title["expected_zh_chars_smooth"].notna()
        ].to_dict("records")
    }
    overall_rows = title[
        (title["scope"] == "unit_type") & title["expected_zh_chars_smooth"].notna()
    ].to_dict("records")
    overall = overall_rows[0] if overall_rows else None
    return bucket_rows, overall


def _confidence(n: int, min_samples: int) -> str:
    if n >= max(min_samples, 5):
        return "high"
    if n >= 5:
        return "medium"
    if n > 0:
        return "low"
    return "missing"


def apply_length_prior_v3(
    units: pd.DataFrame,
    prior: LengthPriorV3,
    *,
    progress: Callable[[str], None] | None = None,
) -> LengthPriorV3ApplyResult:
    """Apply v2 prior, sau đó override title bằng smoothed title range."""
    base = apply_length_prior_v2(
        units,
        prior.v2_prior,
        config=prior.config.v2,
        progress=progress,
    )
    result = base.units.copy()
    bucket_rows, overall = _title_lookup(prior.table)
    result["length_prior_version"] = 3
    result["length_prior_smoothing"] = "none"

    title_mask = result["unit_type"] == "title"
    for index in result.index[title_mask]:
        row = result.loc[index]
        bucket = str(row.get("vi_char_bucket", ""))
        stat = bucket_rows.get(bucket) or overall
        if stat is None:
            continue
        expected = float(stat["expected_zh_chars_smooth"])
        low = float(stat["zh_chars_p05_smooth"])
        high = float(stat["zh_chars_p95_smooth"])
        n = int(stat["n"])
        result.at[index, "expected_zh_value_chars"] = int(max(1, round(expected)))
        result.at[index, "zh_value_chars_p05"] = int(max(1, math.floor(low)))
        result.at[index, "zh_value_chars_p95"] = int(max(1, math.ceil(high)))
        result.at[index, "expected_zh_text_chars"] = int(max(1, round(expected)))
        result.at[index, "zh_text_chars_p05"] = int(max(1, math.floor(low)))
        result.at[index, "zh_text_chars_p95"] = int(max(1, math.ceil(high)))
        result.at[index, "length_prior_scope"] = (
            "title_bucket_hierarchical" if bucket in bucket_rows else "title_hierarchical"
        )
        result.at[index, "length_prior_n"] = n
        result.at[index, "length_prior_confidence"] = _confidence(
            n, prior.config.v2.min_group_samples
        )
        result.at[index, "length_prior_soft_qc"] = bool(prior.config.title_soft_qc)
        result.at[index, "length_prior_smoothing"] = "hierarchical"
        vi_chars = int(row.get("vi_value_chars", 0) or 0)
        if vi_chars > 0 and expected > 0:
            result.at[index, "ratio_vi_zh_p50"] = round(vi_chars / expected, 4)

    stats = dict(base.stats)
    stats["version"] = 3
    stats["scope_counts"] = {
        str(key): int(value)
        for key, value in result["length_prior_scope"].value_counts().items()
    }
    stats["smoothing_counts"] = {
        str(key): int(value)
        for key, value in result["length_prior_smoothing"].value_counts().items()
    }
    stats["title_smoothing"] = prior.stats["title_smoothing"]
    return LengthPriorV3ApplyResult(units=result, stats=stats)


def evaluate_length_prior_v3(
    pairs: pd.DataFrame,
    prior: LengthPriorV3,
    *,
    split: str,
) -> dict[str, Any]:
    """Đánh giá v3 trên dev/test, dùng actual value chars."""
    test = pairs[
        pairs["grain"].isin(PRIMARY_GRAINS) & (pairs["split"] == split)
    ].copy()
    if test.empty:
        return {"split": split, "n": 0, "version": 3}

    units = pd.DataFrame({
        "text_vi": test["text_vi"].to_numpy(),
        "value_vi": test["value_vi"].to_numpy(),
        "unit_type": test["unit_type"].to_numpy(),
        "value_class": test["value_class"].to_numpy(),
        "category": test["category"].to_numpy(),
        "key_zh": test["key_zh"].to_numpy(),
        "key_vi": test["key_vi"].to_numpy(),
        "key_class": test["key_class"].to_numpy(),
    })
    applied = apply_length_prior_v3(units, prior).units
    evaluated = test.reset_index(drop=True).copy()
    for column in (
        "expected_zh_value_chars",
        "zh_value_chars_p05",
        "zh_value_chars_p95",
        "expected_zh_text_chars",
        "length_prior_scope",
    ):
        evaluated[column] = applied[column].to_numpy()

    by_grain = {
        str(grain): _metric_block(group)
        for grain, group in evaluated.groupby("grain", dropna=False)
    }
    by_value_class = {
        str(value_class): _metric_block(group)
        for value_class, group in evaluated.groupby(
            "value_class", dropna=False
        )
    }
    return {
        "split": split,
        "version": 3,
        "n": int(len(evaluated)),
        "overall": _metric_block(evaluated),
        "by_grain": by_grain,
        "by_value_class": by_value_class,
    }
