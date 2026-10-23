# -*- coding: utf-8 -*-
"""Length prior v2 — value-only, value-class, CJK-aware, split-safe.

Bản v1 trong :mod:`cado_core.length_prior` học trên ``key: value`` và vẫn được
giữ nguyên làm audit. Module này tạo prior v2 riêng:

* target chính là **value-only** (``zh_value_chars``);
* tách value 1688 theo dấu phẩy/liệt kê có guard cho số thập phân;
* fit theo ``value_class`` (text / numeric / admin / model);
* đo song song ``zh_cjk_chars`` và các token không phải CJK;
* split theo ``product_id`` (train/dev/test), không leakage giữa sản phẩm.

Không gọi model/MT.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import LATIN_WORD_RE
from .length_prior import (
    DEFAULT_BUCKETS,
    SOURCE_1688,
    length_bucket,
    measure_text,
)
from .text import count_cjk, normalize_text
from .tiki import parse_attributes
from .tiki_units import classify_key, classify_value, normalize_key

SPLITS = ("train", "dev", "test")
PRIMARY_GRAINS = ("title", "attribute_value")
DEFAULT_PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_VALUE_SPLIT_RE = re.compile(r"(?<!\d)[,，、;；](?!\d)")


# ═══════════════════════════════════════════════════════════════════
# CONFIG / RESULT TYPES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LengthPriorV2Config:
    """Cấu hình fit/apply prior v2."""

    min_group_samples: int = 20
    min_category_samples: int = 100
    min_value_class_samples: int = 50
    min_value_class_category_samples: int = 100
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES
    buckets: tuple[tuple[int, int | None, str], ...] = DEFAULT_BUCKETS
    split_seed: int = 42
    train_pct: int = 80
    dev_pct: int = 10
    fit_split: str = "train"  # train | all
    separator_chars: int = 2
    title_soft_qc: bool = True


@dataclass
class HumanValuePairs:
    """Cặp human 1688 ở grain title / attribute_value / attribute_group."""

    pairs: pd.DataFrame
    split_audit: pd.DataFrame
    stats: dict[str, Any]


@dataclass
class LengthPriorV2:
    """Bảng prior v2 đã fit và metadata."""

    table: pd.DataFrame
    stats: dict[str, Any]
    config: LengthPriorV2Config
    split_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class LengthPriorV2ApplyResult:
    units: pd.DataFrame
    stats: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def split_for_product(
    product_id: object,
    *,
    seed: int = 42,
    train_pct: int = 80,
    dev_pct: int = 10,
) -> str:
    """Split ổn định theo product_id, không phụ thuộc thứ tự dòng."""
    if train_pct < 0 or dev_pct < 0 or train_pct + dev_pct > 100:
        raise ValueError("train_pct/dev_pct không hợp lệ")
    digest = hashlib.blake2b(
        f"{seed}:{product_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    bucket = int.from_bytes(digest, "big") % 1000
    if bucket < train_pct * 10:
        return "train"
    if bucket < (train_pct + dev_pct) * 10:
        return "dev"
    return "test"


def split_value_list(value: object) -> list[str]:
    """Tách một value thành các token liệt kê.

    Guard quan trọng: không tách dấu phẩy giữa hai chữ số, để ``1,5 kg`` và
    ``1,000`` không bị biến thành hai value.
    """
    text = normalize_text(value, collapse_whitespace=True).strip()
    if not text:
        return []
    parts = _VALUE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def measure_zh(text: object) -> dict[str, Any]:
    """Đo độ dài ZH theo tổng ký tự, CJK thuần và token không phải CJK."""
    normalized = normalize_text(text, collapse_whitespace=True).strip()
    chars = len("".join(normalized.split())) if normalized else 0
    cjk_chars = count_cjk(normalized) if normalized else 0
    return {
        "zh_chars": int(chars),
        "zh_cjk_chars": int(cjk_chars),
        "zh_non_cjk_chars": int(max(0, chars - cjk_chars)),
        "zh_digits": int(len(re.findall(r"\d", normalized))),
        "zh_latin_tokens": int(len(LATIN_WORD_RE.findall(normalized))),
        "cjk_ratio": round(cjk_chars / chars, 4) if chars else 0.0,
    }


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


def _attribute_key(attribute: dict[str, Any]) -> str:
    return normalize_text(attribute.get("name", ""), collapse_whitespace=True).strip()


def _attribute_value(attribute: dict[str, Any]) -> str:
    values = [
        normalize_text(value, collapse_whitespace=True).strip()
        for value in _as_list(attribute.get("values"))
        if str(value).strip()
    ]
    return ", ".join(values)


def _numeric_overlap(text_zh: object, text_vi: object) -> float:
    """Jaccard của các cụm số/model đơn giản giữa ZH và VI."""
    numbers_zh = set(re.findall(r"\d+(?:[.,]\d+)?", str(text_zh or "")))
    numbers_vi = set(re.findall(r"\d+(?:[.,]\d+)?", str(text_vi or "")))
    if not numbers_zh and not numbers_vi:
        return 0.0
    union = numbers_zh | numbers_vi
    return round(len(numbers_zh & numbers_vi) / len(union), 4) if union else 0.0


def _describe(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(data.size),
        "mean": round(float(data.mean()), 2),
        "p05": round(float(np.percentile(data, 5)), 2),
        "p50": round(float(np.percentile(data, 50)), 2),
        "p95": round(float(np.percentile(data, 95)), 2),
        "min": round(float(data.min()), 2),
        "max": round(float(data.max()), 2),
    }


def _percentiles(
    values: np.ndarray,
    percentiles: Sequence[float],
    *,
    prefix: str,
) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_p{int(round(p * 100)):02d}": 0.0 for p in percentiles}
    return {
        f"{prefix}_p{int(round(p * 100)):02d}": round(
            float(np.percentile(values, p * 100)), 4
        )
        for p in percentiles
    }


# ═══════════════════════════════════════════════════════════════════
# ALIGNMENT MAP / AUDIT (PHASE B)
# ═══════════════════════════════════════════════════════════════════


def build_key_alignment_map(
    audit: pd.DataFrame,
    *,
    min_freq: int = 20,
    min_share: float = 0.60,
) -> tuple[dict[str, str], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Học map ``key_zh_norm -> key_vi_norm`` từ description-order audit.

    Chỉ map key có đủ tần suất và top-1 chiếm ưu thế. Các key còn lại không
    được map để tránh fallback sai.
    """
    if audit.empty:
        return {}, {}, {
            "n_groups": 0,
            "n_zh_keys": 0,
            "n_mapped_keys": 0,
            "min_freq": min_freq,
            "min_share": min_share,
        }

    pair_counts: Counter[tuple[str, str]] = Counter()
    zh_counts: Counter[str] = Counter()
    for row in audit[["key_zh", "key_vi"]].to_dict("records"):
        key_zh = normalize_key(row.get("key_zh"))
        key_vi = normalize_key(row.get("key_vi"))
        if not key_zh or not key_vi:
            continue
        pair_counts[(key_zh, key_vi)] += 1
        zh_counts[key_zh] += 1

    by_zh: dict[str, Counter[str]] = defaultdict(Counter)
    for (key_zh, key_vi), count in pair_counts.items():
        by_zh[key_zh][key_vi] += count

    key_map: dict[str, str] = {}
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key_zh, vi_counts in by_zh.items():
        total = int(zh_counts[key_zh])
        top_vi, top_count = vi_counts.most_common(1)[0]
        top_share = top_count / total if total else 0.0
        mapped = top_count >= min_freq and top_share >= min_share
        if mapped:
            key_map[key_zh] = top_vi
        for key_vi, count in vi_counts.items():
            pair_stats[(key_zh, key_vi)] = {
                "freq": int(count),
                "zh_total": total,
                "share": round(count / total, 4) if total else 0.0,
                "top_vi": top_vi,
                "top_share": round(top_share, 4),
                "mapped": bool(mapped),
            }

    stats = {
        "n_groups": int(len(audit)),
        "n_zh_keys": int(len(by_zh)),
        "n_mapped_keys": int(len(key_map)),
        "n_pairs_high_freq": int(
            sum(1 for stat in pair_stats.values() if stat["freq"] >= min_freq)
        ),
        "min_freq": int(min_freq),
        "min_share": float(min_share),
        "n_pair_keys": int(len(pair_stats)),
    }
    return key_map, pair_stats, stats


# ═══════════════════════════════════════════════════════════════════
# EXTRACT 1688 VALUE-LEVEL PAIRS
# ═══════════════════════════════════════════════════════════════════


def extract_human_value_pairs(
    df: pd.DataFrame,
    *,
    config: LengthPriorV2Config | None = None,
    key_map: dict[str, str] | None = None,
    key_pair_stats: dict[tuple[str, str], dict[str, Any]] | None = None,
    min_key_pair_freq: int = 20,
    align_key_fallback: bool = True,
    progress: Callable[[str], None] | None = None,
) -> HumanValuePairs:
    """Tạo cặp value-only từ 1688, kèm audit tách value.

    Nếu truyền ``key_map``/``key_pair_stats`` (Phase B), các group có key-pair
    description-order ít mẫu sẽ thử align lại VI attribute theo key map.
    """
    config = config or LengthPriorV2Config()
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
    audit: list[dict[str, Any]] = []
    rows = df.to_dict("records")
    description_aligned_rows = 0
    description_mismatch_rows = 0

    def add_pair(
        *,
        grain: str,
        unit_type: str,
        key_zh: str,
        key_vi: str,
        value_zh: str,
        value_vi: str,
        product_id: str,
        category: str,
        split: str,
        unit_index: int,
        value_index: int,
        split_status: str,
        n_tokens_zh: int,
        n_tokens_vi: int,
        alignment_method: str,
        key_pair_freq: int = 0,
        key_pair_share: float = 0.0,
        alignment_confidence: str = "high",
    ) -> bool:
        vi = normalize_text(value_vi, collapse_whitespace=True).strip()
        zh = normalize_text(value_zh, collapse_whitespace=True).strip()
        if not vi or not zh:
            return False
        vi_chars, vi_words = measure_text(vi)
        zh_fields = measure_zh(zh)
        if vi_chars <= 0 or zh_fields["zh_chars"] <= 0:
            return False

        key_vi = normalize_text(key_vi, collapse_whitespace=True).strip()
        key_zh = normalize_text(key_zh, collapse_whitespace=True).strip()
        key_class = classify_key(key_vi) if key_vi else ""
        value_class = (
            classify_value(vi, key_class or "text")
            if unit_type == "attribute"
            else ""
        )
        key_zh_chars = measure_text(key_zh)[0] if key_zh else 0
        key_vi_chars = measure_text(key_vi)[0] if key_vi else 0

        records.append({
            "product_id": product_id,
            "source": SOURCE_1688,
            "category": category,
            "split": split,
            "unit_type": unit_type,
            "grain": grain,
            "unit_index": unit_index,
            "value_index": value_index,
            "key_zh": key_zh,
            "key_vi": key_vi,
            "key_class": key_class,
            "value_class": value_class,
            "value_zh": zh,
            "value_vi": vi,
            "text_vi": vi,
            "text_zh": zh,
            "vi_chars": int(vi_chars),
            "vi_words": int(vi_words),
            "zh_chars": int(zh_fields["zh_chars"]),
            "zh_cjk_chars": int(zh_fields["zh_cjk_chars"]),
            "zh_non_cjk_chars": int(zh_fields["zh_non_cjk_chars"]),
            "zh_digits": int(zh_fields["zh_digits"]),
            "zh_latin_tokens": int(zh_fields["zh_latin_tokens"]),
            "cjk_ratio": float(zh_fields["cjk_ratio"]),
            "key_zh_chars": int(key_zh_chars),
            "key_vi_chars": int(key_vi_chars),
            "split_status": split_status,
            "n_tokens_zh": int(n_tokens_zh),
            "n_tokens_vi": int(n_tokens_vi),
            "alignment_method": alignment_method,
            "alignment_confidence": alignment_confidence,
            "key_pair_freq": int(key_pair_freq),
            "key_pair_share": round(float(key_pair_share), 4),
        })
        return True

    for index, row in enumerate(rows):
        product_id = str(row.get("product_id", "")).strip()
        category = str(row.get("category", "")).strip()
        split = split_for_product(
            product_id,
            seed=config.split_seed,
            train_pct=config.train_pct,
            dev_pct=config.dev_pct,
        )
        description_vi = normalize_text(
            row.get("description_vi"), collapse_whitespace=True
        ).strip()
        description_zh = normalize_text(
            row.get("description_zh"), collapse_whitespace=True
        ).strip()

        add_pair(
            grain="title",
            unit_type="title",
            key_zh="",
            key_vi="",
            value_zh=str(row.get("title_zh") or ""),
            value_vi=str(row.get("title_vi") or ""),
            product_id=product_id,
            category=category,
            split=split,
            unit_index=-1,
            value_index=-1,
            split_status="not_applicable",
            n_tokens_zh=1,
            n_tokens_vi=1,
            alignment_method="native",
        )

        attributes_vi = parse_attributes(None, description_vi)
        attributes_zh = parse_attributes(None, description_zh)
        if len(attributes_vi) != len(attributes_zh):
            description_mismatch_rows += 1
            continue
        description_aligned_rows += 1

        raw_vi_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw_attribute in _as_list(row.get("attributes_vi")):
            if not isinstance(raw_attribute, dict):
                continue
            raw_key = normalize_key(_attribute_key(raw_attribute))
            if raw_key:
                raw_vi_by_key[raw_key].append(raw_attribute)

        for attribute_index, (attribute_vi, attribute_zh) in enumerate(
            zip(attributes_vi, attributes_zh)
        ):
            key_vi = _attribute_key(attribute_vi)
            key_zh = _attribute_key(attribute_zh)
            group_vi = _attribute_value(attribute_vi)
            group_zh = _attribute_value(attribute_zh)
            if not key_vi or not key_zh or not group_vi or not group_zh:
                continue

            key_zh_norm = normalize_key(key_zh)
            key_vi_norm = normalize_key(key_vi)
            pair_stat = (key_pair_stats or {}).get(
                (key_zh_norm, key_vi_norm), {}
            )
            key_pair_freq = int(pair_stat.get("freq", 0) or 0)
            key_pair_share = float(pair_stat.get("share", 0.0) or 0.0)
            alignment_method = "description_order"
            alignment_confidence = (
                "high"
                if key_pair_freq >= min_key_pair_freq
                and key_pair_share >= 0.60
                else "low"
            )

            if (
                alignment_confidence == "low"
                and align_key_fallback
                and key_map
            ):
                predicted_vi = key_map.get(key_zh_norm)
                matches = raw_vi_by_key.get(predicted_vi, []) if predicted_vi else []
                if len(matches) == 1:
                    alternate = matches[0]
                    alternate_key = _attribute_key(alternate)
                    alternate_value = _attribute_value(alternate)
                    if alternate_key and alternate_value:
                        key_vi = alternate_key
                        group_vi = alternate_value
                        key_vi_norm = normalize_key(key_vi)
                        alt_stat = (key_pair_stats or {}).get(
                            (key_zh_norm, key_vi_norm), {}
                        )
                        key_pair_freq = int(
                            alt_stat.get("freq", key_pair_freq) or key_pair_freq
                        )
                        key_pair_share = float(
                            alt_stat.get("share", key_pair_share)
                            or key_pair_share
                        )
                        alignment_method = "key_fallback"
                        alignment_confidence = "medium"

            tokens_zh = split_value_list(group_zh)
            tokens_vi = split_value_list(group_vi)
            n_zh = len(tokens_zh)
            n_vi = len(tokens_vi)
            key_class = classify_key(key_vi)
            value_class_group = classify_value(group_vi, key_class)

            if n_zh == 0 or n_vi == 0:
                split_status = "empty"
                n_values_added = 0
            elif n_zh == n_vi == 1:
                split_status = "single"
                n_values_added = 1
            elif n_zh == n_vi:
                split_status = "multi_matched"
                n_values_added = n_zh
            else:
                split_status = "mismatch"
                n_values_added = 0

            if split_status in {"mismatch", "empty"}:
                alignment_confidence = "low"

            group_zh_chars = measure_text(group_zh)[0]
            group_vi_chars = measure_text(group_vi)[0]
            audit.append({
                "product_id": product_id,
                "category": category,
                "split": split,
                "unit_index": attribute_index,
                "key_zh": key_zh,
                "key_vi": key_vi,
                "key_class": key_class,
                "value_class_group": value_class_group,
                "value_zh_raw": group_zh,
                "value_vi_raw": group_vi,
                "n_tokens_zh": n_zh,
                "n_tokens_vi": n_vi,
                "split_status": split_status,
                "n_values_added": n_values_added,
                "alignment_method": alignment_method,
                "alignment_confidence": alignment_confidence,
                "key_pair_freq": key_pair_freq,
                "key_pair_share": round(key_pair_share, 4),
                "zh_chars_group": int(group_zh_chars),
                "vi_chars_group": int(group_vi_chars),
                "length_ratio_vi_zh": round(
                    group_vi_chars / group_zh_chars, 4
                ) if group_zh_chars else 0.0,
                "numeric_overlap": _numeric_overlap(group_zh, group_vi),
            })

            if split_status == "empty":
                continue

            if split_status == "single":
                add_pair(
                    grain="attribute_value",
                    unit_type="attribute",
                    key_zh=key_zh,
                    key_vi=key_vi,
                    value_zh=tokens_zh[0],
                    value_vi=tokens_vi[0],
                    product_id=product_id,
                    category=category,
                    split=split,
                    unit_index=attribute_index,
                    value_index=0,
                    split_status=split_status,
                    n_tokens_zh=n_zh,
                    n_tokens_vi=n_vi,
                    alignment_method=alignment_method,
                    key_pair_freq=key_pair_freq,
                    key_pair_share=key_pair_share,
                    alignment_confidence=alignment_confidence,
                )
                continue

            if split_status == "multi_matched":
                # Group pair giữ lại để audit grain; value pairs là prior chính.
                add_pair(
                    grain="attribute_group",
                    unit_type="attribute",
                    key_zh=key_zh,
                    key_vi=key_vi,
                    value_zh=group_zh,
                    value_vi=group_vi,
                    product_id=product_id,
                    category=category,
                    split=split,
                    unit_index=attribute_index,
                    value_index=-1,
                    split_status=split_status,
                    n_tokens_zh=n_zh,
                    n_tokens_vi=n_vi,
                    alignment_method=alignment_method,
                    key_pair_freq=key_pair_freq,
                    key_pair_share=key_pair_share,
                    alignment_confidence=alignment_confidence,
                )
                for value_index, (value_zh, value_vi) in enumerate(
                    zip(tokens_zh, tokens_vi)
                ):
                    add_pair(
                        grain="attribute_value",
                        unit_type="attribute",
                        key_zh=key_zh,
                        key_vi=key_vi,
                        value_zh=value_zh,
                        value_vi=value_vi,
                        product_id=product_id,
                        category=category,
                        split=split,
                        unit_index=attribute_index,
                        value_index=value_index,
                        split_status=split_status,
                        n_tokens_zh=n_zh,
                        n_tokens_vi=n_vi,
                        alignment_method=alignment_method,
                        key_pair_freq=key_pair_freq,
                        key_pair_share=key_pair_share,
                        alignment_confidence=alignment_confidence,
                    )
                continue

            # mismatch: không đoán alignment; giữ group để audit.
            add_pair(
                grain="attribute_group",
                unit_type="attribute",
                key_zh=key_zh,
                key_vi=key_vi,
                value_zh=group_zh,
                value_vi=group_vi,
                product_id=product_id,
                category=category,
                split=split,
                unit_index=attribute_index,
                value_index=-1,
                split_status=split_status,
                n_tokens_zh=n_zh,
                n_tokens_vi=n_vi,
                alignment_method=alignment_method,
                key_pair_freq=key_pair_freq,
                key_pair_share=key_pair_share,
                alignment_confidence=alignment_confidence,
            )

        if progress and (index + 1) % 2000 == 0:
            progress(f"  value pairs: {index + 1:,}/{len(rows):,}")

    pairs = pd.DataFrame(records)
    audit_frame = pd.DataFrame(audit)
    if pairs.empty:
        raise ValueError("Không tạo được cặp value-only nào từ 1688.")

    pairs["ratio_vi_zh"] = pairs["vi_chars"] / pairs["zh_chars"]
    pairs["vi_char_bucket"] = pairs["vi_chars"].map(
        lambda value: length_bucket(value, config.buckets)
    )
    pairs = pairs.reset_index(drop=True)
    audit_frame = audit_frame.reset_index(drop=True)

    status_counts = (
        audit_frame["split_status"].value_counts().to_dict()
        if not audit_frame.empty
        else {}
    )
    split_counts = pairs["split"].value_counts().to_dict()
    grain_counts = pairs["grain"].value_counts().to_dict()
    alignment_method_counts = (
        audit_frame["alignment_method"].value_counts().to_dict()
        if not audit_frame.empty
        else {}
    )
    alignment_confidence_counts = (
        audit_frame["alignment_confidence"].value_counts().to_dict()
        if not audit_frame.empty
        else {}
    )
    value_pairs = pairs[pairs["grain"] == "attribute_value"]
    value_class_counts = (
        value_pairs["value_class"].value_counts().to_dict()
        if not value_pairs.empty
        else {}
    )
    stats = {
        "source": SOURCE_1688,
        "n_products": int(pairs["product_id"].nunique()),
        "n_pairs_total": int(len(pairs)),
        "n_title_pairs": int((pairs["grain"] == "title").sum()),
        "n_value_pairs": int(len(value_pairs)),
        "n_group_pairs": int((pairs["grain"] == "attribute_group").sum()),
        "grain_counts": {str(k): int(v) for k, v in grain_counts.items()},
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "value_class_counts": {
            str(k): int(v) for k, v in value_class_counts.items()
        },
        "split_status_counts": {
            str(k): int(v) for k, v in status_counts.items()
        },
        "alignment_method_counts": {
            str(k): int(v) for k, v in alignment_method_counts.items()
        },
        "alignment_confidence_counts": {
            str(k): int(v) for k, v in alignment_confidence_counts.items()
        },
        "n_attribute_groups": int(len(audit_frame)),
        "description_aligned_rows": int(description_aligned_rows),
        "description_mismatch_rows": int(description_mismatch_rows),
    }
    return HumanValuePairs(
        pairs=pairs,
        split_audit=audit_frame,
        stats=stats,
    )


# ═══════════════════════════════════════════════════════════════════
# FIT
# ═══════════════════════════════════════════════════════════════════


def _summarize_group_v2(
    group: pd.DataFrame,
    percentiles: tuple[float, ...],
) -> dict[str, Any]:
    vi_chars = group["vi_chars"].to_numpy(dtype=float)
    vi_words = group["vi_words"].to_numpy(dtype=float)
    zh_chars = group["zh_chars"].to_numpy(dtype=float)
    zh_cjk = group["zh_cjk_chars"].to_numpy(dtype=float)
    zh_non_cjk = group["zh_non_cjk_chars"].to_numpy(dtype=float)
    ratios = group["ratio_vi_zh"].to_numpy(dtype=float)
    cjk_ratios = group["cjk_ratio"].to_numpy(dtype=float)
    cjk_positive = zh_cjk > 0
    ratio_vi_cjk = (
        vi_chars[cjk_positive] / zh_cjk[cjk_positive]
        if cjk_positive.any()
        else np.asarray([], dtype=float)
    )

    result: dict[str, Any] = {
        "n": int(len(group)),
        "n_cjk_positive": int(cjk_positive.sum()),
        "vi_chars_p05": round(float(np.percentile(vi_chars, 5)), 2),
        "vi_chars_p50": round(float(np.percentile(vi_chars, 50)), 2),
        "vi_chars_p95": round(float(np.percentile(vi_chars, 95)), 2),
        "vi_words_p50": round(float(np.percentile(vi_words, 50)), 2),
        "cjk_ratio_p50": round(float(np.percentile(cjk_ratios, 50)), 4),
        "zh_non_cjk_chars_p50": round(float(np.percentile(zh_non_cjk, 50)), 2),
    }
    result.update(_percentiles(zh_chars, percentiles, prefix="zh_chars"))
    result.update(_percentiles(zh_cjk, percentiles, prefix="zh_cjk_chars"))
    result.update(_percentiles(ratios, percentiles, prefix="ratio_vi_zh"))
    result.update(_percentiles(ratio_vi_cjk, percentiles, prefix="ratio_vi_cjk"))
    return result


def fit_length_prior_v2(
    pairs: pd.DataFrame,
    *,
    config: LengthPriorV2Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> LengthPriorV2:
    """Fit prior value-only, có value_class và split-safe."""
    config = config or LengthPriorV2Config()
    if config.fit_split not in {"train", "all"}:
        raise ValueError("fit_split phải là 'train' hoặc 'all'")
    required = (
        "grain",
        "unit_type",
        "value_class",
        "category",
        "split",
        "vi_chars",
        "vi_words",
        "zh_chars",
        "zh_cjk_chars",
        "zh_non_cjk_chars",
        "cjk_ratio",
        "ratio_vi_zh",
        "vi_char_bucket",
        "key_zh_chars",
        "key_class",
    )
    missing = [column for column in required if column not in pairs.columns]
    if missing:
        raise KeyError(f"Thiếu cột pairs v2: {missing}")

    primary = pairs[pairs["grain"].isin(PRIMARY_GRAINS)].copy()
    if config.fit_split != "all":
        primary = primary[primary["split"] == config.fit_split].copy()
    if primary.empty:
        raise ValueError(
            f"Không có cặp primary cho fit_split={config.fit_split!r}."
        )

    records: list[dict[str, Any]] = []

    def add_group(
        *,
        scope: str,
        group: pd.DataFrame,
        grain: str,
        unit_type: str,
        value_class: str = "",
        category: str = "",
        bucket: str = "",
    ) -> None:
        if group.empty:
            return
        record = {
            "scope": scope,
            "grain": grain,
            "unit_type": unit_type,
            "value_class": value_class,
            "category": category,
            "vi_char_bucket": bucket,
        }
        record.update(_summarize_group_v2(group, config.percentiles))
        records.append(record)

    add_group(
        scope="global",
        group=primary,
        grain="",
        unit_type="",
    )

    for grain, grain_group in primary.groupby("grain", dropna=False):
        grain = str(grain)
        unit_type = "title" if grain == "title" else "attribute"
        add_group(
            scope="unit_type",
            group=grain_group,
            grain=grain,
            unit_type=unit_type,
        )
        for bucket, bucket_group in grain_group.groupby(
            "vi_char_bucket", dropna=False
        ):
            add_group(
                scope="unit_type+bucket",
                group=bucket_group,
                grain=grain,
                unit_type=unit_type,
                bucket=str(bucket),
            )
        for category, category_group in grain_group.groupby(
            "category", dropna=False
        ):
            category = str(category)
            add_group(
                scope="unit_type+category",
                group=category_group,
                grain=grain,
                unit_type=unit_type,
                category=category,
            )
            for bucket, category_bucket_group in category_group.groupby(
                "vi_char_bucket", dropna=False
            ):
                add_group(
                    scope="unit_type+category+bucket",
                    group=category_bucket_group,
                    grain=grain,
                    unit_type=unit_type,
                    category=category,
                    bucket=str(bucket),
                )

        if grain == "attribute_value":
            for value_class, class_group in grain_group.groupby(
                "value_class", dropna=False
            ):
                value_class = str(value_class)
                add_group(
                    scope="value_class",
                    group=class_group,
                    grain=grain,
                    unit_type=unit_type,
                    value_class=value_class,
                )
                for bucket, class_bucket_group in class_group.groupby(
                    "vi_char_bucket", dropna=False
                ):
                    add_group(
                        scope="value_class+bucket",
                        group=class_bucket_group,
                        grain=grain,
                        unit_type=unit_type,
                        value_class=value_class,
                        bucket=str(bucket),
                    )
                for category, class_category_group in class_group.groupby(
                    "category", dropna=False
                ):
                    category = str(category)
                    add_group(
                        scope="value_class+category",
                        group=class_category_group,
                        grain=grain,
                        unit_type=unit_type,
                        value_class=value_class,
                        category=category,
                    )
                    for bucket, class_category_bucket_group in (
                        class_category_group.groupby(
                            "vi_char_bucket", dropna=False
                        )
                    ):
                        add_group(
                            scope="value_class+category+bucket",
                            group=class_category_bucket_group,
                            grain=grain,
                            unit_type=unit_type,
                            value_class=value_class,
                            category=category,
                            bucket=str(bucket),
                        )

        if progress:
            progress(f"  fit v2: {grain} ({len(grain_group):,} cặp)")

    table = pd.DataFrame(records)
    scope_order = {
        "global": 0,
        "unit_type": 1,
        "unit_type+bucket": 2,
        "unit_type+category": 3,
        "unit_type+category+bucket": 4,
        "value_class": 5,
        "value_class+bucket": 6,
        "value_class+category": 7,
        "value_class+category+bucket": 8,
    }
    table["_scope_order"] = table["scope"].map(scope_order).fillna(99).astype(int)
    table = table.sort_values(
        [
            "_scope_order",
            "grain",
            "unit_type",
            "value_class",
            "category",
            "vi_char_bucket",
        ],
        kind="stable",
    ).drop(columns="_scope_order").reset_index(drop=True)

    global_row = table[table["scope"] == "global"].iloc[0].to_dict()
    by_unit_type = {
        row["unit_type"]: row
        for row in table[table["scope"] == "unit_type"].to_dict("records")
    }
    by_value_class = {
        row["value_class"]: row
        for row in table[table["scope"] == "value_class"].to_dict("records")
    }

    key_rows = primary[
        (primary["grain"] == "attribute_value")
        & (primary["key_zh_chars"] > 0)
    ]
    key_chars_by_class: dict[str, Any] = {}
    for key_class, class_group in key_rows.groupby("key_class", dropna=False):
        values = class_group["key_zh_chars"].to_numpy(dtype=float)
        key_chars_by_class[str(key_class)] = {
            "n": int(len(values)),
            "p50": round(float(np.percentile(values, 50)), 2),
            "p95": round(float(np.percentile(values, 95)), 2),
        }
    key_chars_overall = (
        round(float(np.percentile(key_rows["key_zh_chars"], 50)), 2)
        if not key_rows.empty
        else 0.0
    )

    stats = {
        "source": SOURCE_1688,
        "version": 2,
        "target": "value_only",
        "fit_split": config.fit_split,
        "n_pairs": int(len(primary)),
        "n_products": int(primary["product_id"].nunique()),
        "grain_counts": {
            str(k): int(v) for k, v in primary["grain"].value_counts().items()
        },
        "split_counts": {
            str(k): int(v) for k, v in primary["split"].value_counts().items()
        },
        "value_class_counts": {
            str(k): int(v)
            for k, v in primary.loc[
                primary["grain"] == "attribute_value", "value_class"
            ].value_counts().items()
        },
        "overall": global_row,
        "by_unit_type": by_unit_type,
        "by_value_class": by_value_class,
        "key_chars": {
            "overall_p50": key_chars_overall,
            "by_key_class": key_chars_by_class,
        },
        "n_groups": int(len(table)),
        "vi_chars": _describe(primary["vi_chars"]),
        "zh_chars": _describe(primary["zh_chars"]),
        "zh_cjk_chars": _describe(primary["zh_cjk_chars"]),
        "ratio_vi_zh": _describe(primary["ratio_vi_zh"]),
        "config": asdict(config),
    }
    return LengthPriorV2(table=table, stats=stats, config=config)


# ═══════════════════════════════════════════════════════════════════
# APPLY TO TIKI
# ═══════════════════════════════════════════════════════════════════


def _lookup_v2(
    table: pd.DataFrame,
) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row["scope"]),
            str(row["grain"]),
            str(row["unit_type"]),
            str(row["value_class"]),
            str(row["category"]),
            str(row["vi_char_bucket"]),
        ): row
        for row in table.to_dict("records")
    }


def _select_prior_v2(
    lookup: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    *,
    grain: str,
    unit_type: str,
    value_class: str,
    category: str,
    bucket: str,
    config: LengthPriorV2Config,
) -> tuple[dict[str, Any] | None, str]:
    candidates: list[tuple[str, str, str, str, str, str, int]] = []

    def add(
        scope: str,
        *,
        grain_value: str = grain,
        unit_type_value: str = unit_type,
        value_class_value: str = "",
        category_value: str = "",
        bucket_value: str = "",
        min_samples: int = 1,
    ) -> None:
        candidates.append((
            scope,
            grain_value,
            unit_type_value,
            value_class_value,
            category_value,
            bucket_value,
            min_samples,
        ))

    if grain == "attribute_value" and value_class:
        if category:
            add(
                "value_class+category+bucket",
                value_class_value=value_class,
                category_value=category,
                bucket_value=bucket,
                min_samples=max(
                    config.min_value_class_category_samples,
                    config.min_category_samples,
                ),
            )
        add(
            "value_class+bucket",
            value_class_value=value_class,
            bucket_value=bucket,
            min_samples=config.min_value_class_samples,
        )
        add(
            "value_class",
            value_class_value=value_class,
            min_samples=config.min_value_class_samples,
        )
        if category:
            add(
                "value_class+category",
                value_class_value=value_class,
                category_value=category,
                min_samples=config.min_value_class_category_samples,
            )

    if category:
        add(
            "unit_type+category+bucket",
            category_value=category,
            bucket_value=bucket,
            min_samples=max(
                config.min_group_samples,
                config.min_category_samples,
            ),
        )
    add(
        "unit_type+bucket",
        bucket_value=bucket,
        min_samples=config.min_group_samples,
    )
    # Bucket là điều kiện quan trọng hơn category; dùng bucket ít mẫu trước
    # category/unit_type để tránh dự đoán lệch do title ngắn.
    add(
        "unit_type+bucket",
        bucket_value=bucket,
        min_samples=1,
    )
    if category:
        add(
            "unit_type+category",
            category_value=category,
            min_samples=config.min_category_samples,
        )
    add("unit_type", min_samples=config.min_group_samples)
    add("global", grain_value="", unit_type_value="", min_samples=1)

    for scope, g, u, vc, cat, b, min_samples in candidates:
        record = lookup.get((scope, g, u, vc, cat, b))
        if record is not None and int(record["n"]) >= min_samples:
            return record, scope
    return None, "missing"


def _estimated_key_chars(
    row: dict[str, Any],
    prior: LengthPriorV2,
    *,
    key_zh: str,
    key_vi: str,
    unit_type: str,
) -> tuple[int, str]:
    if unit_type == "title":
        return 0, "title"
    if key_zh:
        return measure_text(key_zh)[0], "actual"
    key_class = str(row.get("key_class", "") or "").strip()
    if not key_class and key_vi:
        key_class = classify_key(key_vi)
    key_chars = prior.stats.get("key_chars", {})
    by_class = key_chars.get("by_key_class", {})
    if key_class in by_class:
        return int(round(float(by_class[key_class]["p50"]))), "estimated_key_class"
    overall = float(key_chars.get("overall_p50", 0) or 0)
    if overall > 0:
        return int(round(overall)), "estimated_overall"
    return 0, "missing"


def apply_length_prior_v2(
    units: pd.DataFrame,
    prior: LengthPriorV2,
    *,
    config: LengthPriorV2Config | None = None,
    progress: Callable[[str], None] | None = None,
) -> LengthPriorV2ApplyResult:
    """Gắn prior value-only + key correction + CJK range cho Tiki units."""
    config = config or prior.config or LengthPriorV2Config()
    if "text_vi" not in units.columns:
        raise KeyError("Thiếu cột text_vi trong units.")
    if "unit_type" not in units.columns:
        raise KeyError("Thiếu cột unit_type trong units.")

    if units.empty:
        result = units.copy()
        defaults = {
            "length_prior_source": SOURCE_1688,
            "length_prior_version": 2,
            "length_prior_grain": "missing",
            "length_prior_value_class": "",
            "length_prior_scope": "missing",
            "length_prior_n": 0,
            "length_prior_confidence": "missing",
            "length_prior_soft_qc": False,
            "vi_value_chars": 0,
            "vi_value_words": 0,
            "vi_text_chars": 0,
            "vi_char_bucket": "0",
            "ratio_vi_zh_p05": 0.0,
            "ratio_vi_zh_p50": 0.0,
            "ratio_vi_zh_p95": 0.0,
            "ratio_vi_cjk_p05": 0.0,
            "ratio_vi_cjk_p50": 0.0,
            "ratio_vi_cjk_p95": 0.0,
            "expected_zh_value_chars": 0,
            "zh_value_chars_p05": 0,
            "zh_value_chars_p95": 0,
            "expected_zh_cjk_chars": 0,
            "zh_cjk_chars_p05": 0,
            "zh_cjk_chars_p95": 0,
            "expected_zh_key_chars": 0,
            "key_len_source": "missing",
            "expected_zh_text_chars": 0,
            "zh_text_chars_p05": 0,
            "zh_text_chars_p95": 0,
        }
        for column, default in defaults.items():
            result[column] = default
        return LengthPriorV2ApplyResult(
            units=result,
            stats={"n_units": 0, "source": SOURCE_1688, "version": 2},
        )

    lookup = _lookup_v2(prior.table)
    records: list[dict[str, Any]] = []
    rows = units.to_dict("records")

    for index, row in enumerate(rows):
        unit_type = str(row.get("unit_type", "") or "").strip()
        grain = "title" if unit_type == "title" else "attribute_value"
        value_class = str(row.get("value_class", "") or "").strip()
        category = str(row.get("category", "") or "").strip()
        key_zh = str(row.get("key_zh", "") or "").strip()
        key_vi = str(row.get("key_vi", "") or "").strip()

        if unit_type == "attribute":
            value_vi = row.get("value_vi")
            if value_vi is None or not str(value_vi).strip():
                value_vi = row.get("text_vi")
        else:
            value_vi = row.get("text_vi")

        vi_value_chars, vi_value_words = measure_text(value_vi)
        vi_text_chars, _ = measure_text(row.get("text_vi"))
        bucket = length_bucket(vi_value_chars, config.buckets)

        stat, scope = _select_prior_v2(
            lookup,
            grain=grain,
            unit_type=unit_type,
            value_class=value_class,
            category=category,
            bucket=bucket,
            config=config,
        )

        if stat is None or vi_value_chars <= 0:
            expected_value = value_low = value_high = 0
            expected_cjk = cjk_low = cjk_high = 0
            ratio_p05 = ratio_p50 = ratio_p95 = 0.0
            cjk_p05 = cjk_p50 = cjk_p95 = 0.0
            prior_n = 0
        else:
            ratio_p05 = float(stat["ratio_vi_zh_p05"])
            ratio_p50 = float(stat["ratio_vi_zh_p50"])
            ratio_p95 = float(stat["ratio_vi_zh_p95"])
            cjk_p05 = float(stat["ratio_vi_cjk_p05"])
            cjk_p50 = float(stat["ratio_vi_cjk_p50"])
            cjk_p95 = float(stat["ratio_vi_cjk_p95"])
            prior_n = int(stat["n"])

            if ratio_p50 > 0:
                expected_value = max(1, int(round(vi_value_chars / ratio_p50)))
                value_low = (
                    max(1, int(math.floor(vi_value_chars / ratio_p95)))
                    if ratio_p95 > 0
                    else expected_value
                )
                value_high = (
                    max(1, int(math.ceil(vi_value_chars / ratio_p05)))
                    if ratio_p05 > 0
                    else expected_value
                )
                value_low = min(value_low, expected_value)
                value_high = max(value_high, expected_value)
            else:
                expected_value = value_low = value_high = 0

            if cjk_p50 > 0:
                expected_cjk = max(1, int(round(vi_value_chars / cjk_p50)))
                cjk_low = (
                    max(1, int(math.floor(vi_value_chars / cjk_p95)))
                    if cjk_p95 > 0
                    else expected_cjk
                )
                cjk_high = (
                    max(1, int(math.ceil(vi_value_chars / cjk_p05)))
                    if cjk_p05 > 0
                    else expected_cjk
                )
                cjk_low = min(cjk_low, expected_cjk)
                cjk_high = max(cjk_high, expected_cjk)
            else:
                expected_cjk = cjk_low = cjk_high = 0

        key_chars, key_len_source = _estimated_key_chars(
            row,
            prior,
            key_zh=key_zh,
            key_vi=key_vi,
            unit_type=unit_type,
        )
        separator = config.separator_chars if key_chars > 0 else 0
        expected_text = expected_value + key_chars + separator
        text_low = value_low + key_chars + separator
        text_high = value_high + key_chars + separator

        if prior_n >= max(config.min_group_samples, config.min_value_class_samples):
            confidence = "high"
        elif prior_n >= 5:
            confidence = "medium"
        elif prior_n > 0:
            confidence = "low"
        else:
            confidence = "missing"

        records.append({
            "length_prior_source": SOURCE_1688,
            "length_prior_version": 2,
            "length_prior_grain": grain,
            "length_prior_value_class": value_class,
            "length_prior_scope": scope,
            "length_prior_n": prior_n,
            "length_prior_confidence": confidence,
            "length_prior_soft_qc": bool(
                (unit_type == "title" and config.title_soft_qc)
                or value_class in {"admin", "model", "numeric"}
            ),
            "vi_value_chars": int(vi_value_chars),
            "vi_value_words": int(vi_value_words),
            "vi_text_chars": int(vi_text_chars),
            "vi_char_bucket": bucket,
            "ratio_vi_zh_p05": round(ratio_p05, 4),
            "ratio_vi_zh_p50": round(ratio_p50, 4),
            "ratio_vi_zh_p95": round(ratio_p95, 4),
            "ratio_vi_cjk_p05": round(cjk_p05, 4),
            "ratio_vi_cjk_p50": round(cjk_p50, 4),
            "ratio_vi_cjk_p95": round(cjk_p95, 4),
            "expected_zh_value_chars": int(expected_value),
            "zh_value_chars_p05": int(value_low),
            "zh_value_chars_p95": int(value_high),
            "expected_zh_cjk_chars": int(expected_cjk),
            "zh_cjk_chars_p05": int(cjk_low),
            "zh_cjk_chars_p95": int(cjk_high),
            "expected_zh_key_chars": int(key_chars),
            "key_len_source": key_len_source,
            "expected_zh_text_chars": int(expected_text),
            "zh_text_chars_p05": int(text_low),
            "zh_text_chars_p95": int(text_high),
        })

        if progress and (index + 1) % 20000 == 0:
            progress(f"  apply v2: {index + 1:,}/{len(rows):,}")

    prior_frame = pd.DataFrame(records)
    result = units.copy().reset_index(drop=True)
    for column in prior_frame.columns:
        result[column] = prior_frame[column].to_numpy()

    stats = {
        "n_units": int(len(result)),
        "source": SOURCE_1688,
        "version": 2,
        "grain_counts": {
            str(k): int(v)
            for k, v in prior_frame["length_prior_grain"].value_counts().items()
        },
        "value_class_counts": {
            str(k): int(v)
            for k, v in prior_frame["length_prior_value_class"].value_counts().items()
        },
        "scope_counts": {
            str(k): int(v)
            for k, v in prior_frame["length_prior_scope"].value_counts().items()
        },
        "key_len_source_counts": {
            str(k): int(v)
            for k, v in prior_frame["key_len_source"].value_counts().items()
        },
        "confidence_counts": {
            str(k): int(v)
            for k, v in prior_frame["length_prior_confidence"].value_counts().items()
        },
        "n_zero_value_chars": int((prior_frame["vi_value_chars"] <= 0).sum()),
        "expected_zh_value_chars": _describe(
            prior_frame["expected_zh_value_chars"]
        ),
        "expected_zh_text_chars": _describe(
            prior_frame["expected_zh_text_chars"]
        ),
        "expected_zh_cjk_chars": _describe(
            prior_frame["expected_zh_cjk_chars"]
        ),
        "ratio_vi_zh_p50": _describe(prior_frame["ratio_vi_zh_p50"]),
    }
    return LengthPriorV2ApplyResult(units=result, stats=stats)


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════


def _metric_block(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "n": 0,
            "coverage_value_p05_p95": 0.0,
            "mae_value_chars": 0.0,
            "mean_abs_error": 0.0,
        }
    actual = frame["zh_chars"].to_numpy(dtype=float)
    expected = frame["expected_zh_value_chars"].to_numpy(dtype=float)
    low = frame["zh_value_chars_p05"].to_numpy(dtype=float)
    high = frame["zh_value_chars_p95"].to_numpy(dtype=float)
    in_range = (actual >= low) & (actual <= high)
    errors = np.abs(actual - expected)
    return {
        "n": int(len(frame)),
        "coverage_value_p05_p95": round(float(in_range.mean()), 4),
        "mae_value_chars": round(float(errors.mean()), 3),
        "median_abs_error": round(float(np.median(errors)), 3),
    }


def evaluate_length_prior_v2(
    pairs: pd.DataFrame,
    prior: LengthPriorV2,
    *,
    split: str,
    config: LengthPriorV2Config | None = None,
) -> dict[str, Any]:
    """Đánh giá prior trên dev/test, so với actual value-only chars."""
    config = config or prior.config or LengthPriorV2Config()
    test = pairs[
        pairs["grain"].isin(PRIMARY_GRAINS) & (pairs["split"] == split)
    ].copy()
    if test.empty:
        return {"split": split, "n": 0}

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
    applied = apply_length_prior_v2(units, prior, config=config).units
    evaluated = test.reset_index(drop=True).copy()
    for column in (
        "expected_zh_value_chars",
        "zh_value_chars_p05",
        "zh_value_chars_p95",
        "expected_zh_text_chars",
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
        "n": int(len(evaluated)),
        "overall": _metric_block(evaluated),
        "by_grain": by_grain,
        "by_value_class": by_value_class,
    }
