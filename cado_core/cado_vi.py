# -*- coding: utf-8 -*-
"""CADO-VI — xoá từ tiếng Anh dư thừa trong văn bản tiếng Việt.

Nguyên tắc giống CADO: chỉ xoá khi **bản dịch tiếng Việt của từ đó đã xuất
hiện trong chính văn bản**. Mapping nằm trong ``csv/vi_english_map.csv`` để
có thể review/sửa mà không cần đụng code.

Ví dụ: ``"Túi xách nữ bags"`` → ``"Túi xách nữ"`` (vì đã có "túi xách"),
nhưng ``"Balo thể thao bags"`` vẫn giữ ``bags`` nếu không có "túi"/"ba lô".
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import (
    DESC_VI,
    DESC_VI_CLEANED,
    TITLE_VI,
    TITLE_VI_CLEANED,
    VI_MAP_CSV,
)
from .text import extract_latin_words, normalize_text, remove_words, word_pattern

_VI_CLEANED_COLUMNS = {
    TITLE_VI: TITLE_VI_CLEANED,
    DESC_VI: DESC_VI_CLEANED,
}

_TRAILING_LATIN = re.compile(r"([A-Za-z]+)$")
_LEADING_LATIN = re.compile(r"^([A-Za-z]+)")


@dataclass
class ViCleanConfig:
    columns: tuple[str, ...] = (TITLE_VI, DESC_VI)
    map_csv: Path = VI_MAP_CSV
    min_word_len: int = 3
    normalize: bool = True
    # Không xoá từ nằm trong cụm Latin (brand nhiều từ: "MAKI BAG", "Bio-Mask").
    avoid_latin_phrases: bool = True
    progress: Callable[[str], None] | None = None


@dataclass
class ViCleanResult:
    df: pd.DataFrame
    occurrences: pd.DataFrame
    stats: dict = field(default_factory=dict)


def load_vi_map(path: str | Path = VI_MAP_CSV) -> dict[str, tuple[str, ...]]:
    """Đọc mapping ``word → các cách viết tiếng Việt``.

    File CSV có cột ``word,vi_equivalents,category,reason``; nhiều cách viết
    được ngăn bởi ``|``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mapping CADO-VI: {path}")

    mapping: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            word = (row.get("word") or "").strip().lower()
            raw = (row.get("vi_equivalents") or "").strip()
            if not word or not raw:
                continue
            equivalents = tuple(
                part.strip().lower() for part in raw.split("|") if part.strip()
            )
            if equivalents:
                mapping[word] = equivalents
    return mapping


def _is_brand_token(token: str) -> bool:
    """Token Latin viết hoa (MAKI, Bio, New) → có thể là brand, không xoá."""
    return (
        len(token) >= 2
        and all("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in token)
        and (token.isupper() or token[0].isupper())
    )


def _inside_latin_phrase(text: str, word: str) -> bool:
    """True nếu ``word`` nằm cạnh một token Latin viết hoa (brand nhiều từ)."""
    for match in word_pattern(word).finditer(text):
        before = text[: match.start()].rstrip(" -_/")
        after = text[match.end():].lstrip(" -_/")

        before_match = _TRAILING_LATIN.search(before)
        if before_match and _is_brand_token(before_match.group(1)):
            return True

        after_match = _LEADING_LATIN.match(after)
        if after_match and _is_brand_token(after_match.group(1)):
            return True
    return False


def clean_vi_text(
    text: object,
    mapping: dict[str, tuple[str, ...]],
    *,
    min_word_len: int = 3,
    avoid_latin_phrases: bool = True,
) -> tuple[str, list[str]]:
    """Trả về ``(text đã làm sạch, danh sách từ bị xoá)``."""
    source = normalize_text(text)
    text_lower = source.lower()
    to_remove: list[str] = []

    for word in extract_latin_words(source):
        if len(word) < min_word_len:
            continue
        equivalents = mapping.get(word.lower())
        if not equivalents:
            continue
        if not any(equivalent in text_lower for equivalent in equivalents):
            continue
        if avoid_latin_phrases and _inside_latin_phrase(source, word):
            continue
        to_remove.append(word)

    if not to_remove:
        return source, []
    return remove_words(source, set(to_remove)), to_remove


def run_cado_vi(
    df: pd.DataFrame,
    cfg: ViCleanConfig | None = None,
    *,
    mapping: dict[str, tuple[str, ...]] | None = None,
) -> ViCleanResult:
    cfg = cfg or ViCleanConfig()
    mapping = mapping if mapping is not None else load_vi_map(cfg.map_csv)

    missing = [c for c in cfg.columns if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột trong DataFrame: {missing}")

    out = df.copy()
    occurrences: list[dict] = []
    changed_by_column: Counter[str] = Counter()

    for column in cfg.columns:
        cleaned_column = _VI_CLEANED_COLUMNS.get(column, f"{column}_cleaned")
        cleaned_values: list[str] = []

        for row_index, raw in enumerate(out[column].tolist()):
            cleaned, removed = clean_vi_text(
                raw,
                mapping,
                min_word_len=cfg.min_word_len,
                avoid_latin_phrases=cfg.avoid_latin_phrases,
            )
            cleaned_values.append(cleaned)
            if cleaned != normalize_text(raw):
                changed_by_column[column] += 1
            for word in removed:
                occurrences.append({
                    "row": row_index,
                    "product_id": str(out["product_id"].iloc[row_index])
                    if "product_id" in out.columns else "",
                    "column": column,
                    "word": word,
                    "reason": "vi_equivalent_present",
                })

        out[cleaned_column] = cleaned_values

    stats = {
        "total_removed": len(occurrences),
        "changed_by_column": dict(changed_by_column),
        "removed_words": Counter(o["word"] for o in occurrences).most_common(30),
        "map_size": len(mapping),
    }

    return ViCleanResult(
        df=out,
        occurrences=pd.DataFrame(occurrences),
        stats=stats,
    )
