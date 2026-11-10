# -*- coding: utf-8 -*-
"""Luật phân loại của CADO: pattern, auto loanword, keep/remove.

Thứ tự tầng là bất biến (strict ordering):
    T1 whitelist → T2 pattern → T3 loanword → T4 CEDICT → T5 bge-m3
Một từ đã được giữ ở tầng trước sẽ không bao giờ bị tầng sau xoá.
"""

from __future__ import annotations

import re

from .dictionary import Cedict, Overlap, Whitelist
from .text import extract_latin_words, normalize_text

# ═══════════════════════════════════════════════════════════════════
# TẦNG 2 — PATTERN
# ═══════════════════════════════════════════════════════════════════

_MODEL_CODE_RE = re.compile(r"\d")
_SHORT_CAPS_RE = re.compile(r"^[A-Z]{1,5}$")
_CAMEL_BRAND_RE = re.compile(r"^[A-Z][a-z]+[A-Z]")


def pattern_keep(word: str) -> str | None:
    """Trả về tên pattern nếu từ thuộc nhóm luôn giữ, ngược lại ``None``."""
    if not word:
        return None
    if _MODEL_CODE_RE.search(word):
        return "model_code"           # iPhone14, x200cm, M416
    if _SHORT_CAPS_RE.match(word):
        return "short_abbreviation"   # USB, ABS, PU
    if _CAMEL_BRAND_RE.match(word):
        return "camelcase_brand"      # MagSafe, ProMax
    if len(word) <= 2:
        return "too_short"            # AI, UV, 5G (đã qua model_code)
    return None


# ═══════════════════════════════════════════════════════════════════
# TẦNG 3 — AUTO LOANWORD
# ═══════════════════════════════════════════════════════════════════

def build_candidates(
    rows: list[tuple[str, str]],
    whitelist: Whitelist,
) -> list[dict]:
    """Gom các từ chưa được T1/T2 quyết định, kèm ngữ cảnh xuất hiện.

    ``rows`` là danh sách ``(product_id, text)``.
    """
    pool: dict[str, dict] = {}

    for product_id, text in rows:
        for word in extract_latin_words(text):
            if word in whitelist or pattern_keep(word):
                continue
            record = pool.get(word)
            if record is None:
                record = pool[word] = {
                    "word": word,
                    "total_count": 0,
                    "contexts": [],
                }
            record["total_count"] += 1
            record["contexts"].append((product_id, normalize_text(text)))

    return sorted(pool.values(), key=lambda r: -r["total_count"])


def detect_loanwords(
    candidates: list[dict],
    cedict: Cedict,
    *,
    min_freq: int = 3,
    overlap_threshold: float = 0.20,
    progress=None,
) -> tuple[set[str], list[dict]]:
    """Phát hiện từ mượn: xuất hiện ≥ ``min_freq`` nhưng hiếm khi có bản dịch
    tiếng Trung trong câu (overlap rate < ``overlap_threshold``).

    Trả về ``(tập từ mượn lowercase, report rows)``.
    """
    loanwords: set[str] = set()
    report: list[dict] = []
    cache: dict[tuple[str, str], Overlap] = {}

    for index, candidate in enumerate(candidates, 1):
        word = candidate["word"]
        count = candidate["total_count"]

        if count < min_freq:
            report.append({
                "word": word,
                "total_count": count,
                "overlap_count": 0,
                "overlap_rate": 0.0,
                "decision": "insufficient_data",
                "sample_match": "",
            })
            continue

        overlap_count = 0
        sample_match = ""
        for _pid, text in candidate["contexts"]:
            key = (word, text)
            result = cache.get(key)
            if result is None:
                result = cedict.overlap(word, text)
                cache[key] = result
            if result.overlap:
                overlap_count += 1
                if not sample_match:
                    sample_match = result.match

        overlap_rate = overlap_count / count
        is_loanword = overlap_rate < overlap_threshold
        if is_loanword:
            loanwords.add(word.lower())

        report.append({
            "word": word,
            "total_count": count,
            "overlap_count": overlap_count,
            "overlap_rate": round(overlap_rate, 3),
            "decision": "loanword_keep" if is_loanword else "not_loanword",
            "sample_match": sample_match,
        })

        if progress and index % 500 == 0:
            progress(f"    ... đã phân tích {index:,}/{len(candidates):,} từ")

    return loanwords, report


# ═══════════════════════════════════════════════════════════════════
# PHÂN LOẠI MỘT LẦN XUẤT HIỆN
# ═══════════════════════════════════════════════════════════════════

def classify_occurrence(
    word: str,
    text: str,
    whitelist: Whitelist,
    loanwords: set[str] | frozenset[str],
    cedict: Cedict,
    *,
    max_tier: int = 4,
    use_loanwords: bool = True,
) -> tuple[str, str]:
    """Quyết định ``(action, reason)`` cho một từ trong một văn bản.

    ``action`` ∈ {``keep``, ``remove``, ``pending``}. ``pending`` chỉ xuất hiện
    khi pipeline dừng sớm (``max_tier`` < 4); khi đó từ được coi là GIỮ.
    """
    if not word:
        return "keep", "empty"

    # T1 — whitelist
    if word in whitelist:
        return "keep", "whitelist"

    if max_tier < 2:
        return "pending", "stopped_at_t1"

    # T2 — pattern
    pattern = pattern_keep(word)
    if pattern:
        return "keep", f"pattern:{pattern}"

    if max_tier < 3:
        return "pending", "stopped_before_t3"

    # T3 — auto loanword (có thể tắt; khi tắt thì đi tiếp xuống T4)
    if use_loanwords and word.lower() in loanwords:
        return "keep", "auto_loanword"

    if max_tier < 4:
        return "pending", "stopped_before_t4"

    # T4 — CEDICT context overlap
    result = cedict.overlap(word, text)
    if result.overlap:
        return "remove", f"cedict:{result.match}"
    if result.status == "no_entry":
        return "keep", "no_cedict_entry"
    return "keep", "cedict_no_overlap"


# ═══════════════════════════════════════════════════════════════════
# REPORT HELPERS
# ═══════════════════════════════════════════════════════════════════

def aggregate_word_report(occurrences) -> list[dict]:
    """Tổng hợp quyết định theo từ để lưu CSV."""
    stats: dict[tuple[str, str], dict] = {}

    for row in occurrences:
        key = (row["word"], row["action"])
        record = stats.get(key)
        if record is None:
            record = stats[key] = {
                "word": row["word"],
                "action": row["action"],
                "reason": row["reason"],
                "count": 0,
                "columns": set(),
            }
        record["count"] += 1
        record["columns"].add(row["column"])

    return [
        {**record, "columns": "|".join(sorted(record["columns"]))}
        for record in sorted(stats.values(), key=lambda r: (-r["count"], r["word"].lower()))
    ]
