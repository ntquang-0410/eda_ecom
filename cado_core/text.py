# -*- coding: utf-8 -*-
"""Tiện ích xử lý văn bản Trung–Việt.

Điểm quan trọng: ranh giới từ Latin **không** dùng ``\\b`` vì Python coi
chữ Hán là ký tự word. Ta dùng lookbehind/lookahead chỉ loại trừ chữ Latin,
nhờ đó xoá được ``bags`` trong ``双肩包bags男`` nhưng không đụng ``handbags``.
"""

from __future__ import annotations

import re
import unicodedata

from .config import (
    CJK_BLOCK_RE,
    CJK_CHAR_RE,
    INLINE_SPACE_RE,
    LATIN_WORD_RE,
    ZERO_WIDTH_RE,
)

# Ký tự được phép nằm sát nhau như một phần của cùng một "từ Latin".
_LATIN_CLASS = r"A-Za-z\u00C0-\u024F"
_WORD_PATTERN_CACHE: dict[tuple[str, bool], re.Pattern[str]] = {}


# ═══════════════════════════════════════════════════════════════════
# NORMALIZE
# ═══════════════════════════════════════════════════════════════════

def normalize_text(text: object, *, collapse_whitespace: bool = False) -> str:
    """Chuẩn hoá NFC, bỏ ký tự vô hình và NBSP.

    ``collapse_whitespace=True`` chỉ dùng cho title; mặc định giữ nguyên
    newline để không phá format bảng trong ``description_zh``.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def has_latin(text: object) -> bool:
    return bool(LATIN_WORD_RE.search(normalize_text(text)))


def has_cjk(text: object) -> bool:
    return bool(CJK_CHAR_RE.search(normalize_text(text)))


def count_cjk(text: object) -> int:
    return len(CJK_CHAR_RE.findall(normalize_text(text)))


def count_vi_tokens(text: object) -> int:
    return len(normalize_text(text).split())


def strip_cjk(text: object) -> str:
    """Xoá toàn bộ khối CJK khỏi văn bản (dùng cho cột tiếng Việt).

    Giữ newline để không phá format description.
    """
    cleaned = CJK_BLOCK_RE.sub("", normalize_text(text))
    cleaned = INLINE_SPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return cleaned.strip(" \t")


# ═══════════════════════════════════════════════════════════════════
# EXTRACT
# ═══════════════════════════════════════════════════════════════════

def extract_latin_words(text: object) -> list[str]:
    """Trả về các chuỗi Latin xuất hiện trong text, theo thứ tự."""
    return LATIN_WORD_RE.findall(normalize_text(text))


# ═══════════════════════════════════════════════════════════════════
# REMOVE
# ═══════════════════════════════════════════════════════════════════

def word_pattern(word: str, *, ignore_case: bool = True) -> re.Pattern[str]:
    """Regex khớp nguyên từ, coi CJK/dấu câu/số là ranh giới hợp lệ.

    Chỉ chữ cái Latin bị loại khỏi ranh giới, nên:
      - ``bags`` khớp trong ``包bags男`` và ``包 bags 男``
      - ``bags`` **không** khớp trong ``handbags``
      - ``cm`` khớp trong ``x200cm`` (vì ``0`` không phải chữ cái)
    """
    key = (word, ignore_case)
    cached = _WORD_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached

    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(
        rf"(?<![{_LATIN_CLASS}]){re.escape(word)}(?![{_LATIN_CLASS}])",
        flags,
    )
    _WORD_PATTERN_CACHE[key] = pattern
    return pattern


def _tidy_after_removal(text: str) -> str:
    """Dọn khoảng trắng thừa nhưng giữ nguyên newline / format bảng."""
    text = INLINE_SPACE_RE.sub(" ", text)
    # "包 bags 男" → "包男" chỉ khi hai bên đều là CJK.
    text = re.sub(rf"(?<={CJK_CHAR_RE.pattern})[ \t]+(?={CJK_CHAR_RE.pattern})", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip(" \t")


def remove_words(text: object, words: set[str] | list[str] | tuple[str, ...]) -> str:
    """Xoá danh sách từ Latin khỏi text, an toàn với ranh giới CJK.

    Từ dài được xoá trước để tránh xoá substring của từ khác.
    Không thay đổi newline (an toàn cho description).
    """
    cleaned = normalize_text(text)
    if not words:
        return cleaned

    unique_words = {w.strip() for w in words if isinstance(w, str) and w.strip()}
    for word in sorted(unique_words, key=len, reverse=True):
        cleaned = word_pattern(word).sub("", cleaned)

    return _tidy_after_removal(cleaned)


# Alias giữ tương thích với code cũ.
clean_text = remove_words
