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
# Gồm Latin-1/Extended-A/B và Latin Extended Additional (nguyên âm tiếng Việt).
_LATIN_CLASS = r"A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F\u1E00-\u1EFF"
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

# Ngoặc trang trí thường gặp trong dữ liệu TMĐT -> ASCII, theo quy ước bitext.
_BRACKET_MAP = str.maketrans({
    "【": "[", "】": "]",
    "〖": "[", "〗": "]",
    "〔": "[", "〕": "]",
    "丨": "|",
})


def normalize_brackets(text: object) -> str:
    """Chuẩn hoá ngoặc trang trí: 【】〖〗〔〕 -> [] và 丨 -> |."""
    if not isinstance(text, str):
        return ""
    return text.translate(_BRACKET_MAP)

# ── Chuẩn hoá ký tự nhiễu ──────────────────────────────────────────────
_BRACKET_MAP = str.maketrans({
    "【": "[", "】": "]",
    "〖": "[", "〗": "]",
    "〔": "[", "〕": "]",
    "［": "[", "］": "]",
    "｛": "{", "｝": "}",
    "（": "(", "）": ")",
    "〈": "<", "〉": ">",
    "丨": "|",
})
_FULLWIDTH_ALNUM = str.maketrans(
    {chr(0xFF10 + i): str(i) for i in range(10)}
    | {chr(0xFF21 + i): chr(0x41 + i) for i in range(26)}
    | {chr(0xFF41 + i): chr(0x61 + i) for i in range(26)}
)
_INVISIBLE_RE = re.compile(
    r"[\u200b-\u200f\u2028-\u202f\u2060-\u2065\ufeff\u00ad\u034f\u061c\u180e"
    r"\u202a-\u202e\u2066-\u2069\ufe00-\ufe0f\U000E0100-\U000E01EF"
    r"\ue000-\uf8ff"
    r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_SPECIAL_SPACE_RE = re.compile(r"[\u2000-\u200a\u202f\u205f\u3000\u00a0]")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
_TEMPLATE_RE = re.compile(r"\{([^{}\n]{1,60})\}")
_REPEAT_PUNCT_RE = re.compile(r"([!?。，,;:])\1+")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]{0,80}>")
_HTML_ENTITY_RE = re.compile(
    r"&(amp|lt|gt|quot|apos|nbsp|copy|reg|trade|hellip|mdash|ndash|ldquo|rdquo"
    r"|lsquo|rsquo|times|divide|deg|middot|bull|laquo|raquo|#\d+|#x[0-9a-fA-F]+);"
)
_ESCAPE_RE = re.compile(r"\\n|\\t|\\u[0-9a-fA-F]{4}")

_BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("《", "》"),
                  ("「", "」"), ("『", "』")]


def normalize_brackets(text: object) -> str:
    """Chuẩn hoá ngoặc full-width/trang trí về ASCII."""
    if not isinstance(text, str):
        return ""
    return text.translate(_BRACKET_MAP)


def remove_unmatched_brackets(text: str) -> str:
    """Bỏ ngoặc lẻ (không có cặp) trong cùng một chuỗi."""
    for opener, closer in _BRACKET_PAIRS:
        if text.count(opener) == text.count(closer):
            continue
        stack: list[int] = []
        drop: set[int] = set()
        for index, char in enumerate(text):
            if char == opener:
                stack.append(index)
            elif char == closer:
                if stack:
                    stack.pop()
                else:
                    drop.add(index)
        drop.update(stack)
        if drop:
            text = "".join(ch for i, ch in enumerate(text) if i not in drop)
    return text


def clean_noise_characters(text: object) -> str:
    """Chuẩn hoá/loại ký tự nhiễu; giữ dấu câu và ký hiệu có nghĩa."""
    if not isinstance(text, str):
        return ""
    value = unicodedata.normalize("NFC", text)
    value = value.translate(_FULLWIDTH_ALNUM)
    value = value.translate(_BRACKET_MAP)
    value = _HTML_TAG_RE.sub(" ", value)
    value = _HTML_ENTITY_RE.sub(" ", value)
    value = _INVISIBLE_RE.sub("", value)
    value = _SPECIAL_SPACE_RE.sub(" ", value)
    value = _EMOJI_RE.sub(" ", value)
    value = _TEMPLATE_RE.sub(r"\1", value)
    value = _ESCAPE_RE.sub(" ", value)
    value = _REPEAT_PUNCT_RE.sub(r"\1", value)
    value = _LONG_DOTS_RE.sub("...", value)
    value = remove_unmatched_brackets(value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()

