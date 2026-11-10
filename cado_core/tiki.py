# -*- coding: utf-8 -*-
"""EDA cho corpus Tiki (đơn ngữ tiếng Việt) trước back-translation VI → ZH.

Module này **không** gọi model. Nó làm sạch, đo chất lượng, gắn cờ và chia
route cho từng sản phẩm Tiki, xuất ra một bảng product-level để có thể dùng
trực tiếp cho bước dịch ngược trên Kaggle/Colab.

Điểm khác biệt so với CADO:
  * Tiki không có ``title_zh`` / ``description_zh`` (rỗng 100%), nên đây là
    corpus đơn ngữ VI, không thể chấm cosine cặp ZH–VI.
  * ``description_vi`` là chuỗi thuộc tính ``key: value; ...``, còn
    ``description_prose_vi`` mới là văn xuôi. Hai cột này được xử lý riêng.
  * Không dùng ``title_vi_cleaned`` của CADO-VI vì bản này đã xoá nhầm một số
    từ nằm trong tên sản phẩm (ví dụ ``Shoes Dryer`` → ``Dryer``).
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import WHITELIST_CSV
from .dictionary import Whitelist
from .text import extract_latin_words, normalize_text

# ═══════════════════════════════════════════════════════════════════
# REGEX / HẰNG SỐ
# ═══════════════════════════════════════════════════════════════════

_TIKI_BOILERPLATE_RE = re.compile(
    r"\s*Giá sản phẩm trên Tiki đã bao gồm thuế.*$",
    re.IGNORECASE | re.DOTALL,
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
_PHONE_CONTEXT_RE = re.compile(
    r"(?i)(sđt|sdt|hotline|liên hệ|lien he|zalo|phone|điện thoại|dien thoai)"
    r"[^\n]{0,30}?((?:\+?84|0)\d{9,10})"
)
_HASHTAG_RE = re.compile(r"(?<!\w)#[\wÀ-ỹĐđ]+", re.UNICODE)
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B00-\u2BFF]"
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_MOJIBAKE_RE = re.compile(r"Ã[\x80-\xBF]|â€|ðŸ|�")
_SEPARATOR_RE = re.compile(r"(?:[=\-_*•·]\s*){4,}")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_MULTI_PUNCT_RE = re.compile(r"([!?.,])\1{3,}")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:%|cm|mm|km|kg|mg|ml|m2|m3|mah|hz|kw|w|v|g|l|m|inch|ft|pcs)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_MODEL_RE = re.compile(r"(?=[A-Za-z0-9\-_/]*\d)[A-Za-z][A-Za-z0-9\-_/]{1,}")
_ALL_CAPS_RE = re.compile(r"\b[A-ZÀ-ỸĐ]{4,}\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]+|\d+[.)])\s*")
_DECIMAL_RE = re.compile(r"(\d)[.,](\d)")
_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_NEWLINE_RE = re.compile(r"\n{2,}")

_ABBREVIATIONS = (
    "tp.", "q.", "p.", "tx.", "h.", "x.", "v.v.", "vv.", "st.",
    "mr.", "mrs.", "dr.", "vs.", "đ.", "sđt.", "sdt.", "tel.",
)

_VI_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
)

_UNITS = {
    "cm", "mm", "km", "kg", "mg", "ml", "m2", "m3", "mah", "hz",
    "kw", "w", "v", "g", "l", "m", "inch", "ft", "pcs", "%",
}

# Từ tiếng Anh/tiếp thị rất phổ biến, dùng để tách "nhiễu" khỏi brand/model.
_COMMON_EN_NOISE = {
    "the", "and", "for", "with", "new", "hot", "sale", "free", "ship",
    "high", "quality", "best", "good", "super", "smart", "mini", "plus",
    "max", "pro", "premium", "fashion", "style", "model", "set", "combo",
    "gift", "size", "color", "home", "shop", "store", "official",
}


def _unaccent(word: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp văn bản VI không dấu."""
    word = word.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


# ═══════════════════════════════════════════════════════════════════
# TỪ ĐIỂN ÂM TIẾT VIỆT
# ═══════════════════════════════════════════════════════════════════

def load_vi_lexicon() -> set[str]:
    """Nạp danh sách âm tiết/từ đơn tiếng Việt từ pyvi (nếu có).

    Không dùng ``underthesea`` để tránh kéo theo model nặng; pyvi có sẵn
    ``models/words.txt`` (~31k mục) và chạy tốt trên CPU.
    """
    try:
        import pyvi  # type: ignore
        path = Path(pyvi.__file__).resolve().parent / "models" / "words.txt"
        if not path.exists():
            return set()
        lexicon: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                word = line.strip().lower()
                if not word or " " in word:
                    continue
                lexicon.add(word)
                # Thêm dạng không dấu: "những" → "nhung", "mình" → "minh".
                # Giúp nhận ra tiếng Việt không dấu thay vì coi là ngoại lai.
                unaccented = _unaccent(word)
                if unaccented:
                    lexicon.add(unaccented)
        return lexicon
    except Exception:
        return set()


def is_vi_token(token: str, lexicon: set[str]) -> bool:
    """Token có phải tiếng Việt (có dấu hoặc nằm trong lexicon) không."""
    word = token.strip().lower()
    if not word:
        return False
    if any(char in _VI_DIACRITICS for char in token):
        return True
    return word in lexicon


def _looks_like_model(token: str) -> bool:
    return bool(_MODEL_RE.fullmatch(token))


def _looks_like_brand(token: str) -> bool:
    if len(token) < 2 or len(token) > 24:
        return False
    if token.isupper():
        return True
    if re.match(r"^[A-Z][a-z]+[A-Z]", token):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# CLEAN TEXT
# ═══════════════════════════════════════════════════════════════════

def clean_text(text: object, *, title: bool = False) -> tuple[str, dict[str, int]]:
    """Chuẩn hoá và bỏ nhiễu Tiki, trả về ``(text_sạch, thống kê_nhiễu)``.

    Không xoá brand/model/số/đơn vị. Chỉ bỏ các thành phần gần như chắc chắn
    là nhiễu: boilerplate pháp lý Tiki, URL/email/hotline, hashtag, emoji,
    dòng kẻ, HTML entity.
    """
    raw = normalize_text(text)
    info = {
        "boilerplate_chars": 0,
        "n_urls": 0,
        "n_emails": 0,
        "n_phones": 0,
        "n_hashtags": 0,
        "n_emoji": 0,
        "n_html": 0,
        "n_separators": 0,
        "n_cjk": 0,
        "n_cyrillic": 0,
        "n_thai": 0,
        "n_mojibake": 0,
    }
    if not raw.strip():
        return "", info

    info["n_urls"] = len(_URL_RE.findall(raw))
    info["n_emails"] = len(_EMAIL_RE.findall(raw))
    info["n_hashtags"] = len(_HASHTAG_RE.findall(raw))
    info["n_emoji"] = len(_EMOJI_RE.findall(raw))
    info["n_html"] = len(_HTML_ENTITY_RE.findall(raw))
    info["n_separators"] = len(_SEPARATOR_RE.findall(raw))
    info["n_cjk"] = len(_CJK_RE.findall(raw))
    info["n_cyrillic"] = len(_CYRILLIC_RE.findall(raw))
    info["n_thai"] = len(_THAI_RE.findall(raw))
    info["n_mojibake"] = len(_MOJIBAKE_RE.findall(raw))

    cleaned = html.unescape(raw)

    # Boilerplate pháp lý Tiki thường ở cuối; cắt từ lần xuất hiện đầu tiên.
    before = len(cleaned)
    cleaned = _TIKI_BOILERPLATE_RE.sub("", cleaned)
    info["boilerplate_chars"] = before - len(cleaned)

    # Hotline/phone chỉ xoá khi có ngữ cảnh liên hệ, tránh xoá model number.
    def _drop_phone(match: re.Match[str]) -> str:
        info["n_phones"] += 1
        return match.group(1)

    cleaned = _PHONE_CONTEXT_RE.sub(_drop_phone, cleaned)

    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EMAIL_RE.sub(" ", cleaned)
    cleaned = _HASHTAG_RE.sub(" ", cleaned)
    cleaned = _EMOJI_RE.sub(" ", cleaned)
    cleaned = _SEPARATOR_RE.sub(" ", cleaned)
    cleaned = _MULTI_PUNCT_RE.sub(r"\1", cleaned)

    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = _NEWLINE_RE.sub("\n", cleaned)
    cleaned = cleaned.strip(" \t\r\n")

    # Dấu câu treo ở hai đầu title.
    if title:
        cleaned = cleaned.strip(" -_|/,;:")
    return cleaned, info


# ═══════════════════════════════════════════════════════════════════
# SENTENCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def _protect_sentence_markers(text: str) -> tuple[str, dict[str, str]]:
    """Bảo vệ số thập phân/viết tắt để không bị tách câu sai."""
    protected: dict[str, str] = {}

    def _store(value: str) -> str:
        key = f"\x00{len(protected)}\x00"
        protected[key] = value
        return key

    text = _DECIMAL_RE.sub(lambda m: _store(m.group(1) + "." + m.group(2)), text)
    text = text.replace("...", _store("..."))
    for abbreviation in _ABBREVIATIONS:
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(abbreviation)}")
        text = pattern.sub(lambda m: _store(m.group(0)), text)
    return text, protected


def _restore_sentence_markers(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def chunk_sentence(sentence: str, max_words: int = 60) -> list[str]:
    """Cắt câu quá dài tại dấu ``;``/``,`` rồi tại khoảng trắng."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    parts = re.split(r"(?<=[;,])\s+", sentence)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if current and len(candidate.split()) > max_words:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)

    result: list[str] = []
    for chunk in chunks:
        chunk_words = chunk.split()
        if len(chunk_words) <= max_words:
            result.append(chunk)
        else:
            for start in range(0, len(chunk_words), max_words):
                result.append(" ".join(chunk_words[start:start + max_words]))
    return result


def split_sentences(text: object, *, max_words: int = 60) -> list[str]:
    """Tách câu tiếng Việt bằng regex có bảo vệ viết tắt/số thập phân."""
    source = normalize_text(text)
    if not source.strip():
        return []

    protected_source, protected = _protect_sentence_markers(source)
    parts = _SENTENCE_SPLIT_RE.split(protected_source)

    sentences: list[str] = []
    for part in parts:
        restored = _restore_sentence_markers(part, protected)
        restored = _BULLET_RE.sub("", restored).strip()
        restored = _SPACE_RE.sub(" ", restored).strip()
        if not restored:
            continue
        sentences.extend(chunk_sentence(restored, max_words=max_words))

    # Gộp mảnh quá ngắn vào câu bên cạnh.
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence.split()) < 3:
            merged[-1] = f"{merged[-1]} {sentence}".strip()
        else:
            merged.append(sentence)
    return merged


# ═══════════════════════════════════════════════════════════════════
# ATTRIBUTES
# ═══════════════════════════════════════════════════════════════════

def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def parse_attributes(
    structured: Any,
    description: object,
) -> list[dict[str, Any]]:
    """Chuẩn hoá ``attributes_vi`` (list struct) hoặc ``description_vi`` (KV)."""
    attributes: list[dict[str, Any]] = []

    for item in _as_list(structured):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        values = [
            str(value).strip()
            for value in _as_list(item.get("values"))
            if str(value).strip()
        ]
        if name and values:
            attributes.append({"name": name, "values": values})

    if attributes:
        return attributes

    # Fallback: parse chuỗi "Thương hiệu: X; Xuất xứ: Y".
    text = normalize_text(description)
    for part in re.split(r";\s*", text):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        name = name.strip()
        value = value.strip()
        if name and value:
            attributes.append({"name": name, "values": [value]})
    return attributes


def attributes_to_text(attributes: list[dict[str, Any]], limit: int = 40) -> str:
    pairs: list[str] = []
    for attribute in attributes[:limit]:
        name = clean_text(attribute["name"], title=True)[0]
        values = [
            clean_text(value, title=True)[0]
            for value in attribute["values"]
            if clean_text(value, title=True)[0]
        ]
        if name and values:
            pairs.append(f"{name}: {', '.join(values)}")
    return "; ".join(pairs)


# ═══════════════════════════════════════════════════════════════════
# METRICS / ENTITIES
# ═══════════════════════════════════════════════════════════════════

def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zà-ỹđ0-9]+", text.lower())
        if len(token) >= 2
    }


def title_prose_coverage(title: str, prose: str) -> float:
    title_tokens = _content_tokens(title)
    if not title_tokens:
        return 0.0
    prose_tokens = _content_tokens(prose)
    return len(title_tokens & prose_tokens) / len(title_tokens)


def extract_protected_spans(
    text: str,
    *,
    brand_tokens: set[str],
    whitelist: set[str],
    limit: int = 80,
) -> list[str]:
    """Trích các span cần giữ nguyên khi dịch: số/đơn vị, model, brand."""
    spans: list[str] = []

    # Số + đơn vị (20W, 60x30cm, 1.45kg, 220V, 100%...).
    for match in re.finditer(
        r"\d+(?:[.,]\d+)?(?:\s*[x*×]\s*\d+(?:[.,]\d+)?)*\s*"
        r"(?:%|cm|mm|km|kg|mg|ml|m2|m3|mah|hz|kw|w|v|g|l|m|inch|ft|pcs)?",
        text,
        re.IGNORECASE,
    ):
        span = match.group(0).strip()
        if span and any(char.isdigit() for char in span):
            spans.append(span)

    # Model code: T4-40LED, iPhone15, USB-C, M416...
    for match in _MODEL_RE.finditer(text):
        spans.append(match.group(0))

    # Brand/model từ cột brand + whitelist.
    for token in extract_latin_words(text):
        lowered = token.lower()
        if lowered in brand_tokens or lowered in whitelist:
            spans.append(token)
        elif _looks_like_brand(token) and lowered not in _COMMON_EN_NOISE:
            spans.append(token)

    # Bỏ trùng, giữ thứ tự, giới hạn độ dài.
    unique: list[str] = []
    seen: set[str] = set()
    for span in spans:
        key = span.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(span.strip())
        if len(unique) >= limit:
            break
    return unique


def latin_metrics(
    text: str,
    *,
    lexicon: set[str],
    whitelist: set[str],
    brand_tokens: set[str],
) -> dict[str, Any]:
    """Đo tỉ lệ tiếng Việt / ngoại lai / brand / model / số / đơn vị."""
    tokens = extract_latin_words(text)
    vi_count = 0
    protected_count = 0
    foreign_count = 0
    foreign_tokens: list[str] = []

    for token in tokens:
        lowered = token.lower()
        if lowered in _UNITS or lowered in whitelist or lowered in brand_tokens:
            protected_count += 1
        elif _looks_like_model(token):
            protected_count += 1
        elif lowered in _COMMON_EN_NOISE:
            foreign_count += 1
            foreign_tokens.append(token)
        elif is_vi_token(token, lexicon):
            vi_count += 1
        elif len(token) <= 2:
            # Mảnh viết tắt/âm tiết VI không dấu (ng, ch, th, nh...):
            # không đủ bằng chứng để coi là ngoại lai.
            protected_count += 1
        elif _looks_like_brand(token):
            protected_count += 1
        else:
            foreign_count += 1
            foreign_tokens.append(token)

    total = vi_count + protected_count + foreign_count
    return {
        "n_latin": total,
        "n_vi": vi_count,
        "n_protected": protected_count,
        "n_foreign": foreign_count,
        "vi_rate": vi_count / total if total else 0.0,
        "foreign_rate": foreign_count / total if total else 0.0,
        "foreign_tokens": foreign_tokens,
        "n_numbers": len(_NUMBER_RE.findall(text)),
        "n_units": len(_UNIT_RE.findall(text)),
        "n_models": len(_MODEL_RE.findall(text)),
        "n_allcaps": len(_ALL_CAPS_RE.findall(text)),
        "n_chars": len(text),
        "n_words": len(text.split()),
    }


# ═══════════════════════════════════════════════════════════════════
# SCORING / ROUTING
# ═══════════════════════════════════════════════════════════════════

def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clean_score(info: dict[str, int], n_words: int) -> float:
    noise = (
        info["n_urls"] + info["n_emails"] + info["n_phones"]
        + info["n_hashtags"] + info["n_emoji"] + info["n_cjk"]
        + info["n_mojibake"] + info["n_separators"] + info["n_html"]
    )
    if noise == 0:
        return 1.0
    return _clip(1.0 - noise / max(8.0, n_words / 4.0))


def _title_length_score(n_words: int) -> float:
    if 4 <= n_words <= 30:
        return 1.0
    if 3 <= n_words <= 40:
        return 0.7
    return 0.3


def _prose_length_score(n_words: int) -> float:
    if 30 <= n_words <= 600:
        return 1.0
    if 15 <= n_words < 30 or 600 < n_words <= 900:
        return 0.7
    if n_words >= 10:
        return 0.4
    return 0.0


def _entity_score(n_numbers: int, n_words: int) -> float:
    if n_words == 0:
        return 0.0
    density = n_numbers / n_words
    if density <= 0.25:
        return 1.0
    if density <= 0.40:
        return 0.7
    return 0.4


def score_title(metrics: dict[str, Any], info: dict[str, int]) -> float:
    lang = metrics["vi_rate"] if metrics["n_latin"] else 0.5
    return _clip(
        0.35 * _title_length_score(metrics["n_words"])
        + 0.30 * _clean_score(info, metrics["n_words"])
        + 0.25 * lang
        + 0.10 * _entity_score(metrics["n_numbers"], metrics["n_words"])
    )


def score_prose(
    metrics: dict[str, Any],
    info: dict[str, int],
    coverage: float,
) -> float:
    lang = metrics["vi_rate"] if metrics["n_latin"] else 0.5
    return _clip(
        0.25 * _prose_length_score(metrics["n_words"])
        + 0.25 * _clean_score(info, metrics["n_words"])
        + 0.25 * lang
        + 0.15 * _entity_score(metrics["n_numbers"], metrics["n_words"])
        + 0.10 * _clip(coverage)
    )


def _split_for_product(product_id: str, title_key: str, prose_key: str) -> str:
    payload = f"{product_id}|{title_key}|{prose_key}".encode("utf-8")
    bucket = int(hashlib.md5(payload).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "dev"
    return "test"


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TikiEdaResult:
    df: pd.DataFrame
    stats: dict = field(default_factory=dict)


def run_tiki_eda(
    df: pd.DataFrame,
    *,
    progress=None,
) -> TikiEdaResult:
    """Chạy EDA đầy đủ trên toàn bộ DataFrame Tiki (1 dòng/sản phẩm)."""
    required = ("product_id", "title_vi", "description_vi", "description_prose_vi")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột Tiki: {missing}")

    whitelist_obj = Whitelist.load(WHITELIST_CSV)
    whitelist = set(whitelist_obj.words)
    lexicon = load_vi_lexicon()
    if progress:
        progress(f"Whitelist: {len(whitelist):,} | VI lexicon: {len(lexicon):,}")

    # ── Pass 1: clean + segment ──────────────────────────────────
    intermediates: list[dict[str, Any]] = []
    title_counts: Counter[str] = Counter()
    prose_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    sentence_counts: Counter[str] = Counter()
    foreign_counter: Counter[str] = Counter()
    rows = df.to_dict("records")

    for index, row in enumerate(rows):
        title_raw = normalize_text(row.get("title_vi"))
        prose_raw = normalize_text(row.get("description_prose_vi"))
        desc_raw = normalize_text(row.get("description_vi"))

        title, title_info = clean_text(title_raw, title=True)
        prose, prose_info = clean_text(prose_raw)
        desc, _ = clean_text(desc_raw)

        title_key = re.sub(r"[^a-zà-ỹđ0-9]+", " ", title.lower()).strip()
        prose_key = re.sub(r"[^a-zà-ỹđ0-9]+", " ", prose.lower()).strip()

        # Bỏ phần prose lặp lại title ở đầu, nhưng chỉ khi title đóng vai trò
        # một dòng tiêu đề (theo sau là dấu câu hoặc chữ hoa). Nếu title nằm
        # giữa câu ("Hộp cơm ... bằng nhựa ..."), giữ nguyên để không tạo mảnh vụn.
        prose_for_bt = prose
        title_prefix = bool(title and prose.lower().startswith(title.lower()))
        title_echo_removed = False
        if title_prefix:
            remainder = prose[len(title):]
            stripped = remainder.lstrip(" \t\n")
            starts_heading = bool(stripped) and (
                stripped[0] in ".:;-–—|,\"'("
                or stripped[0].isupper()
                or stripped[0].isdigit()
            )
            if not remainder.strip():
                prose_for_bt = ""
                title_echo_removed = True
            elif starts_heading:
                prose_for_bt = stripped.lstrip(" -–—:.,;|")
                title_echo_removed = True

        sentences = split_sentences(prose_for_bt)
        # Nếu câu đầu vẫn là title (sau khi đã cắt prefix), bỏ câu đó.
        if sentences and title:
            first_key = re.sub(r"[^a-zà-ỹđ0-9]+", " ", sentences[0].lower()).strip()
            if first_key and first_key == title_key:
                sentences = sentences[1:]

        attributes = parse_attributes(row.get("attributes_vi"), desc)
        attributes_text = attributes_to_text(attributes)

        brand_value = str(row.get("brand") or "").strip()
        brand_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-zÀ-ỹĐđ0-9]+", brand_value)
            if len(token) >= 2
        }
        if brand_value:
            brand_tokens.add(brand_value.lower())

        title_counts[title_key] += 1
        prose_counts[prose_key] += 1
        pair_counts[f"{title_key}\x1f{prose_key}"] += 1
        for sentence in sentences:
            sentence_key = re.sub(r"[^a-zà-ỹđ0-9]+", " ", sentence.lower()).strip()
            if sentence_key:
                sentence_counts[sentence_key] += 1

        intermediates.append({
            "row": row,
            "title": title,
            "title_info": title_info,
            "title_key": title_key,
            "prose": prose,
            "prose_for_bt": prose_for_bt,
            "prose_info": prose_info,
            "prose_key": prose_key,
            "desc": desc,
            "attributes": attributes,
            "attributes_text": attributes_text,
            "brand_tokens": brand_tokens,
            "sentences": sentences,
            "title_prefix": title_prefix,
            "title_echo_removed": title_echo_removed,
        })

        if progress and (index + 1) % 2000 == 0:
            progress(f"  clean/segment: {index + 1:,}/{len(rows):,}")

    # ── Pass 2: metrics, flags, score, route ────────────────────
    output: list[dict[str, Any]] = []
    for index, item in enumerate(intermediates):
        row = item["row"]
        title = item["title"]
        prose = item["prose"]
        prose_for_bt = item["prose_for_bt"]
        desc = item["desc"]
        title_info = item["title_info"]
        prose_info = item["prose_info"]
        title_key = item["title_key"]
        prose_key = item["prose_key"]
        brand_tokens = item["brand_tokens"]

        title_metrics = latin_metrics(
            title, lexicon=lexicon, whitelist=whitelist, brand_tokens=brand_tokens
        )
        prose_metrics = latin_metrics(
            prose_for_bt, lexicon=lexicon, whitelist=whitelist, brand_tokens=brand_tokens
        )
        desc_metrics = latin_metrics(
            desc, lexicon=lexicon, whitelist=whitelist, brand_tokens=brand_tokens
        )
        for token in title_metrics["foreign_tokens"] + prose_metrics["foreign_tokens"]:
            foreign_counter[token.lower()] += 1

        coverage = title_prose_coverage(title, prose)
        title_score = score_title(title_metrics, title_info)
        prose_score = score_prose(prose_metrics, prose_info, coverage)
        attr_score = 1.0 if item["attributes"] and desc_metrics["n_words"] >= 2 else 0.0

        title_dup = title_counts[title_key]
        prose_dup = prose_counts[prose_key]
        pair_dup = pair_counts[f"{title_key}\x1f{prose_key}"]

        sentences = item["sentences"]
        sentence_freqs = [
            sentence_counts[
                re.sub(r"[^a-zà-ỹđ0-9]+", " ", sentence.lower()).strip()
            ]
            for sentence in sentences
        ]
        sentence_freq_max = max(sentence_freqs) if sentence_freqs else 0
        template_count = sum(1 for freq in sentence_freqs if freq >= 50)
        sentence_keys = [
            re.sub(r"[^a-zà-ỹđ0-9]+", " ", sentence.lower()).strip()
            for sentence in sentences
        ]
        duplicate_sentences = sum(
            1 for key, count in Counter(sentence_keys).items() if key and count > 1
        )

        flags: list[str] = []
        drop_reasons: list[str] = []

        if title_info["boilerplate_chars"] or prose_info["boilerplate_chars"]:
            flags.append("boilerplate_removed")
        if title_info["n_hashtags"] or prose_info["n_hashtags"]:
            flags.append("hashtag_removed")
        if title_info["n_urls"] or prose_info["n_urls"]:
            flags.append("url_removed")
        if title_info["n_emoji"] or prose_info["n_emoji"]:
            flags.append("emoji_removed")
        if title_info["n_phones"] or prose_info["n_phones"]:
            flags.append("phone_removed")
        if title_info["n_separators"] or prose_info["n_separators"]:
            flags.append("separator_removed")
        if item["title_echo_removed"]:
            flags.append("title_echo_removed")
        if title_info["n_cjk"] or prose_info["n_cjk"]:
            flags.append("cjk_present")
        if title_info["n_cyrillic"] or prose_info["n_cyrillic"]:
            flags.append("cyrillic_present")
        if title_info["n_mojibake"] or prose_info["n_mojibake"]:
            flags.append("mojibake")
        if title_metrics["n_words"] < 3:
            flags.append("title_short")
        if title_metrics["n_words"] > 40:
            flags.append("title_long")
        if prose_metrics["n_words"] < 10:
            flags.append("prose_short")
        if prose_metrics["n_words"] > 900:
            flags.append("prose_long")
        if title_dup > 1:
            flags.append("title_duplicate")
        if prose_dup > 1:
            flags.append("prose_duplicate")
        if pair_dup > 1:
            flags.append("pair_duplicate")
        if coverage < 0.30:
            flags.append("title_prose_mismatch")
        if (
            title_metrics["foreign_rate"] > 0.35
            or prose_metrics["foreign_rate"] > 0.15
        ):
            flags.append("foreign_high")
        if (
            title_metrics["n_allcaps"] > 3
            or prose_metrics["n_allcaps"] > 10
        ):
            flags.append("all_caps_high")
        if template_count >= 3:
            flags.append("template_heavy")
        if duplicate_sentences > 0:
            flags.append("duplicate_sentences")
        if not item["attributes"]:
            flags.append("attributes_missing")

        # Price/sales chỉ là cảnh báo, không ảnh hưởng route dịch.
        price = row.get("price")
        try:
            price_value = float(price) if price is not None and not pd.isna(price) else None
        except (TypeError, ValueError):
            price_value = None
        if price_value is not None and (price_value <= 0 or price_value > 50_000_000):
            flags.append("price_outlier")
        if row.get("sales") is None or pd.isna(row.get("sales")):
            flags.append("sales_missing")

        # Hard fail title
        title_hard = False
        if not title:
            title_hard = True
            drop_reasons.append("title_empty")
        elif title_metrics["n_words"] < 3:
            title_hard = True
            drop_reasons.append("title_too_short")
        elif title_metrics["n_words"] > 50:
            title_hard = True
            drop_reasons.append("title_too_long")
        elif title_info["n_cjk"] or title_info["n_mojibake"]:
            title_hard = True
            drop_reasons.append("title_wrong_script")

        # Hard fail prose
        prose_hard = False
        if not prose_for_bt:
            prose_hard = True
            drop_reasons.append("prose_empty")
        elif prose_metrics["n_words"] < 10:
            prose_hard = True
            drop_reasons.append("prose_too_short")
        elif prose_info["n_cjk"] or prose_info["n_mojibake"]:
            prose_hard = True
            drop_reasons.append("prose_wrong_script")
        elif not sentences:
            prose_hard = True
            drop_reasons.append("prose_no_sentence")

        if title_hard and prose_hard and not item["attributes"]:
            drop_reasons.append("nothing_to_translate")

        title_ready = (not title_hard) and title_score >= 0.60
        prose_ready = (
            (not prose_hard) and prose_score >= 0.60 and coverage >= 0.20
        )
        attrs_ready = attr_score >= 1.0

        def _route(ready: bool, hard: bool, score: float) -> str:
            if ready:
                return "BT"
            if not hard and score >= 0.45:
                return "REVIEW"
            return "DROP"

        title_route = _route(title_ready, title_hard, title_score)
        prose_route = _route(prose_ready, prose_hard, prose_score)
        attr_route = "BT" if attrs_ready else "DROP"

        if title_ready and prose_ready:
            overall_route = "BT_TITLE+PROSE"
        elif title_ready:
            overall_route = "BT_TITLE"
        elif prose_ready:
            overall_route = "BT_PROSE"
        elif attrs_ready:
            overall_route = "BT_ATTR"
        elif "REVIEW" in (title_route, prose_route):
            overall_route = "REVIEW"
        else:
            overall_route = "DROP"

        if prose_metrics["n_words"] > 0:
            quality_score = _clip(
                0.45 * title_score + 0.45 * prose_score + 0.10 * attr_score
            )
        else:
            quality_score = _clip(0.80 * title_score + 0.20 * attr_score)

        protected_spans = extract_protected_spans(
            f"{title}\n{prose_for_bt}",
            brand_tokens=brand_tokens,
            whitelist=whitelist,
        )

        product_id = str(row.get("product_id", ""))
        row_hash = hashlib.md5(
            f"{product_id}|{item['title_key']}|{item['prose_key']}".encode("utf-8")
        ).hexdigest()[:16]

        output.append({
            **row,
            "row_hash": row_hash,
            "split": _split_for_product(product_id, title_key, prose_key),
            "bt_title": title,
            "bt_prose": prose_for_bt,
            "bt_attributes_text": item["attributes_text"],
            "bt_sentences": sentences[:300],
            "bt_sentence_freqs": sentence_freqs[:300],
            "protected_spans": protected_spans,
            "n_protected_spans": len(protected_spans),
            "n_sentences": len(sentences),
            "title_n_chars": title_metrics["n_chars"],
            "title_n_words": title_metrics["n_words"],
            "title_n_latin": title_metrics["n_latin"],
            "title_n_vi": title_metrics["n_vi"],
            "title_n_foreign": title_metrics["n_foreign"],
            "title_vi_rate": round(title_metrics["vi_rate"], 4),
            "title_foreign_rate": round(title_metrics["foreign_rate"], 4),
            "title_n_numbers": title_metrics["n_numbers"],
            "title_n_units": title_metrics["n_units"],
            "title_n_models": title_metrics["n_models"],
            "title_n_allcaps": title_metrics["n_allcaps"],
            "prose_n_chars": prose_metrics["n_chars"],
            "prose_n_words": prose_metrics["n_words"],
            "prose_n_latin": prose_metrics["n_latin"],
            "prose_n_vi": prose_metrics["n_vi"],
            "prose_n_foreign": prose_metrics["n_foreign"],
            "prose_vi_rate": round(prose_metrics["vi_rate"], 4),
            "prose_foreign_rate": round(prose_metrics["foreign_rate"], 4),
            "prose_n_numbers": prose_metrics["n_numbers"],
            "prose_n_units": prose_metrics["n_units"],
            "prose_n_models": prose_metrics["n_models"],
            "prose_n_allcaps": prose_metrics["n_allcaps"],
            "desc_n_words": desc_metrics["n_words"],
            "desc_n_attributes": len(item["attributes"]),
            "desc_n_numbers": desc_metrics["n_numbers"],
            "title_prose_coverage": round(coverage, 4),
            "title_prefix_in_prose": bool(item["title_prefix"]),
            "boilerplate_chars_removed": (
                title_info["boilerplate_chars"] + prose_info["boilerplate_chars"]
            ),
            "n_hashtags_removed": (
                title_info["n_hashtags"] + prose_info["n_hashtags"]
            ),
            "n_urls_removed": title_info["n_urls"] + prose_info["n_urls"],
            "n_emails_removed": title_info["n_emails"] + prose_info["n_emails"],
            "n_phones_removed": title_info["n_phones"] + prose_info["n_phones"],
            "n_emoji_removed": title_info["n_emoji"] + prose_info["n_emoji"],
            "n_cjk_chars": title_info["n_cjk"] + prose_info["n_cjk"],
            "n_mojibake_chars": title_info["n_mojibake"] + prose_info["n_mojibake"],
            "n_html_entities": title_info["n_html"] + prose_info["n_html"],
            "n_separator_blocks": (
                title_info["n_separators"] + prose_info["n_separators"]
            ),
            "title_duplicate_count": title_dup,
            "prose_duplicate_count": prose_dup,
            "pair_duplicate_count": pair_dup,
            "n_duplicate_sentences": duplicate_sentences,
            "template_sentence_count": template_count,
            "sentence_freq_max": sentence_freq_max,
            "title_score": round(title_score, 4),
            "prose_score": round(prose_score, 4),
            "attr_score": round(attr_score, 4),
            "quality_score": round(quality_score, 4),
            "title_route": title_route,
            "prose_route": prose_route,
            "attr_route": attr_route,
            "overall_route": overall_route,
            "bt_ready_title": title_ready,
            "bt_ready_prose": prose_ready,
            "bt_ready_attrs": attrs_ready,
            "bt_ready": bool(title_ready or prose_ready),
            "eda_flags": flags,
            "eda_flags_text": "|".join(flags),
            "drop_reasons": drop_reasons,
            "drop_reasons_text": "|".join(drop_reasons),
            "attributes_vi_json": json.dumps(
                item["attributes"], ensure_ascii=False
            ),
        })

        if progress and (index + 1) % 2000 == 0:
            progress(f"  score/route: {index + 1:,}/{len(rows):,}")

    out = pd.DataFrame(output)
    stats = {
        "rows": len(out),
        "columns": len(out.columns),
        "bt_ready": int(out["bt_ready"].sum()),
        "overall_routes": out["overall_route"].value_counts().to_dict(),
        "title_ready": int(out["bt_ready_title"].sum()),
        "prose_ready": int(out["bt_ready_prose"].sum()),
        "attrs_ready": int(out["bt_ready_attrs"].sum()),
        "boilerplate_removed_rows": int(
            (out["boilerplate_chars_removed"] > 0).sum()
        ),
        "title_duplicates": int((out["title_duplicate_count"] > 1).sum()),
        "prose_duplicates": int((out["prose_duplicate_count"] > 1).sum()),
        "top_foreign_tokens": foreign_counter.most_common(30),
        "flags": Counter(
            flag for flags in out["eda_flags"] for flag in flags
        ).most_common(),
    }
    return TikiEdaResult(df=out, stats=stats)
