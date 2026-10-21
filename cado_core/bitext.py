# -*- coding: utf-8 -*-
"""Bitext filter: preprocess + chấm điểm cặp song ngữ Trung–Việt.

Thay đổi so với bản cũ:
  * Dùng trực tiếp cột ``*_cleaned`` do CADO tạo ra (pipeline liên thông).
  * Giữ newline khi chuẩn hoá (không phá format description).
  * Lexical overlap "không áp dụng" = NaN thay vì 1.0 (không thổi phồng điểm).
  * ``label_mode`` cho phép dùng hybrid threshold (trước đây config bị bỏ quên).
  * Lưu riêng các dòng bị loại ở V1 thay vì làm mất.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import (
    BITEXT_BATCH_SIZE,
    BITEXT_COSINE_DISCARD,
    BITEXT_COSINE_WEIGHT,
    BITEXT_CSV_DIR,
    BITEXT_HYBRID_KEEP,
    BITEXT_LEXICAL_WEIGHT,
    BITEXT_MODEL,
    BITEXT_READY_PARQUET,
    BITEXT_SCORED_PARQUET,
    BITEXT_TOKENS_PER_CJK_MAX,
    BITEXT_TOKENS_PER_CJK_MIN,
    CADO_PARQUET,
    DESC_VI,
    DESC_VI_CLEANED,
    DESC_ZH,
    DESC_ZH_CLEANED,
    LATIN_WORD_RE,
    TITLE_VI,
    TITLE_VI_CLEANED,
    TITLE_ZH,
    TITLE_ZH_CLEANED,
    WHITELIST_CSV,
)
from .dictionary import Cedict, Whitelist
from .text import count_cjk, normalize_text, strip_cjk

_ASCII_TRANSLATION = {
    "【": "[", "】": "]", "丨": "|",
}

_TRAILING_PUNCT_RE = re.compile(r"[.,;!?。，；！？]+$")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_STAR_OPERATOR_RE = re.compile(r"(\d+)\*(\d+)")
_ARABIC_NUMBER_RE = re.compile(r"\d+")
_HAN_NUMBER_RUN_RE = re.compile(r"[零〇一二三四五六七八九十百千万亿两]+")

_HAN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_HAN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}

# Đơn vị ZH → VI (chuẩn hoá để so khớp)
_UNIT_MAP: dict[str, str] = {
    "寸": "inch", "英寸": "inch", "斤": "kg", "克": "g",
    "厘米": "cm", "米": "m", "升": "l", "毫升": "ml", "千克": "kg",
}


# ═══════════════════════════════════════════════════════════════════
# CONFIG / RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BitextConfig:
    input_path: Path = CADO_PARQUET
    output_parquet: Path = BITEXT_SCORED_PARQUET
    ready_parquet: Path = BITEXT_READY_PARQUET
    discard_csv: Path = BITEXT_CSV_DIR / "bitext_discard.csv"
    scores_csv: Path = BITEXT_CSV_DIR / "bitext_full_scores.csv"
    discarded_rows_csv: Path = BITEXT_CSV_DIR / "bitext_discarded_rows.csv"

    model_name: str = BITEXT_MODEL
    batch_size: int = BITEXT_BATCH_SIZE
    cosine_discard: float = BITEXT_COSINE_DISCARD
    hybrid_keep: float = BITEXT_HYBRID_KEEP
    cosine_weight: float = BITEXT_COSINE_WEIGHT
    lexical_weight: float = BITEXT_LEXICAL_WEIGHT
    label_mode: str = "cosine"          # "cosine" | "hybrid"
    min_ratio: float = BITEXT_TOKENS_PER_CJK_MIN
    max_ratio: float = BITEXT_TOKENS_PER_CJK_MAX

    limit: int | None = None
    save_ready: bool = False
    entity_signals: bool = False
    entity_embedding: bool = False
    progress: Callable[[str], None] | None = None


@dataclass
class BitextResult:
    df: pd.DataFrame
    discarded: pd.DataFrame
    stats: dict


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESS
# ═══════════════════════════════════════════════════════════════════

def _ensure_cleaned_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo cột ``*_cleaned`` từ cột gốc nếu chưa có (không ghi đè CADO)."""
    pairs = [
        (TITLE_ZH_CLEANED, TITLE_ZH),
        (DESC_ZH_CLEANED, DESC_ZH),
        (TITLE_VI_CLEANED, TITLE_VI),
        (DESC_VI_CLEANED, DESC_VI),
    ]
    for cleaned, source in pairs:
        if cleaned not in df.columns:
            df[cleaned] = df[source] if source in df.columns else ""
    return df


def is_cjk_copy(row) -> bool:
    """Dòng có title_vi chép nguyên title_zh (chứa Hán tự)."""
    zh = normalize_text(row.get(TITLE_ZH, ""))
    vi = normalize_text(row.get(TITLE_VI, ""))
    return zh == vi and bool(re.search(r"[\u4e00-\u9fff]", vi))


def _fix_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = [
        TITLE_ZH, DESC_ZH, TITLE_VI, DESC_VI,
        TITLE_ZH_CLEANED, DESC_ZH_CLEANED, TITLE_VI_CLEANED, DESC_VI_CLEANED,
    ]
    for column in text_columns:
        if column not in df.columns:
            continue
        df[column] = df[column].map(
            lambda value: normalize_text(value).translate(_ASCII_TRANSLATION)
        )

    # V3: bỏ CJK khỏi cột tiếng Việt (chỉ cột cleaned)
    for column in (TITLE_VI_CLEANED, DESC_VI_CLEANED):
        df[column] = df[column].map(strip_cjk)

    # V4/V5: dọn dấu câu & số thập phân cho tiêu đề VI
    df[TITLE_VI_CLEANED] = df[TITLE_VI_CLEANED].map(
        lambda value: _TRAILING_PUNCT_RE.sub("", value.strip())
    )
    df[TITLE_VI_CLEANED] = df[TITLE_VI_CLEANED].map(
        lambda value: _DECIMAL_COMMA_RE.sub(".", value)
    )

    # V16: chuẩn hoá toán tử trong ZH
    for column in (TITLE_ZH_CLEANED, DESC_ZH_CLEANED):
        df[column] = df[column].map(
            lambda value: _STAR_OPERATOR_RE.sub(r"\1x\2", value).replace("丨", "|")
        )
    return df


def _add_flags(df: pd.DataFrame, cfg: BitextConfig) -> pd.DataFrame:
    def stuffing(value: str) -> bool:
        tokens = str(value).lower().split()
        return bool(tokens) and max(Counter(tokens).values()) >= 5

    df["flag_stuffing"] = df[TITLE_VI_CLEANED].map(stuffing)
    zh_cjk = df[TITLE_ZH_CLEANED].map(count_cjk)
    vi_tokens = df[TITLE_VI_CLEANED].map(lambda value: len(str(value).split()))
    ratio = vi_tokens / zh_cjk.replace(0, np.nan)
    df["flag_truncated"] = ratio < cfg.min_ratio
    df["flag_bloated"] = ratio > cfg.max_ratio
    return df


def preprocess(df: pd.DataFrame, cfg: BitextConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    _log(cfg, "─" * 50)
    _log(cfg, "STEP 1: PREPROCESS")
    _log(cfg, "─" * 50)

    initial = len(df)

    # V1: loại dòng chép nguyên CJK
    mask = df.apply(is_cjk_copy, axis=1)
    discarded = df[mask].copy()
    df = df[~mask].copy()
    _log(cfg, f"  V1 CJK copy → DISCARD: {len(discarded):,}")

    df = _ensure_cleaned_columns(df)
    df = _fix_text_columns(df)
    _log(cfg, "  V2 NFC/NFKC + V3 CJK in VI + V4/V5/V10/V16")

    df = _add_flags(df, cfg)
    _log(
        cfg,
        f"  Flags: stuffing={int(df['flag_stuffing'].sum()):,}, "
        f"truncated={int(df['flag_truncated'].sum()):,}, "
        f"bloated={int(df['flag_bloated'].sum()):,}",
    )
    _log(cfg, f"  Result: {len(df):,} rows (removed {initial - len(df):,})")

    return df, discarded


# ═══════════════════════════════════════════════════════════════════
# STEP 2a — COSINE
# ═══════════════════════════════════════════════════════════════════

def compute_cosine(df: pd.DataFrame, cfg: BitextConfig) -> pd.DataFrame:
    _log(cfg, "─" * 50)
    _log(cfg, "STEP 2a: BGE-M3 COSINE SIMILARITY")
    _log(cfg, "─" * 50)

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(cfg.model_name)
    _log(cfg, f"  Model loaded ({time.time() - t0:.0f}s)")

    zh_texts = df[TITLE_ZH_CLEANED].astype(str).tolist()
    vi_texts = df[TITLE_VI_CLEANED].astype(str).tolist()
    texts = zh_texts + vi_texts

    _log(cfg, f"  Encoding {len(texts):,} texts...")
    embeddings = model.encode(
        texts,
        batch_size=cfg.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    zh_emb = embeddings[: len(zh_texts)]
    vi_emb = embeddings[len(zh_texts):]
    cosine = np.sum(zh_emb * vi_emb, axis=1)

    df["cosine_bge_m3"] = cosine
    _log(
        cfg,
        f"  Done ({time.time() - t0:.0f}s) — mean={cosine.mean():.4f} "
        f"median={np.median(cosine):.4f} std={cosine.std():.4f}",
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2b — LEXICAL OVERLAP
# ═══════════════════════════════════════════════════════════════════

def parse_han_numbers(text: str) -> set[int]:
    """Chuyển số viết bằng chữ Hán thành số nguyên (hỗ trợ đến hàng ức)."""
    results: set[int] = set()
    for match in _HAN_NUMBER_RUN_RE.finditer(normalize_text(text)):
        run = match.group()
        total = 0
        section = 0
        number = 0
        for char in run:
            if char in _HAN_DIGITS:
                number = _HAN_DIGITS[char]
            elif char in _HAN_UNITS:
                unit = _HAN_UNITS[char]
                if unit >= 10_000:
                    section = (section + number) * unit
                    total += section
                    section = 0
                else:
                    section += (number or 1) * unit
                number = 0
        value = total + section + number
        if value > 0:
            results.add(value)
    return results


def extract_entities(text: str, *, is_zh: bool, whitelist: Whitelist) -> set[str]:
    """Trích brand/model/số/đơn vị để so khớp giữa ZH và VI."""
    source = normalize_text(text)
    entities: set[str] = set()

    pattern = LATIN_WORD_RE
    minimum = 2 if is_zh else 3
    for word in pattern.findall(source):
        if len(word) < minimum:
            continue
        lowered = word.lower()
        if (
            lowered in whitelist
            or re.search(r"\d", word)
            or re.match(r"^[A-Z]{2,6}$", word)
        ):
            entities.add(lowered)

    entities.update(_ARABIC_NUMBER_RE.findall(source))
    entities.update(str(number) for number in parse_han_numbers(source))

    if is_zh:
        for zh_unit, canonical in _UNIT_MAP.items():
            if zh_unit in source:
                entities.add(canonical)
    else:
        lowered = source.lower()
        for canonical in set(_UNIT_MAP.values()):
            if canonical in lowered:
                entities.add(canonical)

    return entities


def lexical_overlap(zh: str, vi: str, whitelist: Whitelist) -> float | None:
    """Jaccard trên entity. ``None`` = không có entity nào → không có bằng chứng."""
    zh_entities = extract_entities(zh, is_zh=True, whitelist=whitelist)
    vi_entities = extract_entities(vi, is_zh=False, whitelist=whitelist)
    union = zh_entities | vi_entities
    if not union:
        return None
    return len(zh_entities & vi_entities) / len(union)


def compute_lexical(df: pd.DataFrame, cfg: BitextConfig) -> pd.DataFrame:
    _log(cfg, "─" * 50)
    _log(cfg, "STEP 2b: LEXICAL OVERLAP")
    _log(cfg, "─" * 50)

    whitelist = Whitelist.load(WHITELIST_CSV)
    t0 = time.time()
    scores = [
        lexical_overlap(zh, vi, whitelist)
        for zh, vi in zip(
            df[TITLE_ZH_CLEANED].astype(str),
            df[TITLE_VI_CLEANED].astype(str),
        )
    ]
    df["lexical_overlap"] = pd.to_numeric(pd.Series(scores, index=df.index), errors="coerce")

    values = df["lexical_overlap"]
    _log(
        cfg,
        f"  Done ({time.time() - t0:.0f}s) — mean={values.mean():.4f} "
        f"median={values.median():.4f}",
    )
    _log(
        cfg,
        f"  Neutral (không có entity): {int(values.isna().sum()):,} | "
        f"Zero overlap: {int((values == 0).sum()):,}",
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2c — HYBRID + LABEL
# ═══════════════════════════════════════════════════════════════════

def compute_hybrid(df: pd.DataFrame, cfg: BitextConfig) -> pd.DataFrame:
    cosine = df["cosine_bge_m3"]
    lexical = df["lexical_overlap"]
    weighted = cfg.cosine_weight * cosine + cfg.lexical_weight * lexical.fillna(0.0)
    # Không có entity → chỉ dùng cosine, không thổi phồng bằng lexical.
    df["hybrid_score"] = np.where(lexical.notna(), weighted, cosine)
    return df


def classify(df: pd.DataFrame, cfg: BitextConfig) -> pd.DataFrame:
    if cfg.label_mode not in {"cosine", "hybrid"}:
        raise ValueError("label_mode phải là 'cosine' hoặc 'hybrid'")

    cosine = df["cosine_bge_m3"].fillna(0.0)
    discard = cosine < cfg.cosine_discard
    if cfg.label_mode == "hybrid":
        discard = discard | (df["hybrid_score"] < cfg.hybrid_keep)

    df["final_label"] = np.where(discard, "DISCARD", "KEEP")
    df["confidence"] = np.select(
        [cosine >= 0.75, cosine >= cfg.cosine_discard],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# ENTITY SIGNALS (tuỳ chọn — tương thích cado_tier2_entity.py)
# ═══════════════════════════════════════════════════════════════════

_ENTITY_WEIGHTS = {"S1": 0.15, "S2": 0.10, "S3": 0.10, "S4": 0.40,
                   "S5": 0.10, "S6": 0.05, "S7": 0.10}


def _extract_brands(text: str, *, is_zh: bool, whitelist: Whitelist) -> set[str]:
    """Chỉ trích brand/model/SKU (không lấy số/đơn vị như ``extract_entities``)."""
    source = normalize_text(text)
    minimum = 2 if is_zh else 3
    brands: set[str] = set()
    for word in LATIN_WORD_RE.findall(source):
        if len(word) < minimum:
            continue
        lowered = word.lower()
        if (
            lowered in whitelist
            or re.search(r"\d", word)
            or re.match(r"^[A-Z]{2,6}$", word)
        ):
            brands.add(lowered)
    return brands


def _signal_s1(zh: str, vi: str, whitelist: Whitelist) -> int:
    zh_brands = _extract_brands(zh, is_zh=True, whitelist=whitelist)
    vi_brands = _extract_brands(vi, is_zh=False, whitelist=whitelist)
    if not zh_brands and not vi_brands:
        return 1
    return 1 if zh_brands & vi_brands else 0


def _signal_s2(zh: str, vi: str) -> int:
    zh_numbers = {int(n) for n in _ARABIC_NUMBER_RE.findall(zh)} | parse_han_numbers(zh)
    vi_numbers = {int(n) for n in _ARABIC_NUMBER_RE.findall(vi)} | parse_han_numbers(vi)
    if not zh_numbers and not vi_numbers:
        return 1
    if zh_numbers and not vi_numbers:
        return 1   # số Hán có thể đã được dịch thành chữ ("五合一" → "năm trong một")
    if not zh_numbers and vi_numbers:
        return 0   # VI tự thêm số không có trong ZH
    return 1 if zh_numbers & vi_numbers else 0


def _signal_s3(zh: str, vi: str) -> int:
    vi_lower = vi.lower()
    for zh_unit, canonical in _UNIT_MAP.items():
        if zh_unit in zh and canonical not in vi_lower:
            return 0
    return 1


def _signal_s4_cdek(zh: str, vi: str, cedict: Cedict) -> int:
    cjk = [c for c in zh if "\u4e00" <= c <= "\u9fff"]
    if len(cjk) < 2:
        return 1
    for i in range(len(cjk) - 1):
        bigram = cjk[i] + cjk[i + 1]
        for english in cedict.bigram_english(bigram):
            if english in vi.lower():
                return 1
    return 0


def _signal_s5(zh: str, vi: str, cfg: BitextConfig) -> int:
    zh_words = count_cjk(zh) + len(LATIN_WORD_RE.findall(zh))
    vi_words = len(vi.split())
    if not zh_words:
        return 0
    return 1 if cfg.min_ratio <= vi_words / zh_words <= cfg.max_ratio else 0


def _signal_s6(vi: str) -> int:
    return 0 if re.search(r"[\u4e00-\u9fff]", vi) else 1


def _signal_s7(vi: str) -> int:
    tokens = vi.lower().split()
    if not tokens:
        return 1
    return 0 if max(Counter(tokens).values()) >= 5 else 1


def compute_entity_signals(
    df: pd.DataFrame,
    cfg: BitextConfig,
    *,
    use_embedding: bool = False,
) -> pd.DataFrame:
    """7 tín hiệu rule-based cho các cặp AMBIGUOUS (hoặc toàn bộ nếu không có
    ``tier1_label``). Sửa lỗi ``np`` chưa import và ghi đè file input của bản cũ.
    """
    whitelist = Whitelist.load(WHITELIST_CSV)
    cedict = Cedict.load(reverse=True)

    for column in (
        "signal_S1", "signal_S2", "signal_S3", "signal_S4", "signal_S4a",
        "signal_S4b", "signal_S5", "signal_S6", "signal_S7", "entity_score",
    ):
        df[column] = np.nan

    if "tier1_label" in df.columns:
        target_mask = df["tier1_label"] == "AMBIGUOUS"
        df["tier2_label"] = df["tier1_label"]
    else:
        target_mask = pd.Series(True, index=df.index)
        df["tier2_label"] = None

    positions = np.flatnonzero(target_mask.to_numpy())
    _log(cfg, f"  Entity signals: {len(positions):,} rows")

    # S4b — batch encode bigram nếu bật embedding
    bigram_vectors: dict[str, np.ndarray] = {}
    vi_vectors: dict[int, np.ndarray] = {}
    if use_embedding and len(positions):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vi_texts = [str(df[TITLE_VI_CLEANED].iloc[pos]) for pos in positions]
        vi_embeddings = model.encode(
            vi_texts, batch_size=256, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        vi_vectors = {pos: vector for pos, vector in zip(positions, vi_embeddings)}

        unique_bigrams = sorted({
            bigram
            for pos in positions
            for bigram in _top_bigrams(str(df[TITLE_ZH_CLEANED].iloc[pos]))
        })
        if unique_bigrams:
            vectors = model.encode(
                unique_bigrams, batch_size=256, normalize_embeddings=True,
                convert_to_numpy=True,
            )
            bigram_vectors = dict(zip(unique_bigrams, vectors))

    for position in positions:
        zh = str(df[TITLE_ZH_CLEANED].iloc[position])
        vi = str(df[TITLE_VI_CLEANED].iloc[position])

        s1 = _signal_s1(zh, vi, whitelist)
        s2 = _signal_s2(zh, vi)
        s3 = _signal_s3(zh, vi)
        s4a = _signal_s4_cdek(zh, vi, cedict)
        s4b = _signal_s4_embedding(
            zh, vi_vectors.get(position), bigram_vectors
        ) if use_embedding else 0
        s5 = _signal_s5(zh, vi, cfg)
        s6 = _signal_s6(vi)
        s7 = _signal_s7(vi)

        signals = {"S1": s1, "S2": s2, "S3": s3, "S4": max(s4a, s4b),
                   "S5": s5, "S6": s6, "S7": s7}
        score = round(sum(_ENTITY_WEIGHTS[key] * value for key, value in signals.items()), 2)

        df.loc[df.index[position], "signal_S4a"] = s4a
        df.loc[df.index[position], "signal_S4b"] = s4b
        for key, value in signals.items():
            df.loc[df.index[position], f"signal_{key}"] = value
        df.loc[df.index[position], "entity_score"] = score

        label = "KEEP" if score >= 0.60 else "REVIEW" if score >= 0.40 else "DISCARD"
        df.loc[df.index[position], "tier2_label"] = label

    return df


def _top_bigrams(text: str, limit: int = 3) -> list[str]:
    cjk = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return list(dict.fromkeys(bigrams))[:limit]


def _signal_s4_embedding(
    zh: str,
    vi_vector: np.ndarray | None,
    bigram_vectors: dict[str, np.ndarray],
    threshold: float = 0.65,
) -> int:
    if vi_vector is None:
        return 0
    vectors = [bigram_vectors[b] for b in _top_bigrams(zh) if b in bigram_vectors]
    if not vectors:
        return 0
    similarities = [float(np.dot(vector, vi_vector)) for vector in vectors]
    return 1 if max(similarities) > threshold else 0


# ═══════════════════════════════════════════════════════════════════
# RUN + EXPORT
# ═══════════════════════════════════════════════════════════════════

def run_bitext(
    df: pd.DataFrame,
    cfg: BitextConfig | None = None,
    *,
    skip_cosine: bool = False,
    preprocess_only: bool = False,
) -> BitextResult:
    cfg = cfg or BitextConfig()

    if cfg.limit is not None and cfg.limit < len(df):
        _log(cfg, f"⚠️  Giới hạn {cfg.limit:,} dòng để test (tổng {len(df):,})")
        df = df.head(cfg.limit).copy()

    input_rows = len(df)
    df, discarded = preprocess(df, cfg)

    if preprocess_only:
        _ensure_directories(cfg)
        df.to_parquet(cfg.ready_parquet, index=False)
        _log(cfg, f"✅ {cfg.ready_parquet} ({len(df):,} rows)")
        _export_discarded(discarded, cfg)
        return BitextResult(df=df, discarded=discarded, stats={
            "input": input_rows,
            "discarded_v1": len(discarded),
            "preprocessed": len(df),
            "preprocess_only": True,
        })

    if skip_cosine:
        if "cosine_bge_m3" not in df.columns:
            raise ValueError("--skip-cosine cần cột cosine_bge_m3 đã có trong input")
        _log(cfg, "⏩ Bỏ qua cosine (dùng cột có sẵn)")
    else:
        df = compute_cosine(df, cfg)

    df = compute_lexical(df, cfg)
    df = compute_hybrid(df, cfg)
    df = classify(df, cfg)

    if cfg.entity_signals:
        df = compute_entity_signals(df, cfg, use_embedding=cfg.entity_embedding)

    stats = _summarize(df, input_rows, len(discarded), cfg)
    _print_summary(cfg, stats)
    _export(df, discarded, cfg)

    if cfg.save_ready:
        ready = df.drop(columns=[c for c in ("final_label", "confidence") if c in df.columns])
        ready.to_parquet(cfg.ready_parquet, index=False)
        _log(cfg, f"✅ {cfg.ready_parquet} ({len(ready):,} rows)")

    return BitextResult(df=df, discarded=discarded, stats=stats)


def _summarize(df: pd.DataFrame, input_rows: int, discarded: int, cfg: BitextConfig) -> dict:
    keep = int((df["final_label"] == "KEEP").sum())
    return {
        "input": input_rows,
        "discarded_v1": discarded,
        "preprocessed": len(df),
        "keep": keep,
        "discard": len(df) - keep,
        "training_ready_pct": round(100 * keep / len(df), 1) if len(df) else 0.0,
        "cosine_mean": round(float(df["cosine_bge_m3"].mean()), 4),
        "cosine_median": round(float(df["cosine_bge_m3"].median()), 4),
        "hybrid_mean": round(float(df["hybrid_score"].mean()), 4),
        "lexical_neutral": int(df["lexical_overlap"].isna().sum()),
        "label_mode": cfg.label_mode,
    }


def _print_summary(cfg: BitextConfig, stats: dict) -> None:
    _log(cfg, "═" * 50)
    _log(cfg, "📊 BITEXT TALLY")
    _log(cfg, "═" * 50)
    _log(cfg, f"  Input:              {stats['input']:,}")
    _log(cfg, f"  Discarded (V1):     {stats['discarded_v1']:,}")
    _log(cfg, f"  Preprocessed:       {stats['preprocessed']:,}")
    _log(cfg, f"  KEEP:               {stats['keep']:,}")
    _log(cfg, f"  DISCARD:            {stats['discard']:,}")
    _log(cfg, f"  Training-ready:     {stats['training_ready_pct']}%")
    _log(cfg, f"  Label mode:         {stats['label_mode']}")


def _export(df: pd.DataFrame, discarded: pd.DataFrame, cfg: BitextConfig) -> None:
    _ensure_directories(cfg)

    df.to_parquet(cfg.output_parquet, index=False)
    _log(cfg, f"✅ {cfg.output_parquet} ({len(df):,} rows)")

    score_columns = [
        "product_id", "category", "cosine_bge_m3", "lexical_overlap",
        "hybrid_score", "final_label", "confidence",
        "title_zh_cleaned", "title_vi_cleaned",
        "flag_stuffing", "flag_truncated", "flag_bloated",
    ]
    available = [column for column in score_columns if column in df.columns]

    df[df["final_label"] == "DISCARD"][available].to_csv(
        cfg.discard_csv, index=False, encoding="utf-8-sig"
    )
    _log(cfg, f"✅ {cfg.discard_csv} ({int((df['final_label'] == 'DISCARD').sum()):,} rows)")

    df[available].to_csv(cfg.scores_csv, index=False, encoding="utf-8-sig")
    _log(cfg, f"✅ {cfg.scores_csv} ({len(df):,} rows)")

    _export_discarded(discarded, cfg)


def _export_discarded(discarded: pd.DataFrame, cfg: BitextConfig) -> None:
    if not len(discarded):
        return
    columns = [c for c in ("product_id", "title_zh", "title_vi", "category")
               if c in discarded.columns]
    discarded[columns].to_csv(cfg.discarded_rows_csv, index=False, encoding="utf-8-sig")
    _log(cfg, f"✅ {cfg.discarded_rows_csv} ({len(discarded):,} rows)")


def _ensure_directories(cfg: BitextConfig) -> None:
    for path in (cfg.output_parquet, cfg.ready_parquet, cfg.discard_csv,
                 cfg.scores_csv, cfg.discarded_rows_csv):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _log(cfg: BitextConfig, message: str) -> None:
    if cfg.progress:
        cfg.progress(message)
