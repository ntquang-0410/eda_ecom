# -*- coding: utf-8 -*-
"""CADO — pipeline làm sạch từ tiếng Anh trong văn bản Trung (T0–T5).

Đầu vào là DataFrame thô; đầu ra là bản copy có thêm các cột
``title_zh_cleaned`` / ``description_zh_cleaned`` (và report theo từ).

Ví dụ::

    from cado_core.cado import CadoConfig, run_cado
    result = run_cado(df, CadoConfig(columns=("title_zh",)))
    result.df.to_parquet("out.parquet")
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .config import (
    DESC_ZH,
    DESC_ZH_CLEANED,
    TIER3_MIN_FREQ,
    TIER3_OVERLAP_THRESHOLD,
    TIER5_BATCH_SIZE,
    TIER5_MAX_PAIRS,
    TIER5_MODEL,
    TIER5_SIMILARITY_THRESHOLD,
    TITLE_ZH,
    TITLE_ZH_CLEANED,
)
from .dictionary import Cedict, Whitelist
from .rules import (
    build_candidates,
    classify_occurrence,
    detect_loanwords,
)
from .text import extract_latin_words, normalize_text, remove_words

_CLEANED_COLUMNS = {
    TITLE_ZH: TITLE_ZH_CLEANED,
    DESC_ZH: DESC_ZH_CLEANED,
}


def cleaned_column_name(column: str) -> str:
    return _CLEANED_COLUMNS.get(column, f"{column}_cleaned")


def _tier_of(reason: str) -> int | None:
    if reason == "whitelist":
        return 1
    if reason.startswith("pattern:"):
        return 2
    if reason == "auto_loanword":
        return 3
    if reason.startswith("cedict:"):
        return 4
    if reason == "tier5_bge_m3":
        return 5
    return None


@dataclass
class CadoConfig:
    """Cấu hình pipeline CADO (mặc định = hành vi đã kiểm chứng trên title)."""

    columns: tuple[str, ...] = (TITLE_ZH,)
    loanword_column: str = TITLE_ZH

    max_tier: int = 4                 # 1..5; 5 = bật bge-m3 fallback
    use_loanwords: bool = True        # bật/tắt T3
    min_freq: int = TIER3_MIN_FREQ
    overlap_threshold: float = TIER3_OVERLAP_THRESHOLD

    # T5
    tier5_model: str = TIER5_MODEL
    tier5_threshold: float = TIER5_SIMILARITY_THRESHOLD
    tier5_batch_size: int = TIER5_BATCH_SIZE
    tier5_device: str = "cpu"
    tier5_max_pairs: int | None = TIER5_MAX_PAIRS
    tier5_progress_bar: bool = True

    normalize: bool = True
    progress: Callable[[str], None] | None = None


@dataclass
class CadoResult:
    df: pd.DataFrame
    occurrences: pd.DataFrame
    loanword_report: pd.DataFrame
    t5_report: pd.DataFrame
    stats: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# TẦNG 5 — BGE-M3
# ═══════════════════════════════════════════════════════════════════

def _run_tier5(
    candidates: list[dict],
    cfg: CadoConfig,
) -> tuple[dict[tuple[str, str], tuple[bool, float]], list[dict]]:
    """Chấm điểm các cặp (text, word) còn pending.

    Logic: nếu xoá từ mà embedding câu gần như không đổi (cosine > ngưỡng)
    thì từ đó dư thừa → xoá. Quyết định được đưa ra **theo từng lần xuất
    hiện**, không áp dụng toàn cục theo word (sửa bug rò rỉ của bản cũ).
    """
    if not candidates:
        return {}, []

    # Dedup theo (text, word) để không encode trùng.
    pairs: dict[tuple[str, str], dict] = {}
    for item in candidates:
        key = (item["text"], item["word"].lower())
        if key not in pairs:
            pairs[key] = item

    items = list(pairs.values())
    if cfg.tier5_max_pairs is not None and len(items) > cfg.tier5_max_pairs:
        _log(
            cfg,
            f"    ⚠️  T5 giới hạn {cfg.tier5_max_pairs:,} cặp; "
            f"{len(items) - cfg.tier5_max_pairs:,} cặp còn lại được GIỮ nguyên.",
        )
        items = items[: cfg.tier5_max_pairs]

    keep_texts = [item["text"] for item in items]
    remove_texts = [
        remove_words(item["text"], {item["word"]}) for item in items
    ]

    # Bỏ các cặp mà sau khi xoá không thay đổi gì.
    valid_positions = [i for i, t in enumerate(remove_texts) if t != keep_texts[i]]
    if not valid_positions:
        return {}, []
    items = [items[i] for i in valid_positions]
    keep_texts = [keep_texts[i] for i in valid_positions]
    remove_texts = [remove_texts[i] for i in valid_positions]

    _log(cfg, f"    Load {cfg.tier5_model} ({len(items):,} cặp)...")
    from sentence_transformers import SentenceTransformer  # lazy import
    import numpy as np

    model = SentenceTransformer(cfg.tier5_model, device=cfg.tier5_device)
    _log(cfg, "    Encode text gốc...")
    emb_keep = model.encode(
        keep_texts,
        batch_size=cfg.tier5_batch_size,
        normalize_embeddings=True,
        show_progress_bar=cfg.tier5_progress_bar,
        convert_to_numpy=True,
    )
    _log(cfg, "    Encode text sau khi xoá...")
    emb_remove = model.encode(
        remove_texts,
        batch_size=cfg.tier5_batch_size,
        normalize_embeddings=True,
        show_progress_bar=cfg.tier5_progress_bar,
        convert_to_numpy=True,
    )

    similarities = np.sum(emb_keep * emb_remove, axis=1)

    decisions: dict[tuple[str, str], tuple[bool, float]] = {}
    report: list[dict] = []
    for item, similarity in zip(items, similarities):
        key = (item["text"], item["word"].lower())
        remove = bool(similarity > cfg.tier5_threshold)
        decisions[key] = (remove, float(similarity))
        report.append({
            "word": item["word"],
            "similarity": round(float(similarity), 4),
            "decision": "remove" if remove else "keep",
            "text": item["text"],
            "text_no_word": remove_words(item["text"], {item["word"]}),
        })

    report.sort(key=lambda r: -r["similarity"])
    return decisions, report


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_cado(
    df: pd.DataFrame,
    cfg: CadoConfig | None = None,
    *,
    whitelist: Whitelist | None = None,
    cedict: Cedict | None = None,
) -> CadoResult:
    cfg = cfg or CadoConfig()
    whitelist = whitelist or Whitelist.load()
    cedict = cedict or Cedict.load(reverse=False)

    missing = [c for c in cfg.columns if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột trong DataFrame: {missing}")

    _log(cfg, f"🛡️  Whitelist: {len(whitelist):,} từ")
    _log(cfg, f"📚  CC-CEDICT: {cedict.english_keys:,} English keys")

    out = df.copy()

    # ── Chuẩn hoá text nguồn (không ghi đè cột gốc) ──
    sources: dict[str, list[str]] = {}
    for column in cfg.columns:
        values = out[column].tolist()
        sources[column] = [
            normalize_text(value, collapse_whitespace=(column == TITLE_ZH))
            if cfg.normalize else (value if isinstance(value, str) else "")
            for value in values
        ]

    # ── T3: phát hiện từ mượn trên cột title ──
    loanwords: set[str] = set()
    loanword_report: list[dict] = []
    if cfg.max_tier >= 3 and cfg.use_loanwords:
        source_texts = sources.get(cfg.loanword_column, [])
        rows = [
            (str(pid), text)
            for pid, text in zip(out["product_id"], source_texts)
        ]
        candidates = build_candidates(rows, whitelist)
        _log(cfg, f"🔄  T3: {len(candidates):,} ứng viên (chưa T1/T2 quyết định)")
        loanwords, loanword_report = detect_loanwords(
            candidates,
            cedict,
            min_freq=cfg.min_freq,
            overlap_threshold=cfg.overlap_threshold,
        )
        _log(cfg, f"    → {len(loanwords):,} từ mượn được giữ tự động")

    # ── T1–T4: phân loại từng lần xuất hiện ──
    occurrences: list[dict] = []
    removed_by_row: dict[int, dict[str, set[str]]] = {}
    product_ids = out["product_id"].astype(str).tolist() if "product_id" in out.columns else [""] * len(out)

    for column in cfg.columns:
        for row_index, text in enumerate(sources[column]):
            words = extract_latin_words(text)
            if not words:
                continue
            for word in words:
                action, reason = classify_occurrence(
                    word,
                    text,
                    whitelist,
                    loanwords,
                    cedict,
                    max_tier=cfg.max_tier,
                    use_loanwords=cfg.use_loanwords,
                )
                occurrences.append({
                    "row": row_index,
                    "product_id": product_ids[row_index],
                    "column": column,
                    "word": word,
                    "action": action,
                    "reason": reason,
                    "tier": _tier_of(reason),
                    "similarity": None,
                })
                if action == "remove":
                    removed_by_row.setdefault(row_index, {}).setdefault(column, set()).add(word)

    # ── T5: xử lý các occurrence còn no_cedict_entry ──
    t5_report: list[dict] = []
    if cfg.max_tier >= 5:
        pending = [
            {
                "row": occ["row"],
                "column": occ["column"],
                "word": occ["word"],
                "text": sources[occ["column"]][occ["row"]],
            }
            for occ in occurrences
            if occ["reason"] == "no_cedict_entry"
        ]
        _log(cfg, f"🧠  T5: {len(pending):,} cặp (word, text) cần kiểm tra")
        decisions, t5_report = _run_tier5(pending, cfg)

        for occ in occurrences:
            if occ["reason"] != "no_cedict_entry":
                continue
            text = sources[occ["column"]][occ["row"]]
            key = (text, occ["word"].lower())
            decision = decisions.get(key)
            if decision is None:
                continue
            should_remove, similarity = decision
            occ["similarity"] = round(similarity, 4)
            if should_remove:
                occ["action"] = "remove"
                occ["reason"] = "tier5_bge_m3"
                occ["tier"] = 5
                removed_by_row.setdefault(occ["row"], {}).setdefault(occ["column"], set()).add(occ["word"])

        _log(
            cfg,
            f"    → T5 xoá {sum(1 for o in occurrences if o['reason'] == 'tier5_bge_m3'):,} từ",
        )

    # ── Tạo cột cleaned ──
    changed_by_column: Counter[str] = Counter()
    for column in cfg.columns:
        cleaned_column = cleaned_column_name(column)
        cleaned_values: list[str] = []
        source_values = sources[column]
        for row_index, text in enumerate(source_values):
            words = removed_by_row.get(row_index, {}).get(column, set())
            cleaned = remove_words(text, words) if words else text
            cleaned_values.append(cleaned)
            if cleaned != text:
                changed_by_column[column] += 1
        out[cleaned_column] = cleaned_values

    # ── Thống kê ──
    actions = Counter(occ["action"] for occ in occurrences)
    reasons = Counter(occ["reason"] for occ in occurrences)
    tiers = Counter(occ["tier"] for occ in occurrences if occ["tier"] is not None)
    changed_rows: set[int] = set()
    for row_index, columns in removed_by_row.items():
        if any(columns.values()):
            changed_rows.add(row_index)

    _log(cfg, "📊  Hoàn tất CADO")
    _log(cfg, f"    Tổng occurrence: {len(occurrences):,}")
    _log(cfg, f"    ✅ keep: {actions['keep']:,} | 🔴 remove: {actions['remove']:,} | ⏳ pending: {actions['pending']:,}")
    for column, count in changed_by_column.items():
        _log(cfg, f"    Sản phẩm đổi {cleaned_column_name(column)}: {count:,}")

    stats = {
        "total_occurrences": len(occurrences),
        "keep": actions["keep"],
        "remove": actions["remove"],
        "pending": actions["pending"],
        "changed_rows": len(changed_rows),
        "changed_by_column": dict(changed_by_column),
        "actions": dict(actions),
        "reasons": dict(reasons),
        "tiers": dict(tiers),
        "loanwords": len(loanwords),
        "removed_words": Counter(
            occ["word"] for occ in occurrences if occ["action"] == "remove"
        ).most_common(30),
    }

    return CadoResult(
        df=out,
        occurrences=pd.DataFrame(occurrences),
        loanword_report=pd.DataFrame(loanword_report),
        t5_report=pd.DataFrame(t5_report),
        stats=stats,
    )


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _log(cfg: CadoConfig, message: str) -> None:
    if cfg.progress:
        cfg.progress(message)
