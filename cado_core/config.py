# -*- coding: utf-8 -*-
"""Đường dẫn, hằng số và regex dùng chung cho toàn bộ pipeline.

Mọi đường dẫn mặc định đều được neo vào gốc dự án (``PROJECT_ROOT``),
không phụ thuộc vào thư mục đang đứng khi chạy script.
"""

from __future__ import annotations

import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PARQUET = PROJECT_ROOT / "data/raw/bilingual_zh_vi.parquet"
CADO_PARQUET = PROJECT_ROOT / "data/cleaned/bilingual_zh_vi_cado.parquet"
BITEXT_READY_PARQUET = PROJECT_ROOT / "data/cleaned/bilingual_zh_vi_bitext_ready.parquet"
BITEXT_SCORED_PARQUET = PROJECT_ROOT / "data/cleaned/bitext_scored.parquet"

WHITELIST_CSV = PROJECT_ROOT / "csv/whitelist.csv"
VI_MAP_CSV = PROJECT_ROOT / "csv/vi_english_map.csv"
CEDICT_FILE = PROJECT_ROOT / "dictionary/cedict_ts.u8"

CSV_DIR = PROJECT_ROOT / "csv"
CADO_CSV_DIR = CSV_DIR / "cado"
BITEXT_CSV_DIR = CSV_DIR / "bitext_mining"
ENGLISH_REMOVER_CSV_DIR = CSV_DIR / "english_remover"

# Tiki — Phase 1: profile title + description key-value, build BT units.
TIKI_INPUT_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_vi_cado.parquet"
TIKI_PROFILE_JSON = PROJECT_ROOT / "data/cleaned/tiki_profile.json"
TIKI_PRODUCTS_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_products.parquet"
TIKI_BT_UNITS_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_bt_units.parquet"
TIKI_UNITS_STATS_JSON = PROJECT_ROOT / "data/cleaned/tiki_units_stats.json"
TIKI_KEY_INVENTORY_CSV = CSV_DIR / "tiki/tiki_key_inventory.csv"
TIKI_PARSE_ERRORS_CSV = CSV_DIR / "tiki/tiki_parse_errors.csv"
TIKI_KEY_MAP_CSV = CSV_DIR / "tiki/tiki_key_map.csv"

# Phase 1.5 — length prior học từ 1688 human, áp dụng cho Tiki units.
LENGTH_PRIOR_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688.json"
LENGTH_PRIOR_CSV = CSV_DIR / "length_prior/length_prior_by_bucket.csv"
TIKI_UNITS_LENGTH_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_bt_units_length.parquet"
TIKI_LENGTH_PRIOR_STATS_JSON = PROJECT_ROOT / "data/cleaned/tiki_length_prior_stats.json"

# Phase A — length prior v2: value-only, value_class, CJK, split-safe.
LENGTH_PRIOR_V2_TRAIN_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688_v2_train.json"
LENGTH_PRIOR_V2_ALL_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688_v2_all.json"
LENGTH_PRIOR_V2_HOLDOUT_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688_v2_holdout_metrics.json"
LENGTH_PRIOR_V2_CSV = CSV_DIR / "length_prior_v2/prior_value_by_class_bucket.csv"
LENGTH_PRIOR_V2_TRAIN_CSV = CSV_DIR / "length_prior_v2/prior_value_by_class_bucket_train.csv"
LENGTH_PRIOR_V2_SPLIT_MISMATCH_CSV = CSV_DIR / "length_prior_v2/attribute_split_mismatch.csv"
LENGTH_PRIOR_V2_SPLIT_SAMPLE_CSV = CSV_DIR / "length_prior_v2/attribute_split_sample.csv"
LENGTH_PRIOR_V2_SPLIT_SUMMARY_CSV = CSV_DIR / "length_prior_v2/attribute_split_summary.csv"
TIKI_UNITS_LENGTH_V2_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_bt_units_length_v2.parquet"
TIKI_LENGTH_PRIOR_V2_STATS_JSON = PROJECT_ROOT / "data/cleaned/tiki_length_prior_v2_stats.json"

# Phase B/C — alignment fallback + hierarchical title smoothing.
LENGTH_PRIOR_V3_TRAIN_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688_v3_train.json"
LENGTH_PRIOR_V3_ALL_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688_v3_all.json"
LENGTH_PRIOR_V3_HOLDOUT_JSON = PROJECT_ROOT / "data/cleaned/length_prior_1688_v3_holdout_metrics.json"
LENGTH_PRIOR_V3_CSV = CSV_DIR / "length_prior_v3/prior_value_by_class_bucket.csv"
LENGTH_PRIOR_V3_TRAIN_CSV = CSV_DIR / "length_prior_v3/prior_value_by_class_bucket_train.csv"
LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV = CSV_DIR / "length_prior_v3/alignment_key_map.csv"
LENGTH_PRIOR_V3_ALIGNMENT_SAMPLE_CSV = CSV_DIR / "length_prior_v3/alignment_audit_sample.csv"
LENGTH_PRIOR_V3_ALIGNMENT_SUMMARY_CSV = CSV_DIR / "length_prior_v3/alignment_summary.csv"
PHASE_EVALUATION_JSON = PROJECT_ROOT / "data/cleaned/phase_evaluation.json"
PHASE_EVALUATION_MD = PROJECT_ROOT / "data/cleaned/phase_evaluation.md"
TIKI_UNITS_LENGTH_V3_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_bt_units_length_v3.parquet"
TIKI_LENGTH_PRIOR_V3_STATS_JSON = PROJECT_ROOT / "data/cleaned/tiki_length_prior_v3_stats.json"

# Plan A — pre-BT preparation: key map, routing, BT input/cache, state machine.
TIKI_BT_UNITS_PLAN_A_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_bt_units_plan_a.parquet"
TIKI_BT_INPUT_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_bt_input.parquet"
TIKI_TRANSLATION_CACHE_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_translation_cache.parquet"
TIKI_PRODUCTS_BT_STATE_PARQUET = PROJECT_ROOT / "data/cleaned/tiki_products_bt_state.parquet"
TIKI_BT_PLAN_A_STATS_JSON = PROJECT_ROOT / "data/cleaned/tiki_bt_plan_a_stats.json"
TIKI_KEY_MAP_REVIEW_CSV = CSV_DIR / "tiki/tiki_key_map_review.csv"
TIKI_BT_ROUTING_CSV = CSV_DIR / "tiki/tiki_bt_routing_summary.csv"
TIKI_BT_INPUT_SAMPLE_CSV = CSV_DIR / "tiki/tiki_bt_input_sample.csv"

# Plan B0 — Qwen key-map review via Alibaba Cloud DashScope.
KEY_MAP_REVIEW_INPUT_CSV = CSV_DIR / "tiki/key_map_review_input.csv"
KEY_MAP_QWEN_VOTES_CSV = CSV_DIR / "tiki/key_map_qwen_votes.csv"
KEY_MAP_QWEN_REVIEW_CSV = CSV_DIR / "tiki/key_map_qwen_review.csv"
KEY_MAP_FINAL_CSV = CSV_DIR / "tiki/key_map_final.csv"
KEY_MAP_SILVER_EVAL_CSV = CSV_DIR / "tiki/key_map_silver_eval.csv"
KEY_MAP_QWEN_STATS_JSON = PROJECT_ROOT / "data/cleaned/key_map_qwen_review_stats.json"


def resolve_path(value: str | Path | None, default: Path) -> Path:
    """Chuẩn hoá đường dẫn: tương đối → tính từ gốc dự án."""
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ═══════════════════════════════════════════════════════════════════
# REGEX
# ═══════════════════════════════════════════════════════════════════

# Chuỗi Latin liên tục (tiếng Anh + các ngôn ngữ dùng bảng chữ Latin).
# \w không dùng được vì Python coi chữ Hán là \w.
# \u1E00-\u1EFF là Latin Extended Additional — chứa phần lớn nguyên âm có dấu
# tiếng Việt (những, được, ...). Thiếu range này sẽ tách "những" thành "nh"+"ng".
LATIN_WORD_RE = re.compile(
    r"[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F\u1E00-\u1EFF]+"
)

# Ký tự Hán (CJK Unified Ideographs + Extension A + Compatibility Ideographs).
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# Khối CJK mở rộng dùng khi cần xoá cả cụm (bao gồm dấu full-width).
CJK_BLOCK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\uff00-\uffef]+")

# Ký tự vô hình thường gặp trong dữ liệu crawl.
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")

# Khoảng trắng không phải newline (dùng để dọn text nhưng giữ format bảng).
INLINE_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")


# ═══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Tầng 3 — Auto loanword
TIER3_MIN_FREQ = 3
TIER3_OVERLAP_THRESHOLD = 0.20

# Tầng 5 — BGE-M3 semantic fallback
TIER5_MODEL = "BAAI/bge-m3"
TIER5_SIMILARITY_THRESHOLD = 0.98
TIER5_BATCH_SIZE = 64
TIER5_MAX_PAIRS: int | None = None  # None = không giới hạn (tránh bỏ sót âm thầm)

# Bitext filter
BITEXT_COSINE_DISCARD = 0.55
BITEXT_HYBRID_KEEP = 0.50
BITEXT_COSINE_WEIGHT = 0.70
BITEXT_LEXICAL_WEIGHT = 0.30
BITEXT_BATCH_SIZE = 256
BITEXT_MODEL = "BAAI/bge-m3"
BITEXT_TOKENS_PER_CJK_MIN = 0.8   # VI tokens / ZH CJK chars
BITEXT_TOKENS_PER_CJK_MAX = 6.0

# Tên cột
TITLE_ZH = "title_zh"
DESC_ZH = "description_zh"
TITLE_VI = "title_vi"
DESC_VI = "description_vi"

# Cột output quy ước
TITLE_ZH_CLEANED = "title_zh_cleaned"
DESC_ZH_CLEANED = "description_zh_cleaned"
TITLE_VI_CLEANED = "title_vi_cleaned"
DESC_VI_CLEANED = "desc_vi_cleaned"  # giữ tên legacy để tương thích artifact cũ
