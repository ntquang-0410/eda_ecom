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
