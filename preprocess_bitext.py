#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess dataset cho Bitext Filtering pipeline.

Sửa các vấn đề ảnh hưởng đến BGE-M3 cosine similarity:
  P0 (critical): V1 CJK copy → DISCARD, V2 Full-width → NFKC, V3 CJK in VI → clean
  P1 (important): V10 CJK brackets → ASCII, V4 Trailing punct → trim
  P2 (nice): V5 Decimal comma → dot, V16 Star/pipes → normalize

Output: data/cleaned/bilingual_zh_vi_bitext_ready.parquet
"""

import pandas as pd, re, os, unicodedata
from collections import Counter


# ═══════════════════════════════════════
# P0: CRITICAL FIXES
# ═══════════════════════════════════════

def fix_fullwidth(text: str) -> str:
    """V2: Chuẩn hóa NFKC — gập full-width về ASCII."""
    if not isinstance(text, str): return text
    return unicodedata.normalize('NFKC', text)


def remove_cjk_from_vi(text: str) -> str:
    """V3: Xóa ký tự CJK khỏi văn bản tiếng Việt."""
    if not isinstance(text, str): return text
    # Xóa các block CJK
    cleaned = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def is_cjk_copy(row) -> bool:
    """V1: Phát hiện title_vi chép nguyên title_zh (có chứa CJK)."""
    zh = str(row['title_zh'])
    vi = str(row['title_vi'])
    return zh == vi and bool(re.search(r'[\u4e00-\u9fff]', vi))


# ═══════════════════════════════════════
# P1: IMPORTANT FIXES
# ═══════════════════════════════════════

def fix_cjk_brackets(text: str) -> str:
    """V10: Chuẩn hóa 【】 → []."""
    if not isinstance(text, str): return text
    trans = str.maketrans({'【': '[', '】': ']'})
    return text.translate(trans)


def trim_trailing_punct(text: str) -> str:
    """V4: Gọt dấu chấm câu đuôi."""
    if not isinstance(text, str): return text
    return re.sub(r'[.,;!?。，；！？]+$', '', text.strip())


# ═══════════════════════════════════════
# P2: NICE-TO-HAVE
# ═══════════════════════════════════════

def fix_decimal_comma(text: str) -> str:
    """V5: 4,3 → 4.3."""
    if not isinstance(text, str): return text
    return re.sub(r'(?<=\d),(?=\d)', '.', text)


def fix_star_operator(text: str) -> str:
    """V16: 10*20 → 10x20."""
    if not isinstance(text, str): return text
    return re.sub(r'(\d+)\*(\d+)', r'\1x\2', text)


def fix_cjk_pipe(text: str) -> str:
    """V16c: 丨 → |."""
    if not isinstance(text, str): return text
    return text.replace('丨', '|')


def has_stuffing(text: str) -> bool:
    """V14: Phát hiện nhồi từ khóa."""
    words = str(text).lower().split()
    return len(words) > 0 and max(Counter(words).values()) >= 5


def len_ratio(row) -> float:
    """Tỉ lệ độ dài VI / ZH."""
    zl = len(str(row['title_zh']))
    vl = len(str(row['title_vi']))
    return vl / zl if zl > 0 else 999


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    INP = 'data/cleaned/bilingual_zh_vi_cado.parquet'
    OUT = 'data/cleaned/bilingual_zh_vi_bitext_ready.parquet'
    DISCARD_CSV = 'csv/bitext_discarded_rows.csv'

    print('='*60)
    print('🔧 PREPROCESS CHO BITEXT FILTERING')
    print('='*60)

    df = pd.read_parquet(INP)
    print(f'\nInput: {len(df):,} rows')

    # ── P0: V1 ──
    cjk_copy_mask = df.apply(is_cjk_copy, axis=1)
    discarded = df[cjk_copy_mask].copy()
    df = df[~cjk_copy_mask].copy()
    print(f'V1 CJK copy → DISCARD: {cjk_copy_mask.sum()} rows')

    # ── P0: V2 (NFKC trên toàn bộ cột ZH + VI) ──
    for col in ['title_zh', 'description_zh', 'title_vi', 'description_vi',
                'title_vi_cleaned', 'desc_vi_cleaned']:
        if col in df.columns:
            df[col] = df[col].apply(fix_fullwidth)
    print('V2 Full-width → NFKC normalized')

    # ── P0: V3 (xóa CJK khỏi VI) ──
    for col in ['title_vi', 'description_vi', 'title_vi_cleaned', 'desc_vi_cleaned']:
        if col in df.columns:
            df[col] = df[col].apply(remove_cjk_from_vi)
    print('V3 CJK in VI → cleaned')

    # ── P1: V10 ──
    for col in ['title_zh', 'description_zh', 'title_vi', 'description_vi',
                'title_vi_cleaned', 'desc_vi_cleaned']:
        if col in df.columns:
            df[col] = df[col].apply(fix_cjk_brackets)
    print('V10 CJK brackets → ASCII []')

    # ── P1: V4 ──
    for col in ['title_vi', 'title_vi_cleaned']:
        if col in df.columns:
            df[col] = df[col].apply(trim_trailing_punct)
    print('V4 Trailing punct → trimmed')

    # ── P2: V5, V16 ──
    for col in ['title_vi', 'title_vi_cleaned']:
        if col in df.columns:
            df[col] = df[col].apply(fix_decimal_comma)
    print('V5 Decimal comma → dot')

    for col in ['title_zh', 'description_zh']:
        if col in df.columns:
            df[col] = df[col].apply(fix_star_operator)
            df[col] = df[col].apply(fix_cjk_pipe)
    print('V16 Star/pipes → normalized')

    # ── Flag remaining issues ──
    df['flag_stuffing'] = df['title_vi'].apply(has_stuffing)
    df['flag_truncated'] = df.apply(lambda r: len_ratio(r) < 0.8, axis=1)
    df['flag_bloated'] = df.apply(lambda r: len_ratio(r) > 8.0, axis=1)

    print(f'\nFlags added:')
    print(f'  stuffing: {df["flag_stuffing"].sum():,}')
    print(f'  truncated: {df["flag_truncated"].sum():,}')
    print(f'  bloated: {df["flag_bloated"].sum():,}')

    # Save discarded
    if len(discarded) > 0:
        os.makedirs('csv', exist_ok=True)
        discarded[['product_id', 'title_zh', 'title_vi', 'category']].to_csv(
            DISCARD_CSV, index=False, encoding='utf-8-sig')
        print(f'\nDiscarded: {DISCARD_CSV} ({len(discarded)} rows)')

    # Save output
    df.to_parquet(OUT, index=False)
    print(f'\n✅ {OUT}')
    print(f'   Rows: {len(df):,} (removed {cjk_copy_mask.sum()})')
    print(f'   Size: {os.path.getsize(OUT)/1024/1024:.1f} MB')
    print(f'   Columns: {list(df.columns)}')
    print(f'\nSẵn sàng cho Bitext Filtering pipeline!')


if __name__ == '__main__':
    main()