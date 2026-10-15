#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO Bitext Filtering — Tầng 2: Entity Validation
====================================================
Áp dụng 7 tín hiệu rule-based cho các cặp AMBIGUOUS (0.55 ≤ cos < 0.75)
để nâng precision:

  S1: Brand/Model overlap     (w=0.20)
  S2: Number alignment         (w=0.15)
  S3: Unit consistency         (w=0.20)
  S4: CC-CEDICT keyword        (w=0.15)
  S5: Length ratio             (w=0.10)
  S6: No CJK in VI             (w=0.10)
  S7: No keyword stuffing      (w=0.10)

Threshold: entity_score ≥ 0.40 → KEEP, 0.25-0.40 → REVIEW, < 0.25 → DISCARD

Input:  data/cleaned/bilingual_zh_vi_bitext_ready.parquet
Output: Same file + entity_score + 7 signal columns + tier2_label
"""

import pandas as pd, re, csv, os
from collections import Counter, defaultdict

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
INP = 'data/cleaned/bilingual_zh_vi_bitext_ready.parquet'
WHITELIST_CSV = 'csv/whitelist.csv'
CEDICT_FILE = 'cedict_ts.u8'

LATIN_ZH = re.compile(r'[a-zA-Z\u00C0-\u024F]{2,}')  # ≥2 chars cho ZH
LATIN_VI = re.compile(r'[a-zA-Z]{3,}')                # ≥3 chars cho VI
CJK = re.compile(r'[\u4e00-\u9fff]')

# Đơn vị ZH → VI
UNIT_MAP = {'寸': 'inch', '英寸': 'inch', '斤': 'kg', '克': 'g',
            '厘米': 'cm', '米': 'm', '升': 'l', '毫升': 'ml', '千克': 'kg'}

# Số Hán → Ả Rập
HAN_NUM = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
           '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
           '两': '2', '百': '00', '千': '000', '万': '0000', '零': '0'}

# ═══════════
# Trọng số — S4 là gatekeeper (0.40), các tín hiệu còn lại là sanity check
# ═══════════
W = {'S1': 0.15, 'S2': 0.10, 'S3': 0.10, 'S4': 0.40,
     'S5': 0.10, 'S6': 0.05, 'S7': 0.10}


# ═══════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════

def load_whitelist(path):
    wl = set()
    if not os.path.exists(path): return wl
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            w = row['word'].strip()
            if w: wl.add(w.lower())
    return wl


def load_cedict(path):
    """Parse CC-CEDICT → {english_word: [chinese_words]} + reverse index."""
    eng2zh = defaultdict(list)
    zh2eng = defaultdict(set)  # reverse: chinese word → english words
    entry_pat = re.compile(r'^(\S+)\s+\S+\s+\[.+?\]\s+/(.+)/$')
    if not os.path.exists(path): return dict(eng2zh), dict(zh2eng)
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            m = entry_pat.match(line)
            if not m: continue
            trad = m.group(1)
            if len(CJK.findall(trad)) < 2: continue
            gloss = m.group(2).lower()
            for token in set(re.findall(r'[a-z]+', gloss)):
                if len(token) >= 2:
                    if trad not in eng2zh[token]:
                        eng2zh[token].append(trad)
                    zh2eng[trad].add(token)
    for k in eng2zh:
        eng2zh[k] = sorted(set(eng2zh[k]), key=len, reverse=True)
    return dict(eng2zh), dict(zh2eng)


# ═══════════════════════════════════════
# SIGNAL FUNCTIONS
# ═══════════════════════════════════════

def extract_brands(text, is_zh, whitelist_lower):
    """Trích brand/model/SKU từ text."""
    if not isinstance(text, str): return set()
    if is_zh:
        words = LATIN_ZH.findall(text)
    else:
        words = LATIN_VI.findall(text)
    brands = set()
    for w in words:
        wl = w.lower()
        if wl in whitelist_lower or re.search(r'\d', w) or re.match(r'^[A-Z]{2,6}$', w):
            brands.add(wl)
    return brands


def signal_s1(row, whitelist_lower):
    """S1: Brand/Model overlap (case-insensitive)."""
    zh_b = extract_brands(row['title_zh'], True, whitelist_lower)
    vi_b = extract_brands(row['title_vi_cleaned'], False, whitelist_lower)
    has_any = len(zh_b) > 0 or len(vi_b) > 0
    if not has_any:
        return 1  # Không có brand → tín hiệu không áp dụng → pass
    return 1 if len(zh_b & vi_b) > 0 else 0


def extract_numbers(text):
    """Trích số Ả Rập + Hán từ text."""
    nums = set()
    for m in re.finditer(r'\d+', str(text)):
        nums.add(int(m.group()))
    for char in str(text):
        if char in HAN_NUM:
            nums.add(int(HAN_NUM[char]))
    return nums


def signal_s2(row):
    """S2: Number alignment (có mặt số)."""
    zh_n = extract_numbers(row['title_zh'])
    vi_n = extract_numbers(row['title_vi_cleaned'])
    has_zh = len(zh_n) > 0
    has_vi = len(vi_n) > 0
    if not has_zh and not has_vi:
        return 1  # Không có số → pass
    if has_zh and not has_vi:
        return 1  # Số Hán → chữ VI (vd: "五合一" → "năm trong một") → pass
    if not has_zh and has_vi:
        return 0  # VI thêm số lạ → nghi vấn
    # Cả 2 có số → kiểm tra overlap
    return 1 if len(zh_n & vi_n) > 0 else 0


def signal_s3(row):
    """S3: Unit consistency."""
    zh = str(row['title_zh'])
    vi = str(row['title_vi_cleaned']).lower()
    has_unit = False
    for zh_unit, vi_equiv in UNIT_MAP.items():
        if zh_unit in zh:
            has_unit = True
            if vi_equiv not in vi:
                return 0
    return 1 if has_unit else 1  # Không có đơn vị → pass


def signal_s4(row, zh2eng):
    """S4a: CC-CEDICT keyword overlap (giữ nguyên — kiểm tra EN keywords)."""
    zh = str(row['title_zh'])
    vi = str(row['title_vi_cleaned']).lower()
    cjk_chars = CJK.findall(zh)
    if len(cjk_chars) < 2:
        return 1  # Không đủ bigram → pass
    for i in range(len(cjk_chars) - 1):
        bg = cjk_chars[i] + cjk_chars[i+1]
        if bg in zh2eng:
            for eng_word in zh2eng[bg]:
                if eng_word in vi:
                    return 1
    return 0


def build_s4_embedding_model():
    """Khởi tạo MiniLM model cho S4 cross-lingual matching (cache toàn cục)."""
    import time
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    return model


def signal_s4_embedding(row, model, vi_emb, vi_idx_map):
    """S4b: Cross-lingual embedding matching.
    Embed top-3 ZH bigrams bằng MiniLM → cosine vs VI embedding.
    Nếu max cosine > 0.65 → ZH bigram có mặt ngữ nghĩa trong VI → pass."""
    zh = str(row['title_zh'])
    cjk_chars = CJK.findall(zh)
    if len(cjk_chars) < 2:
        return 1
    
    # Top-3 bigrams (theo tần suất trong câu)
    bigrams = []
    for i in range(len(cjk_chars) - 1):
        bigrams.append(cjk_chars[i] + cjk_chars[i+1])
    if not bigrams:
        return 1
    
    # Chọn top-3 unique
    unique_bg = list(dict.fromkeys(bigrams))[:3]
    
    # Embed bigrams
    bg_embs = model.encode(unique_bg, normalize_embeddings=True, convert_to_numpy=True)
    
    # Cosine vs VI embedding (đã normalize → dot product)
    vi_emb_vec = vi_emb  # single vector
    cosines = np.sum(bg_embs * vi_emb_vec, axis=1)
    
    return 1 if np.max(cosines) > 0.65 else 0


def signal_s5(row):
    """S5: Length ratio (CJK chars + Latin words cho ZH, space-split cho VI)."""
    zh = str(row['title_zh'])
    vi = str(row['title_vi_cleaned'])
    # ZH: đếm CJK chars (mỗi char ≈ 1 từ) + số Latin words
    zh_cjk = len(CJK.findall(zh))
    zh_latin = len(LATIN_ZH.findall(zh))
    zh_words = zh_cjk + zh_latin
    # VI: space-split
    vi_words = len(vi.split())
    if zh_words == 0: return 0
    ratio = vi_words / zh_words
    return 1 if 0.4 <= ratio <= 3.0 else 0


def signal_s6(row):
    """S6: CJK chars in VI = 0."""
    vi = str(row['title_vi_cleaned'])
    return 0 if CJK.search(vi) else 1


def signal_s7(row):
    """S7: VI không bị keyword stuffing (từ lặp < 5)."""
    words = str(row['title_vi_cleaned']).lower().split()
    if len(words) == 0: return 1
    return 0 if max(Counter(words).values()) >= 5 else 1


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    import numpy as np, time as time_mod
    print('=' * 60)
    print('🔍 TẦNG 2: ENTITY VALIDATION (7 tín hiệu)')
    print('=' * 60)

    # Load resources
    whitelist_lower = load_whitelist(WHITELIST_CSV)
    print(f'Whitelist: {len(whitelist_lower)} từ')

    cedict, zh2eng = load_cedict(CEDICT_FILE)
    print(f'CC-CEDICT: {len(cedict):,} EN entries, {len(zh2eng):,} ZH reverse entries')

    # Load data
    df = pd.read_parquet(INP)
    print(f'Dataset: {len(df):,} rows')

    # Xác định rows cần xử lý
    if 'tier1_label' in df.columns:
        mask = df['tier1_label'] == 'AMBIGUOUS'
        target_rows = mask.sum()
        print(f'Tier1 AMBIGUOUS: {target_rows:,} → cần Tầng 2')
    else:
        mask = pd.Series(True, index=df.index)
        target_rows = len(df)
        print(f'Không có tier1_label → xử lý toàn bộ {target_rows:,} rows')

    # ── S4b: Pre-compute VI embeddings ──
    print(f'\n🧠 S4b: Loading MiniLM + encoding VI texts...')
    t0 = time_mod.time()
    model = build_s4_embedding_model()
    vi_texts = [str(df.at[i, 'title_vi_cleaned']) for i in df.index[mask]]
    vi_embs = model.encode(vi_texts, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    vi_emb_map = {idx: vi_embs[j] for j, idx in enumerate(df.index[mask])}
    print(f'   Encoded {len(vi_texts):,} VI texts ({time_mod.time()-t0:.0f}s)')

    # ── Tính 7 tín hiệu ──
    print(f'\nTính 7 tín hiệu cho {target_rows:,} rows...')
    idx = df.index[mask]

    signals = {f'S{i}': [] for i in range(1, 8)}
    s4a_vals = []
    s4b_vals = []

    for i in idx:
        row = df.iloc[i]
        signals['S1'].append(signal_s1(row, whitelist_lower))
        signals['S2'].append(signal_s2(row))
        signals['S3'].append(signal_s3(row))
        # S4 = S4a (CEDICT) OR S4b (embedding)
        s4a = signal_s4(row, zh2eng)
        s4b = signal_s4_embedding(row, model, vi_emb_map[i], None)
        s4a_vals.append(s4a)
        s4b_vals.append(s4b)
        signals['S4'].append(1 if (s4a or s4b) else 0)
        signals['S5'].append(signal_s5(row))
        signals['S6'].append(signal_s6(row))
        signals['S7'].append(signal_s7(row))

    # ── Entity score ──
    entity_scores = []
    for j in range(len(signals['S1'])):
        score = sum(W[f'S{i+1}'] * signals[f'S{i+1}'][j] for i in range(7))
        entity_scores.append(round(score, 2))

    # ── Gán vào DataFrame ──
    for col_name in ['signal_S1','signal_S2','signal_S3','signal_S4','signal_S4a','signal_S4b',
                      'signal_S5','signal_S6','signal_S7','entity_score']:
        df[col_name] = None

    for k, i in enumerate(idx):
        df.at[i, 'signal_S4a'] = s4a_vals[k]
        df.at[i, 'signal_S4b'] = s4b_vals[k]
        for s in range(1, 8):
            df.at[i, f'signal_S{s}'] = signals[f'S{s}'][k]
        df.at[i, 'entity_score'] = entity_scores[k]

    # ── Tier2 label ──
    df['tier2_label'] = df['tier1_label'] if 'tier1_label' in df.columns else None
    for k, i in enumerate(idx):
        es = entity_scores[k]
        if es >= 0.60:
            df.at[i, 'tier2_label'] = 'KEEP'
        elif es >= 0.40:
            df.at[i, 'tier2_label'] = 'REVIEW'
        else:
            df.at[i, 'tier2_label'] = 'DISCARD'

    # ── Stats ──
    processed = df[df['entity_score'].notna()]
    print(f'\n{"="*60}')
    print(f'📊 KẾT QUẢ TẦNG 2 (Entity Score, ngưỡng KEEP≥0.60)')
    print(f'{"="*60}')
    print(f'  Processed:      {len(processed):,}')
    print(f'  Mean score:     {processed["entity_score"].mean():.2f}')
    for label in ['KEEP', 'REVIEW', 'DISCARD']:
        n = (processed['tier2_label'] == label).sum()
        print(f'  {label:<15} {n:>6,}')

    # ── Signal contributions ──
    print(f'\n📊 TỈ LỆ PASS TỪNG TÍN HIỆU')
    for s, label in [('S4a','S4a CEDICT'),('S4b','S4b Embedding'),('S4','S4 Combined')]:
        if s in df.columns:
            vals = processed[s].dropna()
            if len(vals) > 0:
                rate = (vals == 1).sum() / len(vals) * 100
                print(f'  {label:<20} {rate:5.1f}% pass')
    for i in [1,2,3,5,6,7]:
        col = f'signal_S{i}'
        vals = processed[col].dropna()
        if len(vals) > 0:
            rate = (vals == 1).sum() / len(vals) * 100
            print(f'  S{i} (w={W[f"S{i}"]:.2f}){"":<8} {rate:5.1f}% pass')

    # ── Save ──
    df.to_parquet(INP, index=False)
    print(f'\n✅ Saved: {INP}')

    # ── Final tally ──
    if 'tier1_label' in df.columns:
        print(f'\n{"="*60}')
        print(f'📊 TỔNG HỢP CUỐI CÙNG')
        print(f'{"="*60}')
        keep1 = (df['tier1_label'] == 'KEEP_HIGH').sum()
        keep2 = ((df['tier1_label'] == 'AMBIGUOUS') & (df['tier2_label'] == 'KEEP')).sum()
        review = ((df['tier1_label'] == 'AMBIGUOUS') & (df['tier2_label'] == 'REVIEW')).sum()
        discard1 = (df['tier1_label'] == 'DISCARD_LOW').sum()
        discard2 = ((df['tier1_label'] == 'AMBIGUOUS') & (df['tier2_label'] == 'DISCARD')).sum()

        print(f'  ✅ KEEP:    {keep1+keep2:>6,}  (T1:{keep1:,} + T2:{keep2:,})')
        print(f'  🟡 REVIEW:  {review:>6,}')
        print(f'  🔴 DISCARD: {discard1+discard2:>6,}  (T1:{discard1:,} + T2:{discard2:,})')
        print(f'  ─────────────────────')
        print(f'  Training-ready: {keep1+keep2:,} ({(keep1+keep2)/len(df)*100:.1f}%)')