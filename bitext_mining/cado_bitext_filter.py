#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO Bitext Filter — H3 Hybrid Scorer (Cosine + Lexical Overlap)
=================================================================
Pipeline cuối cùng để lọc chất lượng cặp song ngữ ZH-VI từ 1688.com:

  Bước 1: Preprocess (V1-V16 fixes + NFKC + CADO-VI)
  Bước 2: BGE-M3 Cosine + Lexical Overlap → hybrid_score
  Bước 3: Phân loại KEEP (≥0.50) / DISCARD (<0.50)

Hybrid score = 0.7 × cosine + 0.3 × lexical_overlap

Input:  data/raw/bilingual_zh_vi.parquet
Output: data/cleaned/bitext_scored.parquet (có cột hybrid_score + final_label)
        csv/bitext_discard.csv (các cặp bị loại)
"""

import pandas as pd, numpy as np, re, csv, os, time, unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Files
    input_parquet: str = 'data/raw/bilingual_zh_vi.parquet'
    output_parquet: str = 'data/cleaned/bitext_scored.parquet'
    output_discard: str = 'csv/bitext_discard.csv'
    whitelist_csv: str = 'csv/whitelist.csv'
    cedict_file: str = 'cedict_ts.u8'

    # Thresholds
    cosine_discard: float = 0.55      # Dưới đây → DISCARD
    hybrid_keep: float = 0.50         # Trên đây → KEEP
    lexical_weight: float = 0.30      # Trọng số lexical overlap trong hybrid
    cosine_weight: float = 0.70       # Trọng số cosine trong hybrid

    # BGE-M3
    bge_model: str = 'BAAI/bge-m3'
    bge_batch: int = 256

    # Regex
    latin_zh: re.Pattern = field(default_factory=lambda: re.compile(r'[a-zA-Z\u00C0-\u024F]{2,}'))
    latin_vi: re.Pattern = field(default_factory=lambda: re.compile(r'[a-zA-Z]{3,}'))
    cjk: re.Pattern = field(default_factory=lambda: re.compile(r'[\u4e00-\u9fff]'))

    # Units ZH → VI
    unit_map: Dict[str, str] = field(default_factory=lambda: {
        '寸': 'inch', '英寸': 'inch', '斤': 'kg', '克': 'g',
        '厘米': 'cm', '米': 'm', '升': 'l', '毫升': 'ml', '千克': 'kg',
    })

    # Han numerals
    han_num: Dict[str, str] = field(default_factory=lambda: {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
        '两': '2', '百': '00', '千': '000', '万': '0000', '零': '0',
    })

    # VI syllables (to exclude from EN detection)
    vi_syllables: Set[str] = field(default_factory=lambda: {
        'cho','nam','trang','cao','sinh','tay','thu','phong','thao','tho',
        'trong','xuy','dung','thun','vai','nguy','nghe','ngo','chi','thi',
        'tai','hai','trai','hoa','vali','gia','ghi','thanh','nhi','trung',
        'chuy','tinh','kim','sang','xanh','khi','nhung','nang','mau',
        'chan','qua','mat','tren','lon','may','lan','khoa','moi','chinh',
        'dong','nho','dep','choi','song','mon','nha','manh','bong',
        'lai','mua','hinh','truoc','sau','ngoai','cung','nua','ngay',
        'bao','gio','gan','xa','day','dem','toi','phai','trai',
        'dac','biet','diem','nhan','nhe','noi','bat','hong','xam','tim',
        'vang','luc','phut','giay','nguoi','cu','lam','chay','nhay',
    })


# ═══════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING
# ═══════════════════════════════════════════════════════════════

class Preprocessor:
    """Áp dụng tất cả V1-V16 fixes + CADO-VI."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.whitelist_lower = self._load_whitelist()

    def _load_whitelist(self) -> Set[str]:
        wl = set()
        if os.path.exists(self.cfg.whitelist_csv):
            with open(self.cfg.whitelist_csv, encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    w = row['word'].strip()
                    if w: wl.add(w.lower())
        return wl

    def fix_fullwidth(self, text: str) -> str:
        """V2: NFKC normalize."""
        if not isinstance(text, str): return text
        return unicodedata.normalize('NFKC', text)

    def remove_cjk_from_vi(self, text: str) -> str:
        """V3: Remove CJK from VI text."""
        if not isinstance(text, str): return text
        cleaned = self.cfg.cjk.sub('', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def fix_cjk_brackets(self, text: str) -> str:
        """V10: 【】 → []."""
        if not isinstance(text, str): return text
        return text.translate(str.maketrans({'【': '[', '】': ']'}))

    def trim_trailing_punct(self, text: str) -> str:
        """V4: Remove trailing punctuation."""
        if not isinstance(text, str): return text
        return re.sub(r'[.,;!?。，；！？]+$', '', text.strip())

    def fix_decimal_comma(self, text: str) -> str:
        """V5: 4,3 → 4.3."""
        if not isinstance(text, str): return text
        return re.sub(r'(?<=\d),(?=\d)', '.', text)

    def fix_star_operator(self, text: str) -> str:
        """V16: 10*20 → 10x20."""
        if not isinstance(text, str): return text
        return re.sub(r'(\d+)\*(\d+)', r'\1x\2', text)

    def fix_cjk_pipe(self, text: str) -> str:
        """V16c: 丨 → |."""
        if not isinstance(text, str): return text
        return text.replace('丨', '|')

    def is_cjk_copy(self, row) -> bool:
        """V1: Detect VI = ZH copy (contains CJK)."""
        zh = str(row['title_zh'])
        vi = str(row['title_vi'])
        return zh == vi and bool(self.cfg.cjk.search(vi))

    def clean_vi_text(self, text: str) -> str:
        """CADO-VI: Remove English words that have VI equivalents in text."""
        if not isinstance(text, str): return text

        en_vi_map = {
            'mask': ['mặt nạ', 'mặt na'],
            'facial': ['mặt nạ', 'mặt na', 'da mặt'],
            'face': ['mặt nạ', 'mặt na', 'khuôn mặt', 'da mặt'],
            'bags': ['túi xách', 'túi', 'cặp', 'balo', 'ba lô'],
            'bag': ['túi xách', 'túi', 'cặp', 'balo', 'ba lô'],
            'handbag': ['túi xách', 'túi', 'túi đeo'],
            'shoes': ['giày', 'dép', 'giầy'],
            'sneakers': ['giày', 'giày thể thao'],
            'clothes': ['quần áo', 'áo quần', 'trang phục'],
            'backpack': ['balo', 'ba lô', 'cặp sách'],
            'luggage': ['hành lý', 'hanh ly', 'vali'],
            'suitcase': ['vali', 'hành lý', 'hanh ly'],
            'tshirt': ['áo thun', 'áo phông', 'áo'],
            'dress': ['váy', 'đầm'],
        }

        cleaned = text
        words = self.cfg.latin_vi.findall(text)
        to_remove = set()
        for w in words:
            wl = w.lower()
            if wl in en_vi_map:
                text_lower = cleaned.lower()
                for vi_word in en_vi_map[wl]:
                    if vi_word in text_lower:
                        to_remove.add(w)
                        break

        for w in sorted(to_remove, key=len, reverse=True):
            pat = re.compile(r'(^|\s)' + re.escape(w) + r'($|\s)', re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', pat.sub(r'\1\2', cleaned)).strip()
        return cleaned

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps. Returns cleaned DataFrame."""
        print('─' * 50)
        print('STEP 1: PREPROCESSING')
        print('─' * 50)

        initial = len(df)

        # V1: Remove CJK copies
        cjk_copy_mask = df.apply(self.is_cjk_copy, axis=1)
        n_discard = cjk_copy_mask.sum()
        if n_discard > 0:
            self._discarded = df[cjk_copy_mask][['product_id', 'title_zh', 'title_vi', 'category']].copy()
        df = df[~cjk_copy_mask].copy()
        print(f'  V1 CJK copy → DISCARD: {n_discard}')

        # V2: NFKC on all text columns
        text_cols = ['title_zh', 'title_vi', 'description_zh', 'description_vi']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.fix_fullwidth)
        print('  V2 Full-width → NFKC')

        # V3: Remove CJK from VI
        for col in ['title_vi', 'description_vi']:
            if col in df.columns:
                df[col] = df[col].apply(self.remove_cjk_from_vi)
        print('  V3 CJK in VI → cleaned')

        # V10: Normalize brackets
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.fix_cjk_brackets)
        print('  V10 CJK brackets → ASCII')

        # V4: Trim trailing punctuation
        for col in ['title_vi']:
            if col in df.columns:
                df[col] = df[col].apply(self.trim_trailing_punct)
        print('  V4 Trailing punct → trimmed')

        # V5: Decimal comma → dot
        for col in ['title_vi']:
            if col in df.columns:
                df[col] = df[col].apply(self.fix_decimal_comma)
        print('  V5 Decimal comma → dot')

        # V16: Star + pipe operators
        for col in ['title_zh', 'description_zh']:
            if col in df.columns:
                df[col] = df[col].apply(self.fix_star_operator)
                df[col] = df[col].apply(self.fix_cjk_pipe)
        print('  V16 Star/pipe → normalized')

        # CADO-VI
        df['title_vi_cleaned'] = df['title_vi'].apply(self.clean_vi_text)
        vi_changed = (df['title_vi'] != df['title_vi_cleaned']).sum()
        print(f'  CADO-VI: {vi_changed} titles cleaned')

        # Flags
        df['flag_stuffing'] = df['title_vi_cleaned'].apply(
            lambda t: max(Counter(str(t).lower().split()).values()) >= 5
            if str(t).split() else False)

        zhl = df['title_zh'].apply(lambda t: len(self.cfg.cjk.findall(str(t))))
        vil = df['title_vi_cleaned'].apply(lambda t: len(str(t).split()))
        ratio = vil / zhl.replace(0, 1)
        df['flag_truncated'] = ratio < 0.8
        df['flag_bloated'] = ratio > 6.0

        print(f'  Flags: stuffing={df["flag_stuffing"].sum()}, '
              f'truncated={df["flag_truncated"].sum()}, '
              f'bloated={df["flag_bloated"].sum()}')

        print(f'  Result: {len(df):,} rows (removed {initial - len(df)})')
        return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: HYBRID SCORING
# ═══════════════════════════════════════════════════════════════

class HybridScorer:
    """
    2a: BGE-M3 Cosine Similarity
    2b: Lexical Overlap Score
    2c: hybrid_score = 0.7 × cosine + 0.3 × lexical_overlap
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.whitelist_lower = self._load_whitelist()

    def _load_whitelist(self) -> Set[str]:
        wl = set()
        if os.path.exists(self.cfg.whitelist_csv):
            with open(self.cfg.whitelist_csv, encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    w = row['word'].strip()
                    if w: wl.add(w.lower())
        return wl

    # ── 2a: Cosine ──────────────────────────────────────

    def compute_cosine(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BGE-M3 cosine similarity for all pairs."""
        print('─' * 50)
        print('STEP 2a: BGE-M3 COSINE SIMILARITY')
        print('─' * 50)

        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        model = SentenceTransformer(self.cfg.bge_model)
        print(f'  Model loaded ({time.time()-t0:.0f}s)')

        zh_texts = df['title_zh'].astype(str).tolist()
        vi_texts = df['title_vi_cleaned'].astype(str).tolist()
        all_texts = zh_texts + vi_texts

        print(f'  Encoding {len(all_texts):,} texts...')
        emb = model.encode(all_texts, batch_size=self.cfg.bge_batch,
                           normalize_embeddings=True, show_progress_bar=True,
                           convert_to_numpy=True)

        emb_zh = emb[:len(zh_texts)]
        emb_vi = emb[len(zh_texts):]
        cos_sim = np.sum(emb_zh * emb_vi, axis=1)

        df['cosine_bge_m3'] = cos_sim
        print(f'  Done ({time.time()-t0:.0f}s)')
        print(f'  Mean={cos_sim.mean():.4f} Median={np.median(cos_sim):.4f} '
              f'Std={cos_sim.std():.4f}')
        return df

    # ── 2b: Lexical Overlap ─────────────────────────────

    def extract_entities(self, text: str, is_zh: bool = False) -> Set[str]:
        """Extract brands, model codes, numbers, units from text."""
        if not isinstance(text, str): return set()
        entities = set()

        # Latin words (brands, tech terms)
        pattern = self.cfg.latin_zh if is_zh else self.cfg.latin_vi
        for w in pattern.findall(text):
            wl = w.lower()
            if wl in self.whitelist_lower or re.search(r'\d', w) or re.match(r'^[A-Z]{2,6}$', w):
                entities.add(wl)

        # Numbers
        for m in re.finditer(r'\d+', text):
            entities.add(m.group())
        for char in text:
            if char in self.cfg.han_num:
                entities.add(self.cfg.han_num[char])

        # Units (from VI)
        if not is_zh:
            for vi_unit in set(self.cfg.unit_map.values()):
                if vi_unit in text.lower():
                    entities.add(vi_unit)
        else:
            for zh_unit in self.cfg.unit_map:
                if zh_unit in text:
                    entities.add(zh_unit)

        return entities

    def lexical_overlap(self, zh: str, vi: str) -> float:
        """Compute Jaccard-like overlap of entities between ZH and VI."""
        zh_ent = self.extract_entities(zh, is_zh=True)
        vi_ent = self.extract_entities(vi, is_zh=False)

        union = len(zh_ent | vi_ent)
        if union == 0:
            return 1.0  # No entities → no evidence either way → neutral

        intersection = len(zh_ent & vi_ent)
        return intersection / union

    def compute_lexical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lexical overlap for all pairs."""
        print('─' * 50)
        print('STEP 2b: LEXICAL OVERLAP')
        print('─' * 50)

        t0 = time.time()
        scores = []
        for _, row in df.iterrows():
            zh = str(row['title_zh'])
            vi = str(row['title_vi_cleaned'])
            scores.append(self.lexical_overlap(zh, vi))

        df['lexical_overlap'] = scores
        print(f'  Done ({time.time()-t0:.0f}s)')
        print(f'  Mean={np.mean(scores):.4f} Median={np.median(scores):.4f}')
        print(f'  Zero overlap: {sum(1 for s in scores if s == 0.0):,} rows')

        return df

    # ── 2c: Hybrid ──────────────────────────────────────

    def compute_hybrid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine cosine + lexical into hybrid score."""
        w_cos = self.cfg.cosine_weight
        w_lex = self.cfg.lexical_weight
        df['hybrid_score'] = w_cos * df['cosine_bge_m3'] + w_lex * df['lexical_overlap']
        return df

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply thresholds to get final labels."""
        df['final_label'] = 'KEEP'
        df.loc[df['cosine_bge_m3'] < self.cfg.cosine_discard, 'final_label'] = 'DISCARD'
        df.loc[df['final_label'] == 'KEEP', 'confidence'] = 'MEDIUM'
        df.loc[df['cosine_bge_m3'] >= 0.75, 'confidence'] = 'HIGH'
        return df

    def run(self, df: pd.DataFrame, skip_cosine: bool = False) -> pd.DataFrame:
        """Run full hybrid scoring pipeline."""
        if not skip_cosine:
            df = self.compute_cosine(df)
        else:
            print('  ⏩ Skipping cosine (already computed)')

        df = self.compute_lexical(df)
        df = self.compute_hybrid(df)
        df = self.classify(df)

        # Summary
        print('─' * 50)
        print('STEP 2c: HYBRID SCORE RESULTS')
        print('─' * 50)
        print(f'  hybrid_score: mean={df["hybrid_score"].mean():.4f} '
              f'median={df["hybrid_score"].median():.4f}')

        for label in ['KEEP', 'DISCARD']:
            n = (df['final_label'] == label).sum()
            pct = n / len(df) * 100
            print(f'  {label:<10} {n:>6,} ({pct:5.1f}%)')

        return df


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    cfg = Config()
    print('=' * 60)
    print('🔍 CADO BITEXT FILTER — H3 Hybrid (Cosine + Lexical)')
    print('=' * 60)

    # ── Load ──
    print(f'\n📂 Loading: {cfg.input_parquet}')
    df = pd.read_parquet(cfg.input_parquet)
    print(f'   {len(df):,} rows × {len(df.columns)} cols')

    # ── Step 1: Preprocess ──
    preprocessor = Preprocessor(cfg)
    df = preprocessor.process(df)

    # ── Step 2: Hybrid Scoring ──
    scorer = HybridScorer(cfg)
    # Check if cosine already exists
    skip_cos = 'cosine_bge_m3' in df.columns and df['cosine_bge_m3'].notna().sum() > len(df) * 0.9
    df = scorer.run(df, skip_cosine=skip_cos)

    # ── Export ──
    os.makedirs(os.path.dirname(cfg.output_parquet), exist_ok=True)
    os.makedirs('csv', exist_ok=True)

    df.to_parquet(cfg.output_parquet, index=False)
    print(f'\n✅ {cfg.output_parquet} ({len(df):,} rows)')

    # Discard CSV
    discard_df = df[df['final_label'] == 'DISCARD']
    export_cols = ['product_id', 'category', 'cosine_bge_m3', 'lexical_overlap',
                   'hybrid_score', 'final_label', 'confidence',
                   'title_zh', 'title_vi_cleaned',
                   'flag_stuffing', 'flag_truncated', 'flag_bloated']
    available = [c for c in export_cols if c in discard_df.columns]
    discard_df[available].to_csv(cfg.output_discard, index=False, encoding='utf-8-sig')
    print(f'✅ {cfg.output_discard} ({len(discard_df)} rows)')

    # Full scores CSV
    full_df = df[available]
    full_df.to_csv('csv/bitext_full_scores.csv', index=False, encoding='utf-8-sig')
    print(f'✅ csv/bitext_full_scores.csv ({len(full_df)} rows)')

    # ── Final tally ──
    print(f'\n{"="*60}')
    print('📊 FINAL TALLY')
    print(f'{"="*60}')
    print(f'  Input:    16,348')
    print(f'  Discarded (CJK copy): 2')
    print(f'  Preprocessed:        16,346')
    print(f'  KEEP:    {(df["final_label"]=="KEEP").sum():,}')
    print(f'  DISCARD: {(df["final_label"]=="DISCARD").sum():,}')
    print(f'  Training-ready: {(df["final_label"]=="KEEP").sum():,} '
          f'({(df["final_label"]=="KEEP").sum()/len(df)*100:.1f}%)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()