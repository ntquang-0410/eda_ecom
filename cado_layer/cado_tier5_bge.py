#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO Pipeline — Tầng 5: bge-m3 Semantic Fallback
===================================================
Dùng BAAI/bge-m3 embedding để kiểm tra: nếu xóa từ tiếng Anh khỏi câu
mà embedding của câu không thay đổi đáng kể (cosine sim > 0.98),
thì từ đó thực sự dư thừa → XÓA.

Chỉ áp dụng cho từ đã qua T0-T4 vẫn là "no_cedict_entry":
  - Những từ không có trong CC-CEDICT
  - Không phải brand/loanword đã được Tầng 1,2,3 bảo vệ

Output: 50 dòng mẫu, embedding similarity, quyết định cuối cùng.
"""

import os, re, csv, time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

RAW_PARQUET = "data/raw/bilingual_zh_vi.parquet"
WHITELIST_CSV = "csv/whitelist.csv"
CEDICT_FILE = "cedict_ts.u8"
LATIN_CHARS = re.compile(r"[a-zA-Z\u00C0-\u024F]+")
CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")

# Tầng 5 tham số
BGE_SIMILARITY_THRESHOLD = 0.98   # Dưới threshold này → embedding thay đổi → GIỮ
BGE_MODEL = "BAAI/bge-m3"
BGE_BATCH_SIZE = 64
MAX_TIER5_SAMPLES = 500           # Giới hạn số cặp (word, text) cần embed


# ═══════════════════════════════════════════════════════════════════
# TẦNG 1-4 (giữ nguyên từ script trước)
# ═══════════════════════════════════════════════════════════════════

def load_whitelist(path: str) -> set[str]:
    whitelist = set()
    if not os.path.exists(path): return whitelist
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            word = row["word"].strip()
            if word: whitelist.add(word.lower())
    return whitelist


def is_pattern_keep(word: str) -> str | None:
    if not word: return None
    if re.search(r"\d", word):           return "model_code"
    if re.match(r"^[A-Z]{1,5}$", word):  return "short_abbreviation"
    if re.match(r"^[A-Z][a-z]+[A-Z]", word): return "camelcase_brand"
    if len(word) <= 2:                   return "too_short"
    return None


def parse_cedict(path: str) -> dict[str, list[str]]:
    eng_to_chn = defaultdict(list)
    entry_pat = re.compile(r"^(\S+)\s+\S+\s+\[.+?\]\s+/(.+)/$")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            m = entry_pat.match(line)
            if not m: continue
            trad = m.group(1)
            if len(CJK_CHAR.findall(trad)) < 2: continue
            gloss = m.group(2).lower()
            for token in set(re.findall(r"[a-z]+", gloss)):
                if len(token) >= 2:
                    if trad not in eng_to_chn[token]:
                        eng_to_chn[token].append(trad)
    for k in eng_to_chn:
        eng_to_chn[k] = sorted(set(eng_to_chn[k]), key=len, reverse=True)
    return dict(eng_to_chn)


def lemmatize(word: str) -> list[str]:
    w = word.lower()
    forms = [w]
    if w.endswith("sses"):   forms.append(w[:-2])
    elif w.endswith("ies"):  forms.append(w[:-3] + "y")
    elif w.endswith("ves"):  forms.append(w[:-3] + "f")
    elif w.endswith("es"):   forms.append(w[:-2])
    elif w.endswith("s") and not w.endswith("ss"): forms.append(w[:-1])
    if w.endswith("ing"):    forms.append(w[:-3]); forms.append(w[:-3] + "e")
    if w.endswith("ed"):     forms.append(w[:-2]); forms.append(w[:-1])
    return forms


def check_overlap(word: str, text: str, cedict: dict) -> tuple[bool, str]:
    for lemma in lemmatize(word):
        if lemma not in cedict: continue
        for trans in cedict[lemma]:
            if trans in text: return (True, trans)
            if " " in trans and all(p in text for p in trans.split()):
                return (True, f"fuzzy:{trans}")
    return (False, "no_entry" if word.lower() not in cedict else "no_overlap")


def build_candidate_pool(df_en, whitelist_lower):
    pool = defaultdict(lambda: {"word": "", "contexts": [], "total_count": 0})
    for _, row in df_en.iterrows():
        title = str(row["title_zh"]); pid = str(row["product_id"])
        for w in LATIN_CHARS.findall(title):
            if w.lower() in whitelist_lower: continue
            if is_pattern_keep(w): continue
            pool[w]["word"] = w
            pool[w]["total_count"] += 1
            pool[w]["contexts"].append({"pid": pid, "text": title})
    return list(pool.values())


def detect_loanwords(candidates, cedict, min_freq=3, threshold=0.20):
    loanwords = set(); report = []
    for cand in candidates:
        w, count, contexts = cand["word"], cand["total_count"], cand["contexts"]
        if count < min_freq:
            report.append({"word": w, "total_count": count, "overlap_count": 0,
                           "overlap_rate": 0, "decision": "insufficient_data", "sample_match": ""})
            continue
        overlap_count = 0; sample_match = ""
        for ctx in contexts:
            has, matched = check_overlap(w, ctx["text"], cedict)
            if has: overlap_count += 1; sample_match = sample_match or matched
        rate = overlap_count / count
        decision = "loanword_keep" if rate < threshold else "not_loanword"
        if decision == "loanword_keep": loanwords.add(w.lower())
        report.append({"word": w, "total_count": count, "overlap_count": overlap_count,
                       "overlap_rate": round(rate, 3), "decision": decision, "sample_match": sample_match})
    return loanwords, report


def cleanup_text(text: str) -> str:
    """Chuẩn hóa text để embedding: xóa toàn bộ từ Latin, chỉ giữ CJK."""
    cleaned = LATIN_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# ═══════════════════════════════════════════════════════════════════
# TẦNG 5: BGE-M3 SEMANTIC FALLBACK
# ═══════════════════════════════════════════════════════════════════

def run_tier5(
    candidates: list[dict],   # list of {pid, title, word, reason}
    threshold: float = BGE_SIMILARITY_THRESHOLD,
) -> tuple[set[str], list[dict]]:
    """
    Với mỗi cặp (word, title) có reason="no_cedict_entry":
      1. Tạo 2 phiên bản text: giữ word và xóa word
      2. Embed cả 2 → cosine similarity
      3. Nếu similarity > threshold → word không thay đổi ý nghĩa → XÓA
    Returns: (words_to_remove_tier5, report)
    """
    if not candidates:
        print("   (không có ứng viên cho Tầng 5)")
        return set(), []

    # Load model
    print(f"   Đang load {BGE_MODEL}...")
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("   ❌ sentence-transformers chưa cài. Bỏ qua Tầng 5.")
        return set(), []

    model = SentenceTransformer(BGE_MODEL, device="cpu")
    print(f"   Load xong ({time.time()-t0:.1f}s)")

    # Chuẩn bị pairs
    pairs_keep = []   # text có giữ word
    pairs_remove = [] # text đã xóa word
    pair_meta = []    # metadata

    for cand in candidates[:MAX_TIER5_SAMPLES]:
        title = cand["title"]
        word = cand["word"]
        pid = cand["pid"]

        # Text giữ word (nguyên bản)
        text_keep = title

        # Text xóa word (chỉ giữ phần CJK + từ khác)
        esc = re.escape(word)
        pattern = re.compile(
            r"(^|\s|[\u4e00-\u9fff])" + esc + r"($|\s|[\u4e00-\u9fff])",
            re.IGNORECASE)
        text_remove = pattern.sub(r"\1\2", title)
        text_remove = re.sub(r"\s+", " ", text_remove).strip()

        # Nếu sau khi xóa không còn gì khác → skip
        if text_remove == text_keep:
            continue

        pairs_keep.append(text_keep)
        pairs_remove.append(text_remove)
        pair_meta.append({"pid": pid, "word": word, "title": title,
                          "title_no_word": text_remove})

    if not pairs_keep:
        print("   (không có cặp hợp lệ)")
        return set(), []

    print(f"   Đang embed {len(pairs_keep)} cặp text...")
    t0 = time.time()

    # Batch encode
    emb_keep = model.encode(pairs_keep, batch_size=BGE_BATCH_SIZE,
                            normalize_embeddings=True, show_progress_bar=True)
    emb_remove = model.encode(pairs_remove, batch_size=BGE_BATCH_SIZE,
                              normalize_embeddings=True, show_progress_bar=True)
    print(f"   Embed xong ({time.time()-t0:.1f}s)")

    # Cosine similarity (đã normalize → dot product)
    similarities = np.sum(emb_keep * emb_remove, axis=1)

    # Quyết định
    words_to_remove = set()
    report = []

    for i, sim in enumerate(similarities):
        meta = pair_meta[i]
        should_remove = bool(sim > threshold)
        if should_remove:
            words_to_remove.add(meta["word"].lower())

        report.append({
            **meta,
            "similarity": round(float(sim), 4),
            "decision": "remove" if should_remove else "keep",
        })

    return words_to_remove, report


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔧 CADO TẦNG 5: BGE-M3 SEMANTIC FALLBACK")
    print("=" * 70)

    # --- Load ---
    whitelist_lower = load_whitelist(WHITELIST_CSV)
    print(f"\n🛡️ Whitelist: {len(whitelist_lower)} từ")
    print(f"📚 Parse CC-CEDICT...")
    cedict = parse_cedict(CEDICT_FILE)
    print(f"   Entries: {len(cedict):,}")

    df = pd.read_parquet(RAW_PARQUET)
    has_en = df["title_zh"].apply(
        lambda t: bool(LATIN_CHARS.search(str(t))) if isinstance(t, str) else False)
    df_en = df[has_en].copy()

    # --- Tầng 3 ---
    candidates = build_candidate_pool(df_en, whitelist_lower)
    loanwords_lower, _ = detect_loanwords(candidates, cedict)

    # --- Tầng 4: Phân loại, thu thập no_cedict_entry ---
    print(f"\n🔍 Tầng 4: Phân loại + thu thập ứng viên Tầng 5...")
    tier5_candidates = []
    results_all = []
    stats = Counter()

    for _, row in df_en.iterrows():
        title = str(row["title_zh"]); pid = str(row["product_id"])
        words = LATIN_CHARS.findall(title)
        word_decisions = []; remove_set = set()

        for w in words:
            # T1
            if w.lower() in whitelist_lower:
                word_decisions.append({"word": w, "action": "keep", "reason": "whitelist"})
                stats["keep"] += 1; continue
            # T2
            pat = is_pattern_keep(w)
            if pat:
                word_decisions.append({"word": w, "action": "keep", "reason": f"pattern:{pat}"})
                stats["keep"] += 1; continue
            # T3
            if w.lower() in loanwords_lower:
                word_decisions.append({"word": w, "action": "keep", "reason": "auto_loanword"})
                stats["keep"] += 1; continue
            # T4
            overlap, match = check_overlap(w, title, cedict)
            if overlap:
                word_decisions.append({"word": w, "action": "remove", "reason": f"cedict:{match}"})
                stats["remove"] += 1; remove_set.add(w)
            elif match == "no_entry":
                word_decisions.append({"word": w, "action": "pending_t5", "reason": "no_cedict_entry"})
                stats["pending"] += 1
                tier5_candidates.append({"pid": pid, "title": title, "word": w})
            else:
                word_decisions.append({"word": w, "action": "keep", "reason": "cedict_no_overlap"})
                stats["keep"] += 1

        results_all.append({"pid": pid, "title": title, "words": word_decisions, "remove_set": remove_set})

    print(f"   Ứng viên Tầng 5: {len(tier5_candidates)} cặp (word, text)")
    unique_t5_words = len(set(c["word"] for c in tier5_candidates))
    print(f"   Unique words: {unique_t5_words}")

    # --- Tầng 5: bge-m3 ---
    print(f"\n🧠 TẦNG 5: BGE-M3 Semantic Fallback...")
    t5_remove_words, t5_report = run_tier5(tier5_candidates)

    # --- Áp dụng T5 ---
    t5_remove_count = 0
    for r in results_all:
        for wd in r["words"]:
            if wd["action"] == "pending_t5" and wd["word"].lower() in t5_remove_words:
                wd["action"] = "remove"
                wd["reason"] = "tier5_bge_m3"
                r["remove_set"].add(wd["word"])
                stats["t5_remove"] += 1
                stats["pending"] -= 1
                t5_remove_count += 1

    print(f"   Từ bị xóa bởi Tầng 5: {t5_remove_count}")

    # --- Thống kê cuối ---
    total = stats["keep"] + stats["remove"] + stats["pending"] + stats.get("t5_remove", 0)
    print(f"\n📊 TỔNG KẾT T0-T5:")
    print(f"   ✅ Keep:         {stats['keep']:,} ({stats['keep']/total*100:.1f}%)")
    print(f"   🔴 Remove (T4):  {stats['remove']:,} ({stats['remove']/total*100:.1f}%)")
    print(f"   🧠 Remove (T5):  {stats.get('t5_remove',0):,} ({stats.get('t5_remove',0)/total*100:.1f}%)")
    if stats["pending"] > 0:
        print(f"   🟡 Pending:       {stats['pending']:,} ({stats['pending']/total*100:.1f}%)")

    # --- 50 dòng T5 ---
    if t5_report:
        t5_remove_report = [r for r in t5_report if r["decision"] == "remove"]
        t5_keep_report = [r for r in t5_report if r["decision"] == "keep"]
        sample_t5 = (t5_remove_report + t5_keep_report)[:50]

        print(f"\n{'=' * 105}")
        print(f"📋 KẾT QUẢ TẦNG 5: SIMILARITY SCORES (threshold={BGE_SIMILARITY_THRESHOLD})")
        print(f"{'=' * 105}")
        print(f"   Remove: {len(t5_remove_report)} | Keep: {len(t5_keep_report)}")

        for i, r in enumerate(sample_t5, 1):
            tag = "🔴 XÓA" if r["decision"] == "remove" else "✅ GIỮ"
            print(f"\n── #{i} | {r['pid'][:14]} | sim={r['similarity']:.4f} {tag} ──")
            print(f"   WORD:     {r['word']}")
            print(f"   ORIGINAL: {r['title'][:110]}")
            if r["title_no_word"] != r["title"]:
                print(f"   NO_WORD:  {r['title_no_word'][:110]}")

    # --- Tổng kết ---
    print(f"\n{'─' * 70}")
    print(f"📊 HOÀN THIỆN CADO (T0-T5)")
    print(f"   Giữ:  {stats['keep'] + stats['pending']:,}")
    print(f"   Xóa:  {stats['remove'] + stats.get('t5_remove',0):,}")
    print(f"   Tầng 5 xóa thêm: {stats.get('t5_remove',0)}")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()