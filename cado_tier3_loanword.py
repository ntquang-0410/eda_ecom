#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO Pipeline — Tầng 3: Auto Loanword Detection
==================================================
Tích hợp đầy đủ T0-T1-T2-T3-T4:

  T0: Trích xuất từ Anh (CJK-Latin boundary regex)
  T1: Static Whitelist (whitelist.csv)
  T2: Pattern Detection (model_code, ALL-CAPS, CamelCase, too_short)
  T3: Auto Loanword — tự động phát hiện từ mượn qua tần suất & overlap rate
  T4: CEDICT Context Overlap — xóa nếu bản dịch TQ xuất hiện trong câu

Tầng 3 logic:
  - Đếm tần suất từ W trong toàn dataset
  - Với mỗi câu chứa W, kiểm tra overlap CEDICT
  - Nếu overlap_rate < threshold (20%): W là từ mượn → GIỮ
  - Nếu count(W) < N (3): không đủ dữ liệu → chuyển Tầng 4
"""

import os, re, csv
import pandas as pd
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

RAW_PARQUET = "data/raw/bilingual_zh_vi.parquet"
WHITELIST_CSV = "csv/whitelist.csv"
CEDICT_FILE = "cedict_ts.u8"
OUTPUT_CSV = "csv/cado_tier3_loanword_report.csv"
LATIN_CHARS = re.compile(r"[a-zA-Z\u00C0-\u024F]+")
CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")

# Tầng 3 tham số
MIN_FREQ = 3          # Chỉ phân tích từ xuất hiện ≥ N lần
OVERLAP_THRESHOLD = 0.20  # Dưới ngưỡng này → từ mượn


# ═══════════════════════════════════════════════════════════════════
# TẦNG 1: WHITELIST
# ═══════════════════════════════════════════════════════════════════

def load_whitelist(path: str) -> set[str]:
    whitelist = set()
    if not os.path.exists(path):
        return whitelist
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            word = row["word"].strip()
            if word:
                whitelist.add(word.lower())
    return whitelist


# ═══════════════════════════════════════════════════════════════════
# TẦNG 2: PATTERN
# ═══════════════════════════════════════════════════════════════════

def is_pattern_keep(word: str) -> str | None:
    if not word: return None
    if re.search(r"\d", word):           return "model_code"
    if re.match(r"^[A-Z]{1,5}$", word):  return "short_abbreviation"
    if re.match(r"^[A-Z][a-z]+[A-Z]", word): return "camelcase_brand"
    if len(word) <= 2:                   return "too_short"
    return None


# ═══════════════════════════════════════════════════════════════════
# TẦNG 4: CEDICT
# ═══════════════════════════════════════════════════════════════════

def parse_cedict(path: str) -> dict[str, list[str]]:
    """Parse CC-CEDICT → dict: lowercase English word → list of Chinese translations (≥2 CJK chars)."""
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
            # Trích xuất từ riêng lẻ trong gloss
            for token in set(re.findall(r"[a-z]+", gloss)):
                if len(token) >= 2:
                    if trad not in eng_to_chn[token]:
                        eng_to_chn[token].append(trad)
    for k in eng_to_chn:
        eng_to_chn[k] = sorted(set(eng_to_chn[k]), key=len, reverse=True)
    return dict(eng_to_chn)


def lemmatize(word: str) -> list[str]:
    """Sinh các dạng lemma đơn giản cho stemming fallback."""
    w = word.lower()
    forms = [w]
    # Số nhiều: bags → bag, dresses → dress, boxes → box
    if w.endswith("sses"):   forms.append(w[:-2])
    elif w.endswith("ies"):  forms.append(w[:-3] + "y")
    elif w.endswith("ves"):  forms.append(w[:-3] + "f")
    elif w.endswith("es"):   forms.append(w[:-2])
    elif w.endswith("s") and not w.endswith("ss"):
        forms.append(w[:-1])
    # Động từ: running → run, stopped → stop
    if w.endswith("ing"):    forms.append(w[:-3]); forms.append(w[:-3] + "e")
    if w.endswith("ed"):     forms.append(w[:-2]); forms.append(w[:-1])
    return forms


def check_overlap(word: str, text: str, cedict: dict) -> tuple[bool, str]:
    """Kiểm tra overlap: bản dịch TQ (≥2 ký tự) có trong text không?
    Dùng lemmatization fallback cho dạng số nhiều/động từ."""
    for lemma in lemmatize(word):
        if lemma not in cedict:
            continue
        for trans in cedict[lemma]:
            if trans in text:
                return (True, trans)
            if " " in trans and all(p in text for p in trans.split()):
                return (True, f"fuzzy:{trans}")
    return (False, "no_entry" if word.lower() not in cedict else "no_overlap")


# ═══════════════════════════════════════════════════════════════════
# TẦNG 3: AUTO LOANWORD DETECTION
# ═══════════════════════════════════════════════════════════════════

def build_candidate_pool(
    df_en: pd.DataFrame,
    whitelist_lower: set[str],
) -> list[dict]:
    """
    Gom tất cả từ chưa được T1+T2 quyết định, kèm context.
    Returns: list of {word, contexts: [(product_id, text), ...]}
    """
    pool = defaultdict(lambda: {"word": "", "contexts": [], "total_count": 0})

    for _, row in df_en.iterrows():
        title = str(row["title_zh"])
        pid = str(row["product_id"])
        words = LATIN_CHARS.findall(title)

        for w in words:
            # Bỏ qua nếu đã được T1 hoặc T2 giữ
            if w.lower() in whitelist_lower: continue
            if is_pattern_keep(w): continue

            pool[w]["word"] = w
            pool[w]["total_count"] += 1
            pool[w]["contexts"].append({"pid": pid, "text": title})

    return list(pool.values())


def detect_loanwords(
    candidates: list[dict],
    cedict: dict,
    min_freq: int = MIN_FREQ,
    threshold: float = OVERLAP_THRESHOLD,
) -> tuple[set[str], list[dict]]:
    """
    Với mỗi từ ứng viên, tính overlap_rate.
    Nếu count ≥ min_freq và overlap_rate < threshold → từ mượn → GIỮ.
    Returns: (loanwords_set, report_rows)
    """
    loanwords = set()
    report = []

    for cand in candidates:
        w = cand["word"]
        count = cand["total_count"]
        contexts = cand["contexts"]

        if count < min_freq:
            # Không đủ dữ liệu → chuyển Tầng 4
            report.append({
                "word": w, "total_count": count, "overlap_count": 0,
                "overlap_rate": 0, "decision": "insufficient_data",
                "sample_match": "",
            })
            continue

        # Đếm overlap
        overlap_count = 0
        sample_match = ""
        for ctx in contexts:
            has_overlap, matched = check_overlap(w, ctx["text"], cedict)
            if has_overlap:
                overlap_count += 1
                if not sample_match:
                    sample_match = matched

        overlap_rate = overlap_count / count

        if overlap_rate < threshold:
            loanwords.add(w.lower())
            decision = "loanword_keep"
        else:
            decision = "not_loanword"

        report.append({
            "word": w, "total_count": count, "overlap_count": overlap_count,
            "overlap_rate": round(overlap_rate, 3), "decision": decision,
            "sample_match": sample_match,
        })

    return loanwords, report


# ═══════════════════════════════════════════════════════════════════
# CLASSIFY FULL
# ═══════════════════════════════════════════════════════════════════

def classify_full(
    word: str, whitelist_lower: set[str], loanwords_lower: set[str],
    cedict: dict, source_text: str,
) -> tuple[str, str]:
    """T1 → T2 → T3 → T4."""
    if not word: return ("keep", "empty")

    # T1
    if word.lower() in whitelist_lower: return ("keep", "whitelist")

    # T2
    pat = is_pattern_keep(word)
    if pat: return ("keep", f"pattern:{pat}")

    # T3
    if word.lower() in loanwords_lower: return ("keep", "auto_loanword")

    # T4
    overlap, match = check_overlap(word, source_text, cedict)
    if overlap:     return ("remove", f"cedict:{match}")
    if match == "no_entry": return ("keep", "no_cedict_entry")
    return ("keep", "cedict_no_overlap")


# ═══════════════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════════════

def clean_text(text: str, words_to_remove: set[str]) -> str:
    if not isinstance(text, str) or not words_to_remove: return text
    cleaned = text
    for word in sorted(words_to_remove, key=len, reverse=True):
        esc = re.escape(word)
        pat = re.compile(
            r"(^|\s|[\u4e00-\u9fff])" + esc + r"($|\s|[\u4e00-\u9fff])",
            re.IGNORECASE)
        cleaned = pat.sub(r"\1\2", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔧 CADO TẦNG 3: AUTO LOANWORD DETECTION (T0-T4 tích hợp)")
    print("=" * 70)

    # --- Load ---
    whitelist_lower = load_whitelist(WHITELIST_CSV)
    print(f"\n🛡️ Whitelist: {len(whitelist_lower)} từ")

    print(f"📚 Parse CC-CEDICT...")
    cedict = parse_cedict(CEDICT_FILE)
    print(f"   Entries: {len(cedict):,}")

    df = pd.read_parquet(RAW_PARQUET)
    print(f"📂 Sản phẩm: {len(df):,}")

    has_en = df["title_zh"].apply(
        lambda t: bool(LATIN_CHARS.search(str(t))) if isinstance(t, str) else False)
    df_en = df[has_en].copy()
    print(f"   Có EN trong title: {len(df_en):,}")

    # --- Tầng 3: Phát hiện loanword ---
    print(f"\n🔄 TẦNG 3: Phát hiện từ mượn tự động...")
    candidates = build_candidate_pool(df_en, whitelist_lower)
    print(f"   Ứng viên (chưa T1+T2 quyết định): {len(candidates)} từ unique")

    loanwords_lower, loanword_report = detect_loanwords(candidates, cedict)
    print(f"   Phát hiện từ mượn: {len(loanwords_lower)} từ")

    # Top loanwords
    loanword_detected = [r for r in loanword_report if r["decision"] == "loanword_keep"]
    loanword_detected.sort(key=lambda x: -x["total_count"])
    print(f"\n   Top 30 từ mượn phát hiện được:")
    for r in loanword_detected[:30]:
        print(f"     {r['word']:<25} freq={r['total_count']:>4}  overlap={r['overlap_count']}/{r['total_count']} ({r['overlap_rate']:.1%})")

    # Lưu report
    os.makedirs("csv", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["word", "total_count", "overlap_count", "overlap_rate", "decision", "sample_match"])
        w.writeheader()
        w.writerows(sorted(loanword_report, key=lambda x: -x["total_count"]))
    print(f"\n💾 {OUTPUT_CSV}")

    # --- Phân loại toàn bộ ---
    print(f"\n🔍 Phân loại toàn bộ từ...")
    results = []
    stats = Counter()
    reason_counts = Counter()

    for _, row in df_en.iterrows():
        title = str(row["title_zh"])
        pid = str(row["product_id"])
        words = LATIN_CHARS.findall(title)

        word_decisions = []
        remove_set = set()
        for w in words:
            action, reason = classify_full(w, whitelist_lower, loanwords_lower, cedict, title)
            stats["total"] += 1
            stats[action] += 1
            reason_counts[reason] += 1
            if action == "remove":
                remove_set.add(w)
            word_decisions.append({"word": w, "action": action, "reason": reason})

        cleaned_title = clean_text(title, remove_set)
        results.append({
            "product_id": pid, "title_zh": title,
            "words": word_decisions, "remove_words": remove_set,
            "title_cleaned": cleaned_title,
        })

    # --- Thống kê ---
    print(f"\n📊 PHÂN LOẠI SAU T0-T4:")
    for action in ["keep", "remove"]:
        pct = stats[action] / stats["total"] * 100
        print(f"   {'✅ Keep' if action == 'keep' else '🔴 Remove'}: {stats[action]:,} ({pct:.1f}%)")

    print(f"\n   Lý do chi tiết:")
    for reason, count in reason_counts.most_common(20):
        print(f"     {reason:<32} {count:>6,}")

    # --- 50 dòng mẫu ---
    with_removes = [r for r in results if r["remove_words"]]
    sample = with_removes[:50]

    print(f"\n{'=' * 105}")
    print(f"📋 50 DÒNG CÓ TỪ BỊ XÓA (SAU T0-T4)")
    print(f"{'=' * 105}")

    for i, r in enumerate(sample, 1):
        pid = r["product_id"][:14]
        removed_info = []
        for wd in r["words"]:
            if wd["action"] == "remove":
                removed_info.append(f"{wd['word']}→{wd['reason'].replace('cedict:','')[:25]}")

        print(f"\n── #{i} | {pid} ──")
        print(f"   BEFORE: {r['title_zh'][:120]}")
        print(f"   REMOVE: {' | '.join(removed_info[:4])}")
        print(f"   AFTER:  {r['title_cleaned'][:120]}")

    # --- Tổng kết ---
    print(f"\n{'─' * 70}")
    print(f"📊 TỔNG KẾT CADO (T0-T4)")
    print(f"   Từ giữ:  {stats['keep']:,}")
    print(f"   Từ xóa:  {stats['remove']:,}")
    print(f"   SP có xóa: {len(with_removes):,}/{len(df_en):,}")
    print(f"   Từ mượn tự động: {len(loanwords_lower)}")
    print(f"   Whitelist: {len(whitelist_lower)}")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()