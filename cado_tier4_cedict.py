#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO Pipeline — Tầng 4: CEDICT Context Overlap
=================================================
Tích hợp Tầng 0-1-2-4:
  - T0: Trích xuất từ tiếng Anh (CJK-Latin regex)
  - T1: Static Whitelist (whitelist.csv)
  - T2: Pattern Detection
  - T4: CEDICT Context Overlap — kiểm tra bản dịch TQ có xuất hiện trong câu gốc

Chỉ XÓA khi: bản dịch TQ (≥2 ký tự Hán) xuất hiện dưới dạng exact substring
              trong chính văn bản gốc.

Output: 50 dòng mẫu với kết quả keep/remove cuối cùng.
"""

import os, re, csv
import pandas as pd
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

RAW_PARQUET = "data/raw/bilingual_zh_vi.parquet"
WHITELIST_CSV = "csv/whitelist.csv"
CEDICT_FILE = "cedict_ts.u8"
LATIN_CHARS = re.compile(r"[a-zA-Z\u00C0-\u024F]+")
CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


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
    if not word:
        return None
    if re.search(r"\d", word):
        return "model_code"
    if re.match(r"^[A-Z]{1,5}$", word):
        return "short_abbreviation"
    if re.match(r"^[A-Z][a-z]+[A-Z][a-zA-Z]*$", word):
        return "camelcase_brand"
    if len(word) <= 2:
        return "too_short"
    return None


# ═══════════════════════════════════════════════════════════════════
# TẦNG 4: CEDICT
# ═══════════════════════════════════════════════════════════════════

def parse_cedict(path: str) -> dict[str, list[str]]:
    """
    Parse CC-CEDICT → dict: lowercase English word → list of Chinese translations.
    Chỉ lấy bản dịch có ≥2 ký tự Hán.
    """
    english_to_chinese = defaultdict(list)
    entry_pattern = re.compile(r"^(\S+)\s+\S+\s+\[.+?\]\s+/(.+)/$")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = entry_pattern.match(line)
            if not m:
                continue

            traditional = m.group(1)
            english_gloss = m.group(2)

            # Kiểm tra traditional có ≥2 ký tự CJK
            cjk_count = len(CJK_CHAR.findall(traditional))
            if cjk_count < 2:
                continue

            # Tách English gloss thành các từ
            eng_words = set()
            for part in english_gloss.lower().split("/"):
                for token in re.findall(r"[a-z]+", part):
                    if len(token) >= 2:
                        eng_words.add(token)

            for ew in eng_words:
                if traditional not in english_to_chinese[ew]:
                    english_to_chinese[ew].append(traditional)

    # Sắp xếp theo độ dài giảm dần (ưu tiên bản dịch dài, cụ thể)
    for k in english_to_chinese:
        english_to_chinese[k] = sorted(set(english_to_chinese[k]), key=len, reverse=True)

    return dict(english_to_chinese)


def check_overlap(word: str, text: str, cedict: dict) -> tuple[bool, str]:
    """
    Kiểm tra: bản dịch TQ của word có xuất hiện trong text không?
    Returns: (overlap: bool, matched_translation: str)
    """
    word_lower = word.lower()
    if word_lower not in cedict:
        return (False, "no_cedict_entry")

    translations = cedict[word_lower]

    # Bước 1: Exact substring match (ưu tiên bản dịch dài)
    for trans in translations:
        if trans in text:
            return (True, trans)

    # Bước 2: Fallback fuzzy — bản dịch đa ký tự có khoảng trắng
    for trans in translations:
        if " " in trans:
            parts = trans.split()
            if all(p in text for p in parts):
                return (True, f"fuzzy:{trans}")

    return (False, "no_overlap")


# ═══════════════════════════════════════════════════════════════════
# CLASSIFY
# ═══════════════════════════════════════════════════════════════════

def classify_with_cedict(
    word: str,
    whitelist_lower: set[str],
    cedict: dict,
    source_text: str,
) -> tuple[str, str]:
    """Phân loại đầy đủ: T1 → T2 → T4."""
    if not word:
        return ("keep", "empty")
    if word.lower() in whitelist_lower:
        return ("keep", "whitelist")

    pattern = is_pattern_keep(word)
    if pattern:
        return ("keep", f"pattern:{pattern}")

    # Tầng 4: CEDICT
    overlap, match = check_overlap(word, source_text, cedict)
    if overlap:
        return ("remove", f"cedict_overlap:{match}")
    elif match == "no_cedict_entry":
        return ("keep", "no_cedict_entry")
    else:
        return ("keep", "cedict_no_overlap")


# ═══════════════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════════════

def clean_text(text: str, words_to_remove: set[str]) -> str:
    if not isinstance(text, str) or not words_to_remove:
        return text
    cleaned = text
    for word in sorted(words_to_remove, key=len, reverse=True):
        esc = re.escape(word)
        pattern = re.compile(
            r"(^|\s|[\u4e00-\u9fff])" + esc + r"($|\s|[\u4e00-\u9fff])",
            re.IGNORECASE,
        )
        cleaned = pattern.sub(r"\1\2", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔧 CADO TẦNG 4: CEDICT CONTEXT OVERLAP")
    print("=" * 70)

    # Load
    whitelist_lower = load_whitelist(WHITELIST_CSV)
    print(f"\n🛡️ Whitelist: {len(whitelist_lower)} từ")

    print(f"📚 Đang parse CC-CEDICT ({CEDICT_FILE})...")
    cedict = parse_cedict(CEDICT_FILE)
    print(f"   English entries: {len(cedict):,}")
    # Sample
    for w in ["bag", "shoes", "wholesale", "gps", "iphone", "oversize"]:
        trans = cedict.get(w, [])
        print(f"   {w:>12} → {trans[:3]}")

    df = pd.read_parquet(RAW_PARQUET)
    print(f"\n📂 Sản phẩm: {len(df):,}")

    has_en = df["title_zh"].apply(
        lambda t: bool(LATIN_CHARS.search(str(t))) if isinstance(t, str) else False
    )
    df_en = df[has_en].copy()

    # Phân loại
    results = []
    stats = {"keep": 0, "remove": 0, "total": 0}
    reason_counts = defaultdict(int)

    for _, row in df_en.iterrows():
        title = str(row["title_zh"])
        pid = str(row["product_id"])
        words = LATIN_CHARS.findall(title)

        word_decisions = []
        remove_set = set()

        for w in words:
            action, reason = classify_with_cedict(w, whitelist_lower, cedict, title)
            stats["total"] += 1
            if action == "keep":
                stats["keep"] += 1
            else:
                stats["remove"] += 1
                remove_set.add(w)
            reason_counts[reason] += 1
            word_decisions.append({"word": w, "action": action, "reason": reason})

        cleaned_title = clean_text(title, remove_set)

        results.append({
            "product_id": pid,
            "title_zh": title,
            "words": word_decisions,
            "remove_words": remove_set,
            "title_cleaned": cleaned_title,
        })

    # Thống kê
    print(f"\n📊 PHÂN LOẠI SAU TẦNG 0-4:")
    print(f"   Tổng từ:    {stats['total']:,}")
    print(f"   ✅ Keep:    {stats['keep']:,} ({stats['keep']/stats['total']*100:.1f}%)")
    print(f"   🔴 Remove:  {stats['remove']:,} ({stats['remove']/stats['total']*100:.1f}%)")
    print(f"\n   Lý do keep/remove:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"     {reason:<30} {count:>6,}")

    # 50 dòng mẫu có từ bị REMOVE
    with_removes = [r for r in results if r["remove_words"]]
    print(f"\n   Sản phẩm có từ bị xóa: {len(with_removes):,}")

    sample = with_removes[:50]

    print(f"\n{'=' * 100}")
    print(f"📋 50 DÒNG CÓ TỪ BỊ XÓA BỞI CEDICT OVERLAP")
    print(f"{'=' * 100}")

    for i, r in enumerate(sample, 1):
        pid = r["product_id"][:14]
        title = r["title_zh"]
        words_info = r["words"]

        word_tags = []
        for wd in words_info:
            if wd["action"] == "remove":
                word_tags.append(f"{wd['word']}[✕:{wd['reason'].split(':')[-1][:20]}]")
            elif wd["action"] == "keep":
                tag = "K" if wd["reason"].startswith("pattern") else \
                      "W" if wd["reason"] == "whitelist" else \
                      "?"
                word_tags.append(f"{wd['word']}[{tag}]")

        print(f"\n── #{i} | ID: {pid} ──")
        print(f"   BEFORE: {title[:120]}")
        removed_display = []
        for wd in words_info:
            if wd["action"] == "remove":
                removed_display.append(f"{wd['word']}→{wd['reason'].split(':')[-1][:30]}")
        print(f"   REMOVE: {' | '.join(removed_display[:5])}")
        print(f"   AFTER:  {r['title_cleaned'][:120]}")

    print(f"\n{'─' * 70}")
    print(f"📊 TỔNG KẾT TẦNG 0-4")
    print(f"   Từ được giữ:  {stats['keep']:,}")
    print(f"   Từ bị xóa:    {stats['remove']:,}")
    print(f"   Sản phẩm có xóa: {len(with_removes):,}/{len(df_en):,}")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()