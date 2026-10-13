#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO Pipeline — Tầng 2: Pattern Detection
===========================================
Tích hợp Tầng 0 + Tầng 1 + Tầng 2:
  - Tầng 0: Trích xuất từ tiếng Anh (CJK-Latin boundary regex)
  - Tầng 1: Static Whitelist (đọc từ whitelist.csv)
  - Tầng 2: Pattern Detection (chứa số, ALL-CAPS, CamelCase, quá ngắn)

Output: 50 dòng title_zh với từ đã được phân loại keep/remove và
        demo title đã làm sạch.
"""

import os, re, csv
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

RAW_PARQUET = "data/raw/bilingual_zh_vi.parquet"
WHITELIST_CSV = "csv/whitelist.csv"
LATIN_CHARS = re.compile(r"[a-zA-Z\u00C0-\u024F]+")


# ═══════════════════════════════════════════════════════════════════
# TẦNG 1: LOAD WHITELIST
# ═══════════════════════════════════════════════════════════════════

def load_whitelist(path: str) -> set[str]:
    """Đọc whitelist.csv, trả về set từ (lowercase để so khớp không phân biệt)."""
    whitelist = set()
    if not os.path.exists(path):
        print(f"⚠️ Không tìm thấy {path}, dùng whitelist rỗng.")
        return whitelist
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            word = row["word"].strip()
            if word:
                whitelist.add(word.lower())
    return whitelist


# ═══════════════════════════════════════════════════════════════════
# TẦNG 2: PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════

def is_pattern_keep(word: str) -> str | None:
    """
    Kiểm tra từ có thuộc pattern nên giữ không.
    Trả về tên pattern nếu khớp, None nếu không.
    """
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
# CLASSIFY: Quyết định keep/remove
# ═══════════════════════════════════════════════════════════════════

def classify_word(word: str, whitelist_lower: set[str]) -> tuple[str, str]:
    """
    Phân loại từ: keep hay remove (và lý do).
    Sử dụng Tầng 1 (whitelist) + Tầng 2 (pattern).
    """
    if not word or not isinstance(word, str):
        return ("keep", "empty")

    # Tầng 1: Whitelist
    if word.lower() in whitelist_lower:
        return ("keep", "whitelist")

    # Tầng 2: Pattern
    pattern = is_pattern_keep(word)
    if pattern:
        return ("keep", f"pattern:{pattern}")

    # Chưa quyết định → pending (sẽ xử lý ở Tầng 4)
    return ("pending", "no_decision_yet")


# ═══════════════════════════════════════════════════════════════════
# CLEAN: Xóa từ khỏi text với CJK-Latin boundary đúng
# ═══════════════════════════════════════════════════════════════════

def clean_text(text: str, words_to_remove: set[str]) -> str:
    """Xóa danh sách từ tiếng Anh khỏi text, dùng regex CJK-Latin boundary."""
    if not isinstance(text, str) or not words_to_remove:
        return text

    cleaned = text
    # Sắp xếp theo độ dài giảm dần → xóa từ dài trước, tránh substring conflict
    for word in sorted(words_to_remove, key=len, reverse=True):
        esc = re.escape(word)
        # Pattern: (^|space|CJK) + word + ($|space|CJK)
        pattern = re.compile(
            r"(^|\s|[\u4e00-\u9fff])" + esc + r"($|\s|[\u4e00-\u9fff])",
            re.IGNORECASE,
        )
        cleaned = pattern.sub(r"\1\2", cleaned)

    # Dọn khoảng trắng thừa
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔧 CADO TẦNG 2: PATTERN DETECTION (tích hợp T0+T1+T2)")
    print("=" * 70)

    # Load
    whitelist_lower = load_whitelist(WHITELIST_CSV)
    print(f"\n🛡️ Whitelist: {len(whitelist_lower)} từ")

    df = pd.read_parquet(RAW_PARQUET)
    print(f"📂 Sản phẩm: {len(df):,}")

    # Lọc sản phẩm có EN trong title
    has_en = df["title_zh"].apply(
        lambda t: bool(LATIN_CHARS.search(str(t))) if isinstance(t, str) else False
    )
    df_en = df[has_en].copy()
    print(f"📋 Có EN trong title: {len(df_en):,} ({len(df_en)/len(df)*100:.1f}%)")

    # Phân loại từng từ trong mỗi title
    results = []
    stats = {"keep": 0, "pending": 0, "remove": 0, "total_words": 0}

    for _, row in df_en.iterrows():
        title = str(row["title_zh"])
        pid = str(row["product_id"])
        words = LATIN_CHARS.findall(title)

        word_decisions = []
        remove_set = set()

        for w in words:
            action, reason = classify_word(w, whitelist_lower)
            stats["total_words"] += 1

            if action == "keep":
                stats["keep"] += 1
            elif action == "remove":
                stats["remove"] += 1
                remove_set.add(w)
            else:
                stats["pending"] += 1

            word_decisions.append({"word": w, "action": action, "reason": reason})

        # Clean title
        cleaned_title = clean_text(title, remove_set)

        results.append({
            "product_id": pid,
            "title_zh": title,
            "words": word_decisions,
            "remove_words": remove_set,
            "title_cleaned": cleaned_title,
        })

    # In thống kê
    print(f"\n📊 PHÂN LOẠI SAU TẦNG 0+1+2:")
    print(f"   Tổng từ: {stats['total_words']:,}")
    print(f"   ✅ Keep (whitelist):     {stats['keep']:,} ({stats['keep']/stats['total_words']*100:.1f}%)")
    print(f"   🟡 Pending (Tầng 4):     {stats['pending']:,} ({stats['pending']/stats['total_words']*100:.1f}%)")
    print(f"   🔴 Remove (sẽ xóa):      {stats['remove']:,} ({stats['remove']/stats['total_words']*100:.1f}%)")

    # 50 dòng mẫu: chỉ lấy những dòng có ít nhất 1 từ bị pending hoặc remove
    # (những dòng thú vị — không phải toàn whitelist)
    interesting = [r for r in results if r["remove_words"] or any(
        d["action"] == "pending" for d in r["words"]
    )]
    sample = interesting[:50]

    print(f"\n{'=' * 100}")
    print(f"📋 50 DÒNG CÓ TỪ BỊ ẢNH HƯỞNG (PENDING hoặc REMOVE)")
    print(f"{'=' * 100}")

    for i, r in enumerate(sample, 1):
        pid = r["product_id"][:14]
        title = r["title_zh"]
        words_info = r["words"]

        # Format: từ + tag
        word_tags = []
        for wd in words_info:
            if wd["action"] == "keep":
                word_tags.append(f"{wd['word']}[K]")
            elif wd["action"] == "pending":
                word_tags.append(f"{wd['word']}[?]")
            else:
                word_tags.append(f"{wd['word']}[X]")

        print(f"\n── #{i} | ID: {pid} ──")
        print(f"   BEFORE: {title[:110]}")
        if r["remove_words"]:
            print(f"   REMOVE: {', '.join(sorted(r['remove_words']))}")
        print(f"   WORDS:  {' | '.join(word_tags[:10])}")
        if len(word_tags) > 10:
            print(f"           ... +{len(word_tags)-10} từ nữa")
        if r["remove_words"] and r["title_cleaned"] != title:
            print(f"   AFTER:  {r['title_cleaned'][:110]}")

    print(f"\n{'─' * 70}")
    print(f"📊 TỔNG KẾT TẦNG 0-2")
    print(f"   Sản phẩm có EN trong title: {len(df_en):,}")
    print(f"   Từ được whitelist giữ:      {stats['keep']:,}")
    print(f"   Từ pending (chờ Tầng 4):    {stats['pending']:,}")
    print(f"   Sẵn sàng cho Tầng 4: CEDICT Context Overlap")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()