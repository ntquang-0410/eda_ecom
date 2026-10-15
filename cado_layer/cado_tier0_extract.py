#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADO — Tầng 0: Preprocess
==========================
1. Trích xuất từ tiếng Anh từ title_zh / description_zh với regex CJK-Latin đúng
2. Lưu kết quả ra CSV trung gian
3. In 50 dòng mẫu để kiểm chứng

Khắc phục 2 vấn đề của pipeline cũ:
  - Regex \b không hoạt động với ranh giới CJK
  - product_id mapping sai kiểu dữ liệu
"""

import os
import re
import csv
import pandas as pd
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

RAW_PARQUET = "data/raw/bilingual_zh_vi.parquet"
OUTPUT_DIR = "csv"

# Bộ ký tự Latin (tiếng Anh + ngôn ngữ châu Âu)
LATIN_CHARS = re.compile(r"[a-zA-Z\u00C0-\u024F]+")

# CJK Unified Ideographs + CJK Compatibility + Fullwidth forms
CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\uff00-\uffef]")


# ═══════════════════════════════════════════════════════════════════
# CORE: Regex CJK-Latin boundary đúng
# ═══════════════════════════════════════════════════════════════════

def extract_english_words(text: str) -> list[str]:
    """
    Trích xuất tất cả chuỗi Latin (từ tiếng Anh) từ văn bản,
    dùng LATIN_CHARS (\\w không đủ vì nó bao gồm cả CJK trong Python 3).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return LATIN_CHARS.findall(text)


def build_clean_pattern(word: str) -> re.Pattern:
    """
    Xây dựng regex để xóa từ tiếng Anh khỏi văn bản CJK.
    Dùng capturing groups để bảo toàn ký tự CJK xung quanh,
    thay vì \\b (không hoạt động với CJK).
    """
    esc = re.escape(word)
    return re.compile(
        r"(^|\s|[\u4e00-\u9fff])" + esc + r"($|\s|[\u4e00-\u9fff])",
        re.IGNORECASE,
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("🔧 CADO TẦNG 0: TRÍCH XUẤT TỪ TIẾNG ANH (Regex CJK-Latin)")
    print("=" * 70)

    # 1. Load dữ liệu
    print("\n📂 Load dữ liệu gốc...")
    df = pd.read_parquet(RAW_PARQUET)
    print(f"   Sản phẩm: {len(df):,}")

    # 2. Trích xuất từ tất cả title_zh + description_zh
    print("\n🔍 Trích xuất từ tiếng Anh...")

    # word -> { "product_ids": set(), "count": 0, "columns": set(), "examples": [] }
    word_index: dict[str, dict] = defaultdict(lambda: {
        "product_ids": set(),
        "count": 0,
        "columns": set(),
        "examples": [],
    })

    col_examples = {"title_zh": [], "description_zh": []}
    total_extracted = 0

    for _, row in df.iterrows():
        pid = str(row["product_id"])

        for col in ["title_zh", "description_zh"]:
            text = row.get(col, "")
            words = extract_english_words(text)
            total_extracted += len(words)

            for w in words:
                info = word_index[w]
                info["product_ids"].add(pid)
                info["count"] += 1
                info["columns"].add(col)
                if len(info["examples"]) < 3:
                    info["examples"].append(text[:120])

    print(f"   Từ unique: {len(word_index):,}")
    print(f"   Tổng lượt trích xuất: {total_extracted:,}")

    # 3. Lưu CSV: english_words.csv
    print("\n💾 Lưu english_words.csv...")
    csv_rows = []
    for word, info in sorted(word_index.items()):
        csv_rows.append({
            "word": word,
            "count": info["count"],
            "columns": "|".join(sorted(info["columns"])),
            "product_id": ",".join(sorted(info["product_ids"])),
        })

    csv_path = os.path.join(OUTPUT_DIR, "english_words.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "count", "columns", "product_id"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"   ✅ {csv_path} ({len(csv_rows):,} dòng)")

    # 4. Trích xuất 50 mẫu title_zh có tiếng Anh
    print("\n" + "=" * 70)
    print("📋 50 MẪU TITLE_ZH CÓ TỪ TIẾNG ANH (TRƯỚC/Sau)")
    print("=" * 70)

    # Lọc sản phẩm có ít nhất 1 từ Latin trong title
    has_english = df["title_zh"].apply(
        lambda t: bool(LATIN_CHARS.search(str(t))) if isinstance(t, str) else False
    )
    sample = df[has_english].head(50)

    # Demo clean: xóa các từ tiếng Anh bị coi là thừa
    # (ở Tầng 0 ta chưa quyết định từ nào thừa, chỉ demo regex)
    sample_rows = []
    for _, row in sample.iterrows():
        pid = row["product_id"]
        title = str(row["title_zh"])
        words = extract_english_words(title)

        # Demo: xóa 1 từ bất kỳ (chỉ để kiểm tra regex)
        demo_word = words[0] if words else "N/A"
        if demo_word != "N/A":
            pattern = build_clean_pattern(demo_word)
            # \1\2 giữ lại ký tự ranh giới (CJK hoặc space)
            cleaned = pattern.sub(r"\1\2", title).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
        else:
            cleaned = title

        sample_rows.append({
            "product_id": pid,
            "title_zh": title,
            "words_found": "|".join(words[:5]),
            "demo_remove": demo_word,
            "demo_cleaned": cleaned,
        })

    # In bảng
    print(f"{'#':<4} {'Product ID':<15} {'Từ tìm thấy':<45} {'Demo xóa':<18} {'Kết quả demo':<50}")
    print("-" * 140)
    for i, r in enumerate(sample_rows, 1):
        pid = r["product_id"][:13]
        words = r["words_found"][:43]
        demo = r["demo_remove"][:16]
        cleaned = r["demo_cleaned"][:48]
        print(f"{i:<4} {pid:<15} {words:<45} {demo:<18} {cleaned:<50}")

    print(f"\n{'─' * 70}")
    print(f"📊 THỐNG KÊ TẦNG 0")
    print(f"   Từ unique: {len(word_index):,}")
    print(f"   Sản phẩm có EN trong title: {has_english.sum():,} ({has_english.sum()/len(df)*100:.1f}%)")
    print(f"   File CSV: {csv_path}")
    print(f"   Sẵn sàng cho Tầng 1 (Whitelist)")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()