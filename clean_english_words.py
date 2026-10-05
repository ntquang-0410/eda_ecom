#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xử lý từ tiếng Anh trong title_zh và description_zh.

Chiến lược 3 tầng:
  1. Whitelist: giữ nguyên (thương hiệu, viết tắt, đơn vị, từ vay mượn)
  2. Pattern-based: nhận diện tự động (mã sản phẩm, viết tắt ngắn)
  3. WordNet semantic: xóa physical_object_zh_match, review physical_object_no_zh

Đầu vào:
  - bilingual_zh_vi.parquet        (dữ liệu gốc)
  - english_words_wordnet_flagged.csv  (từ đã gắn cờ)
  - english_words.csv              (từ + danh sách product_id)

Đầu ra:
  - bilingual_zh_vi_cleaned.parquet  (parquet mới có cột title_zh_cleaned, description_zh_cleaned)
  - cleaning_report.csv              (báo cáo chi tiết từng từ: giữ/xóa + lý do)
"""

import os
import re
import pandas as pd

def get_csv_path(filename: str) -> str:
    """Trả về đường dẫn tới file csv (trong folder csv/ hoặc root)."""
    csv_path = os.path.join("csv", filename)
    if os.path.exists(csv_path):
        return csv_path
    return filename

# ═══════════════════════════════════════════════════════════════════
# TẦNG 1: WHITELIST — Từ chắc chắn giữ nguyên
# ═══════════════════════════════════════════════════════════════════

# Thương hiệu / Tên sản phẩm phổ biến
BRANDS = {
    # Điện thoại / Công nghệ
    "iPhone", "OPPO", "vivo", "Samsung", "Xiaomi", "Huawei", "Redmi",
    "Nokia", "Motorola", "OnePlus", "Realme", "POCO", "Honor",
    "iPad", "MacBook", "AirPods", "Apple",
    "MagSafe", "ProMax", "PROMAX",
    # Thương hiệu thời trang / Phụ kiện
    "BAPE", "POLO", "Gucci", "Nike", "Adidas", "Puma", "FILA",
    "Jeep", "BANGE", "Lolita",
    # Thương hiệu khác
    "SPACEXPERT", "Yahababy", "kakashow", "finercare", "kekemood",
    "birstory", "hegen", "eiio", "babypal", "swicky",
    # Model / dòng sản phẩm
    "Pro", "Max", "Plus", "Mini", "mini", "Air", "Ultra", "Lite",
    "Note", "Mate", "Nova", "Galaxy", "Buds",
}

# Viết tắt kỹ thuật / chuyên ngành
TECH_ABBREVIATIONS = {
    # Kết nối / Truyền thông
    "USB", "WIFI", "WiFi", "Wifi", "GPS", "NFC", "HDMI", "VGA", "DP",
    "Bluetooth", "BT", "RF", "IR", "LTE", "SIM", "OTG", "AUX",
    # Hiển thị / Âm thanh
    "LED", "LCD", "OLED", "AMOLED", "HD", "FHD", "QHD", "UHD",
    "ANC", "ENC", "HiFi", "Hi-Fi", "TWS", "DAC",
    # Ô tô / Xe
    "ADAS", "DVR", "ACC", "OBD", "ECU", "ABS", "EPS", "SUV",
    # Chất liệu
    "PU", "PVC", "ABS", "EVA", "TPU", "TPE", "TPR", "PP", "PE", "PC",
    "PPSU", "PET", "HDPE", "LDPE", "PTFE", "PA", "POM",
    # Công nghệ / Tiêu chuẩn
    "OEM", "ODM", "APP", "AI", "AR", "VR", "IoT", "API",
    "IP", "IP65", "IP67", "IP68",
    "TSA", "CE", "FCC", "RoHS", "FDA", "SGS",
    "DIY", "CNC", "CAD",
    # Điện / Năng lượng
    "DC", "AC", "MPPT", "PWM",
    # Tin học
    "GB", "TB", "MB", "SSD", "HDD", "RAM", "ROM", "CPU", "GPU",
    "DDR", "SATA", "NVMe", "PCIe",
    # Khác
    "UV", "SPF", "LOGO", "LOG", "ID", "QC", "WWW", "PDF",
    "CD", "MD", "DJ", "FM", "AM",
    "SS", "XY", "TC", "TK",
}

# Đơn vị đo / kích thước
UNITS_AND_SIZES = {
    "cm", "mm", "m", "km", "kg", "g", "mg", "ml", "mL", "oz", "lb",
    "mAh", "mah", "Wh", "kWh",
    "dBi", "dB", "Hz", "kHz", "MHz", "GHz",
    "XS", "XL", "XXL", "XXXL", "XXXXL", "XXXXXL",
    "SML", "L", "M", "S",
}

# Từ vay mượn phổ biến trong thời trang / e-commerce Trung Quốc
LOANWORDS = {
    "oversize", "cleanfit", "vintage", "ulzzang", "chic", "boxy",
    "dashcam", "cardvr", "magsafe", "tritan",
    "hello", "fit", "slim", "basic", "casual", "retro", "punk",
    "harajuku", "kawaii", "mori", "preppy", "boho",
    "ins", "INS",  # Instagram style
}

# Hợp nhất tất cả whitelist
WHITELIST = set()
for s in [BRANDS, TECH_ABBREVIATIONS, UNITS_AND_SIZES, LOANWORDS]:
    WHITELIST.update(s)

# Tạo bản lowercase để so khớp không phân biệt hoa thường
WHITELIST_LOWER = {w.lower() for w in WHITELIST}


# ═══════════════════════════════════════════════════════════════════
# TẦNG 2: PATTERN-BASED — Nhận diện tự động
# ═══════════════════════════════════════════════════════════════════

def is_pattern_keep(word: str) -> str | None:
    """
    Kiểm tra từ có thuộc pattern nên giữ không.
    Trả về tên pattern nếu khớp, None nếu không.
    """
    if not word:
        return None

    # Mã sản phẩm: chứa số (iPhone14, x200cm, M416, B5, 1080P...)
    if re.search(r'\d', word):
        return "model_code"

    # Viết tắt ngắn toàn chữ IN HOA (1-5 ký tự) — khả năng cao là viết tắt
    if re.match(r'^[A-Z]{1,5}$', word):
        return "short_abbreviation"

    # CamelCase thường là brand name
    if re.match(r'^[A-Z][a-z]+[A-Z][a-zA-Z]*$', word):
        return "camelcase_brand"

    # Từ rất ngắn (1-2 ký tự) — thường không mang nghĩa độc lập
    if len(word) <= 2:
        return "too_short"

    return None


# ═══════════════════════════════════════════════════════════════════
# TẦNG 3: WORDNET SEMANTIC — Quyết định xóa/giữ
# ═══════════════════════════════════════════════════════════════════

# Các từ physical_object_no_zh nên GIỮ (thực ra là brand/tech/style, không phải vật thể)
PHYSICAL_NO_ZH_KEEP = {
    "Pro", "WIFI", "PC", "PM", "samba", "miss", "XY", "Six",
    "DJ", "MARK", "Dodo", "HS1", "tee", "Ginger", "Lolita",
    "Galaxy", "ma", "toast", "Dieter", "Mommy", "Sports",
    "MILO", "LULU", "Number", "Zero", "Extra", "Hi-Fi",
    "Mulberry", "Gel", "cos", "Vegan", "Male", "newborn",
    "rodeo", "gentlewoman", "mollie", "glaze",
    "Ginseng", "hobo", "Turmeric", "Cinnamon", "growth",
    "Slipper", "slippers",  # Thường là tên sản phẩm, không dư thừa
}

# Các từ physical_object_zh_match mà thực ra nên GIỮ (brand/tech trong ngữ cảnh ecom)
PHYSICAL_MATCH_KEEP = {
    "AI", "PET", "Air", "mini", "CD", "GPS", "LED", "tea",
    "baby", "kitty", "MD", "LP", "LCD",
    "Auto", "Car",  # Thường là phần tên sản phẩm
    "bear", "fox", "Peach", "Maple", "Cyme",  # Thường là brand/style name
    "Mate", "Note",  # Model Huawei
    "Pro",
    "GREEN", "Green", "BLACK", "Black", "Pink",  # Thường là tên màu/brand
    "HEAT", "Flash", "Track", "Step", "Cut", "Clip",  # Thường là model/feature name
    "Five", "Six",
    "HA", "MACK", "POL", "COAST", "CURB", "HORIZON",
    "bob", "ape", "pod",
    "Daddy", "pa", "Man", "Men", "Women", "women", "Girls",
    "classic", "Originals",
    "Warrior", "King",
    "camera",  # Thường đi kèm dashcam, không dư thừa
    "HOME", "LIFE",
    "power", "balance",
}


def classify_word(word: str, flag: str) -> tuple[str, str]:
    """
    Phân loại một từ tiếng Anh: giữ hay xóa.

    Returns:
        (action, reason)
        action: "keep" hoặc "remove"
        reason: lý do chi tiết
    """
    if not word or not isinstance(word, str):
        return ("keep", "empty")

    # --- Tầng 1: Whitelist ---
    if word in WHITELIST or word.lower() in WHITELIST_LOWER:
        return ("keep", "whitelist")

    # --- Tầng 2: Pattern-based ---
    pattern = is_pattern_keep(word)
    if pattern:
        return ("keep", f"pattern:{pattern}")

    # --- Tầng 3: WordNet Semantic ---
    if flag == "physical_object_zh_match":
        if word in PHYSICAL_MATCH_KEEP:
            return ("keep", "phys_match_but_keep")
        return ("remove", "physical_object_zh_match")

    if flag == "physical_object_no_zh":
        if word in PHYSICAL_NO_ZH_KEEP:
            return ("keep", "phys_no_zh_but_keep")
        return ("remove", "physical_object_no_zh")

    # not_physical_object — giữ
    return ("keep", "not_physical_object")


# ═══════════════════════════════════════════════════════════════════
# MAIN: Xử lý dữ liệu
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔧 BẮT ĐẦU XỬ LÝ TỪ TIẾNG ANH TRONG TITLE_ZH & DESCRIPTION_ZH")
    print("=" * 70)

    # 1. Load dữ liệu
    print("\n📂 Đang load dữ liệu...")
    df = pd.read_parquet("bilingual_zh_vi.parquet")
    flagged = pd.read_csv(get_csv_path("english_words_wordnet_flagged.csv"))
    flagged["word"] = flagged["word"].fillna("").astype(str)
    english_words_raw = pd.read_csv(get_csv_path("english_words.csv"))
    english_words_raw["word"] = english_words_raw["word"].fillna("").astype(str)

    print(f"  Sản phẩm: {len(df):,}")
    print(f"  Từ tiếng Anh (flagged): {len(flagged):,}")

    # 2. Phân loại từng từ
    print("\n🏷️ Phân loại từ tiếng Anh...")
    report_rows = []
    words_to_remove = set()

    for _, row in flagged.iterrows():
        word = row["word"]
        flag = row["flag"]
        columns = row.get("columns", "")
        count = row.get("count", 0)

        action, reason = classify_word(word, flag)

        report_rows.append({
            "word": word,
            "count": count,
            "columns": columns,
            "flag": flag,
            "action": action,
            "reason": reason,
            "zh_translations": row.get("zh_translations", ""),
        })

        if action == "remove":
            words_to_remove.add(word)

    report_df = pd.DataFrame(report_rows)

    # Thống kê
    keep_count = len(report_df[report_df["action"] == "keep"])
    remove_count = len(report_df[report_df["action"] == "remove"])
    print(f"  ✅ Giữ nguyên: {keep_count:,} từ")
    print(f"  🔴 Xóa: {remove_count:,} từ")
    print(f"\n  Từ sẽ bị xóa:")
    removed = report_df[report_df["action"] == "remove"].sort_values("count", ascending=False)
    # Chỉ hiện trong title
    removed_title = removed[removed["columns"].str.contains("title_zh", na=False)]
    print(f"    Trong title_zh: {len(removed_title)} từ")
    for _, r in removed_title.head(20).iterrows():
        print(f"      {r['word']:20s} count={r['count']:4}  reason={r['reason']}")
    if len(removed_title) > 20:
        print(f"      ... và {len(removed_title) - 20} từ nữa")

    # 3. Xây mapping: product_id → list of words to remove
    print("\n🗺️ Xây mapping product → words to remove...")
    # Parse english_words.csv để lấy product_id cho mỗi word
    word_to_products = {}
    for _, row in english_words_raw.iterrows():
        word = row["word"]
        if word not in words_to_remove:
            continue
        cols = row.get("columns", "")
        pids_str = str(row.get("product_id", ""))
        pids = [p.strip() for p in pids_str.split(",") if p.strip()]
        word_to_products[word] = {"product_ids": pids, "columns": cols}

    # Invert: product_id → set of words to remove
    product_words_to_remove = {}
    for word, info in word_to_products.items():
        for pid in info["product_ids"]:
            if pid not in product_words_to_remove:
                product_words_to_remove[pid] = set()
            product_words_to_remove[pid].add(word)

    affected_products = len(product_words_to_remove)
    print(f"  Sản phẩm bị ảnh hưởng: {affected_products:,}")

    # 4. Làm sạch title_zh và description_zh
    print("\n🧹 Đang làm sạch...")

    def clean_text(text, words_set):
        """Xóa các từ tiếng Anh khỏi text, giữ nguyên cấu trúc câu."""
        if not isinstance(text, str) or not words_set:
            return text

        cleaned = text
        for word in sorted(words_set, key=len, reverse=True):
            # Xóa từ + khoảng trắng thừa, giữ nguyên ký tự tiếng Trung liền kề
            # Pattern: word boundary hoặc ranh giới CJK-Latin
            pattern = re.compile(
                r'(?<=[^\w]|^)' + re.escape(word) + r'(?=[^\w]|$)'
                + r'|'
                + re.escape(word),
                re.IGNORECASE
            )
            cleaned = pattern.sub("", cleaned)

        # Dọn dẹp: xóa khoảng trắng thừa
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    df["title_zh_cleaned"] = df["title_zh"]
    df["description_zh_cleaned"] = df["description_zh"]

    cleaned_count = 0
    for pid_str, words_set in product_words_to_remove.items():
        try:
            pid = int(pid_str)
        except (ValueError, TypeError):
            continue

        mask = df["product_id"] == pid
        if not mask.any():
            continue

        idx = df.index[mask]
        for i in idx:
            orig_title = df.at[i, "title_zh"]
            orig_desc = df.at[i, "description_zh"]

            new_title = clean_text(orig_title, words_set)
            new_desc = clean_text(orig_desc, words_set)

            if new_title != orig_title or new_desc != orig_desc:
                df.at[i, "title_zh_cleaned"] = new_title
                df.at[i, "description_zh_cleaned"] = new_desc
                cleaned_count += 1

    print(f"  Sản phẩm đã được làm sạch: {cleaned_count:,}")

    # 5. Hiển thị mẫu trước/sau
    print("\n📋 MẪU TRƯỚC/SAU:")
    changed = df[df["title_zh"] != df["title_zh_cleaned"]].head(15)
    for _, row in changed.iterrows():
        print(f"  ID: {row['product_id']}")
        print(f"    TRƯỚC: {row['title_zh']}")
        print(f"    SAU  : {row['title_zh_cleaned']}")
        print()

    # 6. Lưu kết quả
    print("💾 Đang lưu...")
    df.to_parquet("bilingual_zh_vi_cleaned.parquet", index=False)
    print(f"  ✅ bilingual_zh_vi_cleaned.parquet ({len(df):,} dòng)")

    report_path = os.path.join("csv", "cleaning_report.csv") if os.path.exists("csv") else "cleaning_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {report_path} ({len(report_df):,} dòng)")

    # 7. Thống kê cuối
    title_changed = (df["title_zh"] != df["title_zh_cleaned"]).sum()
    desc_changed = (df["description_zh"] != df["description_zh_cleaned"]).sum()
    print(f"\n📊 THỐNG KÊ CUỐI:")
    print(f"  Title thay đổi: {title_changed:,} / {len(df):,} ({title_changed/len(df)*100:.1f}%)")
    print(f"  Description thay đổi: {desc_changed:,} / {len(df):,} ({desc_changed/len(df)*100:.1f}%)")
    print("\n✅ HOÀN TẤT!")


if __name__ == "__main__":
    main()
