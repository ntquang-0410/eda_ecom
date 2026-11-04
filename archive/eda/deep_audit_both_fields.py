# -*- coding: utf-8 -*-
"""
deep_audit_both_fields.py
Kiểm toán siêu chi tiết toàn bộ các góc khuất dữ liệu của Tiêu đề (title_vi_cleaned)
và Thông số kỹ thuật (description_vi_cleaned) trên 4.157 dòng Tiki.
"""

import sys
import re
import unicodedata
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_parquet(r"D:\download\NCKH\eda_ecom\data\tiki_vi_cleaned.parquet")
print(f"Tổng số bản ghi: {len(df)}")

# ==================== 1. AUDIT TITLE ====================
print("\n" + "="*40 + " 1. AUDIT TITLE_VI_CLEANED " + "="*40)

# 1.1 Ký tự vô hình, non-breaking space, zero-width space
def check_hidden_chars(s):
    return [f"U+{ord(c):04X}" for c in str(s) if ord(c) in (0x00A0, 0x200B, 0xFEFF, 0x200E, 0x200F, 0x3000) or (ord(c) < 32 and c not in '\t\n\r')]

hidden_title = df[df['title_vi_cleaned'].apply(lambda x: len(check_hidden_chars(x)) > 0)]
print(f"1. Tiêu đề chứa ký tự ẩn/khoảng trắng đặc thù: {len(hidden_title)}")
if len(hidden_title) > 0:
    for idx, r in hidden_title.head(3).iterrows():
        print(f"   [{r.product_id}]: {check_hidden_chars(r.title_vi_cleaned)} trong '{r.title_vi_cleaned}'")

# 1.2 Dấu câu thừa đầu / đuôi
leading_punct = df[df['title_vi_cleaned'].str.contains(r'^[^\w\(\[\{]+', na=False)]
print(f"2. Tiêu đề có dấu câu thừa ở ĐẦU: {len(leading_punct)}")
if len(leading_punct) > 0:
    for idx, r in leading_punct.head(3).iterrows():
        print(f"   [{r.product_id}]: {r.title_vi_cleaned}")

trailing_punct = df[df['title_vi_cleaned'].str.contains(r'[^\w\)\}\]\s]+$', na=False)]
print(f"3. Tiêu đề có dấu câu thừa ở CUỐI: {len(trailing_punct)}")
if len(trailing_punct) > 0:
    for idx, r in trailing_punct.head(3).iterrows():
        print(f"   [{r.product_id}]: {r.title_vi_cleaned}")

# 1.3 Khoảng trắng trước dấu phẩy/chấm phẩy (ví dụ: 'từ , từ')
space_before_punct = df[df['title_vi_cleaned'].str.contains(r'\s+[,;:](?!\d)', na=False)]
print(f"4. Tiêu đề có khoảng trắng TRƯỚC dấu câu (ví dụ: 'từ , '): {len(space_before_punct)}")
if len(space_before_punct) > 0:
    for idx, r in space_before_punct.head(5).iterrows():
        print(f"   [{r.product_id}]: {r.title_vi_cleaned}")

# 1.4 Thiếu khoảng trắng sau dấu phẩy (ví dụ: 'từ,từ')
missing_space_after_comma = df[df['title_vi_cleaned'].str.contains(r'(?<=[a-zA-ZÀ-ỹ]),(?=[a-zA-ZÀ-ỹ])', na=False)]
print(f"5. Tiêu đề dính chữ sau dấu phẩy (ví dụ: 'từ,từ'): {len(missing_space_after_comma)}")
if len(missing_space_after_comma) > 0:
    for idx, r in missing_space_after_comma.head(5).iterrows():
        print(f"   [{r.product_id}]: {r.title_vi_cleaned}")

# 1.5 Dấu ngoặc mở không đóng hoặc ngoặc rỗng
def check_brackets(s):
    s = str(s)
    if re.search(r'\(\s*\)|\[\s*\]|\{\s*\}', s):
        return 'empty_bracket'
    if s.count('(') != s.count(')'):
        return 'unclosed_parenthesis'
    if s.count('[') != s.count(']'):
        return 'unclosed_square_bracket'
    if re.search(r'[【】⌈⌋«»“”‹›<>{}~`^]', s):
        return 'exotic_brackets_symbols'
    return None

bracket_issues = df[df['title_vi_cleaned'].apply(check_brackets).notnull()]
print(f"6. Tiêu đề dính lỗi ngoặc (lệch đóng mở, rỗng, ký tự lạ): {len(bracket_issues)}")
if len(bracket_issues) > 0:
    for idx, r in bracket_issues.head(5).iterrows():
        print(f"   [{r.product_id}] ({check_brackets(r.title_vi_cleaned)}): {r.title_vi_cleaned}")

# 1.6 Đơn vị đo lường dính liền số (chưa cách khoảng trắng)
unspaced_unit = df[df['title_vi_cleaned'].str.contains(r'\b\d+(?:cm|mm|ml|kg|mah|hz|rpm|inch)\b', case=False, na=False)]
print(f"7. Tiêu đề có đơn vị đo lường dính liền số (chưa cách khoảng trắng): {len(unspaced_unit)}")
if len(unspaced_unit) > 0:
    for idx, r in unspaced_unit.head(5).iterrows():
        print(f"   [{r.product_id}]: {r.title_vi_cleaned}")

# 1.7 Toán tử nhân x dính liền (ví dụ 10x20 thay vì 10 x 20)
unspaced_x = df[df['title_vi_cleaned'].str.contains(r'\b\d+x\d+\b', case=False, na=False)]
print(f"8. Tiêu đề có toán tử x dính liền (ví dụ 10x20): {len(unspaced_x)}")
if len(unspaced_x) > 0:
    for idx, r in unspaced_x.head(5).iterrows():
        print(f"   [{r.product_id}]: {r.title_vi_cleaned}")

# 1.8 Từ viết hoa toàn bộ (>= 4 chữ cái) không phải từ viết tắt
ACRONYMS = {'LED', 'UV', 'TWS', 'USB', 'PVC', 'PE', 'ABS', 'INOX', 'OEM', 'SKU', '3D', '4D', '5D', 'SSD', 'RAM', 'CPU', 'GPU', 'ISO', 'FDA', 'CE', 'TFT', 'LCD', 'HD', 'FHD', '2K', '4K', 'RGB', 'WRGB', 'DPI', 'TYPEC', 'TYPE-C', 'QC', 'PD', 'BT', 'PRO', 'MAX', 'PLUS', 'MINI', 'LTD', 'CO', 'V', 'W', 'A', 'MAH', 'GB', 'TB', 'HZ', 'RPM', 'INCH', 'PSI', 'BAR', 'TP-LINK', 'D-LINK', 'ASUS', 'DELL', 'HP', 'LG', 'MSI', 'JBL', 'SONY'}
def find_unnormalized_caps(s):
    words = re.findall(r'\b[A-ZÀ-Ỹ]{4,}\b', str(s))
    weird = [w for w in words if w not in ACRONYMS]
    return weird

caps_issues = df[df['title_vi_cleaned'].apply(lambda x: len(find_unnormalized_caps(x)) > 0)]
print(f"9. Tiêu đề chứa từ viết hoa toàn bộ (>=4 chữ cái) chưa chuẩn hóa: {len(caps_issues)}")
if len(caps_issues) > 0:
    for idx, r in caps_issues.head(5).iterrows():
        print(f"   [{r.product_id}] Từ: {find_unnormalized_caps(r.title_vi_cleaned)} trong: {r.title_vi_cleaned}")

# 1.9 Dấu gạch nối và khoảng trắng lặp (ví dụ '  ' hoặc ' - - ')
double_spaces = df[df['title_vi_cleaned'].str.contains(r'\s{2,}', na=False)]
print(f"10. Tiêu đề có khoảng trắng kép ('  '): {len(double_spaces)}")

# ==================== 2. AUDIT SPECS ====================
print("\n" + "="*40 + " 2. AUDIT DESCRIPTION_VI_CLEANED " + "="*40)

# 2.1 Ký tự vô hình trong specs
hidden_specs = df[df['description_vi_cleaned'].apply(lambda x: len(check_hidden_chars(x)) > 0)]
print(f"1. Specs chứa ký tự ẩn/khoảng trắng đặc thù: {len(hidden_specs)}")

# 2.2 Mục không theo khuôn dạng 'Khóa: Giá trị'
def check_spec_items(s):
    items = [x.strip() for x in str(s).split(';') if x.strip()]
    no_colon = []
    empty_val = []
    empty_key = []
    for it in items:
        if ':' not in it:
            no_colon.append(it)
        else:
            k, v = it.split(':', 1)
            if not k.strip():
                empty_key.append(it)
            if not v.strip():
                empty_val.append(it)
    return no_colon, empty_key, empty_val

no_colon_rows = []
empty_val_rows = []
for idx, r in df.iterrows():
    nc, ek, ev = check_spec_items(r['description_vi_cleaned'])
    if nc: no_colon_rows.append((r.product_id, nc))
    if ev: empty_val_rows.append((r.product_id, ev))

print(f"2. Specs có mục KHÔNG có dấu hai chấm (không phải Khóa: Giá trị): {len(no_colon_rows)}")
if len(no_colon_rows) > 0:
    for pid, nc in no_colon_rows[:5]:
        print(f"   [{pid}]: {nc}")

print(f"3. Specs có Khóa bị RỖNG GIÁ TRỊ ('Khóa: '): {len(empty_val_rows)}")

# 2.3 Khóa trùng lặp trong cùng 1 sản phẩm
def check_dup_keys(s):
    items = [x.strip() for x in str(s).split(';') if x.strip()]
    keys = []
    for it in items:
        if ':' in it:
            keys.append(it.split(':', 1)[0].strip().lower())
    dups = [k for k in keys if keys.count(k) > 1]
    return list(set(dups))

dup_key_rows = df[df['description_vi_cleaned'].apply(lambda x: len(check_dup_keys(x)) > 0)]
print(f"4. Specs có KHÓA TRÙNG LẶP trong cùng sản phẩm: {len(dup_key_rows)}")
if len(dup_key_rows) > 0:
    for idx, r in dup_key_rows.head(5).iterrows():
        print(f"   [{r.product_id}]: {check_dup_keys(r.description_vi_cleaned)}")

# 2.4 Dấu chấm phẩy thừa (chấm phẩy đầu, đuôi, hoặc ';;')
double_semicolon = df[df['description_vi_cleaned'].str.contains(r';\s*;', na=False)]
print(f"5. Specs có chấm phẩy kép (';;'): {len(double_semicolon)}")

trailing_semicolon = df[df['description_vi_cleaned'].str.contains(r';\s*$', na=False)]
print(f"6. Specs có chấm phẩy ở CUỐI: {len(trailing_semicolon)}")

leading_semicolon = df[df['description_vi_cleaned'].str.contains(r'^\s*;', na=False)]
print(f"7. Specs có chấm phẩy ở ĐẦU: {len(leading_semicolon)}")

# 2.5 Đơn vị đo lường dính liền trong Specs (ví dụ: '11W' hoặc '26cm')
unspaced_unit_specs = df[df['description_vi_cleaned'].str.contains(r'\b\d+(?:cm|mm|ml|kg|mah|hz|rpm|inch)\b', case=False, na=False)]
print(f"8. Specs có đơn vị đo dính liền số: {len(unspaced_unit_specs)}")
if len(unspaced_unit_specs) > 0:
    for idx, r in unspaced_unit_specs.head(5).iterrows():
        print(f"   [{r.product_id}]: {r.description_vi_cleaned[:120]}")

# 2.6 Thuộc tính có giá trị rác (như 'null', 'None', 'undefined', 'n/a', '-')
junk_values = df[df['description_vi_cleaned'].str.contains(r':\s*(?:null|none|undefined|n/a|-|\?)\s*(?:;|$)', case=False, na=False)]
print(f"9. Specs chứa giá trị rác ('null', 'None', 'n/a', '-'): {len(junk_values)}")
if len(junk_values) > 0:
    for idx, r in junk_values.head(5).iterrows():
        print(f"   [{r.product_id}]: {r.description_vi_cleaned[:120]}")

# 2.7 Độ dài specs bất thường (< 15 ký tự hoặc > 1000 ký tự)
short_specs = df[df['description_vi_cleaned'].str.len() < 15]
print(f"10. Specs quá ngắn (< 15 ký tự): {len(short_specs)}")
if len(short_specs) > 0:
    for idx, r in short_specs.iterrows():
        print(f"   [{r.product_id}]: {r.description_vi_cleaned}")
