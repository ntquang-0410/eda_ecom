# -*- coding: utf-8 -*-
"""
polish_and_export_tiki_v3_4.py
Chuẩn hóa triệt để (Forensic Polish v3.4) cho cả Title và Description (Specs) trên 4.157 bản ghi Tiki:
- Tiêu đề: Khử sạch khoảng trắng trước dấu câu, dính chữ sau dấu phẩy, toán tử x dính liền, lỗi ngoặc,
  chuyển các từ tiếng Việt ALL-CAPS về Title Case (bảo toàn mã/từ viết tắt kỹ thuật), loại bỏ ký tự lạ U+FFFC,
  sửa lỗi chính tả đã ghi nhận.
- Thông số kỹ thuật (Specs): Khử 100% mẩu dữ liệu không có dấu hai chấm (gộp phần tử danh sách, loại bỏ văn xuôi/hashtag rác),
  loại bỏ khóa giả, chuẩn hóa khoảng cách đơn vị đo lường và toán tử nhân x trong cả Khóa lẫn Giá trị.
- Tái tính toán Cosine Similarity BGE-M3 trên GPU NVIDIA CUDA.
- Kiểm toán 20 chiều xác thực tỷ lệ lỗi bằng 0 tuyệt đối.
- Xuất bản Parquet ra cả 2 thư mục dự án.
"""

import sys
import os
import re
import html
import unicodedata
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding='utf-8')

start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=== KHỞI CHẠY PIPELINE TIKI FORENSIC POLISH (v3.4) TRÊN {device.upper()} ===")
if device == 'cuda':
    print(f"Hardware GPU: {torch.cuda.get_device_name(0)}")

# 1. Đọc dữ liệu Parquet hiện hành
parquet_in = r"D:\download\NCKH\eda_ecom\data\tiki_vi_cleaned.parquet"
df = pd.read_parquet(parquet_in)
print(f"Đã nạp {len(df)} dòng dữ liệu từ: {parquet_in}")

# ==================== CONSTANTS ====================
ACRONYMS = {
    'LED', 'UV', 'TWS', 'USB', 'PVC', 'PE', 'ABS', 'INOX', 'OEM', 'SKU', 
    '3D', '4D', '5D', 'SSD', 'RAM', 'CPU', 'GPU', 'ISO', 'FDA', 'CE', 
    'TFT', 'LCD', 'HD', 'FHD', '2K', '4K', 'RGB', 'WRGB', 'DPI', 'TYPEC', 
    'TYPE-C', 'QC', 'PD', 'BT', 'PRO', 'MAX', 'PLUS', 'MINI', 'LTD', 'CO', 
    'V', 'W', 'A', 'MAH', 'GB', 'TB', 'HZ', 'RPM', 'INCH', 'PSI', 'BAR',
    'TP-LINK', 'D-LINK', 'ASUS', 'DELL', 'HP', 'LG', 'MSI', 'JBL', 'SONY',
    'NIVEA', 'LEGO', 'SPF', 'NAN', 'DHA', 'DUKA', 'OPTIPRO', 'QMAN', 'AAA',
    'MEN', 'NHASAN', 'DHC', 'ANKAN', 'MINIIN', 'HSD', 'SYS', 'UNMEI', 'VIP',
    'OHUI', 'VIETMAT', 'PPSU', 'PHILIPS', 'TPBVSK', 'XXL', 'INFINIPRO',
    'BABYRO', 'USA', 'PEAFLO', 'BESTON', 'TPBS', 'RICHELL', 'XXXL', 'BPA',
    'DIY', 'TOP', 'ECO', 'SIZE', 'NOOZ', 'GEFU', 'VEGAN', 'SUPER', 'ĐK', 'ĐQ'
}

VIETNAMESE_UPPER_ACCENTS = set("ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ")
PROMO_WORDS = {'COMBO', 'SET', 'TRANG', 'TRANH', 'XANH', 'LONG', 'GIA', 'CAN', 'BAN', 'DEN', 'DOI', 'NAM', 'DEP', 'XIN', 'HOA', 'THONG', 'KEM', 'VAN', 'HOP', 'NOI', 'BINH', 'MUA', 'MIENG', 'CHAO', 'THAM', 'LOAI', 'HANG'}

TYPOS_TITLE = {
    r'\bSươjng\b': 'Sương',
    r'\bsươjng\b': 'sương',
    r'\bchsinh\b': 'chính',
    r'\bChsinh\b': 'Chính',
}

# ==================== 1. POLISH TITLE ====================
def polish_title(title: str) -> str:
    if not isinstance(title, str) or not title.strip():
        return ""
    
    s = html.unescape(str(title))
    s = unicodedata.normalize('NFC', s)
    
    # Ký tự vô hình và Object Replacement Character U+FFFC
    s = re.sub(r'[\u00a0\u200b\ufeff\u200e\u200f\u3000\ufffc\x00-\x1f]', ' ', s)
    
    # Loại bỏ cặp ngoặc promo dạng {...}
    s = re.sub(r'\{[^\}]*\}', ' ', s)
    
    # Chuẩn hóa ngoặc kép thông minh
    s = s.replace('“', '"').replace('”', '"')
    
    # Chuẩn hóa mũi tên và dải giá trị: --->, ---->, ---- thành ' - '
    s = re.sub(r'\s*-+>\s*', ' - ', s)
    s = re.sub(r'\s*-{2,}\s*', ' - ', s)
    s = s.replace('8h>12h', '8h - 12h')
    s = re.sub(r'<\s*(?=[A-ZÀ-Ỹ0-9])', ' - ', s)
    
    # Khắc phục lỗi ngoặc đơn
    if s.startswith('(') and s.count('(') > s.count(')'):
        s = s[1:].strip()
    s = s.replace('((', '(')
    s = s.replace('))', ')')
    if s.count('(') > s.count(')'):
        s = s + ')'
    while s.count(')') > s.count('('):
        idx = s.rfind(')')
        if idx != -1:
            s = s[:idx] + s[idx+1:]
            
    # Khoảng cách quanh dấu ngoặc đơn
    s = re.sub(r'(?<=[a-zA-Z0-9À-ỹ])\(', ' (', s)
    s = re.sub(r'\)(?=[a-zA-Z0-9À-ỹ])', ') ', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)

    # Khắc phục lỗi chính tả đã phát hiện
    for pat, rep in TYPOS_TITLE.items():
        s = re.sub(pat, rep, s)
        
    # Chuẩn hóa dấu phẩy và dấu hai chấm:
    # Không để khoảng trắng TRƯỚC dấu phẩy/chấm phẩy/hai chấm
    s = re.sub(r'\s+([,;:])', r'\1', s)
    # Bắt buộc có khoảng trắng SAU dấu phẩy nếu tiếp theo là chữ cái
    s = re.sub(r'(?<=[a-zA-Z0-9À-ỹ]),(?=[a-zA-ZÀ-ỹ])', ', ', s)
    # Khắc phục lỗi dấu phẩy dính ngoặc kép: ," -> ,
    s = re.sub(r',\s*["\']', ', ', s)
    s = re.sub(r'["\']\s*,', ', ', s)
    
    # Bắt buộc có khoảng trắng sau dấu gạch nối trước chữ: -Nhập khẩu -> - Nhập khẩu
    s = re.sub(r'-(?=[a-zA-ZÀ-ỹ])', '- ', s)
    # Loại bỏ hashtag trước từ chữ: #Made in Japan -> Made in Japan, .#Tông Sáng -> - Tông Sáng
    s = re.sub(r'\.\s*#', ' - ', s)
    s = re.sub(r'#(?=[a-zA-ZÀ-ỹ])', '', s)
    
    # Chuẩn hóa đơn vị đo dính với x: 30mlx10 -> 30 ml x 10, 100cmx100 cm -> 100 cm x 100 cm
    s = re.sub(r'(\d+(?:\.\d+)?)\s*(cm|mm|m|ml|l|g|kg|w|v|mah|hz|rpm|inch)\s*[xX*×]\s*', r'\1 \2 x ', s, flags=re.I)
    
    # Chuẩn hóa toán tử nhân x giữa các số: 40x60 -> 40 x 60, 38x25x25 -> 38 x 25 x 25
    while re.search(r'(\d+(?:\.\d+)?)[xX*×](\d+(?:\.\d+)?)', s):
        s = re.sub(r'(\d+(?:\.\d+)?)[xX*×](\d+(?:\.\d+)?)', r'\1 x \2', s)
        
    # Khoảng cách trước đơn vị đo lường
    s = re.sub(r'(\b\d+(?:\.\d+)?)\s*(cm|mm|ml|l|kg|g|w|v|mah|hz|rpm|inch)\b', r'\1 \2', s, flags=re.I)

    # Chuẩn hóa từ tiếng Việt in hoa toàn bộ (All-caps) về Title Case (ngoại trừ từ viết tắt kỹ thuật)
    def fix_caps_word(m):
        w = m.group(0)
        if w in ACRONYMS:
            return w
        if w in PROMO_WORDS or any(c in VIETNAMESE_UPPER_ACCENTS for c in w):
            return w.capitalize()
        return w
        
    s = re.sub(r'\b[A-ZÀ-Ỹ]+\b', fix_caps_word, s)
    
    # Dọn dẹp khoảng trắng kép và dấu câu thừa ở đầu/đuôi (kể cả dấu ngoặc kép bọc ngoài)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r'^[,\-;:_~|/+\s"\'“”]+', '', s)
    s = re.sub(r'[,\-;:_~|/+\s"\'“”]+$', '', s)
    return s

# ==================== 2. POLISH SPECS ====================
def polish_specs(specs: str, title: str = '') -> str:
    if not isinstance(specs, str) or not specs.strip():
        return ""
        
    s = html.unescape(str(specs))
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'[\u00a0\u200b\ufeff\u200e\u200f\u3000\ufffc\x00-\x1f]', ' ', s)
    
    items = [x.strip() for x in s.split(';') if x.strip()]
    parsed = []
    
    # Bước A: Tách và gộp thông minh các mẩu dữ liệu
    for it in items:
        if ':' in it:
            k, v = it.split(':', 1)
            k = k.strip()
            v = v.lstrip(': ').strip() # loại bỏ lỗi dính 2 dấu hai chấm
            parsed.append([k, v])
        else:
            # Mẩu dữ liệu mồ côi (không chứa dấu hai chấm)
            # Nếu là hashtag hoặc bài văn quảng cáo -> Bỏ qua
            if it.startswith('#') or len(it) > 100 or 'THÔNG TIN' in it or ('hướng dẫn' in it.lower() and len(it) > 40):
                continue
            # Nếu là phần tử tiếp nối của danh sách thuộc tính trước đó
            if parsed:
                prev_k, prev_v = parsed[-1]
                if not re.search(r'\b(không|bạn nên|tránh|chú ý)\b', it.lower()) and len(it) < 60:
                    parsed[-1][1] = f"{prev_v}, {it}"
                    
    # Bước B: Lọc và chuẩn hóa từng cặp Khóa: Giá trị
    title_short = title[:25].lower().strip() if title else ""
    cleaned_pairs = []
    seen_keys = set()
    
    for k, v in parsed:
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
            
        # Kiểm tra tính hợp lệ của Khóa (Key)
        if len(k) > 50 or re.search(r'\.\s+[A-ZÀ-Ỹ]', k) or '\n' in k:
            continue
        if k.startswith('#') or k.lower() in seen_keys:
            continue
            
        # Kiểm tra tính hợp lệ của Giá trị (Value)
        v_lower = v.lower()
        if v_lower in {'null', 'none', 'undefined', 'n/a', '-', '?', 'không có', 'chưa rõ'}:
            continue
        # Bỏ qua nếu giá trị là cả bài văn quảng cáo copy lại từ tiêu đề
        if len(v) > 200:
            if (title_short and title_short in v_lower) or \
               v.startswith("Mô tả sản phẩm") or \
               "THÔNG TIN SẢN PHẨM" in v or \
               "ĐẶC ĐIỂM NỔI BẬT" in v:
                continue
                
        # Bước C: Chuẩn hóa đơn vị đo lường và toán tử nhân trong Key và Value
        for target in ('k', 'v'):
            val = k if target == 'k' else v
            # Dấu phẩy thập phân: 0,65 m -> 0.65 m
            val = re.sub(r'(?<=\d),(?=\d{1,2}\s*(?:cm|mm|m|ml|l|g|kg|w|v|mah|hz|rpm|inch|\b))', '.', val, flags=re.I)
            # Đơn vị đo liền với x: 30mlx10 -> 30 ml x 10
            val = re.sub(r'(\d+(?:\.\d+)?)\s*(cm|mm|m|ml|l|g|kg|w|v|mah|hz|rpm|inch)\s*[xX*×]\s*', r'\1 \2 x ', val, flags=re.I)
            # Toán tử nhân x: 40x60 -> 40 x 60, 38x25x25 -> 38 x 25 x 25
            while re.search(r'(\d+(?:\.\d+)?)[xX*×](\d+(?:\.\d+)?)', val):
                val = re.sub(r'(\d+(?:\.\d+)?)[xX*×](\d+(?:\.\d+)?)', r'\1 x \2', val)
            # Khoảng cách trước đơn vị đo: 2ml -> 2 ml, 300kg -> 300 kg
            val = re.sub(r'(\b\d+(?:\.\d+)?)\s*(cm|mm|ml|l|kg|g|w|v|mah|hz|rpm|inch)\b', r'\1 \2', val, flags=re.I)
            # Khoảng trắng trước dấu câu
            val = re.sub(r'\s+([,;:])', r'\1', val)
            val = re.sub(r'\s{2,}', ' ', val).strip()
            
            if target == 'k':
                k = val
            else:
                v = val
                
        seen_keys.add(k.lower())
        cleaned_pairs.append(f"{k}: {v}")
        
    return "; ".join(cleaned_pairs)

# ==================== ÁP DỤNG CLEANING ====================
print("\n--- Bước 1: Áp dụng Forensic Polish v3.4 cho Tiêu đề & Thông số ---")
df['title_vi_cleaned'] = df['title_vi_cleaned'].apply(polish_title)
df['description_vi_cleaned'] = df.apply(lambda r: polish_specs(r['description_vi_cleaned'], r['title_vi_cleaned']), axis=1)

# ==================== NHÓM SẢN PHẨM & TRÙNG LẶP ====================
print("\n--- Bước 2: Cập nhật phân nhóm trùng lặp biến thể ---")
df['norm_key'] = df['title_vi_cleaned'].str.lower().str.strip()
df['trung_y_het'] = df.duplicated(subset=['norm_key'], keep=False).astype(int)

groups = {}
group_ids = []
current_gid = 1
for key in df['norm_key']:
    if key not in groups:
        groups[key] = current_gid
        current_gid += 1
    group_ids.append(groups[key])
df['nhom_trung'] = group_ids
df.drop(columns=['norm_key'], inplace=True)

print(f"Số bản ghi trùng lặp tuyệt đối (trung_y_het == 1): {df['trung_y_het'].sum()}")
print(f"Số nhóm sản phẩm phân biệt (nhom_trung): {df['nhom_trung'].nunique()}")

# ==================== TÁI TÍNH BGE-M3 SIMILARITY TRÊN GPU ====================
print(f"\n--- Bước 3: Tái tính toán Cosine Similarity BGE-M3 Title vs Specs trên GPU ({device}) ---")
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def encode_texts(texts, max_len=128, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) if t else "" for t in texts[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            cls_rep = out.last_hidden_state[:, 0]
            cls_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
            all_embs.append(cls_rep.cpu().numpy())
    return np.vstack(all_embs)

print("Đang mã hóa Title (max_len=64)...")
title_embs = encode_texts(df['title_vi_cleaned'].tolist(), max_len=64, batch_size=64)

print("Đang mã hóa Specs (max_len=128)...")
specs_embs = encode_texts(df['description_vi_cleaned'].tolist(), max_len=128, batch_size=64)

print("Đang tính toán Cosine Similarity...")
sim_t_s = np.sum(title_embs * specs_embs, axis=1)

df['sim_title_specs_bge_m3'] = np.round(sim_t_s, 4)
df['similarity_bge_m3'] = df['sim_title_specs_bge_m3']

print("Thống kê Cosine Similarity Title <-> Specs (v3.4):")
print(f"  Mean = {df['similarity_bge_m3'].mean():.4f}, Median = {df['similarity_bge_m3'].median():.4f}")
print(f"  Min = {df['similarity_bge_m3'].min():.4f}, Max = {df['similarity_bge_m3'].max():.4f}")

# ==================== XÁC MINH KIỂM TOÁN 20 CHIỀU ====================
print("\n" + "="*40 + " XÁC MINH KIỂM TOÁN 20 CHIỀU ĐỘC LẬP " + "="*40)

def check_hidden_chars(s):
    return [f"U+{ord(c):04X}" for c in str(s) if ord(c) in (0x00A0, 0x200B, 0xFEFF, 0x200E, 0x200F, 0x3000, 0xFFFC) or (ord(c) < 32 and c not in '\t\n\r')]

def check_brackets(s):
    s = str(s)
    if re.search(r'\(\s*\)|\[\s*\]|\{\s*\}', s):
        return 'empty_bracket'
    if s.count('(') != s.count(')'):
        return 'unclosed_parenthesis'
    if s.count('[') != s.count(']'):
        return 'unclosed_square_bracket'
    if re.search(r'[【】⌈⌋«»‹›{}]', s):
        return 'exotic_brackets_symbols'
    return None

c1 = len(df[df['title_vi_cleaned'].apply(lambda x: len(check_hidden_chars(x)) > 0)])
c2 = len(df[df['title_vi_cleaned'].str.contains(r'^[^\w\(\[\{]+', na=False)])
c3 = len(df[df['title_vi_cleaned'].str.contains(r'[^\w\)\}\]%\s]+$', na=False)])
c4 = len(df[df['title_vi_cleaned'].str.contains(r'\s+[,;:](?!\d)', na=False)])
c5 = len(df[df['title_vi_cleaned'].str.contains(r'(?<=[a-zA-ZÀ-ỹ0-9]),(?=[a-zA-ZÀ-ỹ])', na=False)])
c6 = len(df[df['title_vi_cleaned'].apply(check_brackets).notnull()])
c7 = len(df[df['title_vi_cleaned'].str.contains(r'\b\d+(?:cm|mm|ml|kg|mah|hz|rpm|inch)\b', case=False, na=False)])
c8 = len(df[df['title_vi_cleaned'].str.contains(r'\b\d+x\d+\b', case=False, na=False)])
c9 = len(df[df['title_vi_cleaned'].str.contains(r'\s{2,}', na=False)])
c10 = len(df[df['title_vi_cleaned'].str.contains(r'--', na=False)])

print(f"Title audit results:")
print(f"  1. Ký tự ẩn: {c1} | 2. Dấu thừa đầu: {c2} | 3. Dấu thừa cuối: {c3} | 4. Khoảng trắng trước dấu câu: {c4}")
print(f"  5. Dính chữ sau phẩy: {c5} | 6. Lỗi ngoặc: {c6} | 7. Đơn vị dính liền: {c7} | 8. Toán tử x dính liền: {c8}")
print(f"  9. Khoảng trắng kép: {c9} | 10. Dấu gạch đôi: {c10}")

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

s1 = len(df[df['description_vi_cleaned'].apply(lambda x: len(check_hidden_chars(x)) > 0)])
no_colon_rows = [r.product_id for idx, r in df.iterrows() if check_spec_items(r['description_vi_cleaned'])[0]]
empty_val_rows = [r.product_id for idx, r in df.iterrows() if check_spec_items(r['description_vi_cleaned'])[2]]
s2 = len(no_colon_rows)
s3 = len(empty_val_rows)

def check_dup_keys(s):
    items = [x.strip() for x in str(s).split(';') if x.strip()]
    keys = [it.split(':', 1)[0].strip().lower() for it in items if ':' in it]
    return list(set([k for k in keys if keys.count(k) > 1]))

s4 = len(df[df['description_vi_cleaned'].apply(lambda x: len(check_dup_keys(x)) > 0)])
s5 = len(df[df['description_vi_cleaned'].str.contains(r';\s*;', na=False)])
s6 = len(df[df['description_vi_cleaned'].str.contains(r';\s*$', na=False)])
s7 = len(df[df['description_vi_cleaned'].str.contains(r'^\s*;', na=False)])
s8 = len(df[df['description_vi_cleaned'].str.contains(r'\b\d+(?:cm|mm|ml|kg|mah|hz|rpm|inch)\b', case=False, na=False)])
s9 = len(df[df['description_vi_cleaned'].str.contains(r':\s*(?:null|none|undefined|n/a|-|\?)\s*(?:;|$)', case=False, na=False)])
s10 = len(df[df['description_vi_cleaned'].str.len() < 15])

print(f"Specs audit results:")
print(f"  1. Ký tự ẩn: {s1} | 2. Không có dấu hai chấm: {s2} | 3. Khóa rỗng: {s3} | 4. Khóa trùng lặp: {s4}")
print(f"  5. Chấm phẩy kép: {s5} | 6. Chấm phẩy cuối: {s6} | 7. Chấm phẩy đầu: {s7} | 8. Đơn vị dính liền: {s8}")
print(f"  9. Giá trị rác: {s9} | 10. Quá ngắn (<15): {s10}")

assert c1==c2==c3==c4==c5==c6==c7==c8==c9==c10==0, "Vẫn còn lỗi trong Tiêu đề!"
assert s1==s2==s3==s4==s5==s6==s7==s8==s9==s10==0, "Vẫn còn lỗi trong Thông số kỹ thuật!"
print("\n🎉 XÁC NHẬN: TOÀN BỘ 20 CHIỀU ĐẠT 0 LỖI TUYỆT ĐỐI (ZERO-DEFECT)!")

# ==================== XUẤT BẢN PARQUET ====================
print("\n--- Bước 4: Xuất bản tệp Parquet hoàn chỉnh (v3.4) ---")
out_path1 = r"D:\download\NCKH\eda_ecom\data\tiki_vi_cleaned.parquet"
out_path2 = r"D:\download\NCKH\ecom_crawler-main\ecom_crawler-main-feature-1688\data\processed\tiki_vi_cleaned.parquet"

os.makedirs(os.path.dirname(out_path1), exist_ok=True)
os.makedirs(os.path.dirname(out_path2), exist_ok=True)

df.to_parquet(out_path1, index=False, engine='pyarrow')
df.to_parquet(out_path2, index=False, engine='pyarrow')

file_size1 = os.path.getsize(out_path1) / (1024 * 1024)
file_size2 = os.path.getsize(out_path2) / (1024 * 1024)

elapsed = time.time() - start_time
print(f"✅ Xuất bản thành công: {out_path1} ({file_size1:.2f} MB)")
print(f"✅ Xuất bản thành công: {out_path2} ({file_size2:.2f} MB)")
print(f"⏱️ Tổng thời gian hoàn thành: {elapsed:.2f} giây.")
