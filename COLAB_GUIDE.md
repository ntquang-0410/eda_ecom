# 🚀 Hướng dẫn chạy Bitext Filter (H3 Hybrid) trên Google Colab

## Bước 1: Clone repo + cài đặt

```bash
# SSH vào Colab, hoặc mở terminal trong Colab notebook
git clone https://github.com/ntquang-0410/eda_ecom.git
cd eda_ecom

# Cài dependencies
pip install pandas numpy pyarrow sentence-transformers python-dotenv
```

## Bước 2: Chạy toàn bộ pipeline

```bash
# Chạy trực tiếp — GPU auto-detect
python3 cado_bitext_filter.py
```

**Pipeline tự động làm:**
1. Load `data/raw/bilingual_zh_vi.parquet`
2. Preprocess: V1-V16 fixes + NFKC + CADO-VI (CPU, ~5s)
3. BGE-M3 cosine (GPU T4, ~2-3 phút cho 32K texts)
4. Lexical overlap (CPU, ~30s)
5. Hybrid score = 0.7×cosine + 0.3×lexical
6. Phân loại KEEP/DISCARD

## Bước 3: Kết quả

```bash
# Xem kết quả:
python3 -c "
import pandas as pd
df = pd.read_parquet('data/cleaned/bitext_scored.parquet')
print(df['final_label'].value_counts())
print(f'KEEP: {(df.final_label==\"KEEP\").sum():,}')
print(f'DISCARD: {(df.final_label==\"DISCARD\").sum():,}')
"

# File đầu ra:
ls -lh data/cleaned/bitext_scored.parquet   # Toàn bộ + scores
ls -lh csv/bitext_discard.csv               # Cặp bị loại
ls -lh csv/bitext_full_scores.csv           # Full scores để review
```

## Bước 4: Upload lên Hugging Face (tuỳ chọn)

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_xxx")  # Thay bằng token của bạn

api.upload_file("data/cleaned/bitext_scored.parquet",
    path_in_repo="data/cleaned/bitext_scored.parquet",
    repo_id="ntquang0410/zh-vie_ecom", repo_type="dataset")

api.upload_file("csv/bitext_discard.csv",
    path_in_repo="bitext_discard.csv",
    repo_id="ntquang0410/zh-vie_ecom", repo_type="dataset")
```

---

## 📐 Kiến trúc H3

```
cado_bitext_filter.py
├── Config              — toàn bộ tham số (thresholds, paths, model)
├── Preprocessor        — V1-V16 fixes + NFKC + CADO-VI
│   ├── fix_fullwidth()          — NFKC normalize
│   ├── remove_cjk_from_vi()     — Xóa CJK khỏi VI
│   ├── fix_cjk_brackets()       — 【】→ []
│   ├── trim_trailing_punct()    — Gọt dấu câu đuôi
│   ├── fix_decimal_comma()      — 4,3 → 4.3
│   └── clean_vi_text()          — CADO-VI
├── HybridScorer        — Cosine + Lexical → hybrid_score
│   ├── compute_cosine()         — BGE-M3 (GPU)
│   ├── extract_entities()       — Brand/số/đơn vị
│   ├── lexical_overlap()        — Jaccard entity overlap
│   ├── compute_hybrid()         — 0.7×cos + 0.3×lex
│   └── classify()               — KEEP/DISCARD
└── main()              — Pipeline hoàn chỉnh
```

## 🎯 Ngưỡng phân loại

| Điều kiện | Nhãn |
|:---|:---|
| cosine ≥ 0.55 AND hybrid ≥ 0.50 | `KEEP` (confidence: HIGH nếu cos≥0.75, còn lại MEDIUM) |
| cosine < 0.55 | `DISCARD` |

## 🔧 Tuỳ chỉnh nhanh

Mở `cado_bitext_filter.py`, sửa class `Config`:

```python
# Thay đổi ngưỡng:
cosine_discard: float = 0.50   # Hạ xuống → giữ nhiều hơn

# Tăng trọng số lexical:
lexical_weight: float = 0.50   # Từ 0.30 lên 0.50

# Dùng model nhẹ hơn nếu không có GPU:
bge_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
bge_batch: int = 512            # Tăng batch size
```