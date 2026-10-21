# E-commerce Song Ngữ Trung–Việt (1688)

Pipeline xử lý dữ liệu sản phẩm thương mại điện tử Trung Quốc → Việt Nam:

1. **CADO** — làm sạch từ tiếng Anh dư thừa trong `title_zh` / `description_zh` và
   `title_vi` / `description_vi` mà vẫn giữ brand, model, đơn vị, từ mượn.
2. **Bitext Filter** — preprocess, chấm điểm và lọc các cặp câu Trung–Việt
   trước khi đưa vào huấn luyện dịch máy.

Repo cũng chứa notebook EDA (`eda/eda-bilingual_zh_vi.ipynb`), whitelist,
CC-CEDICT offline và các artifact CSV/Parquet phục vụ audit.

> **Trạng thái kiểm chứng:** CADO T1–T4 + CADO-VI đã chạy full trên 16.348
> dòng. T5 (bge-m3) và cosine bitext cần tải model ~2.2GB nên **chưa chạy**
> trong lần refactor này; artifact cosine hiện có là từ lần chạy trước.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.10. Các thư viện chính: `pandas`, `numpy`, `pyarrow`,
`sentence-transformers`, `torch`, `huggingface_hub`, `python-dotenv`.

CC-CEDICT được dùng offline tại `dictionary/cedict_ts.u8` (125.207 entry).
Nếu thiếu file, tải bản `cedict_1_0_ts_utf-8_mdbg.zip` từ
[mdbg.net](https://www.mdbg.net/chinese/dictionary?page=cc-cedict) rồi giải nén
vào `dictionary/`.

---

## Chạy nhanh

```bash
# 1) Làm sạch tiếng Anh (T0–T4) + CADO-VI, ghi đè artifact CADO
python run_pipeline.py cado

# 2) Chỉ preprocess bitext (không cần model)
python run_pipeline.py bitext --preprocess-only

# 3) Chấm điểm đầy đủ (cần bge-m3 ~2.2GB)
python run_pipeline.py bitext

# 4) Chạy cả hai
python run_pipeline.py all

# 5) Thống kê artifact
python run_pipeline.py stats

# 6) Kiểm thử nhanh
python -m unittest discover -s tests -v
```

Các tuỳ chọn hữu ích:

```bash
# Bật T5 semantic fallback (bge-m3), quyết định theo từng lần xuất hiện
python run_pipeline.py cado --tier5 --device cpu

# Làm sạch cả description_zh (xem cảnh báo bên dưới)
python run_pipeline.py cado --columns title_zh,description_zh

# Chỉ chạy CADO Trung, bỏ CADO-VI
python run_pipeline.py cado --no-vi

# Thử nhanh 200 dòng, ghi ra file tạm
python run_pipeline.py cado --limit 200 --output /tmp/cado_test.parquet

# Bitext dùng hybrid threshold thay vì chỉ cosine
python run_pipeline.py bitext --label-mode hybrid
```

Vẫn có thể chạy từng script cũ (đã được refactor thành wrapper mỏng):

```bash
python cado_layer/cado_tier0_extract.py     # thống kê từ Latin
python cado_layer/cado_tier2_pattern.py     # preview whitelist/pattern
python cado_layer/cado_tier3_loanword.py    # auto loanword + CEDICT
python cado_layer/cado_tier4_cedict.py      # chỉ CEDICT
python cado_layer/cado_tier5_bge.py         # T0–T5
python cado_layer/cado_tier2_entity.py      # 7 entity signals
python bitext_mining/preprocess_bitext.py   # preprocess
python bitext_mining/cado_bitext_filter.py  # full filter
```

---

## Pipeline

```
data/raw/bilingual_zh_vi.parquet (16.348 × 30)
                │
                ▼
┌──────────────────────────────────────────────┐
│ CADO (cado_core/cado.py)                     │
│  T1 whitelist → T2 pattern → T3 loanword     │
│  → T4 CC-CEDICT overlap → T5 bge-m3 (opt-in) │
│  + CADO-VI (cado_core/cado_vi.py)            │
└──────────────────────────────────────────────┘
                │
                ▼
data/cleaned/bilingual_zh_vi_cado.parquet (16.348 × 33)
  + title_zh_cleaned, title_vi_cleaned, desc_vi_cleaned
                │
                ▼
┌──────────────────────────────────────────────┐
│ Bitext Filter (cado_core/bitext.py)          │
│  V1–V16 preprocess → flags → cosine bge-m3   │
│  → lexical overlap → hybrid → label          │
└──────────────────────────────────────────────┘
                │
                ▼
data/cleaned/bitext_scored.parquet
```

### CADO — thứ tự tầng là bất biến

| Tầng | Tên | Quyết định | Nguồn |
|---|---|---|---|
| T1 | Whitelist | **Giữ** brand, tech term, đơn vị, từ mượn | `csv/whitelist.csv` (289 từ unique) |
| T2 | Pattern | **Giữ** model code, ALL-CAPS ≤5, CamelCase, từ ≤2 ký tự | regex |
| T3 | Auto loanword | **Giữ** từ lặp ≥3 lần nhưng overlap CEDICT <20% | `cado_core/rules.py` |
| T4 | CC-CEDICT | **Xoá** nếu bản dịch tiếng Trung xuất hiện trong cùng văn bản | `dictionary/cedict_ts.u8` |
| T5 | bge-m3 | **Xoá** nếu xoá từ mà embedding câu gần như không đổi (>0.98) | `BAAI/bge-m3` (opt-in) |

Điểm quan trọng:

- **Boundary an toàn với CJK:** không dùng `\b` (Python coi chữ Hán là `\w`).
  Dùng lookbehind/lookahead chỉ loại trừ chữ Latin, nên `bags` bị xoá trong
  `女包bags大容量包包` (nhờ `包包`) nhưng **không** bị xoá trong `handbags`.
- **CC-CEDICT index theo giản thể** (cột 2), không phải phồn thể — dữ liệu
  1688 là giản thể.
- **Newline được giữ nguyên**, an toàn khi xử lý `description_zh`.
- **T5 quyết định theo từng occurrence**, không áp dụng toàn cục theo word như
  bản cũ; không cắt mẫu âm thầm (`--max-pairs` chỉ dùng khi chủ động).

### Bitext scoring

- **Cosine** giữa `title_zh_cleaned` và `title_vi_cleaned` bằng bge-m3.
- **Lexical overlap** = Jaccard trên brand/model/số/đơn vị. Nếu một câu không
  có entity nào → `NaN` (không có bằng chứng), không gán 1.0.
- **Hybrid** = `0.7 * cosine + 0.3 * lexical`; nếu lexical `NaN` thì dùng
  cosine.
- `--label-mode cosine` (mặc định, tương thích artifact cũ): DISCARD nếu
  cosine < 0.55.
- `--label-mode hybrid`: DISCARD nếu cosine < 0.55 **hoặc** hybrid < 0.50.
  Ngưỡng 0.50 khá nghiêm (trên cosine cũ sẽ loại ~5.492 dòng), nên chỉ dùng
  sau khi tune.

---

## Cấu trúc dự án

```
.
├── cado_core/                 # Thư viện lõi dùng chung
│   ├── config.py              # Đường dẫn, regex, hyperparameter
│   ├── text.py                # NFC, trích/xoá từ Latin an toàn CJK
│   ├── dictionary.py          # Whitelist + CC-CEDICT + lemmatize
│   ├── rules.py               # Pattern, loanword, phân loại
│   ├── cado.py                # Pipeline CADO T0–T5
│   ├── cado_vi.py             # CADO-VI (map EN→VI)
│   └── bitext.py              # Preprocess + cosine/lexical/hybrid
├── cado_layer/                # CLI wrapper từng tầng (giữ tương thích)
├── bitext_mining/             # CLI preprocess + filter
├── dictionary/cedict_ts.u8    # CC-CEDICT offline
├── csv/
│   ├── whitelist.csv          # 323 dòng / 289 từ unique
│   ├── vi_english_map.csv     # Map EN→VI cho CADO-VI (16 từ)
│   ├── cado/                  # Report CADO
│   ├── bitext_mining/         # Report bitext
│   ├── changed_rows/          # Artifact legacy
│   └── english_remover/       # Inventory từ tiếng Anh
├── data/
│   ├── raw/                   # 16.348 × 30
│   └── cleaned/               # CADO / bitext / Tiki
├── eda/eda-bilingual_zh_vi.ipynb
├── run_pipeline.py            # CLI thống nhất
├── upload_to_hf.py            # Upload dataset lên Hugging Face
├── tests/test_core.py         # Smoke test boundary/CEDICT/pipeline
├── README.md
└── Walkthrough.md             # Phân tích chi tiết + lịch sử refactor
```

---

## Artifact hiện có

| File | Shape | Ghi chú |
|---|---:|---|
| `data/raw/bilingual_zh_vi.parquet` | 16.348 × 30 | Dữ liệu gốc 1688 |
| `data/cleaned/bilingual_zh_vi_cado.parquet` | 16.348 × 33 | **Đã regenerate**: thêm `title_zh_cleaned`, `title_vi_cleaned`, `desc_vi_cleaned` |
| `data/cleaned/bilingual_zh_vi_bitext_ready.parquet` | 16.346 × 44 | Legacy, có `signal_S1..S7`, `entity_score`, `tier2_label` |
| `data/cleaned/bitext_scored.parquet` | 16.346 × 39 | Legacy: KEEP 16.201 / DISCARD 145, cosine mean 0.7021 |
| `data/cleaned/tiki_vi_cado.parquet` | 13.200 × 37 | Artifact Tiki; **không có raw/source script trong repo** |

Report mới:

- `csv/cado/cado_word_report.csv` — quyết định theo từ.
- `csv/cado/cado_occurrences.csv` — từng lần xuất hiện (9.721 dòng).
- `csv/cado/cado_changed_rows.csv` — 286 dòng thay đổi (ZH title + VI).
- `csv/cado/cado_tier3_loanword_report.csv` — 478 ứng viên, 41 từ mượn.
- `csv/cado/cado_vi_occurrences.csv` — 202 occurrence CADO-VI.
- `csv/cado/cado_tier5_similarity.csv` — chỉ có khi chạy `--tier5`.

---

## Kết quả kiểm chứng (T0–T4)

| Chỉ số | Giá trị |
|---|---:|
| Tổng occurrence trong `title_zh` | 9.721 |
| Từ bị xoá | 183 |
| Sản phẩm đổi `title_zh_cleaned` | 145 |
| Auto loanword được giữ | 41 |
| Ứng viên T3 | 478 |
| CADO-VI đổi `title_vi_cleaned` | 101 |
| CADO-VI đổi `desc_vi_cleaned` | 73 |
| Bitext preprocess còn lại | 16.346 (loại 2 CJK copy) |

So với bản legacy, số từ bị xoá ở title trùng với phân tích trước đây (183).
CADO-VI thay đổi nhiều hơn legacy (101/73 so với 127/68) vì bản mới xử lý đúng
ranh giới dấu câu/chữ số, đồng thời có guard không xoá từ nằm trong cụm brand
(`MAKI BAG`, `Bio-Mask`).

---

## Lưu ý & giới hạn

- **`description_zh` là opt-in.** Chạy `--columns title_zh,description_zh` sẽ
  xoá nhiều hơn đáng kể (985 occurrence / 372 dòng) và có thể phá model code
  kiểu `Type-c`. Chỉ dùng sau khi review `cado_changed_rows.csv`.
- **T5/cosine chưa chạy lại** vì cần tải bge-m3. Artifact cosine hiện tại là
  legacy; chạy `python run_pipeline.py bitext` để tái tạo.
- `--preprocess-only` tạo lại `bitext_ready` từ CADO và **không** tạo các cột
  `signal_S1..S7`; chạy `python cado_layer/cado_tier2_entity.py` nếu cần.
- **Tiki** chỉ có artifact cleaned, không có raw/source script. Không thể tái
  tạo từ repo hiện tại.
- `csv/cado/zh_cado_changed_rows.csv` (193 dòng), `csv/changed_rows/*`,
  `csv/cado/tiki_cado_changed_rows.csv` là artifact legacy, được giữ để đối
  chiếu; schema/count có thể khác pipeline mới.
- Whitelist có 323 dòng nhưng 289 từ unique (lowercase). `match_mode` hiện
  được xử lý như so khớp nguyên từ, không phân biệt hoa/thường.
- `entity_score` mặc định không dùng embedding (S4b tắt), nên điểm thấp hơn
  artifact cũ (0.65 vs 0.86). Bật `--embedding` nếu cần tái tạo.

---

## Upload lên Hugging Face

```bash
cp .env.example .env   # nếu có; điền HF_TOKEN và HF_REPO_ID
python upload_to_hf.py --file data/cleaned/bilingual_zh_vi_cado.parquet
```

Mặc định script chọn file cleaned đầu tiên tồn tại trong danh sách
`bilingual_zh_vi_cado.parquet` → `bitext_scored.parquet` →
`bilingual_zh_vi_bitext_ready.parquet` → raw. Đường dẫn trên repo được giữ
tương đối so với gốc dự án.

---

## Tài liệu

- [`Walkthrough.md`](Walkthrough.md) — phân tích dữ liệu, kiến trúc CADO, chi
  tiết từng tầng, bug fix và lộ trình.
- [`eda/eda-bilingual_zh_vi.ipynb`](eda/eda-bilingual_zh_vi.ipynb) — notebook
  EDA gốc.
