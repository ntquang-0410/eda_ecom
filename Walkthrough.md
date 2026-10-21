# 📖 Walkthrough — Dữ Liệu & Pipeline Song Ngữ Trung–Việt (1688)

Tài liệu này mô tả dữ liệu, các vấn đề phát hiện trong EDA, kiến trúc CADO,
pipeline bitext, những bug đã sửa trong lần refactor 2026-10-21 và cách tái
tạo artifact.

> Đọc nhanh: nếu chỉ muốn chạy pipeline, xem mục **12. Runbook**.
> Nếu muốn hiểu vì sao một từ bị xoá, xem mục **6–8**.

---

## Mục Lục

1. [Tổng quan](#1-tổng-quan)
2. [Dữ liệu & schema](#2-dữ-liệu--schema)
3. [Kết quả EDA](#3-kết-quả-eda)
4. [Vấn đề từ tiếng Anh](#4-vấn-đề-từ-tiếng-anh)
5. [Pipeline legacy & lý do thay thế](#5-pipeline-legacy--lý-do-thay-thế)
6. [Kiến trúc CADO](#6-kiến-trúc-cado)
7. [Chi tiết từng tầng](#7-chi-tiết-từng-tầng)
8. [CADO-VI](#8-cado-vi)
9. [Bitext Filter](#9-bitext-filter)
10. [Refactor & bug fixes](#10-refactor--bug-fixes)
11. [Kết quả kiểm chứng](#11-kết-quả-kiểm-chứng)
12. [Runbook](#12-runbook)
13. [Artifact & CSV](#13-artifact--csv)
14. [Giới hạn & roadmap](#14-giới-hạn--roadmap)
15. [Glossary](#15-glossary)

---

## 1. Tổng quan

Mục tiêu của repo:

- Làm sạch các từ tiếng Anh bị chèn vào tiêu đề/mô tả tiếng Trung và tiếng
  Việt của sản phẩm 1688, **không** xoá brand/model/đơn vị/từ mượn.
- Tạo tập cặp câu Trung–Việt sạch để huấn luyện dịch máy.
- Giữ khả năng audit: mỗi quyết định xoá/giữ đều có lý do và có thể xuất CSV.

Hai thành phần chính:

| Thành phần | Vai trò | Code |
|---|---|---|
| **CADO** | Xoá từ tiếng Anh dư thừa trong ZH/VI | `cado_core/cado.py`, `cado_core/cado_vi.py` |
| **Bitext Filter** | Preprocess + chấm điểm cặp câu | `cado_core/bitext.py` |

---

## 2. Dữ liệu & schema

### 2.1 Nguồn

- 1688.com (Trung Quốc), sản phẩm được crawl và dịch sang tiếng Việt.
- File gốc: `data/raw/bilingual_zh_vi.parquet` — **16.348 dòng × 30 cột**.
- Ngoài ra có artifact Tiki `data/cleaned/tiki_vi_cado.parquet`
  (**13.200 × 38**) nhưng repo **không có raw/source script** cho Tiki.

### 2.2 Nhóm cột

| Nhóm | Cột tiêu biểu |
|---|---|
| Định danh | `product_id`, `url`, `source_site`, `crawl_time` |
| Nội dung ZH | `title_zh`, `description_zh`, `attributes_zh`, `description_extra_zh` |
| Nội dung VI | `title_vi`, `description_vi`, `attributes_vi` |
| Phân loại | `category`, `category_1688`, `category_id_1688`, `keyword` |
| Thương mại | `price`, `sales`, `province`, `city`, `biz_type`, `shop`, `is_ad` |
| Chất lượng | `has_vi`, `n_attributes_*`, `description_images` |

### 2.3 Cột sau CADO

`data/cleaned/bilingual_zh_vi_cado.parquet` = 16.348 × 33:

- Giữ nguyên 30 cột gốc.
- Thêm `title_zh_cleaned`, `title_vi_cleaned`, `desc_vi_cleaned`.
- Nếu chạy `--columns title_zh,description_zh` sẽ thêm
  `description_zh_cleaned` (opt-in).

---

## 3. Kết quả EDA

Notebook: `eda/eda-bilingual_zh_vi.ipynb`.

### 3.1 Chỉ số chính

| Chỉ số | Giá trị |
|---|---:|
| Số dòng | 16.348 |
| Số cột | 30 |
| `title_zh` có từ Latin | 6.252 (38,2%) |
| `description_zh` có từ Latin | 15.958 (97,6%) |
| Từ Latin unique trong title | 1.183 |
| Từ Latin unique trong description | 4.483 |
| Union title + description | 4.970 |
| `title_zh == title_vi` | 2 |
| Title trùng lặp | 485 (~2,97%) |
| Cặp `(title_zh, title_vi)` trùng | 211 |
| Null `biz_type` | ~16,5% |
| Null `sales` | ~3,3% |

### 3.2 Nhận xét

- Tiếng Anh xuất hiện chủ yếu trong `description_zh` (gần như mọi dòng), nhưng
  title mới là nơi ảnh hưởng trực tiếp đến chất lượng cặp câu.
- Phần lớn từ Latin trong title là **brand/model/đơn vị** (Nike, USB, 5cm,
  Type-C...), chỉ một phần là từ tiếng Anh thật sự dư thừa (bags, shoes, mask).
- Vì vậy không thể xoá mọi từ Latin; cần phân tầng và whitelist.
- 2 dòng `title_zh == title_vi` bị loại ở bitext V1.
- 29 cặp câu nghi lệch căn chỉnh (theo EDA) — chưa có bước xử lý riêng.

---

## 4. Vấn đề từ tiếng Anh

### 4.1 Ví dụ điển hình

```text
女包bags大容量包包       → 女包大容量包包      (xoá "bags" nhờ "包包")
2021women bags外贸包包   → 2021women 外贸包包  (xoá "bags")
handbags                → giữ nguyên "handbags" (không phải "bags" độc lập)
BANGE双肩包backpack     → BANGE双肩包        (xoá "backpack" nhờ "双肩包")
```

### 4.2 Ba nhóm ngữ nghĩa

| Nhóm | Ví dụ | Hành động |
|---|---|---|
| 🟢 Brand / model / SKU | Nike, USB, iPhone14, 5cm | **Giữ** |
| 🟡 Từ mượn / phong cách | Carplay, modelY, Bluetooth | **Giữ** |
| 🔴 Từ tiếng Anh dư thừa | bags, shoes, mask, clothes | **Xoá nếu đã có bản dịch** |

### 4.3 Sai lầm của cách tiếp cận cũ

Bản legacy dùng:

```python
re.sub(rf"\b{word}\b", "", text)
```

Nhưng Python coi chữ Hán là `\w`, nên `\b` không hoạt động đúng:

- `包bags男` có thể không khớp.
- Hoặc tệ hơn, `bags` khớp bên trong `handbags` nếu boundary bị nới lỏng.
- Từ dính dấu câu (`bags,`) hoặc chữ số (`bags2024`) bị bỏ sót.

Bản mới dùng:

```python
rf"(?<![A-Za-z\u00C0-\u024F]){re.escape(word)}(?![A-Za-z\u00C0-\u024F])"
```

Chỉ **chữ cái Latin** bị loại khỏi boundary, nên CJK, dấu câu và chữ số đều là
ranh giới hợp lệ.

---

## 5. Pipeline legacy & lý do thay thế

### 5.1 Legacy 3 tầng

1. **Whitelist hardcoded** — một set trong code.
2. **Pattern regex** — model code, ALL-CAPS, CamelCase, từ ngắn.
3. **WordNet semantic** — phân loại từ dư thừa theo nghĩa.

Hạn chế:

- WordNet không hiểu ngữ cảnh TMĐT Trung–Việt.
- Whitelist hardcoded khó bảo trì.
- Không có bằng chứng “bản dịch đã xuất hiện trong câu”.
- Xoá theo word toàn cục, dễ xoá oan.
- Regex boundary sai với CJK.

### 5.2 CADO thay thế

CADO = **Context-Aware Deletion of English**, 5 tầng:

```
T1 whitelist → T2 pattern → T3 auto loanword → T4 CEDICT overlap → T5 bge-m3
```

Nguyên tắc:

- Tầng trước **giữ** thì tầng sau không được xoá.
- T4 chỉ xoá khi bản dịch tiếng Trung của từ xuất hiện **trong chính văn bản**.
- T5 chỉ xoá khi embedding câu gần như không đổi.
- Mọi quyết định đều ghi `action`, `reason`, `tier`.

---

## 6. Kiến trúc CADO

### 6.1 Sơ đồ

```
DataFrame thô
   │
   ├─ Chuẩn hoá NFC, bỏ zero-width, NBSP
   ├─ T3: gom ứng viên trên title_zh
   │      (bỏ qua từ đã thuộc T1/T2)
   │
   ├─ Với từng occurrence:
   │     T1 whitelist      → keep
   │     T2 pattern        → keep
   │     T3 loanword       → keep
   │     T4 CEDICT overlap → remove / keep
   │     (T5 nếu bật)      → remove / keep
   │
   ├─ Ghi nhận removed words theo (row, column)
   ├─ Tạo cột *_cleaned
   └─ Report: occurrences, word report, loanword report, T5 report
```

### 6.2 API

```python
from cado_core.cado import CadoConfig, run_cado

result = run_cado(
    df,
    CadoConfig(
        columns=("title_zh",),
        max_tier=4,
        use_loanwords=True,
    ),
)

result.df                    # DataFrame + cột cleaned
result.occurrences           # từng lần xuất hiện
result.loanword_report       # T3
result.stats                 # dict tổng hợp
```

### 6.3 Cấu hình mặc định

| Tham số | Giá trị | Ghi chú |
|---|---:|---|
| `columns` | `("title_zh",)` | Description là opt-in |
| `max_tier` | 4 | T5 phải bật rõ |
| `use_loanwords` | True | T3 |
| `min_freq` | 3 | T3 |
| `overlap_threshold` | 0.20 | T3 |
| `tier5_threshold` | 0.98 | T5 |
| `tier5_max_pairs` | None | Không cắt âm thầm |

---

## 7. Chi tiết từng tầng

### T1 — Whitelist

File: `csv/whitelist.csv` (323 dòng, **289 từ unique** lowercase).

| Category | Số dòng |
|---|---:|
| `tech_abbreviation` | 110 |
| `keep_exception` | 105 |
| `brand` | 53 |
| `unit` | 31 |
| `loanword` | 24 |

Hành vi: nếu `word.lower()` có trong whitelist → **giữ**.

Lưu ý: `match_mode` (`exact` / `ignore_case`) hiện được xử lý như nhau vì mọi
so sánh đều lowercase. Đây là điểm cần cải thiện nếu muốn phân biệt chính xác
hoa/thường.

### T2 — Pattern

```python
if re.search(r"\d", word):        return "model_code"          # iPhone14, 5cm
if re.match(r"^[A-Z]{1,5}$", w):  return "short_abbreviation"  # USB, ABS
if re.match(r"^[A-Z][a-z]+[A-Z]", w): return "camelcase_brand" # MagSafe
if len(word) <= 2:                return "too_short"           # AI, UV
```

### T3 — Auto loanword

Mục tiêu: giữ các từ mượn lặp nhiều lần nhưng hiếm khi có bản dịch tiếng
Trung trong câu.

- Ứng viên: từ Latin chưa bị T1/T2 quyết định.
- Với mỗi từ có `count >= 3`:
  - Tính `overlap_rate` = số occurrence có bản dịch CEDICT trong câu / tổng.
  - Nếu `< 0.20` → từ mượn → **giữ**.
- Kết quả trên full data: **478 ứng viên**, **41 từ mượn** được giữ.

Ví dụ: `carplay` xuất hiện 3 lần, không lần nào có “carplay” hoặc bản dịch
tương ứng trong câu Trung → giữ.

### T4 — CC-CEDICT overlap

Đây là tầng quan trọng nhất.

1. Lemma hóa từ tiếng Anh:
   - `bags → bag`
   - `dresses → dress`
   - `shoes → shoe`
   - `running → run`
2. Tra CC-CEDICT: English → danh sách headword **giản thể**.
3. Nếu bất kỳ headword nào (≥2 ký tự Hán) xuất hiện trong văn bản → **xoá**.

Ví dụ:

```text
bags  → 包包 / 包 / 手提包 ...
"外贸包包bags" có "包包" → xoá "bags"
"双肩包bags" không có headword khớp trực tiếp → giữ "bags"
```

**Bug đã sửa:** bản cũ index theo **phồn thể** (`m.group(1)`), trong khi dữ
liệu 1688 là **giản thể**. Điều này làm nhiều bản dịch không bao giờ khớp
(ví dụ `书包` vs `書包`). Bản mới dùng cột simplified (`m.group(2)`).

### T5 — bge-m3 semantic fallback (opt-in)

- Chỉ xử lý các occurrence còn `no_cedict_entry`.
- Với mỗi cặp `(text, word)`:
  - `text_keep` = câu gốc.
  - `text_remove` = câu sau khi xoá word.
  - Encode cả hai bằng `BAAI/bge-m3`.
  - Nếu cosine > 0.98 → từ dư thừa → xoá.
- Quyết định **theo từng occurrence**, không theo word toàn cục.
- Không cắt mẫu âm thầm; `--max-pairs` chỉ dùng khi chủ động.

**Bug đã sửa:** bản cũ quyết định theo word (`remove_words` toàn cục), nên một
lần xuất hiện “dư thừa” khiến từ bị xoá ở mọi ngữ cảnh.

---

## 8. CADO-VI

Mục tiêu: xoá từ tiếng Anh dư thừa trong `title_vi` / `description_vi` khi
**bản dịch tiếng Việt đã xuất hiện trong chính văn bản**.

Mapping: `csv/vi_english_map.csv` (16 từ).

Ví dụ:

```text
"Túi xách nữ bags"        → "Túi xách nữ"      (có "túi xách")
"Phụ kiện bags"           → giữ "bags"          (không có "túi"/"ba lô")
"Bio-Mask Mặt nạ..."      → giữ "Bio-Mask"      (guard cụm brand)
"Thương hiệu: MAKI BAG"   → giữ "MAKI BAG"      (guard cụm brand)
"Túi xách nữ bags2024"    → "Túi xách nữ 2024"  (digit là boundary)
```

### Guard cụm Latin

Không xoá từ nếu nó nằm cạnh một token Latin viết hoa (brand nhiều từ):

- `MAKI BAG` → `MAKI` viết hoa → giữ `BAG`.
- `Bio-Mask` → `Bio` viết hoa → giữ `Mask`.

### Kết quả

- `title_vi_cleaned`: **101** dòng đổi.
- `desc_vi_cleaned`: **73** dòng đổi.
- 202 occurrence bị xoá.

Legacy cho 127/68; chênh lệch do bản mới xử lý đúng dấu câu/chữ số và có guard
brand. Các thay đổi được lưu tại `csv/cado/cado_vi_occurrences.csv`.

---

## 9. Bitext Filter

### 9.1 Preprocess

Các bước chính (V1–V16):

| Bước | Nội dung |
|---|---|
| V1 | Loại dòng `title_zh == title_vi` và chứa CJK |
| V2 | NFC/NFKC, bỏ zero-width, NBSP |
| V3 | Bỏ CJK còn sót trong cột VI |
| V4/V5 | Bỏ dấu câu cuối, sửa `,` thập phân |
| V10 | Chuẩn hoá ngoặc `【】`, `丨` |
| V16 | `3*5` → `3x5` |

Điểm mới:

- Dùng trực tiếp `title_zh_cleaned` / `title_vi_cleaned` do CADO tạo.
- Nếu input chưa có cột cleaned, tự tạo bản copy từ cột gốc.
- **Giữ newline** (không collapse `\s+`) để không phá format description.

Kết quả full:

| Chỉ số | Giá trị |
|---|---:|
| Input | 16.348 |
| Loại V1 | 2 |
| Còn lại | 16.346 |
| `flag_stuffing` | 1.119 |
| `flag_truncated` | 1.566 |
| `flag_bloated` | 0 |

`flag_truncated` dùng tỉ lệ **số token VI / số ký tự Hán** (ngưỡng 0.8), thay
cho tỉ lệ độ dài chuỗi của bản cũ. Cách mới phản ánh đúng việc bản dịch bị
cắt cụt; bản cũ chỉ flag 4 dòng vì so sánh ký tự.

### 9.2 Cosine

- Model: `BAAI/bge-m3`, embedding 1024 chiều, normalized L2.
- Encode `title_zh_cleaned` và `title_vi_cleaned`, cosine = dot product.
- Artifact legacy: mean 0.7021, median 0.7053, std ~0.08.

### 9.3 Lexical overlap

Jaccard trên các entity:

- Brand/model/tech term từ whitelist.
- Số Ả Rập.
- Số viết bằng chữ Hán (đã parse đúng: `三百` → 300, không còn tách
  `3` + `00`).
- Đơn vị (`寸` → `inch`, `斤` → `kg`...).

Nếu **không có entity nào** → `NaN` (không có bằng chứng), không gán 1.0.
Khi tính hybrid, `NaN` được thay bằng chỉ dùng cosine.

### 9.4 Hybrid & label

```text
hybrid = 0.7 * cosine + 0.3 * lexical
```

- `--label-mode cosine` (mặc định):
  - `DISCARD` nếu cosine < 0.55.
  - Tương thích artifact cũ: 16.201 KEEP / 145 DISCARD.
- `--label-mode hybrid`:
  - `DISCARD` nếu cosine < 0.55 **hoặc** hybrid < 0.50.
  - Trên cosine cũ: 10.854 KEEP / 5.492 DISCARD (66,4%).
  - Ngưỡng 0.50 khá nghiêm; chỉ dùng sau khi tune.

`confidence` được gán cho **mọi** dòng:

- `HIGH` nếu cosine ≥ 0.75.
- `MEDIUM` nếu ≥ 0.55.
- `LOW` nếu < 0.55.

**Bug đã sửa:** bản cũ để `confidence = NaN` cho các dòng DISCARD và không
dùng `hybrid_keep`.

### 9.5 Entity signals (tuỳ chọn)

7 tín hiệu S1–S7, weights:

| Signal | Weight | Nội dung |
|---|---:|---|
| S1 | 0.15 | Brand/model khớp |
| S2 | 0.10 | Số khớp |
| S3 | 0.10 | Đơn vị khớp |
| S4 | 0.40 | CEDICT bigram / embedding |
| S5 | 0.10 | Tỉ lệ độ dài |
| S6 | 0.05 | Không còn CJK trong VI |
| S7 | 0.10 | Không lặp token |

- `entity_score >= 0.60` → KEEP; `>= 0.40` → REVIEW; còn lại DISCARD.
- Mặc định **không** dùng embedding (S4b tắt) → điểm thấp hơn artifact cũ.
- `--embedding` bật MiniLM cho S4b.

---

## 10. Refactor & bug fixes

### 10.1 Vì sao refactor

Trước đây logic bị copy-paste ở 6 script CADO + 2 script bitext:

- Mỗi nơi tự parse CC-CEDICT.
- Mỗi nơi tự load whitelist.
- Mỗi nơi tự viết regex boundary.
- Sửa bug phải sửa 6 lần, dễ lệch hành vi.

### 10.2 Cấu trúc mới

```
cado_core/
  config.py      đường dẫn, regex, hyperparameter
  text.py        normalize, extract, remove_words
  dictionary.py  Whitelist, Cedict, lemmatize, Overlap
  rules.py       pattern, loanword, classify_occurrence
  cado.py        run_cado (T0–T5)
  cado_vi.py     run_cado_vi
  bitext.py      preprocess, cosine, lexical, hybrid, entity signals
```

Các script trong `cado_layer/` và `bitext_mining/` trở thành **wrapper mỏng**,
giữ nguyên entry point cũ.

### 10.3 Bảng bug fix

| # | Bug | Trước | Sau |
|---|---|---|---|
| 1 | Sai đường dẫn CEDICT | `cedict_ts.u8` | `dictionary/cedict_ts.u8` |
| 2 | Index CEDICT phồn thể | `m.group(1)` | `m.group(2)` (giản thể) |
| 3 | `\b` với CJK | `\bbags\b` | lookaround chỉ loại trừ chữ Latin |
| 4 | `np` NameError | import trong `main()` | import module-level |
| 5 | Ghi đè input | `df.to_parquet(input)` | output riêng |
| 6 | T5 rò rỉ toàn cục | xoá theo word | quyết định từng occurrence |
| 7 | T5 cắt mẫu âm thầm | `MAX=500` | `None`, chỉ giới hạn khi chủ động |
| 8 | `hybrid_keep` không dùng | chỉ cosine | `--label-mode hybrid` |
| 9 | Lexical neutral = 1.0 | thổi phồng điểm | `NaN` + fallback cosine |
| 10 | Map VI hardcoded | 14 từ trong code | `csv/vi_english_map.csv` |
| 11 | Mất dòng V1 | không lưu | `bitext_discarded_rows.csv` |
| 12 | Ratio không nhất quán | 8.0 / 6.0 | token/CJK thống nhất |
| 13 | Confidence NaN | chỉ gán KEEP | gán mọi dòng |
| 14 | `tier2_label` vô dụng | chỉ gán khi có tier1 | luôn gán theo score |
| 15 | Upload default sai | file không tồn tại | chọn file existing |
| 16 | Requirements thiếu | thiếu pandas/pyarrow | bổ sung, bỏ FlagEmbedding bắt buộc |
| 17 | Docs lỗi thời | README nói `similarity_bge_m3.py` | cập nhật theo code thật |

### 10.4 Hành vi thay đổi có chủ ý

- CADO-VI xoá nhiều hơn legacy (101/73 vs 127/68) do boundary đúng và guard
  brand. Đây là thay đổi có chủ ý, có report audit.
- `flag_truncated` tăng từ 4 lên 1.566 do đổi cách đo.
- `entity_score` mặc định thấp hơn do tắt S4b embedding.
- `description_zh` vẫn là opt-in vì dễ xoá oan model code (`Type-c`).

---

## 11. Kết quả kiểm chứng

### 11.1 CADO full (T0–T4)

```bash
python run_pipeline.py cado
```

```text
Tổng occurrence: 9.721
keep: 9.538 | remove: 183
Sản phẩm đổi title_zh_cleaned: 145
T3: 478 ứng viên → 41 từ mượn
CADO-VI: title 101 | desc 73
```

Top từ bị xoá:

| Từ | Số lần | Bản dịch khớp |
|---|---:|---|
| bags | 54 | 包包 |
| bag | 24 | 单肩包 |
| mask | 15 | 面膜 |
| shoes | 15 | 休闲鞋 |
| Facial | 11 | 面膜 |
| handbag | 4 | 包包 |
| sneakers | 4 | 运动鞋 |

Lý do giữ:

| Lý do | Số occurrence |
|---|---:|
| whitelist | 4.723 |
| pattern:short_abbreviation | 2.659 |
| pattern:too_short | 1.407 |
| auto_loanword | 280 |
| no_cedict_entry | 279 |
| cedict_no_overlap | 161 |

### 11.2 Bitext preprocess

```bash
python run_pipeline.py bitext --preprocess-only
```

```text
V1 CJK copy → DISCARD: 2
Result: 16.346 rows
Flags: stuffing=1.119, truncated=1.566, bloated=0
```

### 11.3 Bitext legacy artifact

| Chỉ số | Giá trị |
|---|---:|
| KEEP | 16.201 |
| DISCARD | 145 |
| Cosine mean / median | 0.7021 / 0.7053 |
| Lexical mean (legacy) | 0.1534 |
| Hybrid mean (legacy) | 0.5375 |

Lưu ý: cosine trong artifact này được tính trước khi có `title_zh_cleaned`.
Chạy lại `python run_pipeline.py bitext` để đồng bộ.

---

## 12. Runbook

### 12.1 Cài đặt

```bash
pip install -r requirements.txt
```

### 12.2 CADO

```bash
# Mặc định: title_zh + CADO-VI, T0–T4
python run_pipeline.py cado

# Chỉ CADO Trung
python run_pipeline.py cado --no-vi

# T5 (cần bge-m3)
python run_pipeline.py cado --tier5

# Xử lý cả description (review kỹ)
python run_pipeline.py cado --columns title_zh,description_zh

# Test nhanh
python run_pipeline.py cado --limit 200 --output /tmp/cado_test.parquet
```

### 12.3 Bitext

```bash
# Preprocess, không cần model
python run_pipeline.py bitext --preprocess-only

# Full (cần bge-m3)
python run_pipeline.py bitext

# Hybrid label
python run_pipeline.py bitext --label-mode hybrid

# Dùng cosine có sẵn
python run_pipeline.py bitext --skip-cosine
```

### 12.4 Toàn bộ

```bash
python run_pipeline.py all
```

### 12.5 Từng tầng (debug)

```bash
python cado_layer/cado_tier0_extract.py
python cado_layer/cado_tier2_pattern.py --limit 100
python cado_layer/cado_tier3_loanword.py --limit 500
python cado_layer/cado_tier4_cedict.py --limit 500
python cado_layer/cado_tier5_bge.py --limit 200
python cado_layer/cado_tier2_entity.py --limit 200
```

### 12.6 Kiểm tra nhanh

```bash
python run_pipeline.py stats
python -m unittest discover -s tests -v
```

---

## 13. Artifact & CSV

### 13.1 Parquet

| File | Shape | Trạng thái |
|---|---:|---|
| `data/raw/bilingual_zh_vi.parquet` | 16.348 × 30 | Gốc |
| `data/cleaned/bilingual_zh_vi_cado.parquet` | 16.348 × 33 | **Mới regenerate** |
| `data/cleaned/bilingual_zh_vi_bitext_ready.parquet` | 16.346 × 44 | Legacy |
| `data/cleaned/bitext_scored.parquet` | 16.346 × 39 | Legacy |
| `data/cleaned/tiki_vi_cado.parquet` | 13.200 × 38 | Legacy, không có source |

### 13.2 CSV mới

| File | Nội dung |
|---|---|
| `csv/cado/cado_word_report.csv` | Quyết định theo từ (1.191 từ) |
| `csv/cado/cado_occurrences.csv` | 9.721 occurrence |
| `csv/cado/cado_changed_rows.csv` | 286 dòng thay đổi |
| `csv/cado/cado_tier3_loanword_report.csv` | 478 ứng viên T3 |
| `csv/cado/cado_vi_occurrences.csv` | 202 occurrence VI |
| `csv/cado/cado_tier5_similarity.csv` | Chỉ có khi chạy T5 |

### 13.3 CSV legacy (giữ để đối chiếu)

- `csv/cado/zh_cado_changed_rows.csv` — 193 dòng, có cả desc, schema khác.
- `csv/cado/tiki_cado_changed_rows.csv` — Tiki.
- `csv/changed_rows/vi_cado_*.csv` — CADO-VI legacy.
- `csv/english_remover/english_words.csv` — inventory từ.
- `csv/english_remover/vi_remaining_en_words.csv` — 2.798 từ còn sót trong VI.

---

## 14. Giới hạn & roadmap

### 14.1 Giới hạn hiện tại

1. **Chưa chạy lại bge-m3** trong lần refactor (T5 + cosine). Cần ~2.2GB và
   thời gian encode.
2. **Description ZH** xử lý được nhưng dễ xoá oan model code; mặc định tắt.
3. **Tiki** không có raw/source script; artifact không tái tạo được.
4. **Hybrid threshold 0.50** quá nghiêm trên dữ liệu hiện tại; cần tune.
5. **Whitelist** còn trùng lặp (323 dòng / 289 unique) và `match_mode` chưa
   được phân biệt.
6. **Entity S4** phụ thuộc embedding; nếu tắt S4b, score thấp.
7. **29 cặp nghi lệch căn chỉnh** chưa có filter riêng.
8. **Title trùng lặp** 485 dòng chưa dedupe ở bitext.
9. `--preprocess-only` không tạo `signal_S1..S7`; chạy
   `cado_layer/cado_tier2_entity.py` nếu cần các cột này.

### 14.2 Roadmap đề xuất

- Chạy full bge-m3, regenerate `bitext_scored.parquet` với `title_zh_cleaned`.
- Thêm filter cho 29 cặp suspect (dùng entity + cosine).
- Dedupe title trước khi train.
- Tune hybrid threshold trên tập dev có nhãn người.
- Bổ sung guard cho `description_zh` (model code `Type-c`, `USB-C`).
- Chuyển whitelist sang schema có `match_mode` thực sự.
- Viết script crawl/clean Tiki để tái tạo artifact.

---

## 15. Glossary

| Thuật ngữ | Nghĩa |
|---|---|
| **CADO** | Context-Aware Deletion of English |
| **CC-CEDICT** | Từ điển Trung–Anh offline, dùng để kiểm tra overlap |
| **Headword** | Mục từ trong CEDICT; bản mới dùng dạng giản thể |
| **Occurrence** | Một lần xuất hiện của từ trong một văn bản |
| **Loanword** | Từ mượn (Carplay, Bluetooth...) được giữ tự động |
| **Whitelist** | Danh sách từ không bao giờ xoá |
| **V1–V16** | Các bước preprocess bitext |
| **Cosine** | Độ tương đồng embedding giữa title ZH và VI |
| **Lexical overlap** | Jaccard trên brand/model/số/đơn vị |
| **Hybrid** | Kết hợp cosine và lexical |
| **KEEP / DISCARD** | Nhãn cuối cùng của bitext filter |
