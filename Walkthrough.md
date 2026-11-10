# 📖 Walkthrough — Dữ Liệu & Pipeline Song Ngữ Trung–Việt (1688)

Tài liệu này mô tả dữ liệu, các vấn đề phát hiện trong EDA, kiến trúc CADO,
pipeline bitext, refactor 2026-10-21, và **Tiki Phase 1 / 1.5 / A / B / C**
(2026-10-23) cùng cách tái tạo artifact.

> Đọc nhanh: nếu chỉ muốn chạy pipeline, xem mục **12. Runbook** và **14–18**.
> Nếu muốn hiểu vì sao một từ bị xoá, xem mục **6–8**.
> Nếu chỉ quan tâm Tiki/length prior, đọc **14–18** và **11.4–11.6**.

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
14. [Tiki Phase 1 — title + KV units](#14-tiki-phase-1--title--kv-units)
15. [Tiki Phase 1.5 — length prior v1](#15-tiki-phase-15--length-prior-v1)
16. [Phase A — value-only v2](#16-phase-a--value-only-v2)
17. [Phase B/C — alignment + title v3](#17-phase-bc--alignment--title-v3)
18. [Đánh giá tổng thể A/B/C](#18-đánh-giá-tổng-thể-abc)
19. [Giới hạn & roadmap](#19-giới-hạn--roadmap)
20. [Glossary](#20-glossary)

---

## 1. Tổng quan

Mục tiêu của repo:

- Làm sạch các từ tiếng Anh bị chèn vào tiêu đề/mô tả tiếng Trung và tiếng
  Việt của sản phẩm 1688, **không** xoá brand/model/đơn vị/từ mượn.
- Tạo tập cặp câu Trung–Việt sạch để huấn luyện dịch máy.
- Chuẩn bị Tiki VI → ZH: title + KV units, không dùng prose.
- Học length prior từ 1688 human để QC/ước lượng độ dài ZH cho Tiki.
- Giữ khả năng audit: mỗi quyết định xoá/giữ, split value và alignment đều có
  artifact/CSV riêng.

Các thành phần chính:

| Thành phần | Vai trò | Code |
|---|---|---|
| **CADO** | Xoá từ tiếng Anh dư thừa trong ZH/VI | `cado_core/cado.py`, `cado_core/cado_vi.py` |
| **Bitext Filter** | Preprocess + chấm điểm cặp câu | `cado_core/bitext.py` |
| **Tiki Units** | Title + description KV → BT units | `cado_core/tiki_units.py` |
| **Length Prior v1** | Prior key+value (audit) | `cado_core/length_prior.py` |
| **Length Prior v2** | Value-only + class + CJK + split-safe | `cado_core/length_prior_v2.py` |
| **Length Prior v3** | Alignment audit/fallback + title smoothing | `cado_core/length_prior_v3.py` |

---

## 2. Dữ liệu & schema

### 2.1 Nguồn

- 1688.com (Trung Quốc), sản phẩm được crawl và dịch sang tiếng Việt.
- File gốc: `data/raw/bilingual_zh_vi.parquet` — **16.348 dòng × 30 cột**.
- Ngoài ra có artifact Tiki `data/cleaned/tiki_vi_cado.parquet`
  (**13.200 × 37**) nhưng repo **không có raw/source script** cho Tiki.
  Tiki là corpus đơn ngữ VI; `title_zh` / `description_zh` rỗng 100%.

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

### 11.4 Tiki Phase 1

```text
13.200 product
13.200 title unit
107.684 attribute unit
120.884 unit tổng
45.514 train_eligible
698 key; 200 parse error
```

Value class: admin 61.805; text 32.509; numeric 11.081; model 2.289.

### 11.5 Tiki length prior A/B/C

| Metric | V2 | V3 |
|---|---:|---:|
| Dev coverage | 98,35% | 98,32% |
| Test coverage | 98,38% | 98,36% |
| Dev MAE | 1,43 | 1,42 |
| Test MAE | 1,43 | 1,42 |
| Title test MAE | 2,158 | **1,777** |
| Alignment fallback | — | 213 |
| Low-confidence group | — | 217.276 |

- V2: 120.884 × 57.
- V3: 120.884 × 58.
- V3 title smoothing: 13.200 unit; 107.684 attribute value-only.
- 0 null prior; 0 prose; `text_zh` vẫn null.

### 11.6 Kết luận kiểm chứng

- CADO/bitext vẫn giữ kết quả cũ; không regression.
- Tiki Phase 1 tạo đúng unit grain, không đọc prose.
- V2 sửa lỗi key/grain/class/split; V3 cải thiện title và thêm alignment
  fallback.
- Residual risk lớn nhất là low-confidence alignment long-tail và title
  transfer từ 1688.

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
# 43 test hiện tại
```

### 12.7 Tiki — Phase 1 / 1.5 / A / B / C

```bash
# Phase 1: profile + units (không gọi MT)
python run_pipeline.py tiki-profile
python run_pipeline.py tiki-units

# Phase 1.5: prior key+value (v1, audit)
python run_pipeline.py tiki-length-prior

# Phase A: value-only + class + CJK + split-safe (v2)
python run_pipeline.py tiki-length-prior-v2

# Phase B/C: alignment audit/fallback + title smoothing (v3)
python run_pipeline.py tiki-length-prior-v3
```

Chạy nhanh để test:

```bash
python run_pipeline.py tiki-units --limit 200
python run_pipeline.py tiki-length-prior-v2 --fit-limit 1000 --limit 500
python tiki_mining/tiki_length_prior_v3.py --fit-limit 1000 --limit 500 --quiet
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
| `data/cleaned/tiki_vi_cado.parquet` | 13.200 × 37 | Legacy, không có source |
| `data/cleaned/tiki_products.parquet` | 13.200 × 27 | Phase 1: product-level, không prose |
| `data/cleaned/tiki_bt_units.parquet` | 120.884 × 28 | Phase 1: title + attribute units |
| `data/cleaned/tiki_bt_units_length.parquet` | 120.884 × 43 | Phase 1.5: prior v1 key+value (audit) |
| `data/cleaned/tiki_bt_units_length_v2.parquet` | 120.884 × 57 | Phase A: value-only/class/CJK |
| `data/cleaned/tiki_bt_units_length_v3.parquet` | 120.884 × 58 | Phase B/C: alignment + title smoothing |
| `data/cleaned/tiki_length_prior_stats.json` | — | Stats v1 |
| `data/cleaned/tiki_length_prior_v2_stats.json` | — | Stats v2 |
| `data/cleaned/tiki_length_prior_v3_stats.json` | — | Stats v3 |
| `data/cleaned/phase_evaluation.json` | — | Đánh giá A/B/C |

### 13.2 CSV mới

| File | Nội dung |
|---|---|
| `csv/cado/cado_word_report.csv` | Quyết định theo từ (1.191 từ) |
| `csv/cado/cado_occurrences.csv` | 9.721 occurrence |
| `csv/cado/cado_changed_rows.csv` | 286 dòng thay đổi |
| `csv/cado/cado_tier3_loanword_report.csv` | 478 ứng viên T3 |
| `csv/cado/cado_vi_occurrences.csv` | 202 occurrence VI |
| `csv/cado/cado_tier5_similarity.csv` | Chỉ có khi chạy T5 |
| `csv/tiki/tiki_key_inventory.csv` | 698 key Tiki Phase 1 |
| `csv/tiki/tiki_parse_errors.csv` | 200 phần description lỗi |
| `csv/length_prior/length_prior_by_bucket.csv` | 305 group prior v1 |
| `csv/length_prior_v2/prior_value_by_class_bucket.csv` | Prior v2 all |
| `csv/length_prior_v2/prior_value_by_class_bucket_train.csv` | Prior v2 train |
| `csv/length_prior_v2/attribute_split_mismatch.csv` | 10.713 group split mismatch |
| `csv/length_prior_v2/attribute_split_sample.csv` | Sample audit split |
| `csv/length_prior_v2/attribute_split_summary.csv` | Summary split status |
| `csv/length_prior_v3/alignment_key_map.csv` | Key map ZH→VI |
| `csv/length_prior_v3/alignment_audit_sample.csv` | 500 group audit |
| `csv/length_prior_v3/alignment_summary.csv` | Summary alignment |

### 13.3 CSV legacy (giữ để đối chiếu)

- `csv/cado/zh_cado_changed_rows.csv` — 193 dòng, có cả desc, schema khác.
- `csv/cado/tiki_cado_changed_rows.csv` — Tiki.
- `csv/changed_rows/vi_cado_*.csv` — CADO-VI legacy.
- `csv/english_remover/english_words.csv` — inventory từ.
- `csv/english_remover/vi_remaining_en_words.csv` — 2.798 từ còn sót trong VI.

---

## 14. Tiki Phase 1 — title + KV units

### 14.1 Phạm vi

Phase 1 chỉ dùng:

- `title_vi`;
- `description_vi` dạng `key: value; ...`.

`description_prose_vi` và toàn bộ prose **không** được đọc, không tạo unit và
không xuất vào product-level. File raw `tiki_vi_cado.parquet` vẫn giữ để audit.
Không dùng `title_vi_cleaned` của CADO-VI vì bản đó có thể làm hỏng brand/tên
sản phẩm.

### 14.2 Output

| File | Grain | Nội dung |
|---|---|---|
| `data/cleaned/tiki_profile.json` | corpus | title/KV/key class/parse error |
| `data/cleaned/tiki_products.parquet` | product | 13.200 × 27, không prose |
| `data/cleaned/tiki_bt_units.parquet` | unit | 120.884 × 28 |
| `data/cleaned/tiki_units_stats.json` | corpus | unit/class/method stats |
| `csv/tiki/tiki_key_inventory.csv` | key | 698 key chuẩn hoá |
| `csv/tiki/tiki_parse_errors.csv` | part | 200 phần description lỗi |

Số liệu full:

- 13.200 product; 13.200 title unit; 107.684 attribute unit.
- 120.884 unit tổng; **45.514** unit `train_eligible`.
- 698 unique key; 200 phần description malformed.
- Value class:
  - `admin`: 61.805
  - `text`: 32.509
  - `numeric`: 11.081
  - `model`: 2.289

### 14.3 Product schema

```text
product_id, source, category
title_vi_raw, title_vi, title_zh
description_vi_raw, description_vi, description_zh
attributes_vi, attributes_vi_json, attributes_zh
n_attributes_vi, n_attributes_zh, n_attributes_aligned, n_values_vi
n_units, n_train_eligible_units
description_parse_source, description_n_parts, description_n_malformed_parts
title_duplicate_count, split, bt_status, zh_origin, qc_flags, qc_flags_text
```

- `description_vi` là bản canonical serialize từ `attributes_vi`.
- `description_zh = null`, `attributes_zh = []` ở Phase 1.
- Không có cột prose.

### 14.4 Unit schema

```text
pair_id, product_id, source, category, split
unit_type, unit_index, value_index
key_vi, key_norm, key_class, key_zh
value_vi, value_zh, text_vi, text_zh
value_class, bt_method, bt_status, key_map_status
parse_source, n_words, value_n_words, n_chars
train_eligible, qc_flags, qc_flags_text, duplicate_count
```

- `unit_type = title | attribute`.
- `pair_id = tiki:<product_id>:title` hoặc
  `tiki:<product_id>:attr:<attribute_index>:<value_index>`.
- `text_vi = "<key_vi>: <value_vi>"` với attribute; title dùng nguyên title.
- `key_zh`, `value_zh`, `text_zh` để trống ở Phase 1; `bt_status = pending`.
- `train_eligible` chỉ bật cho title 3–40 từ và attribute value `text`.
- Admin/model/numeric vẫn có trong unit table để reassemble
  `description_zh` nhưng không train bitext.

### 14.5 Phân loại key/value

| Class | Xử lý dự kiến | Ví dụ |
|---|---|---|
| `admin` | dictionary/preserve | thương hiệu, xuất xứ, bảo hành |
| `model` | preserve | model, SKU, mã hàng |
| `numeric` | rule/convert | trọng lượng, kích thước, dung tích |
| `text` | MT | chất liệu, thành phần, màu sắc |

`key_class` lấy theo tên key; `value_class` có thể override theo value thực tế.

### 14.6 Key map Phase 2

`tiki_bt_units.py` tự nạp `csv/tiki/tiki_key_map.csv` nếu tồn tại. Nếu chưa
có, `key_zh = null`, `key_map_status = pending_phase2`, và text unit bị gắn
`key_unmapped`.

---

## 15. Tiki Phase 1.5 — length prior v1

### 15.1 Mục tiêu

Học tỷ lệ độ dài VI/ZH từ **1688 human**, rồi gắn khoảng độ dài ZH kỳ vọng cho
từng unit VI của Tiki. Không gọi model/MT.

### 15.2 Cặp human

| Unit type | Số cặp | Cách lấy |
|---|---:|---|
| `title` | 16.348 | `title_zh` ↔ `title_vi` |
| `description` | 16.348 | `description_zh` ↔ `description_vi` |
| `attribute` | 406.206 | parse KV description, ghép theo thứ tự |
| **Tổng** | **438.902** | |

Với attribute, thứ tự KV trong `description_zh` và `description_vi` khớp
nhau. Không dùng thứ tự `attributes_zh` / `attributes_vi` vì hai list này
không aligned theo index.

### 15.3 Công thức v1

```text
ratio_vi_zh = vi_chars / zh_chars
expected_zh_chars = round(vi_chars / ratio_vi_zh_p50)
zh_chars_p05 = floor(vi_chars / ratio_vi_zh_p95)
zh_chars_p95 = ceil(vi_chars / ratio_vi_zh_p05)
```

Bucket theo `vi_chars`: `0-10`, `11-20`, ..., `301+`.

Ratio p50:

| Unit type | `vi_chars` p50 | `ratio_vi_zh` p50 |
|---|---:|---:|
| `title` | 97 | 3,156 |
| `attribute` | 24 | 3,000 |
| `description` | 906 | 2,600 |

### 15.4 Kết quả và artifact

- 305 group prior.
- Tiki v1: 120.884 × 43.
- Title expected p50 ~30; attribute expected p50 ~7.
- `length_prior_scope`, `length_prior_n`, `zh_too_short` / `zh_too_long` QC.

Artifact:

- `data/cleaned/length_prior_1688.json`
- `data/cleaned/tiki_bt_units_length.parquet`
- `data/cleaned/tiki_length_prior_stats.json`
- `csv/length_prior/length_prior_by_bucket.csv`

**Hạn chế:** v1 học trên `key: value`, key Tiki dài hơn key 1688; 1688 gộp
nhiều value trong một attribute; title 1688 có đặc thù sàn. Vì vậy v1 chỉ giữ
làm audit, không dùng làm prior chính.

---

## 16. Phase A — value-only v2

### 16.1 Vấn đề phải sửa

1. Prior v1 học trên key+value, không phải value-only.
2. Grain 1688 gộp `a, b, c`, Tiki tách từng value.
3. Title 1688 ~30 ký tự ZH do đặc thù sàn.
4. Alignment attribute dựa trên description-order.
5. Chưa stratify `value_class` / CJK.
6. Chưa split-safe.

### 16.2 Những gì v2 đã làm

- Target **value-only** (`zh_value_chars`).
- Key correction:
  ```text
  expected_zh_text_chars =
      expected_zh_key_chars + separator_chars + expected_zh_value_chars
  ```
  `key_zh` dùng actual nếu có; nếu chưa có, estimate theo `key_class`.
- `split_value_list`:
  - tách `,`, `，`, `、`, `;`, `；`;
  - guard `1,5`, `1,000`;
  - mismatch không đoán, chỉ giữ group + audit.
- Fit riêng `text`, `numeric`, `admin`, `model`.
- Đo `zh_cjk_chars`, `zh_non_cjk_chars`, `cjk_ratio`, `ratio_vi_cjk`.
- Split theo `product_id` 80/10/10.
- `length_prior_soft_qc = true` cho title và `admin/model/numeric`; chỉ `text`
  dùng QC cứng hơn.

### 16.3 Kết quả v2

| Chỉ số | Giá trị |
|---|---:|
| Value-only pairs | 592.055 |
| Title pairs | 16.348 |
| Group audit | 51.082 |
| `single` | 355.117 (87,4%) |
| `multi_matched` | 40.374 (9,9%) |
| `mismatch` | 10.708 (2,6%) |
| `empty` | 7 |
| Train prior pairs | 488.969 |
| All prior pairs | 608.403 |
| Prior groups | 660 |

Holdout train prior:

| Split | Coverage p05–p95 | MAE value chars |
|---|---:|---:|
| dev | 98,35% | 1,43 |
| test | 98,38% | 1,43 |

Tiki v2: **120.884 × 57**:

```text
value_class+category+bucket  104.008
unit_type+bucket               9.068
unit_type+category+bucket      4.132
value_class+bucket             3.350
value_class                      326
```

Artifact:

- `data/cleaned/length_prior_1688_v2_train.json`
- `data/cleaned/length_prior_1688_v2_all.json`
- `data/cleaned/length_prior_1688_v2_holdout_metrics.json`
- `data/cleaned/tiki_bt_units_length_v2.parquet`
- `data/cleaned/tiki_length_prior_v2_stats.json`
- `csv/length_prior_v2/*`

---

## 17. Phase B/C — alignment + title v3

### 17.1 Phase B — alignment audit/fallback

V3 học key map từ description-order:

```text
key_zh_norm → key_vi_norm
```

Điều kiện map:

- `freq >= 20`;
- `top_share >= 0.60`.

Với group low-confidence, thử tìm VI attribute theo key map. Nếu tìm được duy
nhất, dùng `key_fallback`; nếu không, giữ description-order và gắn cờ review.

Kết quả full:

| Chỉ số | Giá trị |
|---|---:|
| ZH key được map | 276 / 2.049 |
| Cặp key đủ tần suất | 921 |
| `description_order` | 405.993 |
| `key_fallback` | 213 |
| High confidence | 188.723 |
| Medium confidence | 207 |
| Low confidence | 217.276 |

Audit sample 500:

| `audit_status` | Số group |
|---|---:|
| `review_low_confidence` | 252 |
| `pass_auto` | 233 |
| `review_split_mismatch` | 13 |
| `review_fallback` | 2 |

**Kết luận Phase B:** fallback hoạt động nhưng low-confidence vẫn lớn do key
long-tail. Không nên coi toàn bộ alignment đã sạch; cần review sample hoặc fit
strict high/medium trước khi dùng cho training.

### 17.2 Phase C — hierarchical title smoothing

Title dùng partial pooling:

```text
mu_shrunk = (n * mu_bucket + k * mu_parent) / (n + k)
```

- `k = 20`;
- interval inflation cho bucket ít mẫu;
- monotonic expected length theo bucket;
- chỉ soft QC, không filter/không ép decode.

Kết quả test:

| Metric | V2 | V3 |
|---|---:|---:|
| Title MAE | 2,158 | **1,777** |
| Title coverage | 95,19% | 94,67% |
| Overall coverage | 98,38% | 98,36% |
| Overall MAE | 1,433 | **1,423** |

Tiki v3: **120.884 × 58**:

- 13.200 title unit hierarchical smoothing;
- 107.684 attribute giữ value-only;
- 0 null prior; không prose; `text_zh` vẫn null;
- 88.375 unit soft QC.

Artifact:

- `data/cleaned/length_prior_1688_v3_train.json`
- `data/cleaned/length_prior_1688_v3_all.json`
- `data/cleaned/length_prior_1688_v3_holdout_metrics.json`
- `data/cleaned/tiki_bt_units_length_v3.parquet`
- `data/cleaned/tiki_length_prior_v3_stats.json`
- `csv/length_prior_v3/*`
- `data/cleaned/phase_evaluation.json`
- `data/cleaned/phase_evaluation.md`

---

## 18. Đánh giá tổng thể A/B/C

### 18.1 Bảng tổng hợp

| Hạng mục | Phase A (v2) | Phase B | Phase C (v3) |
|---|---:|---:|---:|
| Value-only pairs | 592.055 | 592.049 | 592.049 |
| Title pairs | 16.348 | 16.348 | 16.348 |
| Alignment fallback | — | 213 | 213 |
| Low-confidence group | — | 217.276 | 217.276 |
| Dev coverage | 98,35% | — | 98,32% |
| Test coverage | 98,38% | — | 98,36% |
| Dev MAE | 1,429 | — | 1,421 |
| Test MAE | 1,433 | — | 1,423 |
| Title test MAE | 2,158 | — | 1,777 |
| Tiki output | 120.884 × 57 | — | 120.884 × 58 |

### 18.2 Kết luận

- **Phase A** giải quyết đúng các lỗi nền: value-only, grain, class, CJK và
  split-safe. Đây là bước cải thiện chất lượng lớn nhất.
- **Phase B** đã có audit/fallback thật, nhưng chưa xử lý được toàn bộ key
  long-tail. Đây là residual risk lớn nhất.
- **Phase C** giảm title MAE ~18%, đổi lại coverage giảm nhẹ. Title chỉ nên
  dùng soft QC.
- **V3 là bản nên dùng cho Phase 2**, nhưng nên:
  1. review 500 sample alignment;
  2. fit strict trên `high + medium` nếu cần training sạch;
  3. dùng `text` cho length QC cứng, `admin/model/numeric` chỉ preserve/soft;
  4. không dùng title làm hard constraint.

---

## 19. Giới hạn & roadmap

### 19.1 Giới hạn hiện tại

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
10. **Tiki value alignment**: Phase B vẫn còn 217.276 group low-confidence
    (key long-tail); fallback chỉ xử lý an toàn 213 group.
11. **Tiki key_zh**: Phase 1 chưa có key map nên key length trong v3 là
    estimate theo `key_class`; cần key map Phase 2 để chính xác hơn.
12. **Title prior**: v3 đã smooth nhưng vẫn học từ 1688; title Tiki ngắn hơn
    và có phân phối khác. Chỉ dùng soft QC.
13. **Admin/model/numeric**: length prior không phản ánh đúng nghĩa preserve;
    các class này đã được đánh dấu soft QC.
14. **Split mismatch**: 10.713 group 1688 (2,64%) không tách value được; các
    group này chỉ dùng audit, không đưa vào value-level prior.

### 19.2 Roadmap đề xuất

- Chạy full bge-m3, regenerate `bitext_scored.parquet` với `title_zh_cleaned`.
- Thêm filter cho 29 cặp suspect (dùng entity + cosine).
- Dedupe title trước khi train.
- Tune hybrid threshold trên tập dev có nhãn người.
- Bổ sung guard cho `description_zh` (model code `Type-c`, `USB-C`).
- Chuyển whitelist sang schema có `match_mode` thực sự.
- Viết script crawl/clean Tiki để tái tạo artifact.
- **Phase B.2:** review 500 alignment sample; fit strict prior chỉ trên
  `high + medium`; bổ sung semantic/key map cho long-tail key.
- **Phase C.2:** đánh giá title prior bằng một tập Tiki có bản dịch người;
  nếu không có, giữ soft QC và không dùng làm hard constraint.
- **Phase 2:** tạo `csv/tiki/tiki_key_map.csv`, chuyển
  `key_len_source=estimated_key_class` sang `actual`, rồi dùng
  `length_qc_v2/v3` để gắn flag sau BT.

---

## 20. Glossary

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
| **Value-only** | Prior chỉ học value, không học key |
| **Key correction** | Cộng độ dài key_zh khi cần độ dài text_zh đầy đủ |
| **Value grain** | Một value riêng lẻ trong attribute nhiều giá trị |
| **Split-safe** | Chia train/dev/test theo product, không leakage |
| **CJK-only** | Độ dài chỉ tính ký tự Hán, bỏ model/số/dấu câu |
| **Key map** | Map `key_zh_norm → key_vi_norm` cho alignment |
| **Key fallback** | Align lại VI attribute theo key map khi confidence thấp |
| **Hierarchical smoothing** | Partial pooling bucket về parent, dùng cho title |
| **Soft QC** | Chỉ cảnh báo độ dài, không filter/ép decode |
