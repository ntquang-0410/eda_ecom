# Tiki EDA — chuẩn bị back-translation VI → ZH

## Phase 1 — Title + description KV

Phạm vi Phase 1 chỉ gồm:

- `title_vi`
- `description_vi` dạng `key: value; ...`

`description_prose_vi` và toàn bộ prose **không** được đọc, không tạo unit, không
xuất vào product-level. File raw `tiki_vi_cado.parquet` vẫn giữ nguyên để audit.

### Chạy

```bash
# Profile title + KV, xuất JSON/CSV
python run_pipeline.py tiki-profile

# Build product-level + pair-level units, chưa gọi MT
python run_pipeline.py tiki-units
```

Hoặc chạy trực tiếp:

```bash
python tiki_mining/tiki_profile.py
python tiki_mining/tiki_bt_units.py
```

Tuỳ chọn hữu ích:

```bash
python tiki_mining/tiki_profile.py --limit 200 --quiet
python tiki_mining/tiki_bt_units.py --limit 200
```

### Output

| File | Grain | Nội dung |
|---|---|---|
| `data/cleaned/tiki_profile.json` | corpus | thống kê title, KV, key, class, parse error |
| `csv/tiki/tiki_key_inventory.csv` | key | 698 key chuẩn hoá, class, số cặp/value, sample |
| `csv/tiki/tiki_parse_errors.csv` | part | các phần description không có dấu `:` |
| `data/cleaned/tiki_products.parquet` | product | title + description KV, không có prose |
| `data/cleaned/tiki_bt_units.parquet` | unit | title + từng attribute value |
| `data/cleaned/tiki_units_stats.json` | corpus | thống kê unit, class, method, key map |

Số liệu full hiện tại:

- 13.200 product; 13.200 title unit.
- 89.302 attribute record; 107.684 attribute value unit.
- 120.884 unit tổng; 45.514 unit `train_eligible` (title 3–40 từ + text value).
- 698 unique key; 200 phần description malformed.
- Value class: admin 61.805; text 32.509; numeric 11.081; model 2.289.

### Product schema

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
- `description_vi_raw` giữ nguyên KV gốc để audit.
- `description_zh = null`, `attributes_zh = []` ở Phase 1.
- Không có `description_prose_vi`.

### Unit schema

```text
pair_id, product_id, source, category, split
unit_type, unit_index, value_index
key_vi, key_norm, key_class, key_zh
value_vi, value_zh, text_vi, text_zh
value_class, bt_method, bt_status, key_map_status
parse_source, n_words, value_n_words, n_chars
train_eligible, qc_flags, qc_flags_text, duplicate_count
```

- `unit_type = title | attribute`
- `pair_id`: `tiki:<product_id>:title` hoặc
  `tiki:<product_id>:attr:<attribute_index>:<value_index>`
- `text_vi = "<key_vi>: <value_vi>"` với attribute; title dùng nguyên title.
- `key_zh`, `value_zh`, `text_zh` để trống ở Phase 1; `bt_status = pending`.

### Phân loại key/value

| Class | Cách xử lý dự kiến | Ví dụ |
|---|---|---|
| `admin` | dictionary/preserve | thương hiệu, xuất xứ, bảo hành, hạn sử dụng |
| `model` | preserve | model, SKU, mã hàng |
| `numeric` | rule/convert đơn vị | trọng lượng, kích thước, dung tích |
| `text` | MT | chất liệu, thành phần, hướng dẫn sử dụng, màu sắc |

`key_class` lấy theo tên key. `value_class` có thể override:

- key `numeric` nhưng value là câu (`Nhiều size`, `Dùng cho...`) → `text`;
- value chỉ gồm số/đơn vị dưới key `text` → `numeric`.

`train_eligible` chỉ bật cho title 3–40 từ và attribute value thuộc `text`.
Admin/model/numeric vẫn có trong unit table để reassemble `description_zh` ở
Phase 2 nhưng không đưa vào bitext training.

### Key map Phase 2

`tiki_bt_units.py` tự nạp `csv/tiki/tiki_key_map.csv` nếu file tồn tại:

```csv
key_vi,key_zh
Chất liệu,材质
Màu sắc,颜色
Thương hiệu,品牌
```

Nếu chưa có file, `key_zh = null`, `key_map_status = pending_phase2` và
`key_unmapped` được gắn cho các text unit.

### Ghi chú số liệu

Số Phase 1 có thể lệch nhỏ so với parse thô vì value được `clean_text` và value
rỗng/nhiễu bị loại. Ví dụ raw parse có 89.373 cặp, sau làm sạch còn 89.302 cặp.
Đây là chủ ý: unit table chỉ chứa value có thể dùng để dịch.

---

## Phase 1.5 — Length prior 1688 human → Tiki VI

Mục tiêu là **hiệu chỉnh độ dài**, không dịch: dùng 1688 cặp human ZH–VI để
học phân bố `vi_chars / zh_chars`, rồi gắn khoảng độ dài ZH kỳ vọng cho từng
unit VI của Tiki. Phase 2 có thể dùng khoảng này để QC hoặc constrained
decoding mà không cần dịch trước.

### Chạy

```bash
python run_pipeline.py tiki-length-prior
# hoặc
python tiki_mining/tiki_length_prior.py
```

Tuỳ chọn hữu ích:

```bash
python tiki_mining/tiki_length_prior.py --fit-limit 2000 --limit 500 --quiet
python tiki_mining/tiki_length_prior.py --min-group-samples 20 --min-category-samples 100
```

### Học prior

Từ `data/cleaned/bilingual_zh_vi_cado.parquet`:

| Unit type | Số cặp | Cách lấy |
|---|---:|---|
| `title` | 16.348 | `title_zh` ↔ `title_vi` |
| `description` | 16.348 | `description_zh` ↔ `description_vi` |
| `attribute` | 406.206 | parse KV description, chỉ lấy hàng có cùng số phần ZH/VI, ghép theo thứ tự |

Tổng **438.902** cặp. Với attribute, thứ tự KV trong `description_zh` và
`description_vi` khớp nhau; không dùng thứ tự `attributes_zh` / `attributes_vi`
vì hai list này không aligned theo index.

Các cột length:

- `vi_chars` / `zh_chars`: số ký tự không tính khoảng trắng.
- `vi_words`: số token tách theo khoảng trắng (chỉ để audit).
- `ratio_vi_zh = vi_chars / zh_chars`.
- Bucket theo **`vi_chars`**: `0-10`, `11-20`, ..., `301+`.

Ratio VI/ZH p50 học được:

| Unit type | `vi_chars` p50 | `ratio_vi_zh` p50 |
|---|---:|---:|
| `title` | 97 | 3,156 |
| `attribute` | 24 | 3,000 |
| `description` | 906 | 2,600 |

Dùng bucket theo ký tự thay vì số từ vì title Tiki ngắn hơn title 1688 đáng kể
(VI chars p50: 59 so với 97). Nếu bucket theo số từ, prior sẽ lệch mạnh giữa
hai nguồn.

### Áp dụng cho Tiki

Với mỗi unit VI:

```text
expected_zh_chars = round(vi_chars / ratio_vi_zh_p50)
zh_chars_p05      = floor(vi_chars / ratio_vi_zh_p95)
zh_chars_p95      = ceil(vi_chars / ratio_vi_zh_p05)
```

Thứ tự fallback:

1. `unit_type + category + bucket` (n ≥ `min_category_samples`)
2. `unit_type + bucket` (n ≥ `min_group_samples`)
3. `unit_type + bucket` (n < `min_group_samples`, vẫn dùng vì bucket là điều
   kiện độ dài quan trọng hơn category)
4. `unit_type + category` (n ≥ `min_category_samples`)
5. `unit_type`
6. `global`

Mặc định `min_group_samples=20`, `min_category_samples=100`. Mỗi unit được
ghi lại `length_prior_scope` và `length_prior_n` để audit; `length_prior_n <
20` là tín hiệu bucket ít mẫu, không nên coi là prior chắc chắn.

### Output

| File | Nội dung |
|---|---|
| `data/cleaned/length_prior_1688.json` | stats + toàn bộ 305 group prior |
| `csv/length_prior/length_prior_by_bucket.csv` | bảng group phẳng để audit |
| `data/cleaned/tiki_bt_units_length.parquet` | 120.884 × 43: unit Phase 1 + prior |
| `data/cleaned/tiki_length_prior_stats.json` | coverage, scope, phân bố expected length |

Cột mới trong parquet:

```text
length_prior_source, length_prior_scope, length_prior_n, vi_char_bucket
vi_chars, vi_words
ratio_vi_zh_p05, ratio_vi_zh_p50, ratio_vi_zh_p95
zh_per_vi_word_p05, zh_per_vi_word_p50, zh_per_vi_word_p95
expected_zh_chars, zh_chars_p05, zh_chars_p95
```

Kết quả full:

- 120.884 unit; 0 unit thiếu prior.
- Title: expected ZH p50 30; Attribute: expected ZH p50 7.
- Scope: 111.816 `unit_type+category+bucket`; 9.068 `unit_type+bucket`.
- Không có prose trong prior; source luôn là `1688_human`.
- 285 title unit rơi vào bucket có n < 20 (chủ yếu title ngắn); range rộng và
  `length_prior_n` thấp. Đây là giới hạn của nguồn 1688 (title 1688 hiếm khi
  ngắn), không phải prior nhân tạo.

### QC Phase 2

Sau khi có `text_zh`, gọi:

```python
from cado_core.length_prior import length_qc

qc = length_qc(text_zh, row)
# qc["length_qc_flags"] -> ["zh_too_short"] / ["zh_too_long"] / []
```

`length_qc` so `zh_chars` với `[zh_chars_p05, zh_chars_p95]`; không tự sửa
hoặc cắt bản dịch.

---

## EDA Tiki cũ (legacy)

Module `cado_core/tiki.py` chạy EDA trên **toàn bộ** Tiki và xuất **một parquet
product-level** (1 dòng/sản phẩm), có cả prose. Bước này không gọi model, không
dịch. Artifact cũ vẫn giữ để tham khảo; Phase 1 không dùng prose từ artifact này.

## File đã publish

- 🤗 Hugging Face: [`ntquang0410/zh-vie_ecom/data/cleaned/tiki/tiki_eda.parquet`](https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/blob/main/data/cleaned/tiki/tiki_eda.parquet)
- Kích thước: **63.558.766 bytes** (~60,6 MiB), **13.200 dòng × 108 cột**
- SHA256: `7af2a2620b8ca2def3b80762cb8f99196bfe7c64baeaeaa8b99d974df570d3bd`
- Manifest toàn bộ cleaned: [`data/cleaned/README.md`](https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/blob/main/data/cleaned/README.md)

## Chạy

```bash
python tiki_mining/tiki_eda.py
# hoặc
python run_pipeline.py tiki-eda
```

Mặc định:

- input: `data/cleaned/tiki_vi_cado.parquet`
- output: `data/cleaned/tiki_eda.parquet`

Test nhanh:

```bash
python tiki_mining/tiki_eda.py --limit 200 --output /tmp/opencode/tiki_eda_test.parquet
```

Yêu cầu thêm `pyvi>=0.1.1` (MIT) để lấy lexicon tiếng Việt. Nếu thiếu pyvi,
script vẫn chạy nhưng tỉ lệ `vi_rate` / `foreign_rate` kém chính xác hơn.

## Grain & quy ước

- **1 dòng = 1 `product_id`**, giữ nguyên toàn bộ cột gốc.
- `bt_title`: title sạch, lấy từ `title_vi` gốc (không dùng
  `title_vi_cleaned` của CADO-VI vì bản đó đã xoá nhầm brand/tên sản phẩm).
- `bt_prose`: prose sạch từ `description_prose_vi`, đã bỏ boilerplate Tiki,
  URL/email/hotline, hashtag, emoji, dòng kẻ. Nếu title lặp lại ở đầu prose và
  đóng vai trò dòng tiêu đề, phần echo được bỏ (flag `title_echo_removed`).
- `bt_sentences`: list câu sạch, tối đa 60 từ/câu; `bt_sentence_freqs` là tần
  suất toàn corpus của từng câu (cùng thứ tự). Dùng để lọc câu template.
- `bt_attributes_text`: các cặp `name: value` đã làm sạch, lấy từ
  `attributes_vi` (fallback `description_vi`).
- `protected_spans`: số/đơn vị, model code, brand — nên giữ nguyên khi dịch.
- `overall_route`:
  - `BT_TITLE+PROSE`: dịch cả title và prose
  - `BT_TITLE`: chỉ title đủ tốt
  - `BT_PROSE`: chỉ prose đủ tốt
  - `BT_ATTR`: chỉ còn attributes
  - `REVIEW` / `DROP`: cần xem lại / bỏ
- `bt_ready` = title **hoặc** prose đủ tốt. `bt_ready_attrs` tách riêng.
- `title_score`, `prose_score`, `quality_score` là điểm heuristic 0–1, không
  phải xác suất. `quality_score` kết hợp title + prose + attributes.
- `eda_flags` / `eda_flags_text`: danh sách cờ EDA. Không xoá dòng nào; người
  dùng tự quyết định filter.
- `split`: train/dev/test 80/10/10 theo hash của `product_id + title + prose`,
  giữ các cặp trùng khít trong cùng split.
- `attributes_vi_json`: bản JSON của attributes để dễ đọc/portable.

## Gợi ý dùng trên Kaggle

```python
import pandas as pd

df = pd.read_parquet("tiki_eda.parquet")

# Title
titles = df.loc[df.bt_ready_title, ["product_id", "bt_title"]].copy()

# Prose theo từng câu
sentences = (
    df.loc[df.bt_ready_prose, ["product_id", "bt_sentences", "bt_sentence_freqs"]]
      .explode(["bt_sentences", "bt_sentence_freqs"])
      .rename(columns={"bt_sentences": "text_vi", "bt_sentence_freqs": "freq"})
)
sentences = sentences[sentences.text_vi.str.split().str.len() >= 4]
sentences = sentences[sentences.freq < 50]  # bỏ câu template lặp nhiều

# Attributes
attributes = df.loc[df.bt_ready_attrs, ["product_id", "bt_attributes_text"]].copy()
```

Khi ghép kết quả dịch, giữ `product_id` và (nếu dịch theo câu) thêm index câu
trước khi `explode` để merge lại chính xác.

## Ghi chú EDA

- Tiki là corpus **đơn ngữ VI**: `title_zh` và `description_zh` rỗng 100%,
  nên không chấm được cosine cặp ZH–VI.
- Boilerplate pháp lý Tiki xuất hiện ở 13.199/13.200 dòng và được bỏ.
- `description_vi` là chuỗi `key: value; ...`; prose thật nằm ở
  `description_prose_vi`.
- Cờ `foreign_high` chỉ cảnh báo khi title có >35% token ngoại lai hoặc prose
  có >15%; brand/model/đơn vị được bảo vệ nên không bị tính là nhiễu.
