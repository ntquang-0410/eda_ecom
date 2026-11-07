# Plan EDA & Processing — bộ A/B

**Ngày:** 2026-11-07  
**Mục tiêu:** chuẩn hoá cách kiểm tra và xử lý 2 bộ dữ liệu trước khi khảo sát trọng số.

---

## 1. Trạng thái dữ liệu

### A — title + attribute

Đã có:

```text
data/cleaned/weight_splits_v2/A_title_attribute.parquet   118.477 dòng (đã dedup)
data/cleaned/weight_splits_v2/A_title_attribute_raw.parquet 118.688 dòng (raw)
data/cleaned/weight_splits_v2/eval_A.parquet              1.200 dòng
data/cleaned/weight_splits_v2/meta_A.json
```

Nhóm:

```text
1688_title, 1688_attribute, tiki_title, tiki_attribute
```

### B — title + description

Đã tải raw BT 4B:

```text
bt_desc_v1/qwen3-4b/predictions.parquet   13.200 dòng
```

Đã tạo:

```text
data/cleaned/weight_splits_v2/B_title_description.parquet
data/cleaned/weight_splits_v2/eval_B.parquet
data/cleaned/weight_splits_v2/meta_B.json
```

Thống kê:

```text
rows           : 59.209
effective token: 14,598M
eval           : 900 dòng
empty target   : 1.150 dòng (raw 4B, chưa lọc)
```

Phân bố natural:

| Nhóm | Token share |
|---|---:|
| 1688_description | 56,57% |
| tiki_description | 34,93% |
| 1688_title | 5,15% |
| tiki_title | 3,34% |

Lưu ý: 8B mới có 11.900/13.200 dòng do time budget; B dùng 4B nên không cần 8B.

---

## 2. EDA cho A

### 2.1. Kiểm tra cấu trúc

```text
rows theo group
token theo group
share theo group
split train/dev/test
số product duy nhất
product overlap train/test
```

Kỳ vọng:

```text
1688_attribute ~65% token
1688_title    ~16%
tiki_title    ~10%
tiki_attribute ~9%
```

### 2.2. Chất lượng bản dịch

```text
empty target
target/source length ratio
CJK ratio
untranslated Vietnamese
duplicate (source, target)
duplicate source
quantity/number mismatch
```

### 2.3. Lỗi theo nhóm

Bảng riêng cho:

```text
1688_title
1688_attribute
tiki_title
tiki_attribute
```

Mục tiêu: biết nhóm nào đang yếu trước khi weight.

### 2.4. Audit split

- Không product nào nằm cả train và test.
- Eval `eval_id` không nằm trong train.
- `eval_A` có đúng 600 1688 + 600 Tiki.
- Không duplicate `pair_id`.

---

## 3. EDA cho B

B có thêm Tiki description raw BT, cần EDA riêng:

```text
boilerplate Tiki
target rỗng
target quá ngắn
target quá dài / truncated
CJK ratio
traditional Chinese
length ratio
number/unit mismatch
duplicate source
```

Đặc biệt:

```text
"Giá sản phẩm trên Tiki đã bao gồm thuế..." 
```

xuất hiện ở gần như mọi description; phải đo tỷ lệ và quyết định strip hay giữ.

### Bảng EDA B

```text
1688_title
1688_description
tiki_title
tiki_description
```

---

## 4. Processing A

### 4.1. Làm sạch tối thiểu

```text
- bỏ target rỗng
- bỏ dòng trùng (source, target), giữ dup_count
- cap eff_tokens <= 960
- quantity normalization (number/unit)
- giữ nguyên split product-level
```

### 4.2. Output

```text
data/cleaned/processed_A/full.parquet
data/cleaned/processed_A/train.parquet
data/cleaned/processed_A/dev.parquet
data/cleaned/processed_A/test.parquet
data/cleaned/processed_A/units.parquet
data/cleaned/processed_A/eda.json
```

`units` = dòng đã sạch, có `group` và `eff_tokens`.

### 4.3. Không làm

```text
- không đổi eval_A
- không trộn description
- không lọc theo kết quả eval
```

---

## 5. Processing B

### 5.1. Làm sạch tối thiểu

```text
- bỏ target rỗng
- strip boilerplate Tiki khỏi source/target nếu xác định được
- normalize Traditional -> Simplified nếu có opencc
- bỏ target trùng
- cap eff_tokens <= 960
- quantity normalization
- chunk description >960 token
```

### 5.2. Output

```text
data/cleaned/processed_B/full.parquet
data/cleaned/processed_B/train.parquet
data/cleaned/processed_B/dev.parquet
data/cleaned/processed_B/test.parquet
data/cleaned/processed_B/units.parquet
data/cleaned/processed_B/eda.json
```

### 5.3. Điều cần thống nhất

B dùng **raw BT trước EDA** để khảo sát weight, nhưng processing có thể strip boilerplate. Vì vậy cần tách:

```text
B_raw       : dùng cho weight survey
B_processed : dùng cho báo cáo chất lượng / huấn luyện final
```

Nếu nhóm muốn weight survey cũng dùng B_processed thì phải ghi rõ và chạy lại split.

---

## 6. Thứ tự chạy

```text
1. Tải raw BT 4B/8B
2. Build B_raw (chưa lọc)
3. EDA A + B_raw
4. Chốt ngưỡng processing
5. Tạo A_processed, B_processed
6. Khảo sát weight trên A_raw/B_raw (4M token, 500 step)
7. Chấm COMET + Gemma
8. Nếu cần: chạy seed 123 cho top mỗi tập
```

---

## 7. Eval

### A

```text
1688 title + attribute : COMET-da
Tiki title + attribute : COMET-QE
```

### B

```text
1688 title + description : COMET-da
Tiki title + description : COMET-QE
```

Tiki description không có reference người.

Gemma credit AAA–C là tín hiệu phụ.

---

## 8. Tiêu chí kết luận

```text
primary   : macro 4 stratum của A hoặc B
guardrail : không stratum nào giảm quá 0,01
bootstrap : 2.000 paired theo eval_id
seed      : 42 trước, 123 cho top nếu còn quota

best - natural < 0,005 hoặc CI chứa 0 -> natural
```

---

## 9. File cần tạo

```text
reports/EDA_A.md
reports/EDA_B.md
reports/PROCESSING_A.md
reports/PROCESSING_B.md
reports/WEIGHT_SURVEY_A.md
reports/WEIGHT_SURVEY_B.md
```
