# Đề xuất EDA & xử lý Tiki trước khi merge với 1688

Ngày: 2026-10-29. Phạm vi: `data/cleaned/bt_tiki` (4 model × 45.709 unit MT).

## 1. Mục tiêu

1. Tiki phải **sạch và nhất quán** trước khi trộn với 1688 human.
2. Bốn bộ combined chỉ khác phần MT của Tiki, mọi thứ khác dùng chung.
3. Đánh giá baseline trên **cùng một test set** để so sánh công bằng.

## 2. Số liệu EDA bổ sung (Tiki vs 1688)

| chỉ số | Tiki | 1688 |
|---|---|---|
| sản phẩm | 13.200 | 16.346 |
| unit MT | 45.709 | — |
| attribute / sản phẩm | **2,46** | **24,9** |
| title VI p50 / p95 (ký tự) | 59 / 112 | 97 / 129 |
| key unique | 612 | 1.464 |
| attribute value unique | 32.509 | 23.648 |

**Key overlap thấp:**

```text
52 / 612 key Tiki có tên tương ứng trong 1688 (theo unique name)
54,4% số unit Tiki có key nằm trong từ vựng 1688
```

**Leakage thấp:**

```text
Tiki source trùng title 1688   : 0        (0,00%)
Tiki source trùng attr value   : 312      (1,13%)
```

Hệ quả: Tiki mang **domain và từ vựng key riêng**; không thể coi 1688 là bản mở
rộng tự nhiên. Merge phải giữ `domain` như một nhãn và cân bằng trọng số.

## 3. EDA cần chạy thêm trước merge

### 3.1. Bắt buộc (P0)

- **Phân bố key theo tần suất** (long tail): key có <5 unit nên gộp hoặc bỏ.
- **Tỉ lệ value_class** (text / numeric / model / admin) sau khi map.
- **Độ phủ key map**: bao nhiêu unit có `key_zh_effective` rỗng → không dùng được.
- **Trùng lặp trong Tiki**: `(source, target)` và `(source)` — đã có `dup_count`.
- **Kiểm tra split**: không sản phẩm nào nằm ở 2 split; test ổn định.
- **So sánh phân bố độ dài** train/dev/test (tránh lệch domain giữa split).

### 3.2. Nên có (P1)

- **Language ID** trên target: tỉ lệ Hán / Latin / Việt / hỗn hợp theo unit_type.
- **Entity/brand density**: % token Latin trong source, theo ngành hàng.
- **Placeholder & số**: tỉ lệ giữ nguyên số, đơn vị, mã model.
- **Trùng gần** (near-duplicate): MinHash/ngram trên source để phát hiện cặp gần giống
  nhưng target khác nhau.
- **Phân bố theo category Tiki** (13.200 sản phẩm) để phát hiện mất cân bằng ngành.

### 3.3. Sau merge (P2)

- So sánh phân bố combined vs 1688 gốc (domain shift).
- Kiểm tra leakage giữa train và test sau khi trộn.

## 4. Pipeline xử lý Tiki

### Bước 1 — Chốt QC gate

```text
LOẠI CỨNG
  target rỗng
  target == key_zh_effective          ← lỗi lớn nhất, gate cũ bỏ lọt
  không có chữ Hán + source có dấu Việt
  lặp vô hạn (degenerate)
  source >= 10 ký tự nhưng target <= 2 ký tự

GẮN CỜ (giữ, hạ weight)
  target còn dấu Việt lẫn trong câu Hán
  target Latin-only cho source tiếng Việt
  lệch số/đơn vị
  tỉ lệ độ dài ngoài [0,15 ; 2,5]
  target == source (phần lớn là brand hợp lệ)
```

### Bước 2 — Repair 17.632 dòng lỗi key

```text
0.6B 14.231 | 1.7B 1.921 | 4B 533 | 8B 937
```

Dùng prompt v5 (`brand_aware`/`no_key`) đã chứng minh hiệu quả ở vòng 2. Ước
~30 phút GPU. Nếu không repair, 0.6B chỉ còn 67% unit dùng được.

### Bước 3 — Dedup

- Dedup `(source_norm, target_norm)` trong từng tập.
- Giữ `dup_count` làm sample weight (tần suất xuất hiện thật).
- Không dedup theo `source` đơn thuần: cùng source có thể có nhiều target hợp lệ
  theo key khác nhau (353–528 trường hợp/tập).

### Bước 4 — Chuẩn hoá text

```text
Unicode NFC, gộp whitespace, bỏ control char
full-width -> half-width cho số và đơn vị
không đổi đơn vị (cm, ml, W...) — giữ nguyên để học cách giữ
giữ nguyên brand/model code
```

### Bước 5 — Gắn nhãn cấu trúc

- `pair_type`: `title` / `attribute` / `description`.
- `value_class`: `text` / `numeric` / `model` / `admin`.
- `key_vi`, `key_zh` cho attribute (đã có).
- `domain = tiki`, `origin = mt | repaired | rule | dictionary | preserve`.

### Bước 6 — Giữ split gốc

- Split theo `product_id` (đã đúng): train 36.434 / dev 4.824 / test 4.451.
- **Test giữ đủ 4.451 unit cho cả 4 model**, không lọc — nếu lọc, mỗi model bị
  chấm trên tập khác nhau và điểm bị thổi phồng.
- Báo cáo thêm `test_pass_rate` = số unit vượt QC / 4.451.

### Bước 7 — Xuất bảng merge-ready

Schema thống nhất với 1688:

```text
pair_id, domain, pair_type, product_id, split,
source_text, target_text, key_vi, key_zh,
value_class, origin, model_id, prompt_version,
qc_status, dup_count, weight
```

## 5. Thiết kế merge với 1688

### 5.1. Trích 1688

| nguồn | số cặp | lọc |
|---|---|---|
| title | 16.346 | bỏ `flag_truncated`, `flag_bloated` |
| attribute | ~407.000 | cap 15/sản phẩm, bỏ cặp số lệch |
| description | 16.346 | chỉ giữ ≤400 ký tự VI, weight 0,8 |

`flag_stuffing` (1.119 sản phẩm): giữ nhưng weight ×0,7.

### 5.2. Dedup chéo domain

- Bỏ cặp Tiki trùng khít cặp 1688 (giữ human).
- Dedup trong nội bộ từng domain theo `(source_norm, target_norm)`.

### 5.3. Trọng số

```text
1688 : title 1,0 | attribute 0,6 | description 0,8
Tiki : title 1,0 | attribute 0,8 | mt 0,5 | repaired 0,4 | preserve 0,3
```

Sau đó chuẩn hoá để **tổng weight 2 domain xấp xỉ 50/50**, tránh 1688 attribute
(400k cặp) nhấn chìm Tiki (45k).

### 5.4. Kiểm tra sau merge

```text
không product_id nào ở 2 split
không cặp trùng khít
tỉ lệ domain 40–60%
độ dài p95 < 512 token (max_seq)
số cặp / split / domain / pair_type
```

## 6. Đánh giá & chọn baseline

| tầng | dữ liệu | metric |
|---|---|---|
| 1 | 1688 test (human) | COMET-22, BLEU, chrF |
| 2 | Tiki test 4.451 (frozen) | pass-rate QC + COMET-QE |
| 3 | 300–500 unit Tiki gán tay | human preference |

Tầng 2 là nơi thể hiện khác biệt giữa 4 model; tầng 1 kiểm tra quên kiến thức;
tầng 3 chốt khi tầng 2 sát nhau (4B vs 8B).

## 7. Kế hoạch thực thi

| ưu tiên | việc | ước lượng |
|---|---|---|
| P0 | repair 17.632 dòng lỗi key | ~30 phút GPU |
| P0 | chốt QC gate + dedup + xuất merge-ready | ~1 giờ |
| P0 | trích 1688 thành pair table | ~1 giờ |
| P1 | merge 4 bộ + kiểm tra leakage/weight | ~1 giờ |
| P1 | frozen test + script đánh giá | ~2 giờ |
| P2 | gold set 300–500 unit | vài giờ người |

## 8. Rủi ro

1. **Key vocabulary lệch** (54,4% phủ): model có thể học key quen từ 1688 rồi áp
   sai cho key Tiki mới → cần giữ `key_vi` trong prompt khi train.
2. **Mật độ attribute lệch 10×**: 1688 dạy "liệt kê nhiều attribute", Tiki chỉ có
   2,46/sản phẩm → weight theo domain để tránh lệch hành vi.
3. **Synthetic noise**: 0.6B còn ~33% lỗi nếu không repair → không nên đưa vào
   cùng một thí nghiệm so sánh nếu chưa sửa.
4. **Đánh giá trên tập đã lọc**: dễ thổi phồng điểm; luôn báo pass-rate trên
   test đầy đủ.
