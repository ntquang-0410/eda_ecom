# Đề xuất thực nghiệm hyperparameter cho mix title / attribute / description

Ngày: 2026-10-31. Bối cảnh: chuẩn bị dựng pair table 1688 + Tiki để fine-tune
4 model Qwen3 và chọn baseline.

## 0. Vì sao phải thực nghiệm mix

Đo token thực tế (ước lượng VI/3,6 + ZH/1,6 ký tự/token):

| nguồn | cặp | token | tỉ lệ tự nhiên |
|---|---:|---:|---:|
| 1688 description | 16.133 | 8,38M | **55,5%** |
| 1688 attribute | 395.575 | 4,80M | 31,8% |
| Tiki MT | 45.709 | 1,05M | 7,0% |
| 1688 title | 16.133 | 0,87M | 5,8% |
| **tổng** | | **15,1M** | |

Nếu trộn theo số dòng, description chiếm hơn nửa token. Title — loại dữ liệu sát
với Tiki nhất — chỉ 5,8%. Vì vậy phải điều chỉnh weight theo **token mass**, không
theo row count.

## 1. Hyperparameter và miền giá trị

### 1.1. Mix weight (theo token)

| tham số | miền | mặc định |
|---|---|---|
| `share_title_1688` | 5–25% | 15% |
| `share_attr_1688` | 25–50% | 35% |
| `share_desc_1688` | 0–35% | 25% |
| `share_tiki` | 10–40% | 25% |

Ràng buộc: tổng = 100%. Weight từng nguồn:

```text
w_source = share_source / tokens_source
sample ~ w_source
```

Cách này khiến tỉ lệ token quan sát được đúng bằng `share_*`, thay vì phó mặc
cho độ dài văn bản.

### 1.2. Attribute

| tham số | miền | mặc định | lý do |
|---|---|---|---|
| `attr_cap` (số attribute/sản phẩm) | 5 / 10 / 15 / 20 / all | 15 | 1688 có p50 22, p95 41 |
| `attr_pack` | `per_value` / `joined` | `per_value` | 1 cặp/1 value sạch hơn |
| `attr_with_key` | True / False | True | key giúp phân biệt "S" size vs "S" màu |
| `attr_max_values` (nếu joined) | 3 / 5 / 10 | 5 | tránh cặp quá dài |

### 1.3. Description

| tham số | miền | mặc định |
|---|---|---|
| `desc_mode` | off / full / chunk | chunk |
| `desc_max_chars` | 400 / 800 / 1200 / 2000 | 800 |
| `desc_chunk_chars` | 300 / 400 / 500 | 400 |
| `desc_overlap` | 0 / 50 | 0 |
| `desc_weight` | 0,3 / 0,5 / 1,0 | 0,5 |

Lưu ý: description VI p50 986 ký tự, p95 2.354 → nhiều dòng vượt 1.280 token nếu
ghép cả ZH. `chunk` cắt theo đoạn/câu để không vượt `max_seq`.

### 1.4. Lọc & trọng số phụ

| tham số | miền | mặc định |
|---|---|---|
| `dedup` | on / off | on |
| `dup_weight` | on / off | on (`dup_count`) |
| `number_mismatch_weight` | 1,0 / 0,5 / 0 | 0,5 |
| `partial_vi_weight` | 1,0 / 0,5 | 0,5 |
| `flag_truncated_weight` | 1,0 / 0,5 / 0 | 0,5 |
| `qc_min` | `ok` / `repaired` | `ok+repaired` |

### 1.5. Huấn luyện (trục riêng, không trộn vào mix)

`max_seq` 1.024 / 1.280 / 1.536; token budget; LR; LoRA rank; effective batch.

## 2. Thiết kế thực nghiệm 3 pha

### Pha 0 — sàng lọc offline (không train, ~10 phút)

Với mỗi cấu hình weight:
1. Tính token mass thực tế sau lọc/cắt.
2. Kiểm tra share quan sát được lệch mục tiêu < 2%.
3. Loại cấu hình vi phạm: Tiki < 10%, title < 5%, desc > 40%.
4. Kiểm tra vocab coverage (key/domain) và duplicate rate.

Mục tiêu: chỉ đưa 6–8 cấu hình hợp lệ vào train.

### Pha 1 — proxy screening (0.6B, ngân sách cố định)

- **Token budget cố định**: 5M token/cấu hình (không dùng epoch).
- 3 seed, cùng thứ tự dữ liệu sau khi seed.
- Train 0.6B, eval trên:
  - Tiki test 4.451 (frozen)
  - 1688 test 817 sản phẩm
- Chọn top 2 theo metric tổng hợp.

Thiết kế: **OFAT trước, factorial sau**

```text
Vòng 1 (OFAT, 6 runs): thay đổi từng trục so với baseline
  baseline, desc off, desc full, attr cap 5, attr cap all, dedup off
Vòng 2 (factorial 2^3, 8 runs): share_tiki × attr_cap × desc_mode
```

### Pha 2 — tinh chỉnh (1.7B hoặc 4B, ~20M token/run)

- Bayesian optimization (Optuna TPE) trên 4 chiều liên tục:
  `share_tiki`, `share_desc`, `attr_cap`, `desc_max_chars`.
- 8–10 trials, pruning sau 25% budget nếu metric dưới median.
- Ràng buộc tổng share = 100%.

### Pha 3 — xác nhận (8B, full budget)

- 2 cấu hình tốt nhất + baseline.
- Chạy 2 seed để kiểm tra độ ổn định.
- Chỉ cấu hình thắng mới dùng để chọn baseline giữa 4 model.

## 3. Metric và kiểm định

| tầng | metric | vai trò |
|---|---|---|
| chính | Tiki test **pass-rate QC** + COMET-QE | chọn cấu hình |
| phụ | 1688 test COMET-22, chrF | chống quên domain |
| guardrail | untranslated rate, number consistency, length ratio | loại cấu hình hỏng |
| phụ | latency, VRAM | điều kiện triển khai |

**Kiểm định:**

- Paired bootstrap 1.000 lần trên test unit, báo 95% CI.
- Chỉ coi là thắng khi chênh lệch > CI và nhất quán ở ≥2 seed.
- Với 4.451 unit, CI của pass-rate khoảng ±1,3% → đừng tin chênh lệch < 2%.

## 4. Cách triển khai

### 4.1. Config

```yaml
mix:
  share_title_1688: 0.15
  share_attr_1688: 0.35
  share_desc_1688: 0.25
  share_tiki: 0.25
attribute:
  cap: 15
  pack: per_value
  with_key: true
description:
  mode: chunk
  max_chars: 800
  chunk_chars: 400
filter:
  dedup: true
  dup_weight: true
  number_mismatch_weight: 0.5
train:
  max_seq: 1280
  token_budget: 20000000
  seed: 42
```

### 4.2. Sampler theo token mass

```python
tokens = sum(len(tok(x)) for x in pairs)
w = share / tokens
idx = np.random.choice(n, size=batch, p=w/w.sum())
```

### 4.3. Logging bắt buộc

Mỗi run ghi: config hash, token thực tế theo domain/type, số cặp, số epoch hiệu
dụng, metric + CI, thời gian, VRAM. Thiếu token exposure thì kết quả không so
sánh được.

## 5. Chi phí ước tính

| pha | model | runs | token/run | thời gian |
|---|---|---:|---:|---|
| 1 | 0.6B | 8 | 5M | ~1,5h |
| 2 | 4B | 8 | 20M | ~5–6h |
| 3 | 8B | 3 | 40M | ~4–5h |

Tổng ~12h GPU, nằm trong quota 30h/tuần.

## 6. Cạm bẫy cần tránh

1. **So sánh theo epoch** → mỗi cấu hình thấy lượng token khác nhau. Luôn dùng
   token budget.
2. **Weight theo row** → description dài lấn át. Luôn weight theo token.
3. **Đánh giá trên tập đã lọc** → pass-rate bị thổi phồng. Giữ test đủ 4.451.
4. **Một seed** → nhiễu lớn hơn hiệu ứng. Tối thiểu 2 seed ở pha xác nhận.
5. **Tối ưu trên 1688 test** → có thể hy sinh Tiki. Metric chính phải là Tiki.
6. **Description cắt tuỳ tiện** → cắt giữa câu làm hỏng cặp. Cắt theo đoạn/câu.
7. **Không log token exposure** → không biết cấu hình thực sự học gì.

## 7. Thứ tự đề xuất

```text
1. Dựng pair table với config mặc định (baseline)
2. Pha 0: kiểm tra token share, loại cấu hình sai
3. Pha 1: 6–8 run 0.6B, chọn top 2
4. Pha 2: Optuna 4B
5. Pha 3: xác nhận 8B, rồi mới so 4 model
```
