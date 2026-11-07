# Báo cáo thực nghiệm trọng số dữ liệu

> **CẢNH BÁO KỸ THUẬT (2026-11-07):** báo cáo này dùng sampler cũ
> `p_i ∝ w_g * t_i`; token share thực tế bị lệch do `sum(t_i²)`.
> Ví dụ A natural: attr 88,6% thay vì 65,1%; title 3,3% thay vì 16%.
> Kết quả chỉ nên xem là tham khảo; A/B sẽ chạy lại bằng quota-based sampling.


**Ngày:** 2026-11-07  
**Model:** Qwen3-4B + LoRA  
**Bài toán:** dịch thương mại điện tử Việt → Trung (1688 + Tiki)  
**Câu hỏi:** gán trọng số cho `title` / `attribute` / `description` / `Tiki` có làm tăng chất lượng model không?

---

## 1. Tóm tắt điều hành

Kết quả chính:

```text
CÓ. Gán trọng số làm tăng chất lượng so với natural.

Hai cấu hình tốt nhất:
  balanced      — mặc định, tăng đều title/attribute, giữ description
  title_heavy   — tốt nhất cho 1688 title

Hai cấu hình này tương đương trên Tiki:
  chênh +0,0019, CI 95% [-0,0086, +0,0116]

Loại:
  no_desc       — phá description 1688 (-0,0505 COMET)
  natural       — Tiki thấp nhất, 1688 title thấp
  tiki_heavy    — chỉ 1 seed, không vượt balanced/title_heavy
```

---

## 2. Thiết kế thực nghiệm

### 2.1. Dữ liệu

```text
train      : 1688 + Tiki, đã dedup/chunk
budget     : 5.000.000 token/run
optimizer  : 625 step/run, ~8.000 token/step
model      : Qwen3-4B, LoRA r=16, alpha=32, all-linear
precision  : bf16
hardware   : RTX Pro 6000 Blackwell 102GB
seed       : 42 và 123
```

Phân bố token natural quan sát được:

| Nhóm | Token share |
|---|---:|
| 1688 description | ~63,8% |
| 1688 attribute | ~23,5% |
| Tiki | ~6,5% |
| 1688 title | ~6,2% |

### 2.2. Eval set cố định

```text
1688 : 300 title + 300 attribute + 150 description
       COMET-da (reference-based)
Tiki : 300 title + 300 attribute
       COMET-QE (reference-free)
```

Mọi cấu hình dùng đúng cùng `eval_id`, cùng prompt, cùng decoding.

### 2.3. Metric và kiểm định

```text
primary   : COMET-QE Tiki macro (title + attribute)
guardrail : COMET-da 1688 macro không giảm quá 0,01
bootstrap : 2.000 lần paired theo eval_id
judge     : Gemma 4 31B, blind, credit AAA..C
```

Gemma dùng làm tín hiệu phụ; quyết định chính dựa trên COMET paired.

---

## 3. Toàn bộ trọng số đã thực nghiệm

Trọng số là **tỷ lệ token mục tiêu**, không phải tỷ lệ số dòng.

| Tên | 1688 title | 1688 attribute | 1688 description | Tiki | Ý tưởng |
|---|---:|---:|---:|---:|---|
| `natural` | ~6,2% | ~23,5% | ~63,8% | ~6,5% | token-proportional, baseline |
| `balanced` | 15% | 35% | 25% | 25% | cân bằng 1688 và Tiki |
| `no_desc` | 25% | 45% | 0% | 30% | bỏ description, dồn title/attr |
| `tiki_heavy` | 10% | 25% | 20% | 45% | ưu tiên domain Tiki |
| `title_heavy` | 30% | 30% | 15% | 25% | ưu tiên title |

Chưa chạy:

```text
desc_heavy
attr_heavy
```

Muốn kết luận riêng cho việc **tăng** description hoặc attribute, cần bổ sung 2 cấu hình này.

---

## 4. Kết quả COMET

### 4.1. Điểm theo stratum (2 seed nếu có)

| Cấu hình | 1688 title | 1688 attr | 1688 desc | Tiki title | Tiki attr | Tiki macro | 1688 macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| natural | 0,7920 | 0,8657 | 0,9143 | -0,0878 | 0,0645 | -0,0117 | 0,8573 |
| balanced | 0,8127 | 0,8675 | 0,9153 | -0,0312 | 0,0754 | 0,0221 | 0,8652 |
| no_desc | 0,8247 | 0,8621 | 0,8638 | -0,0370 | 0,0811 | 0,0221 | 0,8502 |
| tiki_heavy | 0,8008 | 0,8635 | 0,9141 | -0,0331 | 0,0836 | 0,0252 | 0,8595 |
| title_heavy | 0,8234 | 0,8658 | 0,9144 | -0,0326 | 0,0806 | 0,0240 | 0,8679 |

### 4.2. So với natural

| Cấu hình | Tiki macro diff | CI 95% | 1688 macro diff | CI 95% |
|---|---:|---|---:|---|
| balanced | +0,0337 | [+0,0201, +0,0480] | +0,0092 | [+0,0057, +0,0134] |
| no_desc | +0,0337 | [+0,0160, +0,0501] | +0,0015 | [-0,0031, +0,0067] |
| tiki_heavy | +0,0369 | [+0,0193, +0,0538] | +0,0026 | [-0,0020, +0,0074] |
| title_heavy | +0,0356 | [+0,0210, +0,0502] | +0,0126 | [+0,0086, +0,0169] |

`tiki_heavy` chỉ có seed 42; các cấu hình còn lại có 2 seed hoặc được so 2 seed.

### 4.3. So sánh trực tiếp balanced và title_heavy

```text
Tiki macro : +0,0019  CI [-0,0086, +0,0116]  -> không phân biệt
1688 macro : +0,0034  CI [+0,0008, +0,0059]  -> title_heavy nhỉnh nhẹ
```

Kết luận: hai cấu hình **đồng hạng**, không nên tuyên bố một cái hơn tuyệt đối.

---

## 5. Kết quả Gemma credit

Gemma chấm 2.000 dòng, blind, thang:

```text
AAA = 9, AA = 8, A = 7, BBB = 6, BB = 5,
B = 4, CCC = 3, CC = 2, C = 1
```

| Cấu hình | Mean | Grade | n |
|---|---:|---|---:|
| balanced | **7,868** | AA | 500 |
| title_heavy | 7,846 | AA | 500 |
| tiki_heavy | 7,704 | AA | 250 |
| no_desc | 7,700 | AA | 250 |
| natural | 7,680 | AA | 500 |

Theo seed:

```text
seed 42 : balanced 7,872 > title_heavy 7,744 > tiki_heavy 7,704
          > no_desc 7,700 > natural 7,600
seed 123: title_heavy 7,948 > balanced 7,864 > natural 7,760
```

Gemma cũng xác nhận `no_desc` làm description tệ đi:

```text
no_desc 1688 description: 6,556
```

---

## 6. Kết quả Phase 1 cũ

Phase 1 dùng metric heuristic “Tiki pass”, không cùng thang với COMET-QE; chỉ tham khảo.

| Cấu hình | Tiki pass | 1688 chrF | Tiki qty |
|---|---:|---:|---:|
| natural | **0,9630** | 0,5707 | 0,9755 |
| balanced | 0,9605 | 0,5664 | **0,9815** |
| no_desc | 0,9655 | 0,4905 | 0,9815 |
| tiki_heavy | 0,9610 | 0,5512 | 0,9805 |
| title_heavy | 0,9530 | **0,5733** | 0,9785 |

Lưu ý: v1 right-padding bị lỗi hoàn toàn; bảng trên là v2 left-padding.

---

## 7. Điểm mạnh / yếu từng cấu hình

### `natural`

**Mạnh**
- Không can thiệp, ít rủi ro làm lệch phân bố thật.
- Phase 1 có Tiki pass cao nhất, untranslated thấp nhất.
- 1688 description rất tốt: COMET 0,9143.

**Yếu**
- Tiki chỉ ~6,5% token → Tiki macro thấp nhất (-0,0117).
- 1688 title yếu nhất (0,7920).
- Không phù hợp nếu Tiki là mục tiêu chính.

**Dùng khi:** ưu tiên 1688, baseline an toàn.

---

### `balanced` — 15/35/25/25

**Mạnh**
- Tăng đều các stratum:
  - 1688 title +0,0208
  - 1688 attr +0,0018
  - 1688 desc +0,0010
  - Tiki title +0,0566
  - Tiki attr +0,0109
- 1688 macro +0,0092; Tiki macro +0,0337.
- Gemma credit cao nhất 7,868.
- Giữ description tốt, không phá 1688.

**Yếu**
- Description giảm từ ~64% xuống 25% token.
- Gain 1688 attribute nhỏ, CI gần 0.
- Không phải tốt nhất cho riêng 1688 title.

**Dùng khi:** cần một cấu hình mặc định cho cả 1688 + Tiki.

---

### `no_desc` — 25/45/0/30

**Mạnh**
- Dồn token cho title/attribute:
  - 1688 title +0,0327
  - Tiki title +0,0508
- Câu ngắn hơn, throughput tốt.
- Phase 1 Tiki pass 0,9655.

**Yếu**
- Phá description:
  - COMET 1688 description -0,0505
  - chrF 1688 chỉ còn 0,4905
  - Gemma description 6,556
- 1688 macro thấp nhất 0,8502.
- Không dùng được cho bài toán cần dịch description.

**Dùng khi:** model chỉ phục vụ title/attribute.

---

### `tiki_heavy` — 10/25/20/45

**Mạnh**
- Tiki chiếm 45% token, ưu tiên rõ domain đích.
- Tiki macro +0,0369 (seed 42), 1688 gần như không giảm.
- Tốt nếu sản phẩm cuối chỉ dùng cho Tiki.

**Yếu**
- Mới có 1 seed.
- Tiki macro thấp hơn title_heavy/balanced khi so 2 seed.
- 1688 title chỉ +0,0088.
- Gemma 7,704, thấp hơn balanced/title_heavy.
- Description chỉ 20%, có nguy cơ giảm nhẹ.

**Dùng khi:** Tiki-only, chấp nhận 1688 yếu hơn.

---

### `title_heavy` — 30/30/15/25

**Mạnh**
- 1688 title mạnh nhất: +0,0315.
- 1688 macro cao nhất: 0,8679 (+0,0126).
- Tiki macro +0,0356; description 1688 gần như đi ngang.
- Gemma 7,846.

**Yếu**
- 1688 attribute gần như không tăng (+0,0001).
- Description chỉ 15% token; không tăng desc.
- Tiki ngang balanced, chênh +0,0019 với CI chứa 0.
- Gemma thấp hơn balanced 0,022, không đáng kể nhưng không vượt.

**Dùng khi:** ưu tiên 1688 title.

---

## 8. Kết luận và khuyến nghị

```text
Mặc định tốt nhất : balanced
Ưu tiên 1688 title: title_heavy
Tiki-only         : tiki_heavy (cần thêm seed)
Không nên dùng    : no_desc
Baseline an toàn  : natural
```

`balanced` và `title_heavy` là **đồng hạng**:

- Tiki chênh +0,0019, CI chứa 0.
- 1688 chênh +0,0034, title_heavy nhỉnh nhẹ.
- Gemma chênh 0,022, balanced nhỉnh nhẹ.

Không có cơ sở để nói một cấu hình vượt trội tuyệt đối. Chọn theo mục tiêu triển khai.

---

## 9. Hạn chế

1. Tiki không có reference người; phải dùng COMET-QE.
2. QE dùng `wmt20-comet-qe-da`; điểm tuyệt đối thấp, nhưng so sánh paired vẫn hợp lệ.
3. `no_desc` và `tiki_heavy` chỉ có seed 42.
4. Chưa chạy `desc_heavy` và `attr_heavy`, nên chưa kết luận được việc **tăng** description/attribute.
5. Tiki target là synthetic; không dùng làm gold reference.
6. Phase 1 dùng metric heuristic, không so trực tiếp với COMET.

---

## 10. Đường dẫn tái lập

```text
data/cleaned/weight_eval/eval_1688.parquet
data/cleaned/weight_eval/eval_tiki.parquet
data/cleaned/weight_eval/train_view_4b.parquet

tools/train_weight_gpu.py
tools/tpu_weight_train.py
tools/score_comet_weight.py
tools/build_judge_input_multi.py
tools/gemma_judge_credit.py
tools/aggregate_weight_eval.py
tools/report_credit.py

reports/WEIGHT_EVAL_FINAL.md
reports/WEIGHT_EVAL.md
```

Dataset:

```text
https://huggingface.co/datasets/ntquang0410/tiki-4model-eval
```