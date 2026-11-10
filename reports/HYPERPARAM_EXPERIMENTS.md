# Thực nghiệm hyperparameter cho mix title / attribute / description (bản 2)

Ngày: 2026-11-01. Thay thế `HYPERPARAM_PLAN.md` bằng bản actionable hơn:
5 cách thực nghiệm, so sánh, và lộ trình cụ thể theo ngân sách GPU hiện có
(23,76h còn lại).

## 0. Ràng buộc & số liệu nền

Token mass đo trên pair table dự kiến:

| nguồn | cặp | token | tỉ lệ tự nhiên |
|---|---:|---:|---:|
| 1688 description | 16.133 | 8,38M | **55,5%** |
| 1688 attribute | 395.575 | 4,80M | 31,8% |
| Tiki MT | 45.709 | 1,05M | 7,0% |
| 1688 title | 16.133 | 0,87M | 5,8% |

Ràng buộc thực tế:

- Test Tiki cố định **4.451 unit** (dùng chung 4 model).
- 1688 test 817 sản phẩm (human) để kiểm tra quên domain.
- Ngân sách ~20h GPU; 0.6B/1.7B/4B/8B đều có sẵn.
- Mọi run phải so sánh được → **cùng token budget**, cùng seed, cùng eval.

## 1. Năm cách thực nghiệm

### Cách 1 — Leave-one-out ablation (LOO)

```text
R0: title + attr + desc + tiki   (baseline)
R1: bỏ desc
R2: bỏ title
R3: bỏ attr
R4: chỉ tiki
```

- **Trả lời:** nguồn nào đóng góp thực sự, nguồn nào chỉ tốn token.
- **Chi phí:** 5 run × 3M token (0.6B) ≈ 40 phút.
- **Ưu:** rẻ nhất, kết luận rõ ràng, ít giả định.
- **Nhược:** không tìm được tỉ lệ tối ưu, chỉ biết "có/không".

**Nên chạy đầu tiên.** Nếu bỏ desc mà Tiki không giảm, loại desc khỏi mọi thí
nghiệm sau → tiết kiệm một chiều.

### Cách 2 — OFAT (one-factor-at-a-time)

```text
baseline → đổi share_tiki (15/25/40%)
         → đổi attr_cap (5/15/all)
         → đổi desc_mode (off/chunk/full)
```

- **Trả lời:** xu hướng từng trục, nhanh.
- **Chi phí:** ~8 run × 5M token (0.6B) ≈ 1,5h.
- **Ưu:** dễ hiểu, dễ debug.
- **Nhược:** bỏ qua tương tác (desc dài × attr cap, tiki share × model size).

### Cách 3 — Factorial / fractional factorial

```text
2^3 yếu tố: share_tiki {15,35%} × attr_cap {5,20} × desc_mode {off,chunk}
= 8 run, đo cả main effect và interaction
```

- **Trả lời:** tương tác giữa các trục — thứ OFAT bỏ sót.
- **Chi phí:** 8 run × 5M token (0.6B) ≈ 1,5h.
- **Ưu:** hiệu quả thông tin/chi phí cao nhất khi có 3–4 yếu tố.
- **Nhược:** chỉ 2 mức mỗi yếu tố; không tối ưu tinh.

### Cách 4 — Random search + Bayesian (Optuna TPE)

```text
4 chiều liên tục: share_tiki, share_desc, attr_cap, desc_max_chars
8–10 trial, pruning sau 25% budget (ASHA)
```

- **Trả lời:** vùng tối ưu, có thể vượt xa lưới factorial.
- **Chi phí:** 10 trial × 10M token (1.7B) ≈ 4–5h.
- **Ưu:** tận dụng kết quả các trial trước; phù hợp 4+ chiều.
- **Nhược:** cần hạ tầng logging tốt; kết quả phụ thuộc seed.

### Cách 5 — Học trọng số domain (DoReMi / Group DRO)

```text
1. Train proxy 0.6B với group DRO trên 4 domain
2. Lấy domain weights học được
3. Áp weights đó cho 4B/8B
```

- **Trả lời:** tỉ lệ tối ưu theo mục tiêu, không cần lưới.
- **Chi phí:** ~10h GPU (proxy + áp dụng).
- **Ưu:** phương pháp mạnh nhất về lý thuyết, đã chứng minh ở LM.
- **Nhược:** phức tạp, khó debug, cần eval sạch để không overfit test.

### So sánh nhanh

| cách | run | giờ (0.6B) | tìm được tỉ lệ? | tương tác? | rủi ro |
|---|---|---:|---|---|---|
| LOO | 5 | 0,7 | không | không | thấp |
| OFAT | 8 | 1,5 | một phần | không | thấp |
| Factorial 2³ | 8 | 1,5 | có (2 mức) | **có** | thấp |
| Bayes/ASHA | 10 | 4–5 (1.7B) | **có** | có | trung bình |
| DoReMi | 1+ | 10 | **có** | có | cao |

## 2. Lộ trình đề xuất (theo ngân sách 20h)

### Pha 0 — offline, miễn phí (~15 phút)

Với từng cấu hình: tính token share thực tế sau lọc/cắt, loại nếu lệch mục tiêu
>2 điểm %, hoặc vi phạm: Tiki <10%, title <5%, desc >40%.

### Pha 1 — LOO trên 0.6B (5 run × 3M token ≈ 40 phút)

Mục tiêu: loại nguồn vô dụng. Quyết định desc có xứng đáng 55% token không.

### Pha 2 — Factorial 2³ trên 0.6B (8 run × 5M token ≈ 1,5h)

```text
share_tiki  : 15% | 35%
attr_cap    : 5   | 20
desc_mode   : off | chunk400
```

Phân tích main effect + interaction. Chọn 2 cấu hình tốt nhất.

### Pha 3 — Bayesian multi-fidelity trên 4B (8 trial × 10M token ≈ 4h)

- Optuna TPE, pruning ASHA sau 25% budget.
- 4 chiều: `share_tiki`, `share_desc`, `attr_cap`, `desc_max_chars`.
- Ràng buộc tổng share = 100%.

### Pha 4 — Xác nhận trên 8B (3 run × 20M token × 2 seed ≈ 4,5h)

Top 2 từ Pha 3 + baseline. Chỉ cấu hình thắng mới dùng để so 4 model.

**Tổng: ~11h**, còn dư ~9h cho lần chạy lại nếu kết quả không ổn định.

## 3. Thiết kế đo lường

### 3.1. Metric theo pair type

Không chỉ nhìn điểm tổng. Báo cáo tách:

| nhóm | Tiki test | 1688 test |
|---|---|---|
| title | pass-rate + COMET-QE | COMET-22 |
| attribute | pass-rate + number consistency | COMET-22 |
| description | COMET-QE | COMET-22 |

Mix ảnh hưởng khác nhau lên từng loại; nhìn tổng sẽ che mất trade-off.

### 3.2. Multi-fidelity

```text
3M token  → xếp hạng thô, loại 50%
5M token  → chọn top 4
10M token → chọn top 2
20M token → xác nhận
```

Cùng một seed cho mọi mức để so sánh theo cặp.

### 3.3. Kiểm định

- Paired bootstrap 1.000 lần trên test unit → 95% CI.
- Với 4.451 unit, CI ≈ ±1,3% → chênh <2% chưa kết luận.
- 2 seed ở pha xác nhận; báo cáo cả 2, không chỉ seed tốt.

### 3.4. Guardrail

Loại cấu hình nếu:

```text
untranslated rate tăng >2 điểm
number consistency giảm >3 điểm
1688 test giảm >2 điểm COMET
output length ratio lệch >20% so baseline
```

## 4. Cấu hình chạy

```yaml
run:
  model: qwen3-0.6b        # pha 1-2
  method: lora             # giữ nguyên mọi pha để so sánh được
  token_budget: 5000000
  max_seq: 1280
  seed: 42
  eval:
    tiki_test: frozen_4451
    tiki_metrics: [pass_rate, comet_qe, number_consistency]
    g1688_test: human_817
mix:
  share_title_1688: 0.15
  share_attr_1688: 0.35
  share_desc_1688: 0.25
  share_tiki: 0.25
```

Bắt buộc log: config hash, tokens thực tế theo domain, số cặp, steps, metric + CI.

## 5. Vì sao không dùng epoch

Hai cấu hình khác weight sẽ thấy số token khác nhau trong cùng một epoch. So
sánh theo epoch là so sánh sai. Token budget cố định là điều kiện duy nhất để
kết luận "mix nào tốt hơn".

## 6. Rủi ro & đối sách

| rủi ro | đối sách |
|---|---|
| Test Tiki nhiễu (4.451 unit) | paired bootstrap + 2 seed |
| Overfit vào 1688 test | metric chính là Tiki, 1688 chỉ guardrail |
| Desc cắt giữa câu | cắt theo đoạn/câu, không cắt ký tự thô |
| Weight tính theo row | sampler theo token mass |
| Chạy nhiều cấu hình → multiple testing | giới hạn 10 trial, hiệu chỉnh Bonferroni nếu cần |
| LoRA vs full FT khác kết luận | giữ một phương pháp cho toàn bộ thí nghiệm |

## 7. Kết luận đề xuất

```text
1. LOO trước (40 phút) — trả lời "desc có đáng 55% token không"
2. Factorial 2³ (1,5h) — đo tương tác, chọn 2 cấu hình
3. Bayesian 4B (4h) — tối ưu tinh
4. Xác nhận 8B (4,5h) — 2 seed
```

Nếu chỉ có 2h: chạy LOO + factorial, bỏ Pha 3–4.
Nếu muốn tối đa chất lượng: thay Pha 2–3 bằng DoReMi (10h) sau khi LOO xác nhận
nguồn nào đáng giữ.
