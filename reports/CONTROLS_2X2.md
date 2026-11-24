# Control cells — 2×2 quota×field, seed 123 và NLLB

**Cập nhật:** 2026-11-17 10:10 UTC  
**Eval:** `eval_F_v2_common` — 831 dòng (1688: 231, Tiki: 600)  
**Chấm:** COMET-da `Unbabel/wmt22-comet-da` (1688), COMET-QE `Unbabel/wmt20-comet-qe-da` (Tiki), batch 8–16 trên A100 40GB.

---

## 1. Ma trận 2×2 Qwen3-4B (seed 42)

Hai biến được tách riêng: **tập field** (A = title+attribute, B = title+description) và **cách lấy mẫu** (natural = giữ phân bố gốc, balanced = 25% token mỗi nhóm).

| Cell | Field | Sampling | COMET 1688 | COMET-QE Tiki |
|---|---|---|---:|---:|
| `qwen4b_a` | A | natural | **0,8584** | **0,0573** |
| `qwen4b_a_balanced` | A | balanced | 0,8477 | 0,0524 |
| `qwen4b_b` | B | balanced | 0,8471 | 0,0461 |
| `qwen4b_b_natural` | B | natural | 0,8436 | 0,0471 |

Paired bootstrap 2.000 lần, baseline `qwen4b_a`:

| Cell | Δ 1688 | Δ Tiki |
|---|---:|---:|
| A-balanced | −0,0107 [−0,0202; −0,0019]* | −0,0049 [−0,0200; +0,0104] |
| B-balanced | −0,0113 [−0,0217; −0,0014]* | −0,0112 [−0,0262; +0,0046] |
| B-natural | −0,0148 [−0,0271; −0,0027]* | −0,0102 [−0,0265; +0,0059] |

### Đọc kết quả

1. **A-natural tốt nhất**, hơn 3 cell còn lại 0,011–0,015 COMET trên 1688 (đều có ý nghĩa thống kê).
2. **Quota balanced không tạo lợi thế:** trên A nó làm giảm 0,0107 (có ý nghĩa); trên B nó chỉ nhỉnh hơn natural +0,0035 (không ý nghĩa).
3. **Hiệu ứng field nhỏ hơn hiệu ứng sampling:** B-natural (0,8436) ≈ A-balanced (0,8477) — chênh 0,004. Trước đây khi chỉ so A-natural với B-balanced, ta thấy khoảng cách 0,011 và dễ quy cho "B thiếu attribute"; thực tế phần lớn khoảng cách đến từ việc **đổi sang balanced**.
4. Trên Tiki QE, mọi khác biệt đều **không có ý nghĩa** (CI chứa 0) — QE không phân biệt được 4 cell này.

> Hệ quả cho báo cáo: không được viết "B-balanced tốt hơn/tệ hơn A-natural vì field mix" khi chưa tách quota. Với Qwen3-4B, **A-natural là cấu hình nên dùng**.

## 2. NLLB-200-distilled-600M (control pretrained NMT)

Cùng quota 4M token, cùng `eval_F_v2_common`, LoRA r16.

| Cell | COMET 1688 | COMET-QE Tiki | n 1688 | n Tiki |
|---|---:|---:|---:|---:|
| NLLB A | 0,6344 | −0,5379 | 231 | 600 |
| NLLB B | 0,5517 | −0,6753 | 230 | 597 |
| OpenNMT A (from-scratch) | 0,3488 | −1,4177 | 231 | 600 |
| OpenNMT B-natural | 0,3387 | −1,4074 | 231 | 600 |

**Đọc:** NLLB pretrained 600M **hơn OpenNMT from-scratch 37M rất xa** (+0,29/+0,21 trên 1688), xác nhận kết luận trong `OPENNMT_FAIRNESS.md`: vấn đề của OpenNMT là **pretraining + capacity + recipe**, không phải "paradigm NMT". Tuy vậy NLLB vẫn kém Qwen3-4B ~0,22 COMET trên 1688 và QE âm trên Tiki — pretrained NMT cỡ 600M không thay thế được LLM trong thiết lập này.

## 3. Seed 42 vs 123 (Qwen3-4B)

| Cell | Seed 42 | Seed 123 | Δ (123−42) | CI 95% |
|---|---:|---:|---:|---|
| A-natural 1688 | 0,8584 | 0,8511 | **−0,0072** | [−0,0144; −0,0009]* |
| A-natural Tiki QE | 0,0573 | 0,0637 | +0,0064 | [−0,0040; +0,0174] |
| B-balanced 1688 | 0,8471 | 0,8479 | +0,0008 | [−0,0030; +0,0048] |
| B-balanced Tiki QE | 0,0461 | 0,0517 | +0,0056 | [−0,0054; +0,0171] |

**Đọc kết quả — quan trọng cho mọi kết luận so sánh:**

1. Seed variance **không nhỏ như ước lượng ban đầu từ B-balanced**. Với A-natural, đổi seed làm điểm đổi **−0,0072** trên 1688 (CI không chứa 0). Với B-balanced chỉ +0,0008. Vậy seed noise phụ thuộc cấu hình, ước lượng tốt nhất hiện có là **0,005–0,007 COMET trên 1688**.
2. Khoảng cách A-natural (0,8584) so với A-balanced (0,8477) là **0,0107**, tức chỉ ~1,5 lần seed noise. Điểm của A-balanced thấp hơn **cả hai** seed của A-natural (0,8584 và 0,8511), nên hiệu ứng "balanced làm giảm A" vẫn có khả năng thật, nhưng **không đủ mạnh để tuyên bố chắc chắn** khi chỉ có 1 seed cho A-balanced.
3. Hệ quả: câu "A-natural tốt nhất" phải viết là **"A-natural có điểm cao nhất, nhưng mức hơn 0,011 nằm trong khoảng 1–1,5 lần dao động seed; cần ≥3 seed để kết luận"**. Không dùng paired bootstrap (chỉ phủ variance của eval) làm CI cho so sánh giữa cấu hình.
4. Trên Tiki QE, mọi Δ seed đều không có ý nghĩa.

## 4. Trạng thái artefact

| File | Nội dung |
|---|---|
| `.artifacts/hyp/hyp_qwen4b_a_balanced_seed42.parquet` | A-balanced, 1.500 dòng, 15 rỗng |
| `.artifacts/hyp/hyp_qwen4b_b_natural_seed42.parquet` | B-natural, 1.500 dòng, 15 rỗng |
| `.artifacts/hyp/hyp_qwen4b_b_balanced_seed123.parquet` | B-balanced seed 123, 1.500 dòng, 15 rỗng (đã chấm: 0,8479 / 0,0517) |
| `.artifacts/hyp/hyp_qwen4b_a_seed123.parquet` | A-natural seed 123, 1.500 dòng, 15 rỗng (đã chấm: 0,8511 / 0,0637) |
| `.artifacts/hyp/hyp_nllb_a_seed42.parquet` / `hyp_nllb_b_seed42.parquet` | NLLB A/B, 1.500 dòng (2 / 9 rỗng) |
| `.artifacts/comet_gpu/scores/` | Điểm COMET 5 cell mới + 10 cell cũ (v2_common) |

## 5. Việc còn lại

- (đã xong) inference + chấm COMET `a-s123`.

2. Cập nhật `MATRIX_5X2_EVAL_F_V2.md` §9 với kết luận 2×2 (A-natural là cấu hình chính).
3. Bổ sung NLLB vào `OPENNMT_FAIRNESS.md` §6.

## 6. Cell bổ sung — chấm lại trên `eval_F_v2_common` (2026-11-24 17:10 UTC)

Chấm bằng `venv_comet` trên Colab A100, lọc đúng 831 `eval_id` của `data/cleaned/eval_F_v2_common.parquet` (1688 = 231, Tiki = 600). **Đã có score từng dòng** trong `.artifacts/comet_s7_common/` (chưa chạy paired bootstrap cho các cell này).

| Cell | 1688 da | Tiki QE | n (1688/Tiki) | Trạng thái artefact |
|---|---:|---:|---|---|
| Qwen3-8B A-balanced seed 42 | 0,8553 | 0,0562 | 231/600 | ✅ `hyp_qwen8b_a_balanced_seed42.parquet` (điểm cũ, chưa score lại từng dòng) |
| Qwen3-8B B-natural seed 42 | 0,8514 | 0,0624 | 231/600 | ⚠️ **PARTIAL** — hyp thất lạc; chờ re-infer từ `qwen8b-b-ckpt` step 500 |
| Qwen3-4B B-balanced seed 7 | ~~0,8010~~ | ~~0,0774~~ | 231/600 | ❌ **KHÔNG DÙNG** — kernel infer chọn nhầm adapter `sample_submission/adapters/main_lora` của competition; đang chạy lại từ adapter thật (475 → 500 step) |
| NLLB A-natural seed 7 | 0,6304 | −0,5427 | 231/600 | ✅ `hyp_nllb_a_seed7.parquet` (1.500 dòng, 2 rỗng) |
| NLLB B-balanced seed 7 | 0,6382 | −0,5192 | 231/600 | ✅ `hyp_nllb_b_seed7.parquet` (1.500 dòng, 0 rỗng) |
| EcomGPT-7B A-balanced seed 42 (tái lập) | 0,8156 | 0,0020 | 231/600 | ✅ `hyp_ecomgpt_a_balanced_seed42.parquet`; adapter step 500 ở `nguynthanhquang/ecomgpt-ckpt` |

**Đọc kết quả:**

1. **NLLB B seed 7 (0,6382) cao hơn NLLB B seed 42 (0,5517) rất nhiều**, trong khi seed 42 cho B < A. Vì vậy kết luận cũ "NLLB B kém A" là **nhiễu seed**, không phải hiệu ứng dữ liệu — với control pretrained NMT, chênh lệch A/B nằm trong dao động seed.
2. **EcomGPT A tái lập 0,8156 vs điểm cũ 0,8250** (Δ 0,0094): cùng bậc, khác biệt do lần chạy mới bỏ mega-batch ở step 500 và cắt ngắn chunk cuối (xem `CHECKPOINT_2026-11-24_1640.md` §3). Bản tái lập là bản có artefact đầy đủ (adapter step 500), nên dùng bản này.
3. **Bài học kiểm tra artefact:** kernel infer dùng `rglob('adapter_config.json')` trên `/kaggle/input` có thể bắt nhầm adapter mẫu của competition. Đã vá builder resume; mọi cell cần assert đường dẫn adapter chứa tên dataset đã train.

Bảng NLLB ở §2 chỉ nên trích cho seed 42; khi trích seed 7 phải ghi rõ là cell bổ sung chưa có CI.

## 7. EcomGPT-7B — hai cell control bổ sung (2026-11-24)

| Cell | 1688 da | Tiki QE | Kết luận |
|---|---:|---:|---|
| EcomGPT A-balanced seed 42 | 0,8156 | 0,0020 | ✅ hợp lệ (adapter step 500 trên Kaggle) |
| EcomGPT B-natural seed 42 | — | — | ❌ **FAILED** — generation thoái hoá |

**B-natural thoái hoá:** train đủ 500 step nhưng loss về ~1e-4 rất sớm; khi inference, **1.471/1.500 hyp rỗng**, 29 dòng còn lại là chuỗi lặp `"......"` / `"* * * *"`. Không chấm COMET cell này (điểm sẽ vô nghĩa). Artefact giữ để điều tra: `.artifacts/hyp/hyp_ecomgpt_b_natural_seed42_FAILED.parquet`.

**Giả thuyết:** B-natural rút 13.601 dòng (trung bình 294 token/dòng) → mỗi step chỉ ~20 dòng; target là description dài, nhiều boilerplate lặp giữa các sản phẩm → model học thuộc rồi suy biến khi sinh tự do. Recipe A2 vốn chọn `balanced` cho dataset B, không phải `natural`. Muốn có cell B-natural hợp lệ cần hạ LR (≤5e-5) hoặc dừng sớm (~200 step) — chưa làm.

**Hệ quả cho báo cáo:** không dùng cell EcomGPT B-natural để so sánh; EcomGPT A-balanced là control hợp lệ duy nhất bổ sung hôm nay.
