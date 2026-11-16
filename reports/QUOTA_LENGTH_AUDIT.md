# Quota sampling & length bias — audit và phạm vi kết luận

**Ngày:** 2026-11-16  
**Nguồn:** `review_audit.json` (khóa `quota`, `common_eval_ids`, `tiki_description_common_subset`), sample parquet thật của kernel, tokenizer Qwen3-4B.

---

## 1. Điều đã sửa đúng

Quota sampler `tools/quota_sampling.py` lấy đều dòng trong từng nhóm đến hạn ngạch token, thay vì `p ∝ t`. Manifest thật xác nhận **trong nội bộ nhóm** phân bố độ dài được bảo toàn:

| cell B-balanced | 1688_desc | 1688_title | tiki_desc | tiki_title |
|---|---:|---:|---:|---:|
| Token share (target 25%) | 25,00% | 25,00% | 25,00% | 25,00% |
| Median sample / population | 478,1 / 472,5 | 46,5 / 46,5 | 631,0 / 623,8 | 34,4 / 34,4 |
| P90 sample / population | 794,2 / 775,3 | 53,2 / 53,2 | 960,0 / 960,0 | 57,9 / 57,9 |

→ Lỗi "chỉ lấy câu dài/ngắn trong nhóm" đã được sửa. Đây là kết luận **được phép**.

## 2. Length bias còn ở cấp thí nghiệm

### 2.1. Exposure theo nhóm (A/B)

| recipe | nhóm | population rows | sample draws | unique | coverage | epoch token | draws/row | rows/step | target >960* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1688_attribute | 70.114 | 62.491 | 41.252 | 58,8% | 0,885 | 1,51 | 125,0 | 0,62% |
| A | 1688_title | 15.350 | 13.582 | 9.038 | 58,9% | 0,885 | 1,50 | 27,2 | 0% |
| A | tiki_attribute | 14.012 | 12.460 | 8.224 | 58,7% | 0,885 | 1,52 | 24,9 | 0% |
| A | tiki_title | 11.653 | 10.293 | 6.839 | 58,7% | 0,885 | 1,51 | 20,6 | 0% |
| **A** | **TOTAL** | **111.129** | **98.826** | **65.353** | **58,8%** | **0,885** | **1,51** | **197,7** | **0,1%** |
| B | 1688_description | 15.986 | 2.025 | 1.894 | 11,8% | **0,128** | 1,07 | **4,05** | **4,94%** |
| B | 1688_title | 15.350 | 21.449 | 11.558 | 75,3% | **1,397** | 1,86 | 42,9 | 0% |
| B | tiki_description | 11.017 | 1.602 | 1.488 | 13,5% | **0,146** | 1,08 | **3,20** | **30,40%** |
| B | tiki_title | 11.653 | 27.055 | 10.513 | 90,2% | **2,318** | 2,57 | 54,1 | 0% |
| **B** | **TOTAL** | **54.006** | **52.131** | **25.453** | **47,1%** | **0,253** | **2,05** | **104,3** | **0,6%** |

\* target bị cắt tính từ `tokens_est` **không cap** (`src_chars/3.6 + tgt_chars/1.6 > 960`). Bản cap 960 làm mất đuôi: 1688_desc 4,5% và tiki_desc 12,9% theo cách đếm cũ; uncapped cho thấy tiki_desc tới **30,4%**.

**Kết luận:** `balanced token` ≠ `balanced learning`. B chỉ có **3,2–4,1 mẫu description/step**, trong khi title bị lặp 1,4–2,3 lần; **~87% description không bao giờ được học**. Scalar `epochs_seen=0,2529` che khoảng biến thiên **0,128×–2,318× (18×)**.

### 2.2. Token accounting

- `tokens_est = |src|/3.6 + |tgt|/1.6`, `eff_tokens = clip(960)` — **không phải tokenizer thật**, và cap 960 làm giảm khối lượng đuôi dài.
- Tokenizer Qwen3-4B thật:

| recipe | sample thật (Qwen tokens) | population thật | epoch thật |
|---|---:|---:|---:|
| A | 6,560M | 7,408M | 0,8855 |
| B | 5,926M | 23,920M | 0,2477 |

→ "4,000M token" thực tế là **6,56M / 5,93M Qwen tokens**. Con số epoch gần như không đổi (0,885 / 0,248), nhưng absolute token budget phải ghi đúng.

### 2.3. Eval protocol

- `eval_F` gốc: n khác nhau theo cell (1688: 747/750; Tiki: 738/729/750), vì Qwen thiếu 15 prompt >952 token và EcomGPT-B thiếu 21 Tiki attribute.
- `eval_F_v2_common`: **831 dòng**, cùng `eval_id` cho 10 cell (1688: 231; Tiki: 600). Đã re-score toàn bộ; xem `MATRIX_5X2_EVAL_F_V2.md`.
- Macro-average theo stratum và bucket độ dài đã báo cáo; kết luận stratum description không còn bị phóng đại bởi 12 dòng Qwen không chấm.

## 3. Thiết kế A/B chưa tách được quota khỏi field mix

A = `{title, attribute}` + natural; B = `{title, description}` + balanced → đổi **2 biến cùng lúc**. Vì vậy không thể claim "quota/balanced giúp description" tách khỏi "B có description còn A thì không".

**Control còn thiếu:** A-balanced, B-natural cho từng model. Hiện OpenNMT-B cũ chính là B-natural (nhưng bị dán nhãn balanced); đang retrain balanced. Kế hoạch 2×2:
- Qwen3-4B: A-natural (đã có), A-balanced, B-natural, B-balanced (đã có).
- Qwen3-8B, EcomGPT, Moses, OpenNMT: bổ sung 2 control/model.

## 4. Seed

Toàn bộ matrix hiện **1 seed training (42)**. Paired bootstrap 2.000 lần chỉ resample sản phẩm eval, không bao phủ variance khởi tạo/seed. Vì vậy CI của các so sánh model **hẹp hơn thực tế**; không được dùng để tuyên bố "Qwen hơn EcomGPT 0,0346" ở mức 1 seed.

**Kế hoạch:** seed 123 cho Qwen3-4B A/B trước (2 run), sau đó Qwen3-8B/EcomGPT nếu quota cho phép; sampling seed 123 để kiểm tra composition.

## 5. Trả lời trực tiếp

> "Quota Sampling đã thực sự khắc phục lỗi lấy mẫu thiên vị độ dài chưa?"

**Đã sửa một tầng:** xác suất chọn dòng không còn tỉ lệ với độ dài trong nhóm (có manifest). **Chưa sửa end-to-end:** exposure lệch 18×, title lặp 1,4–2,3 lần, 87% description không học, target dài bị cắt, eval trước đây khác nhau theo cell.

**Đã bổ sung:** per-group exposure, tokenizer thật, eval cố định 831 dòng, macro-average + bucket độ dài, sensitivity v1/v2.

**Chưa xong:** 2×2 đầy đủ + ≥3 seed — xem plan P1/P2 trong `REVIEW_RESPONSE.md`.
