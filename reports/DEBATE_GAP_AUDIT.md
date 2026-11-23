# Đối chiếu 3 phản biện — việc gì còn chưa xử lý

**Ngày:** 2026-11-19  
**Nguồn:** `REVIEW_1_DATA_LEAKAGE.md`, `REVIEW_2_OPENNMT_FAIRNESS.md`, `REVIEW_3_QUOTA_LENGTH_BIAS.md`  
**Nguyên tắc:** chỉ đánh dấu DONE khi có artefact tái lập được; mọi mục thiếu ghi rõ `PARTIAL`/`MISSING`.

---

## 0. Tóm tắt

| Phản biện | Điều kiện rút phản biện | Trạng thái hiện tại |
|---|---|---|
| 1 — Leakage | eval disjoint nội dung 2 chiều + bảng điểm 10 cell tái lập | **ĐẠT**, còn 2 điểm nhỏ (preflight assertion, hash trong run JSON) |
| 2 — OpenNMT fairness | NMT pretrained cùng protocol + nhãn B đúng + bảng compute + ≥2–3 seed | **3/4 ĐẠT**; thiếu seed thứ 2–3 cho NLLB/OpenNMT; recipe chuẩn chưa làm |
| 3 — Quota/length | per-group exposure + eval đồng nhất + 2×2 (hoặc B-natural) + ≥3 seed | **2.5/4 ĐẠT**; 2×2 mới có Qwen3-4B, chưa có seed thứ 3 |

---

## 1. Phản biện 1 — Data leakage

| # | Yêu cầu | Trạng thái | Bằng chứng / thiếu gì |
|---:|---|---|---|
| 1 | Chuẩn hoá hash nội dung + near-dup (MinHash) + nhóm trùng (`nhom_trung`, title cluster) | **DONE** | `tools/verify_review.py`, `review_audit.json` |
| 2 | `eval_F_v2` disjoint **5 khoá** + assertion **fail cứng trong preflight cell** | **PARTIAL** | Đã có `eval_F_v2` 858 dòng + assert trong `verify_review.py`. Nhưng preflight của **kernel** vẫn chỉ `assert not overlap` theo `product_id` → chưa đúng chữ "trong preflight cell" |
| 3 | Bảng overlap theo stratum + re-score 10 cell + sensitivity | **DONE** | `reports/LEAKAGE_AUDIT.md` §2, §4; `reports/MATRIX_5X2_EVAL_F_V2.md`; `EVAL_F_V2_SENSITIVITY.csv` |
| 4 | Báo cáo leakage gốc + audit script + `dataset_hash`/`content_hash` trong **mọi** `run_*.json` | **PARTIAL** | Có report + script + `RUN_HASH_MANIFEST.json`; nhưng chỉ **3** `run_*.json` trong repo có hash. 5 run mới (abal, bnat, a-s123, b-s123, NLLB) có `run_*.json` tải từ Kaggle nhưng **chưa backfill hash** |
| 5 | Nếu giữ `eval_F` cũ: bảng leaked rows + điểm loại trừ | **DONE** | `LEAKAGE_AUDIT.md` §4.1; `EVAL_F_V2_SENSITIVITY.csv` |

**Kết luận R1:** phần khoa học đã đạt; 2 việc còn lại là **hạ tầng** (preflight + hash), không cần GPU.

---

## 2. Phản biện 2 — OpenNMT fairness

| # | Yêu cầu | Trạng thái | Bằng chứng / thiếu gì |
|---:|---|---|---|
| 1 | Sửa nhãn/recipe: retrain B-balanced **hoặc** đổi tên B-natural | **DONE (relabel)** | Mọi bảng đã ghi `OpenNMT B-natural`; dòng B−A của OpenNMT bị loại khỏi bảng hiệu ứng |
| 2 | Baseline NMT pretrained (mBART/NLLB) cùng quota 4M, cùng eval protocol | **DONE** | NLLB-600M A: 0,6344 / −0,5379; B: 0,5517 / −0,6753 (eval_F_v2_common) |
| 3 | Hoặc recipe chuẩn (6 layer/512, BPE 16k, 50–100M token, early stop, averaging, beam+length penalty) + ≥3 seed + sanity IWSLT/WMT | **MISSING** | Chưa chạy. Đây là nhánh thay thế, không bắt buộc nếu đã có #2 |
| 4 | Bảng compute công bằng + bỏ câu "compute-matched" | **DONE** | `reports/OPENNMT_FAIRNESS.md` §2: 37M params, 245,8M/188,3M token-pass, 61,4×/47,1× |
| 5 | Xử lý độ dài giống nhau + công bố stratum description dài | **PARTIAL** | Đã báo cáo bucket short/medium/long và truncation 49 dòng >512; nhưng OpenNMT vẫn cắt trong khi LLM không → chưa "giống nhau", chỉ "đã công bố" |

**Điều kiện rút:** cần thêm **≥2–3 seed** cho NLLB/OpenNMT (hiện 1 seed).

---

## 3. Phản biện 3 — Quota sampling & length bias

| # | Yêu cầu | Trạng thái | Bằng chứng / thiếu gì |
|---:|---|---|---|
| 1 | Per-group epoch, unique coverage, draws/row trong **mọi** `run_*.json` + bảng kết quả | **PARTIAL** | Đã có trong `review_audit.json` (`quota.A/B.groups`) và `QUOTA_LENGTH_AUDIT.md` §2.1. Nhưng `run_*.json` của cell vẫn chỉ có scalar `epochs_seen`/`actual_share`, **không có coverage/draws/row** |
| 2 | Token budget bằng tokenizer thật, bỏ trần 960 trong accounting, báo cáo % cắt | **DONE** | Qwen tokenizer thật: A 6,560M (0,8855 epoch), B 5,926M (0,2477); truncation 4,94%/30,4% (uncapped) |
| 3 | Eval protocol cố định + sensitivity + metric theo bucket + macro-average | **DONE** | `eval_F_v2_common` 831 dòng; `MATRIX_5X2_EVAL_F_V2.md` §7 (macro), §8 (bucket) |
| 4 | 2×2 (field × quota) cho mọi model, hoặc B-natural + A-balanced cho cả 5 model | **PARTIAL** | **Chỉ Qwen3-4B đủ 2×2** (A-natural 0,8584; A-balanced 0,8477; B-balanced 0,8471; B-natural 0,8436). Qwen3-8B, EcomGPT-7B, Moses, OpenNMT **chưa có** A-balanced/B-natural |
| 5 | ≥3 seed + CI theo seed + sampling seed riêng | **PARTIAL** | Có 2 seed cho Qwen3-4B (A-natural: 0,8584/0,8511; B-balanced: 0,8471/0,8479). Seed variance 0,005–0,007. **Chưa có seed thứ 3**; NLLB/OpenNMT/EcomGPT/Moses vẫn 1 seed |

**Điều kiện rút:** thiếu **seed thứ 3** và **2×2 cho 4 model còn lại**.

---

## 4. Việc còn lại — theo thứ tự ưu tiên

### P0 — không cần GPU (làm được ngay)

| # | Việc | Đóng phản biện |
|---:|---|---|
| 1 | Thêm preflight cell assert **5 khoá** (product/source/pair/near-dup/nhom_trung) vào template kernel, fail cứng | R1.2 |
| 2 | Backfill `dataset_hash`, `sample_hash`, `eval_hash`, `content_hash` + **per-group exposure** vào 5 `run_*.json` mới; cập nhật `RUN_HASH_MANIFEST.json` | R1.4, R3.1 |
| 3 | `eval_F_v3` thiết kế **trước** inference, split theo `nhom_trung`/near-dup group | khuyến nghị nội bộ (R1) |

### P1 — cần GPU (A100 Colab hiện có, không tốn quota Kaggle)

| # | Việc | Ước lượng A100 | Đóng phản biện |
|---:|---|---:|---|
| 4 | Qwen3-8B **A-balanced** + **B-natural** | ~6–8h/run → 12–16h | R3.4 |
| 5 | EcomGPT-7B **A-balanced** + **B-natural** | ~6–8h/run → 12–16h | R3.4 |
| 6 | Qwen3-4B **seed thứ 3** (A-natural) + NLLB **seed thứ 3** | ~4h + ~1,5h | R2 điều kiện, R3.5 |
| 7 | Moses A-balanced + B-natural (CPU, không cần GPU) | ~1h | R3.4 |
| 8 | OpenNMT recipe chuẩn 6/512 + BPE 16k (Colab có internet để cài OpenNMT-py) | ~2–3h + train | R2.3 |

### P2 — chấp nhận giới hạn

- 3 seed cho **cả 5 model**: ~40h GPU, không khả thi trong phạm vi hiện tại → chỉ làm seed thứ 3 cho Qwen3-4B + NLLB, ghi rõ phần còn lại là 1 seed.
- Retrain OpenNMT B-balanced trên Kaggle: bị chặn bởi môi trường offline; có thể làm trên Colab nếu cần.

---

## 5. Trạng thái artefact hỗ trợ

| Artefact | Trạng thái |
|---|---|
| `review_audit.json` | có, tái lập 83 giây |
| `eval_F_v2` / `eval_F_v2_common` | có (858 / 831) |
| `LEAKAGE_AUDIT.md`, `MATRIX_5X2_EVAL_F_V2.md`, `OPENNMT_FAIRNESS.md`, `QUOTA_LENGTH_AUDIT.md` | có |
| `CONTROLS_2X2.md` | có, đã gồm 2×2 Qwen3-4B + seed 42/123 + NLLB |
| `RUN_HASH_MANIFEST.json` | có cho 10 cell gốc; **chưa gồm 5 cell mới** |
| `EVAL_F_V2_SENSITIVITY.csv` | có |
