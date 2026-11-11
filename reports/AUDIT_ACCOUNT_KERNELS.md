# Audit kernels — account `nguynthanhquang`

**Ngày:** 2026-11-11  
**Dataset:** `nguynthanhquang/tiki-matrix-data`  
**Quota GPU tại thời điểm audit:** ~11,5h / 30h

---

## 1. Dataset matrix (đã kiểm tra)

| file | rows | token | split |
|---|---:|---:|---|
| `units_A.parquet` | 111.129 | 4,518M | train 103.626 / dev 7.503 |
| `units_B.parquet` | 54.006 | 15,825M | train 49.774 / dev 4.232 |
| `eval_A.parquet` | 1.500 | 0,171M | eval_F, dùng chung |
| `eval_B.parquet` | 1.500 | 0,171M | giống hệt eval_A |

- `eval_A` và `eval_B` **trùng nhau** (cùng 1.500 eval_id) → chính là `eval_F`.
- Leakage check: `product_id(eval_F) ∩ product_id(units_A) = 0`,
  `∩ units_B = 0` → **đạt**.
- 4 cell đã chạy dùng đúng hash `units_A/B` của dataset này.
- Epoch ở budget 4M: **A ≈ 0,885**; **B ≈ 0,253** → B chỉ học ~1/4 epoch.

---

## 2. Toàn bộ kernel trên account

| kernel | trạng thái | việc đã làm | output |
|---|---|---|---|
| `tiki-matrix-m06b-a` | COMPLETE | cell 1: 0.6B, A natural, 4,000M, 77,0 phút | hyp 1.500, adapter |
| `tiki-matrix-m06b-b` | COMPLETE | cell 2: 0.6B, B balanced, 4,002M, 67,5 phút | hyp 1.500, adapter |
| `tiki-matrix-m4b-a` | COMPLETE | cell 6: 4B, A natural, 4,000M, 151,1 phút | hyp 1.500, adapter |
| `tiki-matrix-m8b-a` | COMPLETE | cell 8: 8B, A natural, 4,000M, 216,3 phút | hyp 1.500, adapter |
| `tiki-matrix-m06b-b-natural` | RUNNING | cell 3: 0.6B, B natural (control H2) | chưa có |
| `tiki-matrix-m8b-b` | chờ slot | cell 9: 8B, B balanced | chưa chạy |
| `tiki-matrix-nmt-a` | COMPLETE | baseline NMT from-scratch, A | `hyp_nmt_a_seed42` |
| `tiki-matrix-moses-a` | COMPLETE | baseline Moses SMT, A | `hyp_moses_a_seed42` |
| `tiki-matrix-moses-b` | COMPLETE | baseline Moses SMT, B | `hyp_moses_b_seed42` |
| `tiki-matrix-ecomgpt-a` | RUNNING | EcomGPT-7B, A | chưa có output |
| `tiki-probe-py` | COMPLETE | kiểm tra `pyonmttok` | log |
| `tiki-onmt-smoke` | ERROR | smoke test OpenNMT | `ModuleNotFoundError: pyonmttok` |
| `tiki-weight-final-s123` | COMPLETE | final recipe 4M/500 seed 123 (ngoài matrix 10 cell) | hyp 1.500 |
| `notebooke9c31e34fe` | 404 | notebook không còn tồn tại | — |

---

## 3. Độ phủ 10 cell

| # | cell | trạng thái |
|---:|---|---|
| 1 | 0.6B — A natural | **DONE** |
| 2 | 0.6B — B balanced | **DONE** |
| 3 | 0.6B — B natural (control) | **RUNNING** |
| 4 | 1.7B — A natural | thiếu |
| 5 | 1.7B — B balanced | thiếu |
| 6 | 4B — A natural | **DONE** |
| 7 | 4B — B balanced | thiếu |
| 8 | 8B — A natural | **DONE** |
| 9 | 8B — B balanced | chờ slot |
| 10 | 8B — B natural (control) | thiếu |

**Đã xong 4/10; đang chạy 1; chờ 1; còn thiếu 4.**

---

## 4. Phát hiện quan trọng

1. **4 cell đã chạy không lưu `eff_tokens` sample** — chỉ có `actual_share` và tổng token trong `run_*.json`.
   Dataset trainer đã được cập nhật để các cell sau ghi `sample_<config>_seed42.parquet`
   (`pair_id`, `group`, `eff_tokens`) + `epochs_seen`.
   Nếu cần kiểm tra chéo tuyệt đối cho 4 cell cũ, phải chạy lại (≈8,5h GPU).
2. **Không có assertion leakage trong notebook cũ**, nhưng dataset đã sạch (kiểm tra ở mục 1).
   Kernel mới đã thêm cell assert.
3. **15 hypothesis rỗng mỗi cell** — cùng các prompt dài >952 token, giống hệt nhau giữa các cell.
4. **EcomGPT-7B đang chạy** (từ 04:29) và chiếm 1 GPU session; đây là job ngoài 10 cell.
   Nó đang chặn kernel 8B B và tiêu quota.
5. **Quota còn ~11,5h**; 4 cell còn lại ước tính ~10h, cộng 8B B (~3,6h) → **không đủ**
   nếu EcomGPT tiếp tục chạy.

---

## 5. Đề xuất

- Huỷ `tiki-matrix-ecomgpt-a` nếu không cần cho báo cáo → giải phóng slot + quota.
- Chạy tiếp: 8B B (đang chờ), 1.7B A, 1.7B B, 4B B, 8B B-natural.
- Nếu vẫn thiếu quota: dùng account thứ hai cho 2 cell cuối.
- Với 4 cell cũ: chấp nhận `actual_share` thay cho sample manifest, hoặc chạy lại nếu yêu cầu kiểm tra chéo nghiêm ngặt.
