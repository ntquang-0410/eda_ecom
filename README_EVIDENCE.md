# Evidence bundle — phản hồi 3 phản biện

**Cập nhật:** 2026-11-16  
**Repo chính:** `eda_ecom/`

## 1. Data leakage

- `eda_ecom/tools/verify_review.py` — audit tái lập (hash, near-dup, nhom_trung, quota, OpenNMT).
- `review_audit.json` — kết quả audit (workspace root + `eda_ecom/`).
- `eda_ecom/data/cleaned/eval_F_v2.parquet` — 858 dòng sạch nội dung.
- `eda_ecom/data/cleaned/eval_F_v2_common.parquet` — 831 dòng dùng chung 10 cell.
- `eda_ecom/reports/LEAKAGE_AUDIT.md` — báo cáo leakage gốc + bảng overlap + sensitivity.
- `eda_ecom/reports/MATRIX_5X2_EVAL_F_V2.md` — re-score 10 cell trên v2.
- `eda_ecom/reports/EVAL_F_V2_SENSITIVITY.csv` — so sánh v1/v2 từng cell.

## 2. OpenNMT fairness

- `eda_ecom/reports/OPENNMT_FAIRNESS.md` — 5 trục, bảng compute, lỗi nhãn B-natural, phạm vi kết luận.
- `eda_ecom/reports/MATRIX_5X2_EVAL_F_V2.md` — OpenNMT-B đã relabel **B-natural**.
- `kaggle/matrix_nllb/` — NLLB-200-distilled-600M control (L4x4, offline).
- `kaggle/matrix_onmt_balanced/` — builder OpenNMT-B balanced (không chạy được vì offline env).

## 3. Quota / length bias

- `eda_ecom/reports/QUOTA_LENGTH_AUDIT.md` — per-group exposure, tokenizer thật, truncation, eval protocol.
- `review_audit.json` — khóa `quota`, `common_eval_ids`, `tiki_description_common_subset`.
- `eda_ecom/reports/RUN_HASH_MANIFEST.json` — dataset/sample/eval hash + exposure cho 10 cell.

## 4. Tổng hợp & kế hoạch

- `eda_ecom/reports/REVIEW_RESPONSE.md` — bảng từng yêu cầu + P0/P1/P2/P3.
- `eda_ecom/reports/CLAIMS_REVISION.md` — các câu phải sửa trước khi bảo vệ.
- `eda_ecom/reports/MODEL_CAUSE_ANALYSIS.md` — phân tích nguyên nhân (đã gắn caveat).

## 5. Chạy lại audit

```bash
cd /content
python3 verify_review.py
```

Kết quả ghi `review_audit.json` ở workspace root và `eda_ecom/review_audit.json`.

## 6. GPU

- Kernel mới: `machine_shape: NvidiaL4x4`, `competition_sources: ["gemma-4-developer-agent"]`, dataset `metric/gemma-4-developer-agent-wheelhouse`, internet off.
