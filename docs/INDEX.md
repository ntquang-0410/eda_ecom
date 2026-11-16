# Docs index

Các báo cáo đang dùng:

- `reports/CLAIMS_REVISION.md` — các câu phải sửa trước khi bảo vệ
- `reports/REVIEW_RESPONSE.md` — phản hồi 3 phản biện + plan P0–P3
- `reports/LEAKAGE_AUDIT.md` — audit leakage nội dung + eval_F_v2
- `reports/MATRIX_5X2_EVAL_F_V2.md` — re-score 10 cell trên eval_F_v2_common
- `reports/OPENNMT_FAIRNESS.md` — compute/fairness OpenNMT + NLLB
- `reports/QUOTA_LENGTH_AUDIT.md` — exposure, tokenizer thật, eval protocol
- `reports/MATRIX_5X2_RESULTS.md` — **matrix 5 model × 2 recipe** (COMET + bootstrap)
- `reports/MODEL_CAUSE_ANALYSIS.md` — phân tích nguyên nhân theo từng model
- `reports/MODEL_DIAGNOSTICS.md` — chẩn đoán lỗi định lượng, ví dụ
- `reports/FINETUNE_CONFIGS_5MODELS.md` — cấu hình fine-tune/train 5 model
- `reports/WEIGHT_SURVEY_A_B.md` — **kết quả chính**: A giữ natural, B chọn balanced
- `reports/SEED123_SUMMARY.md` — xác nhận 2 seed 42/123
- `reports/CLUSTER_BOOTSTRAP.md` — CI theo product/title_heavy
- `reports/WEIGHT_SURVEY_A.md` / `WEIGHT_SURVEY_B.md` — bảng COMET chi tiết từng suite
- `reports/WEIGHT_SURVEY_TECHNIQUE_A_B.md` — kỹ thuật quota sampling cho A/B
- `reports/EDA_A_B.md` / `EDA_A_DEDUP.md` — EDA split A/B trước processing
- `reports/EDA_PROCESSED_A_B.md` — EDA sau processing (units dùng để train)
- `reports/TIKI_DESC_BT_PROCESSING.md` — QC/processing BT description 4 model

Layout dataset trên HF: xem `README.md` / `INDEX.md` của repo
https://huggingface.co/datasets/ntquang0410/tiki-4model-eval
- `reports/WEIGHT_EVAL_FINAL.md` — kết luận weight 4B, COMET + Gemma credit (sampler cũ, tham khảo)
- `reports/WEIGHT_EVAL.md` — bảng COMET chi tiết, 2 seed
- `reports/WEIGHT_EVAL_PIPELINE.md` — cách chạy lại weight
- `data/cleaned/tiki_desc_split/README.md` — split attribute/description Tiki
- `reports/CHAR_AUDIT.md` — audit ký tự nhiễu
- `reports/QUANTITY_NORMALIZATION.md` — chuẩn hoá đại lượng
- `reports/PHASE0_DEDUP.md` — dedup/chunk
- `reports/1688_eda.md` — EDA 1688

Archive (không còn active):

- `archive/reports_v3/`
- `archive/reports_v5/`
- `archive/kaggle_legacy/`

Dataset HF:

- https://huggingface.co/datasets/ntquang0410/tiki-4model-eval
