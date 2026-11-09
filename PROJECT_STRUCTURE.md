# Cấu trúc repo (2026-11-07)

## Canonical / đang dùng

```text
README.md
PROJECT_STRUCTURE.md
requirements.txt
run_pipeline.py

cado_core/              CADO + bitext core
cado_layer/             wrapper từng tầng CADO
bitext_mining/          preprocess/filter bitext
tiki_mining/            EDA/unit Tiki

data/
  raw/                  dữ liệu gốc
  cleaned/
    1688/               1688 đã clean/align
    pairs_dedup/        pair table 1688 + 4 Tiki model
    weight_eval/        eval set weight + train view
    weight_splits_v2/   A/B split trước EDA + eval_A/eval_B + meta
    processed_A/        units.parquet cho survey A (quota)
    processed_B/        units.parquet cho survey B (quota)
    tiki_desc_split/    attribute/description split mới
    bt_tiki/            BT title/attribute v5 (4 model)
    bt_tiki_desc/       BT description đang hoàn thiện
    bt_tiki_clean*/     bản clean cũ (giữ audit)
    judge/              Gemma judge input/output

tools/
  split_tiki_description.py
  process_bt_desc.py
  build_weight_eval_sets.py
  dedup_weight_A.py / eda_weight_A_dedup.py
  process_weight_ab.py / eda_processed_ab.py
  quota_sampling.py
  train_weight_gpu.py / tpu_weight_train.py
  train_weight_quota_gpu.py
  score_comet_weight.py
  build_judge_input_multi.py / gemma_judge*.py
  aggregate_weight_eval.py / aggregate_weight_ab.py / report_credit.py
  watch_ab_survey.sh

kaggle/                 kernel đang dùng
  rtx_pro_6000*         weight experiment
  weight_ab_a/          survey A (quota, 5 config)
  weight_ab_b/          survey B (quota, 5 config)
  comet_weight_offline  COMET offline
  rtx_bt_desc*          BT description
  configs/, kernel/, notify/

reports/
  EDA_A_B.md / EDA_A_DEDUP.md / EDA_PROCESSED_A_B.md
  WEIGHT_SURVEY_TECHNIQUE_A_B.md
  WEIGHT_EVAL_FINAL.md
  WEIGHT_EVAL.md
  WEIGHT_EVAL_PIPELINE.md
  TIKI_DESC_*.md
  NEXT_STEPS.md (khi có)
```

## Archive

```text
archive/
  kaggle_legacy/        kernel/experiment cũ (1688, phase1, BT cũ, TPU cũ)
  reports_v3/           report cũ, giữ symlink reports_v3
  reports_v5/           report cũ, giữ symlink reports_v5
  root_build_weight_eval_sets.py
```

Không xoá archive; chỉ move để root gọn.

## Dataset trên HF

```text
https://huggingface.co/datasets/ntquang0410/tiki-4model-eval
```

Layout trên HF (refactor 2026-11-08):

```text
README.md / INDEX.md
code/                 script + requirements
docs/                 migration, TIKI_DESC, WEIGHT_SPLITS, RUN_ON_RTX6000
data/1688/            train/dev/test + tokenized
data/pairs/           pair table dedup
data/bt/              title_attribute/ + description/{processed,raw}
data/weight/          eval/ + splits_v2/ + processed_A|B/ + survey_ab/
data/tiki_desc_split/
reports/              báo cáo + figures/{eda_ab,eda_a_dedup,processed_ab,desc_bt}
```

Bản cập nhật 2026-11-08 có thêm:

```text
data/bt_qwen3-4b/            toàn bộ output BT Qwen3-4B từ Kaggle
  title_attribute/           predictions v5 (29.446 dòng, good 99,68%)
  description/               predictions full 13.200 dòng (good 88,73%)
data/bt_tiki_desc/qwen3-4b/  description BT đã process (11.404 unit, 6,363M token)
data/weight_survey_ab/       hyps + COMET/credit scores của survey A/B
reports/TIKI_DESC_4B_PROCESSING.md
reports/WEIGHT_SURVEY_A_B.md
```

Bản cập nhật 2026-11-07 có thêm:

```text
data/tiki_desc_split/
data/weight_eval/eval_tiki_ext.parquet
data/weight_eval/eval_tiki_desc.parquet
bt_desc_partial/
TIKI_DESC.md
manifest_tiki_desc.json
```

BT 4B/8B final sẽ được version sau khi chạy xong.
