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
    tiki_desc_split/    attribute/description split mới
    bt_tiki/            BT title/attribute v5 (4 model)
    bt_tiki_desc/       BT description đang hoàn thiện
    bt_tiki_clean*/     bản clean cũ (giữ audit)
    judge/              Gemma judge input/output

tools/
  split_tiki_description.py
  process_bt_desc.py
  build_weight_eval_sets.py
  train_weight_gpu.py / tpu_weight_train.py
  score_comet_weight.py
  build_judge_input_multi.py / gemma_judge*.py
  aggregate_weight_eval.py / report_credit.py

kaggle/                 kernel đang dùng
  rtx_pro_6000*         weight experiment
  comet_weight_offline  COMET offline
  rtx_bt_desc*          BT description
  configs/, kernel/, notify/

reports/
  WEIGHT_EVAL_FINAL.md
  WEIGHT_EVAL.md
  WEIGHT_EVAL_PIPELINE.md
  TIKI_DESC_*.md (khi có)
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
