# code/ — script cho dataset

Các script dùng trực tiếp với cấu trúc `data/` trong repo này.

## Pipeline dữ liệu

| script | việc |
|---|---|
| `split_tiki_description.py` | tách Tiki attribute vs description |
| `process_bt_desc.py` | QC/dedup/chunk BT description |
| `build_weight_splits_v2.py` | tạo split A/B trước EDA + eval_A/eval_B |
| `dedup_weight_A.py` | dedup exact pair cho A |
| `process_weight_ab.py` | processing A/B → units |
| `build_weight_eval_sets.py` | eval set + train view cũ |
| `eda_weight_splits.py`, `eda_weight_A_dedup.py`, `eda_processed_ab.py` | EDA + figures |

## Weight survey (quota)

| script | việc |
|---|---|
| `quota_sampling.py` | quota sampler: `T_g = s_g × budget` |
| `train_weight_quota_gpu.py` | train 5 config, 4M token, 500 step |
| `train_weight_gpu.py` | bản weight cũ (sampler theo `t_i`) |
| `train_weight_t4.py` | bản GPU đơn giản |
| `tpu_weight_train.py` | bản TPU 8 chip |

## Chấm điểm

| script | việc |
|---|---|
| `score_comet_weight.py` | COMET-da 1688, COMET-QE Tiki |
| `build_judge_input_multi.py` | judge input mù (id `j000001…`) |
| `gemma_judge_credit.py` | credit AAA..C (1–9), xoay vòng API key |
| `gemma_judge.py` | judge 3 trục (1–5) |
| `aggregate_weight_ab.py` | tổng hợp A/B + paired bootstrap |
| `aggregate_weight_eval.py` | tổng hợp weight cũ |
| `summarize_credit.py` | tổng hợp credit từ batch (kể cả partial) |

## Thứ tự chạy survey A/B

```bash
# 1. processing
python code/process_weight_ab.py

# 2. train + generate (GPU)
python code/train_weight_quota_gpu.py \
  --dataset A --data-root data --out /runs/ab_A \
  --model /path/Qwen3-4B --seed 42 --tokens 4000000 --steps 500

# 3. COMET
python code/score_comet_weight.py --hyp-dir /runs/ab_A/weight_A \
  --out-dir /runs/scores_A --gpus 1

# 4. tổng hợp
python code/aggregate_weight_ab.py --suite A --scores-dir /runs/scores_A \
  --out reports/WEIGHT_SURVEY_A.md --summary /runs/summary_A.json
```

Xem `WEIGHT_EVAL_PIPELINE.md` cho pipeline weight cũ và
`reports/WEIGHT_SURVEY_TECHNIQUE_A_B.md` cho thiết kế quota.
