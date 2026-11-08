# Code cho dataset `tiki-4model-eval`

Các script này dùng trực tiếp với cấu trúc `data/` trong repo.

## Train/eval 4 model

- `build_weight_eval_sets.py`: tạo eval set + train view từ `pairs_dedup/`.
- `train_weight_t4.py`: train LoRA trên GPU đơn (T4/RTX), sampler đơn giản.
- `train_weight_gpu.py`: train GPU với đúng 625 step / 8k token mỗi step (công bằng).
- `tpu_weight_train.py`: train LoRA trên TPU 8 chip, cùng token budget/step.
- `score_comet_weight.py`: COMET-da cho 1688, COMET-QE cho Tiki.
- `build_judge_input_multi.py`: tạo judge input mù, id `j000001...`.
- `gemma_judge.py`: chấm Gemma 3 trục (1-5), xoay vòng nhiều API key.
- `gemma_judge_credit.py`: chấm Gemma theo thang tín dụng AAA..C (1-9), blind.
- `aggregate_weight_eval.py`: paired bootstrap, bảng stratum, kết luận.

## Thứ tự chạy weight experiment

```bash
python code/train_weight_t4.py \
  --model /path/Qwen3-4B \
  --train-view data/weight_eval/train_view_4b.parquet \
  --eval-dir data/weight_eval \
  --configs natural balanced no_desc tiki_heavy title_heavy \
  --seeds 42 123 --tokens 5000000 --out /runs/weight

python code/score_comet_weight.py \
  --hyp-dir /runs/weight --out-dir /runs/weight --batch-size 8

python code/build_judge_input_multi.py \
  --hyp-dir /runs/weight --out-dir /runs/judge --n 50

python code/gemma_judge.py \
  --input /runs/judge/judge_input.jsonl \
  --map /runs/judge/judge_map.jsonl \
  --keys keys.txt --out-dir /runs/judge \
  --model gemma-4-31b-it --batch-size 10

python code/aggregate_weight_eval.py \
  --scores-dir /runs/weight --eval-dir data/weight_eval \
  --gemma /runs/judge/gemma_scores.json \
  --out reports/WEIGHT_EVAL.md
```

Xem `WEIGHT_EVAL_PIPELINE.md` để biết thiết kế và quy tắc kết luận.
