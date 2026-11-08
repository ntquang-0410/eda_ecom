# Chạy trên RTX Pro 6000

## 1. Tải dataset

```bash
pip install -U huggingface_hub
huggingface-cli download ntquang0410/tiki-4model-eval \
  --repo-type dataset --local-dir tiki-4model-eval
cd tiki-4model-eval
pip install -r requirements.txt
```

Repo đang **public**; nếu cần private, đổi repo sang private và thêm account GPU vào
Settings → Collaborators.

## 2. Đánh giá 4 model (baseline natural)

```bash
for m in 0.6b 1.7b 4b 8b; do
  python code/train_weight_t4.py \
    --model Qwen/Qwen3-$m \
    --train-view data/weight_eval/train_view_$m.parquet \
    --eval-dir data/weight_eval \
    --configs natural --seeds 42 --tokens 5000000 \
    --out /runs/4models/$m --dtype bf16 \
    --batch-size 1 --accum 8
done
```

- 0.6B/1.7B có thể tăng `--batch-size 4 --accum 2`.
- 4B dùng `--batch-size 2 --accum 4`.
- 8B dùng `--batch-size 1 --accum 8` (hoặc gradient checkpointing mặc định).

## 3. Weight experiment 4B (đúng 625 step, công bằng)

```bash
python code/train_weight_gpu.py \
  --model Qwen/Qwen3-4B \
  --train-view data/weight_eval/train_view_4b.parquet \
  --eval-dir data/weight_eval \
  --configs natural balanced no_desc tiki_heavy title_heavy \
  --seeds 42 123 --tokens 5000000 --steps 625 \
  --out /runs/weight_4b --dtype bf16 --micro-batch 4
```

`train_weight_gpu.py` dùng cùng scheduler với bản TPU: 625 optimizer step,
~8.000 token/step, nên không bị confound số step giữa các cấu hình.

## 4. Chấm điểm

```bash
python code/score_comet_weight.py \
  --hyp-dir /runs/weight_4b --out-dir /runs/weight_4b --batch-size 8
```

- 1688: `Unbabel/wmt22-comet-da`
- Tiki: `Unbabel/wmt20-comet-qe-da` (bản `wmt22-cometkiwi-da` bị gated)

Nếu RAM thấp, chạy từng file bằng `--filter <substring>` trong process riêng.

## 5. Gemma judge

```bash
python code/build_judge_input_multi.py \
  --hyp-dir /runs/weight_4b --out-dir /runs/judge --n 50

python code/gemma_judge_credit.py \
  --input /runs/judge/judge_input.jsonl \
  --map /runs/judge/judge_map.jsonl \
  --keys keys.txt --out-dir /runs/judge \
  --model gemma-4-31b-it --batch-size 10
```

## 6. Tổng hợp

```bash
python code/aggregate_weight_eval.py \
  --scores-dir /runs/weight_4b --eval-dir data/weight_eval \
  --gemma /runs/judge/gemma_scores.json \
  --out reports/WEIGHT_EVAL.md
```
