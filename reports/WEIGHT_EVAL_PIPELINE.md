# Thực nghiệm weight desc / title / attribute

Mục tiêu: trả lời **gán weight cho description/title/attribute có làm tăng chất lượng học
của model không**, bằng COMET + Gemma, với thiết kế paired công bằng.

## Thiết kế

```text
model        : Qwen3-4B + LoRA (r=16, all-linear)
train        : 1688 + Tiki, đã dedup/chunk, bỏ dòng > 960 token ước lượng
budget       : 5.000.000 token / run
optimizer    : 625 step / run, mỗi step ~8.000 token chia 8 chip
seed         : 42 và 123
config       : natural | balanced | no_desc | tiki_heavy | title_heavy
eval 1688    : 300 title + 300 attribute + 150 description (COMET-da)
eval Tiki    : 300 title + 300 attribute (COMET-QE)
judge        : Gemma blind, id j000001..., map riêng
bootstrap    : 2.000 lần paired theo eval_id, gộp 2 seed
```

Điểm quan trọng: **cùng số optimizer step và cùng token/step**. Nếu chỉ cố định
token budget mà để số step thay đổi, cấu hình nhiều title/attribute ngắn sẽ có
nhiều update hơn — đó là confound, không còn là so sánh weight.

## Cấu hình weight

```text
natural      : phân bố token tự nhiên (desc ~64%)
balanced     : title 15%, attr 35%, desc 25%, tiki 25%
no_desc      : title 25%, attr 45%, desc 0%,  tiki 30%
tiki_heavy   : title 10%, attr 25%, desc 20%, tiki 45%
title_heavy  : title 30%, attr 30%, desc 15%, tiki 25%
```

## Chạy trên Kaggle TPU

```bash
# seed 42
kaggle kernels push -p kaggle/tpu_weight_seed42_v3
kaggle kernels status ntquang0410/tiki-weight-tpu-v3-seed42
kaggle kernels output ntquang0410/tiki-weight-tpu-v3-seed42 -p /tmp/opencode/weight_runs

# seed 123 (sau khi seed 42 xong)
kaggle kernels push -p kaggle/tpu_weight_seed123
```

Notebook tự chạy smoke 2 step trên 8 chip trước; nếu smoke qua mới chạy full.
Output:

```text
hyp_<config>_seed<seed>_1688.parquet
hyp_<config>_seed<seed>_tiki.parquet
adapter_<config>_seed<seed>/
run_<config>_seed<seed>.json
```

## Chấm COMET (GPU/CPU nhẹ)

```bash
export HF_TOKEN=...        # cần cho một số model gated
python tools/score_comet_weight.py \
  --hyp-dir /tmp/opencode/weight_runs \
  --out-dir /tmp/opencode/weight_runs \
  --batch-size 8
```

- 1688: `Unbabel/wmt22-comet-da`
- Tiki: `Unbabel/wmt20-comet-qe-da` (bản `wmt22-cometkiwi-da` bị gated)

Nếu máy ít RAM, chạy từng file bằng `--filter <substring>` trong process riêng.

## Chấm Gemma (blind)

```bash
python tools/build_judge_input_multi.py \
  --hyp-dir /tmp/opencode/weight_runs \
  --out-dir data/cleaned/judge \
  --n 50

python tools/gemma_judge.py \
  --input data/cleaned/judge/judge_input.jsonl \
  --map data/cleaned/judge/judge_map.jsonl \
  --keys keys.txt \
  --out-dir data/cleaned/judge \
  --model gemma-4-31b-it \
  --batch-size 10
```

`judge_input` chỉ chứa id mù, source, hypothesis, reference; không có config/seed.

## Tổng hợp

```bash
python tools/aggregate_weight_eval.py \
  --scores-dir /tmp/opencode/weight_runs \
  --eval-dir data/cleaned/weight_eval \
  --gemma data/cleaned/judge/gemma_scores.json \
  --out reports/WEIGHT_EVAL.md \
  --summary data/cleaned/weight_eval/summary.json
```

## Quy tắc kết luận

```text
primary   : COMET-QE Tiki macro (title + attribute)
guardrail : COMET-da 1688 macro không thấp hơn natural quá 0,01
bootstrap : CI 95% paired theo eval_id
seed      : cả 2 seed phải cùng dấu

nếu best - natural < 0,005 hoặc CI chứa 0 hoặc 2 seed lệch dấu
    -> natural (không cần weight)
ngược lại
    -> chọn config tốt nhất
```

Kết luận `natural` = phân bố token tự nhiên đã tốt nhất; không cần can thiệp.
Kết luận `title_heavy`/`tiki_heavy`/... = weight có tác dụng, kèm stratum tăng.
