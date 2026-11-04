#!/usr/bin/env bash
# Một lệnh: tạo input từ parquet output của model rồi chấm bằng Gemma.
# Cách dùng:
#   cp keys.example.txt keys.txt   # dán key vào đây
#   bash tools/run_gemma_judge.sh data/cleaned/hyp_4b.parquet 200
set -euo pipefail
INPUT="${1:?parquet có id/source/hypothesis[/reference]}"
N="${2:-200}"
python tools/build_judge_input.py --input "$INPUT" --out-dir data/cleaned/judge --n "$N"
python tools/gemma_judge.py \
  --input data/cleaned/judge/judge_input.jsonl \
  --keys keys.txt \
  --out-dir data/cleaned/judge \
  --batch-size 25
