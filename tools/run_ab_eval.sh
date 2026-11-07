#!/bin/bash
# Chấm COMET + Gemma + aggregate cho một suite A/B đã tải hypothesis.
# Dùng: bash tools/run_ab_eval.sh A
set -euo pipefail
export USE_TF=0
export USE_FLAX=0
SUITE=${1:?dùng A hoặc B}
LOWER=$(echo "$SUITE" | tr 'A-Z' 'a-z')
ROOT=/content/eda_ecom
HYP=$ROOT/.artifacts/weight_ab_$LOWER
SCORES=$ROOT/.artifacts/scores_$SUITE
JUDGE=$ROOT/.artifacts/judge_$SUITE
mkdir -p "$SCORES" "$JUDGE"
cd "$ROOT"

echo "== COMET $SUITE =="
python tools/score_comet_weight.py \
  --hyp-dir "$HYP" --out-dir "$SCORES" --gpus 0 --batch-size 8

echo "== Judge input $SUITE =="
python tools/build_judge_input_multi.py \
  --hyp-dir "$HYP" --out-dir "$JUDGE" --n 50 --seed 42

echo "== Gemma credit $SUITE =="
python tools/gemma_judge_credit.py \
  --input "$JUDGE/judge_input.jsonl" \
  --map "$JUDGE/judge_map.jsonl" \
  --keys keys.txt --out-dir "$JUDGE" --batch-size 10

echo "== Aggregate $SUITE =="
python tools/aggregate_weight_ab.py \
  --suite "$SUITE" --scores-dir "$SCORES" \
  --gemma "$JUDGE/gemma_scores.json" \
  --comet-json "$SCORES/comet_scores.json" \
  --out "$ROOT/reports/WEIGHT_SURVEY_$SUITE.md" \
  --summary "$ROOT/.artifacts/summary_$SUITE.json"
