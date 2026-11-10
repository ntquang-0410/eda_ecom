#!/bin/bash
# Theo dõi 2 kernel survey A/B, tải output khi xong.
set -u
ROOT=/content/eda_ecom
LOG=$ROOT/.artifacts/ab_survey_watch.log
mkdir -p "$ROOT/.artifacts"
echo "START $(date -u +%FT%TZ)" >> "$LOG"
while true; do
  a=$(kaggle kernels status nadsa091/tiki-weight-ab-a 2>/dev/null | tail -1)
  b=$(kaggle kernels status nadsa091/tiki-weight-ab-b 2>/dev/null | tail -1)
  echo "$(date -u +%H:%M:%S) A[$a] B[$b]" >> "$LOG"
  case "$a$b" in
    *RUNNING*|*QUEUED*) sleep 300 ;;
    *) break ;;
  esac
done
kaggle kernels output nadsa091/tiki-weight-ab-a -p "$ROOT/.artifacts/weight_ab_a" >> "$LOG" 2>&1 || true
kaggle kernels output nadsa091/tiki-weight-ab-b -p "$ROOT/.artifacts/weight_ab_b" >> "$LOG" 2>&1 || true
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
