#!/bin/bash
# Chờ 2 kernel BT-fix xong rồi tải output.
set -u
ROOT=/content/eda_ecom
LOG=$ROOT/.artifacts/bt_fix_watch.log
mkdir -p "$ROOT/.artifacts"
echo "START $(date -u +%FT%TZ)" >> "$LOG"
while true; do
  a=$(kaggle kernels status anhtamxinhdep/tiki-bt-fix-small 2>/dev/null | tail -1)
  b=$(kaggle kernels status anhtamxinhdep/tiki-bt-fix-big 2>/dev/null | tail -1)
  echo "$(date -u +%H:%M:%S) small[$a] big[$b]" >> "$LOG"
  case "$a$b" in
    *RUNNING*|*QUEUED*) sleep 300 ;;
    *) break ;;
  esac
done
kaggle kernels output anhtamxinhdep/tiki-bt-fix-small -p "$ROOT/.artifacts/bt_fix_out/small" >> "$LOG" 2>&1 || true
kaggle kernels output anhtamxinhdep/tiki-bt-fix-big -p "$ROOT/.artifacts/bt_fix_out/big" >> "$LOG" 2>&1 || true
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
