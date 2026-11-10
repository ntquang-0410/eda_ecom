#!/bin/bash
# Chờ C1, tải; chờ 8B xong để lấy slot; chạy C2 và tải.
set -u
ROOT=/content/eda_ecom
LOG=$ROOT/.artifacts/weight_c_watch.log
mkdir -p "$ROOT/.artifacts/weight_c_out"
wait_done() {
  local ref=$1
  while true; do
    s=$(kaggle kernels status "$ref" 2>/dev/null | tail -1)
    echo "$(date -u +%H:%M:%S) $ref [$s]" >> "$LOG"
    case "$s" in *RUNNING*|*QUEUED*) sleep 180;; *) break;; esac
  done
}
echo "START $(date -u +%FT%TZ)" >> "$LOG"
wait_done anhtamxinhdep/tiki-weight-c1
kaggle kernels output anhtamxinhdep/tiki-weight-c1 -p "$ROOT/.artifacts/weight_c_out/c1" >> "$LOG" 2>&1 || true
wait_done anhtamxinhdep/tiki-bt-fix-8b-full
cd "$ROOT"
kaggle kernels push -p kaggle/tiki-weight-c2 >> "$LOG" 2>&1 || true
wait_done anhtamxinhdep/tiki-weight-c2
kaggle kernels output anhtamxinhdep/tiki-weight-c2 -p "$ROOT/.artifacts/weight_c_out/c2" >> "$LOG" 2>&1 || true
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
