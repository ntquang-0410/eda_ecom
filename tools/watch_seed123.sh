#!/bin/bash
# Chờ seed123 A/B, tải output; sau đó chạy kernel 8B full và tải output.
set -u
ROOT=/content/eda_ecom
LOG=$ROOT/.artifacts/seed123_watch.log
mkdir -p "$ROOT/.artifacts/seed123_out"

wait_done() {
  local ref=$1
  while true; do
    s=$(kaggle kernels status "$ref" 2>/dev/null | tail -1)
    echo "$(date -u +%H:%M:%S) $ref [$s]" >> "$LOG"
    case "$s" in
      *RUNNING*|*QUEUED*) sleep 300 ;;
      *) break ;;
    esac
  done
}

echo "START $(date -u +%FT%TZ)" >> "$LOG"
wait_done anhtamxinhdep/tiki-weight-seed123-a
wait_done anhtamxinhdep/tiki-weight-seed123-b
kaggle kernels output anhtamxinhdep/tiki-weight-seed123-a -p "$ROOT/.artifacts/seed123_out/A" >> "$LOG" 2>&1 || true
kaggle kernels output anhtamxinhdep/tiki-weight-seed123-b -p "$ROOT/.artifacts/seed123_out/B" >> "$LOG" 2>&1 || true

# 8B full (chạy sau khi slot GPU trống)
cd "$ROOT"
kaggle kernels push -p kaggle/tiki-bt-fix-8b-full >> "$LOG" 2>&1 || true
wait_done anhtamxinhdep/tiki-bt-fix-8b-full
kaggle kernels output anhtamxinhdep/tiki-bt-fix-8b-full -p "$ROOT/.artifacts/seed123_out/bt8b" >> "$LOG" 2>&1 || true
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
