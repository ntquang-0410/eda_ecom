#!/usr/bin/env bash
# Chạy watcher mốc, rồi đẩy kết quả vào Telegram.
#   watch_and_notify.sh <kernel> <label> <state_file> [max_minutes]
set -u

KERNEL="$1"
LABEL="$2"
STATE="$3"
MINUTES="${4:-30}"

OUT="$(MILESTONE_STATE="$STATE" python -u /tmp/opencode/watch_milestones.py \
        --kernel "$KERNEL" --max-minutes "$MINUTES" 2>&1 | tail -n 2)"

echo "$OUT"

STATUS="$(kaggle kernels status "$KERNEL" 2>&1 | tail -n 1 | sed 's/.*status "//; s/"$//')"
MSG="🔔 ${LABEL}
${OUT}
kernel: ${STATUS}
$(date -u +%H:%MZ)"

python /tmp/opencode/telegram.py "$MSG" >/dev/null 2>&1 || echo "telegram: gửi thất bại"
