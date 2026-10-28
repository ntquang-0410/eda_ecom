#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watcher ngắn hạn: thoát ngay khi có milestone, để agent được đánh thức.

Khác orchestrator (sống nhiều giờ, chỉ thoát khi xong hết): script này chỉ tồn tại
tối đa `--max-minutes`, và **exit** ngay khi:
  - có thêm 1 model hoàn thành (dòng `: completed in` hoặc `: time_budget in`)
  - kernel chuyển sang trạng thái terminal
  - có Traceback/Error trong log
  - hết max-minutes (check-in định kỳ)

Nhờ vậy agent nhận thông báo ở từng mốc, không phải chờ tới cuối.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATE = Path(os.environ.get("MILESTONE_STATE", "/tmp/opencode/milestone_state.json"))
TERMINAL = ("COMPLETE", "ERROR", "CANCEL", "TIMEOUT")

DONE_RE = re.compile(r"^(?P<model>[\w.\-]+):\s+(?P<how>completed|time_budget|max_batches)\s+in\s+(?P<minutes>[\d.]+)\s+min")
SKIP_RE = re.compile(r"^SKIP\s+(?P<model>[\w.\-]+):")
ERR_RE = re.compile(r"Traceback \(most recent call last\)|^\w*Error:")


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"seen_done": [], "seen_skip": []}


def save_state(state: dict) -> None:
    state["updated_at"] = ts()
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def kernel_status(kernel: str) -> str:
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True, text=True, timeout=120,
        )
        match = re.search(r'has status "([^"]+)"', (result.stdout or "") + (result.stderr or ""))
        return match.group(1) if match else "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--max-minutes", type=float, default=25)
    args = parser.parse_args()

    state = load_state()
    seen_done = set(state.get("seen_done", []))
    seen_skip = set(state.get("seen_skip", []))
    reason = "stream_end"
    status = "UNKNOWN"
    new_events: list[str] = []

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.max_minutes * 60))

    proc = subprocess.Popen(
        ["kaggle", "kernels", "logs", args.kernel, "-f"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            done = DONE_RE.match(line.strip())
            skip = SKIP_RE.match(line.strip())
            if done and done.group("model") not in seen_done:
                seen_done.add(done.group("model"))
                new_events.append(f"done:{done.group('model')}({done.group('how')},{done.group('minutes')}m)")
                reason = "model_done"
                break
            if skip and skip.group("model") not in seen_skip:
                seen_skip.add(skip.group("model"))
                new_events.append(f"skipped:{skip.group('model')}")
                reason = "model_skipped"
                break
            if ERR_RE.search(line):
                new_events.append(f"error:{line.strip()[:160]}")
                reason = "error_in_log"
                break
    except _Timeout:
        reason = "checkpoint_timeout"
    finally:
        signal.alarm(0)
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    status = kernel_status(args.kernel)
    if any(marker in status for marker in TERMINAL):
        reason = f"terminal:{status}"
        new_events.append(f"status:{status}")

    state["seen_done"] = sorted(seen_done)
    state["seen_skip"] = sorted(seen_skip)
    save_state(state)

    print(f"[{ts()}] WATCHER_EXIT reason={reason} status={status} "
          f"done={len(seen_done)} skipped={len(seen_skip)} events={new_events}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
