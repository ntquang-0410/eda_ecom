#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watcher cho Kaggle kernel — sống ngắn, exit để đánh thức agent.

    python tools/watch_kernel.py --kernel owner/slug --until terminal --max-minutes 90
    python tools/watch_kernel.py --kernel owner/slug --until running

`--until terminal` : exit khi COMPLETE/ERROR/CANCEL/TIMEOUT
`--until running`  : exit khi bắt đầu RUNNING (hoặc terminal)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = ("COMPLETE", "ERROR", "CANCEL", "TIMEOUT")
STATE = Path("/tmp/opencode/watch_kernel_state.json")


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def status_of(kernel: str) -> str:
    try:
        result = subprocess.run(["kaggle", "kernels", "status", kernel],
                                capture_output=True, text=True, timeout=120)
        match = re.search(r'has status "([^"]+)"', (result.stdout or "") + (result.stderr or ""))
        return match.group(1) if match else "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--until", choices=("terminal", "running"), default="terminal")
    ap.add_argument("--max-minutes", type=float, default=90)
    ap.add_argument("--interval", type=float, default=45)
    args = ap.parse_args()

    deadline = time.time() + args.max_minutes * 60
    status, reason = "UNKNOWN", "timeout"
    while time.time() < deadline:
        status = status_of(args.kernel)
        if any(t in status for t in TERMINAL):
            reason = "terminal"
            break
        if args.until == "running" and "RUNNING" in status:
            reason = "running"
            break
        time.sleep(args.interval)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"kernel": args.kernel, "reason": reason,
                                 "status": status, "updated_at": ts()},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{ts()}] WATCH_EXIT reason={reason} status={status} kernel={args.kernel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
