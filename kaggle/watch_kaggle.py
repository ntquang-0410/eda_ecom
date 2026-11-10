#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theo dõi Kaggle kernel cho tới khi session kết thúc.

Không chứa credential; dùng Kaggle CLI đã authenticate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time


def status(kernel: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        text=True,
        capture_output=True,
        timeout=60,
    )
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[-1] if text else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--interval", type=int, default=900, help="giây giữa các lần poll")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()

    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        current = status(args.kernel)
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] {args.kernel}: {current}", flush=True)
        lowered = current.lower()
        if "complete" in lowered or "error" in lowered or "cancel" in lowered:
            print("Kernel finished; kiểm tra output/checkpoint trước khi resume.")
            return
        time.sleep(args.interval)
    print("Watcher timeout; kiểm tra Kaggle UI/CLI thủ công.")


if __name__ == "__main__":
    main()
