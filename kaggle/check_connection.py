#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra Kaggle CLI/credentials mà không in key."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    print("Kaggle connection check")
    print("=" * 50)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    print("kaggle.json:", "present" if kaggle_json.exists() else "missing")
    print("access_token:", "present" if access_token.exists() else "missing")
    print("KAGGLE_USERNAME:", "set" if os.getenv("KAGGLE_USERNAME") else "missing")
    print("KAGGLE_KEY:", "set" if os.getenv("KAGGLE_KEY") else "missing")
    try:
        version = subprocess.run(
            ["kaggle", "--version"],
            text=True,
            capture_output=True,
            timeout=20,
        )
        print("CLI:", (version.stdout or version.stderr).strip().splitlines()[0][:120])
    except Exception as error:  # noqa: BLE001
        print("CLI error:", error)
        raise SystemExit(1)

    config = subprocess.run(
        ["kaggle", "config", "view"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    print("config:", (config.stdout or config.stderr).strip()[:500])

    result = subprocess.run(
        ["kaggle", "datasets", "list", "--csv"],
        text=True,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        print("AUTH: FAILED")
        print((result.stderr or result.stdout).strip()[:800])
        raise SystemExit(2)
    print("AUTH: OK")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if lines:
        print("datasets response lines:", len(lines))


if __name__ == "__main__":
    main()
