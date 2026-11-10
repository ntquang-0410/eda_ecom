#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh kernel-metadata.json cho Kaggle từ owner/dataset slug."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default=os.getenv("KAGGLE_USERNAME"))
    parser.add_argument("--dataset-slug", default="tiki-plan-a-bt-input")
    parser.add_argument("--kernel-slug", default="tiki-plan-b-env-check")
    parser.add_argument("--title", default="Tiki Plan B Env Check")
    parser.add_argument("--code-file", default="plan_b_env_check.py")
    parser.add_argument(
        "--accelerator",
        choices=["tpu", "gpu", "none"],
        default="tpu",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "kernel-metadata.json"),
    )
    args = parser.parse_args()
    if not args.owner:
        raise SystemExit("Thiếu --owner hoặc KAGGLE_USERNAME")
    metadata = {
        "id": f"{args.owner}/{args.kernel_slug}",
        "title": args.title,
        "code_file": args.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": args.accelerator == "gpu",
        "enable_tpu": args.accelerator == "tpu",
        "enable_internet": True,
        "dataset_sources": [f"{args.owner}/{args.dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    Path(args.output).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
