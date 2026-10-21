#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess bitext (V1–V16) — không cần model.

    python bitext_mining/preprocess_bitext.py
    python bitext_mining/preprocess_bitext.py --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.bitext import BitextConfig, run_bitext
from cado_core.config import BITEXT_READY_PARQUET, CADO_PARQUET, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Parquet đầu vào (mặc định CADO output)")
    parser.add_argument("--output", help="Parquet preprocessed đầu ra")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    input_path = resolve_path(args.input, CADO_PARQUET)
    output_path = resolve_path(args.output, BITEXT_READY_PARQUET)

    df = pd.read_parquet(input_path)
    config = BitextConfig(
        input_path=input_path,
        ready_parquet=output_path,
        limit=args.limit,
        progress=print,
    )
    run_bitext(df, config, preprocess_only=True)


if __name__ == "__main__":
    main()
