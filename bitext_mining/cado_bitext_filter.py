#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bitext filter đầy đủ: preprocess + cosine + lexical + label.

Mặc định đọc output CADO (đã có ``title_zh_cleaned`` / ``title_vi_cleaned``),
không đọc raw như bản cũ — nhờ đó CADO thực sự có tác dụng lên filter.

    python bitext_mining/cado_bitext_filter.py
    python bitext_mining/cado_bitext_filter.py --label-mode hybrid
    python bitext_mining/cado_bitext_filter.py --preprocess-only
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
from cado_core.config import CADO_PARQUET, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Parquet đầu vào (mặc định CADO output)")
    parser.add_argument("--output", help="Parquet scored đầu ra")
    parser.add_argument("--ready-output", help="Parquet preprocessed đầu ra")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--skip-cosine", action="store_true",
                        help="Dùng cột cosine_bge_m3 có sẵn trong input")
    parser.add_argument("--label-mode", choices=("cosine", "hybrid"), default="cosine")
    parser.add_argument("--save-ready", action="store_true")
    parser.add_argument("--entity-signals", action="store_true")
    parser.add_argument("--entity-embedding", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    input_path = resolve_path(args.input, CADO_PARQUET)
    defaults = BitextConfig()
    config = BitextConfig(
        input_path=input_path,
        output_parquet=resolve_path(args.output, defaults.output_parquet),
        ready_parquet=resolve_path(args.ready_output, defaults.ready_parquet),
        label_mode=args.label_mode,
        limit=args.limit,
        save_ready=args.save_ready,
        entity_signals=args.entity_signals,
        entity_embedding=args.entity_embedding,
        progress=print,
    )

    df = pd.read_parquet(input_path)
    run_bitext(
        df,
        config,
        skip_cosine=args.skip_cosine,
        preprocess_only=args.preprocess_only,
    )


if __name__ == "__main__":
    main()
