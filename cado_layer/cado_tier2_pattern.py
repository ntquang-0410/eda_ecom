#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 — Preview quyết định của whitelist + pattern (không xoá gì).

    python cado_layer/cado_tier2_pattern.py --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.cado import CadoConfig, run_cado
from cado_core.config import RAW_PARQUET, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Parquet đầu vào (mặc định data/raw/bilingual_zh_vi.parquet)")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    df = pd.read_parquet(resolve_path(args.input, RAW_PARQUET))
    if args.limit:
        df = df.head(args.limit)

    result = run_cado(
        df,
        CadoConfig(
            columns=("title_zh",),
            max_tier=2,
            use_loanwords=False,
            progress=print,
        ),
    )

    occurrences = result.occurrences
    print("\n📋 Quyết định T1/T2 (mẫu):")
    if occurrences.empty:
        print("  (không có từ Latin nào)")
    else:
        counts = occurrences.groupby(["reason", "action"]).size().sort_values(ascending=False)
        print(counts.to_string())
        print("\n  Ví dụ:")
        for _, row in occurrences.head(15).iterrows():
            print(f"    {row['word']:<20} → {row['action']:<8} ({row['reason']})")


if __name__ == "__main__":
    main()
