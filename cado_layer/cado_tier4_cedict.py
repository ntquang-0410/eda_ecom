#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4 — Chỉ dùng CC-CEDICT (không auto loanword) để kiểm tra overlap.

    python cado_layer/cado_tier4_cedict.py --limit 500
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
    parser.add_argument("--output", help="Parquet đầu ra (tuỳ chọn)")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    df = pd.read_parquet(resolve_path(args.input, RAW_PARQUET))
    if args.limit:
        df = df.head(args.limit)

    result = run_cado(
        df,
        CadoConfig(columns=("title_zh",), max_tier=4, use_loanwords=False, progress=print),
    )

    removed = result.occurrences[result.occurrences["action"] == "remove"]
    print(f"\n🔴 {len(removed):,} occurrence bị xoá bởi CEDICT")
    if not removed.empty:
        print("\nTop 20 từ:")
        print(removed["word"].value_counts().head(20).to_string())
        print("\nVí dụ:")
        for _, row in removed.head(15).iterrows():
            print(f"  #{row['product_id']}  {row['word']:<18} → {row['reason']}")

    if args.output:
        output_path = resolve_path(args.output, None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.df.to_parquet(output_path, index=False)
        print(f"\n💾 {output_path}")


if __name__ == "__main__":
    main()
