#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA toàn bộ tập Tiki và xuất một file parquet product-level.

Ví dụ::

    python tiki_mining/tiki_eda.py
    python tiki_mining/tiki_eda.py --limit 200 --output /tmp/tiki_eda_test.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.config import PROJECT_ROOT as ROOT, resolve_path
from cado_core.tiki import run_tiki_eda


DEFAULT_INPUT = ROOT / "data/cleaned/tiki_vi_cado.parquet"
DEFAULT_OUTPUT = ROOT / "data/cleaned/tiki_eda.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Parquet Tiki đầu vào (mặc định data/cleaned/tiki_vi_cado.parquet)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Parquet EDA đầu ra (mặc định data/cleaned/tiki_eda.parquet)",
    )
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    parser.add_argument("--quiet", action="store_true", help="Không in tiến độ")
    args = parser.parse_args()

    input_path = resolve_path(args.input, DEFAULT_INPUT)
    output_path = resolve_path(args.output, DEFAULT_OUTPUT)

    print("=" * 68)
    print("📊 TIKI EDA — CHUẨN BỊ BACK-TRANSLATION")
    print("=" * 68)
    print(f"📂 Input : {input_path}")
    print(f"📂 Output: {output_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"⚠️  Giới hạn {args.limit:,} dòng để test")

    if df["product_id"].duplicated().any():
        duplicated = int(df["product_id"].duplicated().sum())
        raise ValueError(f"product_id bị trùng {duplicated:,} dòng; cần xử lý trước EDA.")

    progress = None if args.quiet else print
    result = run_tiki_eda(df, progress=progress)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_parquet(output_path, index=False, compression="snappy")

    stats = result.stats
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 68)
    print("✅ HOÀN TẤT")
    print("=" * 68)
    print(f"Rows     : {stats['rows']:,}")
    print(f"Columns  : {stats['columns']:,}")
    print(f"File     : {output_path} ({size_mb:.1f} MB)")
    print(f"BT ready : {stats['bt_ready']:,} / {stats['rows']:,}  (title hoặc prose)")
    print(f"  title  : {stats['title_ready']:,}")
    print(f"  prose  : {stats['prose_ready']:,}")
    print(f"  attrs  : {stats['attrs_ready']:,}")
    print("\nRoutes:")
    for route, count in sorted(
        stats["overall_routes"].items(), key=lambda item: -item[1]
    ):
        print(f"  {route:<18} {count:>7,}")
    print("\nTop flags:")
    for flag, count in stats["flags"][:20]:
        print(f"  {flag:<28} {count:>7,}")
    print("\nTop foreign tokens:")
    for token, count in stats["top_foreign_tokens"][:20]:
        print(f"  {token:<28} {count:>7,}")
    print(f"\nBoilerplate removed : {stats['boilerplate_removed_rows']:,} rows")
    print(f"Title duplicates    : {stats['title_duplicates']:,} rows")
    print(f"Prose duplicates    : {stats['prose_duplicates']:,} rows")


if __name__ == "__main__":
    main()
