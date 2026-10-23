#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Tiki Phase 1 BT units: title + description key-value.

Không gọi model. Output gồm product-level parquet và pair-level parquet.

Ví dụ::

    python tiki_mining/tiki_bt_units.py
    python tiki_mining/tiki_bt_units.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.config import (
    TIKI_BT_UNITS_PARQUET,
    TIKI_INPUT_PARQUET,
    TIKI_KEY_MAP_CSV,
    TIKI_PRODUCTS_PARQUET,
    TIKI_UNITS_STATS_JSON,
    resolve_path,
)
from cado_core.tiki_units import build_tiki_units


def _validate_product_ids(df: pd.DataFrame) -> None:
    ids = df["product_id"].astype(str).str.strip()
    if (ids == "").any():
        raise ValueError("Có product_id rỗng; cần xử lý trước khi tạo units.")
    duplicated = int(ids.duplicated().sum())
    if duplicated:
        raise ValueError(f"product_id bị trùng {duplicated:,} dòng; cần xử lý trước khi tạo units.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(TIKI_INPUT_PARQUET),
        help="Parquet Tiki đầu vào (mặc định data/cleaned/tiki_vi_cado.parquet)",
    )
    parser.add_argument(
        "--products-output",
        default=str(TIKI_PRODUCTS_PARQUET),
        help="Parquet product-level (mặc định data/cleaned/tiki_products.parquet)",
    )
    parser.add_argument(
        "--units-output",
        default=str(TIKI_BT_UNITS_PARQUET),
        help="Parquet pair-level (mặc định data/cleaned/tiki_bt_units.parquet)",
    )
    parser.add_argument(
        "--stats-output",
        default=str(TIKI_UNITS_STATS_JSON),
        help="JSON stats (mặc định data/cleaned/tiki_units_stats.json)",
    )
    parser.add_argument(
        "--key-map",
        default=str(TIKI_KEY_MAP_CSV),
        help="Glossary VI→ZH nếu có (mặc định csv/tiki/tiki_key_map.csv)",
    )
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    parser.add_argument("--quiet", action="store_true", help="Không in tiến độ")
    args = parser.parse_args()

    input_path = resolve_path(args.input, TIKI_INPUT_PARQUET)
    products_path = resolve_path(args.products_output, TIKI_PRODUCTS_PARQUET)
    units_path = resolve_path(args.units_output, TIKI_BT_UNITS_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_UNITS_STATS_JSON)
    key_map_path = resolve_path(args.key_map, TIKI_KEY_MAP_CSV)

    print("=" * 68)
    print("🧱 TIKI PHASE 1 — BUILD BT UNITS")
    print("=" * 68)
    print(f"📂 Input        : {input_path}")
    print(f"📂 Products     : {products_path}")
    print(f"📂 Units        : {units_path}")
    print(f"📂 Stats        : {stats_path}")
    print(f"📂 Key map      : {key_map_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"⚠️  Giới hạn {args.limit:,} dòng để test")
    _validate_product_ids(df)

    progress = None if args.quiet else print
    result = build_tiki_units(
        df,
        key_map_path=key_map_path,
        progress=progress,
    )
    stats = dict(result.stats)
    stats["input"] = str(input_path)
    stats["products_output"] = str(products_path)
    stats["units_output"] = str(units_path)
    stats["key_map_path"] = str(key_map_path)

    products_path.parent.mkdir(parents=True, exist_ok=True)
    result.products.to_parquet(products_path, index=False, compression="snappy")
    units_path.parent.mkdir(parents=True, exist_ok=True)
    result.units.to_parquet(units_path, index=False, compression="snappy")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print("✅ HOÀN TẤT")
    print("=" * 68)
    print(f"Products            : {stats['products']:,}")
    print(f"Title units         : {stats['title_units']:,}")
    print(f"Attribute units     : {stats['attribute_units']:,}")
    print(f"Total units         : {stats['total_units']:,}")
    print(f"Train-eligible      : {stats['train_eligible_units']:,}")
    print(f"Empty products      : {stats['empty_products']:,}")
    print(f"Malformed desc parts: {stats['malformed_description_parts']:,}")
    print(f"Key map entries     : {stats['key_map_entries']:,}")
    print(f"Pending key map     : {stats['pending_key_map']:,} units")
    print("\nValue classes:")
    for key_class, count in stats["value_classes"].items():
        print(f"  {key_class:<10} {count:>8,}")
    print("\nBT methods (attribute units):")
    for method, count in stats["bt_methods"].items():
        print(f"  {method:<12} {count:>8,}")
    print(f"\n💾 {products_path} ({products_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"💾 {units_path} ({units_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"💾 {stats_path}")


if __name__ == "__main__":
    main()
