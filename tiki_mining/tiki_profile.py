#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile Tiki Phase 1: title + description key-value.

Ví dụ::

    python tiki_mining/tiki_profile.py
    python tiki_mining/tiki_profile.py --limit 200 --quiet
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
    TIKI_INPUT_PARQUET,
    TIKI_KEY_INVENTORY_CSV,
    TIKI_PARSE_ERRORS_CSV,
    TIKI_PROFILE_JSON,
    resolve_path,
)
from cado_core.tiki_units import profile_tiki


def _validate_product_ids(df: pd.DataFrame) -> None:
    ids = df["product_id"].astype(str).str.strip()
    if (ids == "").any():
        raise ValueError("Có product_id rỗng; cần xử lý trước khi profile.")
    duplicated = int(ids.duplicated().sum())
    if duplicated:
        raise ValueError(f"product_id bị trùng {duplicated:,} dòng; cần xử lý trước khi profile.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(TIKI_INPUT_PARQUET),
        help="Parquet Tiki đầu vào (mặc định data/cleaned/tiki_vi_cado.parquet)",
    )
    parser.add_argument(
        "--profile-output",
        default=str(TIKI_PROFILE_JSON),
        help="JSON profile đầu ra (mặc định data/cleaned/tiki_profile.json)",
    )
    parser.add_argument(
        "--key-inventory-output",
        default=str(TIKI_KEY_INVENTORY_CSV),
        help="CSV key inventory (mặc định csv/tiki/tiki_key_inventory.csv)",
    )
    parser.add_argument(
        "--parse-errors-output",
        default=str(TIKI_PARSE_ERRORS_CSV),
        help="CSV các phần description lỗi parse (mặc định csv/tiki/tiki_parse_errors.csv)",
    )
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    parser.add_argument("--quiet", action="store_true", help="Không in tiến độ")
    args = parser.parse_args()

    input_path = resolve_path(args.input, TIKI_INPUT_PARQUET)
    profile_path = resolve_path(args.profile_output, TIKI_PROFILE_JSON)
    inventory_path = resolve_path(args.key_inventory_output, TIKI_KEY_INVENTORY_CSV)
    errors_path = resolve_path(args.parse_errors_output, TIKI_PARSE_ERRORS_CSV)

    print("=" * 68)
    print("📊 TIKI PHASE 1 — PROFILE TITLE + DESCRIPTION KV")
    print("=" * 68)
    print(f"📂 Input          : {input_path}")
    print(f"📂 Profile        : {profile_path}")
    print(f"📂 Key inventory  : {inventory_path}")
    print(f"📂 Parse errors   : {errors_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"⚠️  Giới hạn {args.limit:,} dòng để test")
    _validate_product_ids(df)

    progress = None if args.quiet else print
    result = profile_tiki(df, progress=progress)
    stats = dict(result.stats)
    stats["input"] = str(input_path)

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    result.key_inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    result.parse_errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    title = stats["title"]
    attributes = stats["attributes"]
    description = stats["description"]
    print("\n" + "=" * 68)
    print("✅ HOÀN TẤT")
    print("=" * 68)
    print(f"Rows                : {stats['rows']:,}")
    print(f"Title 3–40 từ       : {title['n_3_40']:,} / {title['n_titles']:,}")
    print(f"Title trùng         : {title['duplicate_rows']:,} dòng / {title['duplicate_groups']:,} nhóm")
    print(f"Description rỗng    : {description['n_empty']:,}")
    print(f"Description lỗi     : {description['n_malformed_parts']:,} phần")
    print(f"Attribute pairs     : {attributes['n_pairs']:,}")
    print(f"Attribute values    : {attributes['n_values']:,}")
    print(f"Unique keys         : {stats['key_inventory_rows']:,}")
    print("\nValue classes:")
    for key_class, item in stats["value_classes"].items():
        print(
            f"  {key_class:<10} {item['n_values']:>8,} values "
            f"({item['value_words']:>9,} từ)"
        )
    print("\nTop keys:")
    for item in stats["top_keys"][:12]:
        print(
            f"  {item['n_pairs']:>7,}  {item['key_norm']:<42} "
            f"{item['key_class']}"
        )
    print(f"\n💾 {profile_path}")
    print(f"💾 {inventory_path} ({len(result.key_inventory):,} key)")
    print(f"💾 {errors_path} ({len(result.parse_errors):,} phần lỗi)")


if __name__ == "__main__":
    main()
