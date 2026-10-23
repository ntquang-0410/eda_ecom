#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan A — chuẩn bị Tiki trước Back-Translation (không gọi MT).

Ví dụ::

    python tiki_mining/tiki_bt_prepare.py
    python tiki_mining/tiki_bt_prepare.py --limit 500 --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.config import (
    LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV,
    TIKI_BT_INPUT_PARQUET,
    TIKI_BT_INPUT_SAMPLE_CSV,
    TIKI_BT_PLAN_A_STATS_JSON,
    TIKI_BT_ROUTING_CSV,
    TIKI_BT_UNITS_PLAN_A_PARQUET,
    TIKI_KEY_MAP_CSV,
    TIKI_KEY_MAP_REVIEW_CSV,
    TIKI_PRODUCTS_BT_STATE_PARQUET,
    TIKI_PRODUCTS_PARQUET,
    TIKI_TRANSLATION_CACHE_PARQUET,
    TIKI_UNITS_LENGTH_V3_PARQUET,
    resolve_path,
)
from cado_core.tiki_bt_prepare import (
    build_tiki_key_map,
    export_plan_a,
    prepare_bt_input,
)


def run_from_args(args: argparse.Namespace) -> None:
    units_path = resolve_path(args.units, TIKI_UNITS_LENGTH_V3_PARQUET)
    products_path = resolve_path(args.products, TIKI_PRODUCTS_PARQUET)
    alignment_map_path = resolve_path(
        args.alignment_map, LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV
    )
    key_map_path = resolve_path(args.key_map, TIKI_KEY_MAP_CSV)
    key_map_review_path = resolve_path(
        args.key_map_review, TIKI_KEY_MAP_REVIEW_CSV
    )
    units_output_path = resolve_path(
        args.units_output, TIKI_BT_UNITS_PLAN_A_PARQUET
    )
    input_output_path = resolve_path(args.input_output, TIKI_BT_INPUT_PARQUET)
    cache_output_path = resolve_path(
        args.cache_output, TIKI_TRANSLATION_CACHE_PARQUET
    )
    product_state_output_path = resolve_path(
        args.product_state_output, TIKI_PRODUCTS_BT_STATE_PARQUET
    )
    routing_output_path = resolve_path(
        args.routing_output, TIKI_BT_ROUTING_CSV
    )
    sample_output_path = resolve_path(
        args.sample_output, TIKI_BT_INPUT_SAMPLE_CSV
    )
    stats_output_path = resolve_path(
        args.stats_output, TIKI_BT_PLAN_A_STATS_JSON
    )

    print("=" * 72)
    print("🧭 PLAN A — PRE-BT PREPARATION (KEY MAP / ROUTING / CACHE / STATE)")
    print("=" * 72)
    print(f"📂 Units v3       : {units_path}")
    print(f"📂 Products       : {products_path}")
    print(f"📂 Alignment map  : {alignment_map_path}")
    print(f"📂 Key map        : {key_map_path}")
    print(f"📂 BT input       : {input_output_path}")
    print(f"📂 Cache          : {cache_output_path}")
    print(f"📂 Product state  : {product_state_output_path}")
    print(f"📂 Stats          : {stats_output_path}")

    units = pd.read_parquet(units_path)
    products = pd.read_parquet(products_path)
    if args.limit:
        units = units.head(args.limit).copy()
        products = products[
            products["product_id"].isin(units["product_id"].unique())
        ].copy()
        print(f"⚠️  Giới hạn {args.limit:,} unit")

    key_map = build_tiki_key_map(
        units,
        alignment_map_path=alignment_map_path,
        existing_map_path=key_map_path,
        min_key_freq=args.min_key_freq,
        min_key_share=args.min_key_share,
    )
    result = prepare_bt_input(
        units,
        products,
        key_map,
        separator_chars=args.separator_chars,
    )
    paths = export_plan_a(
        result,
        units_path=units_output_path,
        input_path=input_output_path,
        cache_path=cache_output_path,
        product_state_path=product_state_output_path,
        key_map_path=key_map_path,
        key_map_review_path=key_map_review_path,
        routing_path=routing_output_path,
        sample_path=sample_output_path,
        stats_path=stats_output_path,
    )

    stats = result.stats
    print("\n" + "=" * 72)
    print("✅ PLAN A HOÀN TẤT")
    print("=" * 72)
    print(f"Keys mapped       : {key_map.stats['n_mapped_keys']:,}/{key_map.stats['n_keys']:,}")
    print(f"Unit coverage     : {key_map.stats['attribute_unit_coverage']:.2%}")
    print(f"Train coverage    : {key_map.stats['train_attribute_unit_coverage']:.2%}")
    print(f"BT ready units    : {stats['n_bt_ready']:,}/{stats['n_units']:,}")
    print(f"Merge ready units : {stats['n_merge_ready']:,}")
    print(f"Routes            : {stats['route_counts']}")
    print(f"BT status         : {stats['bt_status_counts']}")
    print(f"Cache keys        : {stats['n_cache_keys']:,} (reduction {stats['cache_reduction']:.2%})")
    print(f"Products ready    : {stats['n_product_ready_for_bt']:,}")
    print(f"Products blocked  : key_map={stats['n_product_blocked_key_map']:,}; source={stats['n_product_blocked_source']:,}")
    print(f"Merge eligible    : {stats['n_product_merge_eligible']:,} (chờ BT/QC)")
    for label, path in paths.items():
        print(f"💾 {label:<14} {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", default=str(TIKI_UNITS_LENGTH_V3_PARQUET))
    parser.add_argument("--products", default=str(TIKI_PRODUCTS_PARQUET))
    parser.add_argument("--alignment-map", default=str(LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV))
    parser.add_argument("--key-map", default=str(TIKI_KEY_MAP_CSV))
    parser.add_argument("--key-map-review", default=str(TIKI_KEY_MAP_REVIEW_CSV))
    parser.add_argument("--units-output", default=str(TIKI_BT_UNITS_PLAN_A_PARQUET))
    parser.add_argument("--input-output", default=str(TIKI_BT_INPUT_PARQUET))
    parser.add_argument("--cache-output", default=str(TIKI_TRANSLATION_CACHE_PARQUET))
    parser.add_argument("--product-state-output", default=str(TIKI_PRODUCTS_BT_STATE_PARQUET))
    parser.add_argument("--routing-output", default=str(TIKI_BT_ROUTING_CSV))
    parser.add_argument("--sample-output", default=str(TIKI_BT_INPUT_SAMPLE_CSV))
    parser.add_argument("--stats-output", default=str(TIKI_BT_PLAN_A_STATS_JSON))
    parser.add_argument("--min-key-freq", type=int, default=20)
    parser.add_argument("--min-key-share", type=float, default=0.60)
    parser.add_argument("--separator-chars", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
