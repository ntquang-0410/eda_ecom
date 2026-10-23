#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Học length prior từ 1688 human và áp dụng cho Tiki units.

Ví dụ::

    python tiki_mining/tiki_length_prior.py
    python tiki_mining/tiki_length_prior.py --fit-limit 2000 --limit 500 --quiet
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
    CADO_PARQUET,
    LENGTH_PRIOR_CSV,
    LENGTH_PRIOR_JSON,
    TIKI_BT_UNITS_PARQUET,
    TIKI_LENGTH_PRIOR_STATS_JSON,
    TIKI_UNITS_LENGTH_PARQUET,
    resolve_path,
)
from cado_core.length_prior import (
    LengthPriorConfig,
    apply_length_prior,
    extract_human_length_pairs,
    fit_length_prior,
    write_prior_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bilingual",
        default=str(CADO_PARQUET),
        help="Parquet 1688 human ZH–VI (mặc định data/cleaned/bilingual_zh_vi_cado.parquet)",
    )
    parser.add_argument(
        "--tiki-units",
        default=str(TIKI_BT_UNITS_PARQUET),
        help="Parquet Tiki units (mặc định data/cleaned/tiki_bt_units.parquet)",
    )
    parser.add_argument(
        "--prior-output",
        default=str(LENGTH_PRIOR_JSON),
        help="JSON prior (mặc định data/cleaned/length_prior_1688.json)",
    )
    parser.add_argument(
        "--prior-csv",
        default=str(LENGTH_PRIOR_CSV),
        help="CSV bảng prior theo bucket (mặc định csv/length_prior/length_prior_by_bucket.csv)",
    )
    parser.add_argument(
        "--units-output",
        default=str(TIKI_UNITS_LENGTH_PARQUET),
        help="Parquet Tiki units đã gắn prior (mặc định data/cleaned/tiki_bt_units_length.parquet)",
    )
    parser.add_argument(
        "--stats-output",
        default=str(TIKI_LENGTH_PRIOR_STATS_JSON),
        help="JSON thống kê apply (mặc định data/cleaned/tiki_length_prior_stats.json)",
    )
    parser.add_argument(
        "--min-group-samples",
        type=int,
        default=20,
        help="Số cặp tối thiểu cho group unit_type+bucket (mặc định 20)",
    )
    parser.add_argument(
        "--min-category-samples",
        type=int,
        default=100,
        help="Số cặp tối thiểu cho group có category (mặc định 100)",
    )
    parser.add_argument("--fit-limit", type=int, help="Chỉ dùng N dòng 1688 đầu (để test)")
    parser.add_argument("--limit", type=int, help="Chỉ áp dụng cho N Tiki unit đầu (để test)")
    parser.add_argument("--quiet", action="store_true", help="Không in tiến độ")
    args = parser.parse_args()

    bilingual_path = resolve_path(args.bilingual, CADO_PARQUET)
    tiki_units_path = resolve_path(args.tiki_units, TIKI_BT_UNITS_PARQUET)
    prior_path = resolve_path(args.prior_output, LENGTH_PRIOR_JSON)
    prior_csv_path = resolve_path(args.prior_csv, LENGTH_PRIOR_CSV)
    units_output_path = resolve_path(args.units_output, TIKI_UNITS_LENGTH_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_LENGTH_PRIOR_STATS_JSON)

    print("=" * 72)
    print("📏 LENGTH PRIOR — 1688 HUMAN → TIKI VI")
    print("=" * 72)
    print(f"📂 1688 human    : {bilingual_path}")
    print(f"📂 Tiki units    : {tiki_units_path}")
    print(f"📂 Prior JSON    : {prior_path}")
    print(f"📂 Prior CSV     : {prior_csv_path}")
    print(f"📂 Units output  : {units_output_path}")
    print(f"📂 Apply stats   : {stats_path}")

    bilingual = pd.read_parquet(bilingual_path)
    if args.fit_limit:
        bilingual = bilingual.head(args.fit_limit).copy()
        print(f"⚠️  Giới hạn fit {args.fit_limit:,} dòng 1688")

    progress = None if args.quiet else print
    pairs = extract_human_length_pairs(bilingual, progress=progress)
    config = LengthPriorConfig(
        min_group_samples=args.min_group_samples,
        min_category_samples=args.min_category_samples,
    )
    prior = fit_length_prior(pairs, config=config, progress=progress)

    prior_payload = dict(prior.stats)
    prior_payload["bilingual_input"] = str(bilingual_path)
    prior_payload["groups"] = prior.table.to_dict("records")
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    with prior_path.open("w", encoding="utf-8") as handle:
        json.dump(prior_payload, handle, ensure_ascii=False, indent=2)
    write_prior_table(prior.table, prior_csv_path)

    units = pd.read_parquet(tiki_units_path)
    if args.limit:
        units = units.head(args.limit).copy()
        print(f"⚠️  Giới hạn apply {args.limit:,} Tiki unit")
    applied = apply_length_prior(
        units,
        prior,
        config=config,
        progress=progress,
    )
    apply_stats = dict(applied.stats)
    apply_stats.update({
        "bilingual_input": str(bilingual_path),
        "tiki_units_input": str(tiki_units_path),
        "units_output": str(units_output_path),
        "prior_json": str(prior_path),
        "prior_csv": str(prior_csv_path),
    })

    units_output_path.parent.mkdir(parents=True, exist_ok=True)
    applied.units.to_parquet(units_output_path, index=False, compression="snappy")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(apply_stats, handle, ensure_ascii=False, indent=2)

    overall = prior.stats["overall"]
    by_unit = prior.stats["by_unit_type"]
    print("\n" + "=" * 72)
    print("✅ HOÀN TẤT")
    print("=" * 72)
    print(f"Human pairs         : {prior.stats['n_pairs']:,}")
    for unit_type, count in prior.stats["pair_counts"].items():
        print(f"  {unit_type:<12} {count:>9,}")
    print(f"Ratio VI/ZH p50     : {overall['ratio_vi_zh_p50']:.3f}")
    for unit_type in ("title", "attribute", "description"):
        if unit_type in by_unit:
            print(
                f"  {unit_type:<12} n={by_unit[unit_type]['n']:>7,} "
                f"ratio p50={by_unit[unit_type]['ratio_vi_zh_p50']:.3f}"
            )
    print(f"Tiki units          : {apply_stats['n_units']:,}")
    print(f"Expected ZH chars   : {apply_stats['expected_zh_chars']}")
    print("Scope counts:")
    for scope, count in apply_stats["scope_counts"].items():
        print(f"  {scope:<28} {count:>9,}")
    print(f"\n💾 {prior_path}")
    print(f"💾 {prior_csv_path} ({len(prior.table):,} group)")
    print(f"💾 {units_output_path}")
    print(f"💾 {stats_path}")


if __name__ == "__main__":
    main()
