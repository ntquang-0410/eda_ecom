#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A — length prior v2: value-only, value_class, CJK, split-safe.

Ví dụ::

    python tiki_mining/tiki_length_prior_v2.py
    python tiki_mining/tiki_length_prior_v2.py --fit-limit 2000 --limit 500 --quiet
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
    LENGTH_PRIOR_V2_ALL_JSON,
    LENGTH_PRIOR_V2_CSV,
    LENGTH_PRIOR_V2_HOLDOUT_JSON,
    LENGTH_PRIOR_V2_SPLIT_MISMATCH_CSV,
    LENGTH_PRIOR_V2_SPLIT_SAMPLE_CSV,
    LENGTH_PRIOR_V2_SPLIT_SUMMARY_CSV,
    LENGTH_PRIOR_V2_TRAIN_CSV,
    LENGTH_PRIOR_V2_TRAIN_JSON,
    TIKI_BT_UNITS_PARQUET,
    TIKI_LENGTH_PRIOR_V2_STATS_JSON,
    TIKI_UNITS_LENGTH_V2_PARQUET,
    resolve_path,
)
from cado_core.length_prior_v2 import (
    LengthPriorV2Config,
    apply_length_prior_v2,
    evaluate_length_prior_v2,
    extract_human_value_pairs,
    fit_length_prior_v2,
)


def _write_prior(
    prior,
    pairs_result,
    path: Path,
    *,
    bilingual_path: Path,
) -> None:
    payload = dict(prior.stats)
    payload["bilingual_input"] = str(bilingual_path)
    payload["split_audit"] = pairs_result.stats
    payload["groups"] = prior.table.to_dict("records")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_split_audit(
    audit: pd.DataFrame,
    *,
    mismatch_path: Path,
    sample_path: Path,
    summary_path: Path,
    sample_size: int,
    seed: int,
) -> dict:
    summary = (
        audit.groupby("split_status", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    total = int(summary["n"].sum())
    summary["pct"] = (
        summary["n"] / total if total else 0.0
    ).round(4)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    mismatch = audit[audit["split_status"] == "mismatch"].copy()
    mismatch_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    if len(audit) > sample_size:
        sample = audit.sample(n=sample_size, random_state=seed)
    else:
        sample = audit
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    return {
        "summary_path": str(summary_path),
        "mismatch_path": str(mismatch_path),
        "sample_path": str(sample_path),
        "n_mismatch": int(len(mismatch)),
        "n_sample": int(len(sample)),
        "summary": summary.to_dict("records"),
    }


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
        "--prior-train-output",
        default=str(LENGTH_PRIOR_V2_TRAIN_JSON),
        help="JSON prior fit train (mặc định data/cleaned/length_prior_1688_v2_train.json)",
    )
    parser.add_argument(
        "--prior-all-output",
        default=str(LENGTH_PRIOR_V2_ALL_JSON),
        help="JSON prior fit all (mặc định data/cleaned/length_prior_1688_v2_all.json)",
    )
    parser.add_argument(
        "--holdout-output",
        default=str(LENGTH_PRIOR_V2_HOLDOUT_JSON),
        help="JSON metrics dev/test (mặc định data/cleaned/length_prior_1688_v2_holdout_metrics.json)",
    )
    parser.add_argument(
        "--prior-csv",
        default=str(LENGTH_PRIOR_V2_CSV),
        help="CSV prior all (mặc định csv/length_prior_v2/prior_value_by_class_bucket.csv)",
    )
    parser.add_argument(
        "--prior-train-csv",
        default=str(LENGTH_PRIOR_V2_TRAIN_CSV),
        help="CSV prior train (mặc định csv/length_prior_v2/prior_value_by_class_bucket_train.csv)",
    )
    parser.add_argument(
        "--split-mismatch-output",
        default=str(LENGTH_PRIOR_V2_SPLIT_MISMATCH_CSV),
        help="CSV các group split mismatch",
    )
    parser.add_argument(
        "--split-sample-output",
        default=str(LENGTH_PRIOR_V2_SPLIT_SAMPLE_CSV),
        help="CSV sample audit tách value",
    )
    parser.add_argument(
        "--split-summary-output",
        default=str(LENGTH_PRIOR_V2_SPLIT_SUMMARY_CSV),
        help="CSV summary split status",
    )
    parser.add_argument(
        "--units-output",
        default=str(TIKI_UNITS_LENGTH_V2_PARQUET),
        help="Parquet Tiki units v2 (mặc định data/cleaned/tiki_bt_units_length_v2.parquet)",
    )
    parser.add_argument(
        "--stats-output",
        default=str(TIKI_LENGTH_PRIOR_V2_STATS_JSON),
        help="JSON stats apply v2 (mặc định data/cleaned/tiki_length_prior_v2_stats.json)",
    )
    parser.add_argument("--min-group-samples", type=int, default=20)
    parser.add_argument("--min-category-samples", type=int, default=100)
    parser.add_argument("--min-value-class-samples", type=int, default=50)
    parser.add_argument("--min-value-class-category-samples", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--fit-limit", type=int, help="Chỉ dùng N dòng 1688 đầu (để test)")
    parser.add_argument("--limit", type=int, help="Chỉ áp dụng N Tiki unit đầu (để test)")
    parser.add_argument("--quiet", action="store_true", help="Không in tiến độ")
    args = parser.parse_args()

    bilingual_path = resolve_path(args.bilingual, CADO_PARQUET)
    tiki_units_path = resolve_path(args.tiki_units, TIKI_BT_UNITS_PARQUET)
    prior_train_path = resolve_path(args.prior_train_output, LENGTH_PRIOR_V2_TRAIN_JSON)
    prior_all_path = resolve_path(args.prior_all_output, LENGTH_PRIOR_V2_ALL_JSON)
    holdout_path = resolve_path(args.holdout_output, LENGTH_PRIOR_V2_HOLDOUT_JSON)
    prior_csv_path = resolve_path(args.prior_csv, LENGTH_PRIOR_V2_CSV)
    prior_train_csv_path = resolve_path(args.prior_train_csv, LENGTH_PRIOR_V2_TRAIN_CSV)
    mismatch_path = resolve_path(args.split_mismatch_output, LENGTH_PRIOR_V2_SPLIT_MISMATCH_CSV)
    sample_path = resolve_path(args.split_sample_output, LENGTH_PRIOR_V2_SPLIT_SAMPLE_CSV)
    split_summary_path = resolve_path(args.split_summary_output, LENGTH_PRIOR_V2_SPLIT_SUMMARY_CSV)
    units_output_path = resolve_path(args.units_output, TIKI_UNITS_LENGTH_V2_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_LENGTH_PRIOR_V2_STATS_JSON)

    print("=" * 72)
    print("📏 PHASE A — LENGTH PRIOR V2 (VALUE-ONLY / CLASS / CJK / SPLIT-SAFE)")
    print("=" * 72)
    print(f"📂 1688 human    : {bilingual_path}")
    print(f"📂 Tiki units    : {tiki_units_path}")
    print(f"📂 Prior train   : {prior_train_path}")
    print(f"📂 Prior all     : {prior_all_path}")
    print(f"📂 Holdout       : {holdout_path}")
    print(f"📂 Units v2      : {units_output_path}")
    print(f"📂 Apply stats   : {stats_path}")

    bilingual = pd.read_parquet(bilingual_path)
    if args.fit_limit:
        bilingual = bilingual.head(args.fit_limit).copy()
        print(f"⚠️  Giới hạn fit {args.fit_limit:,} dòng 1688")

    base_config = LengthPriorV2Config(
        min_group_samples=args.min_group_samples,
        min_category_samples=args.min_category_samples,
        min_value_class_samples=args.min_value_class_samples,
        min_value_class_category_samples=args.min_value_class_category_samples,
        split_seed=args.split_seed,
    )
    progress = None if args.quiet else print

    extracted = extract_human_value_pairs(
        bilingual,
        config=base_config,
        progress=progress,
    )
    pairs = extracted.pairs
    print(
        f"\n✅ Value pairs: {extracted.stats['n_value_pairs']:,} "
        f"+ title {extracted.stats['n_title_pairs']:,} "
        f"(group audit {extracted.stats['n_group_pairs']:,})"
    )
    print(f"   Split status: {extracted.stats['split_status_counts']}")
    print(f"   Split counts: {extracted.stats['split_counts']}")

    train_config = LengthPriorV2Config(
        **{
            **base_config.__dict__,
            "fit_split": "train",
        }
    )
    all_config = LengthPriorV2Config(
        **{
            **base_config.__dict__,
            "fit_split": "all",
        }
    )
    train_prior = fit_length_prior_v2(
        pairs,
        config=train_config,
        progress=progress,
    )
    all_prior = fit_length_prior_v2(
        pairs,
        config=all_config,
        progress=progress,
    )

    _write_prior(
        train_prior,
        extracted,
        prior_train_path,
        bilingual_path=bilingual_path,
    )
    _write_prior(
        all_prior,
        extracted,
        prior_all_path,
        bilingual_path=bilingual_path,
    )
    prior_train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_prior.table.to_csv(
        prior_train_csv_path, index=False, encoding="utf-8-sig"
    )
    prior_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_prior.table.to_csv(prior_csv_path, index=False, encoding="utf-8-sig")

    split_audit_info = _write_split_audit(
        extracted.split_audit,
        mismatch_path=mismatch_path,
        sample_path=sample_path,
        summary_path=split_summary_path,
        sample_size=args.sample_size,
        seed=args.split_seed,
    )

    holdout = {
        "fit_split": "train",
        "train_prior": str(prior_train_path),
        "splits": {},
    }
    for split in ("dev", "test"):
        holdout["splits"][split] = evaluate_length_prior_v2(
            pairs,
            train_prior,
            split=split,
            config=train_config,
        )
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with holdout_path.open("w", encoding="utf-8") as handle:
        json.dump(holdout, handle, ensure_ascii=False, indent=2)

    units = pd.read_parquet(tiki_units_path)
    if args.limit:
        units = units.head(args.limit).copy()
        print(f"⚠️  Giới hạn apply {args.limit:,} Tiki unit")
    applied = apply_length_prior_v2(
        units,
        all_prior,
        config=all_config,
        progress=progress,
    )
    apply_stats = dict(applied.stats)
    apply_stats.update({
        "bilingual_input": str(bilingual_path),
        "tiki_units_input": str(tiki_units_path),
        "units_output": str(units_output_path),
        "prior_used": str(prior_all_path),
        "prior_train": str(prior_train_path),
        "holdout_metrics": str(holdout_path),
        "split_audit": split_audit_info,
    })
    units_output_path.parent.mkdir(parents=True, exist_ok=True)
    applied.units.to_parquet(
        units_output_path, index=False, compression="snappy"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(apply_stats, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("✅ PHASE A HOÀN TẤT")
    print("=" * 72)
    print(f"Train prior pairs : {train_prior.stats['n_pairs']:,}")
    print(f"All prior pairs   : {all_prior.stats['n_pairs']:,}")
    print(f"Prior groups all  : {all_prior.stats['n_groups']:,}")
    for split in ("dev", "test"):
        metrics = holdout["splits"][split]
        print(
            f"Holdout {split:<4}    : n={metrics.get('n', 0):,} "
            f"coverage={metrics.get('overall', {}).get('coverage_value_p05_p95', 0):.3f} "
            f"MAE={metrics.get('overall', {}).get('mae_value_chars', 0):.3f}"
        )
    print(f"Tiki units        : {apply_stats['n_units']:,}")
    print(f"Expected value p50: {apply_stats['expected_zh_value_chars']['p50']}")
    print(f"Expected text p50 : {apply_stats['expected_zh_text_chars']['p50']}")
    print(f"Scopes            : {apply_stats['scope_counts']}")
    print(f"Confidence        : {apply_stats['confidence_counts']}")
    print(f"\n💾 {prior_train_path}")
    print(f"💾 {prior_all_path}")
    print(f"💾 {holdout_path}")
    print(f"💾 {prior_csv_path}")
    print(f"💾 {prior_train_csv_path}")
    print(f"💾 {split_summary_path}")
    print(f"💾 {mismatch_path}")
    print(f"💾 {sample_path}")
    print(f"💾 {units_output_path}")
    print(f"💾 {stats_path}")


if __name__ == "__main__":
    main()
