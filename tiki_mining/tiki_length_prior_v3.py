#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase B/C — alignment fallback + hierarchical title smoothing (v3).

Ví dụ::

    python tiki_mining/tiki_length_prior_v3.py
    python tiki_mining/tiki_length_prior_v3.py --fit-limit 2000 --limit 500 --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.config import (
    CADO_PARQUET,
    LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV,
    LENGTH_PRIOR_V3_ALIGNMENT_SAMPLE_CSV,
    LENGTH_PRIOR_V3_ALIGNMENT_SUMMARY_CSV,
    LENGTH_PRIOR_V3_ALL_JSON,
    LENGTH_PRIOR_V3_CSV,
    LENGTH_PRIOR_V3_HOLDOUT_JSON,
    LENGTH_PRIOR_V3_TRAIN_CSV,
    LENGTH_PRIOR_V3_TRAIN_JSON,
    PHASE_EVALUATION_JSON,
    PHASE_EVALUATION_MD,
    TIKI_BT_UNITS_PARQUET,
    TIKI_LENGTH_PRIOR_V3_STATS_JSON,
    TIKI_UNITS_LENGTH_V3_PARQUET,
    resolve_path,
)
from cado_core.length_prior_v2 import (
    LengthPriorV2Config,
    build_key_alignment_map,
    evaluate_length_prior_v2,
    extract_human_value_pairs,
)
from cado_core.length_prior_v3 import (
    LengthPriorV3Config,
    apply_length_prior_v3,
    evaluate_length_prior_v3,
    fit_length_prior_v3,
)


def _write_prior(
    prior,
    pairs_result,
    alignment_stats: dict[str, Any],
    path: Path,
    *,
    bilingual_path: Path,
) -> None:
    payload = dict(prior.stats)
    payload["bilingual_input"] = str(bilingual_path)
    payload["split_audit"] = pairs_result.stats
    payload["alignment"] = alignment_stats
    payload["groups"] = prior.table.to_dict("records")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _alignment_table(pair_stats: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (key_zh, key_vi), stats in pair_stats.items():
        rows.append({
            "key_zh_norm": key_zh,
            "key_vi_norm": key_vi,
            **stats,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "key_zh_norm", "key_vi_norm", "freq", "zh_total", "share",
            "top_vi", "top_share", "mapped",
        ])
    return pd.DataFrame(rows).sort_values(
        ["mapped", "freq"], ascending=[False, False]
    ).reset_index(drop=True)


def _stratified_sample(
    audit: pd.DataFrame,
    *,
    n: int,
    seed: int,
) -> pd.DataFrame:
    if audit.empty or len(audit) <= n:
        return audit.copy()
    audit = audit.copy()
    audit["_stratum"] = (
        audit["alignment_confidence"].astype(str)
        + "|"
        + audit["split_status"].astype(str)
    )
    allocations: list[tuple[pd.DataFrame, int]] = []
    total = len(audit)
    for _, group in audit.groupby("_stratum", dropna=False):
        allocation = max(1, int(round(n * len(group) / total)))
        allocations.append((group, min(allocation, len(group))))
    # Điều chỉnh nếu tổng allocation vượt/thiếu n.
    while sum(k for _, k in allocations) > n:
        largest_index = max(range(len(allocations)), key=lambda i: allocations[i][1])
        group, k = allocations[largest_index]
        if k <= 1:
            break
        allocations[largest_index] = (group, k - 1)
    sampled = [
        group.sample(n=k, random_state=seed + index)
        for index, (group, k) in enumerate(allocations)
        if k > 0
    ]
    sample = pd.concat(sampled, ignore_index=False)
    if len(sample) < n:
        remaining = audit.drop(index=sample.index, errors="ignore")
        if not remaining.empty:
            sample = pd.concat([
                sample,
                remaining.sample(
                    n=min(n - len(sample), len(remaining)),
                    random_state=seed + 999,
                ),
            ])
    sample = sample.drop(columns="_stratum", errors="ignore")
    return sample.head(n).reset_index(drop=True)


def _alignment_summary(
    audit: pd.DataFrame,
    *,
    key_map_stats: dict[str, Any],
    sample_rows: int,
) -> dict[str, Any]:
    method_counts = {
        str(k): int(v)
        for k, v in audit["alignment_method"].value_counts().items()
    } if not audit.empty else {}
    confidence_counts = {
        str(k): int(v)
        for k, v in audit["alignment_confidence"].value_counts().items()
    } if not audit.empty else {}
    status_counts = {
        str(k): int(v)
        for k, v in audit["split_status"].value_counts().items()
    } if not audit.empty else {}
    return {
        "key_map": key_map_stats,
        "alignment_method_counts": method_counts,
        "alignment_confidence_counts": confidence_counts,
        "split_status_counts": status_counts,
        "fallback_count": int(method_counts.get("key_fallback", 0)),
        "low_confidence_count": int(confidence_counts.get("low", 0)),
        "audit_sample_rows": int(sample_rows),
    }


def _write_phase_evaluation(
    *,
    phase_a: dict[str, Any],
    phase_b: dict[str, Any],
    phase_c: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    payload = {
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Đánh giá Phase A / B / C",
        "",
        "## Phase A — value-only + class + CJK + split-safe",
        "",
        f"- Value-only pairs: {phase_a.get('value_pairs', 0):,}",
        f"- Title pairs: {phase_a.get('title_pairs', 0):,}",
        f"- V2 holdout dev coverage/MAE: "
        f"{phase_a.get('holdout', {}).get('dev', {}).get('overall', {}).get('coverage_value_p05_p95', 0):.4f} / "
        f"{phase_a.get('holdout', {}).get('dev', {}).get('overall', {}).get('mae_value_chars', 0):.4f}",
        f"- V2 holdout test coverage/MAE: "
        f"{phase_a.get('holdout', {}).get('test', {}).get('overall', {}).get('coverage_value_p05_p95', 0):.4f} / "
        f"{phase_a.get('holdout', {}).get('test', {}).get('overall', {}).get('mae_value_chars', 0):.4f}",
        "",
        "## Phase B — alignment audit + key fallback",
        "",
        f"- Key map entries: {phase_b.get('key_map', {}).get('n_mapped_keys', 0):,}",
        f"- Key fallback groups: {phase_b.get('fallback_count', 0):,}",
        f"- Low-confidence groups: {phase_b.get('low_confidence_count', 0):,}",
        f"- Alignment methods: {phase_b.get('alignment_method_counts', {})}",
        f"- Audit sample status: {phase_b.get('audit_status_counts', {})}",
        "- Lưu ý: low-confidence chủ yếu là key long-tail; cần review sample "
        "trước khi dùng các cặp này cho training.",
        "",
        "## Phase C — hierarchical title smoothing",
        "",
        f"- Smoothing k: {phase_c.get('title_smoothing', {}).get('k', '')}",
        f"- V3 holdout dev coverage/MAE: "
        f"{phase_c.get('holdout', {}).get('dev', {}).get('overall', {}).get('coverage_value_p05_p95', 0):.4f} / "
        f"{phase_c.get('holdout', {}).get('dev', {}).get('overall', {}).get('mae_value_chars', 0):.4f}",
        f"- V3 holdout test coverage/MAE: "
        f"{phase_c.get('holdout', {}).get('test', {}).get('overall', {}).get('coverage_value_p05_p95', 0):.4f} / "
        f"{phase_c.get('holdout', {}).get('test', {}).get('overall', {}).get('mae_value_chars', 0):.4f}",
        f"- Title V2 vs V3 (test MAE): "
        f"{phase_c.get('title_v2_v3', {}).get('v2_test_mae', 0):.4f} → "
        f"{phase_c.get('title_v2_v3', {}).get('v3_test_mae', 0):.4f}",
        f"- Title V2 vs V3 (test coverage): "
        f"{phase_c.get('title_v2_v3', {}).get('v2_test_coverage', 0):.4f} → "
        f"{phase_c.get('title_v2_v3', {}).get('v3_test_coverage', 0):.4f}",
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_from_args(args: argparse.Namespace) -> None:
    bilingual_path = resolve_path(args.bilingual, CADO_PARQUET)
    tiki_units_path = resolve_path(args.tiki_units, TIKI_BT_UNITS_PARQUET)
    prior_train_path = resolve_path(args.prior_train_output, LENGTH_PRIOR_V3_TRAIN_JSON)
    prior_all_path = resolve_path(args.prior_all_output, LENGTH_PRIOR_V3_ALL_JSON)
    holdout_path = resolve_path(args.holdout_output, LENGTH_PRIOR_V3_HOLDOUT_JSON)
    prior_csv_path = resolve_path(args.prior_csv, LENGTH_PRIOR_V3_CSV)
    prior_train_csv_path = resolve_path(args.prior_train_csv, LENGTH_PRIOR_V3_TRAIN_CSV)
    alignment_map_path = resolve_path(args.alignment_map_output, LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV)
    alignment_sample_path = resolve_path(args.alignment_sample_output, LENGTH_PRIOR_V3_ALIGNMENT_SAMPLE_CSV)
    alignment_summary_path = resolve_path(args.alignment_summary_output, LENGTH_PRIOR_V3_ALIGNMENT_SUMMARY_CSV)
    phase_eval_json_path = resolve_path(args.phase_eval_output, PHASE_EVALUATION_JSON)
    phase_eval_md_path = resolve_path(args.phase_eval_md_output, PHASE_EVALUATION_MD)
    units_output_path = resolve_path(args.units_output, TIKI_UNITS_LENGTH_V3_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_LENGTH_PRIOR_V3_STATS_JSON)

    print("=" * 72)
    print("📏 PHASE B/C — ALIGNMENT FALLBACK + TITLE HIERARCHICAL SMOOTHING")
    print("=" * 72)
    print(f"📂 1688 human    : {bilingual_path}")
    print(f"📂 Tiki units    : {tiki_units_path}")
    print(f"📂 Prior train   : {prior_train_path}")
    print(f"📂 Prior all     : {prior_all_path}")
    print(f"📂 Alignment map : {alignment_map_path}")
    print(f"📂 Units v3      : {units_output_path}")
    print(f"📂 Phase eval    : {phase_eval_json_path}")

    bilingual = pd.read_parquet(bilingual_path)
    if args.fit_limit:
        bilingual = bilingual.head(args.fit_limit).copy()
        print(f"⚠️  Giới hạn fit {args.fit_limit:,} dòng 1688")

    base_v2_config = LengthPriorV2Config(
        min_group_samples=args.min_group_samples,
        min_category_samples=args.min_category_samples,
        min_value_class_samples=args.min_value_class_samples,
        min_value_class_category_samples=args.min_value_class_category_samples,
        split_seed=args.split_seed,
        fit_split="all",
    )
    progress = None if args.quiet else print

    base = extract_human_value_pairs(
        bilingual,
        config=base_v2_config,
        progress=progress,
    )
    key_map, pair_stats, key_map_stats = build_key_alignment_map(
        base.split_audit,
        min_freq=args.alignment_min_freq,
        min_share=args.alignment_min_share,
    )
    print(
        f"\n✅ Key map: {key_map_stats['n_mapped_keys']:,}/"
        f"{key_map_stats['n_zh_keys']:,} ZH keys"
    )

    aligned = extract_human_value_pairs(
        bilingual,
        config=base_v2_config,
        key_map=key_map,
        key_pair_stats=pair_stats,
        min_key_pair_freq=args.alignment_min_freq,
        align_key_fallback=not args.no_key_fallback,
        progress=progress,
    )
    alignment_summary = _alignment_summary(
        aligned.split_audit,
        key_map_stats=key_map_stats,
        sample_rows=args.alignment_sample_size,
    )
    alignment_table = _alignment_table(pair_stats)
    alignment_map_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_table.to_csv(alignment_map_path, index=False, encoding="utf-8-sig")

    sample = _stratified_sample(
        aligned.split_audit,
        n=args.alignment_sample_size,
        seed=args.split_seed,
    )
    sample = sample.copy()
    if not sample.empty:
        def _audit_status(row: pd.Series) -> str:
            if row.get("split_status") == "mismatch":
                return "review_split_mismatch"
            if row.get("alignment_method") == "key_fallback":
                return "review_fallback"
            if row.get("alignment_confidence") == "low":
                return "review_low_confidence"
            return "pass_auto"

        sample["audit_status"] = sample.apply(_audit_status, axis=1)
    alignment_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(alignment_sample_path, index=False, encoding="utf-8-sig")
    alignment_summary["audit_sample_rows"] = int(len(sample))
    alignment_summary["audit_status_counts"] = (
        {
            str(k): int(v)
            for k, v in sample["audit_status"].value_counts().items()
        }
        if "audit_status" in sample.columns
        else {}
    )
    summary_frame = pd.DataFrame([
        {"metric": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in alignment_summary.items()
        if key not in {"key_map"}
    ] + [
        {"metric": f"key_map.{key}", "value": json.dumps(value, ensure_ascii=False)}
        for key, value in key_map_stats.items()
    ])
    alignment_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(alignment_summary_path, index=False, encoding="utf-8-sig")

    print(f"   Alignment methods: {alignment_summary['alignment_method_counts']}")
    print(f"   Alignment confidence: {alignment_summary['alignment_confidence_counts']}")
    print(f"   Sample rows: {len(sample):,}")

    train_v2_config = LengthPriorV2Config(
        **{
            **base_v2_config.__dict__,
            "fit_split": "train",
        }
    )
    all_v2_config = LengthPriorV2Config(
        **{
            **base_v2_config.__dict__,
            "fit_split": "all",
        }
    )
    train_config = LengthPriorV3Config(
        v2=train_v2_config,
        title_smoothing_k=args.title_smoothing_k,
        title_interval_inflation=args.title_interval_inflation,
    )
    all_config = LengthPriorV3Config(
        v2=all_v2_config,
        title_smoothing_k=args.title_smoothing_k,
        title_interval_inflation=args.title_interval_inflation,
    )
    train_prior = fit_length_prior_v3(
        aligned.pairs,
        config=train_config,
        progress=progress,
    )
    all_prior = fit_length_prior_v3(
        aligned.pairs,
        config=all_config,
        progress=progress,
    )

    _write_prior(
        train_prior,
        aligned,
        alignment_summary,
        prior_train_path,
        bilingual_path=bilingual_path,
    )
    _write_prior(
        all_prior,
        aligned,
        alignment_summary,
        prior_all_path,
        bilingual_path=bilingual_path,
    )
    prior_train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_prior.table.to_csv(
        prior_train_csv_path, index=False, encoding="utf-8-sig"
    )
    prior_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_prior.table.to_csv(prior_csv_path, index=False, encoding="utf-8-sig")

    holdout = {
        "fit_split": "train",
        "v2": {},
        "v3": {},
    }
    for split in ("dev", "test"):
        holdout["v2"][split] = evaluate_length_prior_v2(
            aligned.pairs,
            train_prior.v2_prior,
            split=split,
            config=train_v2_config,
        )
        holdout["v3"][split] = evaluate_length_prior_v3(
            aligned.pairs,
            train_prior,
            split=split,
        )
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with holdout_path.open("w", encoding="utf-8") as handle:
        json.dump(holdout, handle, ensure_ascii=False, indent=2)

    units = pd.read_parquet(tiki_units_path)
    if args.limit:
        units = units.head(args.limit).copy()
        print(f"⚠️  Giới hạn apply {args.limit:,} Tiki unit")
    applied = apply_length_prior_v3(
        units,
        all_prior,
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
        "alignment_summary": str(alignment_summary_path),
        "alignment_sample": str(alignment_sample_path),
        "phase_evaluation": str(phase_eval_json_path),
    })
    units_output_path.parent.mkdir(parents=True, exist_ok=True)
    applied.units.to_parquet(
        units_output_path, index=False, compression="snappy"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(apply_stats, handle, ensure_ascii=False, indent=2)

    v2_test = holdout["v2"]["test"]
    v3_test = holdout["v3"]["test"]
    title_v2_v3 = {
        "v2_test_mae": v2_test.get("by_grain", {}).get("title", {}).get(
            "mae_value_chars", 0
        ),
        "v3_test_mae": v3_test.get("by_grain", {}).get("title", {}).get(
            "mae_value_chars", 0
        ),
        "v2_test_coverage": v2_test.get("by_grain", {}).get("title", {}).get(
            "coverage_value_p05_p95", 0
        ),
        "v3_test_coverage": v3_test.get("by_grain", {}).get("title", {}).get(
            "coverage_value_p05_p95", 0
        ),
    }
    phase_a = {
        "value_pairs": int(aligned.stats["n_value_pairs"]),
        "title_pairs": int(aligned.stats["n_title_pairs"]),
        "group_pairs": int(aligned.stats["n_group_pairs"]),
        "split_status_counts": aligned.stats["split_status_counts"],
        "holdout": holdout["v2"],
    }
    phase_b = {
        **alignment_summary,
        "alignment_map_csv": str(alignment_map_path),
        "alignment_sample_csv": str(alignment_sample_path),
        "alignment_summary_csv": str(alignment_summary_path),
    }
    phase_c = {
        "title_smoothing": train_prior.stats["title_smoothing"],
        "holdout": holdout["v3"],
        "title_v2_v3": title_v2_v3,
    }
    _write_phase_evaluation(
        phase_a=phase_a,
        phase_b=phase_b,
        phase_c=phase_c,
        json_path=phase_eval_json_path,
        md_path=phase_eval_md_path,
    )

    print("\n" + "=" * 72)
    print("✅ PHASE B/C HOÀN TẤT")
    print("=" * 72)
    print(f"Alignment fallback : {alignment_summary['fallback_count']:,}")
    print(f"Low confidence     : {alignment_summary['low_confidence_count']:,}")
    print(f"Train prior pairs  : {train_prior.stats['n_pairs']:,}")
    print(f"All prior pairs    : {all_prior.stats['n_pairs']:,}")
    for split in ("dev", "test"):
        v3 = holdout["v3"][split].get("overall", {})
        print(
            f"Holdout {split:<4}     : n={holdout['v3'][split].get('n', 0):,} "
            f"coverage={v3.get('coverage_value_p05_p95', 0):.4f} "
            f"MAE={v3.get('mae_value_chars', 0):.4f}"
        )
    print(
        f"Title test MAE     : v2={title_v2_v3['v2_test_mae']:.4f} "
        f"→ v3={title_v2_v3['v3_test_mae']:.4f}"
    )
    print(f"Tiki units         : {apply_stats['n_units']:,}")
    print(f"Title smoothed     : {apply_stats['smoothing_counts']}")
    print(f"Expected value p50 : {apply_stats['expected_zh_value_chars']['p50']}")
    print(f"\n💾 {prior_train_path}")
    print(f"💾 {prior_all_path}")
    print(f"💾 {holdout_path}")
    print(f"💾 {alignment_map_path}")
    print(f"💾 {alignment_sample_path}")
    print(f"💾 {alignment_summary_path}")
    print(f"💾 {phase_eval_json_path}")
    print(f"💾 {phase_eval_md_path}")
    print(f"💾 {units_output_path}")
    print(f"💾 {stats_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bilingual", default=str(CADO_PARQUET))
    parser.add_argument("--tiki-units", default=str(TIKI_BT_UNITS_PARQUET))
    parser.add_argument("--prior-train-output", default=str(LENGTH_PRIOR_V3_TRAIN_JSON))
    parser.add_argument("--prior-all-output", default=str(LENGTH_PRIOR_V3_ALL_JSON))
    parser.add_argument("--holdout-output", default=str(LENGTH_PRIOR_V3_HOLDOUT_JSON))
    parser.add_argument("--prior-csv", default=str(LENGTH_PRIOR_V3_CSV))
    parser.add_argument("--prior-train-csv", default=str(LENGTH_PRIOR_V3_TRAIN_CSV))
    parser.add_argument("--alignment-map-output", default=str(LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV))
    parser.add_argument("--alignment-sample-output", default=str(LENGTH_PRIOR_V3_ALIGNMENT_SAMPLE_CSV))
    parser.add_argument("--alignment-summary-output", default=str(LENGTH_PRIOR_V3_ALIGNMENT_SUMMARY_CSV))
    parser.add_argument("--phase-eval-output", default=str(PHASE_EVALUATION_JSON))
    parser.add_argument("--phase-eval-md-output", default=str(PHASE_EVALUATION_MD))
    parser.add_argument("--units-output", default=str(TIKI_UNITS_LENGTH_V3_PARQUET))
    parser.add_argument("--stats-output", default=str(TIKI_LENGTH_PRIOR_V3_STATS_JSON))
    parser.add_argument("--min-group-samples", type=int, default=20)
    parser.add_argument("--min-category-samples", type=int, default=100)
    parser.add_argument("--min-value-class-samples", type=int, default=50)
    parser.add_argument("--min-value-class-category-samples", type=int, default=100)
    parser.add_argument("--alignment-min-freq", type=int, default=20)
    parser.add_argument("--alignment-min-share", type=float, default=0.60)
    parser.add_argument("--alignment-sample-size", type=int, default=500)
    parser.add_argument("--title-smoothing-k", type=int, default=20)
    parser.add_argument("--title-interval-inflation", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fit-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-key-fallback", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
