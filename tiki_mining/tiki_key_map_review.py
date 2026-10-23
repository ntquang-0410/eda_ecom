#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review key map bằng Qwen qua Alibaba Cloud DashScope.

Ví dụ::

    export DASHSCOPE_API_KEY=...
    python tiki_mining/tiki_key_map_review.py --model qwen3.8-max --votes 3

    # Chỉ dựng prompt, chưa gọi API
    python tiki_mining/tiki_key_map_review.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.config import (
    KEY_MAP_FINAL_CSV,
    KEY_MAP_QWEN_REVIEW_CSV,
    KEY_MAP_QWEN_STATS_JSON,
    KEY_MAP_QWEN_VOTES_CSV,
    KEY_MAP_REVIEW_INPUT_CSV,
    KEY_MAP_SILVER_EVAL_CSV,
    LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV,
    TIKI_BT_UNITS_PLAN_A_PARQUET,
    TIKI_KEY_MAP_CSV,
    resolve_path,
)
from cado_core.key_map_review import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    QwenReviewer,
    aggregate_reviews,
    apply_reviews,
    build_review_input,
    build_silver_input,
    evaluate_silver,
    run_reviews,
    write_review_artifacts,
)


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Nạp .env đơn giản nếu có; không ghi đè env đã set."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key(args: argparse.Namespace) -> str:
    names = [args.api_key_env, "DASHSCOPE_API_KEY", "ALIBABA_CLOUD_API_KEY"]
    for name in names:
        if name and os.getenv(name):
            return str(os.environ[name])
    return ""


def run_from_args(args: argparse.Namespace) -> None:
    _load_dotenv()
    model = args.model or os.getenv("QWEN_REVIEW_MODEL", DEFAULT_MODEL)
    base_url = args.base_url or os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    review_path = resolve_path(args.review, KEY_MAP_REVIEW_INPUT_CSV.parent / "tiki_key_map_review.csv")
    key_map_path = resolve_path(args.key_map, TIKI_KEY_MAP_CSV)
    units_path = resolve_path(args.units, TIKI_BT_UNITS_PLAN_A_PARQUET)
    alignment_path = resolve_path(
        args.alignment_map, LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV
    )
    input_output = resolve_path(args.input_output, KEY_MAP_REVIEW_INPUT_CSV)
    silver_output = resolve_path(args.silver_output, KEY_MAP_SILVER_EVAL_CSV)
    votes_output = resolve_path(args.votes_output, KEY_MAP_QWEN_VOTES_CSV)
    review_output = resolve_path(args.review_output, KEY_MAP_QWEN_REVIEW_CSV)
    final_output = resolve_path(args.final_output, KEY_MAP_FINAL_CSV)
    stats_output = resolve_path(args.stats_output, KEY_MAP_QWEN_STATS_JSON)

    print("=" * 72)
    print("🧠 QWEN KEY-MAP REVIEW — Alibaba Cloud DashScope")
    print("=" * 72)
    print(f"📂 Review input : {review_path}")
    print(f"📂 Key map      : {key_map_path}")
    print(f"📂 Units         : {units_path}")
    print(f"📂 Model         : {model}")
    print(f"📂 Base URL      : {base_url}")
    print(f"📂 Output        : {review_output}")

    review_frame = pd.read_csv(review_path, dtype=str, encoding="utf-8-sig").fillna("")
    key_map = pd.read_csv(key_map_path, dtype=str, encoding="utf-8-sig").fillna("")
    units = pd.read_parquet(units_path)
    if args.limit:
        review_frame = review_frame.head(args.limit)
        print(f"⚠️  Giới hạn {args.limit:,} key")

    review_input = build_review_input(
        units,
        review_frame,
        alignment_map_path=alignment_path,
    )
    silver = (
        build_silver_input(
            key_map,
            units,
            n=args.silver_size,
            seed=args.seed,
            alignment_map_path=alignment_path,
        )
        if args.silver_size > 0 and not args.skip_silver
        else pd.DataFrame()
    )
    print(f"   Review keys : {len(review_input):,}")
    print(f"   Silver keys : {len(silver):,}")

    if args.dry_run:
        empty_votes = pd.DataFrame()
        empty_review = review_input.assign(
            qwen_key_zh="", qwen_status="qwen_low", qwen_votes=0,
            qwen_total_votes=0, qwen_agreement=0.0, qwen_confidence=0.0,
            qwen_alternatives="[]",
        )
        paths = write_review_artifacts(
            review_input=review_input,
            silver=silver,
            votes=empty_votes,
            review=empty_review,
            key_map=key_map,
            final_map=key_map,
            stats={"dry_run": True, "n_review_keys": len(review_input)},
            input_path=input_output,
            silver_path=silver_output,
            votes_path=votes_output,
            review_path=review_output,
            final_path=final_output,
            stats_path=stats_output,
            write_key_map=False,
        )
        print("\n✅ DRY RUN: đã tạo prompt input, chưa gọi API.")
        for label, path in paths.items():
            print(f"💾 {label:<8} {path}")
        return

    api_key = _api_key(args)
    if not api_key:
        raise SystemExit(
            "Chưa có API key. Hãy export DASHSCOPE_API_KEY hoặc "
            "ALIBABA_CLOUD_API_KEY trong shell/Colab, rồi chạy lại."
        )

    reviewer = QwenReviewer(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=args.timeout,
    )
    silver_metrics: dict = {"n": 0}
    if not silver.empty:
        silver_votes = run_reviews(
            reviewer,
            silver,
            votes=args.silver_votes,
            workers=args.workers,
        )
        silver_metrics = evaluate_silver(silver, silver_votes)
        silver.to_csv(silver_output, index=False, encoding="utf-8-sig")
        silver_votes.to_csv(
            silver_output.with_name("key_map_silver_votes.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"   Silver: {json.dumps(silver_metrics, ensure_ascii=False)}")

    votes = run_reviews(
        reviewer,
        review_input,
        votes=args.votes,
        workers=args.workers,
        progress=None if args.quiet else print,
    )
    # Ghi raw votes ngay để không mất nếu bước aggregate gặp lỗi.
    votes_output.parent.mkdir(parents=True, exist_ok=True)
    votes.to_csv(votes_output, index=False, encoding="utf-8-sig")
    review_result = aggregate_reviews(
        votes,
        review_input,
        votes_expected=args.votes,
    )

    apply_ok = True
    if silver_metrics.get("n", 0) and not args.force_apply:
        if silver_metrics.get("exact", 0) < args.min_silver_accuracy:
            apply_ok = False
    if not args.no_apply and apply_ok:
        statuses = ["qwen_high"]
        if args.apply_medium:
            statuses.append("qwen_medium")
        final_map = apply_reviews(key_map, review_result, statuses=statuses)
    else:
        final_map = key_map.copy()

    stats = {
        "model": model,
        "base_url": base_url,
        "votes_per_key": int(args.votes),
        "silver_votes": int(args.silver_votes),
        "silver": silver_metrics,
        "n_review_keys": int(len(review_input)),
        "n_review_rows": int(len(review_result)),
        "n_votes": int(len(votes)),
        "status_counts": {
            str(k): int(v)
            for k, v in review_result["qwen_status"].value_counts().items()
        },
        "accepted_statuses": (
            ["qwen_high", "qwen_medium"] if args.apply_medium else ["qwen_high"]
        ),
        "apply_ok": bool(apply_ok and not args.no_apply),
        "key_map_before_mapped": int(
            (key_map.get("key_zh", pd.Series(dtype=str)).astype(str).str.strip() != "").sum()
        ),
        "key_map_after_mapped": int(
            (final_map.get("key_zh", pd.Series(dtype=str)).astype(str).str.strip() != "").sum()
        ),
    }
    paths = write_review_artifacts(
        review_input=review_input,
        silver=silver,
        votes=votes,
        review=review_result,
        key_map=key_map,
        final_map=final_map,
        stats=stats,
        input_path=input_output,
        silver_path=silver_output,
        votes_path=votes_output,
        review_path=review_output,
        final_path=final_output,
        stats_path=stats_output,
        write_key_map=bool(apply_ok and not args.no_apply),
    )
    print("\n" + "=" * 72)
    print("✅ QWEN KEY-MAP REVIEW HOÀN TẤT")
    print("=" * 72)
    print(f"Status counts : {stats['status_counts']}")
    print(f"Silver        : {silver_metrics}")
    print(f"Applied       : {stats['apply_ok']}")
    print(f"Key map       : {stats['key_map_before_mapped']} → {stats['key_map_after_mapped']} mapped")
    for label, path in paths.items():
        print(f"💾 {label:<8} {path}")
    print("\nSau khi review, chạy lại:")
    print("  python run_pipeline.py tiki-bt-prepare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        default=str(KEY_MAP_REVIEW_INPUT_CSV.parent / "tiki_key_map_review.csv"),
    )
    parser.add_argument("--key-map", default=str(TIKI_KEY_MAP_CSV))
    parser.add_argument("--units", default=str(TIKI_BT_UNITS_PLAN_A_PARQUET))
    parser.add_argument(
        "--alignment-map", default=str(LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV)
    )
    parser.add_argument("--input-output", default=str(KEY_MAP_REVIEW_INPUT_CSV))
    parser.add_argument("--silver-output", default=str(KEY_MAP_SILVER_EVAL_CSV))
    parser.add_argument("--votes-output", default=str(KEY_MAP_QWEN_VOTES_CSV))
    parser.add_argument("--review-output", default=str(KEY_MAP_QWEN_REVIEW_CSV))
    parser.add_argument("--final-output", default=str(KEY_MAP_FINAL_CSV))
    parser.add_argument("--stats-output", default=str(KEY_MAP_QWEN_STATS_JSON))
    parser.add_argument(
        "--model",
        default=None,
        help="Model DashScope, ví dụ qwen3.8-max / qwen3.7-max",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="DashScope OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Tên biến môi trường chứa API key",
    )
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--silver-votes", type=int, default=3)
    parser.add_argument("--silver-size", type=int, default=100)
    parser.add_argument("--min-silver-accuracy", type=float, default=0.80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply-medium", action="store_true")
    parser.add_argument("--no-apply", action="store_true")
    parser.add_argument("--force-apply", action="store_true")
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
