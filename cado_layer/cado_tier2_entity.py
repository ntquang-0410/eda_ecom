#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tính 7 entity signals (S1–S7) cho các cặp bitext.

Sửa các bug của bản cũ:
  * ``np`` được import ở module level (trước đây NameError khi bật embedding).
  * Không ghi đè file input; output mặc định là ``bitext_entity_scored.parquet``.
  * S4b encode bigram theo batch thay vì gọi model cho từng dòng.

    python cado_layer/cado_tier2_entity.py
    python cado_layer/cado_tier2_entity.py --embedding --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.bitext import BitextConfig, compute_entity_signals
from cado_core.config import (
    BITEXT_READY_PARQUET,
    CADO_PARQUET,
    DESC_ZH_CLEANED,
    TITLE_VI_CLEANED,
    TITLE_ZH_CLEANED,
    resolve_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(BITEXT_READY_PARQUET if BITEXT_READY_PARQUET.exists() else CADO_PARQUET),
        help="Parquet đầu vào (mặc định bitext_ready nếu có)",
    )
    parser.add_argument(
        "--output",
        default="data/cleaned/bitext_entity_scored.parquet",
        help="Parquet đầu ra",
    )
    parser.add_argument("--embedding", action="store_true",
                        help="Bật S4b bằng MiniLM (chậm, cần tải model)")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    input_path = resolve_path(args.input, CADO_PARQUET)
    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()

    # Đảm bảo có cột cleaned để tính signal.
    if TITLE_ZH_CLEANED not in df.columns:
        df[TITLE_ZH_CLEANED] = df["title_zh"]
    if TITLE_VI_CLEANED not in df.columns:
        df[TITLE_VI_CLEANED] = df["title_vi"]
    if DESC_ZH_CLEANED not in df.columns:
        df[DESC_ZH_CLEANED] = df["description_zh"]

    print(f"📂 {input_path} — {len(df):,} dòng")
    result = compute_entity_signals(
        df,
        BitextConfig(progress=print),
        use_embedding=args.embedding,
    )

    output_path = resolve_path(args.output, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\n💾 {output_path} ({len(result):,} dòng, {len(result.columns)} cột)")

    if "entity_score" in result.columns:
        print("\n📊 entity_score:")
        print(result["entity_score"].describe().to_string())
        if "tier2_label" in result.columns:
            print("\ntier2_label:")
            print(result["tier2_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
