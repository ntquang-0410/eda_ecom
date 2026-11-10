#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ghép 1.300 description 8B còn thiếu vào raw 8B, rồi reprocess.

Input:
  --new   predictions.parquet của kernel 8B full
  --raw   data/bt/raw/description/qwen3-8b/predictions.parquet
  --out-dir  thư mục tạm chứa qwen3-8b/predictions.parquet cho process_bt_desc
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--raw", default="data/bt/raw/description/qwen3-8b/predictions.parquet")
    ap.add_argument("--out-dir", default=".artifacts/bt8b_full")
    args = ap.parse_args()

    new = pd.read_parquet(args.new)
    raw = pd.read_parquet(args.raw)
    new = new[new["target_text_bt"].fillna("").str.strip().ne("")].copy()
    new["attempt"] = 2
    new["prompt_version"] = new["prompt_version"].fillna("plan_b_desc_v1")
    new["retry_method"] = new.get("retry_method", "full_run")
    for col in raw.columns:
        if col not in new.columns:
            new[col] = None
    new = new[raw.columns]

    before = len(raw)
    merged = pd.concat([raw, new], ignore_index=True)
    merged = merged.drop_duplicates("pair_id", keep="first").reset_index(drop=True)
    print(json.dumps({
        "raw_before": before, "new_rows": len(new), "merged": len(merged),
        "added": len(merged) - before,
        "remaining_products_missing": int(
            13200 - merged["product_id"].nunique()),
    }, ensure_ascii=False))

    Path(args.raw).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.raw, index=False)
    out = Path(args.out_dir) / "qwen3-8b"
    out.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out / "predictions.parquet", index=False)
    print("wrote", args.raw, "and", out / "predictions.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
