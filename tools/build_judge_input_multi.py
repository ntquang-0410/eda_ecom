#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tạo judge input cho nhiều cấu hình weight.

Đọc các file hyp_<config>_1688.parquet / hyp_<config>_tiki.parquet,
sample N dòng mỗi tập, gắn id dạng "<config>::<id gốc>".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CONFIGS = ("natural", "balanced", "no_desc", "tiki_heavy", "title_heavy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="/tmp/opencode/weight_eval")
    ap.add_argument("--out-dir", default="data/cleaned/judge")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hyp = Path(args.hyp_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for config in CONFIGS:
        for dataset in ("1688", "tiki"):
            path = hyp / f"hyp_{config}_{dataset}.parquet"
            if not path.exists():
                print("skip", path)
                continue
            df = pd.read_parquet(path)
            if args.n and args.n < len(df):
                df = df.sample(n=args.n, random_state=args.seed)
            for row in df.itertuples():
                obj = {
                    "id": f"{config}::{row.id}",
                    "config": config,
                    "dataset": dataset,
                    "source": str(row.source),
                    "hypothesis": str(row.hypothesis),
                }
                if hasattr(row, "reference") and str(getattr(row, "reference") or "").strip():
                    obj["reference"] = str(row.reference)
                records.append(obj)
            print(config, dataset, len(df))

    (out / "judge_input.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    for index in range(0, len(records), args.batch_size):
        part = records[index:index + args.batch_size]
        (out / f"judge_input_batch{index // args.batch_size + 1:02d}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part), encoding="utf-8")

    print(f"wrote {len(records)} dòng, {len(records)//args.batch_size + 1} batch -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
