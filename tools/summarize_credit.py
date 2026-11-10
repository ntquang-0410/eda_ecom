#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp credit score từ các batch đã chấm (kể cả khi chưa chấm hết).

Dùng khi judge bị dừng giữa chừng do quota: đọc toàn bộ
`batches_credit/batch_*.json`, map với judge_map.jsonl, xuất summary
cùng format `gemma_scores.json` để aggregate_weight_ab.py đọc được.

Dùng:
  python tools/summarize_credit.py \
    --batches-dir .artifacts/judge_B_flash/batches_credit \
    --map .artifacts/judge_B_flash/judge_map.jsonl \
    --model gemini-3.8-flash \
    --out .artifacts/judge_B_flash/gemma_scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_judge_credit import summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches-dir", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    for path in sorted(Path(args.batches_dir).glob("batch_*.json")):
        try:
            records.extend(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if not records:
        raise SystemExit("không có batch nào")

    mapping = {}
    for line in Path(args.map).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            mapping[str(item["judge_id"])] = item

    summary = summarize(records, mapping)
    result = {"model": args.model, "scale": "credit_AAA_C", "partial": True,
              "summary": summary}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("n =", summary["overall"]["n"], "->", args.out)
    for key, value in summary["by_config"].items():
        print(f"  {key:14s} n={value['n']:4d} mean={value['mean']:.2f} ({value['mean_rating']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
