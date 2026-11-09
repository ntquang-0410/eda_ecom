#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gộp credit cũ (đã chấm) + batch mới, xuất gemma_scores.json đầy đủ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_judge_credit import summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--batches-dir", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    records = list(old.get("summary", {}).get("rows", []))
    seen = {str(r["id"]) for r in records}

    for path in sorted(Path(args.batches_dir).glob("batch_*.json")):
        try:
            for r in json.loads(path.read_text(encoding="utf-8")):
                rid = str(r.get("id"))
                if rid and rid not in seen:
                    records.append(r)
                    seen.add(rid)
        except json.JSONDecodeError:
            continue

    mapping = {}
    for line in Path(args.map).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            mapping[str(item["judge_id"])] = item

    summary = summarize(records, mapping)
    result = {"model": args.model, "scale": "credit_AAA_C", "partial": False,
              "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("n =", summary["overall"]["n"], "->", args.out)
    for k, v in summary["by_config"].items():
        print(f"  {k:14s} n={v['n']:4d} mean={v['mean']:.2f} ({v['mean_rating']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
