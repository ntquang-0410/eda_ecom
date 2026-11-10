#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp điểm Gemma judge xuất từ Google AI Studio.

Input: một hoặc nhiều file JSON/JSONL, mỗi object:
  {"id": "...", "adequacy": 1-5, "fluency": 1-5, "terminology": 1-5, "note": "..."}

Output: JSON tổng hợp + CSV theo id, kèm trung bình theo dataset/pair_type.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in "[]":
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


def dataset_of(identifier: str) -> str:
    value = str(identifier)
    if value.startswith("1688"):
        return "1688"
    if value.startswith("tiki"):
        return "tiki"
    return "unknown"


def pair_type_of(identifier: str) -> str:
    value = str(identifier)
    if ":title" in value or value.endswith(":title"):
        return "title"
    if ":attr" in value:
        return "attribute"
    if ":desc" in value:
        return "description"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/cleaned/judge/gemma_scores.json")
    args = ap.parse_args()

    records = []
    for pattern in args.inputs:
        for path in sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(pattern)]:
            if path.exists():
                got = load_records(path)
                print(path, len(got))
                records.extend(got)

    rows = []
    for rec in records:
        try:
            rows.append({
                "id": str(rec.get("id")),
                "adequacy": float(rec.get("adequacy")),
                "fluency": float(rec.get("fluency")),
                "terminology": float(rec.get("terminology")),
                "note": str(rec.get("note") or ""),
            })
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(rows).drop_duplicates("id")
    df["dataset"] = df["id"].map(dataset_of)
    df["pair_type"] = df["id"].map(pair_type_of)
    df["mean"] = df[["adequacy", "fluency", "terminology"]].mean(axis=1)

    summary = {
        "n": int(len(df)),
        "overall": {k: round(float(df[k].mean()), 3) for k in ("adequacy", "fluency", "terminology", "mean")},
        "by_dataset": {
            name: {
                "n": int(len(part)),
                **{k: round(float(part[k].mean()), 3) for k in ("adequacy", "fluency", "terminology", "mean")},
            }
            for name, part in df.groupby("dataset")
        },
        "by_pair_type": {
            name: {
                "n": int(len(part)),
                **{k: round(float(part[k].mean()), 3) for k in ("adequacy", "fluency", "terminology", "mean")},
            }
            for name, part in df.groupby("pair_type")
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": df.to_dict("records")},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out, "and", out.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
