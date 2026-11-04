#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tạo file JSONL để chấm bằng Gemma trên AI Studio.

Input parquet cần cột: id, source, hypothesis; tùy chọn reference.
Xuất:
  - judge_input.jsonl  (mỗi dòng 1 object)
  - judge_prompt.txt   (prompt đã điền sẵn hướng dẫn, chỗ data là {{DATA}})
  - các batch judge_input_batchNN.jsonl (25 dòng/batch)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROMPT = """You are a strict bilingual evaluator for Vietnamese-to-Simplified-Chinese e-commerce translation.

For each item, score the hypothesis on three axes (1-5):
- adequacy: does it preserve the meaning of the source?
- fluency: is the Chinese natural and grammatical?
- terminology: are brand names, numbers, units and product terms correct?

Rules:
- Use the reference when provided; otherwise judge against the source only.
- Penalize untranslated Vietnamese, missing numbers/units, and hallucinated content.
- Do not reward literal word-for-word copying.

Return ONLY a JSON array, one object per input, with this schema:
{"id": "...", "adequacy": 1-5, "fluency": 1-5, "terminology": 1-5, "note": "<=20 words"}

Data:
{{DATA}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="parquet có id/source/hypothesis[/reference]")
    ap.add_argument("--out-dir", default="data/cleaned/judge")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)
    df = df.reset_index(drop=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for row in df.itertuples():
        obj = {"id": str(row.id), "source": str(row.source), "hypothesis": str(row.hypothesis)}
        if hasattr(row, "reference") and str(getattr(row, "reference") or "").strip():
            obj["reference"] = str(row.reference)
        records.append(obj)

    (out / "judge_input.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    (out / "judge_prompt.txt").write_text(PROMPT, encoding="utf-8")

    for index in range(0, len(records), args.batch_size):
        part = records[index:index + args.batch_size]
        (out / f"judge_input_batch{index // args.batch_size + 1:02d}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part), encoding="utf-8")

    print(f"wrote {len(records)} dòng, {len(records)//args.batch_size + 1} batch -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
