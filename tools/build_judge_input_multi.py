#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tạo judge input mù (blind) cho nhiều cấu hình weight.

- Chọn cùng một tập eval_id cho mọi config/seed (paired).
- Sinh id ngẫu nhiên j000001...; config/seed/dataset/pair_type nằm trong map riêng.
- Gemma chỉ thấy id, source, hypothesis, reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = ("natural", "balanced", "no_desc", "tiki_heavy", "title_heavy")


def load_hyp(hyp_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(hyp_dir.glob("hyp_*.parquet")):
        df = pd.read_parquet(path)
        if "dataset" not in df.columns:
            name = path.stem
            df["dataset"] = "1688" if "_1688" in name else ("tiki" if "_tiki" in name else "unknown")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"không có hyp_*.parquet trong {hyp_dir}")
    df = pd.concat(frames, ignore_index=True)
    required = {"eval_id", "source", "hypothesis", "dataset", "pair_type", "config", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"thiếu cột: {missing}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="/kaggle/working/weight_runs")
    ap.add_argument("--out-dir", default="data/cleaned/judge")
    ap.add_argument("--n", type=int, default=50, help="số eval_id mỗi stratum")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()

    hyp_dir, out = Path(args.hyp_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_hyp(hyp_dir)

    rng = np.random.default_rng(args.seed)
    selected: dict[tuple[str, str], list[str]] = {}
    for (dataset, pair_type), part in df.groupby(["dataset", "pair_type"]):
        ids = sorted(part["eval_id"].astype(str).unique())
        n = min(args.n, len(ids))
        selected[(dataset, pair_type)] = rng.choice(ids, size=n, replace=False).tolist()

    chosen = df[df.apply(lambda r: str(r["eval_id"]) in selected.get(
        (str(r["dataset"]), str(r["pair_type"])), []), axis=1)].copy()
    chosen = chosen.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    records, mapping = [], []
    for i, row in enumerate(chosen.itertuples(), start=1):
        jid = f"j{i:06d}"
        item = {"id": jid, "source": str(row.source),
                "hypothesis": str(row.hypothesis)}
        if str(getattr(row, "reference", "") or "").strip():
            item["reference"] = str(row.reference)
        records.append(item)
        mapping.append({
            "judge_id": jid, "config": str(row.config), "seed": int(row.seed),
            "dataset": str(row.dataset), "pair_type": str(row.pair_type),
            "eval_id": str(row.eval_id),
        })

    (out / "judge_input.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    (out / "judge_map.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in mapping), encoding="utf-8")

    # chia batch theo số ký tự để tránh vượt context
    batch, size = [], 0
    batch_index = 0
    for record in records:
        length = len(record.get("source", "")) + len(record.get("hypothesis", ""))
        if batch and (len(batch) >= args.batch_size or size + length > 30000):
            (out / f"judge_batch_{batch_index:03d}.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in batch), encoding="utf-8")
            batch_index += 1
            batch, size = [], 0
        batch.append(record)
        size += length
    if batch:
        (out / f"judge_batch_{batch_index:03d}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in batch), encoding="utf-8")
        batch_index += 1

    counts = chosen.groupby(["dataset", "pair_type", "config", "seed"]).size()
    print(f"wrote {len(records)} dòng, {batch_index} batch -> {out}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
