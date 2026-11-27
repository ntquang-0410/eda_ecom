#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build sample_{step,row_index,...} cho NLLB từ một units file bất kỳ.

Dùng đúng balance_plan (budget 4M, 500 step). Với B subsample, row_index trỏ vào
file subsample — phải dùng đúng file này khi train.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quota_sampling import balance_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cleaned/nllb_samples"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--config", default="balanced", choices=("balanced", "natural"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 2024, 2025])
    ap.add_argument("--budget", type=int, default=4_000_000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    u = pd.read_parquet(args.units).reset_index(drop=True)
    groups = sorted(u.group.astype(str).unique())
    cfg = None if args.config == "natural" else {g: 0.25 for g in groups}
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        plan, actual = balance_plan(u, cfg, seed, args.budget, args.steps)
        rows = []
        for step, idxs in enumerate(plan):
            for i in idxs:
                rows.append({"step": step, "row_index": int(i), "pair_id": u.pair_id.iloc[i],
                             "group": u.group.iloc[i], "eff_tokens": float(u.eff_tokens.iloc[i])})
        df = pd.DataFrame(rows)
        p = OUT / f"sample_{args.tag}_{args.config}_seed{seed}.parquet"
        df.to_parquet(p, index=False)
        print(f"seed {seed}: {len(df)} draws, {df.step.nunique()} step, "
              f"{df.eff_tokens.sum()/1e6:.3f}M token, unique rows {df.row_index.nunique()} -> {p.name}")


if __name__ == "__main__":
    main()
