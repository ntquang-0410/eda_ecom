#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng cho các cell control (2×2 + seed 123 + NLLB) trên eval_F_v2_common.

Đọc scores_<label>_seed<seed>_<ds>.parquet, lọc theo eval_F_v2_common, in bảng
và paired bootstrap so với baseline chỉ định.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_matrix import boot, fmt

PAT = re.compile(r"scores_(?P<label>.+)_seed(?P<seed>\d+)_(?P<ds>1688|tiki)\.parquet")


def load(scores_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(scores_dir.glob("scores_*.parquet")):
        m = PAT.match(p.name)
        if not m:
            continue
        d = pd.read_parquet(p)
        d["eval_id"] = d["eval_id"].astype(str)
        d["label"] = m.group("label")
        d["seed"] = int(m.group("seed"))
        d["ds"] = m.group("ds")
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=".artifacts/comet_gpu/scores")
    ap.add_argument("--eval", default="data/cleaned/eval_F_v2_common.parquet")
    ap.add_argument("--baseline", default="", help="label@seed để so sánh")
    ap.add_argument("--out", default="reports/CONTROLS_2X2.md")
    args = ap.parse_args()

    ev = pd.read_parquet(args.eval)
    ev["eval_id"] = ev["eval_id"].astype(str)
    ids = set(ev.eval_id)
    df = load(Path(args.scores_dir))
    if df.empty:
        print("no scores")
        return 1
    df = df[df.eval_id.isin(ids)].merge(
        ev[["eval_id", "product_id", "pair_type"]], on="eval_id", how="left")

    rows = []
    for (label, seed, ds), d in df.groupby(["label", "seed", "ds"]):
        rows.append({"cell": f"{label}@s{seed}", "ds": ds, "n": len(d),
                     "score": round(float(d.score.mean()), 4)})
    tab = pd.DataFrame(rows)
    pivot = tab.pivot(index="cell", columns="ds", values="score").round(4)
    print(pivot.to_string())

    lines = ["# Control cells — COMET trên eval_F_v2_common", ""]
    lines += ["| cell | COMET 1688 | COMET-QE Tiki | n 1688 | n Tiki |",
              "|---|---:|---:|---:|---:|"]
    for cell, r in pivot.iterrows():
        n1688 = int(tab[(tab.cell == cell) & (tab.ds == "1688")].n.iloc[0])
        ntiki = int(tab[(tab.cell == cell) & (tab.ds == "tiki")].n.iloc[0])
        lines.append(f"| {cell} | {r.get('1688', float('nan')):.4f} | "
                     f"{r.get('tiki', float('nan')):.4f} | {n1688} | {ntiki} |")

    if args.baseline:
        bl, bseed = (args.baseline.split("@") + ["42"])[:2]
        bseed = int(bseed)
        lines += ["", f"## So với baseline `{bl}@s{bseed}`", "",
                  "| cell | Δ 1688 | Δ Tiki |", "|---|---:|---:|"]
        for (label, seed) in sorted(df[["label", "seed"]].drop_duplicates().itertuples(index=False)):
            if label == bl and seed == bseed:
                continue
            outs = []
            for ds in ("1688", "tiki"):
                a = df[(df.label == label) & (df.seed == seed) & (df.ds == ds)][["eval_id", "score", "product_id"]]
                b = df[(df.label == bl) & (df.seed == bseed) & (df.ds == ds)][["eval_id", "score"]]
                m = a.merge(b, on="eval_id", suffixes=("", "_b"))
                if len(m) < 5:
                    outs.append("n/a")
                    continue
                info = boot((m.score - m.score_b).to_numpy(),
                            m.product_id.fillna(m.eval_id).to_numpy())
                outs.append(fmt(info))
            lines.append(f"| {label}@s{seed} | {outs[0]} | {outs[1]} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
