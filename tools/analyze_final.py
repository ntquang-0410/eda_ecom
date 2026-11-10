#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""So sánh final 4B (8M/1000) với các baseline 4M/500 trên eval_F.

Final dùng eval_F (1500 dòng). Baseline A/B dùng eval_A/eval_B tương ứng,
merge theo eval_id + stratum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def boot(diff: np.ndarray, clusters: np.ndarray, n: int = 4000, seed: int = 42) -> dict:
    order = np.argsort(clusters, kind="stable")
    diff, clusters = diff[order], clusters[order]
    _, starts = np.unique(clusters, return_index=True)
    ends = np.r_[starts[1:], len(diff)]
    groups = [diff[s:e] for s, e in zip(starts, ends)]
    lens = np.array([len(g) for g in groups], float)
    sums = np.array([g.sum() for g in groups], float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        means[i] = sums[idx].sum() / max(1.0, lens[idx].sum())
    return {"mean": round(float(diff.mean()), 4),
            "lo": round(float(np.percentile(means, 2.5)), 4),
            "hi": round(float(np.percentile(means, 97.5)), 4),
            "n": int(len(diff))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", default=".artifacts/final_out/comet/scores")
    ap.add_argument("--out", default="reports/FINAL_4B.md")
    args = ap.parse_args()

    root = Path(".")
    final = []
    for ds in ("1688", "tiki"):
        f = Path(args.final_dir) / f"scores_final_seed42_{ds}.parquet"
        if not f.exists():
            raise FileNotFoundError(f)
        d = pd.read_parquet(f); d["dataset"] = ds; final.append(d)
    F = pd.concat(final, ignore_index=True)
    F["eval_id"] = F["eval_id"].astype(str)

    products = pd.concat([
        pd.read_parquet(root / "data/cleaned/weight_splits_v2/eval_A.parquet")[["eval_id", "product_id"]],
        pd.read_parquet(root / "data/cleaned/weight_splits_v2/eval_B.parquet")[["eval_id", "product_id"]],
    ]).drop_duplicates("eval_id")
    products["eval_id"] = products["eval_id"].astype(str)
    F = F.merge(products, on="eval_id", how="left")

    baselines = {
        "natural_4M_seed42": (root / ".artifacts/scores_A", root / ".artifacts/scores_B", "natural", 42),
        "natural_4M_seed123": (root / ".artifacts/seed123_out/comet/scores/A",
                               root / ".artifacts/seed123_out/comet/scores/B", "natural", 123),
        "balanced_4M_seed42": (root / ".artifacts/scores_A", root / ".artifacts/scores_B", "balanced", 42),
        "balanced_4M_seed123": (root / ".artifacts/seed123_out/comet/scores/A",
                                root / ".artifacts/seed123_out/comet/scores/B", "balanced", 123),
    }

    strata = [("1688", "title"), ("1688", "attribute"), ("1688", "description"),
              ("tiki", "title"), ("tiki", "attribute"), ("tiki", "description")]
    lines = ["# Final 4B — so sánh với baseline 4M", "",
             "Final: recipe A+B, **8M token / 1.000 step**, seed 42.", "",
             "Baseline: 4M token / 500 step, seed 42/123. Khác budget nên chỉ tham khảo.", ""]

    # điểm tuyệt đối
    lines += ["## COMET theo stratum", "",
              "| stratum | final | n |", "|---|---:|---:|"]
    for ds, pt in strata:
        cell = F[(F.dataset == ds) & (F.pair_type == pt)]
        if len(cell) == 0:
            continue
        lines.append(f"| {ds} {pt} | {cell.score.mean():.4f} | {len(cell)} |")
    lines.append("")

    for name, (da, db, config, seed) in baselines.items():
        parts = []
        for ds, ddir in (("1688", da), ("tiki", db)):
            p = ddir / f"scores_{config}_seed{seed}_{ds}.parquet"
            if p.exists():
                x = pd.read_parquet(p); x["dataset"] = ds; parts.append(x)
        if not parts:
            continue
        base = pd.concat(parts, ignore_index=True); base["eval_id"] = base["eval_id"].astype(str)
        m = F.merge(base[["eval_id", "score"]], on="eval_id", suffixes=("", "_base"))
        lines += [f"## Final − {name}", "",
                  "| stratum | Δ | CI 95% | n |", "|---|---:|---|---:|"]
        for ds, pt in strata:
            cell = m[(m.dataset == ds) & (m.pair_type == pt)]
            if len(cell) == 0:
                continue
            info = boot((cell.score - cell.score_base).to_numpy(),
                        cell.product_id.fillna(cell.eval_id).to_numpy())
            lines.append(f"| {ds} {pt} | {info['mean']:+.4f} | "
                         f"[{info['lo']:+.4f}; {info['hi']:+.4f}] | {info['n']} |")
        info = boot((m.score - m.score_base).to_numpy(),
                    m.product_id.fillna(m.eval_id).to_numpy())
        lines += ["", f"Toàn bộ: **{info['mean']:+.4f}** CI [{info['lo']:+.4f}; {info['hi']:+.4f}], "
                  f"n={info['n']}", ""]

    lines += ["## Đọc kết quả", "",
              "- Gain chỉ có ý nghĩa nếu CI không chứa 0 ở stratum tương ứng.",
              "- Final train 8M gấp đôi baseline 4M: chênh lệch dương có thể do budget, không chỉ recipe.",
              "- Muốn tách recipe khỏi budget cần natural 8M/1.000 step (chưa chạy).", ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
