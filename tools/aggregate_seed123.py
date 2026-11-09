#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp seed 42 vs seed 123 cho weight survey A/B.

Output: reports/SEED123_<suite>.md + .artifacts/seed123_<suite>.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r"scores_(?P<config>[a-z_]+)_seed(?P<seed>\d+)_(?P<dataset>1688|tiki)\.parquet")


def load_scores(path: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(path.glob("scores_*.parquet")):
        m = PATTERN.match(f.name)
        if not m:
            continue
        df = pd.read_parquet(f)
        df["config"] = m.group("config")
        df["dataset"] = m.group("dataset")
        df["seed"] = int(m.group("seed"))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(path)
    out = pd.concat(frames, ignore_index=True)
    out["eval_id"] = out["eval_id"].astype(str)
    return out


def cluster_boot(diffs: np.ndarray, clusters: np.ndarray, n: int = 4000, seed: int = 42) -> dict:
    order = np.argsort(clusters, kind="stable")
    diffs, clusters = diffs[order], clusters[order]
    uniq, starts = np.unique(clusters, return_index=True)
    ends = np.r_[starts[1:], len(diffs)]
    groups = [diffs[s:e] for s, e in zip(starts, ends)]
    lens = np.array([len(g) for g in groups], float)
    sums = np.array([g.sum() for g in groups], float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        means[i] = sums[idx].sum() / max(1.0, lens[idx].sum())
    return {"mean": round(float(diffs.mean()), 4),
            "lo": round(float(np.percentile(means, 2.5)), 4),
            "hi": round(float(np.percentile(means, 97.5)), 4),
            "p": round(float((means > 0).mean()), 3),
            "n_products": int(len(groups))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--scores42", required=True)
    ap.add_argument("--scores123", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    products = pd.read_parquet(args.eval)[["eval_id", "product_id"]].drop_duplicates()
    products["eval_id"] = products["eval_id"].astype(str)
    frames = []
    for path in (args.scores42, args.scores123):
        d = load_scores(Path(path))
        d = d.merge(products, on="eval_id", how="left", suffixes=("", "_eval"))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    seeds = sorted(df.seed.unique())
    configs = set(df.config)
    configs.discard("natural")

    base = {(s, ds): None for s in seeds for ds in ("1688", "tiki")}
    for s in seeds:
        for ds in ("1688", "tiki"):
            part = df[(df.seed == s) & (df.dataset == ds) & (df.config == "natural")]
            base[(s, ds)] = part.set_index("eval_id")["score"]

    rows, verdict = [], []
    for config in sorted(configs):
        rec = {"config": config}
        for ds in ("1688", "tiki"):
            deltas = []
            for s in seeds:
                part = df[(df.seed == s) & (df.dataset == ds) & (df.config == config)]
                if part.empty or base[(s, ds)] is None:
                    continue
                m = part.merge(base[(s, ds)].rename("score_base"), left_on="eval_id", right_index=True)
                info = cluster_boot((m.score - m.score_base).to_numpy(),
                                     m.product_id.fillna(m.eval_id).to_numpy())
                rec[f"{ds}_{s}"] = info
                deltas.append(info["mean"])
            if deltas:
                rec[f"{ds}_mean"] = round(float(np.mean(deltas)), 4)
                rec[f"{ds}_same_sign"] = all(x > 0 for x in deltas) or all(x < 0 for x in deltas)
        rows.append(rec)

    lines = [f"# Seed 42 vs 123 — tập {args.suite}", "",
             "Cluster bootstrap theo `product_id`, 4.000 lần. Δ so với `natural` cùng seed.", ""]
    for ds in ("1688", "tiki"):
        lines += [f"## {ds}", "",
                  "| config | Δ seed 42 | CI 42 | Δ seed 123 | CI 123 | Δ trung bình | cùng dấu |",
                  "|---|---:|---|---:|---|---:|---|"]
        for r in rows:
            a = r.get(f"{ds}_42"); b = r.get(f"{ds}_123")
            if not a and not b:
                continue
            fa = f"{a['mean']:+.4f} [{a['lo']:+.4f}; {a['hi']:+.4f}]" if a else "-"
            fb = f"{b['mean']:+.4f} [{b['lo']:+.4f}; {b['hi']:+.4f}]" if b else "-"
            mean = r.get(f"{ds}_mean")
            same = r.get(f"{ds}_same_sign")
            lines.append(f"| {r['config']} | {fa} | {fb} | "
                         f"{mean:+.4f} | {'có' if same else 'không'} |")
        lines.append("")

    lines += ["## Cách đọc", "",
              "- Config chỉ được coi là tái lập nếu Δ cùng dấu ở cả 2 seed và CI từng seed không chứa 0.",
              "- Nếu seed 123 đảo dấu hoặc CI chứa 0, gain seed 42 là nhiễu → giữ `natural`.",
              ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.summary).write_text(json.dumps({"suite": args.suite, "seeds": seeds, "configs": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
