#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cluster bootstrap theo product_id cho weight survey A/B.

Eval có nhiều dòng/sản phẩm (A 1,31; B 1,14) nên resample theo eval_id
đánh giá thấp CI. Script này resample theo product_id.

Output: reports/CLUSTER_BOOTSTRAP.md + .artifacts/cluster_bootstrap.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"scores_(?P<config>[a-z_]+)_seed(?P<seed>\d+)_(?P<dataset>1688|tiki)\.parquet")
CONFIGS = {
    "A": ["attr_heavy", "balanced", "title_heavy", "tiki_heavy"],
    "B": ["balanced", "desc_heavy", "title_heavy", "tiki_heavy"],
}


def load_suite(scores_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        frame = pd.read_parquet(path)
        frame["config"] = match.group("config")
        frame["dataset"] = match.group("dataset")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["eval_id"] = df["eval_id"].astype(str)
    return df


def cluster_bootstrap(diffs: np.ndarray, clusters: np.ndarray, n: int = 4000,
                      seed: int = 42) -> dict:
    """Resample product clusters; trả mean + CI + P(win)."""
    order = np.argsort(clusters, kind="stable")
    diffs, clusters = diffs[order], clusters[order]
    uniq, starts = np.unique(clusters, return_index=True)
    ends = np.r_[starts[1:], len(diffs)]
    groups = [diffs[s:e] for s, e in zip(starts, ends)]
    lengths = np.array([len(g) for g in groups], dtype=float)
    sums = np.array([g.sum() for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(groups), size=len(groups))
        means[i] = sums[idx].sum() / max(1.0, lengths[idx].sum())
    return {
        "n_rows": int(len(diffs)),
        "n_products": int(len(groups)),
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "p_win": round(float((means > 0).mean()), 3),
    }


def main() -> int:
    out = []
    lines = ["# Cluster bootstrap theo product_id", "",
             "Paired bootstrap 4.000 lần, resample **sản phẩm** thay vì `eval_id`.",
             "So sánh với CI theo `eval_id` để thấy mức độ hẹp giả tạo.", ""]
    for suite in ("A", "B"):
        scores_dir = ROOT / f".artifacts/scores_{suite}"
        eval_path = ROOT / f"data/cleaned/weight_splits_v2/eval_{suite}.parquet"
        if not scores_dir.exists():
            print("skip", suite)
            continue
        df = load_suite(scores_dir)
        products = pd.read_parquet(eval_path)[["eval_id", "product_id"]].drop_duplicates()
        products["eval_id"] = products["eval_id"].astype(str)
        df = df.merge(products, on="eval_id", how="left", suffixes=("", "_eval"))
        base = df[df.config == "natural"].set_index("eval_id")["score"]
        lines += [f"## Tập {suite}", "",
                  "| config | dataset | Δ | CI 95% (cluster) | P(win) | n dòng | n sản phẩm |",
                  "|---|---|---:|---|---:|---:|---:|"]
        for config in CONFIGS[suite]:
            part = df[df.config == config]
            if part.empty:
                continue
            merged = part.merge(base.rename("score_base"), left_on="eval_id", right_index=True)
            for dataset in ("1688", "tiki", "all"):
                cell = merged if dataset == "all" else merged[merged.dataset == dataset]
                diffs = (cell["score"] - cell["score_base"]).to_numpy()
                clusters = cell["product_id"].fillna(cell["eval_id"]).to_numpy()
                info = cluster_bootstrap(diffs, clusters)
                lines.append(f"| {config} | {dataset} | {info['mean_diff']:+.4f} | "
                             f"[{info['ci_low']:+.4f}; {info['ci_high']:+.4f}] | "
                             f"{info['p_win']:.3f} | {info['n_rows']} | {info['n_products']} |")
                out.append({"suite": suite, "config": config, "dataset": dataset, **info})
        lines.append("")

    # so sánh trực tiếp balanced vs title_heavy ở B
    scores_dir = ROOT / ".artifacts/scores_B"
    if scores_dir.exists():
        df = load_suite(scores_dir)
        products = pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/eval_B.parquet")[
            ["eval_id", "product_id"]].drop_duplicates()
        products["eval_id"] = products["eval_id"].astype(str)
        df = df.merge(products, on="eval_id", how="left", suffixes=("", "_eval"))
        a = df[df.config == "balanced"][["eval_id", "dataset", "score", "product_id"]]
        b = df[df.config == "title_heavy"][["eval_id", "dataset", "score", "product_id"]]
        m = a.merge(b, on=["eval_id", "dataset"], suffixes=("_bal", "_title"))
        info = cluster_bootstrap((m.score_bal - m.score_title).to_numpy(),
                                 m.product_id_bal.fillna(m.eval_id).to_numpy())
        lines += ["## B: balanced − title_heavy", "",
                  f"Δ = {info['mean_diff']:+.4f}  CI [{info['ci_low']:+.4f}; {info['ci_high']:+.4f}]  "
                  f"P(win) = {info['p_win']:.3f}  ({info['n_products']} sản phẩm)", ""]

    lines += ["## Đọc kết quả", "",
              "- Nếu CI cluster rộng hơn CI theo `eval_id`, kết luận trước đó đang lạc quan.",
              "- Chỉ coi config thắng khi CI cluster không chứa 0 **và** P(win) ≥ 0,95.",
              "- A/B dùng chung sản phẩm giữa các config nên đây vẫn là paired bootstrap.", ""]
    (ROOT / "reports/CLUSTER_BOOTSTRAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / ".artifacts/cluster_bootstrap.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
