#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp kết quả weight survey A/B (quota sampler).

Mỗi suite (A, B) có thư mục score riêng vì tên file
`scores_<config>_seed<seed>_<dataset>.parquet` trùng nhau giữa A và B.

Điểm 1688 = COMET-da (có reference), Tiki = COMET-QE (không reference).
Hai thang điểm khác nhau nên KHÔNG lấy trung bình gộp; báo cáo riêng từng
dataset + từng stratum, paired bootstrap theo eval_id.

Output:
  - markdown + summary json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r"scores_(?P<config>[a-z_]+)_seed(?P<seed>\d+)_(?P<dataset>1688|tiki)\.parquet")


def paired_bootstrap(diff: np.ndarray, n: int = 4000, seed: int = 42) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return {"n": 0, "mean_diff": None, "ci_low": None, "ci_high": None, "p_win": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return {
        "n": int(len(diff)),
        "mean_diff": round(float(diff.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "p_win": round(float((means > 0).mean()), 3),
    }


def load_scores(path: Path) -> dict[tuple[str, int], pd.DataFrame]:
    frames = []
    for score_file in sorted(path.glob("scores_*.parquet")):
        if not PATTERN.match(score_file.name):
            continue
        frames.append(pd.read_parquet(score_file))
    if not frames:
        raise FileNotFoundError(f"không có scores_*.parquet trong {path}")
    df = pd.concat(frames, ignore_index=True)
    df["eval_id"] = df["eval_id"].astype(str)
    return {key: part.reset_index(drop=True) for key, part in df.groupby(["config", "seed"])}


def load_gemma(path: Path | None) -> dict[str, float]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    strata = data.get("summary", {}).get("by_stratum", {})
    return {key: float(value.get("mean", float("nan"))) for key, value in strata.items()}


def mean_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, part in df.groupby("config"):
        row = {"config": config, "n": len(part)}
        for dataset, cell in part.groupby("dataset"):
            row[dataset] = cell["score"].mean()
        for (dataset, pair_type), cell in part.groupby(["dataset", "pair_type"]):
            row[f"{dataset}_{pair_type}"] = cell["score"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("1688", ascending=False).reset_index(drop=True)


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="A hoặc B")
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--gemma", default="")
    ap.add_argument("--comet-json", default="")
    ap.add_argument("--baseline", default="natural")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    scores = load_scores(Path(args.scores_dir))
    gemma = load_gemma(Path(args.gemma) if args.gemma else None)
    frame = pd.concat(scores.values(), ignore_index=True)
    table = mean_table(frame)

    boot = {}
    for (config, seed), part in scores.items():
        base_key = (args.baseline, seed)
        if config == args.baseline or base_key not in scores:
            continue
        merged = part.merge(scores[base_key], on="eval_id", suffixes=("_cfg", "_base"))
        info = {}
        for name, subset in [("all", merged),
                             ("1688", merged[merged["dataset_cfg"] == "1688"]),
                             ("tiki", merged[merged["dataset_cfg"] == "tiki"])]:
            diff = (subset["score_cfg"] - subset["score_base"]).to_numpy()
            info[name] = paired_bootstrap(diff)
        boot[f"{config}::seed{seed}"] = info

    comet_raw = {}
    if args.comet_json and Path(args.comet_json).exists():
        comet_raw = json.loads(Path(args.comet_json).read_text(encoding="utf-8"))

    gemma_rows, gemma_cfg = {}, {}
    for key, value in gemma.items():
        parts = key.split("::")
        if len(parts) != 3:
            continue
        gemma_rows[tuple(parts)] = value
        gemma_cfg.setdefault(parts[0], []).append(value)

    summary = {
        "suite": args.suite,
        "baseline": args.baseline,
        "configs": table.to_dict("records"),
        "bootstrap_vs_baseline": boot,
        "gemma_mean": {k: float(np.mean(v)) for k, v in gemma_cfg.items()},
        "gemma_by_stratum": {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in gemma_rows.items()},
        "comet_raw": comet_raw,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    strata = [c for c in table.columns if "_" in c and c not in ("1688", "tiki")]
    lines = [
        f"# Weight survey {args.suite} — quota sampler",
        "",
        f"Baseline: `{args.baseline}`. 1688 = COMET-da, Tiki = COMET-QE.",
        "Hai thang điểm khác nhau nên báo cáo riêng, không lấy trung bình gộp.",
        "Paired bootstrap 4.000 lần trên eval_id dùng chung.",
        "",
        "## COMET",
        "",
        "| config | n | 1688 (da) | Tiki (qe) | " + " | ".join(strata) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(strata),
    ]
    for _, row in table.iterrows():
        cells = " | ".join(fmt(row[s]) for s in strata)
        lines.append(f"| {row['config']} | {int(row['n'])} | **{fmt(row.get('1688'))}** | "
                     f"**{fmt(row.get('tiki'))}** | {cells} |")

    lines += ["", "## Paired bootstrap vs baseline", "",
              "| config | Δ1688 | CI 95% | P(win) | ΔTiki | CI 95% | P(win) |",
              "|---|---:|---|---:|---:|---|---:|"]
    for name, info in sorted(boot.items(), key=lambda x: -(x[1]["1688"]["mean_diff"] or 0)):
        a, t = info["1688"], info["tiki"]
        lines.append(f"| {name} | {a['mean_diff']:+.4f} | "
                     f"[{a['ci_low']:+.4f}, {a['ci_high']:+.4f}] | {a['p_win']:.3f} | "
                     f"{t['mean_diff']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
                     f"{t['p_win']:.3f} |")

    if gemma_cfg:
        lines += ["", "## Gemma credit (1..9, mẫu 50 eval_id/stratum/config)", "",
                  "| config | mean | n stratum |", "|---|---:|---:|"]
        for config, values in sorted(gemma_cfg.items(), key=lambda x: -np.mean(x[1])):
            lines.append(f"| {config} | {np.mean(values):.2f} | {len(values)} |")
    else:
        lines += ["", "_Chưa có Gemma score._"]

    if comet_raw:
        lines += ["", "### COMET system score (raw)", "",
                  "| key | system | BLEU | chrF |", "|---|---:|---:|---:|"]
        for key, value in sorted(comet_raw.items()):
            lines.append(f"| {key} | {value.get('system_score', float('nan')):.3f} | "
                         f"{fmt(value.get('bleu'))} | {fmt(value.get('chrf'))} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
