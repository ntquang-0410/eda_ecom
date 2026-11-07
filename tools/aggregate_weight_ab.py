#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp kết quả weight survey A/B (quota sampler).

Mỗi suite (A, B) có thư mục score riêng vì tên file
`scores_<config>_seed<seed>_<dataset>.parquet` trùng nhau giữa A và B.

Output:
  - bảng COMET trung bình theo config và theo stratum
  - paired bootstrap vs config baseline (natural)
  - Gemma credit trung bình (nếu có)
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
    """Trả về {(config, seed): frame gồm tất cả dataset}."""
    frames = []
    for score_file in sorted(path.glob("scores_*.parquet")):
        match = PATTERN.match(score_file.name)
        if not match:
            continue
        frame = pd.read_parquet(score_file)
        frames.append(frame)
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
    out = {}
    for key, value in strata.items():
        out[key] = float(value.get("mean", float("nan")))
    return out


def mean_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, part in df.groupby("config"):
        row = {"config": config, "n": len(part), "comet": part["score"].mean()}
        for (dataset, pair_type), cell in part.groupby(["dataset", "pair_type"]):
            row[f"{dataset}_{pair_type}"] = cell["score"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("comet", ascending=False).reset_index(drop=True)


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
        base = scores[base_key]
        merged = part.merge(base, on="eval_id", suffixes=("_cfg", "_base"))
        diff = (merged["score_cfg"] - merged["score_base"]).to_numpy()
        boot[f"{config}::seed{seed}"] = paired_bootstrap(diff)

    comet_raw = {}
    if args.comet_json and Path(args.comet_json).exists():
        comet_raw = json.loads(Path(args.comet_json).read_text(encoding="utf-8"))

    # gộp gemma: key dạng config::dataset::pair_type
    gemma_cfg: dict[str, float] = {}
    gemma_rows = {}
    for key, value in gemma.items():
        parts = key.split("::")
        if len(parts) != 3:
            continue
        config, dataset, pair_type = parts
        gemma_rows[(config, dataset, pair_type)] = value
        gemma_cfg.setdefault(config, []).append(value)
    gemma_mean = {k: float(np.mean(v)) for k, v in gemma_cfg.items()}

    summary = {
        "suite": args.suite,
        "baseline": args.baseline,
        "configs": table.to_dict("records"),
        "bootstrap_vs_baseline": boot,
        "gemma_mean": gemma_mean,
        "gemma_by_stratum": {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in gemma_rows.items()},
        "comet_raw": comet_raw,
        "n_scores": {f"{c}::{s}": int(len(p)) for (c, s), p in scores.items()},
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [c for c in table.columns if c in ("config", "comet", "n")]
    strata = [c for c in table.columns if c not in ("config", "comet", "n")]
    lines = [
        f"# Weight survey {args.suite} — quota sampler",
        "",
        f"Baseline: `{args.baseline}`. Score: COMET-da cho 1688, COMET-QE cho Tiki.",
        "Paired bootstrap 4.000 lần trên eval_id dùng chung.",
        "",
        "## COMET",
        "",
        "| config | n | COMET | " + " | ".join(s for s in strata) + " |",
        "|---|---:|---:|" + "---:|" * len(strata),
    ]
    for _, row in table.iterrows():
        cells = " | ".join(f"{row[s]:.2f}" for s in strata)
        lines.append(f"| {row['config']} | {int(row['n'])} | **{row['comet']:.2f}** | {cells} |")

    lines += ["", "## Paired bootstrap vs baseline", "",
              "| config | n | ΔCOMET | CI 95% | P(win) |", "|---|---:|---:|---|---:|"]
    for name, info in sorted(boot.items(), key=lambda x: -x[1]["mean_diff"]):
        lines.append(f"| {name} | {info['n']} | {info['mean_diff']:+.3f} | "
                     f"[{info['ci_low']:+.3f}, {info['ci_high']:+.3f}] | {info['p_win']:.3f} |")

    if gemma_mean:
        lines += ["", "## Gemma credit (1..9)", "", "| config | mean |", "|---|---:|"]
        for config, value in sorted(gemma_mean.items(), key=lambda x: -x[1]):
            lines.append(f"| {config} | {value:.2f} |")
    else:
        lines += ["", "_Chưa có Gemma score._"]

    if comet_raw:
        lines += ["", "## COMET system score (raw)", "",
                  "| key | system | BLEU | chrF |", "|---|---:|---:|---:|"]
        for key, value in sorted(comet_raw.items()):
            lines.append(f"| {key} | {value.get('system_score', float('nan')):.2f} | "
                         f"{value.get('bleu', float('nan')):.2f} | "
                         f"{value.get('chrf', float('nan')):.2f} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
