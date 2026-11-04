#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp COMET + Gemma để quyết định có cần gán weight cho desc/title/attr.

Input:
  --comet   weight_eval_comet.json (từ kernel)
  --gemma   gemma_scores.json (từ tools/gemma_judge.py)
  --scores-dir  thư mục chứa scores_<config>_{1688,tiki}.parquet (per-segment)

Output:
  reports/WEIGHT_EVAL.md + data/cleaned/weight_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = ("natural", "balanced", "no_desc", "tiki_heavy", "title_heavy")


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000) -> dict:
    diff = a - b
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return {
        "mean_diff": round(float(diff.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "p_win": round(float((means > 0).mean()), 3),
    }


def load_scores(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comet", default="data/cleaned/weight_eval_comet.json")
    ap.add_argument("--gemma", default="data/cleaned/judge/gemma_scores.json")
    ap.add_argument("--scores-dir", default=".")
    ap.add_argument("--out", default="reports/WEIGHT_EVAL.md")
    args = ap.parse_args()

    comet = json.loads(Path(args.comet).read_text(encoding="utf-8"))
    gemma = None
    if Path(args.gemma).exists():
        gemma = json.loads(Path(args.gemma).read_text(encoding="utf-8"))

    scores_dir = Path(args.scores_dir)
    bootstrap = {}
    natural_tiki = load_scores(scores_dir / "scores_natural_tiki.parquet")
    natural_1688 = load_scores(scores_dir / "scores_natural_1688.parquet")
    for name in CONFIGS:
        if name == "natural":
            continue
        row = {}
        t = load_scores(scores_dir / f"scores_{name}_tiki.parquet")
        if t is not None and natural_tiki is not None:
            merged = t.merge(natural_tiki, on="id", suffixes=("", "_nat"))
            row["tiki"] = paired_bootstrap(merged["comet_qe"].to_numpy(),
                                           merged["comet_qe_nat"].to_numpy())
        g = load_scores(scores_dir / f"scores_{name}_1688.parquet")
        if g is not None and natural_1688 is not None:
            merged = g.merge(natural_1688, on="id", suffixes=("", "_nat"))
            row["1688"] = paired_bootstrap(merged["comet"].to_numpy(),
                                           merged["comet_nat"].to_numpy())
        bootstrap[name] = row

    # bảng chính
    rows = []
    for name in CONFIGS:
        if name not in comet:
            continue
        m = comet[name]
        g = (gemma or {}).get("summary", {}).get("by_config", {}).get(name, {}) if gemma else {}
        rows.append({
            "config": name,
            "comet_1688": m.get("comet_1688"),
            "comet_qe_tiki": m.get("comet_qe_tiki"),
            "bleu_1688": m.get("bleu_1688"),
            "chrf_1688": m.get("chrf_1688"),
            "gemma_mean": g.get("mean"),
            "gemma_adequacy": g.get("adequacy"),
            "gemma_fluency": g.get("fluency"),
            "gemma_terminology": g.get("terminology"),
        })
    table = pd.DataFrame(rows)

    # quyết định
    recommendation = {"decision": "unknown", "reason": ""}
    if not table.empty and table["comet_qe_tiki"].notna().any():
        best_1688 = table["comet_1688"].max()
        guard = table[table["comet_1688"] >= best_1688 - 0.01]
        best = guard.sort_values("comet_qe_tiki", ascending=False).iloc[0]
        natural = table[table["config"] == "natural"].iloc[0]
        diff = float(best["comet_qe_tiki"] - natural["comet_qe_tiki"])
        ci = bootstrap.get(best["config"], {}).get("tiki")
        significant = bool(ci and ci["ci_low"] > 0)
        if best["config"] == "natural" or diff < 0.005 or not significant:
            recommendation = {
                "decision": "natural",
                "reason": f"chênh Tiki COMET-QE chỉ {diff:+.4f}, không vượt CI / không đáng đánh đổi",
            }
        else:
            recommendation = {
                "decision": best["config"],
                "reason": f"{best['config']} hơn natural {diff:+.4f} COMET-QE và giữ 1688 trong ngưỡng 0,01",
            }

    summary = {"table": table.to_dict("records"), "bootstrap_vs_natural": bootstrap,
               "recommendation": recommendation}
    Path("data/cleaned/weight_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Đánh giá weight desc/title/attribute (COMET + Gemma)", ""]
    if table.empty:
        lines.append("Chưa có dữ liệu COMET.")
    else:
        lines += ["| cấu hình | COMET 1688 | COMET-QE Tiki | BLEU 1688 | Gemma mean |",
                  "|---|---:|---:|---:|---:|"]
        for row in table.itertuples():
            lines.append(f"| {row.config} | {row.comet_1688:.4f} | {row.comet_qe_tiki:.4f} | "
                         f"{row.bleu_1688:.4f} | {row.gemma_mean if row.gemma_mean is not None else '-'} |")
        lines += ["", "## So với natural (paired bootstrap)", "",
                  "| cấu hình | Tiki diff | CI 95% | P(win) | 1688 diff | CI 95% |",
                  "|---|---:|---|---:|---:|---|"]
        for name, parts in bootstrap.items():
            t = parts.get("tiki", {})
            g = parts.get("1688", {})
            lines.append(f"| {name} | {t.get('mean_diff','-')} | "
                         f"[{t.get('ci_low','-')}, {t.get('ci_high','-')}] | {t.get('p_win','-')} | "
                         f"{g.get('mean_diff','-')} | [{g.get('ci_low','-')}, {g.get('ci_high','-')}] |")
        lines += ["", "## Kết luận", "",
                  f"**{recommendation['decision']}** — {recommendation['reason']}", "",
                  "Nếu kết luận là `natural`: không cần gán weight, giữ phân bố token tự nhiên.", ""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
