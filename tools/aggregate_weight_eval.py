#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp COMET + Gemma cho thực nghiệm weight, có paired bootstrap theo eval_id.

Input:
  --scores-dir  thư mục chứa scores_<config>_seed<seed>_<dataset>.parquet
                và comet_scores.json
  --gemma       gemma_scores.json (tùy chọn)
  --eval-dir    chứa eval_1688.parquet / eval_tiki.parquet (để lấy product_id)

Output:
  reports/WEIGHT_EVAL.md
  data/cleaned/weight_eval/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = ("natural", "balanced", "no_desc", "tiki_heavy", "title_heavy")
STRATA = (("1688", "title"), ("1688", "attribute"), ("1688", "description"),
          ("tiki", "title"), ("tiki", "attribute"))
PATTERN = re.compile(r"scores_(?P<config>[a-z_]+)_seed(?P<seed>\d+)_(?P<dataset>1688|tiki)\.parquet")


def paired_bootstrap(diff: np.ndarray, n: int = 2000, seed: int = 42) -> dict:
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


def load_scores(scores_dir: Path, eval_dir: Path) -> pd.DataFrame:
    frames = []
    products, pair_types = {}, {}
    for dataset in ("1688", "tiki"):
        path = eval_dir / f"eval_{dataset}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            products.update(dict(zip(frame["eval_id"], frame["product_id"])))
            pair_types.update(dict(zip(frame["eval_id"], frame["pair_type"])))
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        df = pd.read_parquet(path)
        if "score" not in df.columns:
            for candidate in ("comet", "comet_qe", "score_qe"):
                if candidate in df.columns:
                    df["score"] = df[candidate]
                    break
        if "score" not in df.columns or "eval_id" not in df.columns:
            continue
        df = df[["eval_id", "score"]].dropna()
        df["config"] = match.group("config")
        df["seed"] = int(match.group("seed"))
        df["dataset"] = match.group("dataset")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["eval_id", "score", "config", "seed", "dataset"])
    df = pd.concat(frames, ignore_index=True)
    df["product_id"] = df["eval_id"].map(products).fillna(df["eval_id"])
    df["pair_type"] = df["eval_id"].map(pair_types).fillna(
        df["eval_id"].map(derive_pair_type))
    return df


def derive_pair_type(eval_id: str) -> str:
    text = str(eval_id)
    if ":title" in text:
        return "title"
    if ":attr" in text:
        return "attribute"
    if ":desc" in text:
        return "description"
    return "unknown"


def strata_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["pair_type"] = df["pair_type"] if "pair_type" in df else df["eval_id"].map(derive_pair_type)
    rows = []
    for config in CONFIGS:
        part = df[df["config"] == config]
        if part.empty:
            continue
        row = {"config": config}
        for dataset, pair_type in STRATA:
            cell = part[(part["dataset"] == dataset) & (part["pair_type"] == pair_type)]
            row[f"{dataset}_{pair_type}"] = round(float(cell["score"].mean()), 4) if len(cell) else None
        tiki = [row[f"tiki_{p}"] for p in ("title", "attribute")
                if row[f"tiki_{p}"] is not None]
        c1688 = [row[f"1688_{p}"] for p in ("title", "attribute", "description")
                 if row[f"1688_{p}"] is not None]
        allv = [row[f"{d}_{p}"] for d, p in STRATA if row[f"{d}_{p}"] is not None]
        row["tiki_macro"] = round(float(np.mean(tiki)), 4) if tiki else None
        row["macro_1688"] = round(float(np.mean(c1688)), 4) if c1688 else None
        row["all_macro"] = round(float(np.mean(allv)), 4) if allv else None
        rows.append(row)
    return pd.DataFrame(rows)


def seed_consistency(df: pd.DataFrame) -> dict:
    out = {}
    for config in CONFIGS:
        if config == "natural":
            continue
        for dataset, pair_type in STRATA:
            part = df[(df["dataset"] == dataset) & (df["pair_type"] == pair_type)]
            means = part.groupby(["config", "seed"])["score"].mean().unstack("config")
            if config not in means or "natural" not in means:
                continue
            diff = (means[config] - means["natural"]).dropna()
            out[f"{config}::{dataset}::{pair_type}"] = {
                "seed_diffs": {str(k): round(float(v), 4) for k, v in diff.items()},
                "n_seeds": int(len(diff)),
                "same_sign": bool(len(diff) >= 2 and ((diff > 0).all() or (diff < 0).all())),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="/kaggle/working/weight_runs")
    ap.add_argument("--eval-dir", default="data/cleaned/weight_eval")
    ap.add_argument("--gemma", default="data/cleaned/judge/gemma_scores.json")
    ap.add_argument("--out", default="reports/WEIGHT_EVAL.md")
    ap.add_argument("--summary", default="data/cleaned/weight_eval/summary.json")
    ap.add_argument("--interim", action="store_true",
                    help="chế độ 1 seed: không bắt buộc đồng thuận 2 seed")
    args = ap.parse_args()

    scores_dir, eval_dir = Path(args.scores_dir), Path(args.eval_dir)
    df = load_scores(scores_dir, eval_dir)
    table = strata_table(df)

    # paired vs natural, gộp 2 seed theo từng eval_id
    paired = {}
    if not df.empty:
        avg = (df.groupby(["config", "dataset", "pair_type", "eval_id"])["score"]
               .mean().reset_index())
        for config in CONFIGS:
            if config == "natural":
                continue
            for dataset, pair_type in STRATA:
                part = avg[(avg["dataset"] == dataset) & (avg["pair_type"] == pair_type)]
                piv = part.pivot_table(index="eval_id", columns="config", values="score")
                if config not in piv or "natural" not in piv:
                    continue
                diff = (piv[config] - piv["natural"]).dropna().to_numpy()
                paired[f"{config}::{dataset}::{pair_type}"] = paired_bootstrap(diff)

    # macro Tiki: trung bình 2 stratum, bootstrap theo eval_id từng stratum
    macro_paired = {}
    if not df.empty:
        rng = np.random.default_rng(7)
        for config in CONFIGS:
            if config == "natural":
                continue
            draws = []
            for dataset, pair_type in (("tiki", "title"), ("tiki", "attribute")):
                part = avg[(avg["dataset"] == dataset) & (avg["pair_type"] == pair_type)]
                piv = part.pivot_table(index="eval_id", columns="config", values="score")
                if config not in piv or "natural" not in piv:
                    continue
                diff = (piv[config] - piv["natural"]).dropna().to_numpy()
                idx = rng.integers(0, len(diff), size=(2000, len(diff)))
                draws.append(diff[idx].mean(axis=1))
            if len(draws) == 2:
                macro = (draws[0] + draws[1]) / 2
                macro_paired[config] = {
                    "mean_diff": round(float(macro.mean()), 4),
                    "ci_low": round(float(np.percentile(macro, 2.5)), 4),
                    "ci_high": round(float(np.percentile(macro, 97.5)), 4),
                    "p_win": round(float((macro > 0).mean()), 3),
                }

    consistency = seed_consistency(df) if not df.empty else {}

    gemma = None
    if Path(args.gemma).exists():
        gemma = json.loads(Path(args.gemma).read_text(encoding="utf-8"))
    gemma_table = []
    if gemma:
        by_stratum = gemma.get("summary", {}).get("by_stratum", {})
        for key, value in sorted(by_stratum.items()):
            config, dataset, pair_type = key.split("::", 2)
            gemma_table.append({"config": config, "dataset": dataset,
                                "pair_type": pair_type, "gemma_mean": value.get("mean"),
                                "gemma_rating": value.get("mean_rating", ""),
                                "n": value.get("n")})

    # quyết định
    decision = {"decision": "natural", "reason": "chưa có dữ liệu"}
    if not table.empty:
        natural = table[table["config"] == "natural"]
        natural_tiki = float(natural["tiki_macro"].iloc[0]) if len(natural) else None
        natural_1688 = float(natural["macro_1688"].iloc[0]) if len(natural) else None
        candidates = table[table["config"] != "natural"].copy()
        # chỉ so các cấu hình có đủ số seed như natural (trừ chế độ interim)
        if not args.interim:
            seed_counts = df.groupby("config")["seed"].nunique()
            max_seeds = int(seed_counts.max()) if len(seed_counts) else 0
            eligible = seed_counts[seed_counts >= max_seeds].index
            candidates = candidates[candidates["config"].isin(eligible)]
        # guardrail 1688: không thấp hơn natural quá 0.01
        if natural_1688 is not None:
            candidates = candidates[candidates["macro_1688"] >= natural_1688 - 0.01]
        if len(candidates) and natural_tiki is not None:
            best = candidates.sort_values("tiki_macro", ascending=False).iloc[0]
            diff = float(best["tiki_macro"] - natural_tiki)
            ci = macro_paired.get(best["config"], {})
            sig = bool(ci.get("ci_low") is not None and ci["ci_low"] > 0)
            consistent = all(
                consistency.get(f"{best['config']}::tiki::{p}", {}).get("same_sign", False)
                for p in ("title", "attribute"))
            if args.interim:
                consistent = True
            if diff >= 0.005 and sig and consistent:
                decision = {
                    "decision": best["config"],
                    "reason": (f"{best['config']} hơn natural {diff:+.4f} COMET-QE Tiki "
                               f"(CI {ci['ci_low']:+.4f}..{ci['ci_high']:+.4f}), "
                               f"{'interim 1 seed' if args.interim else 'đồng thuận 2 seed'}, "
                               f"1688 không giảm quá 0,01"),
                }
            else:
                reasons = []
                if diff < 0.005:
                    reasons.append(f"chênh Tiki chỉ {diff:+.4f} (<0,005)")
                if not sig:
                    reasons.append("CI 95% chứa 0")
                if not consistent:
                    reasons.append("2 seed không cùng dấu")
                decision = {"decision": "natural", "reason": "; ".join(reasons)}

    summary = {
        "protocol": {
            "budget_tokens": 5_000_000,
            "steps": 625,
            "seeds": sorted(df["seed"].unique().tolist()) if not df.empty else [],
            "eval_1688": {"title": 300, "attribute": 300, "description": 150},
            "eval_tiki": {"title": 300, "attribute": 300},
            "metric_1688": "COMET-da (reference)",
            "metric_tiki": "COMET-QE (reference-free)",
        },
        "table": table.to_dict("records") if not table.empty else [],
        "paired_vs_natural": paired,
        "tiki_macro_paired": macro_paired,
        "seed_consistency": consistency,
        "gemma": gemma_table,
        "decision": decision,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    lines = ["# Trọng số desc / title / attribute có làm tăng chất lượng?",
             "", "## Giao thức", "",
             "- Qwen3-4B LoRA, 5M token/run, 625 optimizer step",
             f"- 5 cấu hình × {'1 seed (interim)' if args.interim else '2 seed (42, 123)'}",
             "- eval cố định: 1688 title 300 / attr 300 / desc 150; Tiki title 300 / attr 300",
             "- 1688: COMET-da có reference; Tiki: COMET-QE không reference",
             "- paired bootstrap 2.000 lần theo eval_id", ""]
    if table.empty:
        lines.append("Chưa có file scores. Hãy chạy COMET trước.")
    else:
        columns = ["config", "1688_title", "1688_attribute", "1688_description",
                   "tiki_title", "tiki_attribute", "tiki_macro", "macro_1688", "all_macro"]
        lines += ["## Điểm COMET theo stratum", "",
                  "| " + " | ".join(columns) + " |",
                  "|" + "---|" * len(columns)]
        for row in table.to_dict("records"):
            vals = [row.get(c) for c in columns]
            lines.append("| " + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals) + " |")

        lines += ["", "## So với natural (paired bootstrap, gộp 2 seed)", "",
                  "| cấu hình | stratum | diff | CI 95% | P(win) |", "|---|---|---:|---|---:|"]
        for key, val in paired.items():
            config, dataset, pair_type = key.split("::", 2)
            lines.append(f"| {config} | {dataset}/{pair_type} | {val['mean_diff']:+.4f} | "
                         f"[{val['ci_low']:+.4f}, {val['ci_high']:+.4f}] | {val['p_win']} |")

        lines += ["", "## Tiki macro", "",
                  "| cấu hình | diff | CI 95% | P(win) |", "|---|---:|---|---:|"]
        for config, val in macro_paired.items():
            lines.append(f"| {config} | {val['mean_diff']:+.4f} | "
                         f"[{val['ci_low']:+.4f}, {val['ci_high']:+.4f}] | {val['p_win']} |")

        if gemma_table:
            lines += ["", "## Gemma credit (1-9, blind)", "",
                      "| cấu hình | dataset | pair_type | mean | rating | n |",
                      "|---|---|---|---:|---|---:|"]
            for row in gemma_table:
                lines.append(f"| {row['config']} | {row['dataset']} | {row['pair_type']} | "
                             f"{row['gemma_mean']} | {row['gemma_rating']} | {row['n']} |")

        lines += ["", "## Kết luận", "",
                  f"**{decision['decision']}** — {decision['reason']}", ""]
        if decision["decision"] == "natural":
            lines += ["Không có bằng chứng đủ mạnh rằng gán weight làm tăng chất lượng.",
                      "Giữ phân bố token tự nhiên (desc ~64%).", ""]
        else:
            lines += [f"Dùng weight của `{decision['decision']}`; xem bảng trên để biết "
                      "stratum nào tăng và stratum nào phải giữ guardrail.", ""]
        lines += ["## Đọc kết quả", "",
                  "- Nếu một cấu hình tăng đúng stratum mà nó được ưu tiên (ví dụ title_heavy tăng 1688/title)",
                  "  và không làm giảm stratum khác quá 0,01, đó là bằng chứng weight có tác dụng.",
                  "- Nếu mọi chênh lệch đều <0,005 hoặc CI chứa 0, kết luận là natural.",
                  "- Gemma chỉ là tín hiệu phụ; ưu tiên COMET paired.", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
