#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pha 0 — kiểm tra token share trước khi train.

Không train. Mục tiêu:
  1. Đo token mass thật của từng nguồn (title/attribute/description/1688-Tiki).
  2. Xem share tự nhiên và share sau base weight.
  3. Tính weight cần thiết để đạt share mục tiêu.
  4. Kiểm tra cap 15 attribute/sản phẩm và độ dài description so với max_seq.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CONFIGS = {
    "natural": None,
    "balanced": {"title_1688": 0.15, "attr_1688": 0.35, "desc_1688": 0.25, "tiki": 0.25},
    "no_desc": {"title_1688": 0.25, "attr_1688": 0.45, "desc_1688": 0.0, "tiki": 0.30},
    "tiki_heavy": {"title_1688": 0.10, "attr_1688": 0.25, "desc_1688": 0.20, "tiki": 0.45},
    "title_heavy": {"title_1688": 0.30, "attr_1688": 0.30, "desc_1688": 0.15, "tiki": 0.25},
}


def group_of(row) -> str:
    if row["domain"] == "tiki":
        return "tiki"
    return {"title": "title_1688", "attribute": "attr_1688",
            "description": "desc_1688"}[row["pair_type"]]


def apply_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    attr = df[(df["domain"] == "1688") & (df["pair_type"] == "attribute")].copy()
    attr["_idx"] = attr["pair_id"].str.extract(r":attr:(\d+)$").astype(int)
    attr = attr.sort_values(["product_id", "_idx"]).groupby("product_id").head(cap)
    rest = df[~((df["domain"] == "1688") & (df["pair_type"] == "attribute"))]
    return pd.concat([rest, attr.drop(columns=["_idx"])], ignore_index=True)


def shares(df: pd.DataFrame, token_col: str = "tokens_est", weight_col: str | None = None) -> dict:
    data = df.copy()
    data["_group"] = data.apply(group_of, axis=1)
    data["_mass"] = data[token_col] * (data[weight_col] if weight_col else 1.0)
    mass = data.groupby("_group")["_mass"].sum()
    rows = data.groupby("_group").size()
    total = mass.sum()
    return {
        "tokens": {k: round(float(v) / 1e6, 2) for k, v in mass.items()},
        "share_pct": {k: round(float(v) / total * 100, 1) for k, v in mass.items()},
        "rows": {k: int(v) for k, v in rows.items()},
        "total_million": round(float(total) / 1e6, 2),
    }


def required_weights(df: pd.DataFrame, target: dict, cap: int) -> dict:
    data = apply_cap(df, cap)
    natural = shares(data)
    tokens = {k: v * 1e6 for k, v in natural["tokens"].items()}
    weights = {}
    for group, share in target.items():
        if share <= 0:
            weights[group] = 0.0
        else:
            weights[group] = share / max(1, tokens.get(group, 0))
    mean = sum(weights.values()) / len(weights)
    return {k: round(v / mean, 3) for k, v in weights.items()}, natural


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/cleaned/pairs")
    ap.add_argument("--report", default="reports/PHASE0_TOKEN_SHARE.md")
    args = ap.parse_args()

    d1688 = pd.read_parquet(f"{args.pairs}/1688.parquet")
    tiki = pd.read_parquet(f"{args.pairs}/tiki_qwen3-8b.parquet")
    df = pd.concat([d1688, tiki], ignore_index=True)

    report = {"raw": shares(df),
              "capped_15": shares(apply_cap(df, 15)),
              "base_weight_capped_15": shares(apply_cap(df, 15), weight_col="base_weight"),
              "configs": {}}

    for name, target in CONFIGS.items():
        if target is None:
            report["configs"][name] = {"target": "natural"}
            continue
        weights, natural = required_weights(df, target, 15)
        report["configs"][name] = {
            "target_share_pct": {k: round(v * 100, 1) for k, v in target.items()},
            "required_weight": weights,
            "tokens": natural["tokens"],
        }

    # description length
    desc = d1688[d1688["pair_type"] == "description"]["tokens_est"]
    report["description_tokens"] = {
        "p50": round(float(desc.quantile(0.5)), 1),
        "p95": round(float(desc.quantile(0.95)), 1),
        "p99": round(float(desc.quantile(0.99)), 1),
        "max": round(float(desc.max()), 1),
        "over_1280": int((desc > 1280).sum()),
        "over_512": int((desc > 512).sum()),
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Pha 0 — token share\n\n```json\n"
        + json.dumps(report, ensure_ascii=False, indent=2)
        + "\n```\n", encoding="utf-8")
    (Path(args.pairs) / "phase0_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
