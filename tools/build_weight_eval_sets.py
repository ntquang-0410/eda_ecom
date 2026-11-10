#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng eval set cố định và train view cho thực nghiệm weight.

Mục tiêu: mọi cấu hình weight và mọi seed nhìn đúng cùng một eval set,
cùng id, cùng thứ tự; chỉ khác phân bố token lúc train.

Output:
  data/cleaned/weight_eval/eval_1688.parquet
  data/cleaned/weight_eval/eval_tiki.parquet
  data/cleaned/weight_eval/train_view_4b.parquet
  data/cleaned/weight_eval/train_view_1.7b.parquet
  data/cleaned/weight_eval/meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def target_tokens(row) -> float:
    """Ước lượng token phía target (ZH ~1.6 ký tự/token)."""
    return float(row.get("tgt_chars", 0) or 0) / 1.6


def make_eval_1688(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    test["target_tokens"] = test["target_text"].map(
        lambda x: len(str(x).replace(" ", "")) / 1.6)
    rows = []
    quotas = {"title": 300, "attribute": 300, "description": 150}
    for pair_type, quota in quotas.items():
        part = test[test["pair_type"] == pair_type]
        part = part[(part["tokens_est"] <= 1024) & (part["target_tokens"] <= 700)]
        n = min(quota, len(part))
        rows.append(part.sample(n=n, random_state=seed))
    out = pd.concat(rows, ignore_index=True)
    out["dataset"] = "1688"
    out["eval_id"] = [f"1688v1:{i:05d}" for i in range(len(out))]
    out["max_new"] = out["pair_type"].map(
        {"title": 64, "attribute": 256, "description": 768})
    out["source"] = out["source_text"]
    out["reference"] = out["target_text"]
    cols = ["eval_id", "dataset", "pair_type", "product_id", "source",
            "reference", "max_new", "tokens_est"]
    return out[cols].reset_index(drop=True)


def make_eval_tiki(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    test = test[test["qty_status"].fillna("qty_ok") == "qty_ok"]
    rows = []
    quotas = {"title": 300, "attribute": 300}
    for pair_type, quota in quotas.items():
        part = test[(test["pair_type"] == pair_type) & (test["tokens_est"] <= 1024)]
        n = min(quota, len(part))
        rows.append(part.sample(n=n, random_state=seed))
    out = pd.concat(rows, ignore_index=True)
    out["dataset"] = "tiki"
    out["eval_id"] = [f"tikiv1:{i:05d}" for i in range(len(out))]
    out["max_new"] = out["pair_type"].map({"title": 64, "attribute": 256})
    out["source"] = out["source_text"]
    # Tiki không có reference người: dùng COMET-QE, không dùng target_text.
    out["reference"] = ""
    cols = ["eval_id", "dataset", "pair_type", "product_id", "source",
            "reference", "max_new", "tokens_est"]
    return out[cols].reset_index(drop=True)


def make_train_view(path: Path, max_seq: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    train = df[df["split"] == "train"].copy()
    train = train[(train["tokens_est"] >= 2) & (train["tokens_est"] <= max_seq)]
    train = train[train["target_text"].astype(str).str.strip().ne("")]
    if "domain" in train.columns:
        train["group"] = "tiki"
        is_1688 = train["domain"] == "1688"
        train.loc[is_1688, "group"] = train.loc[is_1688, "pair_type"].map(
            {"title": "title_1688", "attribute": "attr_1688",
             "description": "desc_1688"})
    else:
        train["group"] = train["pair_type"].map(
            {"title": "title_1688", "attribute": "attr_1688",
             "description": "desc_1688"}).fillna("tiki")
    train["eff_tokens"] = train["tokens_est"].clip(lower=1).astype(float)
    return train[["pair_id", "domain", "pair_type", "product_id", "group",
                  "source_text", "target_text", "key_vi", "tokens_est",
                  "eff_tokens"]].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/cleaned/pairs_dedup")
    ap.add_argument("--out", default="data/cleaned/weight_eval")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()

    pairs, out = Path(args.pairs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d1688 = pd.read_parquet(pairs / "1688.parquet")
    eval_1688 = make_eval_1688(d1688, args.seed)
    eval_1688.to_parquet(out / "eval_1688.parquet", index=False)

    views = {}
    for model in ("0.6b", "1.7b", "4b", "8b"):
        path = pairs / f"tiki_qwen3-{model}.parquet"
        tiki = pd.read_parquet(path)
        if model == "4b":
            eval_tiki = make_eval_tiki(tiki, args.seed)
            eval_tiki.to_parquet(out / "eval_tiki.parquet", index=False)
        train = pd.concat([make_train_view(pairs / "1688.parquet", args.max_seq),
                           make_train_view(path, args.max_seq)],
                          ignore_index=True)
        train.to_parquet(out / f"train_view_{model}.parquet", index=False)
        mass = train.groupby("group")["eff_tokens"].sum()
        views[model] = {
            "rows": int(len(train)),
            "tokens": round(float(mass.sum())),
            "group_tokens": {k: round(float(v)) for k, v in mass.items()},
            "group_share": {k: round(float(v / mass.sum()), 4) for k, v in mass.items()},
            "group_rows": {k: int(v) for k, v in train.groupby("group").size().items()},
        }

    meta = {
        "max_seq": args.max_seq,
        "seed": args.seed,
        "eval_1688": {
            "rows": int(len(eval_1688)),
            "by_type": {k: int(v) for k, v in eval_1688.groupby("pair_type").size().items()},
        },
        "eval_tiki": {
            "rows": int(len(eval_tiki)),
            "by_type": {k: int(v) for k, v in eval_tiki.groupby("pair_type").size().items()},
        },
        "views": views,
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
