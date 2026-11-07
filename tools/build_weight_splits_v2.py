#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tạo 2 bài toán weight từ dữ liệu TRƯỚC EDA/processing.

A) title + attribute:
   1688 title, 1688 attribute, Tiki title, Tiki attribute

B) title + description:
   1688 title, 1688 description, Tiki title, Tiki description
   Tiki description lấy từ raw BT predictions (trước QC/EDA).

Không sửa `pairs_dedup`; output là dataset riêng.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SPACE = re.compile(r"\s+")


def compact_len(text) -> int:
    return len(SPACE.sub("", str(text or "")))


def token_est(src, tgt) -> float:
    return compact_len(src) / 3.6 + compact_len(tgt) / 1.6


def load_a(pairs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(pairs / "1688.parquet")
    t = pd.read_parquet(pairs / "tiki_qwen3-4b.parquet")
    a = pd.concat([
        d[d["pair_type"].isin(["title", "attribute"])],
        t[t["pair_type"].isin(["title", "attribute"])],
    ], ignore_index=True)
    a["group"] = [
        ("1688_" if dom == "1688" else "tiki_") + pt
        for dom, pt in zip(a["domain"], a["pair_type"])
    ]

    e1688 = pd.read_parquet("data/cleaned/weight_eval/eval_1688.parquet")
    etiki = pd.read_parquet("data/cleaned/weight_eval/eval_tiki.parquet")
    e = pd.concat([
        e1688[e1688["pair_type"].isin(["title", "attribute"])],
        etiki,
    ], ignore_index=True)
    return a, e


def load_b(pairs: Path, tiki_desc: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(pairs / "1688.parquet")
    t = pd.read_parquet(pairs / "tiki_qwen3-4b.parquet")
    b = pd.concat([
        d[d["pair_type"].isin(["title", "description"])],
        t[t["pair_type"] == "title"],
    ], ignore_index=True)
    b["group"] = [
        ("1688_" if dom == "1688" else "tiki_") + pt
        for dom, pt in zip(b["domain"], b["pair_type"])
    ]

    if tiki_desc is not None and tiki_desc.exists():
        pred = pd.read_parquet(tiki_desc).drop_duplicates("pair_id")
        desc = pd.DataFrame({
            "pair_id": pred["pair_id"],
            "domain": "tiki",
            "pair_type": "description",
            "product_id": pred["product_id"].astype(str),
            "split": pred["split"],
            "source_text": pred["source_text"],
            "target_text": pred["target_text_bt"].fillna(""),
            "key_vi": "", "key_zh": "",
            "origin": "bt",
            "qc_status": "pre_eda",
            "qty_status": "", "qty_weight": 1.0,
            "token_est": pred["tokens_est"],
            "token_source": "raw_bt",
        })
        desc["group"] = "tiki_description"
        b = pd.concat([b, desc], ignore_index=True)

    e1688 = pd.read_parquet("data/cleaned/weight_eval/eval_1688.parquet")
    etiki = pd.read_parquet("data/cleaned/weight_eval/eval_tiki.parquet")
    edesc = pd.read_parquet("data/cleaned/weight_eval/eval_tiki_desc.parquet")
    e = pd.concat([
        e1688[e1688["pair_type"].isin(["title", "description"])],
        etiki[etiki["pair_type"] == "title"],
        edesc,
    ], ignore_index=True)
    return b, e


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source_text" not in df.columns and "source" in df.columns:
        df["source_text"] = df["source"]
    if "target_text" not in df.columns and "reference" in df.columns:
        df["target_text"] = df["reference"]
    df["source_text"] = df["source_text"].astype(str)
    df["target_text"] = df["target_text"].fillna("").astype(str)
    if "tokens_est" not in df.columns:
        df["tokens_est"] = [token_est(s, t) for s, t in zip(df["source_text"], df["target_text"])]
    df["tokens_est"] = pd.to_numeric(df["tokens_est"], errors="coerce").fillna(0.0)
    df["eff_tokens"] = df["tokens_est"].clip(lower=1.0)
    df["src_chars"] = df["source_text"].map(compact_len)
    df["tgt_chars"] = df["target_text"].map(compact_len)
    return df


CONFIGS_A = {
    "natural": None,
    "balanced": {"1688_title": 0.25, "1688_attribute": 0.25,
                 "tiki_title": 0.25, "tiki_attribute": 0.25},
    "title_heavy": {"1688_title": 0.40, "1688_attribute": 0.20,
                    "tiki_title": 0.25, "tiki_attribute": 0.15},
    "attr_heavy": {"1688_title": 0.15, "1688_attribute": 0.45,
                   "tiki_title": 0.15, "tiki_attribute": 0.25},
    "tiki_heavy": {"1688_title": 0.10, "1688_attribute": 0.20,
                   "tiki_title": 0.30, "tiki_attribute": 0.40},
}
CONFIGS_B = {
    "natural": None,
    "balanced": {"1688_title": 0.25, "1688_description": 0.25,
                 "tiki_title": 0.25, "tiki_description": 0.25},
    "title_heavy": {"1688_title": 0.40, "1688_description": 0.10,
                    "tiki_title": 0.40, "tiki_description": 0.10},
    "desc_heavy": {"1688_title": 0.10, "1688_description": 0.40,
                   "tiki_title": 0.10, "tiki_description": 0.40},
    "tiki_heavy": {"1688_title": 0.10, "1688_description": 0.15,
                   "tiki_title": 0.35, "tiki_description": 0.40},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/cleaned/pairs_dedup")
    ap.add_argument("--tiki-desc", default="/content/bt_desc_out/bt_desc_v1/qwen3-4b/predictions.parquet")
    ap.add_argument("--out", default="data/cleaned/weight_splits_v2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = Path(args.pairs)

    a, ea = load_a(pairs)
    b, eb = load_b(pairs, Path(args.tiki_desc) if args.tiki_desc else None)
    a, b, ea, eb = normalize(a), normalize(b), normalize(ea), normalize(eb)

    a.to_parquet(out / "A_title_attribute.parquet", index=False)
    b.to_parquet(out / "B_title_description.parquet", index=False)
    ea.to_parquet(out / "eval_A.parquet", index=False)
    eb.to_parquet(out / "eval_B.parquet", index=False)

    meta = {}
    for name, df, configs in (("A", a, CONFIGS_A), ("B", b, CONFIGS_B)):
        mass = df.groupby("group")["eff_tokens"].sum().sort_values(ascending=False)
        meta[name] = {
            "rows": int(len(df)),
            "tokens_million": round(float(mass.sum()) / 1e6, 3),
            "group_tokens": {k: round(float(v)) for k, v in mass.items()},
            "group_share": {k: round(float(v / mass.sum()), 4) for k, v in mass.items()},
            "split": {k: int(v) for k, v in df["split"].value_counts().items()},
            "configs": configs,
        }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
