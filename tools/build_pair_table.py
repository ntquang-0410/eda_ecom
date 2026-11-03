#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng pair table thống nhất từ 1688 + Tiki.

Schema:
  pair_id, domain, pair_type, product_id, split,
  source_text, target_text, key_vi, key_zh,
  origin, qc_status, qty_status, qty_weight,
  src_chars, tgt_chars, tokens_est, dup_count, base_weight

1688 split: hash product_id -> 90/5/5.
Tiki split: lấy từ units.
Token ước lượng: VI/3.6 + ZH/1.6 ký tự (sẽ thay bằng tokenizer thật khi train).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
SPACE = re.compile(r"\s+")


def compact_len(text) -> int:
    return len(SPACE.sub("", str(text or "")))


def token_est(src: str, tgt: str) -> float:
    return compact_len(src) / 3.6 + compact_len(tgt) / 1.6


def split_1688(product_id: str) -> str:
    value = int(hashlib.md5(str(product_id).encode()).hexdigest()[:8], 16) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "dev"
    return "test"


def build_1688() -> pd.DataFrame:
    b = pd.read_parquet("data/cleaned/1688/bilingual_1688_clean.parquet")
    a = pd.read_parquet("data/cleaned/1688/attributes_aligned.parquet")

    rows = []
    for row in b.itertuples():
        pid = str(row.product_id)
        split = split_1688(pid)
        rows.append({
            "pair_id": f"1688:{pid}:title", "domain": "1688", "pair_type": "title",
            "product_id": pid, "split": split,
            "source_text": row.title_vi_clean, "target_text": row.title_zh_clean,
            "key_vi": "", "key_zh": "", "origin": "human", "qc_status": "ok",
            "qty_status": "", "qty_weight": 1.0,
            "flag_stuffing": bool(row.flag_stuffing_new),
            "flag_truncated": bool(row.flag_truncated_new),
        })
        rows.append({
            "pair_id": f"1688:{pid}:desc", "domain": "1688", "pair_type": "description",
            "product_id": pid, "split": split,
            "source_text": row.description_vi_clean, "target_text": row.description_zh_clean,
            "key_vi": "", "key_zh": "", "origin": "human", "qc_status": "ok",
            "qty_status": "", "qty_weight": 1.0,
            "flag_stuffing": False, "flag_truncated": False,
        })

    for row in a.itertuples():
        attrs = row.attributes_aligned
        if attrs is None:
            continue
        if isinstance(attrs, np.ndarray):
            attrs = list(attrs)
        pid = str(row.product_id)
        split = split_1688(pid)
        for index, attr in enumerate(attrs):
            if attr["status"] == "unmatched":
                continue
            if len(attr["values_vi"]) == 0 or len(attr["values_zh"]) == 0:
                continue
            rows.append({
                "pair_id": f"1688:{pid}:attr:{index}", "domain": "1688", "pair_type": "attribute",
                "product_id": pid, "split": split,
                "source_text": " ; ".join(map(str, attr["values_vi"])),
                "target_text": " ; ".join(map(str, attr["values_zh"])),
                "key_vi": str(attr["name_vi"] or ""), "key_zh": str(attr["name_zh"] or ""),
                "origin": "human", "qc_status": "ok", "qty_status": "", "qty_weight": 1.0,
                "flag_stuffing": False, "flag_truncated": False,
            })

    df = pd.DataFrame(rows)
    df["src_chars"] = df["source_text"].map(compact_len)
    df["tgt_chars"] = df["target_text"].map(compact_len)
    df["tokens_est"] = [token_est(s, t) for s, t in zip(df["source_text"], df["target_text"])]
    df["dup_count"] = df.groupby(["source_text", "target_text"])["pair_id"].transform("size")
    base = df["pair_type"].map({"title": 1.0, "attribute": 0.6, "description": 0.8}).fillna(1.0)
    df["base_weight"] = base * np.where(df["flag_stuffing"], 0.7, 1.0) * np.where(df["flag_truncated"], 0.5, 1.0)
    return df


def build_tiki(model: str) -> pd.DataFrame:
    units = pd.read_parquet(f"data/cleaned/bt_tiki/{model}/units.parquet")
    units = units[units["qc_status"] == "ok"].copy()
    df = pd.DataFrame({
        "pair_id": units["pair_id"].astype(str),
        "domain": "tiki",
        "pair_type": np.where(units["unit_type"] == "title", "title", "attribute"),
        "product_id": units["product_id"].astype(str),
        "split": units["split"],
        "source_text": units["source_text_bt"],
        "target_text": units["target_text_bt"],
        "key_vi": units["key_vi"].fillna(""),
        "key_zh": units["key_zh_effective"].fillna(""),
        "origin": "mt",
        "qc_status": units["qc_status"],
        "qty_status": units["qty_status"].fillna(""),
        "qty_weight": units["qty_weight"].fillna(1.0),
        "flag_stuffing": False,
        "flag_truncated": False,
    })
    df["src_chars"] = df["source_text"].map(compact_len)
    df["tgt_chars"] = df["target_text"].map(compact_len)
    df["tokens_est"] = [token_est(s, t) for s, t in zip(df["source_text"], df["target_text"])]
    df["dup_count"] = df.groupby(["source_text", "target_text"])["pair_id"].transform("size")
    df["base_weight"] = df["qty_weight"].astype(float)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned/pairs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d1688 = build_1688()
    d1688.to_parquet(out / "1688.parquet", index=False)
    print("1688:", len(d1688), d1688["pair_type"].value_counts().to_dict())

    summary = {"1688": {"rows": int(len(d1688)),
                        "by_type": d1688["pair_type"].value_counts().to_dict(),
                        "split": d1688["split"].value_counts().to_dict()}}
    for model in MODELS:
        dt = build_tiki(model)
        dt.to_parquet(out / f"tiki_{model}.parquet", index=False)
        summary[model] = {"rows": int(len(dt)),
                          "by_type": dt["pair_type"].value_counts().to_dict(),
                          "split": dt["split"].value_counts().to_dict()}
        print(model, len(dt), dt["pair_type"].value_counts().to_dict())

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
