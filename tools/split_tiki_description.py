#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tách description Tiki thành attribute (key-value) và description (prose).

Nguồn: data/cleaned/tiki_vi_cado.parquet
  - attributes_vi / desc_vi_cleaned  -> attribute
  - desc_prose_vi_cleaned            -> description

Lưu ý: Tiki không có description_zh. Target để trống; đây là split source-only
cho eval QE và cho bước BT sau. Không trộn với target BT title/attribute hiện có.
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


def tokens_est(src: str, tgt: str = "") -> float:
    return compact_len(src) / 3.6 + compact_len(tgt) / 1.6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/cleaned/tiki_vi_cado.parquet")
    ap.add_argument("--pairs", default="data/cleaned/pairs_dedup/tiki_qwen3-4b.parquet")
    ap.add_argument("--out", default="data/cleaned/tiki_desc_split")
    args = ap.parse_args()

    raw = pd.read_parquet(args.raw)
    pairs = pd.read_parquet(args.pairs)
    split_map = (pairs.drop_duplicates("product_id")
                 .set_index(pairs.drop_duplicates("product_id")["product_id"].astype(str))["split"]
                 .to_dict())

    attr_rows, desc_rows = [], []
    for row in raw.itertuples():
        pid = str(row.product_id)
        split = split_map.get(pid, "train")
        attrs = row.attributes_vi
        if attrs is not None:
            for index, attr in enumerate(attrs):
                values = [str(v).strip() for v in attr.get("values", []) if str(v).strip()]
                if not values:
                    continue
                source = " ; ".join(values)
                attr_rows.append({
                    "pair_id": f"tiki:{pid}:attr:{index}",
                    "domain": "tiki", "pair_type": "attribute",
                    "product_id": pid, "split": split,
                    "source_text": source, "target_text": "",
                    "key_vi": str(attr.get("name") or ""), "key_zh": "",
                    "origin": "human_vi", "qc_status": "ok",
                    "src_chars": compact_len(source), "tgt_chars": 0,
                    "tokens_est": tokens_est(source),
                })
        prose = str(row.desc_prose_vi_cleaned or "").strip()
        if prose:
            desc_rows.append({
                "pair_id": f"tiki:{pid}:desc",
                "domain": "tiki", "pair_type": "description",
                "product_id": pid, "split": split,
                "source_text": prose, "target_text": "",
                "key_vi": "", "key_zh": "",
                "origin": "human_vi", "qc_status": "ok",
                "src_chars": compact_len(prose), "tgt_chars": 0,
                "tokens_est": tokens_est(prose),
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    attr = pd.DataFrame(attr_rows)
    desc = pd.DataFrame(desc_rows)
    combined = pd.concat([attr, desc], ignore_index=True)

    attr.to_parquet(out / "attribute.parquet", index=False)
    desc.to_parquet(out / "description.parquet", index=False)
    combined.to_parquet(out / "tiki_desc_split.parquet", index=False)

    summary = {
        "products": int(raw["product_id"].nunique()),
        "attribute_rows": int(len(attr)),
        "description_rows": int(len(desc)),
        "description_tokens": round(float(desc["tokens_est"].sum()), 1),
        "description_chars": {
            "mean": round(float(desc["src_chars"].mean()), 1),
            "p50": int(desc["src_chars"].quantile(0.5)),
            "p95": int(desc["src_chars"].quantile(0.95)),
            "max": int(desc["src_chars"].max()),
        },
        "description_tokens_est": {
            "mean": round(float(desc["tokens_est"].mean()), 1),
            "p95": round(float(desc["tokens_est"].quantile(0.95)), 1),
            "max": round(float(desc["tokens_est"].max()), 1),
        },
        "split": {k: int(v) for k, v in desc["split"].value_counts().items()},
        "note": "Tiki description_zh is empty; target_text is intentionally empty.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
