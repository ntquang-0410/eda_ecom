#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedup tập A (title+attribute), giữ bản raw để audit.

Quy tắc:
  - dedup exact pair (source_text, target_text) sau khi strip/normalize khoảng trắng
  - nếu cùng cặp nằm ở nhiều split, ưu tiên giữ train > dev > test
  - không dedup source-only (cùng source khác target là biến thể hợp lệ)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/cleaned/weight_splits_v2/A_title_attribute.parquet"
RAW = ROOT / "data/cleaned/weight_splits_v2/A_title_attribute_raw.parquet"
OUT = ROOT / "data/cleaned/weight_splits_v2/A_title_attribute.parquet"
META = ROOT / "data/cleaned/weight_splits_v2/meta_A_dedup.json"
SPACE = re.compile(r"\s+")
PRIORITY = {"train": 0, "dev": 1, "test": 2}


def norm(text) -> str:
    return SPACE.sub(" ", str(text or "")).strip()


def main() -> int:
    if not RAW.exists():
        RAW.write_bytes(SRC.read_bytes())
    raw = pd.read_parquet(RAW).copy()
    df = raw.copy()
    df["_src"] = df["source_text"].map(norm)
    df["_tgt"] = df["target_text"].map(norm)
    df["_rank"] = df["split"].map(PRIORITY).fillna(9)
    df["dup_count_raw"] = df.groupby(["_src", "_tgt"])["pair_id"].transform("size")

    groups = df.groupby(["_src", "_tgt"])
    cross_split = int((groups["split"].nunique() > 1).sum())
    exact_extra = int(df.duplicated(["_src", "_tgt"]).sum())
    source_extra = int(df.duplicated(["source_text"]).sum())

    dedup = (df.sort_values(["_rank", "pair_id"])
             .drop_duplicates(["_src", "_tgt"], keep="first")
             .drop(columns=["_src", "_tgt", "_rank"])
             .reset_index(drop=True))
    dedup["dup_count"] = dedup["dup_count_raw"].astype(int)
    dedup["dedup_rule"] = "exact_pair_train_first"

    dedup.to_parquet(OUT, index=False)
    meta = {
        "raw_rows": int(len(raw)),
        "dedup_rows": int(len(dedup)),
        "removed": int(len(raw) - len(dedup)),
        "exact_pair_extra": exact_extra,
        "cross_split_duplicate_groups": cross_split,
        "source_duplicate_extra_kept": source_extra,
        "group_rows": {k: int(v) for k, v in dedup.groupby("group").size().items()},
        "split": {k: int(v) for k, v in dedup["split"].value_counts().items()},
        "rule": "exact pair, train > dev > test",
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
