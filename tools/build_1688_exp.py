#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng tập thực nghiệm 1688-only cho so sánh 4 hệ dịch.

- Chỉ dùng title + attribute (bỏ description dài, tránh bất công cho SMT/NMT).
- Split có sẵn: train/dev/test theo product_id.
- test_small: 500 title + 500 attribute, seed cố định, dùng chung cho cả 4 hệ.

Output:
  data/cleaned/1688_exp/
    train.parquet, dev.parquet, test.parquet, test_small.parquet
    train.vi, train.zh, dev.vi, dev.zh, test.vi, test.zh,
    test_small.vi, test_small.zh
    test_small.raw.vi, test_small.raw.zh
    meta.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SPACE = re.compile(r"\s+")
PUNCT = re.compile(r"([,.;:!?，。；：！？()（）\[\]【】])")


def clean(text) -> str:
    return SPACE.sub(" ", str(text or "")).strip()


def tok_vi(text: str) -> str:
    value = clean(text)
    value = PUNCT.sub(r" \1 ", value)
    return SPACE.sub(" ", value).strip()


def tok_zh(text: str) -> str:
    value = clean(text)
    return " ".join(list(value))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/cleaned/pairs_dedup/1688.parquet")
    ap.add_argument("--out", default="data/cleaned/1688_exp")
    args = ap.parse_args()

    df = pd.read_parquet(args.pairs)
    df = df[df["pair_type"].isin(["title", "attribute"])].copy()
    df["source_text"] = df["source_text"].map(clean)
    df["target_text"] = df["target_text"].map(clean)
    df = df[(df["source_text"] != "") & (df["target_text"] != "")].reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        part.to_parquet(out / f"{split}.parquet", index=False)
        (out / f"{split}.vi").write_text("\n".join(part["source_text"]), encoding="utf-8")
        (out / f"{split}.zh").write_text("\n".join(part["target_text"]), encoding="utf-8")
        (out / f"{split}.tok.vi").write_text("\n".join(part["source_text"].map(tok_vi)), encoding="utf-8")
        (out / f"{split}.tok.zh").write_text("\n".join(part["target_text"].map(tok_zh)), encoding="utf-8")
        print(split, len(part), part["pair_type"].value_counts().to_dict())

    test = df[df["split"] == "test"].reset_index(drop=True)
    small = pd.concat([
        test[test["pair_type"] == "title"].sample(n=min(500, (test["pair_type"] == "title").sum()), random_state=42),
        test[test["pair_type"] == "attribute"].sample(n=min(500, (test["pair_type"] == "attribute").sum()), random_state=42),
    ]).reset_index(drop=True)
    small.to_parquet(out / "test_small.parquet", index=False)
    (out / "test_small.vi").write_text("\n".join(small["source_text"]), encoding="utf-8")
    (out / "test_small.zh").write_text("\n".join(small["target_text"]), encoding="utf-8")
    (out / "test_small.tok.vi").write_text("\n".join(small["source_text"].map(tok_vi)), encoding="utf-8")
    (out / "test_small.tok.zh").write_text("\n".join(small["target_text"].map(tok_zh)), encoding="utf-8")
    print("test_small", len(small), small["pair_type"].value_counts().to_dict())

    meta = {
        "train": int((df["split"] == "train").sum()),
        "dev": int((df["split"] == "dev").sum()),
        "test": int((df["split"] == "test").sum()),
        "test_small": int(len(small)),
        "types": ["title", "attribute"],
        "tokenization": {"vi": "whitespace+punct", "zh": "character-level"},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "1688 Translation Experiment",
        "id": "ntquang0410/tiki-1688-exp",
        "licenses": [{"name": "CC0-1.0"}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
