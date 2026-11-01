#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn hoá ngoặc trang trí (【】〖〗〔〕丨) trong dataset Tiki BT.

Áp dụng cho source_text_bt / target_text_bt / key_vi / key_zh_effective ở cả
full, train và units. `raw_output` giữ nguyên để audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cado_core.text import normalize_brackets

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
FILES = ("full.parquet", "train.parquet", "units.parquet")
COLUMNS = ("source_text_bt", "target_text_bt", "key_vi", "key_zh_effective")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    args = ap.parse_args()

    root = Path(args.root)
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for filename in FILES:
            path = root / model / filename
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            changed = 0
            for column in COLUMNS:
                if column not in df.columns:
                    continue
                before = df[column].fillna("").astype(str)
                after = before.map(normalize_brackets)
                changed += int((before != after).sum())
                df[column] = after
            df.to_parquet(path, index=False)
            summary[model][filename] = changed
        print(model, summary[model])

    (root / "reports" / "bracket_normalization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
