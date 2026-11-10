#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn bị input repair cho các dòng quantity mismatch.

Mỗi dòng: model, pair_id, source, key, target cũ, danh sách đại lượng lệch.
Kernel sẽ dịch lại với prompt nhấn mạnh giữ số/đơn vị, rồi validate bằng
chính `compare` trước khi chấp nhận.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cado_core.quantity_norm import compare

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    ap.add_argument("--out", default="/tmp/opencode/qty_repair_input")
    args = ap.parse_args()

    rows = []
    for model in MODELS:
        units = pd.read_parquet(f"{args.root}/{model}/units.parquet")
        bad = units[units["qty_status"] == "qty_mismatch"]
        for row in bad.itertuples():
            result = compare(row.source_text_bt, row.target_text_bt, "vi", "zh")
            rows.append({
                "model_key": model,
                "pair_id": str(row.pair_id),
                "source_text_bt": str(row.source_text_bt or ""),
                "key_vi": str(row.key_vi or ""),
                "key_zh_effective": str(row.key_zh_effective or ""),
                "previous_target": str(row.target_text_bt or ""),
                "missing": json.dumps(result["missing"], ensure_ascii=False),
                "extra": json.dumps(result["extra"], ensure_ascii=False),
            })
        print(f"{model}: {len(bad):,} dòng mismatch")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "qty_repair_input.parquet", index=False)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Tiki Qty Repair Input",
        "id": "ntquang0410/tiki-qty-repair-input",
        "licenses": [{"name": "CC0-1.0"}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total {len(df):,} -> {out/'qty_repair_input.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
