#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mở rộng 4 tập BT từ cache-level (29.446) ra unit-level (45.709).

Dataset BT hiện lưu theo `cache_key` (một bản dịch dùng cho nhiều unit trùng
source). Để fine-tune cần pair-level: mỗi unit là một dòng, kèm `pair_id`,
`product_id`, `split`, `unit_type`.

Script recompute `cache_key` cho từng unit MT bằng đúng hàm của Plan A
(`cado_core.tiki_bt_prepare`), join sang target của từng model, rồi xuất:

  data/cleaned/bt_tiki/<model>/units.parquet   45.709 dòng

Kiểm tra: 0 unit không map được, 0 pair_id trùng, đủ 13.200 product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cado_core.text import normalize_text
from cado_core.tiki_bt_prepare import PLAN_A_VERSION
from cado_core.tiki_units import normalize_key

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")

TARGET_COLUMNS = (
    "target_text_bt", "model_id", "prompt_version", "json_ok",
    "qc_status", "repair_pass", "attempt", "latency_ms",
)


def cache_key(source_text: str, key_norm: str, value_class: str, route: str) -> str:
    payload = "\x1f".join([
        normalize_text(source_text, collapse_whitespace=True).strip().lower(),
        normalize_key(key_norm),
        str(value_class or ""),
        str(route or ""),
        PLAN_A_VERSION,
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-root", default="data/cleaned/bt_tiki")
    ap.add_argument("--plan-a", default=".artifacts/plan_a")
    args = ap.parse_args()

    bt_root = Path(args.bt_root)
    plan_a = Path(args.plan_a)

    units = pd.read_parquet(plan_a / "tiki_bt_units_plan_a.parquet")
    units = units[units["bt_method"] == "mt"].copy()

    keys = []
    for row in units.itertuples():
        source = row.text_vi if row.unit_type == "title" else row.value_vi
        keys.append(cache_key(str(source or ""), str(row.key_norm or ""),
                              str(row.value_class or ""), "mt"))
    recomputed_mismatch = int((units["cache_key"].astype(str) != pd.Series(keys, index=units.index)).sum())
    units["cache_key"] = keys

    units["source_text_bt"] = units["text_vi"]
    attribute = units["unit_type"] == "attribute"
    units.loc[attribute, "source_text_bt"] = units.loc[attribute, "value_vi"]
    units["source_text_bt"] = units["source_text_bt"].map(
        lambda value: normalize_text(value, collapse_whitespace=True).strip()
    )

    base_cols = [
        "pair_id", "product_id", "split", "unit_type", "unit_index", "value_index",
        "key_vi", "key_zh", "key_zh_effective", "value_class", "source_text_bt", "cache_key",
    ]
    units = units[base_cols].copy()

    report = {}
    for model in MODELS:
        full = pd.read_parquet(bt_root / model / "full.parquet")
        payload = full[["cache_key", *TARGET_COLUMNS]].drop_duplicates("cache_key")
        merged = units.merge(payload, on="cache_key", how="left", validate="many_to_one")

        missing = int(merged["target_text_bt"].isna().sum() + merged["qc_status"].isna().sum())
        out = merged[base_cols + list(TARGET_COLUMNS)]
        (bt_root / model).mkdir(parents=True, exist_ok=True)
        out_path = bt_root / model / "units.parquet"
        out.to_parquet(out_path, index=False)

        report[model] = {
            "units": int(len(out)),
            "unmapped": missing,
            "cache_key_recomputed_mismatch": recomputed_mismatch,
            "dup_pair_id": int(out["pair_id"].duplicated().sum()),
            "products": int(out["product_id"].nunique()),
            "split": out["split"].value_counts().to_dict(),
            "unit_type": out["unit_type"].value_counts().to_dict(),
            "usable": int((out["qc_status"] != "still_untranslated").sum()),
            "file": str(out_path),
        }
        print(model, json.dumps(report[model], ensure_ascii=False))

    (bt_root / "bt_units_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", bt_root / "bt_units_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
