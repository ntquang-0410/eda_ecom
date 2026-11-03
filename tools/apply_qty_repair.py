#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge kết quả repair quantity vào unit dataset + gán qty_weight.

- Dòng repair good (validate lại bằng compare): thay target, qty_status=qty_ok,
  qty_repaired=1.
- Dòng còn lỗi: giữ target, qty_status=qty_mismatch, weight theo tầng A/B/C.
- Các trạng thái khác: weight 1.0 / 0.7 / 0.5 theo quy ước.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from cado_core.quantity_norm import compare, extract_quantities

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
PEER = {"qwen3-0.6b": "qwen3-8b", "qwen3-1.7b": "qwen3-8b",
        "qwen3-4b": "qwen3-8b", "qwen3-8b": "qwen3-4b"}


def ngrams(text: str, n: int = 2) -> set[str]:
    value = re.sub(r"\s+", "", str(text or "")).lower()
    if len(value) < n:
        return {value} if value else set()
    return {value[i:i + n] for i in range(len(value) - n + 1)}


def similarity(a: str, b: str) -> float:
    sa, sb = ngrams(a), ngrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def tier_weight(model: str, damage: float, consensus: float) -> float:
    if damage > 0.5:
        return 0.1
    high = consensus >= 0.6
    if model == "qwen3-0.6b":
        return 0.4 if high else 0.2
    if model == "qwen3-1.7b":
        return 0.5 if high else 0.3
    return 0.6 if high else 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    ap.add_argument("--repair", default=".artifacts/qty_repair_out/qty_repair.parquet")
    args = ap.parse_args()

    repair = pd.read_parquet(args.repair)

    frames = {}
    for model in MODELS:
        frames[model] = pd.read_parquet(f"{args.root}/{model}/units.parquet")

    summary = {}
    for model in MODELS:
        df = frames[model].copy()
        peer = frames[PEER[model]].set_index("pair_id")["target_text_bt"]
        good = (repair[(repair["good"]) & (repair["model_key"] == model)]
                .drop_duplicates("pair_id", keep="first").set_index("pair_id"))

        repaired = 0
        weights = []
        statuses = []
        for index, row in df.iterrows():
            pair_id = str(row["pair_id"])
            target = row["target_text_bt"]
            repaired_flag = 0

            # 1. áp repair nếu có và validate lại
            if pair_id in good.index:
                new_target = str(good.loc[pair_id, "new_target"] or "")
                result = compare(row["source_text_bt"], new_target, "vi", "zh")
                if new_target and result["match"]:
                    target = new_target
                    repaired_flag = 1
                    repaired += 1

            # 2. trạng thái
            if row["qty_status"] == "skipped":
                status = "skipped"
            elif repaired_flag:
                status = "qty_ok"
            elif row["qty_status"] == "qty_ok":
                status = "qty_ok"
            elif row["qty_status"] in ("qty_resolved",):
                status = "qty_resolved"
            elif row["qty_status"] == "qty_weak":
                status = "qty_weak"
            elif row["qty_status"] == "qty_review":
                status = "qty_review"
            else:
                status = "qty_mismatch"

            # 3. weight
            if status == "skipped":
                weight = 0.0
            elif status in ("qty_ok", "qty_resolved"):
                weight = 1.0
            elif status == "qty_weak":
                weight = 0.7
            elif status == "qty_review":
                weight = 0.5
            else:
                result = compare(row["source_text_bt"], target, "vi", "zh")
                src_q = len(extract_quantities(row["source_text_bt"], "vi")) or 1
                tgt_q = len(extract_quantities(target, "zh")) or 1
                damage = (len(result["missing"]) + len(result["extra"])) / (src_q + tgt_q)
                consensus = similarity(target, peer.get(pair_id, ""))
                weight = tier_weight(model, min(1.0, damage), consensus)

            df.at[index, "target_text_bt"] = target
            df.at[index, "qty_repaired"] = repaired_flag
            df.at[index, "qty_status"] = status
            weights.append(weight)

        df["qty_weight"] = weights
        df.to_parquet(f"{args.root}/{model}/units.parquet", index=False)

        counts = df["qty_status"].value_counts().to_dict()
        summary[model] = {
            "repaired": int(repaired),
            "status": counts,
            "weight_mean": round(float(df["qty_weight"].mean()), 3),
            "weight_1_0_rows": int((df["qty_weight"] == 1.0).sum()),
        }
        print(model, json.dumps(summary[model], ensure_ascii=False))

    out = Path(args.root) / "reports" / "qty_repair_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
