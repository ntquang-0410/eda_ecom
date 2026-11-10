#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Khảo sát trọng số hợp lý cho các dòng quantity mismatch.

Câu hỏi: dòng strong mismatch nên có weight bao nhiêu?

Khảo sát offline (không train):
  1. damage        = số đại lượng lệch / tổng đại lượng nguồn
  2. consensus     = độ giống target của model với target 8B (hoặc 4B nếu là 8B)
  3. length_ratio  = độ dài target / source
  4. so sánh với nhóm qty_ok làm baseline

Từ đó chia tầng và đề xuất weight.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cado_core.quantity_norm import compare, extract_quantities

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    ap.add_argument("--out", default="data/cleaned/qty_weight_survey.json")
    args = ap.parse_args()

    frames = {}
    for model in MODELS:
        df = pd.read_parquet(f"{args.root}/{model}/units.parquet",
                             columns=["pair_id", "source_text_bt", "target_text_bt", "qty_status", "split"])
        frames[model] = df.set_index("pair_id")

    index = frames[MODELS[0]].index
    for model in MODELS[1:]:
        index = index.intersection(frames[model].index)

    report = {}
    for model in MODELS:
        df = frames[model].loc[index]
        peer = "qwen3-8b" if model != "qwen3-8b" else "qwen3-4b"
        peer_target = frames[peer].loc[index, "target_text_bt"]

        rows = []
        for pair_id, row in df.iterrows():
            if row["qty_status"] not in ("qty_ok", "qty_mismatch"):
                continue
            result = compare(row["source_text_bt"], row["target_text_bt"], "vi", "zh")
            src_q = len(extract_quantities(row["source_text_bt"], "vi")) or 1
            tgt_q = len(extract_quantities(row["target_text_bt"], "zh")) or 1
            damage = (len(result["missing"]) + len(result["extra"])) / (src_q + tgt_q)
            rows.append({
                "pair_id": pair_id,
                "status": row["qty_status"],
                "damage": min(1.0, damage),
                "consensus": similarity(row["target_text_bt"], peer_target.loc[pair_id]),
                "src_len": len(str(row["source_text_bt"] or "")),
                "tgt_len": len(str(row["target_text_bt"] or "")),
            })
        data = pd.DataFrame(rows)
        data["length_ratio"] = data["tgt_len"] / data["src_len"].replace(0, 1)

        summary = {}
        for status, part in data.groupby("status"):
            summary[status] = {
                "rows": int(len(part)),
                "damage_mean": round(float(part["damage"].mean()), 3),
                "damage_p50": round(float(part["damage"].median()), 3),
                "consensus_mean": round(float(part["consensus"].mean()), 3),
                "consensus_p10": round(float(part["consensus"].quantile(0.1)), 3),
                "length_ratio_p50": round(float(part["length_ratio"].median()), 2),
            }

        mismatch = data[data["status"] == "qty_mismatch"].copy()
        tiers = []
        if len(mismatch):
            tier_a = mismatch[(mismatch["damage"] <= 0.5) & (mismatch["consensus"] >= 0.6)]
            tier_b = mismatch[(mismatch["damage"] <= 0.5) & (mismatch["consensus"] < 0.6)]
            tier_c = mismatch[mismatch["damage"] > 0.5]
            for name, part, suggested in (
                ("A. lệch ít, đồng thuận cao", tier_a, 0.5),
                ("B. lệch ít, đồng thuận thấp", tier_b, 0.3),
                ("C. lệch nhiều (>50% đại lượng)", tier_c, 0.1),
            ):
                tiers.append({
                    "tier": name,
                    "rows": int(len(part)),
                    "share": round(len(part) / len(mismatch) * 100, 1),
                    "consensus_mean": round(float(part["consensus"].mean()), 3) if len(part) else None,
                    "damage_mean": round(float(part["damage"].mean()), 3) if len(part) else None,
                    "suggested_weight": suggested,
                })

        # thiếu / thừa / cả hai
        kinds = Counter()
        for pair_id, row in df[df["qty_status"] == "qty_mismatch"].iterrows():
            result = compare(row["source_text_bt"], row["target_text_bt"], "vi", "zh")
            if result["missing"] and result["extra"]:
                kinds["both"] += 1
            elif result["missing"]:
                kinds["missing"] += 1
            elif result["extra"]:
                kinds["extra"] += 1

        examples = {}
        if len(mismatch):
            for name, part in (("A", tier_a), ("B", tier_b), ("C", tier_c)):
                if len(part):
                    sample = part.sort_values("consensus", ascending=False).iloc[0]
                    pair_id = sample["pair_id"]
                    examples[name] = {
                        "source": str(df.loc[pair_id, "source_text_bt"])[:120],
                        "target": str(df.loc[pair_id, "target_text_bt"])[:120],
                        "consensus": round(float(sample["consensus"]), 3),
                        "damage": round(float(sample["damage"]), 3),
                    }

        report[model] = {"summary": summary, "tiers": tiers, "kinds": dict(kinds),
                         "peer": peer, "mismatch_rows": int(len(mismatch)),
                         "examples": examples}
        print(f"\n== {model} (so với {peer}) ==")
        print(pd.DataFrame(summary).T.to_string())
        print(pd.DataFrame(tiers).to_string(index=False) if tiers else "  không có mismatch")
        print("  kinds:", dict(kinds))
        for name, ex in examples.items():
            print(f"  [{name}] {ex['source'][:60]} -> {ex['target'][:60]}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
