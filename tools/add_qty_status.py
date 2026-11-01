#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thêm cột qty_status vào dataset unit-level (tập chính).

Với mỗi unit (model, pair):
  qty_ok       : đại lượng vật lý khớp sau chuẩn hoá
  qty_resolved : weak nhưng ngữ cảnh/đơn vị đã xác nhận tương đương
  qty_review   : weak, cần người quyết
  qty_mismatch : cùng thứ nguyên nhưng giá trị khác (strong)
  qty_weak     : weak nhưng chưa có quyết định

Nguồn quyết định: data/cleaned/quantity_weak_needs_decision.csv (cột decision).
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
    ap.add_argument("--decisions", default="data/cleaned/quantity_weak_needs_decision.csv")
    args = ap.parse_args()

    decisions = {}
    if Path(args.decisions).exists():
        d = pd.read_csv(args.decisions, encoding="utf-8-sig")
        for row in d.itertuples():
            if isinstance(row.decision, str) and row.decision.strip():
                decisions[str(row.id)] = row.decision.strip()
        print("decisions:", len(decisions))

    summary = {}
    for model in MODELS:
        path = Path(args.root) / model / "units.parquet"
        units = pd.read_parquet(path)
        status, decision_col = [], []

        for row in units.itertuples():
            if row.qc_status != "ok":
                status.append("skipped")
                decision_col.append("")
                continue
            result = compare(row.source_text_bt, row.target_text_bt, "vi", "zh")
            decided = decisions.get(str(row.pair_id), "")
            if result["match"]:
                value = "qty_ok"
            elif result.get("strong_mismatch"):
                value = "qty_mismatch"
            elif decided == "keep":
                value = "qty_resolved"
            elif decided:
                value = "qty_review"
            else:
                value = "qty_weak"
            status.append(value)
            decision_col.append(decided)

        units["qty_status"] = status
        units["qty_decision"] = decision_col
        units.to_parquet(path, index=False)

        counts = pd.Series(status).value_counts().to_dict()
        summary[model] = counts
        print(model, counts)

    out = Path(args.root) / "reports" / "qty_status_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
