#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Báo cáo Gemma credit AAA..C cho thực nghiệm weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

STRATA = (("1688", "title"), ("1688", "attribute"), ("1688", "description"),
          ("tiki", "title"), ("tiki", "attribute"))
CONFIGS = ("natural", "balanced", "no_desc", "tiki_heavy", "title_heavy")
GRADES = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", default="data/cleaned/judge/gemma_scores.json")
    ap.add_argument("--out", default="reports/WEIGHT_EVAL_CREDIT.md")
    args = ap.parse_args()

    data = json.loads(Path(args.gemma).read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    by_stratum = summary.get("by_stratum", {})

    rows = []
    for config in CONFIGS:
        row = {"config": config}
        for dataset, pair_type in STRATA:
            cell = by_stratum.get(f"{config}::{dataset}::{pair_type}", {})
            row[f"{dataset}_{pair_type}"] = cell.get("mean")
            row[f"{dataset}_{pair_type}_grade"] = cell.get("mean_rating")
        rows.append(row)
    table = pd.DataFrame(rows)

    lines = ["# Gemma credit AAA..C — thực nghiệm weight", "",
             "Thang: AAA=9 (tốt nhất) ... C=1 (kém nhất).", ""]
    if table.empty:
        lines.append("Chưa có dữ liệu.")
    else:
        cols = ["config"] + [f"{d}_{p}" for d, p in STRATA]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * (len(cols))]
        for row in table.to_dict("records"):
            vals = [row["config"]] + [
                f"{row[c]:.2f} ({row[c+'_grade']})" if pd.notna(row.get(c)) else "-"
                for c in cols[1:]]
            lines.append("| " + " | ".join(vals) + " |")

        lines += ["", "## Phân bố grade theo cấu hình", "",
                  "| cấu hình | " + " | ".join(GRADES) + " |",
                  "|---|" + "---:|" * len(GRADES)]
        for config in CONFIGS:
            cell = summary.get("by_config", {}).get(config, {})
            counts = cell.get("rating_counts", {})
            lines.append(f"| {config} | " + " | ".join(
                str(counts.get(g, 0)) for g in GRADES) + " |")

        lines += ["", "## Đọc kết quả", "",
                  "- Credit là tín hiệu phụ, không thay COMET paired.",
                  "- Nếu credit và COMET cùng hướng, kết luận weight chắc hơn.",
                  "- Nếu lệch, ưu tiên COMET; ghi chú credit trong báo cáo.", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
