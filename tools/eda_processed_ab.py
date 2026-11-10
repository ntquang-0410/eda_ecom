#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA cho processed_A / processed_B; xuất report + figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eda_weight_splits import (  # noqa: E402
    GROUPS_A, GROUPS_B, bar_share, boxplot, enrich, group_table, split_bar,
)

OUT = ROOT / "reports"
FIG = OUT / "figures_processed_ab"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> int:
    A = enrich(pd.read_parquet(ROOT / "data/cleaned/processed_A/units.parquet"))
    B = enrich(pd.read_parquet(ROOT / "data/cleaned/processed_B/units.parquet"))
    proc = json.loads((ROOT / "data/cleaned/processed_A_B.json").read_text(encoding="utf-8"))
    ta, tb = group_table(A, GROUPS_A), group_table(B, GROUPS_B)

    bar_share(A, GROUPS_A, "Processed A — dòng vs token", FIG / "A_group_share.png")
    bar_share(B, GROUPS_B, "Processed B — dòng vs token", FIG / "B_group_share.png")
    boxplot(A, GROUPS_A, "eff_tokens", "effective tokens", "Processed A — độ dài",
            FIG / "A_length.png", log=True)
    boxplot(B, GROUPS_B, "eff_tokens", "effective tokens", "Processed B — độ dài",
            FIG / "B_length.png", log=True)
    boxplot(A, GROUPS_A, "ratio", "target/source", "Processed A — tỷ lệ",
            FIG / "A_ratio.png")
    boxplot(B, GROUPS_B, "ratio", "target/source", "Processed B — tỷ lệ",
            FIG / "B_ratio.png")
    boxplot(A, GROUPS_A, "cjk_ratio", "CJK ratio", "Processed A — CJK",
            FIG / "A_cjk.png")
    boxplot(B, GROUPS_B, "cjk_ratio", "CJK ratio", "Processed B — CJK",
            FIG / "B_cjk.png")
    split_bar(A, "Processed A — split", FIG / "A_split.png")
    split_bar(B, "Processed B — split", FIG / "B_split.png")

    def rows_md(tab):
        lines = ["| group | rows | token % | median token | empty | CJK mean |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in tab.itertuples():
            lines.append(f"| {r.group} | {r.rows:,} | {r.token_share*100:.1f}% | "
                         f"{r.median_tokens} | {r.empty:,} | - |")
        return "\n".join(lines)

    lines = [
        "# EDA sau Processing — A/B",
        "",
        "**Ngày:** 2026-11-07  ",
        "**Input:** `data/cleaned/processed_A/units.parquet`, `processed_B/units.parquet`.",
        "",
        "## 1. Tóm tắt processing",
        "",
        "| bộ | raw rows | empty dropped | dedup removed | rows full | rows units | chunked parents | token (M) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| A | {proc['A']['raw_rows']:,} | {proc['A']['empty_dropped']:,} | "
        f"{proc['A']['dedup_removed']:,} | {proc['A']['rows_full']:,} | "
        f"{proc['A']['rows_units']:,} | {proc['A']['chunked_parents']:,} | {proc['A']['tokens_million']} |",
        f"| B | {proc['B']['raw_rows']:,} | {proc['B']['empty_dropped']:,} | "
        f"{proc['B']['dedup_removed']:,} | {proc['B']['rows_full']:,} | "
        f"{proc['B']['rows_units']:,} | {proc['B']['chunked_parents']:,} | {proc['B']['tokens_million']} |",
        "",
        "- A: giữ toàn bộ dòng người/BT đã qua gate; chunk 127 parent >1.280 token.",
        "- B: bỏ 1.532 target rỗng, dedup 485 cặp, chunk 626 parent; Tiki description raw 4B được QC riêng.",
        "",
        "## 2. A sau processing",
        "",
        rows_md(ta),
        "",
        f"Split: `{A['split'].value_counts().to_dict()}`. Eval A: **{proc['A']['eval_rows']:,}** dòng.",
        "",
        "![A group](figures_processed_ab/A_group_share.png)",
        "",
        "![A length](figures_processed_ab/A_length.png)",
        "",
        "![A ratio](figures_processed_ab/A_ratio.png)",
        "",
        "![A CJK](figures_processed_ab/A_cjk.png)",
        "",
        "![A split](figures_processed_ab/A_split.png)",
        "",
        "### Nhận xét A",
        "",
        "- `1688_attribute` vẫn ~66% token; title 1688 ~15,5%; Tiki ~18%.",
        "- Tất cả dòng A được giữ (`qc_ok = rows`), không bị lọc bởi CJK/ratio vì đây là dữ liệu người/BT đã gate.",
        "- Chunk 127 parent làm tăng nhẹ số dòng; token tăng nhẹ do cách pack chunk.",
        "- A sẵn sàng cho quota sampling; không còn exact duplicate.",
        "",
        "## 3. B sau processing",
        "",
        rows_md(tb),
        "",
        f"Split: `{B['split'].value_counts().to_dict()}`. Eval B: **{proc['B']['eval_rows']:,}** dòng.",
        "",
        "![B group](figures_processed_ab/B_group_share.png)",
        "",
        "![B length](figures_processed_ab/B_length.png)",
        "",
        "![B ratio](figures_processed_ab/B_ratio.png)",
        "",
        "![B CJK](figures_processed_ab/B_cjk.png)",
        "",
        "![B split](figures_processed_ab/B_split.png)",
        "",
        "### Nhận xét B",
        "",
        "- Description vẫn ~92% token; title ~8%.",
        "- `tiki_description` còn 11.397 dòng sau khi bỏ 1.532 target rỗng và QC raw 4B.",
        "- 626 parent description >1.280 token được chunk; đây là lý do rows units > rows full.",
        "- Boilerplate Tiki đã được strip khỏi source/target trước khi tính lại token.",
        "- B vẫn description-heavy; `balanced`/`title_heavy` sẽ giảm desc mạnh, `desc_heavy` gần natural.",
        "",
        "## 4. Dùng cho survey",
        "",
        "```text",
        "A survey: data/cleaned/processed_A/units.parquet",
        "B survey: data/cleaned/processed_B/units.parquet",
        "eval A/B giữ nguyên",
        "quota sampling: tools/quota_sampling.py",
        "budget 4M, 500 step",
        "```",
    ]
    (OUT / "EDA_PROCESSED_A_B.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "EDA_PROCESSED_A_B.md")
    print("A", ta.to_dict("records"))
    print("B", tb.to_dict("records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
