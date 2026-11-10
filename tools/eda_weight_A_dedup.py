#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA riêng cho tập A sau dedup exact pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eda_weight_splits import (  # noqa: E402
    GROUPS_A, boxplot, enrich, group_table, split_bar, bar_share,
)

OUT = ROOT / "reports"
FIG = OUT / "figures_A_dedup"
FIG.mkdir(parents=True, exist_ok=True)
CURRENT = ROOT / "data/cleaned/weight_splits_v2/A_title_attribute.parquet"
RAW = ROOT / "data/cleaned/weight_splits_v2/A_title_attribute_raw.parquet"
META = ROOT / "data/cleaned/weight_splits_v2/meta_A_dedup.json"
EVAL = ROOT / "data/cleaned/weight_splits_v2/eval_A.parquet"


def main() -> int:
    raw = enrich(pd.read_parquet(RAW))
    a = enrich(pd.read_parquet(CURRENT))
    tab = group_table(a, GROUPS_A)
    meta = json.loads(META.read_text(encoding="utf-8"))

    bar_share(a, GROUPS_A, "A sau dedup — tỷ lệ dòng vs token", FIG / "A_group_share.png")
    boxplot(a, GROUPS_A, "eff_tokens", "effective tokens", "A sau dedup — độ dài",
            FIG / "A_length.png", log=True)
    boxplot(a, GROUPS_A, "ratio", "target/source chars", "A sau dedup — tỷ lệ độ dài",
            FIG / "A_ratio.png")
    boxplot(a, GROUPS_A, "cjk_ratio", "CJK ratio", "A sau dedup — tỷ lệ chữ Hán",
            FIG / "A_cjk.png")
    split_bar(a, "A sau dedup — split", FIG / "A_split.png")

    # duplicate flags
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = {
        "raw rows": len(raw),
        "dedup rows": len(a),
        "exact copies removed": meta["removed"],
        "cross-split groups": meta["cross_split_duplicate_groups"],
        "source-dup rows kept": int(raw.duplicated(["source_text"], keep=False).sum()),
    }
    ax.barh(list(vals), list(vals.values()),
            color=["#9ecae1", "#3182bd", "#e6550d", "#fd8d3c", "#bdbdbd"])
    ax.set_title("A — dedup exact pair")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "A_dup_flags.png", dpi=160); plt.close(fig)

    # top source duplicates
    top = (a[a.duplicated(["source_text"], keep=False)]
           .groupby("source_text").agg(
               rows=("pair_id", "size"),
               targets=("target_text", lambda x: len(set(map(str, x)))),
           ).sort_values("rows", ascending=False).head(8).reset_index())

    lines = [
        "# EDA tập A — sau dedup exact pair",
        "",
        "**Ngày:** 2026-11-07  ",
        "**Nguồn:** `data/cleaned/weight_splits_v2/A_title_attribute.parquet`  ",
        "**Raw backup:** `A_title_attribute_raw.parquet`",
        "",
        "## 1. Dedup đã làm gì",
        "",
        "```text",
        f"raw rows                 : {meta['raw_rows']:,}",
        f"dedup rows               : {meta['dedup_rows']:,}",
        f"exact pair removed       : {meta['removed']:,}",
        f"cross-split dup groups   : {meta['cross_split_duplicate_groups']:,}",
        f"source-dup rows kept     : {int(raw.duplicated(['source_text'], keep=False).sum()):,}",
        "rule                     : exact (source,target), train > dev > test",
        "```",
        "",
        "Không dedup source-only: cùng source nhưng target khác có thể là biến thể hợp lệ.",
        "Các cặp trùng đúng nhưng nằm ở nhiều split được giữ ở **train** để tránh leakage.",
        "",
        "## 2. Phân bố sau dedup",
        "",
        "| group | rows | row % | tokens | token % | median tok | empty | traditional |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tab.itertuples():
        lines.append(f"| {r.group} | {r.rows:,} | {r.row_share*100:.1f}% | "
                     f"{r.tokens:,} | {r.token_share*100:.1f}% | {r.median_tokens} | "
                     f"{r.empty:,} | {r.trad:,} |")
    lines += [
        "",
        f"Split: `{a['split'].value_counts().to_dict()}`.",
        f"Eval A: `{len(pd.read_parquet(EVAL)):,}` dòng, không đổi.",
        "",
        "![A group](figures_A_dedup/A_group_share.png)",
        "",
        "![A length](figures_A_dedup/A_length.png)",
        "",
        "![A ratio](figures_A_dedup/A_ratio.png)",
        "",
        "![A CJK](figures_A_dedup/A_cjk.png)",
        "",
        "![A split](figures_A_dedup/A_split.png)",
        "",
        "![A dup flags](figures_A_dedup/A_dup_flags.png)",
        "",
        "## 3. Nhận xét",
        "",
        f"- Chỉ mất **{meta['removed']:,} / {meta['raw_rows']:,} dòng** (~0,18%); phân bố gần như không đổi.",
        f"- Có **{meta['cross_split_duplicate_groups']:,}** nhóm cặp trùng nằm ở nhiều split; đã dồn về train, nên hiện không còn exact duplicate cross-split.",
        "- `1688_attribute` vẫn chiếm ~65% token; title 1688 ~16%; Tiki ~19%.",
        "- Tiki attribute giảm nhẹ do các cặp generic (`1 mét`/`1米`, `1 ngày`/`1天`, ...) bị dedup với 1688.",
        f"- Còn **{int(raw.duplicated(['source_text'], keep=False).sum()):,}** dòng source trùng nhưng target khác; giữ lại vì có thể là biến thể/ngữ cảnh khác. Cần theo dõi khi train.",
        "- Không có target rỗng; A vẫn sạch hơn B.",
        "",
        "## 4. Top source trùng (giữ lại, target khác nhau)",
        "",
        "| source | rows | số target khác |",
        "|---|---:|---:|",
    ]
    for r in top.itertuples():
        src = str(r.source_text).replace("|", "\\|")
        if len(src) > 80:
            src = src[:77] + "..."
        lines.append(f"| `{src}` | {r.rows} | {r.targets} |")

    lines += [
        "",
        "## 5. Dùng cho weight survey",
        "",
        "```text",
        "Dùng A_title_attribute.parquet (đã dedup exact pair) cho khảo sát weight.",
        "Không dùng A_title_attribute_raw.parquet trừ khi cần audit.",
        "eval_A.parquet giữ nguyên.",
        "```",
    ]
    (OUT / "EDA_A_DEDUP.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "EDA_A_DEDUP.md")
    print(tab.to_dict("records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
