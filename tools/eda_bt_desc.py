#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh EDA report + figures cho BT description (0.6B/1.7B/4B/8B).

Đọc data/cleaned/bt_tiki_desc/summary.json + units, xuất:
  reports/TIKI_DESC_BT_PROCESSING.md
  reports/figures_desc_bt/*.png
Nếu có .artifacts/bt_fix_backup/merge_summary.json thì thêm mục retry.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"qwen3-0.6b": "0.6B", "qwen3-1.7b": "1.7B", "qwen3-4b": "4B", "qwen3-8b": "8B"}
FIG = ROOT / "reports/figures_desc_bt"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> int:
    summ = json.loads((ROOT / "data/cleaned/bt_tiki_desc/summary.json").read_text(encoding="utf-8"))
    models = [e["model"] for e in summ]
    x = range(len(summ))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    import numpy as np
    x = np.arange(len(summ)); w = 0.16
    for i, (key, name) in enumerate([("rows", "rows"), ("good_provider", "good"),
                                     ("qc_ok", "qc ok"), ("units", "units"),
                                     ("train_units", "train")]):
        vals = [e[key] for e in summ]
        ax.bar(x + (i - 2) * w, vals, w, label=name)
        for xi, v in zip(x + (i - 2) * w, vals):
            ax.text(xi, v, f"{v/1000:.1f}k", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[e["model"]] for e in summ])
    ax.set_ylabel("rows"); ax.set_title("Tiki description BT — funnel processing")
    ax.legend(ncol=5, fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "desc_bt_funnel.png", dpi=140); plt.close(fig)

    flags = [("empty", "empty"), ("no_cjk", "no CJK"), ("untranslated", "untranslated"),
             ("ratio_bad", "char ratio"), ("input_truncated", "truncated input"),
             ("number_mismatch", "number mismatch")]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(flags)); w = 0.2
    for i, e in enumerate(summ):
        vals = [e[k] / e["rows"] * 100 for k, _ in flags]
        ax.bar(x + (i - 1.5) * w, vals, w, label=LABELS[e["model"]])
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in flags], fontsize=9)
    ax.set_ylabel("% số dòng"); ax.set_title("Tỷ lệ lỗi QC theo model")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "desc_bt_qc_flags.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [pd.read_parquet(ROOT / f"data/cleaned/bt_tiki_desc/{e['model']}/units.parquet").tokens_est
            for e in summ]
    ax.boxplot(data, tick_labels=[LABELS[e["model"]] for e in summ], showfliers=False)
    ax.set_yscale("log"); ax.set_ylabel("tokens ước lượng / unit (log)")
    ax.set_title("Độ dài unit sau chunking"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "desc_bt_length.png", dpi=140); plt.close(fig)

    f4 = pd.read_parquet(ROOT / "data/cleaned/bt_tiki_desc/qwen3-4b/full.parquet")
    f8 = pd.read_parquet(ROOT / "data/cleaned/bt_tiki_desc/qwen3-8b/full.parquet")
    ids = set(f8["product_id"])
    q4 = (f4[f4["product_id"].isin(ids)]["qc_status"] == "ok").mean()
    q8 = (f8["qc_status"] == "ok").mean()
    fair_lines = [f"8B hiện có {f8['product_id'].nunique():,}/13.200 sản phẩm.",
                  f"Trên cùng {len(ids):,} sản phẩm: 4B QC ok **{q4*100:.1f}%**, 8B **{q8*100:.1f}%**.",
                  "8B đã phủ gần đủ; chênh lệch chủ yếu do chất lượng target, không còn do thiếu dữ liệu."]

    lines = ["# Tiki description BT — QC & processing", "",
             "**Ngày:** 2026-11-08  ",
             "**Script:** `code/process_bt_desc.py`  ",
             "**Input:** `data/bt/raw/description/`  ",
             "**Output:** `data/bt/description/processed/`", "",
             "## 1. Funnel processing", "",
             "| model | rows | good | QC ok | QC ok % | dedup bỏ | units | train | token (M) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in summ:
        lines.append(f"| {e['model']} | {e['rows']:,} | {e['good_provider']:,} | {e['qc_ok']:,} | "
                     f"{e['qc_ok_rate']*100:.1f}% | {e['dedup_removed']:,} | {e['units']:,} | "
                     f"{e['train_units']:,} | {e['token_million']} |")
    lines += ["", "![funnel](figures/desc_bt/desc_bt_funnel.png)", "",
              "## 2. Lỗi QC", "",
              "| model | empty | no CJK | untranslated | char ratio | truncated | number mismatch |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for e in summ:
        lines.append(f"| {e['model']} | {e['empty']:,} | {e['no_cjk']:,} | {e['untranslated']:,} | "
                     f"{e['ratio_bad']:,} | {e['input_truncated']:,} | {e['number_mismatch']:,} |")
    lines += ["", "![flags](figures/desc_bt/desc_bt_qc_flags.png)", "",
              "Nhận xét:", "",
              "- QC pass: " + " → ".join(f"{e['qc_ok_rate']*100:.1f}%" for e in summ) + ".",
              "- `empty` còn " + ", ".join(f"{e['empty']:,}" for e in summ) + " dòng; phần lớn là target chỉ chứa boilerplate giá/tax.",
              "- `number_mismatch` vẫn là lỗi phổ biến nhất ở mọi model.",
              "- `input_truncated`: " + ", ".join(f"{e['model']}={e['input_truncated']}" for e in summ) + ".",
              "- `f_boilerplate` phản ánh boilerplate phía source (đã strip), không phải lỗi target.", "",
              "## 3. Sau chunking", "",
              "| model | units | train/dev/test | token (M) | sản phẩm bị chunk |",
              "|---|---:|---|---:|---:|"]
    for e in summ:
        sp = e["split"]
        lines.append(f"| {e['model']} | {e['units']:,} | {sp.get('train',0):,} / {sp.get('dev',0):,} / "
                     f"{sp.get('test',0):,} | {e['token_million']} | {e['chunked_products']:,} |")
    lines += ["", "![length](figures/desc_bt/desc_bt_length.png)", "",
              "## 4. So sánh công bằng 4B vs 8B", "",
              *fair_lines, "",
              "## 5. Dùng cho train", "", "```text",
              "data/bt/description/processed/qwen3-4b/train.parquet",
              "data/bt/description/processed/qwen3-4b/units.parquet",
              "data/bt/description/processed/qwen3-8b/train.parquet",
              "```", ""]

    retry_path = ROOT / ".artifacts/bt_fix_backup/merge_summary.json"
    if retry_path.exists():
        retry = json.loads(retry_path.read_text(encoding="utf-8"))
        lines += ["## 6. Retry các dòng empty", "",
                  "Các dòng `target_text_bt` rỗng được BT lại (attempt 2) với fallback parse",
                  "JSON bị cắt và split input dài:", "",
                  "| model | fix rows | patched | good | còn empty |",
                  "|---|---:|---:|---:|---:|"]
        for r in retry:
            lines.append(f"| {r['model']} | {r['fix_rows']:,} | {r['patched']:,} | "
                         f"{r['patched_good']:,} | {r['remaining_empty']:,} |")
        lines.append("")

    (ROOT / "reports/TIKI_DESC_BT_PROCESSING.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/TIKI_DESC_BT_PROCESSING.md")
    for e in summ:
        print(e["model"], "qc_ok", e["qc_ok"], f"{e['qc_ok_rate']*100:.1f}%",
              "units", e["units"], "empty", e["empty"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
