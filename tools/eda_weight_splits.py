#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA cho 2 bộ weight A (title+attribute) và B (title+description).

Output:
  reports/figures_ab/*.png
  reports/EDA_A_B.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures_ab"
FIG.mkdir(parents=True, exist_ok=True)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
SPACE = re.compile(r"\s+")
BOILER = "giá sản phẩm trên tiki"
TRAD = set("產價稅這個與為於後裡說車門開關電話網購買賣東質顏適")
GROUPS_A = ["1688_title", "1688_attribute", "tiki_title", "tiki_attribute"]
GROUPS_B = ["1688_title", "1688_description", "tiki_title", "tiki_description"]


def compact(text) -> int:
    return len(SPACE.sub("", str(text or "")))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["src_len"] = df["source_text"].map(compact)
    df["tgt_len"] = df["target_text"].map(compact)
    df["empty"] = df["tgt_len"] == 0
    df["ratio"] = df["tgt_len"] / df["src_len"].clip(lower=1)
    df["cjk"] = [len(CJK_RE.findall(str(x))) for x in df["target_text"]]
    df["cjk_ratio"] = df["cjk"] / df["tgt_len"].clip(lower=1)
    df["vi"] = [len(VI_RE.findall(str(x))) for x in df["target_text"]]
    df["vi_ratio"] = df["vi"] / df["tgt_len"].clip(lower=1)
    df["boilerplate"] = df["source_text"].str.lower().str.contains(BOILER, regex=False)
    df["traditional"] = [any(ch in TRAD for ch in str(x)) for x in df["target_text"]]
    df["dup_pair"] = df.duplicated(["source_text", "target_text"], keep=False)
    df["dup_source"] = df.duplicated(["source_text"], keep=False)
    return df


def group_table(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    total_tok = float(df["eff_tokens"].sum())
    for g in groups:
        part = df[df["group"] == g]
        rows.append({
            "group": g,
            "rows": int(len(part)),
            "row_share": round(len(part) / max(1, len(df)), 4),
            "tokens": round(float(part["eff_tokens"].sum())),
            "token_share": round(float(part["eff_tokens"].sum()) / max(1, total_tok), 4),
            "mean_tokens": round(float(part["eff_tokens"].mean()), 1),
            "median_tokens": round(float(part["eff_tokens"].median()), 1),
            "empty": int(part["empty"].sum()),
            "empty_rate": round(float(part["empty"].mean()), 4),
            "boilerplate": int(part["boilerplate"].sum()),
            "trad": int(part["traditional"].sum()),
        })
    return pd.DataFrame(rows)


def bar_share(df: pd.DataFrame, groups: list[str], title: str, path: Path) -> None:
    tab = group_table(df, groups)
    x = np.arange(len(tab))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w/2, tab["row_share"] * 100, w, label="% số dòng", color="#9ecae1")
    ax.bar(x + w/2, tab["token_share"] * 100, w, label="% token", color="#3182bd")
    ax.set_xticks(x); ax.set_xticklabels(tab["group"], rotation=15, ha="right")
    ax.set_ylabel("%"); ax.set_title(title)
    ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def boxplot(df: pd.DataFrame, groups: list[str], col: str, ylabel: str,
            title: str, path: Path, log: bool = False) -> None:
    data = [df.loc[df["group"] == g, col].replace([np.inf, -np.inf], np.nan).dropna()
            for g in groups]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.boxplot(data, labels=groups, showfliers=False)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def split_bar(df: pd.DataFrame, title: str, path: Path) -> None:
    tab = df.groupby(["group", "split"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=["train", "dev", "test"], fill_value=0)
    ax = tab.plot(kind="bar", stacked=True, figsize=(10, 4.8),
                  color=["#3182bd", "#9ecae1", "#deebf7"])
    ax.set_ylabel("rows"); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="split")
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def eval_table(path: Path) -> pd.DataFrame:
    e = pd.read_parquet(path)
    return (e.groupby(["dataset", "pair_type"]).size()
            .reset_index(name="rows"))


def main() -> int:
    A = enrich(pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/A_title_attribute.parquet"))
    B = enrich(pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/B_title_description.parquet"))
    ea = pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/eval_A.parquet")
    eb = pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/eval_B.parquet")

    ta, tb = group_table(A, GROUPS_A), group_table(B, GROUPS_B)

    bar_share(A, GROUPS_A, "A — tỷ lệ dòng vs token", FIG / "A_group_share.png")
    bar_share(B, GROUPS_B, "B — tỷ lệ dòng vs token", FIG / "B_group_share.png")
    boxplot(A, GROUPS_A, "eff_tokens", "effective tokens", "A — độ dài theo nhóm",
            FIG / "A_length.png", log=True)
    boxplot(B, GROUPS_B, "eff_tokens", "effective tokens", "B — độ dài theo nhóm",
            FIG / "B_length.png", log=True)
    boxplot(A, GROUPS_A, "ratio", "target/source chars", "A — tỷ lệ độ dài",
            FIG / "A_ratio.png")
    boxplot(B, GROUPS_B, "ratio", "target/source chars", "B — tỷ lệ độ dài",
            FIG / "B_ratio.png")
    boxplot(A, GROUPS_A, "cjk_ratio", "CJK ratio", "A — tỷ lệ chữ Hán",
            FIG / "A_cjk.png")
    boxplot(B, GROUPS_B, "cjk_ratio", "CJK ratio", "B — tỷ lệ chữ Hán",
            FIG / "B_cjk.png")
    split_bar(A, "A — split theo nhóm", FIG / "A_split.png")
    split_bar(B, "B — split theo nhóm", FIG / "B_split.png")

    # B flags
    fig, ax = plt.subplots(figsize=(8, 4.5))
    flags = {
        "empty": int(B["empty"].sum()),
        "boilerplate": int(B["boilerplate"].sum()),
        "traditional": int(B["traditional"].sum()),
        "dup pair": int(B["dup_pair"].sum()),
        "dup source": int(B["dup_source"].sum()),
    }
    ax.barh(list(flags), list(flags.values()), color="#e6550d")
    ax.set_title("B — cờ chất lượng trên raw data")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "B_flags.png", dpi=160); plt.close(fig)

    def table_md(df: pd.DataFrame) -> str:
        lines = ["| group | rows | row % | tokens | token % | median tok | empty | boilerplate | traditional |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in df.itertuples():
            lines.append(f"| {r.group} | {r.rows:,} | {r.row_share*100:.1f}% | "
                         f"{r.tokens:,} | {r.token_share*100:.1f}% | {r.median_tokens} | "
                         f"{r.empty:,} | {r.boilerplate:,} | {r.trad:,} |")
        return "\n".join(lines)

    def eval_md(df: pd.DataFrame) -> str:
        lines = ["| dataset | pair_type | rows |", "|---|---|---:|"]
        for _, r in df.iterrows():
            lines.append(f"| {r['dataset']} | {r['pair_type']} | {int(r['rows']):,} |")
        return "\n".join(lines)

    def overlap(df: pd.DataFrame) -> str:
        tr = set(df[df["split"] == "train"]["product_id"].astype(str))
        te = set(df[df["split"] == "test"]["product_id"].astype(str))
        return f"{len(tr & te):,} product overlap train/test"

    lines = [
        "# EDA — bộ A (title+attribute) và B (title+description)",
        "",
        "**Ngày:** 2026-11-07  ",
        "**Nguồn:** dữ liệu trước EDA/processing; B dùng raw BT 4B cho Tiki description.",
        "",
        "## 0. Tóm tắt nhanh",
        "",
        f"- A: **{len(A):,} dòng**, **{A.eff_tokens.sum()/1e6:.2f}M token**, eval **{len(ea):,}** dòng.",
        f"- B: **{len(B):,} dòng**, **{B.eff_tokens.sum()/1e6:.2f}M token**, eval **{len(eb):,}** dòng.",
        f"- B có **{int(B["empty"].sum()):,} target rỗng** và **{int(B["boilerplate"].sum()):,}** dòng còn boilerplate Tiki.",
        "- A không có description; B không có attribute.",
        "",
        "## 1. Bộ A — title + attribute",
        "",
        table_md(ta),
        "",
        f"Split: `{A['split'].value_counts().to_dict()}`; {overlap(A)}.",
        "",
        "![A group share](figures_ab/A_group_share.png)",
        "",
        "![A length](figures_ab/A_length.png)",
        "",
        "![A ratio](figures_ab/A_ratio.png)",
        "",
        "![A CJK](figures_ab/A_cjk.png)",
        "",
        "![A split](figures_ab/A_split.png)",
        "",
        "### Nhận xét A",
        "",
        "- `1688_attribute` chiếm ~65% token nhưng chỉ ~62% số dòng → attribute là nhóm lớn nhất.",
        "- `1688_title` ~16% token; title bị underweight tự nhiên so với attribute.",
        "- Tiki title + attr ~19% token; Tiki attr rất ngắn, dễ bị chìm nếu sampling theo dòng.",
        "- Không có target rỗng; chất lượng A sạch hơn B.",
        "- Vì vậy A nên khảo sát `title_heavy` và `attr_heavy`; `balanced` sẽ kéo Tiki lên mạnh.",
        "",
        "## 2. Bộ B — title + description",
        "",
        table_md(tb),
        "",
        f"Split: `{B['split'].value_counts().to_dict()}`; {overlap(B)}.",
        "",
        "![B group share](figures_ab/B_group_share.png)",
        "",
        "![B length](figures_ab/B_length.png)",
        "",
        "![B ratio](figures_ab/B_ratio.png)",
        "",
        "![B CJK](figures_ab/B_cjk.png)",
        "",
        "![B split](figures_ab/B_split.png)",
        "",
        "![B flags](figures_ab/B_flags.png)",
        "",
        "### Nhận xét B",
        "",
        "- Description chiếm ~91,5% token; title chỉ ~8,5%.",
        "- `tiki_description` ~35% token dù chỉ ~22% số dòng → Tiki desc rất dài.",
        "- `1688_description` ~57% token.",
        f"- Có **{int(B["empty"].sum()):,}** target rỗng, chủ yếu ở Tiki description raw 4B.",
        f"- **{int(B["boilerplate"].sum()):,}** dòng còn boilerplate Tiki; đây là lý do phải tách `B_raw` và `B_processed`.",
        "- Vì natural đã description-heavy, `title_heavy`/`balanced` thực chất là **giảm description**; `desc_heavy` chỉ tăng nhẹ.",
        "- Cần strip boilerplate, bỏ target rỗng, normalize Traditional → Simplified trước khi train final.",
        "",
        "## 3. Eval",
        "",
        "### A",
        "",
        eval_md(eval_table(ROOT / "data/cleaned/weight_splits_v2/eval_A.parquet")),
        "",
        "### B",
        "",
        eval_md(eval_table(ROOT / "data/cleaned/weight_splits_v2/eval_B.parquet")),
        "",
        "Tiki description không có reference người; chỉ dùng COMET-QE. 1688 dùng COMET-da.",
        "",
        "## 4. Kết luận cho processing",
        "",
        "### A",
        "",
        "```text",
        "bỏ target rỗng (không có)",
        "dedup exact pair",
        "cap eff_tokens <= 960",
        "giữ split product-level",
        "quantity normalization",
        "```",
        "",
        "### B",
        "",
        "```text",
        "tách B_raw (weight survey) và B_processed (train final)",
        "strip boilerplate Tiki ở source/target",
        "bỏ 1.150 target rỗng",
        "normalize Traditional -> Simplified nếu có opencc",
        "dedup exact pair",
        "cap eff_tokens <= 960",
        "chunk description >960 token",
        "```",
        "",
        "## 5. Đề xuất weight survey",
        "",
        "```text",
        "A: natural | balanced | title_heavy | attr_heavy | tiki_heavy",
        "B: natural | balanced | title_heavy | desc_heavy | tiki_heavy",
        "budget 4M token, 500 optimizer step, seed 42 trước",
        "```",
        "",
        "Lưu ý: A chỉ có 4,707M token; 4M là budget an toàn. B có 14,598M token.",
    ]
    (OUT / "EDA_A_B.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "EDA_A_B.md")
    print("A", ta.to_dict("records"))
    print("B", tb.to_dict("records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
