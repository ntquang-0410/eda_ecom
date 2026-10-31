#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA + làm sạch tập 1688 trước khi dựng pair table.

Phát hiện chính: `attributes_vi` và `attributes_zh` **không cùng thứ tự**.
Cùng một sản phẩm, VI[0]="Thương hiệu" nhưng ZH[0]="材质"; bản dịch đúng của
"Thương hiệu" (品牌) lại nằm ở ZH[4]. Vì vậy `n_attributes_aligned` không có
nghĩa và mọi phép ghép theo vị trí đều sai.

Script này:
  1. Chuẩn hoá text (NFC, HTML, zero-width, whitespace).
  2. Dedup sản phẩm theo cặp title.
  3. Chuẩn hoá attribute: gộp key trùng, bỏ hậu tố `.1/.2` giả, cờ value còn Hán.
  4. Tính lại flag stuffing/truncated/bloated và so với flag đã lưu.
  5. Xuất bảng sạch + báo cáo + hình.

Output:
  data/cleaned/1688/bilingual_1688_clean.parquet
  data/cleaned/1688/eda_report.json
  reports/1688_eda.md
  reports/figures/1688_*.png
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CJK = re.compile(r"[\u4e00-\u9fff]")
VI = re.compile(
    r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
TAG = re.compile(r"<[^>]{1,120}>")
ZW = re.compile(r"[\u200b-\u200f\u2028-\u202f\ufeff]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE = re.compile(r"[ \t\u00a0]+")
SUFFIX = re.compile(r"^(?P<base>.+?)\.\d+$")
TOKEN = re.compile(r"[^\s,;&|/]+")
LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-+.]{2,}")
NUM = re.compile(r"\d+")


def clean_text(value, keep_newlines: bool = False) -> str:
    text = html.unescape(str(value or ""))
    text = TAG.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = ZW.sub("", text)
    text = CTRL.sub(" ", text)
    if keep_newlines:
        text = re.sub(r"[ \t\u00a0]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
    else:
        text = SPACE.sub(" ", text)
    return text.strip()


def as_list(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return list(values)
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def clean_attributes(attrs):
    """Chuẩn hoá list attribute, gộp key trùng, bỏ hậu tố .N giả."""
    if attrs is None:
        return [], {"names": 0, "values": 0, "dup_keys": 0, "vi_cjk_values": 0, "suffix_values": 0}

    merged: dict[str, dict] = {}
    dup_keys = 0
    vi_cjk = 0
    suffix = 0
    for attr in attrs:
        name = clean_text(attr.get("name") if isinstance(attr, dict) else None)
        if not name:
            continue
        key = name.lower()
        values = []
        for raw in as_list(attr.get("values") if isinstance(attr, dict) else None):
            value = clean_text(raw)
            if not value:
                continue
            if CJK.search(value):
                vi_cjk += 1
            match = SUFFIX.match(value)
            if match:
                suffix += 1
            values.append(value)
        if not values:
            continue
        if key in merged:
            dup_keys += 1
            existing = merged[key]["values"]
            for value in values:
                if value not in existing:
                    existing.append(value)
        else:
            merged[key] = {"name": name, "values": values}

    # bỏ hậu tố .N khi base đã tồn tại trong cùng attribute
    for entry in merged.values():
        values = entry["values"]
        bases = {SUFFIX.match(v).group("base") for v in values if SUFFIX.match(v)}
        if bases:
            entry["values"] = [
                SUFFIX.match(v).group("base") if SUFFIX.match(v) and SUFFIX.match(v).group("base") in values
                else v for v in values
            ]
            deduped = []
            for value in entry["values"]:
                if value not in deduped:
                    deduped.append(value)
            entry["values"] = deduped

    out = list(merged.values())
    stats = {
        "names": len(out),
        "values": sum(len(a["values"]) for a in out),
        "dup_keys": dup_keys,
        "vi_cjk_values": vi_cjk,
        "suffix_values": suffix,
    }
    return out, stats


def is_stuffed(text: str) -> bool:
    """Lặp token: token phổ biến nhất chiếm >= 18% (p95 của tập)."""
    tokens = [t.lower() for t in TOKEN.findall(text) if len(t) > 1]
    if len(tokens) < 6:
        return False
    top = Counter(tokens).most_common(1)[0][1]
    return top / len(tokens) >= 0.18


def flags_for(title_vi: str, title_zh: str) -> dict:
    """Hiệu chỉnh theo phân bố thật của tập 16.133 sản phẩm.

    zh_chars / vi_chars: p05 0,186 | p50 0,25 | p99 0,517.
    - truncated : < 0,20  (~1.560 dòng, khớp artifact cũ 1.566)
    - bloated   : > 0,60  (rất hiếm)
    """
    vi_chars = len(re.sub(r"\s", "", title_vi))
    cjk_chars = len(CJK.findall(title_zh))
    ratio = cjk_chars / max(1, vi_chars)
    return {
        "flag_stuffing_new": is_stuffed(title_vi),
        "flag_truncated_new": ratio < 0.20,
        "flag_bloated_new": ratio > 0.60,
        "vi_chars": vi_chars,
        "cjk_chars": cjk_chars,
        "zh_vi_ratio": round(ratio, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cleaned/bilingual_zh_vi_bitext_ready.parquet")
    ap.add_argument("--out", default="data/cleaned/1688")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    n0 = len(df)
    report: dict = {"input_rows": n0}

    # --- text ---
    df["title_vi_clean"] = df["title_vi"].map(clean_text)
    df["title_zh_clean"] = df["title_zh"].map(clean_text)
    df["description_vi_clean"] = df["description_vi"].map(lambda x: clean_text(x, keep_newlines=True))
    df["description_zh_clean"] = df["description_zh"].map(lambda x: clean_text(x, keep_newlines=True))

    # --- dedup ---
    dup_pair = int(df.duplicated(["title_zh_clean", "title_vi_clean"]).sum())
    df = df.drop_duplicates(["title_zh_clean", "title_vi_clean"], keep="first").reset_index(drop=True)
    report["dedup"] = {"duplicate_title_pairs": dup_pair, "after_dedup": len(df)}

    # --- attributes ---
    cleaned_attrs = []
    attr_stats = []
    for attrs in df["attributes_vi"]:
        out, stats = clean_attributes(attrs)
        cleaned_attrs.append(out)
        attr_stats.append(stats)
    df["attributes_vi_clean"] = cleaned_attrs
    stats_df = pd.DataFrame(attr_stats)
    report["attributes"] = {
        "products": int(len(df)),
        "names_total": int(stats_df["names"].sum()),
        "values_total": int(stats_df["values"].sum()),
        "names_p50": float(stats_df["names"].median()),
        "values_p50": float(stats_df["values"].median()),
        "dup_keys_merged": int(stats_df["dup_keys"].sum()),
        "vi_values_with_cjk": int(stats_df["vi_cjk_values"].sum()),
        "suffix_values": int(stats_df["suffix_values"].sum()),
    }

    # --- flags ---
    flag_rows = [flags_for(v, z) for v, z in zip(df["title_vi_clean"], df["title_zh_clean"])]
    flag_df = pd.DataFrame(flag_rows)
    df = pd.concat([df, flag_df], axis=1)
    report["flags"] = {
        "stored": {
            "stuffing": int(df["flag_stuffing"].sum()),
            "truncated": int(df["flag_truncated"].sum()),
            "bloated": int(df["flag_bloated"].sum()),
        },
        "recomputed": {
            "stuffing": int(df["flag_stuffing_new"].sum()),
            "truncated": int(df["flag_truncated_new"].sum()),
            "bloated": int(df["flag_bloated_new"].sum()),
        },
        "disagree_stuffing": int((df["flag_stuffing"].astype(bool) != df["flag_stuffing_new"]).sum()),
        "disagree_truncated": int((df["flag_truncated"].astype(bool) != df["flag_truncated_new"]).sum()),
    }

    # --- descriptions ---
    desc_vi_dup = df["description_vi_clean"].duplicated(keep=False)
    report["descriptions"] = {
        "empty_vi": int(df["description_vi_clean"].eq("").sum()),
        "empty_zh": int(df["description_zh_clean"].eq("").sum()),
        "duplicate_vi_rows": int(df["description_vi_clean"].duplicated().sum()),
        "duplicate_zh_rows": int(df["description_zh_clean"].duplicated().sum()),
        "length_vi_p50": float(df["description_vi_clean"].str.len().median()),
        "length_zh_p50": float(df["description_zh_clean"].str.len().median()),
    }
    df["desc_duplicate"] = desc_vi_dup

    # --- quality flags ---
    df["title_vi_has_cjk_clean"] = df["title_vi_clean"].str.contains(CJK)
    df["title_zh_has_vi_clean"] = df["title_zh_clean"].str.contains(VI)
    report["language"] = {
        "title_vi_with_cjk": int(df["title_vi_has_cjk_clean"].sum()),
        "title_zh_with_vi": int(df["title_zh_has_vi_clean"].sum()),
    }

    df["attr_align_status"] = "index_misaligned"
    df["keep_for_pairs"] = ~(
        df["title_vi_clean"].eq("") | df["title_zh_clean"].eq("") | df["title_zh_has_vi_clean"]
    )

    keep_cols = [
        "product_id", "category_1688", "title_vi_clean", "title_zh_clean",
        "description_vi_clean", "description_zh_clean", "attributes_vi_clean",
        "flag_stuffing", "flag_truncated", "flag_bloated",
        "flag_stuffing_new", "flag_truncated_new", "flag_bloated_new",
        "desc_duplicate", "attr_align_status", "keep_for_pairs",
        "entity_score", "n_attributes_aligned",
    ]
    df[keep_cols].to_parquet(out_dir / "bilingual_1688_clean.parquet", index=False)
    report["output_rows"] = int(len(df))
    report["keep_for_pairs"] = int(df["keep_for_pairs"].sum())

    (out_dir / "eda_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["title_vi_clean"].str.len().clip(upper=400), bins=60)
    ax.set_title("1688 title_vi length")
    ax.set_xlabel("chars")
    fig.tight_layout(); fig.savefig(report_dir / "figures/1688_title_len.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats_df["values"], bins=60)
    ax.set_title("1688 values per product")
    ax.set_xlabel("values")
    fig.tight_layout(); fig.savefig(report_dir / "figures/1688_attr_values.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["stored", "recomputed"]
    stuff = [report["flags"]["stored"]["stuffing"], report["flags"]["recomputed"]["stuffing"]]
    trunc = [report["flags"]["stored"]["truncated"], report["flags"]["recomputed"]["truncated"]]
    x = np.arange(2); w = 0.35
    ax.bar(x - w / 2, stuff, w, label="stuffing")
    ax.bar(x + w / 2, trunc, w, label="truncated")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
    ax.set_title("Stored vs recomputed flags")
    fig.tight_layout(); fig.savefig(report_dir / "figures/1688_flags.png", dpi=140); plt.close(fig)

    md = [
        "# EDA — Tập song ngữ 1688",
        "",
        f"- input: **{n0:,}** sản phẩm",
        f"- sau dedup title: **{len(df):,}** (bỏ {dup_pair})",
        f"- giữ cho pair table: **{int(df['keep_for_pairs'].sum()):,}**",
        "",
        "## Attribute",
        "",
        f"- names: {report['attributes']['names_total']:,} (p50 {report['attributes']['names_p50']:.0f}/sản phẩm)",
        f"- values: {report['attributes']['values_total']:,} (p50 {report['attributes']['values_p50']:.0f}/sản phẩm)",
        f"- key trùng đã gộp: {report['attributes']['dup_keys_merged']:,}",
        f"- value VI còn chữ Hán: {report['attributes']['vi_values_with_cjk']:,}",
        f"- hậu tố `.N` giả: {report['attributes']['suffix_values']:,}",
        "",
        "## Flag",
        "",
        "| flag | đã lưu | tính lại |",
        "|---|---|---|",
        f"| stuffing | {report['flags']['stored']['stuffing']:,} | {report['flags']['recomputed']['stuffing']:,} |",
        f"| truncated | {report['flags']['stored']['truncated']:,} | {report['flags']['recomputed']['truncated']:,} |",
        f"| bloated | {report['flags']['stored']['bloated']:,} | {report['flags']['recomputed']['bloated']:,} |",
        "",
        "## Mô tả",
        "",
        f"- rỗng: VI {report['descriptions']['empty_vi']}, ZH {report['descriptions']['empty_zh']}",
        f"- trùng lặp: VI {report['descriptions']['duplicate_vi_rows']:,}, ZH {report['descriptions']['duplicate_zh_rows']:,}",
        "",
        "## Lỗi căn chỉnh attribute",
        "",
        "`attributes_vi` và `attributes_zh` **không cùng thứ tự**. Ví dụ:",
        "",
        "```text",
        "VI[0] Thương hiệu = 'Vũ Cen'     | ZH[0] 材质 = '皮革'",
        "VI[1] Chất liệu   = 'Da'         | ZH[4] 品牌 = '羽岑'   <- đúng là bản dịch của VI[0]",
        "```",
        "",
        "=> Không thể ghép attribute theo index; cần re-align theo tên.",
        "",
    ]
    (report_dir / "1688_eda.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
