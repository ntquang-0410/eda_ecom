#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra tổng thể trước khi dựng pair table.

Mỗi check trả PASS / WARN / FAIL kèm số liệu.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
CJK = re.compile(r"[\u4e00-\u9fff]")
VI = re.compile(r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
NOISE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u2060-\u2064\ufeff\u00ad\ue000-\uf8ff\U0001F300-\U0001FAFF]")

results = []


def add(name, status, detail):
    results.append({"check": name, "status": status, "detail": detail})


def main() -> int:
    # ---------- Tiki ----------
    full_rows, unit_rows = {}, {}
    for model in MODELS:
        full = pd.read_parquet(f"data/cleaned/bt_tiki/{model}/full.parquet")
        train = pd.read_parquet(f"data/cleaned/bt_tiki/{model}/train.parquet")
        units = pd.read_parquet(f"data/cleaned/bt_tiki/{model}/units.parquet")
        full_rows[model] = len(full)
        unit_rows[model] = len(units)

        problems = []
        if len(full) != 29446:
            problems.append(f"full={len(full)}")
        if full["cache_key"].duplicated().any():
            problems.append("dup cache_key")
        if full["cache_key"].isna().any():
            problems.append("null cache_key")
        if (train["qc_status"] != "ok").any():
            problems.append("train có qc_status != ok")
        if len(units) != 45709:
            problems.append(f"units={len(units)}")
        if units["pair_id"].duplicated().any():
            problems.append("dup pair_id")
        if units["product_id"].nunique() != 13200:
            problems.append("thiếu product")
        if units["qty_status"].isna().any():
            problems.append("thiếu qty_status")
        detail = (f"full {len(full):,} | train {len(train):,} | units {len(units):,} "
                  f"| prompt {full['prompt_version'].iloc[0]}")
        add(f"{model}.integrity", "PASS" if not problems else "FAIL",
            detail + ("" if not problems else " | " + "; ".join(problems)))

        noise = (units["source_text_bt"].fillna("").str.contains(NOISE).sum()
                 + units["target_text_bt"].fillna("").str.contains(NOISE).sum())
        add(f"{model}.noise_chars", "PASS" if noise == 0 else "FAIL", f"{int(noise)} dòng còn ký tự nhiễu")

        untr = int(((~units["target_text_bt"].fillna("").str.contains(CJK))
                    & units["source_text_bt"].fillna("").str.contains(VI)).sum())
        add(f"{model}.untranslated_units", "PASS" if untr < 3000 else "WARN",
            f"{untr:,} unit không có chữ Hán & source tiếng Việt")

        qty = units["qty_status"].value_counts().to_dict()
        mismatch = qty.get("qty_mismatch", 0)
        add(f"{model}.qty_mismatch", "WARN" if mismatch else "PASS",
            f"{mismatch:,} strong mismatch / {len(units):,} unit")

        dup = int(units.duplicated(["source_text_bt", "target_text_bt"]).sum())
        add(f"{model}.duplicate_pairs", "WARN" if dup > 0 else "PASS",
            f"{dup:,} cặp (source,target) trùng — nên dedup khi train")

    # split integrity
    units = pd.read_parquet("data/cleaned/bt_tiki/qwen3-8b/units.parquet")
    split_per_product = units.groupby("product_id")["split"].nunique()
    bad_split = int((split_per_product > 1).sum())
    add("tiki.split_integrity", "PASS" if bad_split == 0 else "FAIL",
        f"{bad_split} product nằm ở >1 split")
    add("tiki.split_counts", "PASS",
        str(units["split"].value_counts().to_dict()))

    # ---------- 1688 ----------
    b = pd.read_parquet("data/cleaned/1688/bilingual_1688_clean.parquet")
    a = pd.read_parquet("data/cleaned/1688/attributes_aligned.parquet")
    add("1688.rows", "PASS" if len(b) == 16133 else "FAIL", f"{len(b):,} sản phẩm sau dedup")
    dup_titles = int(b.duplicated(["title_vi_clean", "title_zh_clean"]).sum())
    add("1688.duplicate_titles", "PASS" if dup_titles == 0 else "FAIL", f"{dup_titles} cặp trùng")
    report = json.loads(Path("data/cleaned/1688/attr_align_report.json").read_text(encoding="utf-8"))
    pct = report["align_pct"]
    add("1688.attr_alignment", "PASS" if pct >= 95 else "WARN", f"{pct:.2f}% attribute align được")
    noise_zh = int(b["title_zh_clean"].str.contains(NOISE).sum() + b["description_zh_clean"].str.contains(NOISE).sum())
    add("1688.noise_chars", "PASS" if noise_zh == 0 else "FAIL", f"{noise_zh} dòng còn ký tự nhiễu")
    add("1688.description_duplicates", "WARN" if b["description_vi_clean"].duplicated().any() else "PASS",
        f"{int(b['description_vi_clean'].duplicated().sum())} description VI trùng")

    # ---------- leakage ----------
    tiki_sources = set(units["source_text_bt"].fillna("").str.strip().str.lower())
    titles = set(b["title_vi_clean"].fillna("").str.strip().str.lower())
    overlap = len(tiki_sources & titles)
    add("leakage.title_overlap", "PASS" if overlap == 0 else "WARN",
        f"{overlap} source Tiki trùng title 1688")

    # ---------- pending decisions ----------
    review = pd.read_csv("data/cleaned/quantity_review.csv", encoding="utf-8-sig")
    pending = int(review["decision"].fillna("").eq("").sum())
    add("review.pending", "WARN" if pending else "PASS",
        f"{pending:,} dòng review chưa có decision")

    # ---------- length vs max_seq ----------
    p95_src = units["source_text_bt"].fillna("").str.len().quantile(0.95)
    p99_tgt = units["target_text_bt"].fillna("").str.len().quantile(0.99)
    add("length.max_seq_check", "PASS" if p95_src + p99_tgt < 2000 else "WARN",
        f"source p95 {p95_src:.0f} ký tự + target p99 {p99_tgt:.0f} ký tự")

    # ---------- output ----------
    df = pd.DataFrame(results)
    df.to_json("data/cleaned/final_check.json", orient="records", force_ascii=False, indent=2)
    print(df.to_string(index=False, max_colwidth=80))
    counts = df["status"].value_counts().to_dict()
    print("\n", counts)
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
