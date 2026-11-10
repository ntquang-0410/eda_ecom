#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xuất các cặp "weak" của kiểm tra đại lượng ra CSV để review.

Weak = naive digits khớp (không bị cách cũ bắt) nhưng strict quantity mismatch
VÀ không phải strong (không có cùng thứ nguyên lệch giá trị ở cả hai bên).

Đây là nhóm dễ là false positive của parser (đơn vị chỉ xuất hiện một bên,
cách diễn đạt khác nhau...), cần xem tay trước khi quyết định weight.

Output: data/cleaned/quantity_weak_rows.csv (UTF-8 BOM để mở bằng Excel)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cado_core.quantity_norm import WEAK_DIMS, compare, signature, extract_quantities

DIGIT = re.compile(r"\d+")


def naive_digits(text: str) -> set[str]:
    value = re.sub(r"(?<=\d)[.,](?=\d)", "", str(text or ""))
    return set(DIGIT.findall(value))


def as_json(items) -> str:
    return json.dumps([[dim, value] for dim, value in items], ensure_ascii=False)


def reason_for(missing, extra, src_dims, tgt_dims) -> str:
    if not missing and not extra:
        return ""
    if missing and extra:
        return "both_sides_diff_dims"
    side = missing or extra
    if all(dim not in (tgt_dims if missing else src_dims) for dim, _ in side):
        return "one_side_only"
    return "same_dim_but_weak"


def collect(rows: list[dict], dataset: str, ids, src_texts, tgt_texts) -> int:
    added = 0
    for identifier, src, tgt in zip(ids, src_texts, tgt_texts):
        src, tgt = str(src or ""), str(tgt or "")
        if not src and not tgt:
            continue
        if naive_digits(src) != naive_digits(tgt):
            continue
        result = compare(src, tgt)
        if result["match"] or result["strong_mismatch"]:
            continue
        src_all = signature(extract_quantities(src, "vi"))
        tgt_all = signature(extract_quantities(tgt, "zh"))
        src_dims = {dim for dim, _ in src_all}
        tgt_dims = {dim for dim, _ in tgt_all}
        rows.append({
            "dataset": dataset,
            "id": str(identifier),
            "source_text": src,
            "target_text": tgt,
            "qty_src_all": as_json(src_all),
            "qty_tgt_all": as_json(tgt_all),
            "missing": as_json(result["missing"]),
            "extra": as_json(result["extra"]),
            "reason": reason_for(result["missing"], result["extra"], src_dims, tgt_dims),
        })
        added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned/quantity_weak_rows.csv")
    args = ap.parse_args()

    rows: list[dict] = []

    # 1688 title
    b = pd.read_parquet("data/cleaned/1688/bilingual_1688_clean.parquet")
    added = collect(rows, "1688_title", b["product_id"],
                    b["title_vi_clean"], b["title_zh_clean"])
    print("1688_title:", added)

    # 1688 attribute (mỗi attribute là 1 dòng)
    a = pd.read_parquet("data/cleaned/1688/attributes_aligned.parquet")
    attr_ids, vi_values, zh_values = [], [], []
    for row in a.itertuples():
        attrs = row.attributes_aligned
        if attrs is None:
            continue
        if isinstance(attrs, np.ndarray):
            attrs = list(attrs)
        for index, attr in enumerate(attrs):
            if attr["status"] == "unmatched":
                continue
            attr_ids.append(f"{row.product_id}:{index}")
            vi_values.append(" ; ".join(map(str, attr["values_vi"])))
            zh_values.append(" ; ".join(map(str, attr["values_zh"])))
    added = collect(rows, "1688_attribute", attr_ids, vi_values, zh_values)
    print("1688_attribute:", added)

    # Tiki v5 units (chỉ dòng ok)
    t = pd.read_parquet("data/cleaned/bt_tiki/qwen3-8b/units.parquet")
    t = t[t["qc_status"] == "ok"]
    added = collect(rows, "tiki_units", t["pair_id"], t["source_text_bt"], t["target_text_bt"])
    print("tiki_units:", added)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"wrote {args.out} | {len(df):,} dòng")
    print(df.groupby("dataset").size().to_string())
    print(df["reason"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
