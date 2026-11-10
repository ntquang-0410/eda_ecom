#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo hiệu quả chuẩn hoá đại lượng trên dữ liệu thật (1688 + Tiki).

So sánh 2 cách kiểm tra số/đơn vị:
  - naive  : so tập chữ số (cách cũ, gây false positive "144,5 vs 144.5")
  - quantity: quy đổi đơn vị về SI rồi so (cado_core.quantity_norm)

Báo cáo: độ phủ, tỉ lệ khớp, số cặp được cứu (naive sai → quantity đúng),
số cặp bị bắt thêm (naive đúng → quantity sai), ví dụ.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cado_core.quantity_norm import compare, extract_quantities

DIGIT = re.compile(r"\d+")


def naive_digits(text: str) -> set[str]:
    value = re.sub(r"(?<=\d)[.,](?=\d)", "", str(text or ""))
    return set(DIGIT.findall(value))


def has_quantity(text: str, lang: str) -> bool:
    return bool(extract_quantities(text, lang))


def evaluate(pairs: list[tuple[str, str]], label: str) -> dict:
    total = len(pairs)
    with_qty = 0
    naive_ok = qty_ok = both = rescued = newly_bad = strong_bad = 0
    examples_rescued, examples_bad, examples_missing = [], [], []

    for src, tgt in pairs:
        src, tgt = str(src or ""), str(tgt or "")
        if not (has_quantity(src, "vi") or has_quantity(tgt, "zh")):
            continue
        with_qty += 1
        naive = naive_digits(src) == naive_digits(tgt)
        result = compare(src, tgt)
        naive_ok += int(naive)
        qty_ok += int(result["match"])
        if naive and result.get("strong_mismatch"):
            strong_bad += 1
        both += int(naive and result["match"])
        if result["match"] and not naive:
            rescued += 1
            if len(examples_rescued) < 8:
                examples_rescued.append({"src": src[:90], "tgt": tgt[:90],
                                         "qty": result["src"]})
        if naive and not result["match"]:
            newly_bad += 1
            if len(examples_bad) < 8:
                examples_bad.append({"src": src[:90], "tgt": tgt[:90],
                                     "missing": result["missing"], "extra": result["extra"]})
        if not result["match"] and len(examples_missing) < 8:
            examples_missing.append({"src": src[:90], "tgt": tgt[:90],
                                     "missing": result["missing"], "extra": result["extra"]})

    return {
        "label": label,
        "pairs": total,
        "pairs_with_quantity": with_qty,
        "coverage_pct": round(with_qty / max(1, total) * 100, 2),
        "naive_match_pct": round(naive_ok / max(1, with_qty) * 100, 2),
        "quantity_match_pct": round(qty_ok / max(1, with_qty) * 100, 2),
        "both_match": both,
        "rescued_by_quantity": rescued,
        "newly_flagged_by_quantity": newly_bad,
        "newly_flagged_strong": strong_bad,
        "examples_rescued": examples_rescued,
        "examples_newly_flagged": examples_bad,
        "examples_mismatch": examples_missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned/quantity_audit.json")
    args = ap.parse_args()

    report = {}

    # 1688 title
    b = pd.read_parquet("data/cleaned/1688/bilingual_1688_clean.parquet")
    report["1688_title"] = evaluate(
        list(zip(b["title_vi_clean"], b["title_zh_clean"])), "1688 title")

    # 1688 attribute (join values)
    a = pd.read_parquet("data/cleaned/1688/attributes_aligned.parquet")
    pairs = []
    for row in a.itertuples():
        attrs = row.attributes_aligned
        if attrs is None:
            continue
        if isinstance(attrs, np.ndarray):
            attrs = list(attrs)
        for attr in attrs:
            if attr["status"] == "unmatched":
                continue
            pairs.append((" ; ".join(map(str, attr["values_vi"])),
                          " ; ".join(map(str, attr["values_zh"]))))
    report["1688_attribute"] = evaluate(pairs, "1688 attribute")

    # Tiki v5 units (chỉ dòng qc_status=ok)
    t = pd.read_parquet("data/cleaned/bt_tiki/qwen3-8b/units.parquet")
    t = t[t["qc_status"] == "ok"]
    report["tiki_units"] = evaluate(
        list(zip(t["source_text_bt"], t["target_text_bt"])), "Tiki units (8B)")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in report.items():
        print(f"\n== {value['label']} ==")
        print(f"  pairs {value['pairs']:,} | có đại lượng {value['pairs_with_quantity']:,} "
              f"({value['coverage_pct']}%)")
        print(f"  naive match {value['naive_match_pct']}%  ->  quantity match {value['quantity_match_pct']}%")
        print(f"  cứu được {value['rescued_by_quantity']:,} | bắt thêm {value['newly_flagged_by_quantity']:,} "
              f"(strong {value['newly_flagged_strong']:,})")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
