#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn hoá các dòng weak: tự động phân loại, chỉ để lại phần cần người quyết.

Đầu vào : data/cleaned/quantity_weak_rows.csv
Đầu ra  : data/cleaned/quantity_weak_resolved.csv

Phân loại:
  equivalent      : cùng đại lượng, chỉ khác cách viết (bỏ đơn vị, dấu phân cách)
                    -> giữ, weight 1.0
  parser_artifact : lỗi parser (tên màu "百年青", "元祖灰") -> giữ, weight 1.0
  real_mismatch   : cùng thứ nguyên nhưng giá trị khác -> cần review/loại
  ambiguous       : còn lại -> cần người quyết

Cột `decision` để trống cho người dùng điền.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cado_core.quantity_norm import (
    Quantity,
    WEAK_DIMS,
    extract_quantities,
)

CJK_RANGE = ("\u4e00", "\u9fff")


def is_cjk(char: str) -> bool:
    return bool(char) and CJK_RANGE[0] <= char <= CJK_RANGE[1]


def to_sig(quantities: list[Quantity], include_weak: bool = True):
    items = []
    for q in quantities:
        if not include_weak and q.dimension in WEAK_DIMS:
            continue
        if q.dimension:
            items.append((q.dimension, round(q.base_value, 6)))
        else:
            items.append(("bare", round(q.value, 6)))
    return items


def lenient_match(src_q: list[Quantity], tgt_q: list[Quantity]) -> tuple[bool, list, list]:
    """Khớp chính xác, rồi cho phép bare <-> đơn vị theo **giá trị gốc**.

    Ví dụ "29" ~ "29英寸": bare 29 khớp với quantity có raw value 29.
    Không cho count/year khớp với đơn vị vật lý.
    """
    used_src: set[int] = set()
    used_tgt: set[int] = set()

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(1e-6, 0.02 * max(abs(a), abs(b)))

    # 1. cùng thứ nguyên, cùng giá trị quy đổi
    for i, src in enumerate(src_q):
        for j, tgt in enumerate(tgt_q):
            if j in used_tgt:
                continue
            if src.dimension and src.dimension == tgt.dimension and close(src.base_value, tgt.base_value):
                used_src.add(i); used_tgt.add(j)
                break

    # 2. bare <-> đơn vị, so theo giá trị gốc
    for i, src in enumerate(src_q):
        if i in used_src:
            continue
        for j, tgt in enumerate(tgt_q):
            if j in used_tgt:
                continue
            if (not src.dimension) != (not tgt.dimension) and close(src.value, tgt.value):
                used_src.add(i); used_tgt.add(j)
                break

    # 3. bare <-> count/year cùng giá trị
    for i, src in enumerate(src_q):
        if i in used_src:
            continue
        for j, tgt in enumerate(tgt_q):
            if j in used_tgt:
                continue
            if (not src.dimension) and tgt.dimension in ("count", "year") and close(src.value, tgt.value):
                used_src.add(i); used_tgt.add(j)
                break
            if (not tgt.dimension) and src.dimension in ("count", "year") and close(src.value, tgt.value):
                used_src.add(i); used_tgt.add(j)
                break

    missing = [(q.dimension or "bare", round(q.base_value if q.dimension else q.value, 6))
               for i, q in enumerate(src_q) if i not in used_src]
    extra = [(q.dimension or "bare", round(q.base_value if q.dimension else q.value, 6))
             for j, q in enumerate(tgt_q) if j not in used_tgt]
    return not missing and not extra, missing, extra


def artifact_reason(dim: str, value: float, quantity: Quantity, text: str) -> str | None:
    """Phát hiện đại lượng do parser bắt nhầm từ tên màu/sản phẩm."""
    after = text[quantity.end:quantity.end + 1]
    if dim == "time" and quantity.unit == "年" and value >= 100 * 31536000 and is_cjk(after):
        return f"'{quantity.raw}{quantity.unit}{after}' là tên màu, không phải thời gian"
    if dim == "currency" and quantity.unit in ("元", "¥") and is_cjk(after):
        return f"'{quantity.raw}{quantity.unit}{after}' là tên màu/ thương hiệu, không phải tiền"
    if dim == "length" and quantity.unit == "码":
        return "'码' là size code, không phải yard"
    return None


def find_quantity(quantities: list[Quantity], dim: str, value: float) -> Quantity | None:
    for q in quantities:
        if not q.dimension and dim == "bare" and abs(q.value - value) <= 1e-6:
            return q
        if q.dimension == dim and abs(round(q.base_value, 6) - value) <= 1e-6:
            return q
    return None


def classify(row) -> dict:
    src_q = extract_quantities(str(row.source_text or ""), "vi")
    tgt_q = extract_quantities(str(row.target_text or ""), "zh")

    ok, missing, extra = lenient_match(src_q, tgt_q)
    if ok:
        return {
            "resolution": "equivalent",
            "action": "keep",
            "confidence": "high",
            "note": "cùng đại lượng, khác cách viết (bỏ đơn vị / dấu phân cách)",
        }

    artifacts = []
    for dim, value in list(missing):
        q = find_quantity(src_q, dim, value)
        if q is not None:
            reason = artifact_reason(dim, value, q, str(row.source_text or ""))
            if reason:
                artifacts.append(reason)
    for dim, value in list(extra):
        q = find_quantity(tgt_q, dim, value)
        if q is not None:
            reason = artifact_reason(dim, value, q, str(row.target_text or ""))
            if reason:
                artifacts.append(reason)

    if artifacts:
        return {
            "resolution": "parser_artifact",
            "action": "keep",
            "confidence": "high",
            "note": "; ".join(sorted(set(artifacts))),
        }

    # strong: cùng thứ nguyên vật lý ở cả hai bên, giá trị khác
    src_dims = {q.dimension for q in src_q if q.dimension not in WEAK_DIMS}
    tgt_dims = {q.dimension for q in tgt_q if q.dimension not in WEAK_DIMS}
    strong = any(dim in tgt_dims for dim, _ in missing) or any(dim in src_dims for dim, _ in extra)
    if strong:
        return {
            "resolution": "real_mismatch",
            "action": "review_drop",
            "confidence": "medium",
            "note": "cùng thứ nguyên nhưng giá trị khác — khả năng lỗi dữ liệu",
        }

    return {
        "resolution": "ambiguous",
        "action": "review",
        "confidence": "low",
        "note": "khác thứ nguyên/đơn vị, cần người quyết",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cleaned/quantity_weak_rows.csv")
    ap.add_argument("--out", default="data/cleaned/quantity_weak_resolved.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    results = [classify(row) for row in df.itertuples()]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    out["decision"] = ""

    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"wrote {args.out} | {len(out):,} dòng")
    print("\nresolution:")
    print(out["resolution"].value_counts().to_string())
    print("\naction:")
    print(out["action"].value_counts().to_string())
    print("\nresolution x dataset:")
    print(pd.crosstab(out["dataset"], out["resolution"]).to_string())
    print("\nví dụ cần người quyết:")
    review = out[out["action"].isin(["review", "review_drop"])]
    for row in review.head(10).itertuples():
        print(f"  [{row.dataset}] {str(row.source_text)[:45]} -> {str(row.target_text)[:35]}")
        print(f"     miss {row.missing} extra {row.extra} | {row.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
