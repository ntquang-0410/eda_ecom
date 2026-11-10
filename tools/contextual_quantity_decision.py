#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quyết định weak quantity bằng ngữ cảnh toàn sản phẩm.

Với mỗi dòng weak, thay vì chỉ so source_text vs target_text, ta dựng context:

  1688  : title_vi + toàn bộ attribute VI + description_vi
          title_zh + toàn bộ attribute ZH + description_zh
  Tiki  : title + toàn bộ attribute của product (VI)
          target v5 của toàn bộ unit cùng product (model đang xét)

Nếu đại lượng bị thiếu ở dòng này lại xuất hiện ở **field khác của cùng sản phẩm
bên kia**, thì cặp đó vẫn nhất quán toàn cục -> keep.
Nếu không xuất hiện ở bất kỳ field nào bên kia -> nhiều khả năng thiếu thật -> drop.

Input : data/cleaned/quantity_weak_needs_decision.csv
Output: data/cleaned/quantity_weak_needs_decision.csv (điền decision + evidence)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cado_core.quantity_norm import Quantity, extract_quantities

CJK = ("\u4e00", "\u9fff")


def close(a: float, b: float, rel: float = 0.02) -> bool:
    return abs(a - b) <= max(1e-6, rel * max(abs(a), abs(b)))


def q_items(text: str, lang: str) -> list[Quantity]:
    return extract_quantities(str(text or ""), lang)


def find_in(text: str, lang: str, dim: str, value: float) -> bool:
    """Đại lượng (dim, value) có xuất hiện trong text không."""
    for q in q_items(text, lang):
        qdim = q.dimension or "bare"
        qvalue = q.base_value if q.dimension else q.value
        if qdim == dim and close(qvalue, value):
            return True
        # bare <-> đơn vị theo giá trị gốc
        if (qdim == "bare") != (dim == "bare") and close(q.value, value):
            return True
        if dim in ("capacity", "current") and qdim in ("capacity", "current") and close(q.value, value):
            return True
        if dim == "year" and q.unit in ("年", "năm") and 1900 <= q.value <= 2100 and close(q.value, value):
            return True
    return False


def build_1688_context() -> tuple[dict, dict]:
    b = pd.read_parquet("data/cleaned/1688/bilingual_1688_clean.parquet")
    a = pd.read_parquet("data/cleaned/1688/attributes_aligned.parquet")

    attrs = {}
    for row in a.itertuples():
        values = row.attributes_aligned
        if values is None:
            continue
        if isinstance(values, np.ndarray):
            values = list(values)
        vi, zh = [], []
        for attr in values:
            if attr["status"] == "unmatched":
                continue
            vi.extend(map(str, attr["values_vi"]))
            zh.extend(map(str, attr["values_zh"]))
        attrs[str(row.product_id)] = (" ; ".join(vi), " ; ".join(zh))

    src, tgt = {}, {}
    for row in b.itertuples():
        pid = str(row.product_id)
        vi_attr, zh_attr = attrs.get(pid, ("", ""))
        src[pid] = " | ".join([str(row.title_vi_clean or ""), vi_attr, str(row.description_vi_clean or "")])
        tgt[pid] = " | ".join([str(row.title_zh_clean or ""), zh_attr, str(row.description_zh_clean or "")])
    return src, tgt


def build_tiki_context(model: str = "qwen3-8b") -> tuple[dict, dict]:
    units = pd.read_parquet(f"data/cleaned/bt_tiki/{model}/units.parquet")
    units = units[units["qc_status"] == "ok"]
    src, tgt = defaultdict(list), defaultdict(list)
    for row in units.itertuples():
        pid = str(row.product_id)
        src[pid].append(str(row.source_text_bt or ""))
        tgt[pid].append(str(row.target_text_bt or ""))
    return ({k: " | ".join(v) for k, v in src.items()},
            {k: " | ".join(v) for k, v in tgt.items()})


def decide(row, ctx1688_src, ctx1688_tgt, tiki_src, tiki_tgt) -> dict:
    missing = json.loads(row.missing) if isinstance(row.missing, str) else list(row.missing)
    extra = json.loads(row.extra) if isinstance(row.extra, str) else list(row.extra)
    dims = {dim for dim, _ in missing + extra}
    if not dims:
        return {"decision": "keep", "context_evidence": "không còn đại lượng lệch",
                "confidence": "high"}

    # chọn context
    if row.dataset.startswith("1688"):
        src_ctx, tgt_ctx = ctx1688_src.get(str(row.id).split(":")[0], ""), ctx1688_tgt.get(str(row.id).split(":")[0], "")
    else:
        pid = str(row.id).split(":")[1] if str(row.id).startswith("tiki:") else str(row.id)
        src_ctx, tgt_ctx = tiki_src.get(pid, ""), tiki_tgt.get(pid, "")

    # đại lượng thiếu ở target: tìm trong toàn bộ target context
    missing_found = []
    missing_absent = []
    for dim, value in missing:
        if find_in(tgt_ctx, "zh", dim, value):
            missing_found.append((dim, value))
        else:
            missing_absent.append((dim, value))

    extra_found = []
    extra_absent = []
    for dim, value in extra:
        if find_in(src_ctx, "vi", dim, value):
            extra_found.append((dim, value))
        else:
            extra_absent.append((dim, value))

    evidence = {
        "missing_found_elsewhere": missing_found,
        "missing_not_found": missing_absent,
        "extra_found_elsewhere": extra_found,
        "extra_not_found": extra_absent,
    }

    if not missing_absent and not extra_absent:
        return {"decision": "keep", "context_evidence": json.dumps(evidence, ensure_ascii=False),
                "confidence": "high"}
    if len(missing_absent) + len(extra_absent) < len(missing) + len(extra):
        return {"decision": "keep", "context_evidence": json.dumps(evidence, ensure_ascii=False),
                "confidence": "medium"}
    return {"decision": "review", "context_evidence": json.dumps(evidence, ensure_ascii=False),
            "confidence": "low"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cleaned/quantity_weak_needs_decision.csv")
    ap.add_argument("--model", default="qwen3-8b")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in ("decision", "context_evidence", "confidence") if c in df.columns])
    ctx1688_src, ctx1688_tgt = build_1688_context()
    tiki_src, tiki_tgt = build_tiki_context(args.model)
    print(f"context: 1688 {len(ctx1688_src):,} sản phẩm | tiki {len(tiki_src):,} sản phẩm")

    results = [decide(row, ctx1688_src, ctx1688_tgt, tiki_src, tiki_tgt)
               for row in df.itertuples()]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    out.to_csv(args.input, index=False, encoding="utf-8-sig")

    print(f"cập nhật {args.input} | {len(out):,} dòng")
    print(out["decision"].value_counts().to_string())
    print("\ntheo dataset:")
    print(pd.crosstab(out["dataset"], out["decision"]).to_string())
    print("\nví dụ còn review:")
    for row in out[out["decision"] == "review"].head(8).itertuples():
        print(f"  [{row.dataset}] {str(row.source_text)[:45]} -> {str(row.target_text)[:35]}")
        print(f"     evidence: {row.context_evidence[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
