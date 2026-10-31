#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-align attributes 1688 (VI <-> ZH) trước khi dựng pair table.

Vấn đề: `attributes_vi` và `attributes_zh` không cùng thứ tự. Ví dụ cùng một
sản phẩm:

    VI[0] Thương hiệu = "Vũ Cen"   |  ZH[0] 材质 = "皮革"
    VI[1] Chất liệu   = "Da"       |  ZH[4] 品牌 = "羽岑"   <- đúng là VI[0]

Cách làm:
  1. Bootstrap map tên VI -> ZH từ chính dữ liệu: hai attribute được coi là cặp
     nếu cùng số value và tập value trùng >= 60% (brand, model, số, giá trị Hán
     còn sót...).
  2. Vote để chọn tên ZH cho mỗi tên VI.
  3. Align từng sản phẩm theo tên; nếu tên trùng hoặc thiếu, fallback theo
     value-set.
  4. Báo cáo độ phủ, xuất `attributes_aligned`.

Output:
  data/cleaned/1688/attributes_aligned.parquet   (product_id, attributes_aligned JSON)
  data/cleaned/1688/attr_align_report.json
  data/cleaned/1688/attr_name_map.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ZW = re.compile(r"[\u200b-\u200f\u2028-\u202f\ufeff]")


def as_list(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return list(values)
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def norm(value) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def attr_key(attr) -> str:
    return norm(attr.get("name") if isinstance(attr, dict) else "")


def attr_values(attr) -> list[str]:
    return [str(v).strip() for v in as_list(attr.get("values") if isinstance(attr, dict) else None)]


def value_sets(attrs) -> list[set[str]]:
    return [set(norm(v) for v in attr_values(a)) for a in attrs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cleaned/1688/bilingual_1688_clean.parquet")
    ap.add_argument("--out", default="data/cleaned/1688")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # attributes_vi_clean đã gộp key trùng; ZH đọc từ file gốc đã clean text
    zh_df = pd.read_parquet("data/cleaned/bilingual_zh_vi_bitext_ready.parquet")
    zh_map = dict(zip(zh_df["product_id"].astype(str), zh_df["attributes_zh"]))

    # ---------- 1. bootstrap map từ value overlap ----------
    votes: dict[str, Counter] = defaultdict(Counter)
    vi_freq: Counter = Counter()
    for row in df.itertuples():
        vi_attrs = row.attributes_vi_clean
        zh_attrs = zh_map.get(str(row.product_id))
        if vi_attrs is None or zh_attrs is None:
            continue
        zsets = value_sets(zh_attrs)
        for attr in vi_attrs:
            name = attr_key(attr)
            if not name:
                continue
            vi_freq[name] += 1
            values = set(norm(v) for v in attr_values(attr))
            if not values:
                continue
            for index, zset in enumerate(zsets):
                if len(values) != len(zset):
                    continue
                overlap = len(values & zset) / len(values)
                if overlap >= 0.6:
                    votes[name][attr_key(zh_attrs[index])] += 1

    name_map = {}
    for name, counter in votes.items():
        best, count = counter.most_common(1)[0]
        total = sum(counter.values())
        if count >= 1:
            name_map[name] = {"zh": best, "votes": count, "candidates": total,
                              "confidence": round(count / max(1, total), 3)}

    covered = sum(vi_freq[n] for n in name_map if n in vi_freq)
    report = {
        "vi_names": len(vi_freq),
        "mapped_names": len(name_map),
        "vi_attr_occurrences": int(sum(vi_freq.values())),
        "covered_occurrences": int(covered),
        "coverage_pct": round(covered / max(1, sum(vi_freq.values())) * 100, 2),
    }

    # ---------- 2. align từng sản phẩm ----------
    aligned_rows = []
    total_attrs = matched_name = matched_value = unmatched = 0
    full_products = 0

    for row in df.itertuples():
        vi_attrs = row.attributes_vi_clean
        zh_attrs = zh_map.get(str(row.product_id))
        if vi_attrs is None or zh_attrs is None:
            aligned_rows.append({"product_id": row.product_id, "attributes_aligned": None,
                                 "n_vi": 0, "n_aligned": 0, "status": "missing"})
            continue

        zsets = value_sets(zh_attrs)
        zused = set()
        aligned = []
        for attr in vi_attrs:
            total_attrs += 1
            name = attr_key(attr)
            values = attr_values(attr)
            vset = set(norm(v) for v in values)
            mapped = name_map.get(name, {}).get("zh")
            candidates = [i for i, za in enumerate(zh_attrs)
                          if mapped and attr_key(za) == mapped and i not in zused]
            chosen = None
            how = None
            if len(candidates) == 1:
                chosen = candidates[0]
                how = "name"
            elif len(candidates) > 1:
                same_count = [i for i in candidates if len(zsets[i]) == len(vset)]
                if len(same_count) == 1:
                    chosen = same_count[0]
                    how = "name+count"
                elif same_count:
                    best = max(same_count, key=lambda i: len(vset & zsets[i]))
                    if len(vset & zsets[best]) / max(1, len(vset)) >= 0.5:
                        chosen = best
                        how = "name+overlap"
            if chosen is None and vset:
                overlap = [(i, len(vset & zsets[i]) / len(vset)) for i in range(len(zh_attrs))
                           if i not in zused and len(zsets[i]) == len(vset)]
                overlap = [x for x in overlap if x[1] >= 0.6]
                if len(overlap) == 1:
                    chosen = overlap[0][0]
                    how = "value"

            if chosen is None:
                unmatched += 1
                aligned.append({
                    "name_vi": attr.get("name"), "name_zh": mapped or "",
                    "values_vi": values, "values_zh": [],
                    "status": "unmatched",
                })
                continue

            zused.add(chosen)
            zattr = zh_attrs[chosen]
            aligned.append({
                "name_vi": attr.get("name"), "name_zh": zattr.get("name"),
                "values_vi": values, "values_zh": attr_values(zattr),
                "status": how,
            })
            if how.startswith("name"):
                matched_name += 1
            else:
                matched_value += 1

        n_aligned = sum(1 for a in aligned if a["status"] != "unmatched")
        if n_aligned == len(aligned) and aligned:
            full_products += 1
        aligned_rows.append({
            "product_id": row.product_id,
            "attributes_aligned": aligned,
            "n_vi": len(aligned),
            "n_aligned": n_aligned,
            "status": "ok" if n_aligned == len(aligned) else "partial",
        })

    report.update({
        "attributes_total": int(total_attrs),
        "aligned_by_name": int(matched_name),
        "aligned_by_value": int(matched_value),
        "unmatched": int(unmatched),
        "align_pct": round((matched_name + matched_value) / max(1, total_attrs) * 100, 2),
        "products": int(len(df)),
        "products_fully_aligned": int(full_products),
        "products_fully_aligned_pct": round(full_products / max(1, len(df)) * 100, 2),
    })

    out = pd.DataFrame(aligned_rows)
    out.to_parquet(out_dir / "attributes_aligned.parquet", index=False)

    map_rows = [{"name_vi": k, **v} for k, v in sorted(name_map.items(), key=lambda x: -vi_freq[x[0]])]
    pd.DataFrame(map_rows).to_csv(out_dir / "attr_name_map.csv", index=False, encoding="utf-8-sig")
    (out_dir / "attr_align_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
