#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-align attributes 1688 (VI <-> ZH) — bản 2, dùng anchor + MT + fuzzy.

Vấn đề: `attributes_vi` và `attributes_zh` không cùng thứ tự (xem
reports/1688_eda.md). Bản 1 chỉ khớp theo tên chính xác nên phủ 72,9%.

Bản 2 thêm fuzzy theo từng sản phẩm:
  1. Anchor map (học từ value overlap) — ưu tiên, độ chính xác cao.
  2. MT map (Qwen3-4B dịch tên key) — khớp chính xác.
  3. Fuzzy: so ký tự giữa tên MT và các tên ZH **còn trống trong chính sản phẩm**
     (xử lý trường hợp đồng nghĩa: "是否跨境货源" vs "是否跨境出口专供货源").
  4. Fallback: value-set overlap.

Output:
  data/cleaned/1688/attributes_aligned.parquet
  data/cleaned/1688/attr_align_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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


def value_set(attr) -> set[str]:
    return set(norm(v) for v in attr_values(attr))


def char_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cleaned/1688/bilingual_1688_clean.parquet")
    ap.add_argument("--raw", default="data/cleaned/bilingual_zh_vi_bitext_ready.parquet")
    ap.add_argument("--out", default="data/cleaned/1688")
    ap.add_argument("--mt-map", default=".artifacts/attr_names/attr_names_zh.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    raw = pd.read_parquet(args.raw)
    zh_map = dict(zip(raw["product_id"].astype(str), raw["attributes_zh"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- anchor map ----------
    votes: dict[str, Counter] = defaultdict(Counter)
    vi_freq: Counter = Counter()
    for row in df.itertuples():
        vi_attrs = row.attributes_vi_clean
        zh_attrs = zh_map.get(str(row.product_id))
        if vi_attrs is None or zh_attrs is None:
            continue
        zsets = [value_set(a) for a in zh_attrs]
        for attr in vi_attrs:
            name = attr_key(attr)
            if not name:
                continue
            vi_freq[name] += 1
            vset = value_set(attr)
            if not vset:
                continue
            for index, zset in enumerate(zsets):
                if len(vset) != len(zset):
                    continue
                if len(vset & zset) / len(vset) >= 0.6:
                    votes[name][attr_key(zh_attrs[index])] += 1

    anchors = {}
    for name, counter in votes.items():
        best, count = counter.most_common(1)[0]
        anchors[name] = {"zh": best, "votes": count}

    # ---------- MT map ----------
    mt_map: dict[str, str] = {}
    if Path(args.mt_map).exists():
        mt = pd.read_parquet(args.mt_map)
        for row in mt.itertuples():
            key = norm(row.name_vi)
            target = norm(row.name_zh_mt)
            if key and target:
                mt_map[key] = target

    # ---------- align ----------
    aligned_rows = []
    stats = Counter()
    total_attrs = 0
    full_products = 0

    for row in df.itertuples():
        vi_attrs = row.attributes_vi_clean
        zh_attrs = zh_map.get(str(row.product_id))
        if vi_attrs is None or zh_attrs is None:
            aligned_rows.append({"product_id": row.product_id, "attributes_aligned": None,
                                 "n_vi": 0, "n_aligned": 0, "status": "missing"})
            continue

        znames = [attr_key(a) for a in zh_attrs]
        zsets = [value_set(a) for a in zh_attrs]
        used: set[int] = set()
        aligned = []

        for attr in vi_attrs:
            total_attrs += 1
            name = attr_key(attr)
            values = attr_values(attr)
            vset = value_set(attr)
            anchor = anchors.get(name)
            mt_target = mt_map.get(name)
            desired = anchor["zh"] if anchor else (mt_target or "")
            chosen, how = None, None

            candidates = [i for i in range(len(zh_attrs)) if i not in used]

            # 1. khớp chính xác tên mong muốn
            if desired:
                exact = [i for i in candidates if znames[i] == desired]
                if len(exact) == 1:
                    chosen, how = exact[0], "anchor" if anchor else "mt"
                elif len(exact) > 1:
                    same = [i for i in exact if len(zsets[i]) == len(vset)]
                    pick = same or exact
                    chosen = max(pick, key=lambda i: len(vset & zsets[i]))
                    how = "anchor+dup" if anchor else "mt+dup"

            # 2. fuzzy trên tên MT (đồng nghĩa / containment)
            if chosen is None and mt_target:
                def score_for(index: int) -> float:
                    name = znames[index]
                    if mt_target and (mt_target in name or name in mt_target):
                        return 1.0
                    return char_sim(mt_target, name)

                scored = sorted(((score_for(i), i) for i in candidates), reverse=True)
                if scored:
                    best_score, best_i = scored[0]
                    second = scored[1][0] if len(scored) > 1 else 0.0
                    if best_score >= 0.34:
                        clear = (best_score - second) >= 0.10 or len(scored) == 1
                        if not clear and abs(best_score - second) <= 0.02:
                            # hoà điểm: ưu tiên candidate cùng số value
                            tied = [i for score, i in scored if abs(score - best_score) <= 0.02]
                            same_count = [i for i in tied if len(zsets[i]) == len(vset)]
                            if len(same_count) == 1:
                                best_i, clear = same_count[0], True
                        if clear:
                            chosen, how = best_i, "mt_fuzzy"

            # 3. fallback theo value
            if chosen is None and vset:
                overlap = [(i, len(vset & zsets[i]) / len(vset)) for i in candidates
                           if len(zsets[i]) == len(vset)]
                overlap = [x for x in overlap if x[1] >= 0.6]
                if len(overlap) == 1:
                    chosen, how = overlap[0][0], "value"

            if chosen is None:
                stats["unmatched"] += 1
                aligned.append({
                    "name_vi": attr.get("name"), "name_zh": desired,
                    "values_vi": values, "values_zh": [],
                    "status": "unmatched",
                })
                continue

            used.add(chosen)
            stats[how] += 1
            aligned.append({
                "name_vi": attr.get("name"), "name_zh": zh_attrs[chosen].get("name"),
                "values_vi": values, "values_zh": attr_values(zh_attrs[chosen]),
                "status": how,
            })

        n_aligned = sum(1 for a in aligned if a["status"] != "unmatched")
        if aligned and n_aligned == len(aligned):
            full_products += 1
        aligned_rows.append({
            "product_id": row.product_id,
            "attributes_aligned": aligned,
            "n_vi": len(aligned),
            "n_aligned": n_aligned,
            "status": "ok" if n_aligned == len(aligned) else "partial",
        })

    matched = total_attrs - stats["unmatched"]
    report = {
        "vi_names": len(vi_freq),
        "anchors": len(anchors),
        "mt_names": len(mt_map),
        "attributes_total": int(total_attrs),
        "aligned": int(matched),
        "align_pct": round(matched / max(1, total_attrs) * 100, 2),
        "by_how": {k: int(v) for k, v in stats.items()},
        "products": int(len(df)),
        "products_fully_aligned": int(full_products),
        "products_fully_aligned_pct": round(full_products / max(1, len(df)) * 100, 2),
    }

    pd.DataFrame(aligned_rows).to_parquet(out_dir / "attributes_aligned.parquet", index=False)
    (out_dir / "attr_align_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
