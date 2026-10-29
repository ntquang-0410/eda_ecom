#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit tính đầy đủ của 4 tập Tiki BT sau 2 vòng repair.

Kiểm tra 5 lớp:
  A. Cấu trúc từng file        : số dòng, cache_key trùng/thiếu, field rỗng
  B. Đồng nhất giữa 4 model    : cùng tập cache_key, cùng source_text
  C. Đối chiếu nguồn Plan A    : tập cache_key == 29.446 cache route 'mt'
  D. Khả năng mở rộng unit     : recompute cache_key cho 45.709 unit MT
  E. Vệ sinh dữ liệu           : control char, mojibake, target == source

Output: JSON report + bảng in ra stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from cado_core.text import normalize_text
from cado_core.tiki_bt_prepare import PLAN_A_VERSION
from cado_core.tiki_units import normalize_key

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(
    r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MOJIBAKE_RE = re.compile(r"(Ã[\u0080-\u00bf]|Ä[\u0080-\u00bf]|áº|á»|â€Â|Æ[\u0080-\u00bf])")


def cache_key(source_text: str, key_norm: str, value_class: str, route: str) -> str:
    payload = "\x1f".join([
        normalize_text(source_text, collapse_whitespace=True).strip().lower(),
        normalize_key(key_norm),
        str(value_class or ""),
        str(route or ""),
        PLAN_A_VERSION,
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def audit(args: argparse.Namespace) -> dict:
    bt_root = Path(args.bt_root)
    plan_a = Path(args.plan_a)
    report: dict = {"models": {}, "cross_model": {}, "source": {}, "units": {}, "hygiene": {}}

    frames = {}
    for model in MODELS:
        df = pd.read_parquet(bt_root / model / "full.parquet")
        frames[model] = df
        src = df["source_text_bt"].fillna("")
        tgt = df["target_text_bt"].fillna("")
        has_cjk = tgt.str.contains(CJK_RE)
        report["models"][model] = {
            "rows": int(len(df)),
            "dup_cache_key": int(df["cache_key"].duplicated().sum()),
            "null_cache_key": int(df["cache_key"].isna().sum()),
            "empty_source": int(src.str.strip().eq("").sum()),
            "empty_target": int(tgt.str.strip().eq("").sum()),
            "target_cjk": int(has_cjk.sum()),
            "target_cjk_pct": round(has_cjk.mean() * 100, 2),
            "still_untranslated": int((df["qc_status"] == "still_untranslated").sum()),
            "qc_status": df["qc_status"].value_counts().to_dict(),
            "model_id_unique": df["model_id"].dropna().unique().tolist()[:3],
            "prompt_versions": df["prompt_version"].dropna().value_counts().to_dict(),
            "repair_pass": df["repair_pass"].value_counts().to_dict(),
            "json_ok_false": int((~df["json_ok"].astype(bool)).sum()),
            "value_class": df["value_class"].dropna().value_counts().to_dict(),
            "key_vi_empty": int(df["key_vi"].fillna("").str.strip().eq("").sum()),
        }

    # B. đồng nhất giữa model
    key_sets = {m: set(frames[m]["cache_key"].astype(str)) for m in MODELS}
    base = key_sets[MODELS[0]]
    report["cross_model"]["same_key_set"] = all(key_sets[m] == base for m in MODELS)
    report["cross_model"]["key_set_sizes"] = {m: len(key_sets[m]) for m in MODELS}
    for model in MODELS[1:]:
        a = frames[MODELS[0]].set_index("cache_key")["source_text_bt"]
        b = frames[model].set_index("cache_key")["source_text_bt"]
        common = a.index.intersection(b.index)
        report["cross_model"][f"source_mismatch_{model}"] = int((a.loc[common] != b.loc[common]).sum())

    # C. đối chiếu nguồn Plan A
    cache = pd.read_parquet(plan_a / "tiki_translation_cache.parquet")
    mt_keys = set(cache.loc[cache["bt_route"] == "mt", "cache_key"].astype(str))
    report["source"] = {
        "cache_rows": int(len(cache)),
        "cache_mt_keys": len(mt_keys),
        "dataset_keys": len(base),
        "missing_in_dataset": len(mt_keys - base),
        "extra_in_dataset": len(base - mt_keys),
    }

    # D. mở rộng ra unit
    units = pd.read_parquet(plan_a / "tiki_bt_units_plan_a.parquet")
    mt_units = units[units["bt_method"] == "mt"].copy()
    keys = []
    for row in mt_units.itertuples():
        source = row.text_vi if row.unit_type == "title" else row.value_vi
        keys.append(cache_key(str(source or ""), str(row.key_norm or ""),
                              str(row.value_class or ""), "mt"))
    mt_units["cache_key"] = keys
    mapped = mt_units["cache_key"].isin(base)
    report["units"] = {
        "mt_units": int(len(mt_units)),
        "mapped_units": int(mapped.sum()),
        "unmapped_units": int((~mapped).sum()),
        "mapped_pct": round(mapped.mean() * 100, 2),
        "products_total": int(units["product_id"].nunique()),
        "products_covered": int(mt_units.loc[mapped, "product_id"].nunique()),
        "split_units": mt_units.groupby("split").size().to_dict(),
        "split_mapped": mt_units[mapped].groupby("split").size().to_dict(),
    }
    if (~mapped).any():
        report["units"]["unmapped_sample"] = (
            mt_units.loc[~mapped, ["pair_id", "product_id", "unit_type", "text_vi", "value_class"]]
            .head(10).to_dict("records")
        )

    # E. vệ sinh
    for model, df in frames.items():
        tgt = df["target_text_bt"].fillna("")
        src = df["source_text_bt"].fillna("")
        report["hygiene"][model] = {
            "target_control_chars": int(tgt.str.contains(CTRL_RE).sum()),
            "target_mojibake": int(tgt.str.contains(MOJIBAKE_RE).sum()),
            "source_mojibake": int(src.str.contains(MOJIBAKE_RE).sum()),
            "target_leading_trailing_space": int((tgt != tgt.str.strip()).sum()),
            "target_equals_source": int((tgt.str.strip().str.lower() == src.str.strip().str.lower()).sum()),
            "target_vi_diacritics": int(tgt.str.contains(VI_RE).sum()),
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-root", default="data/cleaned/bt_tiki")
    ap.add_argument("--plan-a", default="/tmp/opencode/plan_a")
    ap.add_argument("--out", default="data/cleaned/bt_tiki/bt_audit.json")
    args = ap.parse_args()

    report = audit(args)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== A. từng model ===")
    for m, r in report["models"].items():
        print(f"{m:14} rows {r['rows']:6,} | dup {r['dup_cache_key']} | empty_tgt {r['empty_target']:4} "
              f"| CJK {r['target_cjk_pct']:5.2f}% | still {r['still_untranslated']:4} | json_ok_false {r['json_ok_false']}")
    print("\n=== B. đồng nhất ===")
    print(json.dumps(report["cross_model"], ensure_ascii=False, indent=1))
    print("\n=== C. nguồn Plan A ===")
    print(json.dumps(report["source"], ensure_ascii=False, indent=1))
    print("\n=== D. mở rộng unit ===")
    print(json.dumps({k: v for k, v in report["units"].items() if k != "unmapped_sample"},
                     ensure_ascii=False, indent=1))
    print("\n=== E. vệ sinh ===")
    for m, r in report["hygiene"].items():
        print(f"{m:14} {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
