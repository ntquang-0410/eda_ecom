#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge kết quả BT-fix (các dòng empty) vào raw predictions gốc.

Input:
  --fix-root   thư mục chứa <model>/predictions.parquet (output kernel)
  --backup     nơi lưu bản gốc trước khi ghi đè
  --merged     nơi tạo source root mới cho process_bt_desc.py

Ghi đè trực tiếp file raw trong data/bt/raw/description/...,
đồng thời tạo bản merged copy cho processing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

RAW = {
    "qwen3-0.6b": "data/bt/raw/description/partial_old/qwen3-0.6b/predictions.parquet",
    "qwen3-1.7b": "data/bt/raw/description/partial_old/qwen3-1.7b/predictions.parquet",
    "qwen3-4b": "data/bt/raw/description/qwen3-4b/predictions_full.parquet",
    "qwen3-8b": "data/bt/raw/description/qwen3-8b/predictions.parquet",
}
ORIG_RAW = RAW.copy()
REQUIRED = ["pair_id", "product_id", "split", "source_text", "tokens_est",
            "target_text_bt", "raw_output", "json_ok", "good", "n_new_tokens",
            "input_truncated", "latency_ms", "model_id", "prompt_version",
            "attempt", "qc_status"]


def merge_one(model: str, fix_path: Path, raw_path: Path, backup: Path,
              merged_root: Path) -> dict:
    fix = pd.read_parquet(fix_path).drop_duplicates("pair_id")
    raw = pd.read_parquet(raw_path)
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, backup / f"{model}_predictions_before.parquet")

    fix = fix[fix["target_text_bt"].fillna("").str.strip().ne("")].copy()
    for col in REQUIRED:
        if col not in fix.columns:
            fix[col] = None
    extra = [c for c in fix.columns if c not in REQUIRED]
    only = fix[REQUIRED + extra]
    only["attempt"] = 2
    only["prompt_version"] = only["prompt_version"].fillna("plan_b_desc_v1_retry")

    key = raw["pair_id"].astype(str)
    patch = only.set_index(only["pair_id"].astype(str))
    hit = key.isin(patch.index)
    raw = raw.copy()
    for col in REQUIRED + extra:
        if col not in raw.columns:
            raw[col] = None
        vals = raw[col].astype(object)
        vals.loc[hit] = key[hit].map(patch[col]).values
        raw[col] = vals
    out = merged_root / model
    out.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    raw.to_parquet(out / "predictions.parquet", index=False)

    stats = {
        "model": model,
        "raw_rows": int(len(raw)),
        "fix_rows": int(len(fix)),
        "patched": int(hit.sum()),
        "patched_good": int(raw.loc[hit, "good"].fillna(False).astype(bool).sum()),
        "remaining_empty": int(raw["target_text_bt"].fillna("").str.strip().eq("").sum()),
        "remaining_empty_in_patched": int(
            raw.loc[hit, "target_text_bt"].fillna("").str.strip().eq("").sum()),
        "source": str(fix_path),
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-root", required=True)
    ap.add_argument("--backup", default=".artifacts/bt_fix_backup")
    ap.add_argument("--merged", default=".artifacts/bt_fix_merged")
    args = ap.parse_args()

    fix_root, backup, merged = Path(args.fix_root), Path(args.backup), Path(args.merged)
    merged.mkdir(parents=True, exist_ok=True)
    results = []
    for model, rel in RAW.items():
        src = fix_root / model / "predictions.parquet"
        if not src.exists():
            print("skip (no fix output)", model)
            continue
        results.append(merge_one(model, src, Path(rel), backup, merged))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    # model không có fix: copy nguyên để process đủ 4 model
    for model, rel in RAW.items():
        dst = merged / model / "predictions.parquet"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rel, dst)

    (merged / "merge_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
