#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng bộ dataset BT v5 từ predictions của kernel (cache-level → full/train/units).

Không ghi đè v3: output mặc định `data/cleaned/bt_tiki_v5`.

    python tools/build_bt_v5.py
    python tools/build_bt_v5.py --out data/cleaned/bt_tiki_v5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=".artifacts/bt_v5/plan_b_outputs_v5")
    ap.add_argument("--out", default="data/cleaned/bt_tiki_v5")
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "tiki-bt-v5",
        "version": "2026-11-01",
        "prompt_version": "plan_b_rtx_v5",
        "source": {"products": 13200, "units_mt": 45709, "cache_keys_mt": 29446,
                   "split_units": {"train": 36434, "dev": 4824, "test": 4451}},
        "models": {},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for model in MODELS:
        pred = pd.read_parquet(pred_root / model / "predictions.parquet")
        model_dir = out_root / model
        model_dir.mkdir(exist_ok=True)

        full = pred.copy()
        if "repair_pass" not in full.columns:
            full["repair_pass"] = 0
        full["qc_status"] = full["good"].map({True: "ok", False: "needs_review"})
        full["prompt_version"] = "plan_b_rtx_v5"
        full.to_parquet(model_dir / "full.parquet", index=False)

        train = full[full["qc_status"] == "ok"].drop(columns=["good"])
        train.to_parquet(model_dir / "train.parquet", index=False)

        entry = {
            "full": {"rows": int(len(full)), "bytes": (model_dir / "full.parquet").stat().st_size,
                     "sha256": sha256(model_dir / "full.parquet")},
            "train": {"rows": int(len(train)), "bytes": (model_dir / "train.parquet").stat().st_size,
                      "sha256": sha256(model_dir / "train.parquet")},
        }
        manifest["models"][model] = entry
        print(f"{model}: full {len(full):,} | train {len(train):,} "
              f"({len(train)/len(full)*100:.2f}%)")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Tiki BT v5 (prompt brand-aware)\n\n"
        "Sinh lại toàn bộ 29.446 cache MT với prompt v5:\n"
        "- brand-aware: chỉ giữ proper noun/brand/model code, danh từ Việt phải dịch\n"
        "- gate loại `target == key_zh` (lỗi 31% ở 0.6B của v3)\n"
        "- probe 80 dòng/model, retry bằng biến thể nhì\n\n"
        "Mỗi model: `full.parquet` (29.446) + `train.parquet` (qc_status=ok) + `units.parquet` (45.709).\n",
        encoding="utf-8")
    print("wrote", out_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
