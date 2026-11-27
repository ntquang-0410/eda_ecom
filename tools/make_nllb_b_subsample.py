#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cắt ngẫu nhiên tập B xuống cùng kích thước A (4,84 M token) cho NLLB B.

Mục tiêu: budget 4 M token ⇒ A và B đều ~0,83 epoch (A: 4,842 M; B sau cắt: 4,842 M).

Lưu ý: lấy mẫu từ **toàn bộ units_B** (đúng pool mà pipeline NLLB dùng để build
sample_*.parquet); giữ nguyên cột gốc + thêm `subsample_seed`, `subsample_rank`.

Chạy: python tools/make_nllb_b_subsample.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
OUT = ROOT / "data/cleaned/nllb_train_b_subsampled.parquet"
META = ROOT / "data/cleaned/nllb_train_b_subsampled.json"


def main() -> None:
    a = pd.read_parquet(ROOT / "data/cleaned/processed_A/units.parquet")
    b = pd.read_parquet(ROOT / "data/cleaned/processed_B/units.parquet")
    target = float(a.eff_tokens.sum())
    total_b = float(b.eff_tokens.sum())

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(b))
    eff = b.eff_tokens.to_numpy(dtype=float)
    cum = np.cumsum(eff[perm])
    k = int(np.searchsorted(cum, target, side="left")) + 1   # dòng làm vượt target
    chosen = np.sort(perm[:k])                               # giữ thứ tự gốc

    out = b.iloc[chosen].copy()
    out["subsample_seed"] = SEED
    out["subsample_rank"] = np.arange(len(out))
    actual = float(out.eff_tokens.sum())
    out.to_parquet(OUT, index=False)

    meta = {
        "seed": SEED,
        "source": "data/cleaned/processed_B/units.parquet",
        "target_tokens": target,
        "actual_tokens": actual,
        "overshoot_tokens": actual - target,
        "rows_source": int(len(b)),
        "rows_subsampled": int(len(out)),
        "fraction_rows": len(out) / len(b),
        "epochs_at_4M": 4_000_000 / actual,
        "baseline_A": {"tokens": target, "rows": int(len(a)), "epochs_at_4M": 4_000_000 / target},
        "baseline_B_full": {"tokens": total_b, "rows": int(len(b)), "epochs_at_4M": 4_000_000 / total_b},
        "group_tokens_share_subsample": (out.groupby("group").eff_tokens.sum() / actual).round(4).to_dict(),
        "group_tokens_share_full_B": (b.groupby("group").eff_tokens.sum() / total_b).round(4).to_dict(),
        "split_rows": out.split.value_counts().to_dict() if "split" in out else None,
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if not k.startswith("group")}, ensure_ascii=False, indent=2))
    print("group share (subsample):", meta["group_tokens_share_subsample"])
    print("group share (full B) :", meta["group_tokens_share_full_B"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
