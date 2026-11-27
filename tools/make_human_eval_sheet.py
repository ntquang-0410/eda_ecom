#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh sheet chấm human eval cho Tiki từ artefact canonical.

Thiết kế: lấy mẫu phân tầng theo stratum, với mỗi item đưa k hệ thống (ẩn danh,
thứ tự ngẫu nhiên). Rater chấm adequacy + fluency 1–5 cho từng hypothesis.
Key ánh xạ slot → cell lưu riêng, không đưa cho rater.

Chạy:
  python tools/make_human_eval_sheet.py \
     --cells qwen4b_a qwen4b_b ecomgpt_a moses_b --n 120 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import canonical_eval as ce

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"
OUTDIR = ROOT / "data/cleaned/human_eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-run", type=int, default=42, help="seed của run dùng để lấy hyp")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    idx = ce.load_index()[["eval_id", "source"]]
    base = None
    for cell in args.cells:
        p = CANON / cell / f"seed{args.seed_run}.parquet"
        if not p.exists():
            raise SystemExit(f"thiếu artefact {p}")
        d = pd.read_parquet(p)
        d = d[d.dataset == "tiki"][["eval_id", "stratum", "ref", "hyp"]].copy()
        d = d.rename(columns={"hyp": cell})
        base = d if base is None else base.merge(d[["eval_id", cell]], on="eval_id", how="inner")
    base = base.merge(idx, on="eval_id", how="left")

    # lấy mẫu phân tầng
    parts = []
    per = max(1, args.n // base.stratum.nunique())
    for st, grp in base.groupby("stratum"):
        take = min(per, len(grp))
        parts.append(grp.sample(take, random_state=int(rng.integers(1e6))))
    sheet = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, key = [], []
    for i, r in sheet.iterrows():
        slots = list(args.cells)
        rng.shuffle(slots)
        for slot, cell in enumerate(slots, start=1):
            rows.append({"item_id": f"tiki:{i:04d}", "eval_id": r.eval_id, "stratum": r.stratum,
                         "source": r["source"], "system_slot": f"S{slot}",
                         "hypothesis": r[cell],
                         "adequacy_1_5": "", "fluency_1_5": "", "note": ""})
            key.append({"item_id": f"tiki:{i:04d}", "system_slot": f"S{slot}", "cell": cell})
    csv = OUTDIR / f"tiki_sheet_{'_'.join(args.cells)}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False, encoding="utf-8-sig")
    (OUTDIR / f"tiki_key_{'_'.join(args.cells)}.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(sheet)} item × {len(args.cells)} hệ thống → {csv}")
    print("key:", OUTDIR / f"tiki_key_{'_'.join(args.cells)}.json")


if __name__ == "__main__":
    main()
