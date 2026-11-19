#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill dataset/eval hash + per-group exposure vào run JSON của các cell mới.

Exposure được tái lập tất định bằng chính `balance_plan` (cùng units/config/seed/
budget/steps) nên không cần manifest sample đã mất khi Colab reset.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from quota_sampling import balance_plan  # noqa: E402

BASE = ROOT / ".artifacts/matrix_out/m4b_a/matrix_data"
OUT = ROOT / "reports/runs_2026-11-19"
BALANCED = {"1688_attribute": 0.25, "1688_title": 0.25,
            "tiki_attribute": 0.25, "tiki_title": 0.25}  # A
BALANCED_B = {"1688_description": 0.25, "1688_title": 0.25,
              "tiki_description": 0.25, "tiki_title": 0.25}

CELLS = [
    # label, dataset, config name, config dict, seed, source run json (or None), note
    ("qwen4b_a_balanced", "A", "balanced", BALANCED, 42,
     "/content/dl/abal/out_A_qwen4b_balanced/run_balanced_seed42.json", ""),
    ("qwen4b_b_natural", "B", "natural", None, 42,
     "/content/dl/bnat/out_B_qwen4b_natural/run_natural_seed42.json", ""),
    ("qwen4b_b_balanced_s123", "B", "balanced", BALANCED_B, 123,
     "/content/dl/b123/out_B_qwen4b_balanced/run_balanced_seed123.json", ""),
    ("qwen4b_a_s123", "A", "natural", None, 123,
     None, "train xong ~590 phut (log Kaggle step 475/500 @561.5 phut); bi cat o inference, chay lai 50.7 phut tren A100"),
    ("nllb_a", "A", "natural", None, 42,
     None, "inference-only tren T4; 482.5 phut/1500 dong (run goc bi cat)" ),
    ("nllb_b", "B", "balanced", BALANCED_B, 42,
     "/content/dl/nllbb/run_nllb_b_infer.json", "inference-only tu adapter; 55.9 phut"),
]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exposure(units: pd.DataFrame, config, seed: int, budget: int = 4_000_000,
             steps: int = 500) -> dict:
    plan, actual = balance_plan(units, config, seed, budget, steps)
    drawn = [i for s in plan for i in s]
    groups = units["group"].to_numpy()
    out = {}
    for g in sorted(set(groups.tolist())):
        pop = int((groups == g).sum())
        d = [i for i in drawn if groups[i] == g]
        uniq = len(set(d))
        out[str(g)] = {"population_rows": pop, "draws": int(len(d)),
                       "unique_rows": uniq,
                       "coverage_pct": round(100 * uniq / max(1, pop), 2),
                       "draws_per_row": round(len(d) / max(1, uniq), 3)}
    return {"steps": len(plan), "draws": len(drawn),
            "unique_rows": len(set(drawn)),
            "actual_share": {k: round(v, 6) for k, v in actual.items()},
            "groups": out}


def main() -> int:
    units = {ds: pd.read_parquet(BASE / f"processed_{ds}/units.parquet") for ds in ("A", "B")}
    units = {k: v[v.split.isin(["train", "dev"])].reset_index(drop=True) for k, v in units.items()}
    ds_hash = {ds: sha_file(BASE / f"processed_{ds}/units.parquet") for ds in ("A", "B")}
    eval_hash = {
        "eval_F": sha_file(BASE / "eval_A.parquet"),
        "eval_F_v2": sha_file(ROOT / "data/cleaned/eval_F_v2.parquet"),
        "eval_F_v2_common": sha_file(ROOT / "data/cleaned/eval_F_v2_common.parquet"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for label, ds, cfgname, cfg, seed, src, note in CELLS:
        exp = exposure(units[ds], cfg, seed)
        rec = {}
        if src and Path(src).exists():
            rec = json.loads(Path(src).read_text(encoding="utf-8"))
        rec.update({
            "cell": label, "dataset": ds, "config": cfgname, "seed": seed,
            "dataset_hash": ds_hash[ds], "eval_hashes": eval_hash,
            "exposure": exp, "note": note,
        })
        p = OUT / f"run_{label}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        made.append(p)
        print("wrote", p.name, "draws", exp["draws"], "unique", exp["unique_rows"])

    manifest = ROOT / "reports/RUN_HASH_MANIFEST.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        m.setdefault("cells_2026_11_19", []).extend(
            [{"cell": c[0], "dataset": c[1], "dataset_hash": ds_hash[c[1]],
              "eval_hashes": eval_hash} for c in CELLS])
        manifest.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("updated", manifest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
