#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đếm dòng train A/B bị guard max_seq=960 loại (prompt_tokens >= 952).

Tính trên:
1. Toàn bộ units (tỉ lệ dòng).
2. Tập mẫu thực tế của từng config/seed (draws + eff_tokens bị mất) — đây mới là
   con số ảnh hưởng đến training.

Chạy: python tools/audit_train_guard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quota_sampling import quota_indices
from tpu_weight_train import build_prompt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".artifacts/train_guard"
THRESH = 952


def prompt_tokens(ds: str, tokenizer) -> pd.Series:
    units = pd.read_parquet(ROOT / f"data/cleaned/processed_{ds}/units.parquet")
    cache = CACHE / f"prompt_tokens_{ds}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).prompt_tokens
    tokens = []
    for s in units.source_text:
        p = build_prompt(tokenizer, s)
        tokens.append(len(tokenizer(p, add_special_tokens=False).input_ids))
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prompt_tokens": tokens}).to_parquet(cache, index=False)
    return pd.Series(tokens)


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    out = {}
    lines = ["# Training-side guard — số dòng bị loại bởi max_seq=960 (ngưỡng prompt 952)", "",
             "Guard: `encode_row()` trả `None` khi `prompt_tokens >= max_seq - 8` (Qwen chat template).",
             "Ảnh hưởng thật đo trên **tập mẫu của từng config/seed** (draws có hoàn lại).", ""]
    for ds in ("A", "B"):
        units = pd.read_parquet(ROOT / f"data/cleaned/processed_{ds}/units.parquet")
        pt = prompt_tokens(ds, tok)
        assert len(units) == len(pt)
        drop = pt >= THRESH
        empty = units.target_text.astype(str).str.strip().eq("") if "target_text" in units else pd.Series(False, index=units.index)
        base = {
            "rows": int(len(units)), "drop_rows": int(drop.sum()),
            "drop_pct": float(drop.mean() * 100),
            "drop_tokens_pct": float(units.loc[drop, "eff_tokens"].sum() / units.eff_tokens.sum() * 100),
            "max_prompt": int(pt.max()),
        }
        lines += [f"## Dataset {ds}", "",
                  f"- Units: **{base['rows']:,}** dòng; bị loại: **{base['drop_rows']:,}** "
                  f"(**{base['drop_pct']:.2f}%** dòng, **{base['drop_tokens_pct']:.2f}%** token); "
                  f"prompt max {base['max_prompt']}.", ""]
        groups = sorted(units.group.astype(str).unique())
        for cfg_name, cfg in (("natural", None), ("balanced", {g: 0.25 for g in groups})):
            rows = []
            for seed in (42, 123, 7):
                idxs, _ = quota_indices(units, cfg, seed, 4_000_000)
                arr = np.array(idxs)
                d = drop.to_numpy()[arr]
                tok_share = float(units.eff_tokens.to_numpy()[arr][d].sum() /
                                  max(1.0, units.eff_tokens.to_numpy()[arr].sum()) * 100)
                rows.append({"seed": seed, "draws": len(arr), "dropped_draws": int(d.sum()),
                             "dropped_pct": float(d.mean() * 100),
                             "dropped_token_pct": tok_share,
                             "unique_dropped": int(pd.Series(arr[d]).nunique())})
                out[f"{ds}_{cfg_name}_{seed}"] = rows[-1]
            lines += [f"### {ds} · {cfg_name}", "",
                      "| seed | draws | draws bị loại | % draws | % eff_tokens mất | unique rows bị loại |",
                      "|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                lines.append(f"| {r['seed']} | {r['draws']:,} | {r['dropped_draws']:,} | "
                             f"{r['dropped_pct']:.2f}% | {r['dropped_token_pct']:.2f}% | {r['unique_dropped']:,} |")
            lines.append("")
        out[ds] = base

    (ROOT / "reports/TRAIN_GUARD_960.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (CACHE / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
