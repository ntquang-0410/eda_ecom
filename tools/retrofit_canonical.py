#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrofit mọi run đã có về artefact canonical (index 831, zero-fill, coverage).

Chạy: python tools/retrofit_canonical.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import canonical_eval as ce

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / ".artifacts/hyp"
OFF = ROOT / ".artifacts/matrix_out/comet5_v2_common/scores"
GPU = ROOT / ".artifacts/comet_gpu/scores"
S7 = ROOT / ".artifacts/comet_s7_common"
A7 = ROOT / ".artifacts/comet_a7"
B8 = ROOT / ".artifacts/comet_8b"

# cell, model, data, recipe, seed, hyp stem, score dir, score stem, token_budget, adapter, note
RUNS = [
    ("qwen8b_a", "Qwen3-8B", "A", "natural", 42, "hyp_qwen_a_seed42", OFF, "qwen_a_seed42", 4_000_000, None, ""),
    ("qwen8b_b", "Qwen3-8B", "B", "balanced", 42, "hyp_qwen_b_seed42", OFF, "qwen_b_seed42", 4_000_000, None, ""),
    ("qwen8b_a_balanced", "Qwen3-8B", "A", "balanced", 42, "hyp_qwen8b_a_balanced_seed42", None, "qwen8b_a_balanced_seed42", 4_000_000, None, "retrofit: chưa có score từng dòng"),
    ("qwen8b_b_natural", "Qwen3-8B", "B", "natural", 42, "hyp_m8b_b_natural_seed42", B8, "m8b_b_natural_seed42", 4_000_000, "nguynthanhquang/qwen8b-b-ckpt@500", "tái lập 24/11"),
    ("qwen4b_a", "Qwen3-4B", "A", "natural", 42, "hyp_qwen4b_a_seed42", OFF, "qwen4b_a_seed42", 4_000_000, None, ""),
    ("qwen4b_b", "Qwen3-4B", "B", "balanced", 42, "hyp_qwen4b_b_seed42", OFF, "qwen4b_b_seed42", 4_000_000, None, ""),
    ("qwen4b_a_balanced", "Qwen3-4B", "A", "balanced", 42, "hyp_qwen4b_a_balanced_seed42", GPU, "qwen4b_a_balanced_seed42", 4_000_000, None, ""),
    ("qwen4b_b_natural", "Qwen3-4B", "B", "natural", 42, "hyp_qwen4b_b_natural_seed42", GPU, "qwen4b_b_natural_seed42", 4_000_000, None, ""),
    ("qwen4b_a", "Qwen3-4B", "A", "natural", 123, "hyp_qwen4b_a_seed123", GPU, "qwen4b_a_seed123", 4_000_000, None, ""),
    ("qwen4b_b", "Qwen3-4B", "B", "balanced", 123, "hyp_qwen4b_b_balanced_seed123", GPU, "qwen4b_b_balanced_seed123", 4_000_000, None, ""),
    ("qwen4b_a", "Qwen3-4B", "A", "natural", 7, "hyp_qwen4b_a_natural_seed7", A7, "natural_seed7", 4_000_000,
     "nguynthanhquang/qwen4b-a-s7-ckpt@500", "train bị cắt 12h, resume từ ckpt 300"),
    ("ecomgpt_a", "EcomGPT-7B", "A", "natural", 42, "hyp_ecomgpt_a_seed42", OFF, "ecomgpt_a_seed42",
     4_000_000, None, "recipe chia token encoded ≈1,42M token hiệu dụng"),
    ("ecomgpt_b", "EcomGPT-7B", "B", "balanced", 42, "hyp_ecomgpt_b_seed42", OFF, "ecomgpt_b_seed42",
     4_000_000, None, "recipe chia token encoded ≈1,42M token hiệu dụng"),
    ("ecomgpt_a_balanced", "EcomGPT-7B", "A", "balanced", 42, "hyp_ecomgpt_a_balanced_seed42", S7, "ecomgpt_a_balanced_seed42",
     4_000_000, "nguynthanhquang/ecomgpt-ckpt@500", "tái lập 24/11; ≈1,42M token hiệu dụng"),
    ("moses_a", "Moses", "A", "natural", 42, "hyp_moses_a_seed42", OFF, "moses_a_seed42", None, None, ""),
    ("moses_b", "Moses", "B", "balanced", 42, "hyp_moses_b_seed42", OFF, "moses_b_seed42", None, None, ""),
    ("onmt_a", "OpenNMT", "A", "natural", 42, "hyp_onmt_a_seed42", OFF, "onmt_a_seed42", None, None, ""),
    ("onmt_b", "OpenNMT", "B", "natural", 42, "hyp_onmt_b_seed42", OFF, "onmt_b_seed42", None, None, ""),
    ("nllb_a", "NLLB-600M", "A", "natural", 42, "hyp_nllb_a_seed42", GPU, "nllb_a_seed42", 4_000_000, None, ""),
    ("nllb_b", "NLLB-600M", "B", "balanced", 42, "hyp_nllb_b_seed42", GPU, "nllb_b_seed42", 4_000_000, None, ""),
    ("nllb_a", "NLLB-600M", "A", "natural", 7, "hyp_nllb_a_seed7", S7, "nllb_a_seed7", 4_000_000, None, ""),
    ("nllb_b", "NLLB-600M", "B", "balanced", 7, "hyp_nllb_b_seed7", S7, "nllb_b_seed7", 4_000_000, None, ""),
]


def load_scores(sdir: Path | None, stem: str, ds: str):
    if sdir is None:
        return None
    p = sdir / f"scores_{stem}_{ds}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["eval_id"] = d.eval_id.astype(str)
    return d[["eval_id", "score"]].drop_duplicates("eval_id")


def main() -> None:
    rows = []
    for cell, model, data, recipe, seed, stem, sdir, sstem, budget, adapter, note in RUNS:
        hp = HYP / f"{stem}.parquet"
        if not hp.exists():
            print("MISSING hyp", stem)
            continue
        hyp = pd.read_parquet(hp)
        cfg = {"cell": cell, "model": model, "data": data, "recipe": recipe,
               "seed": seed, "token_budget": budget}
        man = {
            "model": model, "data": data, "recipe": recipe,
            "commit_hash": ce.commit_hash(), "config_hash": ce.config_hash(cfg),
            "token_budget": budget, "adapter_path": adapter,
            "source": "retrofit", "note": note,
        }
        df, m = ce.build_artefact(cell, seed, hyp,
                                  load_scores(sdir, sstem, "1688"),
                                  load_scores(sdir, sstem, "tiki"), man)
        s1688 = ce.summarise(df, "1688")
        s_tiki = ce.summarise(df, "tiki")
        rows.append({"cell": cell, "seed": seed, "model": model, "data": data,
                     "recipe": recipe,
                     "comet_da": s1688["mean"], "cov1688": s1688["coverage"],
                     "comet_qe": s_tiki["mean"], "covtiki": s_tiki["coverage"],
                     "missing": m["missing_scores"]["1688"] + m["missing_scores"]["tiki"]})
        print(f"{cell:22s} s{seed:<4d} da={s1688['mean']:.4f} cov={s1688['coverage']:.3f} "
              f"qe={s_tiki['mean']:.4f} cov={s_tiki['coverage']:.3f} missing={rows[-1]['missing']}")

    out = ROOT / ".artifacts/canonical/_registry.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
