#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn bị audit 858: tạo index 858 và gói 27 dòng Tiki bổ sung cho từng run.

Output:
  data/cleaned/eval_paired_858.parquet + .json
  /tmp/audit858/hyps/<cell>__seed<seed>.parquet   (eval_id, hypothesis)
  /tmp/audit858/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path("/tmp/audit858")

# cell, seed, hyp stem
RUNS = [
    ("qwen8b_a", 42, "hyp_qwen_a_seed42"),
    ("qwen8b_b", 42, "hyp_qwen_b_seed42"),
    ("qwen8b_a_balanced", 42, "hyp_qwen8b_a_balanced_seed42"),
    ("qwen8b_b_natural", 42, "hyp_m8b_b_natural_seed42"),
    ("qwen4b_a", 42, "hyp_qwen4b_a_seed42"),
    ("qwen4b_a", 123, "hyp_qwen4b_a_seed123"),
    ("qwen4b_a", 7, "hyp_qwen4b_a_natural_seed7"),
    ("qwen4b_b", 42, "hyp_qwen4b_b_seed42"),
    ("qwen4b_b", 123, "hyp_qwen4b_b_balanced_seed123"),
    ("qwen4b_a_balanced", 42, "hyp_qwen4b_a_balanced_seed42"),
    ("qwen4b_b_natural", 42, "hyp_qwen4b_b_natural_seed42"),
    ("ecomgpt_a", 42, "hyp_ecomgpt_a_seed42"),
    ("ecomgpt_b", 42, "hyp_ecomgpt_b_seed42"),
    ("ecomgpt_a_balanced", 42, "hyp_ecomgpt_a_balanced_seed42"),
    ("moses_a", 42, "hyp_moses_a_seed42"),
    ("moses_b", 42, "hyp_moses_b_seed42"),
    ("onmt_a", 42, "hyp_onmt_a_seed42"),
    ("onmt_b", 42, "hyp_onmt_b_seed42"),
    ("nllb_a", 42, "hyp_nllb_a_seed42"),
    ("nllb_a", 7, "hyp_nllb_a_seed7"),
    ("nllb_b", 42, "hyp_nllb_b_seed42"),
    ("nllb_b", 7, "hyp_nllb_b_seed7"),
]


def index_hash(ev: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(ev.eval_id.astype(str))).encode()).hexdigest()


def main() -> None:
    full = pd.read_parquet(ROOT / "data/cleaned/eval_F_v2.parquet")
    ev = full[["eval_id", "dataset", "pair_type", "product_id", "source",
               "reference", "max_new"]].rename(columns={"pair_type": "stratum"}).copy()
    ev["eval_id"] = ev.eval_id.astype(str)
    assert len(ev) == 858, len(ev)
    assert ev.dataset.value_counts().to_dict() == {"tiki": 627, "1688": 231}
    ev = ev.sort_values(["dataset", "eval_id"]).reset_index(drop=True)
    ev.to_parquet(ROOT / "data/cleaned/eval_paired_858.parquet", index=False)
    (ROOT / "data/cleaned/eval_paired_858.json").write_text(json.dumps({
        "n": 858, "by_dataset": {"1688": 231, "tiki": 627},
        "by_stratum": {f"{d}/{s}": int(n) for (d, s), n in
                       ev.groupby(["dataset", "stratum"]).size().items()},
        "eval_ids_sha256": index_hash(ev),
        "note": "858 dòng sạch nội dung; 27 dòng Tiki bị loại khỏi 831 vì có cell hyp rỗng",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    common = pd.read_parquet(ROOT / "data/cleaned/eval_F_v2_common.parquet")
    extra = ev[~ev.eval_id.isin(set(common.eval_id.astype(str)))]
    assert len(extra) == 27 and (extra.dataset == "tiki").all()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "hyps").mkdir(parents=True)
    extra.to_parquet(STAGE / "extra_index.parquet", index=False)

    manifest = []
    for cell, seed, stem in RUNS:
        p = ROOT / f".artifacts/hyp/{stem}.parquet"
        hyp = pd.read_parquet(p)
        hyp["eval_id"] = hyp.eval_id.astype(str)
        sub = extra[["eval_id"]].merge(
            hyp[["eval_id", "hypothesis"]].drop_duplicates("eval_id"), on="eval_id", how="left")
        sub["hypothesis"] = sub.hypothesis.fillna("")
        dst = STAGE / "hyps" / f"{cell}__seed{seed}.parquet"
        sub.to_parquet(dst, index=False)
        manifest.append({"cell": cell, "seed": seed, "file": dst.name,
                         "rows": int(len(sub)),
                         "empty": int(sub.hypothesis.str.strip().eq("").sum()),
                         "source_hyp": str(p.relative_to(ROOT))})
    (STAGE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"index 858: {len(ev)} dòng, extra {len(extra)}")
    print(f"stage {STAGE}: {len(manifest)} run, tổng dòng cần chấm QE = "
          f"{sum(m['rows'] - m['empty'] for m in manifest)}")


if __name__ == "__main__":
    main()
