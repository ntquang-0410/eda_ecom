#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm một run theo chuẩn canonical (index 831) và ghi artefact + manifest.

Hai chế độ:
  # 1) đã có score từng dòng (parquet: eval_id, score)
  python tools/score_canonical.py --cell qwen4b_a --seed 7 \
      --hyp .artifacts/hyp/hyp_qwen4b_a_natural_seed7.parquet \
      --da-scores scores_da.parquet --qe-scores scores_qe.parquet

  # 2) tự chấm COMET (cần GPU + internet để tải model)
  python tools/score_canonical.py --cell qwen4b_a --seed 7 \
      --hyp ... --compute-comet --adapter-path ... --base-model ...

Output:
  .artifacts/canonical/<cell>/seed<seed>.parquet
  .artifacts/canonical/<cell>/seed<seed>.manifest.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import canonical_eval as ce

ROOT = Path(__file__).resolve().parents[1]


def read_scores(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    d = pd.read_parquet(path)
    if "score" not in d.columns:
        raise SystemExit(f"{path} thiếu cột score")
    d["eval_id"] = d.eval_id.astype(str)
    return d[["eval_id", "score"]].drop_duplicates("eval_id")


def compute_comet(hyp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from comet import download_model, load_from_checkpoint
    ev = ce.load_index()
    h = hyp.copy()
    h["eval_id"] = h.eval_id.astype(str)
    h = h.drop_duplicates("eval_id").set_index("eval_id")
    h["hyp"] = [str(h.hypothesis.get(e, "") or "") for e in ev.eval_id]

    da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    qe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))

    a = ev[(ev.dataset == "1688")].copy()
    a["hyp"] = [h.hyp.get(e, "") for e in a.eval_id]
    a = a[a.hyp.str.strip().ne("")]
    out_a = da.predict([{"src": s, "mt": m, "ref": r}
                        for s, m, r in zip(a.source, a.hyp, a.reference)],
                       batch_size=16, gpus=1)
    da_df = pd.DataFrame({"eval_id": a.eval_id.to_numpy(),
                          "score": np.asarray(out_a.scores, dtype=float)})

    t = ev[(ev.dataset == "tiki")].copy()
    t["hyp"] = [h.hyp.get(e, "") for e in t.eval_id]
    t = t[t.hyp.str.strip().ne("")]
    out_t = qe.predict([{"src": s, "mt": m} for s, m in zip(t.source, t.hyp)],
                       batch_size=16, gpus=1)
    qe_df = pd.DataFrame({"eval_id": t.eval_id.to_numpy(),
                          "score": np.asarray(out_t.scores, dtype=float)})
    return da_df, qe_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--da-scores", default="")
    ap.add_argument("--qe-scores", default="")
    ap.add_argument("--compute-comet", action="store_true")
    ap.add_argument("--token-budget", type=int, default=4_000_000)
    ap.add_argument("--adapter-path", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    hyp = pd.read_parquet(args.hyp)
    if args.compute_comet:
        da, qe = compute_comet(hyp)
    else:
        da, qe = read_scores(args.da_scores), read_scores(args.qe_scores)
        if da is None or qe is None:
            print("CẢNH BÁO: thiếu score 1688 hoặc Tiki → cột tương ứng sẽ NaN")

    cfg = {"cell": args.cell, "seed": args.seed, "token_budget": args.token_budget,
           "adapter_path": args.adapter_path, "base_model": args.base_model}
    man = {
        "model": args.model or args.cell, "commit_hash": ce.commit_hash(),
        "config_hash": ce.config_hash(cfg), "token_budget": args.token_budget,
        "adapter_path": args.adapter_path or None, "base_model": args.base_model or None,
        "source": "score_canonical", "note": args.note,
    }
    df, m = ce.build_artefact(args.cell, args.seed, hyp, da, qe, man)
    s1, st = ce.summarise(df, "1688"), ce.summarise(df, "tiki")
    print(f"{args.cell} seed {args.seed}: COMET-da {s1['mean']:.4f} (cov {s1['coverage']*100:.1f}%) | "
          f"QE {st['mean']:.4f} (cov {st['coverage']*100:.1f}%) | missing "
          f"{m['missing_scores']}")


if __name__ == "__main__":
    main()
