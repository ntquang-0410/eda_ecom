#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm lại toàn bộ Qwen3-8B A-balanced trên index 858: COMET-da (1688) + COMET-QE (Tiki).

Chạy trên Colab A100:
  /content/work/venv_comet/bin/python /content/work/score_8b_bal.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd

STAGE = Path("/content/work/audit8b")


def main() -> None:
    import torch
    from comet import download_model, load_from_checkpoint

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    idx = pd.read_parquet(STAGE / "eval_paired_858.parquet")
    idx["eval_id"] = idx.eval_id.astype(str)
    hyp = pd.read_parquet(STAGE / "hyp_qwen8b_a_balanced_seed42.parquet")
    hyp["eval_id"] = hyp.eval_id.astype(str)
    h = hyp.drop_duplicates("eval_id").set_index("eval_id")

    da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    qe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
    print("models loaded", flush=True)

    a = idx[idx.dataset == "1688"].copy()
    a["hyp"] = [str(h.hypothesis.get(e, "") or "") for e in a.eval_id]
    a = a[a.hyp.str.strip().ne("")]
    print("1688 rows", len(a), flush=True)
    out = da.predict([{"src": s, "mt": m, "ref": r}
                      for s, m, r in zip(a.source, a.hyp, a.reference)], batch_size=16, gpus=1)
    pd.DataFrame({"eval_id": a.eval_id.to_numpy(),
                  "score": np.asarray(out.scores, dtype=float)}).to_parquet(STAGE / "da.parquet", index=False)
    print("DA", round(float(out.system_score), 4), flush=True)

    t = idx[idx.dataset == "tiki"].copy()
    t["hyp"] = [str(h.hypothesis.get(e, "") or "") for e in t.eval_id]
    t = t[t.hyp.str.strip().ne("")]
    print("tiki rows", len(t), flush=True)
    out = qe.predict([{"src": s, "mt": m} for s, m in zip(t.source, t.hyp)],
                     batch_size=16, gpus=1)
    pd.DataFrame({"eval_id": t.eval_id.to_numpy(),
                  "score": np.asarray(out.scores, dtype=float)}).to_parquet(STAGE / "qe.parquet", index=False)
    print("QE", round(float(out.system_score), 4), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
