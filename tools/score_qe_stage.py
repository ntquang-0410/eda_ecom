#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm COMET-QE cho một stage bất kỳ (audit858, reinfer, ...).

Input stage:
  <stage>/extra_index.parquet   (eval_id, source, ...)
  <stage>/manifest.json         [{cell, seed, file, ...}]
  <stage>/hyps/<file>           (eval_id, hypothesis)
Output:
  <stage>/qe_extra.parquet      (cell, seed, eval_id, score)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    stage = Path(args.stage)

    import torch
    from comet import download_model, load_from_checkpoint

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    idx = pd.read_parquet(stage / "extra_index.parquet")
    idx["eval_id"] = idx.eval_id.astype(str)
    src = dict(zip(idx.eval_id, idx.source))
    man = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    qe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
    print("QE loaded", flush=True)

    out = []
    for m in man:
        d = pd.read_parquet(stage / "hyps" / m["file"])
        d["eval_id"] = d.eval_id.astype(str)
        d = d[d.hypothesis.astype(str).str.strip().ne("")]
        if not len(d):
            print(f"{m['cell']} s{m['seed']}: 0 hyp", flush=True)
            continue
        data = [{"src": src[e], "mt": str(h)} for e, h in zip(d.eval_id, d.hypothesis)]
        res = qe.predict(data, batch_size=16, gpus=1)
        for e, s in zip(d.eval_id, np.asarray(res.scores, dtype=float)):
            out.append({"cell": m["cell"], "seed": int(m["seed"]), "eval_id": e, "score": float(s)})
        print(f"{m['cell']} s{m['seed']}: {len(d)} dòng", flush=True)

    pd.DataFrame(out).to_parquet(stage / "qe_extra.parquet", index=False)
    print("wrote", stage / "qe_extra.parquet", flush=True)


if __name__ == "__main__":
    main()
