#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm COMET-QE cho 27 dòng Tiki bổ sung (audit 858). Chạy trên Colab A100.

Input : /content/work/audit858/{extra_index.parquet, manifest.json, hyps/*.parquet}
Output: /content/work/audit858/qe_extra.parquet (cell, seed, eval_id, score)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd

STAGE = Path("/content/work/audit858")


def main() -> None:
    import torch
    from comet import download_model, load_from_checkpoint

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    idx = pd.read_parquet(STAGE / "extra_index.parquet")
    idx["eval_id"] = idx.eval_id.astype(str)
    src = dict(zip(idx.eval_id, idx.source))
    man = json.loads((STAGE / "manifest.json").read_text(encoding="utf-8"))

    qe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
    print("QE loaded", flush=True)

    out = []
    for m in man:
        d = pd.read_parquet(STAGE / "hyps" / m["file"])
        d["eval_id"] = d.eval_id.astype(str)
        d = d[d.hypothesis.astype(str).str.strip().ne("")]
        if not len(d):
            print(f"{m['cell']} s{m['seed']}: 0 hyp — bỏ qua", flush=True)
            continue
        data = [{"src": src[e], "mt": str(h)} for e, h in zip(d.eval_id, d.hypothesis)]
        res = qe.predict(data, batch_size=16, gpus=1)
        for e, s in zip(d.eval_id, np.asarray(res.scores, dtype=float)):
            out.append({"cell": m["cell"], "seed": m["seed"], "eval_id": e, "score": float(s)})
        print(f"{m['cell']} s{m['seed']}: {len(d)} dòng", flush=True)

    df = pd.DataFrame(out)
    df.to_parquet(STAGE / "qe_extra.parquet", index=False)
    print(df.groupby(["cell", "seed"]).size().to_string())
    print("wrote", STAGE / "qe_extra.parquet", flush=True)


if __name__ == "__main__":
    main()
