#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score COMET trên CPU cho hypothesis mới (NLLB/control) — dùng khi không có GPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default=".artifacts/hyp")
    ap.add_argument("--out", default=".artifacts/comet_local/scores")
    ap.add_argument("--labels", default="")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    from comet import download_model, load_from_checkpoint

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    qe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
    labels = [x for x in args.labels.split(",") if x] or None
    for p in sorted(Path(args.hyp_dir).glob("hyp_*_seed42.parquet")):
        label = p.name[len("hyp_"):-len("_seed42.parquet")]
        if labels and label not in labels:
            continue
        df = pd.read_parquet(p)
        for ds in ("1688", "tiki"):
            part = df[(df.dataset == ds) & df.hypothesis.fillna("").str.strip().ne("")].copy()
            if ds == "1688":
                part = part[part.reference.fillna("").str.strip().ne("")]
                data = [{"src": str(s), "mt": str(m), "ref": str(r)}
                        for s, m, r in zip(part.source, part.hypothesis, part.reference)]
                res = da.predict(data, batch_size=args.batch, gpus=0)
            else:
                data = [{"src": str(s), "mt": str(m)} for s, m in zip(part.source, part.hypothesis)]
                res = qe.predict(data, batch_size=args.batch, gpus=0)
            part["score"] = np.asarray(res.scores, dtype=float)
            part["config"] = label; part["seed"] = 42; part["dataset"] = ds
            part[["eval_id", "pair_type", "score", "config", "seed", "dataset"]].to_parquet(
                out / f"scores_{label}_seed42_{ds}.parquet", index=False)
            print(label, ds, len(part), round(float(res.system_score), 4), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
