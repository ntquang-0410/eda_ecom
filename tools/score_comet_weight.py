#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm COMET cho các file hypothesis của thực nghiệm weight.

- 1688: COMET-da (reference-based) + BLEU/chrF
- Tiki: COMET-QE (reference-free)

Output:
  <out>/scores_<config>_seed<seed>_<dataset>.parquet
  <out>/comet_scores.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r"hyp_(?P<config>[a-z_]+)_seed(?P<seed>\d+)(?:_(?P<dataset>1688|tiki))?\.parquet")


def discover(hyp_dir: Path) -> list[tuple[str, int, str, Path]]:
    found = []
    for path in sorted(hyp_dir.glob("hyp_*.parquet")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        dataset = match.group("dataset")
        if dataset:
            found.append((match.group("config"), int(match.group("seed")), dataset, path))
            continue
        df = pd.read_parquet(path)
        if "dataset" in df.columns:
            for name, part in df.groupby("dataset"):
                tmp = hyp_dir / f"_tmp_{match.group('config')}_{match.group('seed')}_{name}.parquet"
                part.to_parquet(tmp, index=False)
                found.append((match.group("config"), int(match.group("seed")), str(name), tmp))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="/kaggle/working/weight_runs")
    ap.add_argument("--out-dir", default="/kaggle/working/weight_runs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    hyp_dir, out_dir = Path(args.hyp_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover(hyp_dir)
    if not files:
        raise FileNotFoundError(f"không có hyp_*.parquet trong {hyp_dir}")
    print(f"found {len(files)} files", flush=True)

    from comet import download_model, load_from_checkpoint
    import sacrebleu

    da = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    qe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))

    metrics = {}
    for config, seed, dataset, path in files:
        score_path = out_dir / f"scores_{config}_seed{seed}_{dataset}.parquet"
        if score_path.exists() and not args.force:
            print("skip", score_path.name, flush=True)
            continue
        df = pd.read_parquet(path).drop_duplicates("eval_id")
        df = df[df["hypothesis"].astype(str).str.strip().ne("")]
        if args.limit:
            df = df.head(args.limit)
        print(config, seed, dataset, len(df), flush=True)
        if dataset == "1688":
            df = df[df["reference"].astype(str).str.strip().ne("")]
            data = [{"src": str(s), "mt": str(m), "ref": str(r)}
                    for s, m, r in zip(df["source"], df["hypothesis"], df["reference"])]
            out = da.predict(data, batch_size=args.batch_size, gpus=args.gpus)
            df["score"] = np.asarray(out.scores, dtype=float)
            bleu = sacrebleu.corpus_bleu(df["hypothesis"].tolist(),
                                          [df["reference"].tolist()], tokenize="zh").score
            chrf = sacrebleu.corpus_chrf(df["hypothesis"].tolist(),
                                          [df["reference"].tolist()]).score
            extra = {"bleu": round(float(bleu), 4), "chrf": round(float(chrf), 4)}
        else:
            data = [{"src": str(s), "mt": str(m)}
                    for s, m in zip(df["source"], df["hypothesis"])]
            out = qe.predict(data, batch_size=args.batch_size, gpus=args.gpus)
            df["score"] = np.asarray(out.scores, dtype=float)
            extra = {}
        df["config"] = config
        df["seed"] = seed
        df["dataset"] = dataset
        df[["eval_id", "pair_type", "score", "config", "seed", "dataset"]].to_parquet(
            score_path, index=False)
        key = f"{config}::{seed}::{dataset}"
        metrics[key] = {"system_score": round(float(out.system_score), 4),
                        "n": int(len(df)), **extra}
        print(key, metrics[key], flush=True)
        (out_dir / "comet_scores.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "comet_scores.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
