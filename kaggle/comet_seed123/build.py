#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel COMET GPU cho seed 123 (internet ON để pip install + tải model)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


CELLS = [
    md("# COMET seed 123 — A/B\n\nInternet ON, GPU, tải model từ HuggingFace.\n"),
    code("""import os, subprocess, sys
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'unbabel-comet>=2.2', 'sacrebleu'], check=True)
import torch, json, time
from pathlib import Path
import numpy as np
import pandas as pd
print('cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('no GPU')
print(torch.cuda.get_device_name(0))
"""),
    code("""from comet import download_model, load_from_checkpoint
import sacrebleu
DA = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
QE = load_from_checkpoint(download_model('Unbabel/wmt20-comet-qe-da'))
print('models loaded')
"""),
    code("""HYP = Path('/kaggle/input/tiki-seed123-hyps')
OUT = Path('/kaggle/working/scores'); OUT.mkdir(parents=True, exist_ok=True)
metrics = {}
for suite in ('A', 'B'):
    out_dir = OUT / suite; out_dir.mkdir(exist_ok=True)
    for path in sorted(HYP.glob(f'hyp_{suite}_*.parquet')):
        df = pd.read_parquet(path).drop_duplicates('eval_id')
        config = str(df['config'].iloc[0]); seed = int(df['seed'].iloc[0])
        for dataset in ('1688', 'tiki'):
            part = df[df['dataset'] == dataset].copy()
            part = part[part['hypothesis'].astype(str).str.strip().ne('')]
            if dataset == '1688':
                part = part[part['reference'].astype(str).str.strip().ne('')]
                data = [{'src': str(s), 'mt': str(m), 'ref': str(r)}
                        for s, m, r in zip(part.source, part.hypothesis, part.reference)]
                out = DA.predict(data, batch_size=16, gpus=1)
                bleu = sacrebleu.corpus_bleu(part.hypothesis.tolist(),
                                             [part.reference.tolist()], tokenize='zh').score
                chrf = sacrebleu.corpus_chrf(part.hypothesis.tolist(),
                                             [part.reference.tolist()]).score
                extra = {'bleu': round(float(bleu), 4), 'chrf': round(float(chrf), 4)}
            else:
                data = [{'src': str(s), 'mt': str(m)}
                        for s, m in zip(part.source, part.hypothesis)]
                out = QE.predict(data, batch_size=16, gpus=1)
                extra = {}
            part['score'] = np.asarray(out.scores, dtype=float)
            part['config'] = config; part['seed'] = seed; part['dataset'] = dataset
            part[['eval_id', 'pair_type', 'score', 'config', 'seed', 'dataset']].to_parquet(
                out_dir / f'scores_{config}_seed{seed}_{dataset}.parquet', index=False)
            key = f'{suite}::{config}::{seed}::{dataset}'
            metrics[key] = {'system_score': round(float(out.system_score), 4),
                            'n': int(len(part)), **extra}
            print(key, metrics[key], flush=True)
            (OUT / 'comet_scores.json').write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metrics, ensure_ascii=False, indent=2))
"""),
    code("""for p in sorted(OUT.rglob('*.parquet')): print(p, p.stat().st_size)
print((OUT / 'comet_scores.json').read_text()[:500])
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

folder = ROOT / "kaggle" / "comet_seed123"
folder.mkdir(parents=True, exist_ok=True)
(folder / "comet.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {
    "id": "anhtamxinhdep/tiki-comet-seed123",
    "title": "Tiki COMET Seed123",
    "code_file": "comet.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["anhtamxinhdep/tiki-seed123-hyps"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaRtxPro6000",
}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
print("wrote", folder)
