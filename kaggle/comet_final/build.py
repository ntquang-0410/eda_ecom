#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel COMET cho final 4B trên eval_F (venv Python 3.12)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


SCORE = r'''
import json, os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
import sacrebleu

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no GPU")
DA = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
QE = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
print("models loaded")

hyp = sorted(Path("/kaggle/input").rglob("hyp_final_seed42.parquet"))[0]
df = pd.read_parquet(hyp).drop_duplicates("eval_id")
OUT = Path("/kaggle/working/scores"); OUT.mkdir(parents=True, exist_ok=True)
metrics = {}
for dataset in ("1688", "tiki"):
    part = df[df["dataset"] == dataset].copy()
    part = part[part["hypothesis"].astype(str).str.strip().ne("")]
    if dataset == "1688":
        part = part[part["reference"].astype(str).str.strip().ne("")]
        data = [{"src": str(s), "mt": str(m), "ref": str(r)}
                for s, m, r in zip(part.source, part.hypothesis, part.reference)]
        out = DA.predict(data, batch_size=16, gpus=1)
        bleu = sacrebleu.corpus_bleu(part.hypothesis.tolist(),
                                     [part.reference.tolist()], tokenize="zh").score
        chrf = sacrebleu.corpus_chrf(part.hypothesis.tolist(),
                                     [part.reference.tolist()]).score
        extra = {"bleu": round(float(bleu), 4), "chrf": round(float(chrf), 4)}
    else:
        data = [{"src": str(s), "mt": str(m)}
                for s, m in zip(part.source, part.hypothesis)]
        out = QE.predict(data, batch_size=16, gpus=1)
        extra = {}
    part["score"] = np.asarray(out.scores, dtype=float)
    part["config"] = "final"; part["seed"] = 42; part["dataset"] = dataset
    part[["eval_id", "pair_type", "score", "config", "seed", "dataset"]].to_parquet(
        OUT / f"scores_final_seed42_{dataset}.parquet", index=False)
    metrics[dataset] = {"system_score": round(float(out.system_score), 4),
                        "n": int(len(part)), **extra}
    print(dataset, metrics[dataset], flush=True)
    (OUT / "comet_scores.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
'''

CELLS = [
    md("# COMET final 4B — eval_F\n"),
    code("""import subprocess, sys, os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run(['uv', 'venv', '--python', '3.12', '/kaggle/working/venv'], check=True)
P = '/kaggle/working/venv/bin/python'
subprocess.run(['uv', 'pip', 'install', '--python', P,
                'torch', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', P,
                'numpy<2', 'pandas==2.2.3', 'pyarrow==17.0.0',
                'unbabel-comet>=2.2', 'sacrebleu'], check=True)
print('venv ready')
"""),
    code(f"""from pathlib import Path
Path('/kaggle/working/score.py').write_text({SCORE!r}, encoding='utf-8')
print('score.py written')
"""),
    code("""import subprocess
r = subprocess.run(['/kaggle/working/venv/bin/python', '/kaggle/working/score.py'],
                   capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-4000:]); r.check_returncode()
"""),
    code("""from pathlib import Path
OUT = Path('/kaggle/working/scores')
for p in sorted(OUT.glob('*.parquet')): print(p, p.stat().st_size)
print((OUT / 'comet_scores.json').read_text())
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
folder = ROOT / "kaggle" / "comet_final"
folder.mkdir(parents=True, exist_ok=True)
(folder / "comet.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {
    "id": "anhtamxinhdep/tiki-comet-final",
    "title": "Tiki COMET Final",
    "code_file": "comet.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["anhtamxinhdep/tiki-final-hyps"],
    "kernel_sources": [],
    "model_sources": [],
}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
print("wrote", folder)
