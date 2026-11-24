"""Sinh kernel chấm COMET cho các cell seed 7 (4B B-balanced, NLLB A/B).

Nguồn hyp: dataset nguynthanhquang/tiki-s7-hyps (3 parquet 1.500 dòng).
Dùng: python kaggle/comet_s7/build.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "tiki-comet-s7"

SCORE_PY = r'''
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

HYP = None
for cand in sorted(Path("/kaggle/input").rglob("hyp_*.parquet")):
    if cand.parent.name in ("tiki-s7-hyps", "tiki-seed123-hyps"):
        HYP = cand.parent
        break
assert HYP, "khong thay dataset hyp"
OUT = Path("/kaggle/working/scores"); OUT.mkdir(parents=True, exist_ok=True)
metrics = {}
for path in sorted(HYP.glob("hyp_*.parquet")):
    df = pd.read_parquet(path).drop_duplicates("eval_id")
    config = str(df["config"].iloc[0]); seed = int(df["seed"].iloc[0])
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
        part["config"] = config; part["seed"] = seed; part["dataset"] = dataset
        part[["eval_id", "pair_type", "score", "config", "seed", "dataset"]].to_parquet(
            OUT / f"scores_{config}_seed{seed}_{dataset}.parquet", index=False)
        key = f"{config}::{seed}::{dataset}"
        metrics[key] = {"system_score": round(float(out.system_score), 4),
                        "n": int(len(part)), **extra}
        print(key, metrics[key], flush=True)
        (OUT / "comet_scores.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
'''

CELLS = [
    ("markdown", "# COMET seed 7 — 4B B-balanced + NLLB A/B\n\n"
                 "Venv Python 3.12 riêng (comet yêu cầu numpy<2), GPU + internet."),
    ("code", """import subprocess, sys, os
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
    ("code", f"""from pathlib import Path
Path('/kaggle/working/score.py').write_text({SCORE_PY!r}, encoding='utf-8')
print('score.py', len({SCORE_PY!r}))
"""),
    ("code", """import subprocess
r = subprocess.run(['/kaggle/working/venv/bin/python', '/kaggle/working/score.py'],
                   capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-4000:])
r.check_returncode()
"""),
    ("code", """from pathlib import Path
OUT = Path('/kaggle/working/scores')
for p in sorted(OUT.rglob('*.parquet')): print(p, p.stat().st_size)
print((OUT / 'comet_scores.json').read_text()[:2000])
"""),
]

META = {
    "id": f"nguynthanhquang/{SLUG}",
    "title": "Tiki COMET S7",
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["nguynthanhquang/tiki-s7-hyps"],
    "kernel_sources": [],
    "model_sources": [],
}


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / META["code_file"]).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "kernel-metadata.json").write_text(json.dumps(META, indent=2), encoding="utf-8")
    print("wrote", HERE / META["code_file"])


if __name__ == "__main__":
    main()
