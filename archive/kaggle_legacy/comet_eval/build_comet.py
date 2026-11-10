#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COMET cho output Qwen3-4B trên 1688 (có ref) và Tiki (QE)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# COMET cho Qwen3-4B\n",
    "\n",
    "- 1688: `Unbabel/wmt22-comet-da` (reference-based)\n",
    "- Tiki: `Unbabel/wmt22-cometkiwi-da` (QE, không reference)\n",
]})

code("""import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'unbabel-comet', 'sacrebleu'], check=False)
import pandas as pd, torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
""")

code("""INPUT = Path('/kaggle/input/tiki-4b-hypotheses')
if not (INPUT / 'hyp_1688.parquet').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/hyp_1688.parquet', recursive=True)
    INPUT = Path(hits[0]).parent if hits else INPUT
print('INPUT', INPUT)
h1688 = pd.read_parquet(INPUT / 'hyp_1688.parquet')
htiki = pd.read_parquet(INPUT / 'hyp_tiki.parquet')
print('1688', len(h1688), '| tiki', len(htiki))
print(h1688.head(2).to_string(max_colwidth=50))
""")

code("""from comet import download_model, load_from_checkpoint
import sacrebleu

# 1688 reference-based
model_path = download_model('Unbabel/wmt22-comet-da')
comet_model = load_from_checkpoint(model_path)
data = [{'src': str(s), 'mt': str(h), 'ref': str(r)}
        for s, h, r in zip(h1688['source'], h1688['hypothesis'], h1688['reference'])]
out = comet_model.predict(data, batch_size=8, gpus=1)
h1688['comet'] = out.scores
bleu = sacrebleu.corpus_bleu(h1688['hypothesis'].tolist(), [h1688['reference'].tolist()], tokenize='zh').score
chrf = sacrebleu.corpus_chrf(h1688['hypothesis'].tolist(), [h1688['reference'].tolist()]).score
res1688 = {'n': len(h1688), 'comet': round(float(out.system_score), 4),
           'bleu': round(float(bleu), 4), 'chrf': round(float(chrf), 4)}
print('1688', res1688)
""")

code("""# Tiki QE
qe_path = download_model('Unbabel/wmt22-cometkiwi-da')
qe_model = load_from_checkpoint(qe_path)
data = [{'src': str(s), 'mt': str(h)} for s, h in zip(htiki['source'], htiki['hypothesis'])]
out = qe_model.predict(data, batch_size=8, gpus=1)
htiki['comet_qe'] = out.scores
restiki = {'n': len(htiki), 'comet_qe': round(float(out.system_score), 4)}
print('tiki', restiki)
""")

code("""h1688[['id','source','hypothesis','reference','comet']].to_parquet('/kaggle/working/scored_1688.parquet', index=False)
htiki[['id','source','hypothesis','comet_qe']].to_parquet('/kaggle/working/scored_tiki.parquet', index=False)
result = {'system': 'qwen3-4b', 'g1688': res1688, 'tiki': restiki}
Path('/kaggle/working/comet_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "comet_eval_4b.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
