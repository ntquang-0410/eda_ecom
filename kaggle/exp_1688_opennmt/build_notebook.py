#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenNMT-py baseline trên 1688 (tokenized), eval test_small."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# 1688-only — OpenNMT-py (NMT)\n",
    "\n",
    "Transformer nhỏ (3 layer, 256 hidden), train trên 1688 train tokenized,\n",
    "dịch test_small, BLEU/chrF.\n",
]})

code("""import json, math, os, re, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'OpenNMT-py', 'sacrebleu'], check=False)
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
""")

code("""INPUT = Path('/kaggle/input/tiki-1688-exp')
if not (INPUT / 'train.tok.vi').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/train.tok.vi', recursive=True)
    INPUT = Path(hits[0]).parent if hits else INPUT
print('INPUT', INPUT)

WORK = Path('/kaggle/working/onmt')
WORK.mkdir(parents=True, exist_ok=True)
for name in ['train.tok.vi','train.tok.zh','dev.tok.vi','dev.tok.zh','test_small.tok.vi','test_small.zh']:
    (WORK / name).write_text((INPUT / name).read_text(encoding='utf-8'), encoding='utf-8')
print('files copied')
""")

code("""%%bash
cd /kaggle/working/onmt
onmt_preprocess -train_src train.tok.vi -train_tgt train.tok.zh \\
  -valid_src dev.tok.vi -valid_tgt dev.tok.zh \\
  -save_data prep -src_vocab_size 32000 -tgt_vocab_size 32000 -share_vocab \\
  > preprocess.log 2>&1
tail -5 preprocess.log
""")

code("""%%bash
cd /kaggle/working/onmt
onmt_train -data prep -save_model model/model \\
  -train_steps 12000 -valid_steps 1000 -save_checkpoint_steps 1000 \\
  -gpu_ranks 0 -batch_type tokens -batch_size 4096 \\
  -encoder_type transformer -decoder_type transformer -layers 3 -heads 8 \\
  -hidden_size 256 -transformer_ff 1024 -dropout 0.1 -attention_dropout 0.1 \\
  -warmup_steps 2000 -learning_rate 2 -label_smoothing 0.1 \\
  > train.log 2>&1
tail -10 train.log
""")

code("""%%bash
cd /kaggle/working/onmt
onmt_translate -model model/model_step_12000.pt -src test_small.tok.vi \\
  -output pred.tok.zh -gpu 0 -beam_size 5 -batch_size 16 \\
  > translate.log 2>&1
tail -5 translate.log
wc -l pred.tok.zh
""")

code("""import sacrebleu
WORK = Path('/kaggle/working/onmt')
pred = [re.sub(r'\\s+', '', line.strip()) for line in (WORK / 'pred.tok.zh').read_text(encoding='utf-8').splitlines()]
ref = [line.strip() for line in (WORK / 'test_small.zh').read_text(encoding='utf-8').splitlines()]
n = min(len(pred), len(ref))
pred, ref = pred[:n], ref[:n]

bleu = sacrebleu.corpus_bleu(pred, [ref], tokenize='zh').score
chrf = sacrebleu.corpus_chrf(pred, [ref]).score
result = {'system': 'opennmt', 'n': n, 'bleu': round(bleu, 4), 'chrf': round(chrf, 4)}
Path('/kaggle/working/result_opennmt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
pd.DataFrame({'reference': ref, 'hypothesis': pred}).to_parquet('/kaggle/working/opennmt_outputs.parquet', index=False)
print(result)
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "exp_1688_opennmt.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
