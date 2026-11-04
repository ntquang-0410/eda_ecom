#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Moses SMT baseline trên 1688 (CPU), eval test_small."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# 1688-only — Moses (SMT)\n",
    "\n",
    "Build Moses từ source, alignment bằng eflomal + atools, LM KenLM,\n",
    "train phrase-based trên 1688 train, dịch test_small.\n",
]})

code("""%%bash
set -e
apt-get update -qq
apt-get install -y -qq build-essential git libboost-all-dev zlib1g-dev libbz2-dev cmake liblzma-dev > /kaggle/working/apt.log 2>&1
echo apt done
""")

code("""%%bash
set -e
cd /kaggle/working
if [ ! -d mosesdecoder ]; then git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git; fi
cd mosesdecoder
./bjam -j4 --with-boost=/usr > /kaggle/working/moses_build.log 2>&1
echo build done
ls bin/moses bin/lmplz
""")

code("""%%bash
set -e
cd /kaggle/working
if [ ! -d eflomal ]; then git clone --depth 1 https://github.com/robertostling/eflomal.git; fi
g++ -O3 -o /kaggle/working/atools eflomal/atools/atools.cc
ls -l /kaggle/working/atools
""")

code("""%%bash
set -e
mkdir -p /kaggle/working/exp
cp /kaggle/input/tiki-1688-exp/train.tok.vi /kaggle/working/exp/train.vi
cp /kaggle/input/tiki-1688-exp/train.tok.zh /kaggle/working/exp/train.zh
cp /kaggle/input/tiki-1688-exp/test_small.tok.vi /kaggle/working/exp/test_small.vi
cp /kaggle/input/tiki-1688-exp/test_small.zh /kaggle/working/exp/test_small.zh
wc -l /kaggle/working/exp/*
""")

code("""%%bash
set -e
pip install -q eflomal > /kaggle/working/pip_eflomal.log 2>&1
EG=$(python -c "import eflomal, os; print(os.path.join(os.path.dirname(eflomal.__file__),'bin','eflomal'))")
echo "eflomal: $EG"
cd /kaggle/working/exp
$EG -s train.vi -t train.zh -f fwd --priors 3 --iterations 4 > eflomal_fwd.log 2>&1
$EG -s train.zh -t train.vi -f rev --priors 3 --iterations 4 > eflomal_rev.log 2>&1
/kaggle/working/atools -i fwd -j rev -c grow-diag-final-and > sym.grow-diag-final-and
wc -l fwd rev sym.grow-diag-final-and
""")

code("""%%bash
set -e
cd /kaggle/working/exp
zcat -f train.zh | /kaggle/working/mosesdecoder/bin/lmplz -o 3 > lm.arpa 2> lm.log
gzip -f lm.arpa
ls -lh lm.arpa.gz
""")

code("""%%bash
set -e
cd /kaggle/working/exp
/kaggle/working/mosesdecoder/scripts/training/train-model.perl \\
  -root-directory /kaggle/working/exp/work \\
  -corpus /kaggle/working/exp/train \\
  -f zh -e vi \\
  -alignment grow-diag-final-and \\
  -alignment-file /kaggle/working/exp/sym \\
  -first-step 4 \\
  -lm 0:3:/kaggle/working/exp/lm.arpa.gz \\
  -max-phrase-length 7 \\
  > /kaggle/working/exp/train_model.log 2>&1
tail -20 /kaggle/working/exp/train_model.log
ls -l /kaggle/working/exp/work/model/moses.ini
""")

code("""%%bash
set -e
cd /kaggle/working/exp
/kaggle/working/mosesdecoder/bin/moses -f work/model/moses.ini -threads 4 \\
  < test_small.vi > pred.tok.zh 2> moses_decode.log
wc -l pred.tok.zh
tail -3 moses_decode.log
""")

code("""import json, re
from pathlib import Path
import pandas as pd, sacrebleu
WORK = Path('/kaggle/working/exp')
pred = [re.sub(r'\\s+', '', l.strip()) for l in (WORK / 'pred.tok.zh').read_text(encoding='utf-8').splitlines()]
ref = [l.strip() for l in (WORK / 'test_small.zh').read_text(encoding='utf-8').splitlines()]
n = min(len(pred), len(ref))
pred, ref = pred[:n], ref[:n]
bleu = sacrebleu.corpus_bleu(pred, [ref], tokenize='zh').score
chrf = sacrebleu.corpus_chrf(pred, [ref]).score
result = {'system': 'moses', 'n': n, 'bleu': round(bleu, 4), 'chrf': round(chrf, 4)}
Path('/kaggle/working/result_moses.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
pd.DataFrame({'reference': ref, 'hypothesis': pred}).to_parquet('/kaggle/working/moses_outputs.parquet', index=False)
print(result)
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "exp_1688_moses.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
