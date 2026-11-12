#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate eval_F bằng checkpoint OpenNMT A 30k step (batch nhỏ tránh OOM)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


CELLS = [
    md("# OpenNMT A4 — translate eval_F\n"),
    code("""import subprocess, sys, os
subprocess.run([sys.executable,'-m','pip','install','-q','uv'],check=True)
subprocess.run(['uv','venv','--python','3.11','/kaggle/working/venv'],check=True)
P='/kaggle/working/venv/bin/python'
subprocess.run(['uv','pip','install','--python',P,'torch==2.2.2','--index-url','https://download.pytorch.org/whl/cu121'],check=True)
subprocess.run(['uv','pip','install','--python',P,'numpy<2','OpenNMT-py==3.5.1'],check=True)
subprocess.run([P,'-c','import torch,onmt;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'],check=True)
"""),
    code("""import glob
from pathlib import Path
MODEL = sorted(Path('/kaggle/input').rglob('model_step_30000.pt'))[0]
SRC = sorted(Path('/kaggle/input').rglob('eval.vi'))[0]
P='/kaggle/working/venv/bin/python'
OUT=Path('/kaggle/working/pred.zh')
cmd=[P,'-m','onmt.bin.translate','-model',str(MODEL),'-src',str(SRC),'-output',str(OUT),
     '-batch_size','4','-beam_size','5','-max_length','512','-replace_unk']
print('RUN',' '.join(cmd),flush=True)
subprocess.run(cmd,check=True)
print('lines',len(OUT.read_text().splitlines()))
"""),
    code("""import re
from pathlib import Path
import pandas as pd
DATA=Path('/kaggle/input/tiki-matrix-data')
if not (DATA/'eval_A.parquet').exists():
    DATA=sorted(Path('/kaggle/input').rglob('eval_A.parquet'))[0].parent
ev=pd.read_parquet(DATA/'eval_A.parquet')
def detok(t):
    t=re.sub(r'\\s+',' ',str(t or '')).strip()
    t=re.sub(r'(?<=[\\u4e00-\\u9fff])\\s+(?=[\\u4e00-\\u9fff])','',t)
    t=re.sub(r'\\s+(?=[，。！？；：、）】》])','',t)
    t=re.sub(r'(?<=[（【《])\\s+','',t)
    return t
lines=(Path('/kaggle/working/pred.zh').read_text().splitlines())
lines += ['']*(len(ev)-len(lines))
out=ev[['eval_id','dataset','pair_type','source','reference']].copy()
out['hypothesis']=[detok(x) for x in lines[:len(out)]]
out['config']='onmt_a'; out['seed']=42
out.to_parquet('/kaggle/working/hyp_onmt_a_seed42.parquet',index=False)
print(len(out),'empty',int(out.hypothesis.str.strip().eq('').sum()))
"""),
]
nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.11"}}, "nbformat":4,"nbformat_minor":5}
folder=ROOT/'kaggle'/'onmt_a4_translate'; folder.mkdir(parents=True,exist_ok=True)
(folder/'translate.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
meta={"id":"ntquang0410/onmt-a4-translate","title":"Onmt A4 Translate","code_file":"translate.ipynb",
      "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,
      "enable_tpu":False,"enable_internet":True,
      "dataset_sources":["ntquang0410/onmt-a4-model","ntquang0410/tiki-matrix-data"],
      "kernel_sources":[],"model_sources":[]}
(folder/'kernel-metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print('wrote',folder)
