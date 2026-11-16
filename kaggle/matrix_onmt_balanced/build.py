#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenNMT-B chạy đúng quota balanced 4M (52.131 draw / 25.453 unique),
cùng 30.000 step với OpenNMT-A, trên GPU L4x4."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


SETUP = r'''
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run(['uv', 'venv', '--python', '3.11', '/kaggle/working/venv'], check=True)
P = '/kaggle/working/venv/bin/python'
subprocess.run(['uv', 'pip', 'install', '--python', P, 'torch==2.2.2',
                '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', P, 'OpenNMT-py==3.5.1'], check=True)
subprocess.run([P, '-c', 'import torch, onmt; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'], check=True)
print('venv ready')
'''

DATA = r'''
import json, re
from pathlib import Path
import pandas as pd

DATA = Path('/kaggle/input/tiki-matrix-data')
if not (DATA / 'units_B.parquet').exists():
    hits = sorted(Path('/kaggle/input').rglob('units_B.parquet'))
    DATA = hits[0].parent
CTRL = Path('/kaggle/input/tiki-matrix-controls')
if not (CTRL / 'sample_B_balanced_seed42.parquet').exists():
    hits = sorted(Path('/kaggle/input').rglob('sample_B_balanced_seed42.parquet'))
    CTRL = hits[0].parent

WORK = Path('/kaggle/working/onmt_bal'); WORK.mkdir(exist_ok=True)
PUNCT = re.compile(r'([,.;:!?()\[\]{}"“”‘’、，。；：！？（）【】《》])')
def tok_vi(text):
    t = PUNCT.sub(r' \1 ', str(text or '').strip())
    return re.sub(r'\s+', ' ', t).strip()
def tok_zh(text):
    t = PUNCT.sub(r' \1 ', str(text or '').strip())
    t = re.sub(r'([\u4e00-\u9fff])', r' \1 ', t)
    return re.sub(r'\s+', ' ', t).strip()

u = pd.read_parquet(DATA / 'units_B.parquet').reset_index(drop=True)
s = pd.read_parquet(CTRL / 'sample_B_balanced_seed42.parquet')
idx = s['row_index'].astype(int).tolist()
train = u.iloc[idx].copy()
valid = u[u.split == 'dev'].copy()
ev = pd.read_parquet(DATA / 'eval_B.parquet')
(WORK / 'train.vi').write_text('\n'.join(tok_vi(x) for x in train.source_text) + '\n', encoding='utf-8')
(WORK / 'train.zh').write_text('\n'.join(tok_zh(x) for x in train.target_text) + '\n', encoding='utf-8')
(WORK / 'valid.vi').write_text('\n'.join(tok_vi(x) for x in valid.source_text) + '\n', encoding='utf-8')
(WORK / 'valid.zh').write_text('\n'.join(tok_zh(x) for x in valid.target_text) + '\n', encoding='utf-8')
(WORK / 'eval.vi').write_text('\n'.join(tok_vi(x) for x in ev.source) + '\n', encoding='utf-8')
train_tokens = sum(len(x.split()) for x in (WORK / 'train.vi').read_text().splitlines()) + \
               sum(len(x.split()) for x in (WORK / 'train.zh').read_text().splitlines())
stats = {'train_rows': int(len(train)), 'valid_rows': int(len(valid)),
         'train_tokens': int(train_tokens), 'draws': int(len(s)),
         'unique_rows': int(s.row_index.nunique()), 'eval_rows': int(len(ev)),
         'sample_groups': {str(k): int(v) for k, v in s.group.value_counts().items()}}
(WORK / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print(json.dumps(stats, ensure_ascii=False, indent=2))
'''

TRAIN = r'''
import json, subprocess
from pathlib import Path
P = '/kaggle/working/venv/bin/python'
WORK = Path('/kaggle/working/onmt_bal')
stats = json.loads((WORK / 'stats.json').read_text())
vocab_yaml = WORK / 'vocab.yaml'
vocab_yaml.write_text(f"""src_vocab: {WORK}/vocab.vi
tgt_vocab: {WORK}/vocab.zh
src_vocab_size: 32000
tgt_vocab_size: 32000
data:
  corpus_1:
    path_src: {WORK}/train.vi
    path_tgt: {WORK}/train.zh
""", encoding='utf-8')
subprocess.run([P, '-m', 'onmt.bin.build_vocab', '-config', str(vocab_yaml), '-save_data', str(WORK),
                '-src_vocab_size', '32000', '-tgt_vocab_size', '32000', '-n_sample', '200000'], check=True)
cfg = WORK / 'train.yaml'
cfg.write_text(f"""data:
  corpus_1:
    path_src: {WORK}/train.vi
    path_tgt: {WORK}/train.zh
    weight: 1
  valid:
    path_src: {WORK}/valid.vi
    path_tgt: {WORK}/valid.zh
save_model: {WORK}/model
src_vocab: {WORK}/vocab.vi
tgt_vocab: {WORK}/vocab.zh
world_size: 1
gpu_ranks: [0]
batch_type: tokens
batch_size: 2048
accum_count: [4]
num_workers: 0
bucket_size: 1024
train_steps: 30000
valid_steps: 500
save_checkpoint_steps: 1000
keep_checkpoint: 2
report_every: 200
warmup_steps: 2000
learning_rate: 2.0
decay_method: noam
adam_beta2: 0.998
label_smoothing: 0.1
dropout: 0.1
attention_dropout: 0.1
seed: 42
model_type: text
encoder_type: transformer
decoder_type: transformer
hidden_size: 256
word_vec_size: 256
heads: 8
layers: 4
transformer_ff: 1024
""", encoding='utf-8')
log = (WORK / 'train.log').open('w', encoding='utf-8')
subprocess.run([P, '-m', 'onmt.bin.train', '-config', str(cfg)], stdout=log, stderr=subprocess.STDOUT, check=True)
log.close()
ckpt = sorted(WORK.glob('model_step_*.pt'))[-1]
pred = WORK / 'pred.zh'
subprocess.run([P, '-m', 'onmt.bin.translate', '-model', str(ckpt), '-src', str(WORK/'eval.vi'),
                '-output', str(pred), '-gpu', '0', '-batch_size', '4',
                '-beam_size', '5', '-max_length', '512', '-replace_unk'], check=True)
summary = {'train_steps': 30000, 'epochs_token_pass': round(30000*8192 / max(1, stats['train_tokens']), 2),
           'train_tokens': stats['train_tokens'], 'token_pass': 30000*8192,
           'checkpoint': ckpt.name, 'pred_lines': len(pred.read_text().splitlines()),
           'eval_src_over_512': int(sum(len(x.split()) > 512 for x in (WORK/'eval.vi').read_text().splitlines()))}
(WORK / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
'''

HYP = r'''
import re
from pathlib import Path
import pandas as pd
WORK = Path('/kaggle/working/onmt_bal')
DATA = Path('/kaggle/input/tiki-matrix-data')
if not (DATA/'eval_B.parquet').exists():
    DATA = sorted(Path('/kaggle/input').rglob('eval_B.parquet'))[0].parent
ev = pd.read_parquet(DATA/'eval_B.parquet')
def detok(t):
    t = re.sub(r'\s+', ' ', str(t or '')).strip()
    t = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', t)
    t = re.sub(r'\s+(?=[，。！？；：、）】》])', '', t)
    t = re.sub(r'(?<=[（【《])\s+', '', t)
    return t
lines = (WORK/'pred.zh').read_text().splitlines()
lines += [''] * (len(ev) - len(lines))
out = ev[['eval_id','dataset','pair_type','source','reference']].copy()
out['hypothesis'] = [detok(x) for x in lines[:len(out)]]
out['config'] = 'onmt_b_balanced'; out['seed'] = 42
out.to_parquet('/kaggle/working/hyp_onmt_b_balanced_seed42.parquet', index=False)
print(len(out), 'empty', int(out.hypothesis.str.strip().eq('').sum()))
'''

CELLS = [md("# OpenNMT-B balanced quota (52.131 draw / 25.453 unique), 30k step\n"),
         code(SETUP), code(DATA), code(TRAIN), code(HYP)]
nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
folder = ROOT / "kaggle" / "matrix_onmt_balanced"; folder.mkdir(parents=True, exist_ok=True)
(folder / "onmt_balanced.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {"id": "nguynthanhquang/tiki-onmt-b-balanced", "title": "Tiki Onmt B Balanced",
        "code_file": "onmt_balanced.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/tiki-matrix-data", "nguynthanhquang/tiki-matrix-controls",
                            "metric/gemma-4-developer-agent-wheelhouse"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"], "kernel_sources": [], "model_sources": [],
        "machine_shape": "NvidiaL4"}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", folder)
