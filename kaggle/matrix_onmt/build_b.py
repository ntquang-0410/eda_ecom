#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel OpenNMT-py chuẩn (10 epochs) cho matrix A/B.

Kaggle image dùng Python 3.13, OpenNMT-py pin torch<2.3 và pyonmttok không có
wheel cp313. Kernel tạo venv Python 3.11, cài torch 2.2.2+cu121 + OpenNMT-py 3.5.1.
"""

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
import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run(['uv', 'venv', '--python', '3.11', '/kaggle/working/venv'], check=True)
P = '/kaggle/working/venv/bin/python'
subprocess.run(['uv', 'pip', 'install', '--python', P,
                'torch==2.2.2', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', P, 'OpenNMT-py==3.5.1'], check=True)
subprocess.run([P, '-c', 'import torch, pyonmttok, onmt; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'], check=True)
print('venv ready')
'''

DATA = r'''
import json, re
from pathlib import Path
import pandas as pd

DATA = Path('/kaggle/input/tiki-matrix-data')
if not (DATA / 'units_A.parquet').exists():
    hits = sorted(Path('/kaggle/input').rglob('units_A.parquet'))
    DATA = hits[0].parent
WORK = Path('/kaggle/working/onmt'); WORK.mkdir(exist_ok=True)

PUNCT = re.compile(r'([,.;:!?()\[\]{}"“”‘’、，。；：！？（）【】《》])')
def tok_vi(text):
    t = PUNCT.sub(r' \1 ', str(text or '').strip())
    return re.sub(r'\s+', ' ', t).strip()
def tok_zh(text):
    t = PUNCT.sub(r' \1 ', str(text or '').strip())
    t = re.sub(r'([\u4e00-\u9fff])', r' \1 ', t)
    return re.sub(r'\s+', ' ', t).strip()

stats = {}
for ds in ('A', 'B'):
    u = pd.read_parquet(DATA / f'units_{ds}.parquet')
    ev = pd.read_parquet(DATA / f'eval_{ds}.parquet')
    train = u[u.split == 'train']
    valid = u[u.split == 'dev']
    d = WORK / ds; d.mkdir(exist_ok=True)
    for name, part in (('train', train), ('valid', valid)):
        (d / f'{name}.vi').write_text('\n'.join(tok_vi(x) for x in part.source_text) + '\n', encoding='utf-8')
        (d / f'{name}.zh').write_text('\n'.join(tok_zh(x) for x in part.target_text) + '\n', encoding='utf-8')
    (d / 'eval.vi').write_text('\n'.join(tok_vi(x) for x in ev.source) + '\n', encoding='utf-8')
    n_tok = sum(len(l.split()) for l in (d / 'train.vi').read_text().splitlines()) + \
            sum(len(l.split()) for l in (d / 'train.zh').read_text().splitlines())
    stats[ds] = {'train_rows': len(train), 'valid_rows': len(valid), 'train_tokens': n_tok,
                 'eval_rows': len(ev),
                 'eval_src_over_512': int(sum(len(l.split()) > 512 for l in (d / 'eval.vi').read_text().splitlines()))}
    print(ds, stats[ds])
Path('/kaggle/working/onmt_data_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
'''

TRAIN_A = r'''
import subprocess, math, json
from pathlib import Path
P = '/kaggle/working/venv/bin/python'
WORK = Path('/kaggle/working/onmt')
stats = json.loads(Path('/kaggle/working/onmt_data_stats.json').read_text())

def run(ds, epochs=10):
    d = WORK / ds
    steps_per_epoch = max(1, math.ceil(stats[ds]['train_tokens'] / (4096 * 2)))
    train_steps = min(30000, steps_per_epoch * epochs)
    vocab_yaml = WORK / f'vocab_{ds}.yaml'
    vocab_yaml.write_text(f"""src_vocab: {d}/vocab.vi
tgt_vocab: {d}/vocab.zh
src_vocab_size: 32000
tgt_vocab_size: 32000
data:
  corpus_1:
    path_src: {d}/train.vi
    path_tgt: {d}/train.zh
""", encoding='utf-8')
    subprocess.run([P, '-m', 'onmt.bin.build_vocab', '-config', str(vocab_yaml), '-save_data', str(d), '-n_sample', '200000'], check=True)

    cfg = WORK / f'train_{ds}.yaml'
    cfg.write_text(f"""data:
  corpus_1:
    path_src: {d}/train.vi
    path_tgt: {d}/train.zh
    weight: 1
  valid:
    path_src: {d}/valid.vi
    path_tgt: {d}/valid.zh
save_model: {d}/model
src_vocab: {d}/vocab.vi
tgt_vocab: {d}/vocab.zh
world_size: 1
gpu_ranks: [0]
batch_type: tokens
batch_size: 2048
accum_count: [4]
train_steps: {train_steps}
valid_steps: {max(500, steps_per_epoch // 2)}
save_checkpoint_steps: {max(500, steps_per_epoch)}
keep_checkpoint: 2
report_every: 200
num_workers: 0
bucket_size: 1024
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
    log = (d / 'train.log').open('w', encoding='utf-8')
    subprocess.run([P, '-m', 'onmt.bin.train', '-config', str(cfg)], stdout=log, stderr=subprocess.STDOUT, check=True)
    log.close()
    ckpts = sorted(d.glob('model_step_*.pt'))
    assert ckpts, 'no checkpoint'
    model = ckpts[-1]
    pred = d / 'pred.zh'
    subprocess.run([P, '-m', 'onmt.bin.translate', '-model', str(model), '-src', str(d/'eval.vi'),
                    '-output', str(pred), '-gpu', '0', '-batch_size', '32',
                    '-beam_size', '5', '-max_length', '512', '-replace_unk'], check=True)
    print(ds, 'steps', train_steps, 'ckpt', model.name, 'pred lines', len(pred.read_text().splitlines()))
    return {'dataset': ds, 'train_steps': train_steps, 'epochs': epochs,
            'checkpoint': model.name, 'train_tokens': stats[ds]['train_tokens']}

res = [run('B')]
Path('/kaggle/working/onmt_train_summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=2))
'''

HYPS = r'''
import re, json
from pathlib import Path
import pandas as pd
WORK = Path('/kaggle/working/onmt')
DATA = Path('/kaggle/input/tiki-matrix-data')
if not (DATA/'units_A.parquet').exists():
    DATA = sorted(Path('/kaggle/input').rglob('units_A.parquet'))[0].parent

def detok(t):
    t = re.sub(r'\s+', ' ', str(t or '')).strip()
    t = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', t)
    t = re.sub(r'\s+(?=[，。！？；：、）】》])', '', t)
    t = re.sub(r'(?<=[（【《])\s+', '', t)
    return t

stats = json.loads(Path('/kaggle/working/onmt_data_stats.json').read_text())
for ds in ('B',):
    ev = pd.read_parquet(DATA / f'eval_{ds}.parquet')
    lines = (WORK/ds/'pred.zh').read_text().splitlines()
    lines += [''] * (len(ev)-len(lines))
    out = ev[['eval_id','dataset','pair_type','source','reference']].copy()
    out['hypothesis'] = [detok(x) for x in lines[:len(out)]]
    out['config'] = 'onmt_' + ds.lower(); out['seed'] = 42
    out['eval_src_over_512'] = stats[ds]['eval_src_over_512']
    p = Path(f'/kaggle/working/hyp_onmt_{ds.lower()}_seed42.parquet')
    out.to_parquet(p, index=False)
    print(ds, len(out), 'empty', int(out.hypothesis.str.strip().eq('').sum()), p)
'''

CELLS = [
    md("# OpenNMT-py matrix A/B\n\nTransformer 4 layer, 10 epochs, eval_F 1.500 dòng.\n"),
    code(SETUP), code(DATA), code(TRAIN_A), code(HYPS),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
folder = ROOT / "kaggle" / "tiki-matrix-onmt-b"
folder.mkdir(parents=True, exist_ok=True)
(folder / "onmt.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {
    "id": "ntquang0410/tiki-matrix-onmt-b",
    "title": "Tiki Matrix onmt b",
    "code_file": "onmt.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["ntquang0410/tiki-matrix-data"],
    "kernel_sources": [],
    "model_sources": [],
}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", folder)
