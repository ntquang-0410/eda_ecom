#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLLB-200-distilled-600M fine-tune trên đúng quota 4M token A/B (control pretrained NMT)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


CHECK = r'''
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
print('transformers', transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print('gpu', i, p.name, round(p.total_memory/1e9,1), 'GB')
'''

TRAIN = r'''
import json, math, os, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DATA = Path('/kaggle/input/tiki-matrix-data')
if not (DATA/'units_A.parquet').exists():
    DATA = sorted(Path('/kaggle/input').rglob('units_A.parquet'))[0].parent
CTRL = Path('/kaggle/input/tiki-matrix-controls')
if not (CTRL/'sample_A_natural_seed42.parquet').exists():
    CTRL = sorted(Path('/kaggle/input').rglob('sample_A_natural_seed42.parquet'))[0].parent
MODEL = None
for cfg in Path('/kaggle/input').rglob('config.json'):
    try: d = json.loads(cfg.read_text(encoding='utf-8'))
    except Exception: continue
    if d.get('model_type') == 'm2m_100' or 'M2M100' in str(d.get('architectures')):
        MODEL = str(cfg.parent); break
assert MODEL, 'không thấy NLLB model source'
print('DATA', DATA, 'CTRL', CTRL, 'MODEL', MODEL)

tok = AutoTokenizer.from_pretrained(MODEL, src_lang='vie_Latn', tgt_lang='zho_Hans',
                                    local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tgt_lang_id = tok.convert_tokens_to_ids('zho_Hans')
print('tokenizer', len(tok), 'tgt_lang_id', tgt_lang_id, 'pad', tok.pad_token_id)

def load_model():
    m = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True,
                                              low_cpu_mem_usage=True)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                                     task_type='SEQ_2_SEQ_LM', target_modules='all-linear'))
    m.to('cuda').train()
    m.config.use_cache = False
    return m

def encode(src, tgt, max_seq=768):
    a = tok(str(src or '').strip(), add_special_tokens=True, truncation=True, max_length=max_seq)
    b = tok(str(tgt or '').strip(), add_special_tokens=True, truncation=True, max_length=max_seq)
    ids = a['input_ids']; labs = b['input_ids']
    if len(ids) < 2 or len(labs) < 2:
        return None
    return ids, labs

def collate(items):
    max_in = max(len(x[0]) for x in items); max_out = max(len(x[1]) for x in items)
    inp, att, lab = [], [], []
    for a, b in items:
        inp.append(a + [tok.pad_token_id]*(max_in-len(a)))
        att.append([1]*len(a) + [0]*(max_in-len(a)))
        lab.append(b + [-100]*(max_out-len(b)))
    return {'input_ids': torch.tensor(inp), 'attention_mask': torch.tensor(att),
            'labels': torch.tensor(lab)}

def train_one(ds, cfg, sample_name, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(DATA/f'units_{ds}.parquet').reset_index(drop=True)
    ev = pd.read_parquet(DATA/f'eval_{ds}.parquet')
    s = pd.read_parquet(CTRL/sample_name)
    model = load_model()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0,(step+1)/20)*max(0.0,0.5*(1+math.cos(math.pi*min(1.0,step/500)))))
    start = time.time(); steps = sorted(s.step.unique())
    for si, step in enumerate(steps):
        rows = s[s.step == step]
        idxs = rows.row_index.astype(int).tolist()
        items = []
        for i in idxs:
            x = encode(u.source_text.iloc[i], u.target_text.iloc[i])
            if x is not None: items.append(x)
        if not items: continue
        opt.zero_grad(set_to_none=True)
        step_tokens = sum(len(x[1]) for x in items)
        for j in range(0, len(items), 2):
            batch = collate(items[j:j+2])
            batch = {k:v.to('cuda') for k,v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**batch).loss * (int((batch['labels']!=-100).sum()) / max(1.0, step_tokens))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if si % 25 == 0:
            print(f'{ds} step {si}/{len(steps)} min={(time.time()-start)/60:.1f}', flush=True)
    model.save_pretrained(out_dir/'adapter')
    tok.save_pretrained(out_dir/'adapter')

    # inference đúng max_new theo eval_F
    model.eval(); model.config.use_cache = True
    recs = []
    for max_new, part in ev.groupby('max_new'):
        part = part.reset_index(drop=True)
        bs = 1 if int(max_new) > 300 else 8
        for k in range(0, len(part), bs):
            chunk = part.iloc[k:k+bs]
            enc = tok([str(x) for x in chunk.source], return_tensors='pt', padding=True,
                      truncation=True, max_length=768, add_special_tokens=True).to('cuda')
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=tgt_lang_id,
                                     max_new_tokens=int(max_new), do_sample=False,
                                     num_beams=1, eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for eid, text in zip(chunk.eval_id, texts):
                recs.append({'eval_id': eid, 'hypothesis': text.strip()})
    hyp = ev[['eval_id','dataset','pair_type','source','reference']].merge(pd.DataFrame(recs), on='eval_id', how='left')
    hyp['hypothesis'] = hyp.hypothesis.fillna(''); hyp['config'] = cfg; hyp['seed'] = 42
    hyp.to_parquet(out_dir/f'hyp_{cfg}_seed42.parquet', index=False)
    info = {'config': cfg, 'dataset': ds, 'steps': len(steps), 'train_rows': int(len(s)),
            'unique_rows': int(s.row_index.nunique()), 'minutes': round((time.time()-start)/60,1),
            'hyp_rows': int(len(hyp)), 'empty': int(hyp.hypothesis.str.strip().eq('').sum())}
    (out_dir/f'run_{cfg}_seed42.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    del model, opt
    torch.cuda.empty_cache()

train_one('A', 'nllb_a', 'sample_A_natural_seed42.parquet', '/kaggle/working/nllb_a')
train_one('B', 'nllb_b', 'sample_B_balanced_seed42.parquet', '/kaggle/working/nllb_b')
'''

CELLS = [md("# NLLB-200-distilled-600M — control pretrained NMT, quota 4M token A/B\n"),
         code(CHECK), code(TRAIN)]
nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
folder = ROOT / "kaggle" / "matrix_nllb"; folder.mkdir(parents=True, exist_ok=True)
(folder / "nllb.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {"id": "nguynthanhquang/tiki-nllb-control", "title": "Tiki Nllb Control",
        "code_file": "nllb.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/tiki-matrix-data", "nguynthanhquang/tiki-matrix-controls",
                            "metric/gemma-4-developer-agent-wheelhouse"],
        "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [],
        "model_sources": ["amritha18/nllb-200-distilled-600m/Transformers/600m/1"],
        "machine_shape": "NvidiaL4x4"}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", folder)
