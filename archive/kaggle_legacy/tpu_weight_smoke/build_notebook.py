#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test: Qwen3-4B LoRA vài bước trên TPU v5e-8."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# Smoke test TPU v5e-8 — Qwen3-4B LoRA\n",
]})

code("""import gc, json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
print('torch', torch.__version__)
print('torch_xla', torch_xla.__version__)
print('world_size', xr.world_size(), '| ordinal', xr.global_ordinal())
DEVICE = xm.xla_device()
print('DEVICE', DEVICE)
""")

code("""INPUT = Path('/kaggle/input/tiki-pairs-dedup')
if not (INPUT / '1688.parquet').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/1688.parquet', recursive=True)
    INPUT = Path(hits[0]).parent
EXP = Path('/kaggle/input/tiki-1688-exp')
if not (EXP / 'test_small.parquet').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/test_small.parquet', recursive=True)
    EXP = Path(hits[0]).parent
print('INPUT', INPUT, '| EXP', EXP)

MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)
MAX_SEQ = 768
BATCH = 1

def find_model():
    try:
        import kagglehub
        return kagglehub.model_download(MODEL_SOURCE)
    except Exception as error:
        print('kagglehub failed:', error)
    for cfg in Path('/kaggle/input').rglob('config.json'):
        try:
            payload = json.loads(cfg.read_text(encoding='utf-8'))
        except Exception:
            continue
        if (payload.get('hidden_size'), payload.get('intermediate_size'),
                payload.get('num_hidden_layers')) == SHAPE:
            return str(cfg.parent)
    from huggingface_hub import snapshot_download
    return snapshot_download('Qwen/Qwen3-4B')
MODEL_PATH = find_model()
print('model', MODEL_PATH)
""")

code("""from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

SYSTEM = ('You are a professional Vietnamese-to-Simplified Chinese e-commerce translator. '
          'Translate the user text into Simplified Chinese. Preserve numbers and units. '
          'Return only the translation.')

def encode(row):
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': str(row['source_text'])}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + str(row['target_text']) + tokenizer.eos_token,
                     add_special_tokens=False).input_ids[:MAX_SEQ]
    if len(full) <= len(prompt_ids) + 1:
        return None
    return full, [-100] * len(prompt_ids) + full[len(prompt_ids):]

class DS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        item = encode(self.rows[index])
        while item is None:
            index = (index + 1) % len(self.rows)
            item = encode(self.rows[index])
        return item

def collate(batch):
    length = max(len(x[0]) for x in batch)
    ids, labs, mask = [], [], []
    for seq, lab in batch:
        pad = length - len(seq)
        ids.append(seq + [tokenizer.pad_token_id] * pad)
        labs.append(lab + [-100] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labs),
            'attention_mask': torch.tensor(mask)}
""")

code("""train = pd.read_parquet(INPUT / '1688.parquet')
train = train[train['split'] == 'train'].head(2000).reset_index(drop=True)
print('sample rows', len(train))

base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                            trust_remote_code=True, low_cpu_mem_usage=True)
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                                        task_type='CAUSAL_LM', target_modules='all-linear'))
model.to(DEVICE)
model.print_trainable_parameters()
print('memory after load', xm.get_memory_info(DEVICE))
""")

code("""dataset = DS(train.to_dict('records'))
sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=xr.world_size(),
                                              rank=xr.global_ordinal(), shuffle=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, sampler=sampler, collate_fn=collate)
loader = pl.MpDeviceLoader(loader, DEVICE)

params = [q for q in model.parameters() if q.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4)
model.train()
started = time.time()
step = 0
for batch in loader:
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    loss = model(**batch).loss
    loss.backward()
    xm.optimizer_step(optimizer)
    xm.mark_step()
    step += 1
    if step % 5 == 0:
        print('step', step, '| loss', round(float(loss.item()), 4),
              '| sec/step', round((time.time()-started)/step, 2),
              '| mem', xm.get_memory_info(DEVICE), flush=True)
    if step >= 20:
        break
print('done', step, 'steps in', round((time.time()-started)/60, 2), 'min')
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "smoke_tpu.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
