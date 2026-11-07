#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Qwen3-4B trên 1688+Tiki, sinh output cho 2 tập, lưu parquet.

Output:
  /kaggle/working/hyp_1688.parquet  (id, source, hypothesis, reference)
  /kaggle/working/hyp_tiki.parquet  (id, source, hypothesis)
  /kaggle/working/adapter_4b/
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# Qwen3-4B — train 1688+Tiki, sinh output 2 tập\n",
]})

code("""import gc, json, math, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
""")

code("""INPUT = Path('/kaggle/input/tiki-pairs-dedup')
if not (INPUT / '1688.parquet').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/1688.parquet', recursive=True)
    INPUT = Path(hits[0]).parent
MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)
TOKEN_BUDGET = 5_000_000
MAX_SEQ = 1280
BATCH, ACCUM = 4, 4
LR = 1e-4
EVAL_TIKI = 2000
SEED = 42

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
    raise FileNotFoundError('4b')
MODEL_PATH = find_model()
print('model', MODEL_PATH)
""")

code("""SYSTEM = ('You are a professional Vietnamese-to-Simplified Chinese e-commerce translator. '
          'Translate the user text into Simplified Chinese. Preserve numbers, units and brand names exactly. '
          'Return only the translation, no explanation.')

def user_text(row):
    key = str(row.get('key_vi') or '').strip()
    text = str(row.get('source_text') or '')
    return f'key: {key}\\ntext: {text}' if key else text

def make_example(tokenizer, row):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_text(row)}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(prompt_ids) >= MAX_SEQ - 16:
        return None
    full_ids = tokenizer(prompt + str(row['target_text'] or '') + tokenizer.eos_token,
                         add_special_tokens=False).input_ids[:MAX_SEQ]
    if len(full_ids) <= len(prompt_ids) + 1:
        return None
    return full_ids, [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

def collate(batch, pad_id):
    length = max(len(x[0]) for x in batch)
    ids, labs, mask = [], [], []
    for seq, lab in batch:
        pad = length - len(seq)
        ids.append(seq + [pad_id] * pad); labs.append(lab + [-100] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labs),
            'attention_mask': torch.tensor(mask)}
""")

code("""d1688 = pd.read_parquet(INPUT / '1688.parquet')
tiki = pd.read_parquet(INPUT / 'tiki_qwen3-4b.parquet')
df = pd.concat([d1688, tiki], ignore_index=True)
train = df[df['split'] == 'train'].reset_index(drop=True)
train['tok'] = train['tokens_est'].astype(float)
print('train', len(train))

p = train['tok'].values / train['tok'].values.sum()
chosen, total = [], 0.0
while total < TOKEN_BUDGET:
    for idx in np.random.choice(len(train), size=4096, p=p, replace=True):
        chosen.append(int(idx)); total += float(train['tok'].iloc[idx])
        if total >= TOKEN_BUDGET: break
print('sampled', len(chosen), round(total/1e6, 2), 'M tokens')
""")

code("""tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                            trust_remote_code=True, local_files_only=True,
                                            low_cpu_mem_usage=True)
base.gradient_checkpointing_enable(); base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                                        task_type='CAUSAL_LM', target_modules='all-linear')).to('cuda')
model.print_trainable_parameters()
""")

code("""records = train.to_dict('records')
params = [q for q in model.parameters() if q.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR)
steps_total = max(1, int(total / (BATCH * 700)))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min(1.0, (s+1)/max(1, int(steps_total*0.03))) *
    max(0.0, 0.5*(1+math.cos(math.pi*min(1.0, s/steps_total)))))
model.train(); step = micro = 0; started = time.time()
optimizer.zero_grad(set_to_none=True)
for idx in chosen:
    example = make_example(tokenizer, records[idx])
    if example is None: continue
    batch = collate([example], tokenizer.pad_token_id)
    batch = {k: v.to('cuda') for k, v in batch.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**batch).loss / ACCUM
    loss.backward(); micro += 1
    if micro >= ACCUM:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        micro = 0; step += 1
        if step % 100 == 0:
            print('step', step, '| loss', round(float(loss.item()*ACCUM), 4),
                  '| min', round((time.time()-started)/60, 1), flush=True)
print('train done', step, round((time.time()-started)/60, 1), 'min')
model.save_pretrained('/kaggle/working/adapter_4b')
tokenizer.save_pretrained('/kaggle/working/adapter_4b')
""")

code("""def generate(rows, max_new=256):
    model.eval(); outs = []
    for start in range(0, len(rows), 8):
        batch = rows.iloc[start:start+8]
        prompts = []
        for _, row in batch.iterrows():
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_text(row)}]
            try:
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ).to('cuda')
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        outs.extend([t.strip() for t in tokenizer.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)])
        if (start // 8) % 20 == 0:
            print('  gen', start+len(batch), '/', len(rows), flush=True)
    return outs

# 1688 test_small (1000) từ dataset 1688-exp
exp = Path('/kaggle/input/tiki-1688-exp')
if not (exp / 'test_small.parquet').exists():
    import glob
    exp = Path(glob.glob('/kaggle/input/**/test_small.parquet', recursive=True)[0]).parent
small = pd.read_parquet(exp / 'test_small.parquet').reset_index(drop=True)
small['id'] = small['pair_id'].astype(str)
small['source'] = small['source_text']; small['reference'] = small['target_text']
small['hypothesis'] = generate(small)
small[['id','source','hypothesis','reference']].to_parquet('/kaggle/working/hyp_1688.parquet', index=False)
print('1688 done', len(small))

tiki_test = tiki[tiki['split'] == 'test'].sample(n=min(EVAL_TIKI, int((tiki['split']=='test').sum())), random_state=SEED).reset_index(drop=True)
tiki_test['id'] = tiki_test['pair_id'].astype(str)
tiki_test['source'] = tiki_test['source_text']
tiki_test['hypothesis'] = generate(tiki_test)
tiki_test[['id','source','hypothesis']].to_parquet('/kaggle/working/hyp_tiki.parquet', index=False)
print('tiki done', len(tiki_test))
""")

code("""print(json.dumps({'1688': len(small), 'tiki': len(tiki_test)}, ensure_ascii=False))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "train_generate_4b.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
