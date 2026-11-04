#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-4B fine-tune trên 1688-only, eval BLEU/chrF trên test_small + full test."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# 1688-only — Qwen3-4B\n",
    "\n",
    "Fine-tune LoRA trên 1688 train (title + attribute), eval BLEU/chrF.\n",
]})

code("""import gc, json, math, re, subprocess, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import subprocess
_r = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True, text=True)
print('torchao rc', _r.returncode)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
""")

code("""INPUT = Path('/kaggle/input/tiki-1688-exp')
if not (INPUT / 'train.parquet').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/train.parquet', recursive=True)
    INPUT = Path(hits[0]).parent
print('INPUT', INPUT)

MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)
TOKEN_BUDGET = 5_000_000
MAX_SEQ = 1280
BATCH, ACCUM = 4, 4
LR = 1e-4

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
        ids.append(seq + [pad_id] * pad)
        labs.append(lab + [-100] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labs),
            'attention_mask': torch.tensor(mask)}
""")

code("""def corpus_bleu(hyps, refs, n=4):
    matches, totals = [0]*n, [0]*n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht = list(re.sub(r'\\s+', '', str(h)))
        rt = list(re.sub(r'\\s+', '', str(r)))
        hyp_len += len(ht); ref_len += len(rt)
        for k in range(1, n+1):
            hc = Counter(tuple(ht[i:i+k]) for i in range(max(0, len(ht)-k+1)))
            rc = Counter(tuple(rt[i:i+k]) for i in range(max(0, len(rt)-k+1)))
            matches[k-1] += sum(min(v, rc[g]) for g, v in hc.items())
            totals[k-1] += sum(hc.values())
    prec = [(matches[i]/totals[i] if totals[i] else 0.0) for i in range(n)]
    if min(prec) == 0: return 0.0
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len/max(1, hyp_len))
    return bp * math.exp(sum(math.log(p) for p in prec)/n)

def chrf(hyp, ref, n=4, beta=2.0):
    def grams(t, k):
        t = re.sub(r'\\s+', '', str(t))
        return {t[i:i+k] for i in range(max(0, len(t)-k+1))}
    if not hyp or not ref: return 0.0
    vals = []
    for k in range(1, n+1):
        h, r = grams(hyp, k), grams(ref, k)
        if not h or not r: continue
        p = len(h & r)/len(h); rec = len(h & r)/len(r)
        vals.append((1+beta**2)*p*rec/(beta**2*p+rec+1e-9))
    return float(np.mean(vals)) if vals else 0.0

def generate(model, tokenizer, rows, max_new=256):
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
    return outs
""")

code("""train = pd.read_parquet(INPUT / 'train.parquet').reset_index(drop=True)
test_small = pd.read_parquet(INPUT / 'test_small.parquet').reset_index(drop=True)
test_full = pd.read_parquet(INPUT / 'test.parquet').reset_index(drop=True)
print('train', len(train), '| test_small', len(test_small), '| test', len(test_full))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
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

code("""# sampler thật: tính token theo ký tự
def token_est(row):
    return len(re.sub(r'\\s+', '', str(row['source_text']))) / 3.6 + len(re.sub(r'\\s+', '', str(row['target_text']))) / 1.6

train['tok'] = train.apply(token_est, axis=1)
p = train['tok'].values / train['tok'].values.sum()
chosen, total = [], 0.0
while total < TOKEN_BUDGET:
    for idx in np.random.choice(len(train), size=4096, p=p, replace=True):
        chosen.append(int(idx)); total += float(train['tok'].iloc[idx])
        if total >= TOKEN_BUDGET: break
print('sampled', len(chosen), '|', round(total/1e6, 2), 'M tokens')
""")

code("""model.train()
params = [q for q in model.parameters() if q.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR)
steps_total = max(1, int(total / (BATCH * 700)))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min(1.0, (s+1)/max(1, int(steps_total*0.03))) *
    max(0.0, 0.5*(1+math.cos(math.pi*min(1.0, s/steps_total)))))
records = train.to_dict('records')
step = micro = 0
started = time.time()
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
print('train done', step, 'steps |', round((time.time()-started)/60, 1), 'min')
model.save_pretrained('/kaggle/working/adapter_4b_1688')
tokenizer.save_pretrained('/kaggle/working/adapter_4b_1688')
""")

code("""def evaluate(rows):
    outs = generate(model, tokenizer, rows)
    refs = rows['target_text'].tolist()
    return {
        'n': len(rows),
        'bleu': round(corpus_bleu(outs, refs), 4),
        'chrf': round(float(np.mean([chrf(o, r) for o, r in zip(outs, refs)])), 4),
        'empty': round(float(np.mean([not o for o in outs])), 4),
    }

small = evaluate(test_small)
print('test_small', small)
full = evaluate(test_full)
print('test_full', full)
result = {'system': 'qwen3-4b', 'test_small': small, 'test_full': full,
          'train_tokens': round(total), 'steps': step,
          'minutes': round((time.time()-started)/60, 1)}
Path('/kaggle/working/result_4b_1688.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "exp_1688_4b.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
