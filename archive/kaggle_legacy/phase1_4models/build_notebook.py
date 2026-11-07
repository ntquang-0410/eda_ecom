#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook: fine-tune + eval 4 model Qwen3 (0.6B/1.7B/4B/8B).

Cùng harness v2 (left padding), cấu hình natural (token-proportional),
mỗi model dùng bảng Tiki của chính nó + 1688.
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
    "# Fine-tune 4 model (natural weights)\n",
    "\n",
    "Mỗi model học trên 1688 + bảng Tiki của chính nó, eval trên Tiki test",
    "và 1688 test. LoRA, 5M token/model, left padding.\n",
]})

code("""import gc, json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# torchao 0.10 trong image Kaggle xung đột peft -> gỡ
import subprocess
_res = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
                      capture_output=True, text=True)
print('torchao uninstall rc', _res.returncode)
from peft import LoraConfig, get_peft_model
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))
try:
    import kagglehub
except Exception:
    kagglehub = None
""")

code("""INPUT = Path('/kaggle/input/tiki-pairs-dedup')
if not (INPUT / '1688.parquet').exists():
    import glob
    hits = glob.glob('/kaggle/input/**/1688.parquet', recursive=True)
    INPUT = Path(hits[0]).parent if hits else INPUT
sys.path.insert(0, str(INPUT))
from quantity_norm import compare, extract_quantities
print('INPUT', INPUT)

MODELS = {
    'qwen3-0.6b': {'source': 'qwen-lm/qwen-3/Transformers/0.6b/1', 'shape': (1024, 3072, 28), 'batch': 16, 'accum': 1},
    'qwen3-1.7b': {'source': 'qwen-lm/qwen-3/Transformers/1.7b/1', 'shape': (2048, 6144, 28), 'batch': 8,  'accum': 2},
    'qwen3-4b':   {'source': 'qwen-lm/qwen-3/Transformers/4b/1',   'shape': (2560, 9728, 36), 'batch': 4,  'accum': 4},
    'qwen3-8b':   {'source': 'qwen-lm/qwen-3/Transformers/8b/1',   'shape': (4096, 12288, 36),'batch': 2,  'accum': 8},
}

TOKEN_BUDGET = 5_000_000
MAX_SEQ = 1280
LR = 1e-4
WARMUP = 0.03
EVAL_TIKI = 2000
EVAL_1688 = 2000
SEED = 42
OUT = Path('/kaggle/working/phase1_4models.json')

np.random.seed(SEED); torch.manual_seed(SEED)
""")

code("""SYSTEM = ('You are a professional Vietnamese-to-Simplified Chinese e-commerce translator. '
          'Translate the user text into Simplified Chinese. Preserve numbers, units and brand names exactly. '
          'Return only the translation, no explanation.')

def user_text(row):
    key = str(row.get('key_vi') or '').strip()
    text = str(row.get('source_text') or '')
    return f'key: {key}\\ntext: {text}' if key else text

def find_model(source, shape):
    if kagglehub is not None:
        try:
            return kagglehub.model_download(source)
        except Exception as error:
            print('kagglehub failed:', error)
    import glob
    for cfg in Path('/kaggle/input').rglob('config.json'):
        try:
            payload = json.loads(cfg.read_text(encoding='utf-8'))
        except Exception:
            continue
        if (payload.get('hidden_size'), payload.get('intermediate_size'),
                payload.get('num_hidden_layers')) == shape:
            return str(cfg.parent)
    raise FileNotFoundError(source)

def make_example(tokenizer, row):
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': user_text(row)}]
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
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels[:len(full_ids)]

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

code("""def load_lora(path):
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                trust_remote_code=True, local_files_only=True,
                                                low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                     task_type='CAUSAL_LM', target_modules='all-linear')
    model = get_peft_model(base, cfg)
    model.to('cuda')
    return model

def build_sampler(train):
    p = train['tokens_est'].astype(float).values
    p = p / p.sum()
    chosen, total = [], 0.0
    while total < TOKEN_BUDGET:
        draw = np.random.choice(len(train), size=4096, p=p, replace=True)
        for idx in draw:
            chosen.append(int(idx)); total += float(train.iloc[idx]['tokens_est'])
            if total >= TOKEN_BUDGET: break
    return chosen, total

def train_one(model, tokenizer, train, batch_size, accum):
    chosen, total_tokens = build_sampler(train)
    records = train.to_dict('records')
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)
    total_steps = max(1, int(total_tokens / (batch_size * 700)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, int(total_steps * WARMUP))) *
        max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))))
    started = time.time()
    step, micro = 0, 0
    optimizer.zero_grad(set_to_none=True)
    for idx in chosen:
        example = make_example(tokenizer, records[idx])
        if example is None:
            continue
        batch = collate([example], tokenizer.pad_token_id)
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch).loss / accum
        loss.backward()
        micro += 1
        if micro >= accum:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            micro = 0; step += 1
            if step % 100 == 0:
                print('  step', step, '| loss', round(float(loss.item() * accum), 4),
                      '| min', round((time.time() - started) / 60, 1), flush=True)
    return {'steps': step, 'tokens': round(total_tokens), 'minutes': round((time.time() - started) / 60, 1)}
""")

code("""CJK = re.compile(r'[\\u4e00-\\u9fff]')
VI = re.compile(r'[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]', re.I)

def chrf(hyp, ref, n=4, beta=2.0):
    def grams(text, k):
        text = re.sub(r'\\s+', '', str(text or ''))
        return {text[i:i+k] for i in range(max(0, len(text)-k+1))}
    if not hyp or not ref: return 0.0
    scores = []
    for k in range(1, n+1):
        h, r = grams(hyp, k), grams(ref, k)
        if not h or not r: continue
        p = len(h & r) / len(h); rec = len(h & r) / len(r)
        scores.append((1+beta**2) * p * rec / (beta**2 * p + rec + 1e-9))
    return float(np.mean(scores)) if scores else 0.0

def generate(model, tokenizer, rows, max_new=256):
    model.eval(); outs = []
    for start in range(0, len(rows), 8):
        batch = rows.iloc[start:start+8]
        prompts = []
        for _, row in batch.iterrows():
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': user_text(row)}]
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

def evaluate(model, tokenizer, tiki_test, d1688_test):
    outs = generate(model, tokenizer, tiki_test)
    rows = []
    for (_, row), out in zip(tiki_test.iterrows(), outs):
        cjk = bool(CJK.search(out)); vi = bool(VI.search(out)); src_vi = bool(VI.search(str(row['source_text'])))
        qty = compare(row['source_text'], out, 'vi', 'zh')
        rows.append(bool(out) and (cjk or not src_vi) and (not vi or not src_vi) and qty['match'])
    outs2 = generate(model, tokenizer, d1688_test)
    cf = [chrf(o, r) for o, r in zip(outs2, d1688_test['target_text'])]
    q1688 = [compare(s, o, 'vi', 'zh')['match'] for s, o in zip(d1688_test['source_text'], outs2)]
    return {'tiki_pass_rate': round(float(np.mean(rows)), 4),
            'g1688_chrf': round(float(np.mean(cf)), 4),
            'g1688_qty_match': round(float(np.mean(q1688)), 4)}
""")

code("""d1688 = pd.read_parquet(INPUT / '1688.parquet')
d1688_test = d1688[d1688['split'] == 'test']
d1688_test = d1688_test.groupby('pair_type', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), EVAL_1688 // 3), random_state=SEED)).reset_index(drop=True)
print('eval 1688', len(d1688_test))

results = {}
for model_key, meta in MODELS.items():
    print('=' * 70)
    print('MODEL', model_key)
    tiki = pd.read_parquet(INPUT / f'tiki_{model_key}.parquet')
    df = pd.concat([d1688, tiki], ignore_index=True)
    train = df[df['split'] == 'train'].reset_index(drop=True)
    tiki_test = tiki[tiki['split'] == 'test'].sample(
        n=min(EVAL_TIKI, int((tiki['split'] == 'test').sum())), random_state=SEED).reset_index(drop=True)
    print('train', len(train), '| tiki test', len(tiki_test))

    path = find_model(meta['source'], meta['shape'])
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = load_lora(path)
    info = train_one(model, tokenizer, train, meta['batch'], meta['accum'])
    metrics = evaluate(model, tokenizer, tiki_test, d1688_test)
    metrics.update(info)
    results[model_key] = metrics
    print(model_key, json.dumps(metrics, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

print(json.dumps(results, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "phase1_4models.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
