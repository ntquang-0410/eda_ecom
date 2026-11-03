#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook Pha 1: thực nghiệm weight title/attr/desc/tiki trên Qwen3-4B.

- LoRA, token budget cố định cho mọi cấu hình.
- Sampler theo token mass để đạt share mục tiêu.
- Eval trên Tiki test (frozen) + 1688 test.
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
    "# Pha 1 — weight experiments (Qwen3-4B)\n",
    "\n",
    "5 cấu hình weight, cùng token budget, cùng eval set.\n",
    "LoRA + sampler theo token mass.\n",
]})

code("""import json, math, os, random, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# torchao trong image Kaggle cũ (0.10) không tương thích peft -> gỡ hẳn
import subprocess, sys as _sys
_res = subprocess.run([_sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
                      capture_output=True, text=True)
print('torchao uninstall rc', _res.returncode, (_res.stdout or '')[-160:])
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

MODEL_KEY = 'qwen3-4b'
MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)

TOKEN_BUDGET = 5_000_000
MAX_SEQ = 1280
BATCH = 4
GRAD_ACCUM = 4
LR = 1e-4
WARMUP = 0.03
EVAL_TIKI = 2000
EVAL_1688 = 2000
SEED = 42
OUT = Path('/kaggle/working/phase1_results.json')

CONFIGS = {
    'natural': None,
    'balanced': {'title_1688': 0.15, 'attr_1688': 0.35, 'desc_1688': 0.25, 'tiki': 0.25},
    'no_desc': {'title_1688': 0.25, 'attr_1688': 0.45, 'desc_1688': 0.0, 'tiki': 0.30},
    'tiki_heavy': {'title_1688': 0.10, 'attr_1688': 0.25, 'desc_1688': 0.20, 'tiki': 0.45},
    'title_heavy': {'title_1688': 0.30, 'attr_1688': 0.30, 'desc_1688': 0.15, 'tiki': 0.25},
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
""")

code("""SYSTEM = ('You are a professional Vietnamese-to-Simplified Chinese e-commerce translator. '
          'Translate the user text into Simplified Chinese. Preserve numbers, units and brand names exactly. '
          'Return only the translation, no explanation.')

def user_text(row):
    key = str(row.get('key_vi') or '').strip()
    text = str(row.get('source_text') or '')
    if key:
        return f'key: {key}\\ntext: {text}'
    return text

def find_model():
    if kagglehub is not None:
        try:
            return kagglehub.model_download(MODEL_SOURCE)
        except Exception as error:
            print('kagglehub failed:', error)
    import glob
    for cfg in Path('/kaggle/input').rglob('config.json'):
        try:
            payload = json.loads(cfg.read_text(encoding='utf-8'))
        except Exception:
            continue
        if (payload.get('hidden_size'), payload.get('intermediate_size'),
                payload.get('num_hidden_layers')) == SHAPE:
            return str(cfg.parent)
    raise FileNotFoundError('no 4b weights')

MODEL_PATH = find_model()
print('model', MODEL_PATH)
""")

code("""# nạp dữ liệu
d1688 = pd.read_parquet(INPUT / '1688.parquet')
tiki = pd.read_parquet(INPUT / 'tiki_qwen3-4b.parquet')
df = pd.concat([d1688, tiki], ignore_index=True)
df['group'] = np.where(df['domain'] == 'tiki', 'tiki',
                       df['pair_type'].map({'title': 'title_1688', 'attribute': 'attr_1688', 'description': 'desc_1688'}))
df['tokens'] = df['tokens_est'].astype(float)

train = df[df['split'] == 'train'].reset_index(drop=True)
print('train rows', len(train))
print(train.groupby('group').agg(rows=('pair_id','size'), tokens=('tokens','sum')).to_string())

tiki_test = tiki[tiki['split'] == 'test'].sample(n=min(EVAL_TIKI, (tiki['split']=='test').sum()), random_state=SEED).reset_index(drop=True)
d1688_test = d1688[d1688['split'] == 'test']
d1688_test = d1688_test.groupby('pair_type', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), EVAL_1688 // 3), random_state=SEED)).reset_index(drop=True)
print('eval tiki', len(tiki_test), '| eval 1688', len(d1688_test))
""")

code("""def build_sampler(config):
    groups = train['group'].values
    tokens = train['tokens'].values
    mass = train.groupby('group')['tokens'].sum()
    if config is None:
        # natural: xác suất theo token mass
        weights = {g: 1.0 for g in mass.index}
    else:
        weights = {g: config[g] / max(1e-9, mass[g]) for g in mass.index}
    p = np.array([weights.get(g, 1.0) * t for g, t in zip(groups, tokens)], dtype=np.float64)
    p /= p.sum()
    # lấy mẫu đến khi đủ token budget
    chosen, total = [], 0.0
    while total < TOKEN_BUDGET:
        draw = np.random.choice(len(train), size=4096, p=p, replace=True)
        for idx in draw:
            chosen.append(int(idx)); total += tokens[idx]
            if total >= TOKEN_BUDGET:
                break
    return chosen, total

def make_example(tokenizer, row):
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': user_text(row)}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target = str(row['target_text'] or '')
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + target + tokenizer.eos_token, add_special_tokens=False).input_ids
    full_ids = full_ids[:MAX_SEQ]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    labels = labels[:len(full_ids)]
    return full_ids, labels

def collate(batch, pad_id):
    length = max(len(x[0]) for x in batch)
    input_ids, labels, attention = [], [], []
    for ids, lab in batch:
        pad = length - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention)}
""")

code("""def load_lora():
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                                trust_remote_code=True, local_files_only=True,
                                                low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                     task_type='CAUSAL_LM', target_modules='all-linear')
    model = get_peft_model(base, cfg)
    model.to('cuda')
    model.print_trainable_parameters()
    return model

def train_one(config, tokenizer):
    chosen, total_tokens = build_sampler(config)
    print('sampled', len(chosen), 'examples |', round(total_tokens/1e6, 2), 'M tokens')
    model = load_lora()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps_per_epoch = max(1, len(chosen) // (BATCH * GRAD_ACCUM))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=steps_per_epoch, pct_start=WARMUP)
    started = time.time()
    step, accum = 0, 0
    optimizer.zero_grad(set_to_none=True)
    order = list(range(len(chosen)))
    for i in order:
        row = train.iloc[chosen[i]]
        example = make_example(tokenizer, row)
        if len(example[0]) < 8:
            continue
        batch = collate([example], tokenizer.pad_token_id)
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch).loss / GRAD_ACCUM
        loss.backward()
        accum += 1
        if accum >= GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            accum = 0; step += 1
            if step % 50 == 0:
                print('  step', step, '/', steps_per_epoch, '| loss', round(float(loss.item()*GRAD_ACCUM), 4),
                      '| min', round((time.time()-started)/60, 1), flush=True)
            if step >= steps_per_epoch:
                break
    return model, {'steps': step, 'tokens': total_tokens, 'minutes': round((time.time()-started)/60, 1)}
""")

code("""def generate(model, tokenizer, rows, max_new=256):
    model.eval()
    outputs = []
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
        texts = tokenizer.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        outputs.extend([t.strip() for t in texts])
    return outputs

CJK = re.compile(r'[\\u4e00-\\u9fff]')
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

def evaluate(model, tokenizer):
    # Tiki test: không có gold -> pass-rate
    outs = generate(model, tokenizer, tiki_test)
    tiki_rows = []
    for (_, row), out in zip(tiki_test.iterrows(), outs):
        cjk = bool(CJK.search(out)); vi = bool(VI.search(out)); src_vi = bool(VI.search(str(row['source_text'])))
        qty = compare(row['source_text'], out, 'vi', 'zh')
        ok = bool(out) and (cjk or not src_vi) and (not vi or not src_vi) and qty['match']
        tiki_rows.append({'cjk': cjk, 'vi': vi, 'qty': qty['match'], 'pass': ok,
                          'len_ratio': len(out) / max(1, len(str(row['source_text'])))})
    tdf = pd.DataFrame(tiki_rows)

    outs = generate(model, tokenizer, d1688_test, max_new=256)
    refs = d1688_test['target_text'].tolist()
    cf = [chrf(o, r) for o, r in zip(outs, refs)]
    qty1688 = [compare(s, o, 'vi', 'zh')['match'] for s, o in zip(d1688_test['source_text'], outs)]
    return {
        'tiki_pass_rate': round(float(tdf['pass'].mean()), 4),
        'tiki_qty_match': round(float(tdf['qty'].mean()), 4),
        'tiki_untranslated': round(float((tdf['vi'] & ~tdf['cjk']).mean()), 4),
        'tiki_len_ratio_p50': round(float(tdf['len_ratio'].median()), 3),
        'g1688_chrf': round(float(np.mean(cf)), 4),
        'g1688_qty_match': round(float(np.mean(qty1688)), 4),
        'n_tiki': len(tdf), 'n_1688': len(cf),
    }
""")

code("""results = {}
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

for name, config in CONFIGS.items():
    print('=' * 60)
    print('RUN', name, config)
    model, train_info = train_one(config, tokenizer)
    metrics = evaluate(model, tokenizer)
    metrics.update(train_info)
    results[name] = metrics
    print(name, json.dumps(metrics, ensure_ascii=False))
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

print(json.dumps(results, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "phase1_weights.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
