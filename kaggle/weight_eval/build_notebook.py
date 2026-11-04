#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook đánh giá 5 cấu hình weight bằng COMET.

Mỗi cấu hình: train Qwen3-4B 5M token -> sinh output 1688 test_small (1.000)
và Tiki test (1.000) -> COMET (da cho 1688, QE cho Tiki).

Output:
  /kaggle/working/hyp_<config>_1688.parquet
  /kaggle/working/hyp_<config>_tiki.parquet
  /kaggle/working/weight_eval_comet.json
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
    "# Đánh giá weight bằng COMET — Qwen3-4B\n",
    "\n",
    "5 cấu hình: natural, balanced, no_desc, tiki_heavy, title_heavy.\n",
    "1688 dùng COMET reference-based, Tiki dùng COMET-QE.\n",
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

MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)
TOKEN_BUDGET = 5_000_000
MAX_SEQ = 1280
BATCH, ACCUM = 4, 4
LR = 1e-4
SEED = 42
N_EVAL = 1000

CONFIGS = {
    'natural': None,
    'balanced': {'title_1688': 0.15, 'attr_1688': 0.35, 'desc_1688': 0.25, 'tiki': 0.25},
    'no_desc': {'title_1688': 0.25, 'attr_1688': 0.45, 'desc_1688': 0.0, 'tiki': 0.30},
    'tiki_heavy': {'title_1688': 0.10, 'attr_1688': 0.25, 'desc_1688': 0.20, 'tiki': 0.45},
    'title_heavy': {'title_1688': 0.30, 'attr_1688': 0.30, 'desc_1688': 0.15, 'tiki': 0.25},
}

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
train['group'] = np.where(train['domain'] == 'tiki', 'tiki',
                          train['pair_type'].map({'title': 'title_1688', 'attribute': 'attr_1688', 'description': 'desc_1688'}))
mass = train.groupby('group')['tok'].sum()

small = pd.read_parquet(EXP / 'test_small.parquet').reset_index(drop=True)
small['id'] = small['pair_id'].astype(str)
small['source'] = small['source_text']; small['reference'] = small['target_text']
tiki_test = tiki[tiki['split'] == 'test'].sample(n=min(N_EVAL, int((tiki['split']=='test').sum())), random_state=SEED).reset_index(drop=True)
tiki_test['id'] = tiki_test['pair_id'].astype(str)
tiki_test['source'] = tiki_test['source_text']
print('train', len(train), '| 1688 eval', len(small), '| tiki eval', len(tiki_test))
""")

code("""def build_sampler(config):
    if config is None:
        p = train['tok'].values / train['tok'].values.sum()
    else:
        weights = {g: config[g] / max(1e-9, mass[g]) for g in mass.index}
        p = np.array([weights.get(g, 1.0) * t for g, t in zip(train['group'], train['tok'])])
        p = p / p.sum()
    chosen, total = [], 0.0
    while total < TOKEN_BUDGET:
        for idx in np.random.choice(len(train), size=4096, p=p, replace=True):
            chosen.append(int(idx)); total += float(train['tok'].iloc[idx])
            if total >= TOKEN_BUDGET: break
    return chosen, total

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

code("""tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
records_train = train.to_dict('records')
results = {}

for name, config in CONFIGS.items():
    print('=' * 60); print('RUN', name, config, flush=True)
    chosen, total = build_sampler(config)
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                                trust_remote_code=True, local_files_only=True,
                                                low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                                            task_type='CAUSAL_LM', target_modules='all-linear')).to('cuda')
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)
    steps_total = max(1, int(total / (BATCH * 700)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s+1)/max(1, int(steps_total*0.03))) *
        max(0.0, 0.5*(1+math.cos(math.pi*min(1.0, s/steps_total)))))
    model.train(); step = micro = 0; started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for idx in chosen:
        example = make_example(tokenizer, records_train[idx])
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
            if step % 200 == 0:
                print('  step', step, '| loss', round(float(loss.item()*ACCUM), 4),
                      '| min', round((time.time()-started)/60, 1), flush=True)

    h1688 = small.copy(); h1688['hypothesis'] = generate(model, tokenizer, small)
    h1688['config'] = name
    h1688[['config','id','source','hypothesis','reference']].to_parquet(f'/kaggle/working/hyp_{name}_1688.parquet', index=False)
    htiki = tiki_test.copy(); htiki['hypothesis'] = generate(model, tokenizer, tiki_test)
    htiki['config'] = name
    htiki[['config','id','source','hypothesis']].to_parquet(f'/kaggle/working/hyp_{name}_tiki.parquet', index=False)
    results[name] = {'steps': step, 'tokens': round(total), 'minutes': round((time.time()-started)/60, 1)}
    print(name, results[name], flush=True)
    del model, base; gc.collect(); torch.cuda.empty_cache()

Path('/kaggle/working/train_runs.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
""")

code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'unbabel-comet', 'sacrebleu'], check=False)
from comet import download_model, load_from_checkpoint
import sacrebleu

da_path = download_model('Unbabel/wmt22-comet-da')
qe_path = download_model('Unbabel/wmt22-cometkiwi-da')
da = load_from_checkpoint(da_path)
qe = load_from_checkpoint(qe_path)

metrics = {}
for name in CONFIGS:
    h = pd.read_parquet(f'/kaggle/working/hyp_{name}_1688.parquet')
    data = [{'src': str(s), 'mt': str(m), 'ref': str(r)} for s, m, r in zip(h['source'], h['hypothesis'], h['reference'])]
    out = da.predict(data, batch_size=8, gpus=1)
    bleu = sacrebleu.corpus_bleu(h['hypothesis'].tolist(), [h['reference'].tolist()], tokenize='zh').score
    chrf = sacrebleu.corpus_chrf(h['hypothesis'].tolist(), [h['reference'].tolist()]).score
    t = pd.read_parquet(f'/kaggle/working/hyp_{name}_tiki.parquet')
    data_qe = [{'src': str(s), 'mt': str(m)} for s, m in zip(t['source'], t['hypothesis'])]
    out_qe = qe.predict(data_qe, batch_size=8, gpus=1)
    metrics[name] = {'comet_1688': round(float(out.system_score), 4),
                     'comet_qe_tiki': round(float(out_qe.system_score), 4),
                     'bleu_1688': round(float(bleu), 4),
                     'chrf_1688': round(float(chrf), 4),
                     'n_1688': len(data), 'n_tiki': len(data_qe)}
    print(name, metrics[name], flush=True)

Path('/kaggle/working/weight_eval_comet.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metrics, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "weight_eval_comet.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
