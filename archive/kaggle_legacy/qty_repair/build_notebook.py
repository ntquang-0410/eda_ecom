#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook repair quantity mismatch (per-model, quantity-aware)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# Repair quantity mismatch (per-model)\n",
    "\n",
    "1.891 dòng `qty_mismatch`: dịch lại với prompt nhấn mạnh giữ số/đơn vị.\n",
    "Validate bằng chính `quantity_norm.compare` trước khi chấp nhận.\n",
]})

code("""import json, glob, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))
try:
    import kagglehub
except Exception:
    kagglehub = None
""")

code("""INPUT = Path('/kaggle/input/tiki-qty-repair-input')
if not (INPUT / 'qty_repair_input.parquet').exists():
    hits = glob.glob('/kaggle/input/**/qty_repair_input.parquet', recursive=True)
    INPUT = Path(hits[0]).parent if hits else INPUT
sys.path.insert(0, str(INPUT))
from quantity_norm import compare
print('INPUT', INPUT)

MODEL_SOURCES = {
    'qwen3-0.6b': 'qwen-lm/qwen-3/Transformers/0.6b/1',
    'qwen3-1.7b': 'qwen-lm/qwen-3/Transformers/1.7b/1',
    'qwen3-4b':   'qwen-lm/qwen-3/Transformers/4b/1',
    'qwen3-8b':   'qwen-lm/qwen-3/Transformers/8b/1',
}
EXPECTED_SHAPE = {
    'qwen3-0.6b': (1024, 3072, 28),
    'qwen3-1.7b': (2048, 6144, 28),
    'qwen3-4b':   (2560, 9728, 36),
    'qwen3-8b':   (4096, 12288, 36),
}
BATCH = {'qwen3-0.6b': 16, 'qwen3-1.7b': 16, 'qwen3-4b': 12, 'qwen3-8b': 8}
MAX_NEW = 1536
MAX_INPUT = 1280
OUT = Path('/kaggle/working/qty_repair.parquet')
""")

code("""SYS_A = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
Translate "source_text" into Simplified Chinese.
Rules:
- Preserve EVERY number from the source. Never add a number that is not in the source.
- Convert units correctly: cm->厘米, mm->毫米, m->米, kg->千克, g->克, ml->毫升, l->升, W->瓦, V->伏, inch->英寸.
- Keep brand names, model codes and SKUs unchanged.
- The output must contain Chinese characters. Never return the source unchanged.
Return JSON only: {"translation": "..."}'''

SYS_B = SYS_A + '''

The previous translation had quantity problems:
- missing quantities: __MISSING__
- extra quantities: __EXTRA__
Fix those problems: keep every number from the source, remove numbers that do not exist in the source.'''

def build_prompt(tok, row, variant):
    payload = {'source_text': row['source_text_bt']}
    system = SYS_A
    if variant == 'with_previous':
        system = SYS_B.replace('__MISSING__', str(row['missing'])).replace('__EXTRA__', str(row['extra']))
        payload['previous_translation'] = row['previous_target']
    if row.get('key_vi'):
        payload['key_vi'] = row['key_vi']
        payload['key_zh'] = row.get('key_zh_effective') or ''
    msgs = [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(text):
    raw = (text or '').strip()
    a, b = raw.find('{'), raw.rfind('}')
    if a >= 0 and b > a:
        try:
            obj = json.loads(raw[a:b+1])
            if isinstance(obj, dict) and str(obj.get('translation') or '').strip():
                return str(obj['translation']).strip()
        except json.JSONDecodeError:
            pass
    return ''
""")

code("""def resolve_model(model_key):
    if kagglehub is not None:
        try:
            return kagglehub.model_download(MODEL_SOURCES[model_key])
        except Exception as error:
            print('kagglehub failed:', error)
    for cfg in Path('/kaggle/input').rglob('config.json'):
        try:
            payload = json.loads(cfg.read_text(encoding='utf-8'))
        except Exception:
            continue
        if (payload.get('hidden_size'), payload.get('intermediate_size'),
                payload.get('num_hidden_layers')) == EXPECTED_SHAPE[model_key]:
            return str(cfg.parent)
    raise FileNotFoundError(model_key)

def load_model(path):
    kwargs = dict(trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kwargs)

def generate(model, tok, rows, variant, batch_size):
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        prompts = [build_prompt(tok, row, variant) for row in batch]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_INPUT).to('cuda')
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for row, raw in zip(batch, texts):
            target = parse(raw)
            result = compare(row['source_text_bt'], target, 'vi', 'zh')
            ok = bool(target) and result['match']
            out.append({'model_key': row['model_key'], 'pair_id': row['pair_id'],
                        'source_text_bt': row['source_text_bt'],
                        'new_target': target, 'raw': raw, 'good': ok,
                        'variant': variant})
        if (start // batch_size) % 10 == 0:
            print('  ', start + len(batch), '/', len(rows), flush=True)
    return out
""")

code("""import pandas as pd
jobs = pd.read_parquet(INPUT / 'qty_repair_input.parquet').to_dict('records')
print('jobs', len(jobs))

results = []
for model_key in ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']:
    subset = [r for r in jobs if r['model_key'] == model_key]
    if not subset:
        continue
    path = resolve_model(model_key)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = load_model(path).to('cuda').eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    print(model_key, 'loaded |', len(subset), 'rows', flush=True)

    # probe 24 dòng
    step = max(1, len(subset) // 24)
    probe = subset[::step][:24]
    scores = {}
    for variant in ('strict', 'with_previous'):
        recs = generate(model, tok, probe, variant, max(1, BATCH[model_key]//2))
        scores[variant] = sum(r['good'] for r in recs) / max(1, len(recs))
    order = sorted(scores, key=lambda k: scores[k], reverse=True)
    print(model_key, 'probe', scores, '->', order[0], flush=True)

    pending = subset
    for attempt, variant in enumerate(order, 1):
        if not pending:
            break
        recs = generate(model, tok, pending, variant, BATCH[model_key])
        good_ids = set()
        for rec in recs:
            rec['attempt'] = attempt
            if rec['good']:
                good_ids.add(rec['pair_id'])
                results.append(rec)
        pending = [r for r in pending if r['pair_id'] not in good_ids]

    for row in pending:
        results.append({'model_key': model_key, 'pair_id': row['pair_id'],
                        'source_text_bt': row['source_text_bt'], 'new_target': '',
                        'raw': '', 'good': False, 'variant': 'none', 'attempt': 3})

    del model
    torch.cuda.empty_cache()
    pd.DataFrame(results).to_parquet(OUT, index=False)
    print(model_key, 'done | good', sum(r['good'] for r in results if r['model_key']==model_key),
          '/', len(subset), flush=True)

res = pd.DataFrame(results)
res.to_parquet(OUT, index=False)
summary = res.groupby('model_key')['good'].agg(['sum', 'count']).to_dict('index')
print(json.dumps(summary, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "qty_repair.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
