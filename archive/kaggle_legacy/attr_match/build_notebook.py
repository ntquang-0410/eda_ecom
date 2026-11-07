#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook match tên attribute 1688: chọn 1 key ZH trong danh sách của sản phẩm."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# Match attribute 1688 (VI → key ZH của chính sản phẩm)\n",
    "\n",
    "13.262 job chưa align. Mỗi job: một tên attribute tiếng Việt + danh sách key ZH\n",
    "còn trống của chính sản phẩm đó. Model chọn đúng 1 key hoặc `NONE`.\n",
]})

code("""import json, glob, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
try:
    import kagglehub
except Exception:
    kagglehub = None
""")

code("""INPUT = Path('/kaggle/input/1688-attr-match-jobs')
if not (INPUT / 'match_jobs.parquet').exists():
    hits = glob.glob('/kaggle/input/**/match_jobs.parquet', recursive=True)
    INPUT = Path(hits[0]).parent if hits else INPUT
print('INPUT', INPUT)
MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)
BATCH = 16
MAX_NEW = 32
OUT = Path('/kaggle/working/match_results.parquet')
""")

code("""def resolve_model():
    if kagglehub is not None:
        try:
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
    raise FileNotFoundError('no 4b weights')

SYSTEM = '''You match a Vietnamese e-commerce attribute key to the correct Chinese attribute key of the SAME product.
You are given the Vietnamese key and the list of Chinese keys that are still unused for that product.
Choose exactly ONE candidate with the same meaning. If none fits, answer "NONE".
Return JSON only: {"key_zh": "<one candidate>"} or {"key_zh": "NONE"}

Examples:
{"key_vi": "Sử dụng vị trí", "candidates": ["使用位置", "使用方式", "适用汽车品牌"]} -> {"key_zh": "使用位置"}
{"key_vi": "Thương hiệu xe áp dụng", "candidates": ["使用位置", "使用方式", "适用汽车品牌"]} -> {"key_zh": "适用汽车品牌"}
{"key_vi": "Cách sử dụng", "candidates": ["使用位置", "使用方式", "适用汽车品牌"]} -> {"key_zh": "使用方式"}
{"key_vi": "Chất liệu", "candidates": ["颜色", "尺寸"]} -> {"key_zh": "NONE"}'''

def build_prompt(tok, name, candidates):
    payload = json.dumps({'key_vi': name, 'candidates': candidates}, ensure_ascii=False)
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': payload}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(text, candidates):
    raw = (text or '').strip()
    a, b = raw.find('{'), raw.rfind('}')
    if a >= 0 and b > a:
        try:
            obj = json.loads(raw[a:b+1])
            value = str(obj.get('key_zh') or '').strip()
            if value in candidates:
                return value
            if value.upper() == 'NONE':
                return 'NONE'
        except json.JSONDecodeError:
            pass
    for cand in candidates:
        if cand and cand in raw:
            return cand
    return 'NONE'
""")

code("""import pandas as pd
jobs = pd.read_parquet(INPUT / 'match_jobs.parquet')
print('jobs', len(jobs))

model_dir = resolve_model()
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                             trust_remote_code=True, local_files_only=True,
                                             low_cpu_mem_usage=True).to('cuda').eval()
model.generation_config.pad_token_id = tok.pad_token_id
print('loaded | vram', round(torch.cuda.memory_allocated()/1e9, 1))

out = []
started = time.time()
for start in range(0, len(jobs), BATCH):
    batch = jobs.iloc[start:start+BATCH]
    prompts, meta = [], []
    for row in batch.itertuples():
        cands = json.loads(row.candidates)
        prompts.append(build_prompt(tok, row.name_vi, cands))
        meta.append((row.job_id, row.name_vi, cands))
    enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=768).to('cuda')
    with torch.inference_mode():
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    for (job_id, name, cands), raw in zip(meta, texts):
        out.append({'job_id': job_id, 'name_vi': name,
                    'key_zh_chosen': parse(raw, cands), 'raw': raw})
    if (start // BATCH) % 20 == 0:
        print('batch', start+len(batch), '/', len(jobs), flush=True)

res = pd.DataFrame(out)
res.to_parquet(OUT, index=False)
n_none = int((res['key_zh_chosen'] == 'NONE').sum())
print('wrote', OUT, len(res), '| NONE', n_none, f'({n_none/len(res)*100:.1f}%)',
      '| elapsed_min', round((time.time()-started)/60, 1))
print(res.head(10).to_string())
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "match_attr_keys.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
