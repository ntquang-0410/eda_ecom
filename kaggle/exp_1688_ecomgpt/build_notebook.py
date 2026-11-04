#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EcomGPT-7B zero/few-shot trên 1688 test_small (internet + ModelScope)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# 1688-only — EcomGPT-7B (zero/few-shot)\n",
    "\n",
    "Model: `damo/nlp_ecomgpt_multilingual-7B-ecom` (ModelScope).\n",
    "Prompt BLOOMZ, eval trên test_small.\n",
]})

code("""import json, math, re, subprocess, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'modelscope'], check=False)
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
""")

code("""INPUT = Path('/kaggle/input/tiki-1688-exp')
test = pd.read_parquet(INPUT / 'test_small.parquet').reset_index(drop=True)
train = pd.read_parquet(INPUT / 'train.parquet').reset_index(drop=True)
print('test', len(test), '| train', len(train))

from modelscope import snapshot_download
MODEL_DIR = snapshot_download('damo/nlp_ecomgpt_multilingual-7B-ecom',
                              cache_dir='/kaggle/working/modelscope')
print('model dir', MODEL_DIR)
""")

code("""from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                             trust_remote_code=True, device_map='cuda')
model.eval()
print('loaded')
""")

code("""INSTRUCTION = 'Translate the following Vietnamese e-commerce text into Simplified Chinese. Return only the translation.'
TEMPLATE = ('Below is an instruction that describes a task, paired with an input that provides further context. '
            'Write a response that appropriately completes the request.\\n\\n'
            '### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n')

def build_prompt(row, shots=None):
    text = str(row['source_text'])
    if row.get('key_vi'):
        text = f"{row['key_vi']}: {text}"
    prompt = TEMPLATE.format(instruction=INSTRUCTION, input=text)
    if shots:
        prefix = ''
        for src, tgt in shots:
            prefix += TEMPLATE.format(instruction=INSTRUCTION, input=src) + tgt + '\\n\\n'
        prompt = prefix + prompt
    return prompt

def generate(rows, shots=None, max_new=96, batch_size=8):
    outs = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start+batch_size]
        prompts = [build_prompt(row, shots) for _, row in batch.iterrows()]
        enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,
                        max_length=1024).to('cuda')
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        outs.extend([t.strip().split('###')[0].strip() for t in texts])
        if (start // batch_size) % 10 == 0:
            print('  ', start+len(batch), '/', len(rows), flush=True)
    return outs
""")

code("""def corpus_bleu(hyps, refs, n=4):
    matches, totals = [0]*n, [0]*n
    hl = rl = 0
    for h, r in zip(hyps, refs):
        ht = list(re.sub(r'\\s+', '', str(h))); rt = list(re.sub(r'\\s+', '', str(r)))
        hl += len(ht); rl += len(rt)
        for k in range(1, n+1):
            hc = Counter(tuple(ht[i:i+k]) for i in range(max(0, len(ht)-k+1)))
            rc = Counter(tuple(rt[i:i+k]) for i in range(max(0, len(rt)-k+1)))
            matches[k-1] += sum(min(v, rc[g]) for g, v in hc.items())
            totals[k-1] += sum(hc.values())
    prec = [(matches[i]/totals[i] if totals[i] else 0.0) for i in range(n)]
    if min(prec) == 0: return 0.0
    bp = 1.0 if hl >= rl else math.exp(1 - rl/max(1, hl))
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

def score(rows, outs):
    refs = rows['target_text'].tolist()
    return {'n': len(rows), 'bleu': round(corpus_bleu(outs, refs), 4),
            'chrf': round(float(np.mean([chrf(o, r) for o, r in zip(outs, refs)])), 4),
            'empty': round(float(np.mean([not o for o in outs])), 4)}
""")

code("""# probe 40 dòng: zero-shot vs 3-shot
probe = test.sample(n=min(40, len(test)), random_state=42).reset_index(drop=True)
shots = []
for _, row in train.sample(n=3, random_state=7).iterrows():
    src = str(row['source_text'])
    if row.get('key_vi'): src = f"{row['key_vi']}: {src}"
    shots.append((src, str(row['target_text'])))

zero = generate(probe, shots=None)
few = generate(probe, shots=shots)
print('probe zero-shot', score(probe, zero))
print('probe few-shot', score(probe, few))
use_few = score(probe, few)['chrf'] >= score(probe, zero)['chrf']
print('use_few', use_few)

outs = generate(test, shots=shots if use_few else None)
result = {'system': 'ecomgpt-7b', 'mode': 'few_shot' if use_few else 'zero_shot',
          'test_small': score(test, outs)}
Path('/kaggle/working/result_ecomgpt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
pd.DataFrame({'source': test['source_text'], 'reference': test['target_text'],
              'hypothesis': outs}).to_parquet('/kaggle/working/ecomgpt_outputs.parquet', index=False)
print(json.dumps(result, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "exp_1688_ecomgpt.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
