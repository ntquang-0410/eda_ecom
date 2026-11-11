#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh kernel EcomGPT-7B A/B với LoRA giống Qwen, max_new theo stratum."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path('/content/audit_src/tiki-matrix-ecomgpt-a/tiki-matrix-ecomgpt-a.ipynb')

LEAK = '''import pandas as pd
ev = pd.read_parquet(DATA / 'eval_A.parquet')
for dsname in ('A', 'B'):
    U = pd.read_parquet(DATA / f'units_{dsname}.parquet')
    overlap = set(ev.product_id) & set(U.product_id)
    tok = float(U.eff_tokens.sum())
    print(dsname, 'rows', len(U), 'tokensM', round(tok/1e6, 3),
          'overlap', len(overlap), 'epochs@4M', round(4e6/tok, 3))
    assert not overlap, f'LEAK {dsname}'
print('LEAK CHECK OK')
'''

INFER = '''model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache = True
import re
def build(row):
    return TEMPLATE.format(instruction=INSTRUCTION, input=str(row['source']))

pred = {}
for max_new, part in eval_frame.groupby('max_new'):
    print('generate max_new', max_new, 'rows', len(part), flush=True)
    for start in range(0, len(part), 8):
        batch = part.iloc[start:start+8]
        enc = tokenizer([build(r) for _, r in batch.iterrows()], return_tensors='pt',
                        padding=True, truncation=True, max_length=1024).to('cuda')
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=int(max_new), do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for eid, text in zip(batch.eval_id, texts):
            pred[eid] = text.strip().split('###')[0].strip()
        if start % 160 == 0:
            print(start, '/', len(part), flush=True)

outs = [pred.get(eid, '') for eid in eval_frame.eval_id]
Path('/kaggle/working/work/pred.zh').write_text('\\n'.join(outs) + '\\n', encoding='utf-8')
print('done', len(outs))
'''


def make(ds: str, config: str, out_name: str, title: str) -> None:
    nb = json.loads(SRC.read_text(encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        t = ''.join(c['source'])
        if "DATASET = 'A'" in t:
            t = t.replace("DATASET = 'A'", f"DATASET = '{ds}'")
            t = t.replace("CONFIG = 'natural'", f"CONFIG = '{config}'")
            t = t.replace("idx, actual = quota_indices(units, cfg, 42, BUDGET)",
                          "idx, actual = quota_indices(units, cfg, 42, BUDGET)")
            t = t.replace("print('eval', len(eval_frame))",
                          "print('eval', len(eval_frame))\n"
                          "sample[['pair_id','group','eff_tokens']].to_parquet("
                          f"'/kaggle/working/sample_ecomgpt_{ds.lower()}_seed42.parquet', index=False)\n"
                          "train_tokens = float(units.eff_tokens.sum())\n"
                          "print('epochs_seen', round(float(sample.eff_tokens.sum())/train_tokens, 4))")
            c['source'] = t.splitlines(keepends=True)
        if 'model.eval(); model.gradient_checkpointing_disable()' in t:
            c['source'] = INFER.splitlines(keepends=True)
    leak_cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": LEAK.splitlines(keepends=True)}
    nb['cells'].insert(4, leak_cell)

    folder = ROOT / 'kaggle' / out_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{out_name}.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    meta = {
        "id": f"ntquang0410/{out_name}",
        "title": title,
        "code_file": f'{out_name}.ipynb',
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/ecomgpt-7b", "ntquang0410/tiki-matrix-data"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaRtxPro6000",
    }
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', folder)


if __name__ == '__main__':
    make('A', 'natural', 'tiki-matrix-ecomgpt-a2', 'Tiki Matrix Ecomgpt A2')
    make('B', 'balanced', 'tiki-matrix-ecomgpt-b', 'Tiki Matrix Ecomgpt B')
