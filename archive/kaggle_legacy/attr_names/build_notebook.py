#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook dịch tên attribute 1688 (VI -> ZH) bằng Qwen3-4B."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# Dịch tên attribute 1688 (VI → ZH)\n",
    "\n",
    "Dùng Qwen3-4B dịch 3.881 tên attribute duy nhất sang thuật ngữ thương mại\n",
    "điện tử Trung Quốc chuẩn (2–6 ký tự). Kết quả dùng để re-align\n",
    "`attributes_vi` ↔ `attributes_zh` (hai list đang bị lệch thứ tự).\n",
]})

code("""import json, glob, time
from pathlib import Path
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print('transformers', transformers.__version__)
try:
    import kagglehub
except Exception:
    kagglehub = None
""")

code("""INPUT = Path('/kaggle/input/1688-attr-names-vi')
if not (INPUT / 'attr_names_vi.parquet').exists():
    hits = glob.glob('/kaggle/input/**/attr_names_vi.parquet', recursive=True)
    INPUT = Path(hits[0]).parent if hits else INPUT
print('INPUT', INPUT)
MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/4b/1'
SHAPE = (2560, 9728, 36)
BATCH = 32
MAX_NEW = 32
OUT = Path('/kaggle/working/attr_names_zh.parquet')
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

SYSTEM = '''You are an e-commerce attribute-key translator (Vietnamese -> Simplified Chinese).
Translate the field "key" into the standard Simplified Chinese attribute name used on 1688/Taobao (usually 2-6 characters).
Return JSON only: {"translation": "..."}

Examples:
{"key": "màu sắc"} -> {"translation": "颜色"}
{"key": "chất liệu"} -> {"translation": "材质"}
{"key": "thương hiệu"} -> {"translation": "品牌"}
{"key": "số mặt hàng"} -> {"translation": "货号"}
{"key": "phong cách"} -> {"translation": "风格"}
{"key": "chức năng"} -> {"translation": "功能"}
{"key": "xuất xứ"} -> {"translation": "产地"}
{"key": "kích thước"} -> {"translation": "尺寸"}
{"key": "trọng lượng"} -> {"translation": "重量"}
{"key": "bảo hành"} -> {"translation": "保修"}'''

def build_prompt(tok, name):
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps({'key': name}, ensure_ascii=False)}]
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
    return raw
""")

code("""import pandas as pd
names = pd.read_parquet(INPUT / 'attr_names_vi.parquet')
print('names', len(names))

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
for start in range(0, len(names), BATCH):
    batch = names.iloc[start:start+BATCH]
    prompts = [build_prompt(tok, n) for n in batch['name_vi']]
    enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
    with torch.inference_mode():
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    for name, raw in zip(batch['name_vi'], texts):
        out.append({'name_vi': name, 'name_zh_mt': parse(raw), 'raw': raw})
    if (start // BATCH) % 10 == 0:
        print('batch', start+len(batch), '/', len(names), flush=True)

res = pd.DataFrame(out)
res.to_parquet(OUT, index=False)
print('wrote', OUT, len(res), '| elapsed_min', round((time.time()-started)/60, 1))
print(res.head(10).to_string())
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "translate_attr_names.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, len(CELLS), "cells")
