#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook spike TPU sharding cho Qwen3-8B (PyTorch/XLA SPMD).

Notebook làm 3 việc, theo thứ tự:
1. Probe topology TPU: số XLA device, mesh khả dụng, API SPMD.
2. Kiểm chứng sharding trên Qwen3-0.6B: so output sharded vs single-device.
3. Smoke 8B: nạp sharded + generate vài prompt, đo latency + memory/chip.

Toàn bộ kết quả ghi ra /kaggle/working/shard_spike_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


md(
    """# Tiki Plan B — TPU v5e-8 sharding spike (Qwen3-8B)

Mục tiêu: tìm đường chạy **Qwen3-8B** trên `TpuV5E8` mà không cần RTX 96GB.

Kế hoạch 3 bước:

1. **Probe** — TPU có bao nhiêu XLA device thật, `xs.Mesh` + `mark_sharding` có dùng được không.
2. **Validate** — shard Qwen3-0.6B kiểu tensor-parallel, so output với single-device.
   Nếu text khớp nhau thì sharding đúng.
3. **Smoke 8B** — nạp 8B sharded, generate vài prompt, đo latency + memory/chip.

Quy ước sharding (kiểu Megatron tensor-parallel, chỉ 1 all-reduce mỗi block):

| module | partition | lý do |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj` | `('x', None)` | chia head ra từng chip, attention chạy local |
| `o_proj` | `(None, 'x')` | nhận input đã chia head, all-reduce output |
| `gate_proj`, `up_proj` | `('x', None)` | chia intermediate dim |
| `down_proj` | `(None, 'x')` | all-reduce output |
| `embed_tokens`, `lm_head`, norm | replicated | tránh all-gather vocab mỗi token |

Kết quả mong đợi: ~2.5-3.5 GB/chip cho 8B bf16 (thay vì 16 GB).
"""
)

md("## 1. Probe topology")

code(
    """import os, sys, json, time, glob, gc
os.environ.setdefault('XLA_USE_SPMD', '1')

from pathlib import Path
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return f'ERR:{type(e).__name__}:{e}'

def as_int(value, default=1):
    try:
        return int(value)
    except Exception:
        return default

PROBE = {
    'torch': torch.__version__,
    'torch_xla': torch_xla.__version__,
    'PJRT_DEVICE': os.getenv('PJRT_DEVICE'),
    'XLA_USE_SPMD': os.getenv('XLA_USE_SPMD'),
    'cpu_count': os.cpu_count(),
    'global_device_count': safe(xr.global_device_count),
    'global_runtime_device_count': safe(xr.global_runtime_device_count),
    'addressable_runtime_device_count': safe(xr.addressable_runtime_device_count),
    'supported_devices': safe(lambda: [str(d) for d in xm.get_xla_supported_devices()]),
}
for key, value in PROBE.items():
    print(f'{key}: {value}')
"""
)

md("## 2. Bật SPMD + tạo mesh")

code(
    """import numpy as np
import torch_xla.distributed.spmd as xs

N_DEV = as_int(
    safe(xr.global_device_count),
    as_int(safe(xr.global_runtime_device_count), 1),
)
if N_DEV <= 1:
    N_DEV = len(safe(xm.get_xla_supported_devices) or []) or 1

SPMD_API = 'none'
for module, attr in ((xr, 'use_spmd'), (xs, 'use_spmd')):
    if hasattr(module, attr):
        try:
            getattr(module, attr)()
            SPMD_API = f'{module.__name__}.{attr}'
            break
        except Exception as error:
            SPMD_API = f'ERR {module.__name__}.{attr}: {error}'

print('N_DEV', N_DEV)
print('SPMD_API', SPMD_API)
print('spmd module attrs', [a for a in dir(xs) if 'shard' in a.lower() or a in {'Mesh', 'use_spmd', 'clear_cache'}])

MESH = xs.Mesh(np.arange(N_DEV, dtype=np.int32), (N_DEV,), ('x',)) if N_DEV > 1 else None
print('MESH', MESH)
DEVICE = xm.xla_device()
print('DEVICE', DEVICE)
print('memory', safe(xm.get_memory_info, DEVICE))
"""
)

md("## 3. Cài transformers (bản 4.x để ổn định `apply_chat_template`)")

code(
    """!pip install -q 'transformers>=4.51,<5' accelerate sentencepiece protobuf
import transformers
print('transformers', transformers.__version__)
"""
)

md("## 4. Dataset + config")

code(
    """import pandas as pd

CONFIG_FILES = {
    'qwen3-0.6b': 'qwen3_0_6b.json',
    'qwen3-1.7b': 'qwen3_1_7b.json',
    'qwen3-4b':   'qwen3_4b.json',
    'qwen3-8b':   'qwen3_8b.json',
}
MODEL_SOURCES = {
    'qwen3-0.6b': 'qwen-lm/qwen-3/Transformers/0.6b',
    'qwen3-1.7b': 'qwen-lm/qwen-3/Transformers/1.7b',
    'qwen3-4b':   'qwen-lm/qwen-3/Transformers/4b',
    'qwen3-8b':   'qwen-lm/qwen-3/Transformers/8b',
}
HF_IDS = {
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
    'qwen3-4b':   'Qwen/Qwen3-4B',
    'qwen3-8b':   'Qwen/Qwen3-8B',
}

def find_dataset_dir():
    candidates = [
        Path('/kaggle/input/datasets/ntquang0410/tiki-plan-a-bt-input'),
        Path('/kaggle/input/tiki-plan-a-bt-input'),
    ]
    for base in candidates:
        if (base / 'tiki_bt_input.parquet').exists():
            return base
    matches = glob.glob('/kaggle/input/**/tiki_bt_input.parquet', recursive=True)
    if matches:
        return Path(matches[0]).parent
    raise FileNotFoundError('Không thấy tiki_bt_input.parquet')

DATASET = find_dataset_dir()

def find_config(model_key):
    name = CONFIG_FILES[model_key]
    for base in (DATASET, DATASET / 'configs', Path('/kaggle/input')):
        path = base / name
        if path.exists():
            return path
    matches = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if matches:
        return Path(matches[0])
    return None

def load_config(model_key):
    path = find_config(model_key)
    if path is None:
        return {'model_id': HF_IDS[model_key], 'model_source': MODEL_SOURCES[model_key]}
    cfg = json.loads(path.read_text(encoding='utf-8'))
    cfg['_config_path'] = str(path)
    return cfg

def resolve_model_path(model_key):
    source = MODEL_SOURCES.get(model_key)
    try:
        import kagglehub
        path = kagglehub.model_download(source)
        print('model from kagglehub', source, '->', path)
        return path
    except Exception as error:
        print('kagglehub failed:', type(error).__name__, error, '-> fallback HF', HF_IDS[model_key])
        return HF_IDS[model_key]

print('DATASET', DATASET)
for key in CONFIG_FILES:
    print(key, 'config', find_config(key))
"""
)

md("## 5. Prompt builder (giống notebook BT)")

code(
    """SYSTEM_PROMPT = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
Translate only the source text. Preserve brands, model codes, numbers and units.
Return JSON only: {"translation": "..."}
Do not explain, do not use markdown, do not translate the key itself.'''

def make_prompt(row):
    payload = {
        'task': 'translate_title' if not str(row.get('key_vi') or '').strip() else 'translate_value',
        'source_text': str(row.get('source_text_bt') or ''),
        'key_vi': str(row.get('key_vi') or ''),
        'key_zh': str(row.get('key_zh_effective') or ''),
        'value_class': str(row.get('value_class') or ''),
        'expected_zh_chars': int(row.get('expected_zh_value_chars') or 0),
    }
    return json.dumps(payload, ensure_ascii=False)

def parse_translation(text):
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.replace('```json', '', 1).replace('```', '').strip()
    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and obj.get('translation'):
                return str(obj['translation']).strip()
        except json.JSONDecodeError:
            pass
    return text.strip()

cache = pd.read_parquet(DATASET / 'tiki_translation_cache.parquet')
meta = pd.read_parquet(DATASET / 'tiki_bt_input.parquet')
meta = meta[meta['bt_route'] == 'mt'].drop_duplicates('cache_key')
meta_cols = [c for c in ['cache_key', 'expected_zh_value_chars'] if c in meta.columns]
df = cache[cache['bt_route'] == 'mt'].merge(meta[meta_cols], on='cache_key', how='left')
df['source_chars'] = df['source_text_bt'].fillna('').str.replace(' ', '', regex=False).str.len()
df = df.sort_values('source_chars').reset_index(drop=True)
SMOKE_ROWS = df.iloc[[0, len(df) // 3, 2 * len(df) // 3]].to_dict('records')
print('MT cache rows', len(df))
for row in SMOKE_ROWS:
    print('-', row['source_text_bt'][:90])
"""
)

md(
    """## 6. Sharding utilities

`shard_and_move` nạp từng parameter: CPU → XLA → `mark_sharding` → `mark_step`.
Nhờ vậy chip không bao giờ giữ full model; peak chỉ bằng parameter lớn nhất.
"""
)

code(
    """COLWISE = {'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'}
ROWWISE = {'o_proj', 'down_proj'}

def parent_module(model, param_name):
    parts = param_name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module

def set_parameter(model, name, value):
    parts = name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], value)

def partition_spec(model, name, mode):
    if 'lm_head' in name:
        return None
    module = parent_module(model, name)
    if not isinstance(module, torch.nn.Linear):
        return None
    last = name.split('.')[-1]
    if mode == 'tp':
        if last in COLWISE:
            return ('x', None)
        if last in ROWWISE:
            return (None, 'x')
        return None
    if mode == 'tp_simple':
        return (None, 'x')
    return None

def device_memory():
    info = safe(xm.get_memory_info, DEVICE)
    if isinstance(info, dict):
        used = as_int(info.get('bytes-used'), 0)
        limit = as_int(info.get('bytes-limit'), 0)
        return {'used_gb': round(used / 1e9, 3), 'limit_gb': round(limit / 1e9, 3)}
    return {'raw': str(info)}

def shard_and_move(model, mesh, mode='tp', mem_limit_frac=0.90):
    assert mesh is not None, 'Cần mesh > 1 device mới shard được'
    model.eval()
    model.requires_grad_(False)
    names = [name for name, _ in model.named_parameters()]
    stats = {'mode': mode, 'params': len(names), 'sharded': 0, 'replicated': 0, 'errors': []}
    started = time.time()
    largest = 0
    for index, name in enumerate(names, 1):
        param = model.get_parameter(name)
        size = param.numel() * param.element_size()
        largest = max(largest, size)
        spec = partition_spec(model, name, mode)
        moved = param.detach().to(DEVICE)
        if spec is not None:
            xs.mark_sharding(moved, mesh, spec)
            stats['sharded'] += 1
        else:
            stats['replicated'] += 1
        xm.mark_step()
        set_parameter(model, name, torch.nn.Parameter(moved, requires_grad=False))
        del param, moved
        if index % 40 == 0 or index == len(names):
            mem = device_memory()
            print(f'  [{index}/{len(names)}] sharded={stats["sharded"]} mem={mem}', flush=True)
            if mem.get('limit_gb') and mem['used_gb'] > mem['limit_gb'] * mem_limit_frac:
                stats['errors'].append(f'vượt ngưỡng memory tại {name}: {mem}')
                raise RuntimeError(f'Memory guard: {mem}')
    stats['largest_param_gb'] = round(largest / 1e9, 3)
    stats['seconds'] = round(time.time() - started, 1)
    stats['device_memory'] = device_memory()
    print('shard_and_move done', stats)
    return stats

def build_prompt(tokenizer, row):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': make_prompt(row)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_rows(model, tokenizer, rows, max_new=32, max_input=768, shard_inputs=True):
    outputs = []
    for row in rows:
        prompt = build_prompt(tokenizer, row)
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_input)
        enc = {key: value.to(DEVICE) for key, value in enc.items()}
        if MESH is not None and shard_inputs:
            for value in enc.values():
                xs.mark_sharding(value, MESH, (None, None))
        started = time.time()
        with torch.inference_mode():
            result = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        xm.mark_step()
        generated = result[:, enc['input_ids'].shape[1]:].cpu()
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append({
            'text': text,
            'translation': parse_translation(text),
            'seconds': round(time.time() - started, 2),
            'new_tokens': int(generated.shape[1]),
        })
    return outputs

def free_device():
    gc.collect()
    xm.mark_step()
    if hasattr(xs, 'clear_cache'):
        try:
            xs.clear_cache()
        except Exception:
            pass
    print('memory after free', device_memory())
"""
)

md(
    """## 7. Validate sharding trên Qwen3-0.6B

So sánh greedy output của:

- **baseline**: model nằm trên 1 device;
- **sharded**: cùng model, tensor-parallel trên `N_DEV` chip.

Greedy + bf16 nên text phải khớp (sai khác số học nhỏ có thể làm lệch token hiếm gặp).
"""
)

code(
    """from transformers import AutoTokenizer, AutoModelForCausalLM

VALIDATION = {'model_key': 'qwen3-0.6b', 'runs': {}, 'match': None}

small_key = 'qwen3-0.6b'
small_path = resolve_model_path(small_key)
tokenizer = AutoTokenizer.from_pretrained(small_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('tokenizer ok, vocab', tokenizer.vocab_size)

def load_model(model_key, path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=Path(path).is_dir(),
    )

# --- baseline: 1 device ---
base_model = load_model(small_key, small_path)
base_model.to(DEVICE)
base_model.eval()
base_out = generate_rows(base_model, tokenizer, SMOKE_ROWS, max_new=32, shard_inputs=False)
VALIDATION['runs']['baseline'] = base_out
print('baseline', json.dumps(base_out, ensure_ascii=False, indent=2)[:1200])
del base_model
free_device()
"""
)

code(
    """# --- sharded: tensor-parallel trên N_DEV chip ---
SHARD_MODE = 'tp'
if MESH is None:
    VALIDATION['error'] = 'N_DEV == 1, không có mesh để shard'
    print(VALIDATION['error'])
else:
    shard_model = load_model(small_key, small_path)
    shard_stats = shard_and_move(shard_model, MESH, mode=SHARD_MODE)
    shard_model.eval()
    shard_out = generate_rows(shard_model, tokenizer, SMOKE_ROWS, max_new=32)
    VALIDATION['runs']['sharded'] = shard_out
    VALIDATION['stats'] = shard_stats
    print('sharded', json.dumps(shard_out, ensure_ascii=False, indent=2)[:1200])

    base_texts = [item['text'] for item in VALIDATION['runs'].get('baseline', [])]
    shard_texts = [item['text'] for item in shard_out]
    matches = sum(1 for a, b in zip(base_texts, shard_texts) if a.strip() == b.strip())
    VALIDATION['match'] = f'{matches}/{len(shard_texts)}'
    print('exact match', VALIDATION['match'])
    for a, b in zip(base_texts, shard_texts):
        if a.strip() != b.strip():
            print('--- baseline ---'); print(a)
            print('--- sharded ---'); print(b)
    del shard_model
    free_device()
"""
)

md(
    """## 8. Smoke Qwen3-8B

Nếu bước 7 khớp (`match` = 3/3) thì sharding đúng. Bước này nạp 8B sharded thật,
generate 2 prompt, đo:

- memory/chip sau khi nạp;
- latency/prompt và token/s;
- output có parse được JSON `translation` không.
"""
)

code(
    """RUN_8B = True
EIGHTB = {'attempted': False}

if RUN_8B:
    try:
        path8 = resolve_model_path('qwen3-8b')
        tokenizer8 = AutoTokenizer.from_pretrained(path8, trust_remote_code=True)
        if tokenizer8.pad_token is None:
            tokenizer8.pad_token = tokenizer8.eos_token
        EIGHTB['attempted'] = True
        print('loading 8B on CPU (bf16)...')
        model8 = AutoModelForCausalLM.from_pretrained(
            path8,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=Path(path8).is_dir(),
        )
        print('8B loaded on CPU, params', sum(p.numel() for p in model8.parameters()) / 1e9, 'B')
        t0 = time.time()
        stats8 = shard_and_move(model8, MESH, mode='tp') if MESH is not None else {}
        model8.eval()
        EIGHTB['load_seconds'] = round(time.time() - t0, 1)
        EIGHTB['stats'] = stats8
        rows8 = SMOKE_ROWS[:2]
        out8 = generate_rows(model8, tokenizer8, rows8, max_new=64)
        EIGHTB['outputs'] = out8
        total_new = sum(item['new_tokens'] for item in out8) or 1
        total_sec = sum(item['seconds'] for item in out8) or 1
        EIGHTB['tokens_per_sec'] = round(total_new / total_sec, 2)
        EIGHTB['decode_seconds_per_entry'] = round(total_sec / len(out8), 2)
        print(json.dumps(EIGHTB, ensure_ascii=False, indent=2)[:2500])
        del model8
        free_device()
    except Exception as error:
        import traceback
        traceback.print_exc()
        EIGHTB['error'] = f'{type(error).__name__}: {error}'
        EIGHTB['memory_at_error'] = device_memory()
        print('8B FAILED:', EIGHTB['error'])
"""
)

md("## 9. Report")

code(
    """REPORT = {
    'probe': PROBE,
    'n_dev': N_DEV,
    'spmd_api': SPMD_API,
    'mesh': None if MESH is None else {'n_dev': N_DEV, 'axis': 'x'},
    'validation': VALIDATION,
    'eightb': EIGHTB if 'EIGHTB' in globals() else None,
    'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
out_path = Path('/kaggle/working/shard_spike_report.json')
out_path.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print('wrote', out_path, out_path.stat().st_size, 'bytes')
print(json.dumps({
    'n_dev': REPORT['n_dev'],
    'spmd_api': REPORT['spmd_api'],
    'validation_match': VALIDATION.get('match'),
    'validation_error': VALIDATION.get('error'),
    'eightb_ok': bool(EIGHTB.get('outputs')) if isinstance(EIGHTB, dict) else False,
    'eightb_error': EIGHTB.get('error') if isinstance(EIGHTB, dict) else None,
    'eightb_tokens_per_sec': EIGHTB.get('tokens_per_sec') if isinstance(EIGHTB, dict) else None,
}, ensure_ascii=False, indent=2))
"""
)

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "shard_spike.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(CELLS))
