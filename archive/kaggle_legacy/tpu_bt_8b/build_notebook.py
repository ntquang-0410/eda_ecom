#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook BT full Qwen3-8B trên TPU v5e-8 (SPMD sharding).

Khác notebook BT 0.6B/1.7B/4B:
- nạp 8B sharded trên 8 chip (xem `kaggle/tpu_shard/README.md`);
- chia 29.446 MT entry thành `N_SLICES` phần (round-robin) và ghi từng phần
  thành parquet riêng, để mỗi session TPU chỉ cần xong 1-2 phần;
- có time budget: dừng êm trước giới hạn session, checkpoint vẫn được lưu.
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
    """# Tiki Plan B — Qwen3-8B BT trên TPU v5e-8 (SPMD sharding)

- Input: `ntquang0410/tiki-plan-a-bt-input`
- Chỉ dịch route `mt` → **29.446 cache entry**
- 8B bf16 chạy **sharded trên 8 chip** (tensor-parallel, 1 all-reduce / block)
- Chia việc thành `N_SLICES` phần round-robin, mỗi phần một parquet
- Time budget: dừng êm trước khi session bị cắt

Output:

```text
/kaggle/working/plan_b_outputs/qwen3-8b/slice_00of04.parquet
                                     /slice_01of04.parquet ...
```
"""
)

md("## 1. Probe + SPMD mesh")

code(
    """import os, sys, json, time, glob, gc
os.environ.setdefault('XLA_USE_SPMD', '1')

from pathlib import Path
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

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

print('torch', torch.__version__, '| torch_xla', torch_xla.__version__)
print('PJRT_DEVICE', os.getenv('PJRT_DEVICE'), '| XLA_USE_SPMD', os.getenv('XLA_USE_SPMD'))

N_DEV = as_int(safe(xr.global_device_count), as_int(safe(xr.global_runtime_device_count), 1))
if N_DEV <= 1:
    N_DEV = len(safe(xm.get_xla_supported_devices) or []) or 1
print('N_DEV', N_DEV)

SPMD_API = 'none'
for module, attr in ((xr, 'use_spmd'), (xs, 'use_spmd')):
    if hasattr(module, attr):
        try:
            getattr(module, attr)()
            SPMD_API = f'{module.__name__}.{attr}'
            break
        except Exception as error:
            SPMD_API = f'ERR {module.__name__}.{attr}: {error}'
print('SPMD_API', SPMD_API)

if N_DEV < 2:
    raise RuntimeError('Cần >= 2 XLA device để shard 8B. Kiểm tra machine_shape=TpuV5E8.')

MESH = xs.Mesh(np.arange(N_DEV, dtype=np.int32), (N_DEV,), ('x',))
DEVICE = xm.xla_device()
print('MESH', MESH, '| DEVICE', DEVICE)
print('memory', safe(xm.get_memory_info, DEVICE))
"""
)

md("## 2. Deps")

code(
    """!pip install -q 'transformers>=4.51,<5' accelerate sentencepiece protobuf
import transformers
print('transformers', transformers.__version__)
"""
)

md("## 3. Config")

code(
    """import pandas as pd

MODEL_KEY = 'qwen3-8b'
MODEL_SOURCE = 'qwen-lm/qwen-3/Transformers/8b'
HF_FALLBACK = 'Qwen/Qwen3-8B'
CONFIG_NAME = 'qwen3_8b.json'

# --- chia việc ---
N_SLICES = 4
SLICE_START = 0          # session sau đổi thành 1, 2, 3 nếu chưa xong
MAX_BATCHES_PER_SLICE = None   # None = chạy hết slice (vẫn bị chặn bởi TIME_BUDGET_MIN)

# --- generation ---
BATCH_SIZE = 2
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 96
CHECKPOINT_EVERY = 40
TIME_BUDGET_MIN = 450    # dừng êm trước giới hạn session TPU

PROMPT_VERSION = 'plan_b_tpu_v1'
OUTPUT_ROOT = Path('/kaggle/working/plan_b_outputs')
OUT_DIR = OUTPUT_ROOT / MODEL_KEY
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEADLINE = time.time() + TIME_BUDGET_MIN * 60

CONFIG_FILES = {
    'qwen3-0.6b': 'qwen3_0_6b.json',
    'qwen3-1.7b': 'qwen3_1_7b.json',
    'qwen3-4b':   'qwen3_4b.json',
    'qwen3-8b':   CONFIG_NAME,
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
print('DATASET', DATASET)

def find_config(model_key):
    name = CONFIG_FILES[model_key]
    for base in (DATASET, DATASET / 'configs', Path('/kaggle/input')):
        path = base / name
        if path.exists():
            return path
    matches = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    return Path(matches[0]) if matches else None

def resolve_model_path():
    try:
        import kagglehub
        path = kagglehub.model_download(MODEL_SOURCE)
        print('model from kagglehub ->', path)
        return path
    except Exception as error:
        print('kagglehub failed:', type(error).__name__, error, '-> HF', HF_FALLBACK)
        return HF_FALLBACK

_config_path = find_config(MODEL_KEY)
CFG = json.loads(_config_path.read_text(encoding='utf-8')) if _config_path else {}
print('config', _config_path, CFG)
print('plan', {'N_SLICES': N_SLICES, 'SLICE_START': SLICE_START, 'batch': BATCH_SIZE,
               'max_input': MAX_INPUT_TOKENS, 'max_new': MAX_NEW_TOKENS,
               'time_budget_min': TIME_BUDGET_MIN})
"""
)

md("## 4. Dataset → slice round-robin")

code(
    """cache = pd.read_parquet(DATASET / 'tiki_translation_cache.parquet')
meta = pd.read_parquet(DATASET / 'tiki_bt_input.parquet')
meta = meta[meta['bt_route'] == 'mt'].drop_duplicates('cache_key')
meta_cols = [c for c in ['cache_key', 'expected_zh_value_chars'] if c in meta.columns]
df = cache[cache['bt_route'] == 'mt'].merge(meta[meta_cols], on='cache_key', how='left')
df['source_chars'] = df['source_text_bt'].fillna('').str.replace(' ', '', regex=False).str.len()
df = df.sort_values(['min_priority', 'source_chars', 'cache_key']).reset_index(drop=True)
df['slice_index'] = np.arange(len(df)) % N_SLICES

print('MT entries', len(df), '| slice sizes', df['slice_index'].value_counts().sort_index().to_dict())

def slice_frame(slice_index):
    frame = df[df['slice_index'] == slice_index].reset_index(drop=True)
    return frame.sample(frac=1.0, random_state=slice_index).reset_index(drop=True)

def slice_path(slice_index):
    return OUT_DIR / f'slice_{slice_index:02d}of{N_SLICES:02d}.parquet'
"""
)

md("## 5. Prompt builder")

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
"""
)

md("## 6. Sharding + load 8B")

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

def partition_spec(model, name):
    if 'lm_head' in name:
        return None
    module = parent_module(model, name)
    if not isinstance(module, torch.nn.Linear):
        return None
    last = name.split('.')[-1]
    if last in COLWISE:
        return ('x', None)
    if last in ROWWISE:
        return (None, 'x')
    return None

def device_memory():
    info = safe(xm.get_memory_info, DEVICE)
    if isinstance(info, dict):
        return {'used_gb': round(as_int(info.get('bytes-used'), 0) / 1e9, 2),
                'limit_gb': round(as_int(info.get('bytes-limit'), 0) / 1e9, 2)}
    return {'raw': str(info)}

def shard_and_move(model, mem_limit_frac=0.92):
    model.eval()
    model.requires_grad_(False)
    names = [name for name, _ in model.named_parameters()]
    sharded = replicated = 0
    for index, name in enumerate(names, 1):
        param = model.get_parameter(name)
        spec = partition_spec(model, name)
        moved = param.detach().to(DEVICE)
        if spec is not None:
            xs.mark_sharding(moved, MESH, spec)
            sharded += 1
        else:
            replicated += 1
        xm.mark_step()
        set_parameter(model, name, torch.nn.Parameter(moved, requires_grad=False))
        del param, moved
        if index % 40 == 0 or index == len(names):
            mem = device_memory()
            print(f'  load [{index}/{len(names)}] sharded={sharded} mem={mem}', flush=True)
            if mem.get('limit_gb') and mem['used_gb'] > mem['limit_gb'] * mem_limit_frac:
                raise RuntimeError(f'Memory guard tại {name}: {mem}')
    print('loaded sharded params', sharded, 'replicated', replicated, device_memory())

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = resolve_model_path()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

t_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=Path(model_path).is_dir(),
)
print('CPU load', round(time.time() - t_load, 1), 's | params',
      round(sum(p.numel() for p in model.parameters()) / 1e9, 2), 'B')
shard_and_move(model)
model.eval()
model.generation_config.pad_token_id = tokenizer.pad_token_id
print('device after load', device_memory())
"""
)

md("## 7. Run slices với time budget")

code(
    """def run_slice(slice_index):
    frame = slice_frame(slice_index)
    out_file = slice_path(slice_index)
    state_file = OUT_DIR / f'state_slice_{slice_index:02d}.json'

    existing = pd.DataFrame()
    done = set()
    if out_file.exists():
        existing = pd.read_parquet(out_file)
        done = set(existing['cache_key'].astype(str))
        print(f'[slice {slice_index}] resume {len(done)}')
    todo = frame[~frame['cache_key'].astype(str).isin(done)].reset_index(drop=True)
    print(f'[slice {slice_index}] todo {len(todo)} / {len(frame)}')

    records = []
    n_batches = 0
    started = time.time()
    stopped = 'completed'

    for start in range(0, len(todo), BATCH_SIZE):
        if time.time() >= DEADLINE:
            stopped = 'time_budget'
            print(f'[slice {slice_index}] dừng do time budget tại {start}/{len(todo)}')
            break
        if MAX_BATCHES_PER_SLICE and n_batches >= MAX_BATCHES_PER_SLICE:
            stopped = 'max_batches'
            break

        batch = todo.iloc[start:start + BATCH_SIZE]
        prompts = [build_prompt(tokenizer, row) for row in batch.to_dict('records')]
        enc = tokenizer(prompts, return_tensors='pt', padding=True,
                        truncation=True, max_length=MAX_INPUT_TOKENS)
        enc = {key: value.to(DEVICE) for key, value in enc.items()}
        for value in enc.values():
            xs.mark_sharding(value, MESH, (None, None))

        t0 = time.time()
        with torch.inference_mode():
            result = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        xm.mark_step()
        latency_ms = (time.time() - t0) * 1000
        generated = result[:, enc['input_ids'].shape[1]:].cpu()
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for row, raw in zip(batch.to_dict('records'), texts):
            records.append({
                'cache_key': str(row['cache_key']),
                'source_text_bt': row.get('source_text_bt'),
                'key_vi': row.get('key_vi'),
                'key_zh_effective': row.get('key_zh_effective'),
                'value_class': row.get('value_class'),
                'target_text_bt': parse_translation(raw),
                'raw_output': raw,
                'model_id': MODEL_KEY,
                'prompt_version': PROMPT_VERSION,
                'slice_index': slice_index,
                'n_slices': N_SLICES,
                'qc_status': 'pending',
                'latency_ms': round(latency_ms / max(1, len(batch)), 2),
            })
        n_batches += 1

        if len(records) >= CHECKPOINT_EVERY or start + BATCH_SIZE >= len(todo):
            part = pd.DataFrame(records)
            merged = pd.concat([existing, part], ignore_index=True) if len(existing) else part
            merged.to_parquet(out_file, index=False)
            existing = merged
            records = []
            state_file.write_text(json.dumps({
                'slice_index': slice_index,
                'n_slices': N_SLICES,
                'n_done': int(len(merged)),
                'n_slice_total': int(len(frame)),
                'n_mt_total': int(len(df)),
                'batches': n_batches,
                'elapsed_min': round((time.time() - started) / 60, 1),
                'entries_per_min': round(len(merged) / max(1e-9, (time.time() - started) / 60), 1),
                'device_memory': device_memory(),
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'[slice {slice_index}] saved {len(merged)}/{len(frame)} '
                  f'({round(len(merged) / max(1e-9, (time.time() - started) / 60), 1)} entry/min)', flush=True)

    return {'slice_index': slice_index, 'stopped': stopped, 'n_done': int(len(existing)),
            'n_slice_total': int(len(frame)), 'out_file': str(out_file),
            'elapsed_min': round((time.time() - started) / 60, 1)}

summaries = []
for slice_index in range(SLICE_START, N_SLICES):
    if time.time() >= DEADLINE:
        print('hết time budget, dừng trước slice', slice_index)
        break
    summaries.append(run_slice(slice_index))

print(json.dumps(summaries, ensure_ascii=False, indent=2))
"""
)

md("## 8. Tổng kết + trạng thái resume")

code(
    """done_total = 0
per_slice = {}
for slice_index in range(N_SLICES):
    path = slice_path(slice_index)
    n = len(pd.read_parquet(path)) if path.exists() else 0
    per_slice[slice_index] = n
    done_total += n

missing = [i for i in range(N_SLICES) if per_slice[i] < int((df['slice_index'] == i).sum())]
summary = {
    'model': MODEL_KEY,
    'n_mt_total': int(len(df)),
    'n_done': int(done_total),
    'coverage': round(done_total / max(1, len(df)) * 100, 2),
    'per_slice': per_slice,
    'missing_slices': missing,
    'next_slice_start': missing[0] if missing else None,
    'device_memory': device_memory(),
    'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
Path('/kaggle/working/plan_b_qwen3_8b_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print()
print('Files:')
for path in sorted(OUT_DIR.glob('*')):
    print(' ', path.name, round(path.stat().st_size / 1e6, 2), 'MB')
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
out = ROOT / "plan_b_bt_tpu_8b.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(CELLS))
