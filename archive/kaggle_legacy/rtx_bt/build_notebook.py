#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook BT full 4 model Qwen3 trên RTX Pro 6000 (offline, competition).

Ràng buộc: notebook có `competition_sources` -> Kaggle buộc `enable_internet=false`.
Vì vậy:
- KHÔNG pip install; dùng transformers có sẵn trong image (có preflight check).
- Model lấy từ `model_sources` (Kaggle Models) qua kagglehub + fallback quét /kaggle/input.

Khác notebook TPU: chạy tuần tự 4 model trên **một** GPU 96GB, batch lớn, có time budget.
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
    """# Tiki Plan B — Qwen3 BT 4 model trên RTX Pro 6000

- Input: `ntquang0410/tiki-plan-a-bt-input` (private dataset)
- Route `mt`: **29.446 cache entry**
- 4 model tuần tự: Qwen3-0.6B / 1.7B / 4B / 8B (bf16, 1 GPU 96GB)
- Offline hoàn toàn (competition notebook): model từ Kaggle Models
- Checkpoint mỗi 200 entry, resume theo `cache_key`
- Time budget 11h: dừng êm, giữ output

Output:

```text
/kaggle/working/plan_b_outputs/qwen3-0.6b/predictions.parquet
                             /qwen3-1.7b/predictions.parquet
                             /qwen3-4b/predictions.parquet
                             /qwen3-8b/predictions.parquet
                             /../run_state.json
```
"""
)

md("## 1. Phần cứng")

code(
    """import os, sys, json, glob, time
from pathlib import Path

import torch

print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('Notebook này cần GPU.')

GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
DEVICE = torch.device('cuda')
print('gpu', GPU_NAME)
print('vram_gb', VRAM_GB)
print('capability', torch.cuda.get_device_capability(0))

if VRAM_GB < 15:
    raise RuntimeError(f'VRAM {VRAM_GB}GB quá nhỏ cho pipeline này')
if 'RTX' not in GPU_NAME.upper() and 'PRO' not in GPU_NAME.upper():
    print(f'CẢNH BÁO: GPU không phải RTX Pro 6000 ({GPU_NAME}); vẫn chạy nhưng 8B có thể bị skip.')
"""
)

md("## 2. Preflight transformers (offline, không pip)")

code(
    """import transformers

VERSION = tuple(int(part) for part in transformers.__version__.split('.')[:2])
print('transformers', transformers.__version__)

if VERSION < (4, 51):
    raise RuntimeError(
        f'transformers {transformers.__version__} < 4.51 -> không hỗ trợ Qwen3. '
        'Notebook offline (competition) không pip install được; cần image mới hơn.'
    )

try:
    from transformers import Qwen3ForCausalLM  # noqa: F401
    print('Qwen3ForCausalLM OK')
except Exception as error:
    raise RuntimeError(f'Không import được Qwen3ForCausalLM: {error}')

try:
    import kagglehub
    print('kagglehub', getattr(kagglehub, '__version__', 'installed'))
except Exception as error:
    kagglehub = None
    print('kagglehub không có:', error, '-> sẽ chỉ dùng fallback /kaggle/input')
"""
)

md("## 3. Config")

code(
    """CONFIG_FILES = {
    'qwen3-0.6b': 'qwen3_0_6b.json',
    'qwen3-1.7b': 'qwen3_1_7b.json',
    'qwen3-4b':   'qwen3_4b.json',
    'qwen3-8b':   'qwen3_8b.json',
}
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

MODEL_KEYS_TO_RUN = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
LIMIT = None                 # None = toàn bộ 29.446 entry

# --- giới hạn sinh ---
# Số liệu thật trên 29.446 entry (expected_zh_value_chars):
#   p50 30 | p99 230 | p99.5 336 | p99.9 684 | max 1781
# 96 token (bản cũ) cắt cụt 976 entry (3.31%). 1536 phủ tới ~1200 ký tự Hán,
# phần dư xử lý bằng repair pass riêng (4296 token).
MAX_NEW_TOKENS = 1536
MAX_NEW_TOKENS_RETRY = 4096
# prompt token đo thật: p99 314 | max 1074 -> 1280 phủ 100%
MAX_INPUT_TOKENS = 1280
CHECKPOINT_EVERY = 200
TIME_BUDGET_MIN = 660        # 11h, dừng êm trước giới hạn session
PROMPT_VERSION = 'plan_b_rtx_v3'

if VRAM_GB >= 80:
    BATCH_BY_MODEL = {'qwen3-0.6b': 16, 'qwen3-1.7b': 16, 'qwen3-4b': 12, 'qwen3-8b': 8}
elif VRAM_GB >= 40:
    BATCH_BY_MODEL = {'qwen3-0.6b': 8, 'qwen3-1.7b': 8, 'qwen3-4b': 4, 'qwen3-8b': 2}
else:
    BATCH_BY_MODEL = {'qwen3-0.6b': 4, 'qwen3-1.7b': 2, 'qwen3-4b': 2, 'qwen3-8b': 1}

MIN_VRAM_BY_MODEL = {'qwen3-0.6b': 4, 'qwen3-1.7b': 8, 'qwen3-4b': 18, 'qwen3-8b': 40}

OUTPUT_ROOT = Path('/kaggle/working/plan_b_outputs')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DEADLINE = time.time() + TIME_BUDGET_MIN * 60

print('models', MODEL_KEYS_TO_RUN)
print('limit', LIMIT, '| batch', BATCH_BY_MODEL)
print('time_budget_min', TIME_BUDGET_MIN)
"""
)

md("## 4. Dataset")

code(
    """import pandas as pd

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
    raise FileNotFoundError('Không thấy tiki_bt_input.parquet trong /kaggle/input')

DATASET = find_dataset_dir()
print('DATASET', DATASET)
print('files', sorted(p.name for p in DATASET.glob('*.json')))

cache = pd.read_parquet(DATASET / 'tiki_translation_cache.parquet')
meta = pd.read_parquet(DATASET / 'tiki_bt_input.parquet')
meta = meta[meta['bt_route'] == 'mt'].drop_duplicates('cache_key')
meta_cols = [c for c in ['cache_key', 'expected_zh_value_chars'] if c in meta.columns]
df = cache[cache['bt_route'] == 'mt'].merge(meta[meta_cols], on='cache_key', how='left')
df['source_chars'] = df['source_text_bt'].fillna('').str.replace(' ', '', regex=False).str.len()
if LIMIT:
    df = df.sample(n=min(int(LIMIT), len(df)), random_state=42).reset_index(drop=True)
else:
    df = df.sort_values(['min_priority', 'source_chars', 'cache_key']).reset_index(drop=True)

print('MT rows', len(df), '| duplicate', int(df['cache_key'].duplicated().sum()))
print(df[['source_text_bt', 'key_vi', 'min_priority']].head(3).to_string(index=False))
"""
)

md("## 5. Prompt")

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

md("## 6. Resolve model offline + run")

code(
    """from transformers import AutoTokenizer, AutoModelForCausalLM

def find_config(model_key):
    name = CONFIG_FILES[model_key]
    for base in (DATASET, DATASET / 'configs', Path('/kaggle/input')):
        path = base / name
        if path.exists():
            return path
    matches = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    return Path(matches[0]) if matches else None

def load_config(model_key):
    path = find_config(model_key)
    if path is None:
        return {}
    return json.loads(path.read_text(encoding='utf-8'))

def signature(config_path):
    try:
        payload = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return (payload.get('hidden_size'), payload.get('intermediate_size'), payload.get('num_hidden_layers'))

def resolve_model_path(model_key):
    # Offline: ưu tiên model_sources qua kagglehub, fallback quét /kaggle/input theo signature
    if kagglehub is not None:
        try:
            path = kagglehub.model_download(MODEL_SOURCES[model_key])
            print(f'{model_key}: kagglehub -> {path}')
            return path
        except Exception as error:
            print(f'{model_key}: kagglehub failed ({type(error).__name__}: {error})')

    expected = EXPECTED_SHAPE[model_key]
    for config_path in Path('/kaggle/input').rglob('config.json'):
        if signature(config_path) == expected:
            print(f'{model_key}: fallback {config_path.parent}')
            return str(config_path.parent)
    raise FileNotFoundError(f'Không tìm được weights {model_key} trong model_sources lẫn /kaggle/input')

def parse_json_translation(text):
    # Trả (translation, ok). ok=False = JSON không đóng hoặc model sinh hỏng
    raw = (text or '').strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return '', False
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return '', False
    if not isinstance(payload, dict):
        return '', False
    value = str(payload.get('translation') or '').strip()
    return value, bool(value)

def generate_batch(model, tokenizer, rows, max_new):
    prompts = [build_prompt(tokenizer, row) for row in rows]
    enc = tokenizer(
        prompts, return_tensors='pt', padding=True,
        truncation=True, max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    lengths = enc['attention_mask'].sum(dim=1).tolist()
    input_truncated = [bool(length >= MAX_INPUT_TOKENS) for length in lengths]

    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = (time.time() - started) * 1000

    generated = output[:, enc['input_ids'].shape[1]:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    n_new = [int((row != tokenizer.pad_token_id).sum()) for row in generated]
    return texts, n_new, latency_ms / max(1, len(rows)), input_truncated

def make_record(model_key, row, raw, n_new, input_truncated, latency_ms, attempt=1):
    value, ok = parse_json_translation(raw)
    return {
        'cache_key': str(row['cache_key']),
        'source_text_bt': row.get('source_text_bt'),
        'key_vi': row.get('key_vi'),
        'key_zh_effective': row.get('key_zh_effective'),
        'value_class': row.get('value_class'),
        'target_text_bt': value,
        'raw_output': raw,
        'json_ok': bool(ok),
        'n_new_tokens': int(n_new),
        'input_truncated': bool(input_truncated),
        'attempt': int(attempt),
        'model_id': model_key,
        'prompt_version': PROMPT_VERSION,
        'qc_status': 'pending' if ok else 'needs_review',
        'latency_ms': round(latency_ms, 2),
    }

def write_state(model_key, frame, out_dir, started, extra=None):
    payload = {
        'model_key': model_key,
        'gpu': GPU_NAME,
        'vram_gb': VRAM_GB,
        'n_done': int(len(frame)),
        'n_total': int(len(df)),
        'n_json_ok': int(frame['json_ok'].astype(bool).sum()) if 'json_ok' in frame.columns else None,
        'n_needs_review': int((~frame['json_ok'].astype(bool)).sum()) if 'json_ok' in frame.columns else None,
        'elapsed_min': round((time.time() - started) / 60, 1),
        'vram_alloc_gb': round(torch.cuda.memory_allocated() / 1e9, 1),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    if extra:
        payload.update(extra)
    (out_dir / 'run_state.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def repair_failures(model_key, model, tokenizer, out_file, out_dir, started):
    # Sinh lại các row json_ok=False với cap lớn hơn (batch 1)
    frame = pd.read_parquet(out_file)
    if 'json_ok' not in frame.columns:
        return 0
    bad = frame.index[~frame['json_ok'].astype(bool)]
    print(f'{model_key}: repair {len(bad):,} row lỗi JSON / {len(frame):,}')
    fixed = 0
    for count, idx in enumerate(bad, 1):
        if time.time() >= DEADLINE:
            print(f'{model_key}: repair dừng do time budget tại {count}/{len(bad)}')
            break
        row = frame.loc[idx].to_dict()
        best = None
        for attempt, cap in ((2, MAX_NEW_TOKENS_RETRY), (3, MAX_NEW_TOKENS_RETRY * 2)):
            texts, n_new, latency, input_trunc = generate_batch(model, tokenizer, [row], cap)
            value, ok = parse_json_translation(texts[0])
            best = (texts[0], value, ok, n_new[0], input_trunc[0], latency, attempt)
            if ok:
                break
        raw, value, ok, n_tokens, input_trunc, latency, attempt = best
        frame.at[idx, 'raw_output'] = raw
        frame.at[idx, 'target_text_bt'] = value
        frame.at[idx, 'json_ok'] = bool(ok)
        frame.at[idx, 'n_new_tokens'] = int(n_tokens)
        frame.at[idx, 'input_truncated'] = bool(input_trunc)
        frame.at[idx, 'attempt'] = int(attempt)
        frame.at[idx, 'qc_status'] = 'pending' if ok else 'needs_review'
        frame.at[idx, 'latency_ms'] = round(latency, 2)
        fixed += int(ok)
        if count % 25 == 0 or count == len(bad):
            frame.to_parquet(out_file, index=False)
            print(f'{model_key}: repair {count}/{len(bad)} fixed={fixed}', flush=True)
    frame.to_parquet(out_file, index=False)
    print(f'{model_key}: repair xong, fixed {fixed}/{len(bad)}')
    return fixed

def load_model_weights(model_path):
    # transformers >= 5 đổi torch_dtype -> dtype; thử dtype trước, fallback torch_dtype
    kwargs = dict(trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **kwargs)

def run_model(model_key):
    out_dir = OUTPUT_ROOT / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'predictions.parquet'

    if VRAM_GB < MIN_VRAM_BY_MODEL[model_key]:
        print(f'SKIP {model_key}: cần >= {MIN_VRAM_BY_MODEL[model_key]}GB VRAM, có {VRAM_GB}GB')
        return {'model_key': model_key, 'skipped': True, 'reason': f'vram {VRAM_GB}GB'}

    existing = pd.DataFrame()
    done = set()
    if out_file.exists():
        existing = pd.read_parquet(out_file)
        done = set(existing['cache_key'].astype(str))
        print(f'{model_key}: resume {len(done):,}')

    todo = df[~df['cache_key'].astype(str).isin(done)].reset_index(drop=True)
    print(f'{model_key}: todo {len(todo):,} / {len(df):,}')

    if todo.empty and existing.empty:
        return {'model_key': model_key, 'n_done': 0, 'n_total': len(df), 'skipped': False}

    if not todo.empty:
        model_path = resolve_model_path(model_key)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f'{model_key}: loading weights...')
        model = load_model_weights(model_path).to(DEVICE)
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        print(f'{model_key}: loaded | vram_alloc_gb', round(torch.cuda.memory_allocated() / 1e9, 1))

        batch_size = int(BATCH_BY_MODEL[model_key])
        records = []
        started = time.time()
        stopped = 'completed'
        n_ok = 0

        for start in range(0, len(todo), batch_size):
            if time.time() >= DEADLINE:
                stopped = 'time_budget'
                print(f'{model_key}: dừng do time budget tại {start}/{len(todo)}')
                break

            batch = todo.iloc[start:start + batch_size]
            rows = batch.to_dict('records')
            texts, n_new, latency, input_trunc = generate_batch(model, tokenizer, rows, MAX_NEW_TOKENS)

            for row, raw, n_tokens, truncated, lat in zip(rows, texts, n_new, input_trunc, [latency] * len(rows)):
                record = make_record(model_key, row, raw, n_tokens, truncated, lat)
                records.append(record)
                n_ok += int(record['json_ok'])

            if len(records) >= CHECKPOINT_EVERY or start + batch_size >= len(todo):
                part = pd.DataFrame(records)
                merged = pd.concat([existing, part], ignore_index=True) if len(existing) else part
                merged.to_parquet(out_file, index=False)
                existing = merged
                records = []
                elapsed_min = (time.time() - started) / 60
                write_state(model_key, merged, out_dir, started, {
                    'entries_per_min': round(len(merged) / max(1e-9, elapsed_min), 1),
                })
                print(f'{model_key}: saved {len(merged):,}/{len(df):,} '
                      f'json_ok={n_ok:,} ({len(merged) / max(1e-9, elapsed_min):.0f} entry/min)', flush=True)

        fixed = repair_failures(model_key, model, tokenizer, out_file, out_dir, started)
        del model
        torch.cuda.empty_cache()
        elapsed_min = round((time.time() - started) / 60, 1)
        final = pd.read_parquet(out_file)
        n_ok_final = int(final['json_ok'].astype(bool).sum())
        write_state(model_key, final, out_dir, started, {
            'entries_per_min': round(len(final) / max(1e-9, elapsed_min), 1),
            'repair_fixed': int(fixed),
            'stopped': stopped,
        })
        print(f'{model_key}: {stopped} in {elapsed_min} min | json_ok {n_ok_final:,}/{len(final):,}')
        return {'model_key': model_key, 'n_done': int(len(final)), 'n_total': int(len(df)),
                'n_json_ok': n_ok_final, 'n_needs_review': int(len(final) - n_ok_final),
                'repair_fixed': int(fixed), 'stopped': stopped, 'elapsed_min': elapsed_min}

    # resume-only: vẫn repair nếu còn row lỗi
    return {'model_key': model_key, 'n_done': int(len(existing)), 'n_total': int(len(df)),
            'n_json_ok': int(existing['json_ok'].astype(bool).sum()) if 'json_ok' in existing.columns else None,
            'stopped': 'resumed'}
"""
)

md("## 7. Chạy 4 model")

code(
    """summaries = []
for model_key in MODEL_KEYS_TO_RUN:
    if time.time() >= DEADLINE:
        print('hết time budget, dừng trước', model_key)
        break
    try:
        summaries.append(run_model(model_key))
    except Exception as error:
        import traceback
        traceback.print_exc()
        summaries.append({'model_key': model_key, 'error': f'{type(error).__name__}: {error}'})
        torch.cuda.empty_cache()
    print(json.dumps(summaries[-1], ensure_ascii=False))

print(json.dumps(summaries, ensure_ascii=False, indent=2))
"""
)

md("## 8. Tổng kết")

code(
    """per_model = {}
total_done = 0
for model_key in MODEL_KEYS_TO_RUN:
    path = OUTPUT_ROOT / model_key / 'predictions.parquet'
    n = len(pd.read_parquet(path)) if path.exists() else 0
    per_model[model_key] = n
    total_done += n

summary = {
    'gpu': GPU_NAME,
    'vram_gb': VRAM_GB,
    'n_mt_total': int(len(df)),
    'per_model': per_model,
    'expected_per_model': int(len(df)),
    'runs': summaries,
    'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
Path('/kaggle/working/plan_b_rtx_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))

print('Files:')
for path in sorted(OUTPUT_ROOT.rglob('*')):
    if path.is_file():
        print(' ', path.relative_to(OUTPUT_ROOT), round(path.stat().st_size / 1e6, 2), 'MB')
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
out = ROOT / "plan_b_bt_rtx.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(CELLS))
