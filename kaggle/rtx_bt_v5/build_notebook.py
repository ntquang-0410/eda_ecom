#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook BT v5: chạy lại toàn bộ 4 model với prompt brand-aware.

Khác v3:
- Prompt v5: brand/model giữ nguyên, nhưng danh từ tiếng Việt phải dịch, kể cả
  khi đứng cạnh brand.
- Gate `good` chặt hơn: loại luôn trường hợp model trả về đúng nhãn key
  (`target == key_zh`) — lỗi chiếm 31% ở 0.6B mà gate cũ bỏ lọt.
- Probe 80 dòng/model với 3 biến thể (brand_aware | no_key | v4_fewshot),
  chọn biến thể tốt nhất rồi mới chạy full.
- Dòng lỗi được thử lại bằng biến thể nhì (max_new 2048).

Output: /kaggle/working/plan_b_outputs_v5/<model>/predictions.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": text.splitlines(keepends=True),
    })


md("""# Tiki Plan B — BT v5 (4 model, brand-aware)

Chạy lại **toàn bộ 29.446 cache MT** cho 4 model Qwen3 với prompt v5.

Vì sao cần chạy lại: prompt v3 dạy model "preserve brands" quá mạnh, khiến
0.6B trả về đúng nhãn key (`"Đỏ"` + key `Màu sắc` → `颜色`) tới 31% số dòng.

Thay đổi chính:

1. Prompt brand-aware: chỉ giữ proper noun/brand/model code; danh từ Việt phải dịch.
2. Gate `good` loại `target == key_zh`.
3. Probe 80 dòng/model để chọn biến thể tốt nhất trước khi chạy full.
4. Retry dòng lỗi bằng biến thể nhì.
""")

code("""import json, glob, re, time
from pathlib import Path
import torch

print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    print('vram_gb', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
import transformers
print('transformers', transformers.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Qwen3ForCausalLM OK')
try:
    import kagglehub
    print('kagglehub', kagglehub.__version__)
except Exception as error:
    kagglehub = None
    print('kagglehub unavailable:', error)
""")

code("""OUTPUT_ROOT = Path('/kaggle/working/plan_b_outputs_v5')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
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
BATCH_BY_MODEL = {'qwen3-0.6b': 16, 'qwen3-1.7b': 16, 'qwen3-4b': 12, 'qwen3-8b': 8}

MAX_NEW_TOKENS = 1536
MAX_NEW_RETRY = 2048
MAX_INPUT_TOKENS = 1280
CHECKPOINT_EVERY = 200
PROBE_SIZE = 80
PROMPT_VERSION = 'plan_b_rtx_v5'
TIME_BUDGET_MIN = 660
DEADLINE = time.time() + TIME_BUDGET_MIN * 60
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('time_budget_min', TIME_BUDGET_MIN)
""")

code("""BRAND_AWARE = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
The field "source_text" is a product title or attribute value. Translate ALL Vietnamese words into Simplified Chinese.
Rules:
- Keep only proper nouns, brand names and Latin model codes unchanged (e.g. "Cozy Icetea", "iPhone 13", "La Roche-Posay").
- Vietnamese common nouns MUST be translated even when they stand next to a brand: "Chanh" -> 柠檬, "Túi" -> 袋装, "Hộp" -> 盒, "Ly" -> 杯, "Khóa" -> 锁.
- Never answer with the attribute key. The key is context only, not the answer.
- Keep numbers, sizes and units unchanged.
- The output MUST contain Chinese characters. Never return the source unchanged.
Return JSON only: {"translation": "..."}

Examples:
{"source_text": "Cozy Icetea Chanh 400gr Túi"} -> {"translation": "Cozy Icetea 柠檬味 400克 袋装"}
{"source_text": "Khóa Thắng Đĩa Master Lock 8303 DPS"} -> {"translation": "Master Lock 8303 DPS 碟刹锁"}
{"source_text": "Chì mày vặn 1 đầu Camelo"} -> {"translation": "Camelo 单头旋转眉笔"}
{"source_text": "Đỏ", "key_vi": "Màu sắc", "key_zh": "颜色"} -> {"translation": "红色"}
{"source_text": "Tre", "key_vi": "Chất liệu", "key_zh": "材质"} -> {"translation": "竹"}
{"source_text": "iPhone 15 Pro Max"} -> {"translation": "iPhone 15 Pro Max"}'''

V4_FEWSHOT = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
The field "source_text" is the VALUE that must be translated.
"key_vi"/"key_zh" describe what kind of attribute it is: use them as context only,
never translate or copy the key into the output.
Rules:
- Output MUST be written in Simplified Chinese characters.
- Never copy Vietnamese words, except brand names and model codes.
- Translate even a single noun (e.g. "Hổ" -> "虎").
- Keep numbers, units, sizes, SKUs and Latin model codes unchanged.
Return JSON only: {"translation": "..."}

Examples:
{"source_text": "Hổ", "key_vi": "Họa tiết"} -> {"translation": "虎"}
{"source_text": "Xám", "key_vi": "Mầu Sắc"} -> {"translation": "灰色"}
{"source_text": "Số 4: 5L", "key_vi": "Kích cỡ"} -> {"translation": "4号：5升"}
{"source_text": "Z Fold 3", "key_vi": "Size"} -> {"translation": "Z Fold 3"}'''

VARIANTS = {
    'brand_aware': (BRAND_AWARE, True),
    'no_key': (BRAND_AWARE, False),
    'v4_fewshot': (V4_FEWSHOT, True),
}
CJK_RE = re.compile(r'[\\u4e00-\\u9fff]')
VI_RE = re.compile(r'[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]', re.IGNORECASE)

def norm(value):
    return re.sub(r'\\s+', '', str(value or '')).lower()

def build_prompt(tokenizer, row, system, with_key=True):
    payload = {'source_text': str(row.get('source_text_bt') or '')}
    if with_key:
        payload['key_vi'] = str(row.get('key_vi') or '')
        payload['key_zh'] = str(row.get('key_zh_effective') or '')
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_translation(text):
    raw = (text or '').strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start:end + 1])
            if isinstance(payload, dict) and str(payload.get('translation') or '').strip():
                return str(payload['translation']).strip(), True
        except json.JSONDecodeError:
            pass
    return '', False

def is_good(source, target, key_zh):
    text = (target or '').strip()
    if not text:
        return False
    key = norm(key_zh)
    if key and norm(text) == key and norm(source) != key:
        return False  # model trả về nhãn key thay vì dịch value
    if CJK_RE.search(text):
        return True
    if VI_RE.search(text):
        return False
    return not VI_RE.search(str(source or ''))
""")

code("""import pandas as pd

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

cache = pd.read_parquet(DATASET / 'tiki_translation_cache.parquet')
meta = pd.read_parquet(DATASET / 'tiki_bt_input.parquet')
meta = meta[meta['bt_route'] == 'mt'].drop_duplicates('cache_key')
meta_cols = [c for c in ['cache_key', 'expected_zh_value_chars'] if c in meta.columns]
df = cache[cache['bt_route'] == 'mt'].merge(meta[meta_cols], on='cache_key', how='left')
df['source_chars'] = df['source_text_bt'].fillna('').str.replace(' ', '', regex=False).str.len()
df = df.sort_values(['min_priority', 'source_chars', 'cache_key']).reset_index(drop=True)

print('MT rows', len(df), '| duplicate', int(df['cache_key'].duplicated().sum()))
print(df[['source_text_bt', 'key_vi', 'min_priority']].head(3).to_string(index=False))
""")

code("""def find_model_dir(model_key):
    if kagglehub is not None:
        try:
            path = kagglehub.model_download(MODEL_SOURCES[model_key])
            print(model_key, 'kagglehub ->', path)
            return path
        except Exception as error:
            print(model_key, 'kagglehub failed:', type(error).__name__, error)

    expected = EXPECTED_SHAPE[model_key]
    for config_path in Path('/kaggle/input').rglob('config.json'):
        try:
            payload = json.loads(config_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        shape = (payload.get('hidden_size'), payload.get('intermediate_size'),
                 payload.get('num_hidden_layers'))
        if shape == expected:
            print(model_key, 'fallback ->', config_path.parent)
            return str(config_path.parent)
    raise FileNotFoundError('Không tìm thấy weights cho ' + model_key)

def load_model(model_dir):
    kwargs = dict(trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kwargs)
""")

code("""def generate_batch(model, tokenizer, rows, system, with_key, batch_size, max_new=MAX_NEW_TOKENS):
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start + batch_size].to_dict('records')
        prompts = [build_prompt(tokenizer, row, system, with_key) for row in batch]
        enc = tokenizer(prompts, return_tensors='pt', padding=True,
                        truncation=True, max_length=MAX_INPUT_TOKENS).to(DEVICE)
        input_truncated = (enc['attention_mask'].sum(dim=1) >= MAX_INPUT_TOKENS).tolist()
        started = time.time()
        with torch.inference_mode():
            output = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    num_beams=1, eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)
        latency_ms = (time.time() - started) * 1000 / max(1, len(batch))
        generated = output[:, enc['input_ids'].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        n_new = [int((row != tokenizer.pad_token_id).sum()) for row in generated]
        for row, raw, ntokens, truncated in zip(batch, texts, n_new, input_truncated):
            value, ok = parse_translation(raw)
            records.append({
                'cache_key': str(row['cache_key']),
                'source_text_bt': row.get('source_text_bt'),
                'key_vi': row.get('key_vi'),
                'key_zh_effective': row.get('key_zh_effective'),
                'value_class': row.get('value_class'),
                'target_text_bt': value,
                'raw_output': raw,
                'json_ok': bool(ok),
                'good': bool(is_good(row.get('source_text_bt'), value, row.get('key_zh_effective'))),
                'n_new_tokens': int(ntokens),
                'input_truncated': bool(truncated),
                'latency_ms': round(latency_ms, 2),
            })
        if (start // batch_size) % 10 == 0 or start + len(batch) >= len(rows):
            print('  batch', start + len(batch), '/', len(rows), flush=True)
    return records

def probe(model, tokenizer, rows, batch_size):
    sample = rows.assign(_len=rows['source_text_bt'].fillna('').str.len()).sort_values('_len')
    if len(sample) > PROBE_SIZE:
        idx = [int(i * (len(sample) - 1) / (PROBE_SIZE - 1)) for i in range(PROBE_SIZE)]
        sample = sample.iloc[idx]
    scores = {}
    for name, (system, with_key) in VARIANTS.items():
        recs = generate_batch(model, tokenizer, sample, system, with_key,
                              max(1, batch_size // 2), max_new=512)
        rate = sum(r['good'] for r in recs) / max(1, len(recs))
        scores[name] = round(rate, 4)
        print('probe', name, scores[name], flush=True)
    return scores
""")

code("""def run_model(model_key):
    out_dir = OUTPUT_ROOT / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'predictions.parquet'

    model_dir = find_model_dir(model_key)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(model_key, 'loading weights...')
    model = load_model(model_dir).to(DEVICE).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(model_key, 'loaded | vram_alloc_gb', round(torch.cuda.memory_allocated() / 1e9, 1))

    batch_size = BATCH_BY_MODEL[model_key]
    started = time.time()
    scores = probe(model, tokenizer, df, batch_size)
    order = sorted(scores, key=lambda k: scores[k], reverse=True)
    best = order[0]
    second = order[1] if len(order) > 1 else order[0]
    print(model_key, 'variant order', order, scores)

    frame = pd.DataFrame()
    stopped = 'completed'
    n_good = 0
    for start in range(0, len(df), CHECKPOINT_EVERY):
        if time.time() >= DEADLINE:
            stopped = 'time_budget'
            print(model_key, 'dừng do time budget tại', start, '/', len(df))
            break
        chunk = df.iloc[start:start + CHECKPOINT_EVERY].reset_index(drop=True)
        system, with_key = VARIANTS[best]
        recs = generate_batch(model, tokenizer, chunk, system, with_key, batch_size)
        for rec in recs:
            rec['attempt'] = 1
            rec['prompt_variant'] = best

        pending = [r for r in recs if not r['good']]
        if pending:
            pending_df = pd.DataFrame(pending)
            system2, with_key2 = VARIANTS[second]
            retry = generate_batch(model, tokenizer, pending_df, system2, with_key2,
                                   max(1, batch_size // 2), max_new=MAX_NEW_RETRY)
            for rec in retry:
                rec['attempt'] = 2
                rec['prompt_variant'] = second
            by_key = {r['cache_key']: r for r in recs}
            for rec in retry:
                if rec['good']:
                    by_key[rec['cache_key']] = rec
            recs = list(by_key.values())

        for rec in recs:
            rec['model_id'] = model_key
            rec['prompt_version'] = PROMPT_VERSION
            rec['qc_status'] = 'ok' if rec['good'] else 'needs_review'
            n_good += int(rec['good'])

        frame = pd.concat([frame, pd.DataFrame(recs)], ignore_index=True)
        frame.to_parquet(out_file, index=False)
        print(model_key, 'saved', len(frame), '/', len(df),
              'good', n_good, f'({n_good / len(frame) * 100:.1f}%)', flush=True)

    state = {
        'model_key': model_key,
        'n_done': int(len(frame)),
        'n_total': int(len(df)),
        'n_good': int(frame['good'].sum()) if len(frame) else 0,
        'good_rate': round(float(frame['good'].mean()), 4) if len(frame) else 0.0,
        'variant_scores': scores,
        'chosen_variant': best,
        'stopped': stopped,
        'elapsed_min': round((time.time() - started) / 60, 1),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (out_dir / 'run_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(model_key, 'DONE', json.dumps(state, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()
    return state
""")

code("""summary = []
for model_key in MODELS:
    if time.time() >= DEADLINE:
        print('Hết time budget, dừng trước', model_key)
        break
    summary.append(run_model(model_key))

payload = {'prompt_version': PROMPT_VERSION, 'runs': summary,
           'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
Path('/kaggle/working/plan_b_summary_v5.json').write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False, indent=2))
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
target = ROOT / "plan_b_bt_rtx_v5.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, "cells", len(CELLS))
