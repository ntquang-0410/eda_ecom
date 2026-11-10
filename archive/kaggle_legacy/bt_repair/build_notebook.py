#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook repair BT: dịch lại đúng các dòng "chưa dịch" của từng model.

Mỗi model tự repair dòng của mình (giữ nguyên bản sắc model), dùng prompt chặt hơn
và tự chọn giữa 2 biến thể prompt bằng probe 40 dòng.

Input : dataset ntquang0410/tiki-bt-repair-input (repair_input.parquet)
Output: /kaggle/working/repair_outputs/<model>/repair.parquet + run_state.json
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


md("""# Tiki Plan B — Repair các dòng chưa dịch (per-model)

Mỗi model Qwen3 tự dịch lại **đúng những dòng của chính nó** bị gate QC đánh dấu
chưa dịch (không có chữ Hán + còn dấu tiếng Việt, hoặc JSON hỏng/rỗng).

- Input: `tiki-bt-repair-input` (23.000 dòng: 0.6B 13.738 | 1.7B 8.717 | 4B 253 | 8B 292)
- Prompt mới: nhấn mạnh `source_text` là VALUE, cấm copy tiếng Việt, bắt buộc chữ Hán
- Tự chọn prompt bằng probe 40 dòng/model (strict vs strict+few-shot)
- Output: `repair_outputs/<model>/repair.parquet`
""")

code("""import json, os, re, time, glob
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

code("""INPUT_DIR = Path('/kaggle/input/tiki-bt-repair-input')
if not (INPUT_DIR / 'repair_input.parquet').exists():
    hits = glob.glob('/kaggle/input/**/repair_input.parquet', recursive=True)
    INPUT_DIR = Path(hits[0]).parent if hits else INPUT_DIR
print('INPUT_DIR', INPUT_DIR)

OUTPUT_ROOT = Path('/kaggle/working/repair_outputs')
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
MAX_INPUT_TOKENS = 1280
CHECKPOINT_EVERY = 200
PROBE_SIZE = 40
PROMPT_VERSION = 'plan_b_repair_v4'
TIME_BUDGET_MIN = 600
DEADLINE = time.time() + TIME_BUDGET_MIN * 60
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('time_budget_min', TIME_BUDGET_MIN)
""")

code("""SYSTEM_STRICT = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
The field "source_text" is the VALUE that must be translated.
"key_vi"/"key_zh" describe what kind of attribute it is: use them as context only,
never translate or copy the key into the output.
Rules:
- Output MUST be written in Simplified Chinese characters.
- Never copy Vietnamese words, except brand names and model codes.
- Translate even a single noun (e.g. "Hổ" -> "虎").
- Keep numbers, units, sizes, SKUs and Latin model codes unchanged.
Return JSON only: {"translation": "..."}'''

SYSTEM_FEWSHOT = SYSTEM_STRICT + '''

Examples:
{"source_text": "Hổ", "key_vi": "Họa tiết"} -> {"translation": "虎"}
{"source_text": "Xám", "key_vi": "Mầu Sắc"} -> {"translation": "灰色"}
{"source_text": "Số 4: 5L", "key_vi": "Kích cỡ"} -> {"translation": "4号：5升"}
{"source_text": "Z Fold 3", "key_vi": "Size"} -> {"translation": "Z Fold 3"}
{"source_text": "Đen viền xám", "key_vi": "Màu sắc"} -> {"translation": "黑边灰色"}'''

VARIANTS = {'strict': SYSTEM_STRICT, 'strict_fewshot': SYSTEM_FEWSHOT}
CJK_RE = re.compile(r'[\\u4e00-\\u9fff]')
VI_RE = re.compile(r'[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]', re.IGNORECASE)

def build_prompt(tokenizer, row, system):
    payload = {
        'source_text': str(row.get('source_text_bt') or ''),
        'key_vi': str(row.get('key_vi') or ''),
        'key_zh': str(row.get('key_zh_effective') or ''),
    }
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

def good_translation(source, translation):
    if not translation:
        return False
    if CJK_RE.search(translation):
        return True
    return not VI_RE.search(translation)
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

code("""import pandas as pd

repair = pd.read_parquet(INPUT_DIR / 'repair_input.parquet')
print('repair rows', len(repair))
print(repair.groupby('model_key').size().to_dict())
print(repair.head(3).to_string(max_colwidth=50))
""")

code("""def generate_rows(model, tokenizer, rows, system, batch_size, max_new=MAX_NEW_TOKENS):
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start + batch_size].to_dict('records')
        prompts = [build_prompt(tokenizer, row, system) for row in batch]
        enc = tokenizer(prompts, return_tensors='pt', padding=True,
                        truncation=True, max_length=MAX_INPUT_TOKENS).to(DEVICE)
        started = time.time()
        with torch.inference_mode():
            output = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    num_beams=1, eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)
        latency_ms = (time.time() - started) * 1000 / max(1, len(batch))
        generated = output[:, enc['input_ids'].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        n_new = [int((row != tokenizer.pad_token_id).sum()) for row in generated]
        for row, raw, ntokens in zip(batch, texts, n_new):
            value, ok = parse_translation(raw)
            records.append({
                'model_key': row['model_key'],
                'cache_key': str(row['cache_key']),
                'source_text_bt': row.get('source_text_bt'),
                'key_vi': row.get('key_vi'),
                'key_zh_effective': row.get('key_zh_effective'),
                'target_text_bt': value,
                'raw_output': raw,
                'json_ok': bool(ok),
                'good': bool(good_translation(row.get('source_text_bt'), value)),
                'n_new_tokens': int(ntokens),
                'prompt_variant': '',
                'repair_pass': 1,
                'latency_ms': round(latency_ms, 2),
            })
        if (start // batch_size) % 10 == 0 or start + len(batch) >= len(rows):
            print('  batch', start + len(batch), '/', len(rows), flush=True)
    return records
""")

code("""def probe_variants(model, tokenizer, rows, batch_size):
    # chọn 40 dòng trải đều theo độ dài source
    probe = rows.assign(_len=rows['source_text_bt'].fillna('').str.len())
    probe = probe.sort_values('_len')
    if len(probe) > PROBE_SIZE:
        idx = [int(i * (len(probe) - 1) / (PROBE_SIZE - 1)) for i in range(PROBE_SIZE)]
        probe = probe.iloc[idx]
    results = {}
    for name, system in VARIANTS.items():
        recs = generate_rows(model, tokenizer, probe, system, max(1, batch_size // 2), max_new=512)
        rate = sum(r['good'] for r in recs) / max(1, len(recs))
        results[name] = rate
        print(name, 'probe good rate', round(rate, 3))
    best = max(results, key=lambda k: (results[k], k == 'strict'))
    return best, results

def run_model(model_key):
    out_dir = OUTPUT_ROOT / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'repair.parquet'

    rows = repair[repair['model_key'] == model_key].reset_index(drop=True)
    if rows.empty:
        print('SKIP', model_key, 'không có dòng nào')
        return {'model_key': model_key, 'n_rows': 0}

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
    best, scores = probe_variants(model, tokenizer, rows, batch_size)
    system = VARIANTS[best]
    print(model_key, 'chosen variant', best, scores)

    print(model_key, 'repair rows', len(rows))
    frame = pd.DataFrame()
    stopped = 'completed'
    for start in range(0, len(rows), 1000):
        if time.time() >= DEADLINE:
            stopped = 'time_budget'
            print(model_key, 'dừng do time budget tại', start, '/', len(rows))
            break
        chunk = rows.iloc[start:start + 1000].reset_index(drop=True)
        records = generate_rows(model, tokenizer, chunk, system, batch_size)
        for rec in records:
            rec['prompt_variant'] = best
        frame = pd.concat([frame, pd.DataFrame(records)], ignore_index=True)
        frame.to_parquet(out_file, index=False)
        n_good_so_far = int(frame['good'].sum())
        print(model_key, 'saved', len(frame), '/', len(rows),
              'good', n_good_so_far, f'({n_good_so_far / len(frame) * 100:.1f}%)', flush=True)

    n_good = int(frame['good'].sum())
    state = {
        'model_key': model_key,
        'n_rows': int(len(frame)),
        'n_good': n_good,
        'good_rate': round(n_good / max(1, len(frame)), 4),
        'prompt_variant': best,
        'probe_scores': scores,
        'batch_size': batch_size,
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
Path('/kaggle/working/repair_summary.json').write_text(
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
target = ROOT / "plan_b_bt_repair.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", target, "cells", len(CELLS))
