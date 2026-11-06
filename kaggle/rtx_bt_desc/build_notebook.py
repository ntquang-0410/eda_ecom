#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook BT description Tiki cho 4 model Qwen3 (offline, RTX)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle" / "rtx_bt_desc"
OUT.mkdir(parents=True, exist_ok=True)

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


md("""# Tiki description BT — 4 model Qwen3

Sinh target tiếng Trung cho 13.200 description prose, 4 model:
0.6B / 1.7B / 4B / 8B.

- internet OFF
- checkpoint mỗi 100 dòng
- output: `/kaggle/working/bt_desc_v1/<model>/predictions.parquet`
""")

code("""import glob, json, re, time
from pathlib import Path
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

print('torch', torch.__version__, '| transformers', transformers.__version__)
print('cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('không có GPU')
p = torch.cuda.get_device_properties(0)
print('gpu', p.name, round(p.total_memory/1e9, 1), 'GB')
DEVICE = 'cuda'
""")

code("""MODELS = ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
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
BATCH = {'qwen3-0.6b': 24, 'qwen3-1.7b': 16, 'qwen3-4b': 8, 'qwen3-8b': 4}
MAX_INPUT = 1536
MAX_NEW = 1536
CHECKPOINT = 100
TIME_BUDGET_MIN = 660
DEADLINE = time.time() + TIME_BUDGET_MIN * 60
PROMPT_VERSION = 'plan_b_desc_v1'
OUTPUT_ROOT = Path('/kaggle/working/bt_desc_v1')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print('deadline min', TIME_BUDGET_MIN)
""")

code("""SYSTEM = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
Translate the product description into fluent Simplified Chinese.
Rules:
- Translate all Vietnamese content; do not summarize or omit sentences.
- Keep numbers, units, sizes, brand names and Latin model codes unchanged.
- Keep the original order and meaning.
- The output MUST contain Chinese characters.
Return JSON only: {"translation": "..."}

If the text is already Chinese or has no Vietnamese, return it unchanged.'''

CJK_RE = re.compile(r'[\\u4e00-\\u9fff]')
VI_RE = re.compile(r'[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]', re.IGNORECASE)

def norm(s):
    return re.sub(r'\\s+', '', str(s or '')).lower()

def parse_translation(text):
    raw = (text or '').strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end+1])
            v = str(obj.get('translation') or '').strip()
            if v:
                return v, True
        except Exception:
            pass
    return '', False

def is_good(src, tgt):
    t = (tgt or '').strip()
    if not t:
        return False
    cjk = len(CJK_RE.findall(t))
    if cjk < max(2, int(0.08 * len(t))):
        return False
    ratio = len(t) / max(1, len(str(src)))
    if not (0.15 <= ratio <= 2.5):
        return False
    if norm(t) == norm(src):
        return False
    vi = len(VI_RE.findall(t))
    if vi > max(3, int(0.03 * len(t))):
        return False
    return True

def build_prompt(tok, source):
    payload = {'source_text': str(source)}
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
""")

code("""def find_input():
    hits = sorted(Path('/kaggle/input').rglob('description.parquet'))
    for p in hits:
        try:
            d = pd.read_parquet(p)
            if 'pair_id' in d.columns and 'source_text' in d.columns and len(d) > 10000:
                print('input', p, d.shape)
                return d
        except Exception:
            continue
    raise FileNotFoundError('không thấy description.parquet')

df = find_input().copy()
df = df[df['pair_type'] == 'description'].copy() if 'pair_type' in df.columns else df
df = df.drop_duplicates('pair_id').reset_index(drop=True)
df['source_text'] = df['source_text'].astype(str)
df['tokens_est'] = pd.to_numeric(df.get('tokens_est'), errors='coerce').fillna(0)
df = df.sort_values('tokens_est').reset_index(drop=True)
print('rows', len(df), 'tokens', int(df.tokens_est.sum()))
print(df[['pair_id','source_text']].head(2).to_string()[:600])
""")

code("""def find_model(model_key):
    try:
        import kagglehub
        path = kagglehub.model_download(MODEL_SOURCES[model_key])
        print(model_key, 'kagglehub', path)
        return path
    except Exception as e:
        print(model_key, 'kagglehub failed', type(e).__name__, str(e)[:120])
    shape = EXPECTED_SHAPE[model_key]
    for cfg in Path('/kaggle/input').rglob('config.json'):
        try:
            d = json.loads(cfg.read_text(encoding='utf-8'))
        except Exception:
            continue
        if (d.get('hidden_size'), d.get('intermediate_size'), d.get('num_hidden_layers')) == shape:
            print(model_key, 'fallback', cfg.parent)
            return str(cfg.parent)
    raise FileNotFoundError(model_key)

def load_model(path):
    kw = dict(trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)
""")

code("""def generate_batch(model, tok, rows, batch_size):
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start+batch_size].to_dict('records')
        prompts = [build_prompt(tok, r['source_text']) for r in batch]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                  max_length=MAX_INPUT).to(DEVICE)
        trunc = (enc['attention_mask'].sum(dim=1) >= MAX_INPUT).tolist()
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 num_beams=1, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        lat = (time.time()-t0)*1000/max(1, len(batch))
        texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        ntok = [int((x != tok.pad_token_id).sum()) for x in gen[:, enc['input_ids'].shape[1]:]]
        for r, raw, n, tr in zip(batch, texts, ntok, trunc):
            tgt, ok = parse_translation(raw)
            out.append({
                'pair_id': r['pair_id'], 'product_id': r.get('product_id'),
                'split': r.get('split'), 'source_text': r['source_text'],
                'tokens_est': float(r.get('tokens_est') or 0),
                'target_text_bt': tgt, 'raw_output': raw,
                'json_ok': bool(ok), 'good': bool(is_good(r['source_text'], tgt)),
                'n_new_tokens': int(n), 'input_truncated': bool(tr),
                'latency_ms': round(lat, 2),
            })
    return out
""")

code("""def run_model(key):
    out_dir = OUTPUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'predictions.parquet'
    model_path = find_model(key)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(key, 'loading...')
    model = load_model(model_path).to(DEVICE).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    print(key, 'loaded | alloc GB', round(torch.cuda.memory_allocated()/1e9, 1))
    frame = pd.DataFrame()
    started = time.time()
    stopped = 'completed'
    for start in range(0, len(df), CHECKPOINT):
        if time.time() >= DEADLINE:
            stopped = 'time_budget'; print(key, 'time budget at', start); break
        chunk = df.iloc[start:start+CHECKPOINT].reset_index(drop=True)
        recs = generate_batch(model, tok, chunk, BATCH[key])
        for r in recs:
            r['model_id'] = key; r['prompt_version'] = PROMPT_VERSION
            r['attempt'] = 1; r['qc_status'] = 'ok' if r['good'] else 'needs_review'
        frame = pd.concat([frame, pd.DataFrame(recs)], ignore_index=True)
        frame.to_parquet(out_file, index=False)
        print(key, len(frame), '/', len(df), 'good', round(float(frame.good.mean()), 3), flush=True)
    state = {'model': key, 'rows': len(frame), 'good_rate': round(float(frame.good.mean()), 4) if len(frame) else 0,
             'stopped': stopped, 'minutes': round((time.time()-started)/60, 1)}
    (out_dir / 'run_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(key, 'DONE', state)
    del model; torch.cuda.empty_cache()
    return state
""")

code("""summary = []
for key in MODELS:
    if time.time() >= DEADLINE:
        print('time budget, skip', key); break
    summary.append(run_model(key))
Path('/kaggle/working/bt_desc_summary.json').write_text(
    json.dumps({'prompt_version': PROMPT_VERSION, 'runs': summary}, ensure_ascii=False, indent=2),
    encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
""")

notebook = {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
(OUT / "desc_bt.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

metadata = {
    "id": "nadsa091/tiki-description-bt",
    "title": "Tiki Description BT 4 Models",
    "code_file": "desc_bt.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": ["nadsa091/tiki-description-input",
                        "nadsa091/tiki-4model-eval",
                        "nadsa091/tiki-offline-wheels"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
    "model_sources": ["qwen-lm/qwen-3/Transformers/0.6b/1",
                      "qwen-lm/qwen-3/Transformers/1.7b/1",
                      "qwen-lm/qwen-3/Transformers/4b/1",
                      "qwen-lm/qwen-3/Transformers/8b/1"],
    "machine_shape": "NvidiaRtxPro6000",
}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
