#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh 2 notebook Kaggle để BT lại các dòng description bị empty target.

Kernel A: qwen3-0.6b + qwen3-1.7b
Kernel B: qwen3-4b + qwen3-8b

Input: dataset `anhtamxinhdep/tiki-bt-empty-tasks` với
  tasks_qwen3-<model>.parquet  (pair_id, product_id, split, source_text, tokens_est)

Output: /kaggle/working/bt_fix/<model>/predictions.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


SETUP = """import glob, json, re, time
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
"""

CONFIG = """MODELS = {models}
MODEL_SOURCES = {{
    'qwen3-0.6b': 'qwen-lm/qwen-3/Transformers/0.6b/1',
    'qwen3-1.7b': 'qwen-lm/qwen-3/Transformers/1.7b/1',
    'qwen3-4b':   'qwen-lm/qwen-3/Transformers/4b/1',
    'qwen3-8b':   'qwen-lm/qwen-3/Transformers/8b/1',
}}
EXPECTED_SHAPE = {{
    'qwen3-0.6b': (1024, 3072, 28),
    'qwen3-1.7b': (2048, 6144, 28),
    'qwen3-4b':   (2560, 9728, 36),
    'qwen3-8b':   (4096, 12288, 36),
}}
BATCH = {{'qwen3-0.6b': 24, 'qwen3-1.7b': 16, 'qwen3-4b': 8, 'qwen3-8b': 4}}
MAX_INPUT = 1536
MAX_NEW = 1536
CHECKPOINT = 100
TIME_BUDGET_MIN = 660
DEADLINE = time.time() + TIME_BUDGET_MIN * 60
PROMPT_VERSION = 'plan_b_desc_v1_retry'
OUTPUT_ROOT = Path('/kaggle/working/bt_fix')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print('deadline min', TIME_BUDGET_MIN, 'models', MODELS)
"""

PROMPT = """SYSTEM = '''You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
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

def _clean_candidate(text):
    s = str(text or '').strip()
    s = re.sub(r'^```(?:json)?', '', s).strip()
    s = re.sub(r'```$', '', s).strip()
    s = s.strip().strip('"').strip()
    s = s.replace('\\\\n', '\\n').replace('\\\\"', '"')
    # bỏ đuôi JSON bị cắt
    s = re.sub(r'["\\s}]{{1,4}}$', '', s).strip()
    return s

def parse_translation(text, src=None):
    raw = str(text or '').strip()
    start, end = raw.find('{{'), raw.rfind('}}')
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end+1])
            v = str(obj.get('translation') or '').strip()
            if v:
                return v, True, 'json'
        except Exception:
            pass
    # JSON bị cắt: lấy phần sau "translation":
    m = re.search(r'"translation"\\s*:\\s*"(.*)$', raw, re.S)
    cand = _clean_candidate(m.group(1) if m else raw)
    if not cand or not CJK_RE.search(cand):
        return '', False, 'empty'
    if src is not None and not is_good(src, cand):
        return '', False, 'bad'
    return cand, True, ('json_partial' if m else 'raw_text')

def build_messages(source):
    payload = {{'source_text': str(source)}}
    return [{{'role': 'system', 'content': SYSTEM}},
            {{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}}]

def build_prompt(tok, source):
    messages = build_messages(source)
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""

HELPERS = """def split_source(tok, source, max_tokens=900):
    text = str(source or '')
    parts = [p.strip() for p in re.split(r'(?<=[.。!?！？;；:\\n])', text) if p and p.strip()]
    if not parts:
        parts = [text]
    out, cur = [], ''
    for p in parts:
        candidate = (cur + ' ' + p).strip() if cur else p
        if len(tok(candidate, add_special_tokens=False).input_ids) <= max_tokens:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    # hard split nếu một mảnh vẫn quá dài
    final = []
    for p in out:
        ids = tok(p, add_special_tokens=False).input_ids
        if len(ids) <= max_tokens:
            final.append(p)
        else:
            step = max(1, len(p) // (len(ids) // max_tokens + 1))
            final.extend(p[i:i+step] for i in range(0, len(p), step))
    return [x for x in final if x.strip()]

def find_tasks(key):
    hits = sorted(Path('/kaggle/input').rglob(f'tasks_{{key}}.parquet'))
    if not hits:
        raise FileNotFoundError(f'tasks_{{key}}.parquet')
    df = pd.read_parquet(hits[0]).drop_duplicates('pair_id').reset_index(drop=True)
    df['source_text'] = df['source_text'].astype(str)
    print(key, 'tasks', len(df), hits[0])
    return df

def find_model(model_key):
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
"""

GEN = """def generate_segments(model, tok, segments, batch_size):
    results = {{}}
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start+batch_size]
        prompts = [build_prompt(tok, seg['text']) for seg in batch]
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
        for seg, raw, n, tr in zip(batch, texts, ntok, trunc):
            tgt, ok, method = parse_translation(raw, seg['text'])
            results[seg['id']] = {{'target': tgt, 'raw': raw, 'json_ok': ok,
                                 'method': method, 'n_new': int(n),
                                 'truncated': bool(tr), 'latency': lat}}
    return results
"""

RUN = """def run_model(key):
    out_dir = OUTPUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'predictions.parquet'
    df = find_tasks(key)
    model_path = find_model(key)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(key, 'loading...')
    model = load_model(model_path).to(DEVICE).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    print(key, 'loaded | alloc GB', round(torch.cuda.memory_allocated()/1e9, 1), flush=True)

    started = time.time()
    stopped = 'completed'
    frame = pd.DataFrame()
    for ckpt_start in range(0, len(df), CHECKPOINT):
        if time.time() >= DEADLINE:
            stopped = 'time_budget'; print(key, 'time budget at', ckpt_start); break
        chunk = df.iloc[ckpt_start:ckpt_start+CHECKPOINT].reset_index(drop=True)
        # tạo segment: split nếu prompt quá dài
        segments, row_parts = [], []
        for i, r in chunk.iterrows():
            prompt_ids = tok(build_prompt(tok, r['source_text']), add_special_tokens=False).input_ids
            if len(prompt_ids) <= MAX_INPUT - 64:
                segs = [r['source_text']]
            else:
                segs = split_source(tok, r['source_text'], max_tokens=900)
                if len(segs) <= 1:
                    segs = [r['source_text'][:6000]]
            for j, text in enumerate(segs):
                segments.append({{'id': f'{{i}}_{{j}}', 'text': text}})
            row_parts.append((i, len(segs)))
        results = generate_segments(model, tok, segments, BATCH[key])

        recs = []
        for i, n_parts in row_parts:
            r = chunk.iloc[i]
            parts = [results[f'{{i}}_{{j}}'] for j in range(n_parts)]
            cleaned = [p['target'] for p in parts if p['target']]
            target = '\\n'.join(cleaned)
            good = bool(target) and is_good(r['source_text'], target)
            recs.append({{
                'pair_id': r['pair_id'], 'product_id': r.get('product_id'),
                'split': r.get('split'), 'source_text': r['source_text'],
                'tokens_est': float(r.get('tokens_est') or 0),
                'target_text_bt': target,
                'raw_output': '\\n---\\n'.join(str(p['raw']) for p in parts),
                'json_ok': all(p['json_ok'] for p in parts),
                'good': good,
                'n_new_tokens': int(sum(p['n_new'] for p in parts)),
                'input_truncated': any(p['truncated'] for p in parts),
                'latency_ms': round(float(sum(p['latency'] for p in parts)), 2),
                'attempt': 2,
                'retry_method': ('split' if n_parts > 1 else parts[0]['method']),
                'n_segments': int(n_parts),
                'model_id': key, 'prompt_version': PROMPT_VERSION,
                'qc_status': 'ok' if good else 'needs_review',
            }})
        frame = pd.concat([frame, pd.DataFrame(recs)], ignore_index=True)
        frame.to_parquet(out_file, index=False)
        print(key, len(frame), '/', len(df), 'good', round(float(frame.good.mean()), 3), flush=True)

    state = {{'model': key, 'rows': len(frame),
             'good_rate': round(float(frame.good.mean()), 4) if len(frame) else 0,
             'stopped': stopped, 'minutes': round((time.time()-started)/60, 1)}}
    (out_dir / 'run_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(key, 'DONE', state, flush=True)
    del model; torch.cuda.empty_cache()
    return state
"""

MAIN = """summary = []
for key in MODELS:
    if time.time() >= DEADLINE:
        print('time budget, skip', key); break
    summary.append(run_model(key))
Path('/kaggle/working/bt_fix_summary.json').write_text(
    json.dumps({{'prompt_version': PROMPT_VERSION, 'runs': summary}}, ensure_ascii=False, indent=2),
    encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
"""


def _fix(text: str) -> str:
    return text.replace("{{", "{").replace("}}", "}")


def build(name: str, title: str, models: list[str]) -> None:
    cells = [
        md(f"# BT lại dòng empty target — {', '.join(models)}\n\n"
           "Prompt gốc `plan_b_desc_v1`, thêm fallback parse JSON bị cắt và split input dài.\n"),
        code(SETUP),
        code(CONFIG.format(models=json.dumps(models))),
        code(_fix(PROMPT)),
        code(_fix(HELPERS)),
        code(_fix(GEN)),
        code(_fix(RUN)),
        code(_fix(MAIN)),
    ]
    folder = ROOT / "kaggle" / name
    folder.mkdir(parents=True, exist_ok=True)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (folder / "bt_fix.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": f"anhtamxinhdep/{name}",
        "title": title,
        "code_file": "bt_fix.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["anhtamxinhdep/tiki-bt-empty-tasks"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": ["qwen-lm/qwen-3/Transformers/0.6b/1",
                          "qwen-lm/qwen-3/Transformers/1.7b/1",
                          "qwen-lm/qwen-3/Transformers/4b/1",
                          "qwen-lm/qwen-3/Transformers/8b/1"],
        "machine_shape": "NvidiaRtxPro6000",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build("tiki-bt-fix-small", "Tiki BT Fix Small", ["qwen3-0.6b", "qwen3-1.7b"])
    build("tiki-bt-fix-big", "Tiki BT Fix Big", ["qwen3-4b", "qwen3-8b"])
