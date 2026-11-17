#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLLB-200-600M B: chỉ chạy inference từ adapter đã train (kernel cũ bị cắt ở bước này).

Tác vụ nhẹ -> T4x2. Không train lại.
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


RUN = r'''
import json, os, subprocess, sys, time
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True)
import pandas as pd, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault('HF_HUB_OFFLINE', '1')
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      'gpus', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print('gpu', i, p.name, round(p.total_memory / 1e9, 1), 'GB')

DATA = Path('/kaggle/input/tiki-matrix-data')
if not (DATA / 'eval_B.parquet').exists():
    DATA = sorted(Path('/kaggle/input').rglob('eval_B.parquet'))[0].parent
ADAPTER = Path('/kaggle/input/nllb-b-adapter/adapter')
if not (ADAPTER / 'adapter_config.json').exists():
    ADAPTER = sorted(Path('/kaggle/input').rglob('adapter_config.json'))[0].parent
MODEL = None
for cfg in Path('/kaggle/input').rglob('config.json'):
    if cfg.parent == ADAPTER:
        continue
    try:
        d = json.loads(cfg.read_text(encoding='utf-8'))
    except Exception:
        continue
    if d.get('model_type') == 'm2m_100':
        MODEL = str(cfg.parent)
        break
assert MODEL, 'không thấy NLLB model source'
print('DATA', DATA, 'ADAPTER', ADAPTER, 'MODEL', MODEL)

tok = AutoTokenizer.from_pretrained(MODEL, src_lang='vie_Latn', tgt_lang='zho_Hans',
                                    local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tgt_id = tok.convert_tokens_to_ids('zho_Hans')
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             local_files_only=True, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')
print('loaded adapter', len(tok))

ev = pd.read_parquet(DATA / 'eval_B.parquet')
recs = []
t0 = time.time()
for max_new, part in ev.groupby('max_new'):
    part = part.reset_index(drop=True)
    bs = 1 if int(max_new) > 300 else 8
    for k in range(0, len(part), bs):
        chunk = part.iloc[k:k + bs]
        enc = tok([str(x) for x in chunk.source], return_tensors='pt', padding=True,
                  truncation=True, max_length=768, add_special_tokens=True)
        if torch.cuda.is_available():
            enc = enc.to('cuda')
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                 max_new_tokens=int(max_new), do_sample=False,
                                 num_beams=1, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        for eid, text in zip(chunk.eval_id, tok.batch_decode(gen, skip_special_tokens=True)):
            recs.append({'eval_id': eid, 'hypothesis': text.strip()})
        if k % 80 == 0:
            print('max_new', max_new, k, '/', len(part), 'min', round((time.time() - t0) / 60, 1),
                  flush=True)

hyp = ev[['eval_id', 'dataset', 'pair_type', 'source', 'reference']].merge(
    pd.DataFrame(recs), on='eval_id', how='left')
hyp['hypothesis'] = hyp.hypothesis.fillna('')
hyp['config'] = 'nllb_b'
hyp['seed'] = 42
out = Path('/kaggle/working/hyp_nllb_b_seed42.parquet')
hyp.to_parquet(out, index=False)
info = {'rows': int(len(hyp)), 'empty': int(hyp.hypothesis.str.strip().eq('').sum()),
        'minutes': round((time.time() - t0) / 60, 1)}
Path('/kaggle/working/run_nllb_b_infer.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
print(json.dumps(info, ensure_ascii=False, indent=2))
'''

CELLS = [md("# NLLB-600M B — inference-only từ adapter (train đã xong trước khi kernel cũ bị cắt)\n"),
         code(RUN)]
nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
folder = ROOT / "kaggle" / "nllb_b_infer"
folder.mkdir(parents=True, exist_ok=True)
(folder / "nllb_b_infer.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
meta = {"id": "nguynthanhquang/tiki-nllb-b-infer", "title": "Tiki Nllb B Infer",
        "code_file": "nllb_b_infer.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/tiki-matrix-data",
                            "nguynthanhquang/nllb-b-adapter",
                            "metric/gemma-4-developer-agent-wheelhouse"],
        "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [],
        "model_sources": ["amritha18/nllb-200-distilled-600m/Transformers/600m/1"],
        "machine_shape": "NvidiaTeslaT4"}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
print("wrote", folder)
