#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel train final 4B theo recipe A+B (attribute natural, title/desc balanced)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


SETUP = """import json, shutil, subprocess, sys
from pathlib import Path
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('no GPU')
p = torch.cuda.get_device_properties(0)
print('gpu', p.name, round(p.total_memory/1e9, 1), 'GB')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True)
"""

PREP = """DATA = Path('/kaggle/input/tiki-weight-final-data')
if not (DATA / 'units_F.parquet').exists():
    hits = sorted(Path('/kaggle/input').rglob('units_F.parquet'))
    assert hits, 'không thấy units_F.parquet'
    DATA = hits[0].parent
ROOT = Path('/kaggle/working/f_data')
(ROOT / 'processed_F').mkdir(parents=True, exist_ok=True)
shutil.copy2(DATA / 'units_F.parquet', ROOT / 'processed_F/units.parquet')
shutil.copy2(DATA / 'eval_F.parquet', ROOT / 'eval_F.parquet')
shutil.copy2(DATA / 'meta_F.json', ROOT / 'meta_F.json')

CODE = Path('/kaggle/working/code'); CODE.mkdir(exist_ok=True)
trainer = sorted(Path('/kaggle/input').rglob('train_weight_quota_gpu.py'))[0]
for name in ('train_weight_quota_gpu.py', 'quota_sampling.py', 'tpu_weight_train.py'):
    shutil.copy2(trainer.parent / name, CODE / name)
tr = CODE / 'train_weight_quota_gpu.py'
tr.write_text(tr.read_text().replace('choices=("A", "B")', 'choices=("A", "B", "C", "F")'), encoding='utf-8')

MODEL = None
for cfg in Path('/kaggle/input').rglob('config.json'):
    try:
        d = json.loads(cfg.read_text(encoding='utf-8'))
    except Exception:
        continue
    if (d.get('hidden_size'), d.get('intermediate_size'), d.get('num_hidden_layers')) == (2560, 9728, 36):
        MODEL = str(cfg.parent); break
assert MODEL, 'không thấy Qwen3-4B'
print('DATA', DATA, '| MODEL', MODEL)
"""

RUN = """cmd = [sys.executable, str(CODE / 'train_weight_quota_gpu.py'),
       '--dataset', 'F', '--data-root', str(ROOT),
       '--out', '/kaggle/working/final_4b', '--model', MODEL,
       '--seed', '42', '--tokens', '8000000', '--steps', '1000',
       '--configs', 'final']
print('RUN', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
"""

CELLS = [
    md("# Final 4B — recipe A+B\n\n"
       "attribute natural; title/description cân bằng theo B. 8M token / 1.000 step.\n"),
    code(SETUP), code(PREP), code(RUN),
    code("""out=Path('/kaggle/working/final_4b')
for p in sorted(out.glob('hyp_*.parquet')): print(p.name, p.stat().st_size)
print((out/'summary.json').read_text())
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
folder = ROOT / "kaggle" / "tiki-weight-final"
folder.mkdir(parents=True, exist_ok=True)
(folder / "final.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {
    "id": "anhtamxinhdep/tiki-weight-final",
    "title": "Tiki Weight Final",
    "code_file": "final.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": ["anhtamxinhdep/tiki-weight-final-data"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
    "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
    "machine_shape": "NvidiaRtxPro6000",
}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
print("wrote", folder)
