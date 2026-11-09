#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh 2 kernel chạy seed 123 cho weight survey A/B."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


SETUP = """import glob, json, shutil, subprocess, sys
from pathlib import Path
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('no GPU')
p = torch.cuda.get_device_properties(0)
print('gpu', p.name, round(p.total_memory/1e9, 1), 'GB')
"""

PREP = """DATA = Path('/kaggle/input/tiki-weight-ab-data2')
if not (DATA / 'units_A.parquet').exists():
    hits = sorted(Path('/kaggle/input').rglob('units_A.parquet'))
    assert hits, 'không thấy units_A.parquet'
    DATA = hits[0].parent
ROOT = Path('/kaggle/working/ab_data'); ROOT.mkdir(parents=True, exist_ok=True)
for ds in ('A', 'B'):
    d = ROOT / f'processed_{ds}'; d.mkdir(exist_ok=True)
    shutil.copy2(DATA / f'units_{ds}.parquet', d / 'units.parquet')
    shutil.copy2(DATA / f'eval_{ds}.parquet', ROOT / f'eval_{ds}.parquet')
    shutil.copy2(DATA / f'meta_{ds}.json', ROOT / f'meta_{ds}.json')

CODE = Path('/kaggle/working/code'); CODE.mkdir(exist_ok=True)
trainer = sorted(Path('/kaggle/input').rglob('train_weight_quota_gpu.py'))[0]
for name in ('train_weight_quota_gpu.py', 'quota_sampling.py', 'tpu_weight_train.py'):
    shutil.copy2(trainer.parent / name, CODE / name)

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


def run_cell(dataset: str, configs: str) -> str:
    return f"""cmd = [sys.executable, str(CODE / 'train_weight_quota_gpu.py'),
       '--dataset', '{dataset}', '--data-root', str(ROOT),
       '--out', '/kaggle/working/seed123_{dataset}', '--model', MODEL,
       '--seed', '123', '--tokens', '4000000', '--steps', '500',
       '--configs', '{configs}']
print('RUN', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
"""


def build(suffix: str, dataset: str, configs: str, title: str) -> None:
    name = f"tiki-weight-seed123-{suffix}"
    cells = [
        md(f"# Weight survey {dataset} — seed 123\n\nConfig: `{configs}`.\n"),
        code(SETUP), code(PREP), code(run_cell(dataset, configs)),
        code(f"out=Path('/kaggle/working/seed123_{dataset}')\n"
             "for p in sorted(out.glob('hyp_*.parquet')): print(p.name, p.stat().st_size)\n"
             "print((out/'summary.json').read_text())\n"),
    ]
    folder = ROOT / "kaggle" / name
    folder.mkdir(parents=True, exist_ok=True)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    (folder / "seed123.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": f"anhtamxinhdep/{name}",
        "title": title,
        "code_file": "seed123.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["anhtamxinhdep/tiki-weight-ab-data2"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
        "machine_shape": "NvidiaRtxPro6000",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build("a", "A", "natural,attr_heavy", "Tiki Weight Seed123 A")
    build("b", "B", "natural,balanced,title_heavy", "Tiki Weight Seed123 B")
