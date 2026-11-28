#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh + push 3 kernel Kaggle cho campaign (NLLB resume, EcomGPT resume, 4B 1792)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "kaggle/campaign_1128"

COMMON = """import json, os, shutil, subprocess, sys
from pathlib import Path
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no GPU'
p = torch.cuda.get_device_properties(0)
print('gpu', p.name, round(p.total_memory/1e9, 1), 'GB')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
"""


def cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True),
            "outputs": [], "execution_count": None}


def notebook(cells: list[tuple[str, str]]) -> dict:
    return {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
                      if k == "md" else cell(s) for k, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


KERNELS = {}

# ---------------- NLLB B sub4842 resume ----------------
KERNELS["nllb"] = {
    "meta": {
        "id": "nguynthanhquang/tiki-nllb-b-sub4842-s42",
        "title": "Tiki Nllb B Sub4842 S42",
        "code_file": "nllb.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/nllb-b-sub4842-data",
                            "nguynthanhquang/nllb-b-sub4842-ckpt",
                            "nguynthanhquang/tiki-matrix-tools-v2"],
        "kernel_sources": [], "model_sources": ["amritha18/nllb-200-distilled-600m/Transformers/600m/1"],
        "competition_sources": [],
    },
    "cells": [
        ("md", "# NLLB B subsample 4,842M — seed 42 (resume)\n\nModel: NLLB-600M source Kaggle; sample B_sub4842; resume từ ckpt step đã upload."),
        ("code", COMMON),
        ("code", """DATA = None
for c in Path('/kaggle/input').rglob('units_B_sub4842.parquet'):
    DATA = c.parent; break
assert DATA, 'không thấy units_B_sub4842'
TOOLS = None
for c in Path('/kaggle/input').rglob('train_nllb.py'):
    TOOLS = c.parent; break
assert TOOLS, 'không thấy tools'
CKPT = None
for c in Path('/kaggle/input').rglob('adapter_config.json'):
    CKPT = c.parent; break
STEP = 0
if CKPT:
    f = CKPT/'STEP.txt'
    if f.exists(): STEP = int(f.read_text().strip() or 0)
MODEL = None
for cfg in Path('/kaggle/input').rglob('config.json'):
    try: d = json.loads(cfg.read_text())
    except Exception: continue
    if d.get('model_type') == 'm2m_100' or 'M2M100' in str(d.get('architectures')):
        MODEL = str(cfg.parent); break
assert MODEL, 'không thấy NLLB model'
print('DATA', DATA, '| TOOLS', TOOLS, '| CKPT', CKPT, 'step', STEP, '| MODEL', MODEL)
"""),
        ("code", """OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, str(TOOLS/'train_nllb.py'),
       '--units', str(DATA/'units_B_sub4842.parquet'),
       '--sample', str(DATA/'sample_B_sub4842_balanced_seed42.parquet'),
       '--eval', str(DATA/'eval_B.parquet'),
       '--model', MODEL, '--out', str(OUT), '--seed', '42',
       '--steps', '500', '--save-every', '25']
if STEP > 0:
    cmd += ['--init-adapter', str(CKPT), '--start-step', str(STEP)]
print('RUN', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
src = OUT/'hyp_seed42.parquet'
if src.exists():
    src.rename(OUT/'hyp_nllb_b_sub4842_seed42.parquet')
print('DONE')
"""),
    ],
}

# ---------------- EcomGPT B-natural retry resume ----------------
KERNELS["ecomgpt"] = {
    "meta": {
        "id": "nguynthanhquang/tiki-ecomgpt-b-natural-lr5e5-resume",
        "title": "Tiki Ecomgpt B Natural Lr5e5 Resume",
        "code_file": "ecom.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/ecomgpt-7b", "nguynthanhquang/tiki-matrix-data",
                            "nguynthanhquang/ecomgpt-b-natural-lr5e5-ckpt",
                            "nguynthanhquang/tiki-matrix-tools-v2"],
        "kernel_sources": [], "model_sources": [],
        "competition_sources": [], "machine_shape": "NvidiaRtxPro6000",
    },
    "cells": [
        ("md", "# EcomGPT-7B B-natural retry lr=5e-5 (resume từ step 125)\n\n"
               "Model: nguynthanhquang/ecomgpt-7b (mount sẵn, không tải)."),
        ("code", COMMON),
        ("code", """MODEL = '/kaggle/input/ecomgpt-7b'
if not (Path(MODEL)/'pytorch_model.bin.index.json').exists():
    hits = list(Path('/kaggle/input').rglob('pytorch_model.bin.index.json'))
    MODEL = str(hits[0].parent) if hits else MODEL
DATA_SRC = None
for c in Path('/kaggle/input').rglob('units_A.parquet'):
    DATA_SRC = c.parent; break
assert DATA_SRC, 'không thấy tiki-matrix-data'
ROOT = Path('/kaggle/working/matrix_data')
if ROOT.exists(): shutil.rmtree(ROOT)
(ROOT/'processed_A').mkdir(parents=True); (ROOT/'processed_B').mkdir(parents=True)
for n in ('A','B'):
    shutil.copy2(DATA_SRC/f'units_{n}.parquet', ROOT/f'processed_{n}/units.parquet')
    shutil.copy2(DATA_SRC/f'eval_{n}.parquet', ROOT/f'eval_{n}.parquet')
    shutil.copy2(DATA_SRC/f'meta_{n}.json', ROOT/f'meta_{n}.json')
TOOLS = None
for c in Path('/kaggle/input').rglob('train_ecomgpt.py'):
    TOOLS = c.parent; break
assert TOOLS, 'không thấy tools'
CKPT = None
for c in Path('/kaggle/input').rglob('adapter_config.json'):
    CKPT = c.parent; break
STEP = 0
if CKPT:
    f = CKPT/'STEP.txt'
    if f.exists(): STEP = int(f.read_text().strip() or 0)
print('MODEL', MODEL, '| TOOLS', TOOLS, '| CKPT', CKPT, 'step', STEP)
"""),
        ("code", """OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, str(TOOLS/'train_ecomgpt.py'),
       '--dataset', 'B', '--data-root', str(ROOT), '--out', str(OUT),
       '--model', MODEL, '--seed', '42', '--tokens', '4000000', '--steps', '500',
       '--configs', 'natural', '--micro-batch', '2', '--max-seq', '960',
       '--save-every', '25', '--lr', '5e-5']
if STEP > 0:
    cmd += ['--init-adapter', str(CKPT), '--start-step', str(STEP)]
print('RUN', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
src = OUT/'hyp_natural_seed42.parquet'
if src.exists():
    src.rename(OUT/'hyp_ecomgpt_b_natural_lr5e5_seed42.parquet')
print('DONE')
"""),
    ],
}

# ---------------- Qwen 4B A seed 2024 @1792 ----------------
KERNELS["qwen4b"] = {
    "meta": {
        "id": "nguynthanhquang/tiki-m4b-a-s2024-1792",
        "title": "Tiki M4b A S2024 1792",
        "code_file": "q4b.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["nguynthanhquang/tiki-matrix-data",
                            "nguynthanhquang/tiki-matrix-tools-v2"],
        "kernel_sources": [], "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
        "competition_sources": [], "machine_shape": "NvidiaL4",
    },
    "cells": [
        ("md", "# Qwen3-4B A-natural seed 2024 — max_seq 1792\n\n"
               "Mục tiêu: giữ 100% token (guard 960 làm mất 3,2% A)."),
        ("code", COMMON),
        ("code", """DATA = None
for c in Path('/kaggle/input').rglob('units_A.parquet'):
    DATA = c.parent; break
assert DATA, 'không thấy tiki-matrix-data'
ROOT = Path('/kaggle/working/matrix_data'); ROOT.mkdir(parents=True, exist_ok=True)
for n in ('A','B'):
    (ROOT/f'processed_{n}').mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA/f'units_{n}.parquet', ROOT/f'processed_{n}/units.parquet')
    shutil.copy2(DATA/f'eval_{n}.parquet', ROOT/f'eval_{n}.parquet')
    shutil.copy2(DATA/f'meta_{n}.json', ROOT/f'meta_{n}.json')
TOOLS = None
for c in Path('/kaggle/input').rglob('train_weight_quota_gpu.py'):
    try:
        if 'eval-max-seq' in c.read_text():
            TOOLS = c.parent; break
    except Exception: continue
assert TOOLS, 'không thấy trainer bản mới'
MODEL = None
for cfg in Path('/kaggle/input').rglob('config.json'):
    try: d = json.loads(cfg.read_text())
    except Exception: continue
    if d.get('hidden_size') == 2560 and d.get('num_hidden_layers') == 36:
        MODEL = str(cfg.parent); break
assert MODEL, 'không thấy Qwen3-4B'
print('DATA', DATA, '| TOOLS', TOOLS, '| MODEL', MODEL)
"""),
        ("code", """OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, str(TOOLS/'train_weight_quota_gpu.py'),
       '--dataset', 'A', '--data-root', str(ROOT), '--out', str(OUT),
       '--model', MODEL, '--seed', '2024', '--tokens', '4000000', '--steps', '500',
       '--configs', 'natural', '--micro-batch', '2', '--max-seq', '1792',
       '--eval-max-seq', '1792', '--dtype', 'bf16', '--save-every', '25']
print('RUN', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
src = OUT/'hyp_natural_seed2024.parquet'
if src.exists():
    src.rename(OUT/'hyp_qwen4b_a_natural_seed2024_1792.parquet')
print('DONE')
"""),
    ],
}


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    for name, spec in KERNELS.items():
        d = BASE / name
        d.mkdir(parents=True, exist_ok=True)
        (d / spec["meta"]["code_file"]).write_text(
            json.dumps(notebook(spec["cells"]), ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(
            json.dumps(spec["meta"], indent=2), encoding="utf-8")
        print("wrote", d)
    if "--push" in sys.argv:
        for name in sys.argv[sys.argv.index("--push") + 1:] or list(KERNELS):
            d = BASE / name
            r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                               capture_output=True, text=True)
            print(name, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
