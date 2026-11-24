"""Sinh kernel resume cho Qwen3-4B A-natural seed 7 (sau khi retry2 bị cắt ở 12h).

Dùng: python kaggle/matrix_m4b_a_s7_resume/build.py
Notebook tìm adapter trong /kaggle/input (dataset qwen4b-a-s7-ckpt), đọc STEP.txt,
rồi train tiếp với --init-adapter/--start-step và chạy inference.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "tiki-m4b-a-s7-resume"

CELLS = [
    ("markdown", "# 4B A-natural seed 7 — resume từ checkpoint\n\n"
                 "Kernel retry2 bị cắt ở trần 12h; adapter định kỳ nằm trong dataset "
                 "`anhtamxinhdep/qwen4b-a-s7-ckpt`."),
    ("code", """import json, os, shutil, subprocess, sys
from pathlib import Path
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('no GPU')
p = torch.cuda.get_device_properties(0)
print('gpu', p.name, round(p.total_memory / 1e9, 1), 'GB')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], capture_output=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
"""),
    ("code", """DATA_SRC = None
for cand in sorted(Path('/kaggle/input').rglob('units_A.parquet')):
    DATA_SRC = cand.parent
    break
assert DATA_SRC, 'không thấy tiki-matrix-data'
ROOT = Path('/kaggle/working/matrix_data')
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)
for name in ('A', 'B'):
    (ROOT / f'processed_{name}').mkdir(parents=True)
    shutil.copy2(DATA_SRC / f'units_{name}.parquet', ROOT / f'processed_{name}/units.parquet')
    shutil.copy2(DATA_SRC / f'eval_{name}.parquet', ROOT / f'eval_{name}.parquet')
    shutil.copy2(DATA_SRC / f'meta_{name}.json', ROOT / f'meta_{name}.json')

CODE = Path('/kaggle/working/code'); CODE.mkdir(exist_ok=True)
SRC = None
for cand in Path('/kaggle/input').rglob('train_weight_quota_gpu.py'):
    try:
        if 'save-every' in cand.read_text(encoding='utf-8'):
            SRC = cand.parent; break
    except Exception:
        continue
assert SRC, 'không thấy trainer bản patched (tiki-matrix-tools)'
for name in ('train_weight_quota_gpu.py', 'quota_sampling.py', 'tpu_weight_train.py'):
    shutil.copy2(SRC / name, CODE / name)

HIDDEN = 2560
MODEL = None
for cfg in Path('/kaggle/input').rglob('config.json'):
    try:
        d = json.loads(cfg.read_text(encoding='utf-8'))
    except Exception:
        continue
    if d.get('hidden_size') == HIDDEN and d.get('num_hidden_layers') in (36,):
        MODEL = str(cfg.parent); break
assert MODEL, 'không thấy model 4B'

ADAPTER = None
for cand in sorted(Path('/kaggle/input').rglob('adapter_config.json')):
    ADAPTER = cand.parent; break
assert ADAPTER, 'không thấy adapter checkpoint'
STEP = 0
for cand in Path('/kaggle/input').rglob('STEP.txt'):
    try:
        STEP = int(cand.read_text(encoding='utf-8').strip() or 0)
        break
    except Exception:
        continue
assert STEP > 0, f'STEP.txt không hợp lệ: {STEP}'
print('DATA', DATA_SRC, '| MODEL', MODEL, '| CODE', SRC)
print('ADAPTER', ADAPTER, '| STEP', STEP)
"""),
    ("code", """DATASET = 'A'
CONFIG = 'natural'
MODEL_LABEL = 'qwen4b'
SEED = 7
OUT = Path('/kaggle/working/out_A_qwen4b_natural')
OUT.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, str(CODE / 'train_weight_quota_gpu.py'),
       '--dataset', DATASET, '--data-root', str(ROOT), '--out', str(OUT),
       '--model', MODEL, '--seed', str(SEED), '--tokens', '4000000', '--steps', '500',
       '--configs', CONFIG, '--micro-batch', '4', '--max-seq', '960',
       '--dtype', 'bf16', '--save-every', '50',
       '--init-adapter', str(ADAPTER), '--start-step', str(STEP)]
print('RUN', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)

src = OUT / f'hyp_{CONFIG}_seed{SEED}.parquet'
if src.exists():
    dst = OUT / f'hyp_{MODEL_LABEL}_{DATASET.lower()}_{CONFIG}_seed{SEED}.parquet'
    src.rename(dst)
    print('HYP', dst)
"""),
]

META = {
    "id": "anhtamxinhdep/tiki-m4b-a-s7-resume",
    "title": "Tiki M4b A S7 Resume",
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [
        "metric/gemma-4-developer-agent-wheelhouse",
        "anhtamxinhdep/tiki-matrix-data",
        "anhtamxinhdep/tiki-matrix-tools",
        "anhtamxinhdep/qwen4b-a-s7-ckpt",
    ],
    "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:fc720c15545eb096a510e1f085b17fc118387a007599424d953a4f327b71cacf",
    "machine_shape": "NvidiaL4",
}


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / META["code_file"]).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "kernel-metadata.json").write_text(json.dumps(META, indent=2), encoding="utf-8")
    print("wrote", HERE / META["code_file"])


if __name__ == "__main__":
    main()
