#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh 2 notebook Kaggle TPU v5e-8: seed 42 và seed 123."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def build(seed: int, slug: str, title: str) -> None:
    builder = (TOOLS / "build_weight_eval_sets.py").read_text(encoding="utf-8")
    trainer = (TOOLS / "tpu_weight_train.py").read_text(encoding="utf-8")

    cells = [
        markdown(
            f"# Weight experiment — Qwen3-4B trên TPU v5e-8 (seed {seed})\n\n"
            "5 cấu hình: natural, balanced, no_desc, tiki_heavy, title_heavy.\n\n"
            "- cùng 5M token, 625 optimizer step\n"
            "- cùng eval set 1.050 dòng (1688 title/attr/desc + Tiki title/attr)\n"
            "- chỉ khác phân bố token theo group\n"),
        code(
            "import subprocess, sys, json\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
            "                '-U', 'transformers>=4.51', 'peft', 'kagglehub'], check=False)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],\n"
            "               capture_output=True)\n"
            "import torch, torch_xla\n"
            "import torch_xla.core.xla_model as xm\n"
            "import torch_xla.runtime as xr\n"
            "print('torch', torch.__version__, '| xla', torch_xla.__version__)\n"
            "print('world_size', xr.world_size())\n"
            "print('device', xm.xla_device())\n"
        ),
        code(
            "from pathlib import Path\n"
            f"BUILDER = {json.dumps(builder, ensure_ascii=False)}\n"
            "Path('build_weight_eval_sets.py').write_text(BUILDER, encoding='utf-8')\n"
            f"TRAINER = {json.dumps(trainer, ensure_ascii=False)}\n"
            "Path('tpu_weight_train.py').write_text(TRAINER, encoding='utf-8')\n"
            "print('scripts written')\n"
        ),
        code(
            "import glob\n"
            "from pathlib import Path\n"
            "hits = glob.glob('/kaggle/input/**/1688.parquet', recursive=True)\n"
            "assert hits, 'không thấy 1688.parquet'\n"
            "pairs = Path(hits[0]).parent\n"
            "print('pairs dir', pairs)\n"
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, 'build_weight_eval_sets.py',\n"
            "                '--pairs', str(pairs),\n"
            "                '--out', '/kaggle/working/weight_eval',\n"
            "                '--max-seq', '1024'], check=True)\n"
        ),
        markdown("## Smoke 8 chip: 2 step, natural, không generate\n"),
        code(
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, 'tpu_weight_train.py',\n"
            "                '--eval-dir', '/kaggle/working/weight_eval',\n"
            "                '--out', '/kaggle/working/smoke_runs',\n"
            "                '--smoke'], check=True)\n"
            "print('SMOKE OK')\n"
        ),
        markdown(f"## Full run seed {seed}: 5 cấu hình × 5M token\n"),
        code(
            "import subprocess, sys, time\n"
            "started = time.time()\n"
            "subprocess.run([sys.executable, 'tpu_weight_train.py',\n"
            "                '--eval-dir', '/kaggle/working/weight_eval',\n"
            "                '--out', '/kaggle/working/weight_runs',\n"
            f"                '--seed', '{seed}'], check=True)\n"
            "print('DONE in', round((time.time()-started)/60, 1), 'min')\n"
        ),
        code(
            "from pathlib import Path\n"
            "for p in sorted(Path('/kaggle/working/weight_runs').glob('hyp_*.parquet')):\n"
            "    print(p.name, p.stat().st_size)\n"
            "print('--- adapters ---')\n"
            "for p in sorted(Path('/kaggle/working/weight_runs').glob('adapter_*')):\n"
            "    print(p.name)\n"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    folder = ROOT / "kaggle" / f"tpu_weight_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "weight_tpu.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata = {
        "id": f"ntquang0410/{slug}",
        "title": title,
        "code_file": "weight_tpu.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "dataset_sources": ["ntquang0410/tiki-pairs-dedup"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
        "machine_shape": "TpuV5E8",
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build(42, "tiki-weight-tpu-seed42", "Tiki Weight TPU Seed42")
    build(123, "tiki-weight-tpu-seed123", "Tiki Weight TPU Seed123")
