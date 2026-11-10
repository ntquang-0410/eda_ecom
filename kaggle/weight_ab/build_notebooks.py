#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook Kaggle A/B weight survey (quota sampler)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def build(dataset: str, slug: str, title: str) -> None:
    cells = [
        md(f"# Weight survey {dataset} — quota sampler\n\n"
           "5 cấu hình, 4M token, 500 step, seed 42.\n"),
        code("import glob, json, subprocess, sys\n"
             "from pathlib import Path\n"
             "import torch\n"
             "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())\n"
             "if not torch.cuda.is_available(): raise SystemExit('no GPU')\n"
             "p=torch.cuda.get_device_properties(0)\n"
             "print('gpu', p.name, round(p.total_memory/1e9,1), 'GB')\n"
             "subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'], capture_output=True)\n"
             "wheels=sorted(Path('/kaggle/input').rglob('unbabel_comet-*.whl'))\n"
             "# không cần comet ở bước train; chỉ cần transformers/peft có sẵn\n"),
        code("DATA = Path('/kaggle/input/tiki-weight-ab-processed')\n"
             "if not (DATA/'eval_A.parquet').exists():\n"
             "    hits=sorted(glob.glob('/kaggle/input/**/eval_A.parquet', recursive=True))\n"
             "    assert hits, 'không thấy dataset A/B'\n"
             "    DATA=Path(hits[0]).parent\n"
             "CODE=sorted(Path('/kaggle/input').rglob('train_weight_quota_gpu.py'))\n"
             "assert CODE, 'không thấy code'\n"
             "\n"
             "# Dataset mount có thể lồng thư mục (khi upload dạng zip), nên dựng\n"
             "# một data-root phẳng trong /kaggle/working bằng symlink.\n"
             "ROOT = Path('/kaggle/working/ab_data'); ROOT.mkdir(parents=True, exist_ok=True)\n"
             "for ds in ('A','B'):\n"
             "    for name in (f'eval_{ds}.parquet', f'meta_{ds}.json'):\n"
             "        hits=sorted(Path('/kaggle/input').rglob(name))\n"
             "        assert hits, name\n"
             "        dst=ROOT/name\n"
             "        if not dst.exists(): dst.symlink_to(hits[0])\n"
             "    hits=sorted(Path('/kaggle/input').rglob(f'processed_{ds}/units.parquet'))\n"
             "    assert hits, f'processed_{ds}/units.parquet'\n"
             "    d=ROOT/f'processed_{ds}'; d.mkdir(exist_ok=True)\n"
             "    dst=d/'units.parquet'\n"
             "    if not dst.exists(): dst.symlink_to(hits[0])\n"
             "DATA=ROOT\n"
             "MODEL=None\n"
             "for cfg in Path('/kaggle/input').rglob('config.json'):\n"
             "    try: d=json.loads(cfg.read_text(encoding='utf-8'))\n"
             "    except Exception: continue\n"
             "    if (d.get('hidden_size'),d.get('intermediate_size'),d.get('num_hidden_layers'))==(2560,9728,36):\n"
             "        MODEL=str(cfg.parent); break\n"
             "print('DATA',DATA); print('CODE',CODE[0]); print('MODEL',MODEL)\n"
             "assert (DATA/'processed_A/units.parquet').exists()\n"
             "assert (DATA/'processed_B/units.parquet').exists()\n"
             "assert MODEL, 'không thấy Qwen3-4B'\n"),
        code(f"cmd=[sys.executable, str(CODE[0]), '--dataset', '{dataset}',\n"
             "     '--data-root', str(DATA), '--out', '/kaggle/working/weight_' + "
             f"'{dataset}', '--model', MODEL, '--seed','42', '--tokens','4000000', '--steps','500']\n"
             "print('RUN', ' '.join(cmd), flush=True)\n"
             "subprocess.run(cmd, check=True)\n"),
        code(f"out=Path('/kaggle/working/weight_{dataset}')\n"
             "for p in sorted(out.glob('hyp_*.parquet')): print(p.name,p.stat().st_size)\n"
             "for p in sorted(out.glob('adapter_*')): print(p.name)\n"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    folder = ROOT / "kaggle" / f"weight_ab_{dataset.lower()}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "weight_ab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata = {
        "id": f"nadsa091/{slug}",
        "title": title,
        "code_file": "weight_ab.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["nadsa091/tiki-weight-ab-processed",
                            "nadsa091/tiki-4model-eval",
                            "nadsa091/tiki-offline-wheels"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
        "machine_shape": "NvidiaRtxPro6000",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build("A", "tiki-weight-ab-a", "Tiki Weight AB A")
    build("B", "tiki-weight-ab-b", "Tiki Weight AB B")
