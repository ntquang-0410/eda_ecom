#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed 7 cho Qwen3-4B (A-natural, B-balanced) — Kaggle L4x4, offline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "kaggle" / "tiki-matrix-m4b-a" / "tiki-matrix-m4b-a.ipynb"


def build(name: str, dataset: str, config: str, seed: int, label: str) -> None:
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        text = text.replace("DATASET = 'A'", f"DATASET = '{dataset}'")
        text = text.replace("CONFIG = 'natural'", f"CONFIG = '{config}'")
        text = text.replace("MODEL_LABEL = 'm4b'", f"MODEL_LABEL = '{label}'")
        text = text.replace("OUT = Path('/kaggle/working/out_A_m4b_' + CONFIG)",
                            f"OUT = Path('/kaggle/working/out_{dataset}_{label}_' + CONFIG)")
        text = text.replace("'--seed', '42'", f"'--seed', '{seed}'")
        text = text.replace("'--configs', CONFIG, '--micro-batch', str(4)]",
                            "'--configs', CONFIG, '--micro-batch', str(1), '--max-seq', '960', '--dtype', 'fp16']")
        text = text.replace("import json, shutil, subprocess, sys",
                            "import json, shutil, subprocess, sys, os\nos.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'")
        text = text.replace("OUT / f'hyp_{CONFIG}_seed42.parquet'",
                            f"OUT / f'hyp_{{CONFIG}}_seed{seed}.parquet'")
        text = text.replace("OUT / f'hyp_{MODEL_LABEL}_{DATASET.lower()}_{CONFIG}_seed42.parquet'",
                            f"OUT / f'hyp_{{MODEL_LABEL}}_{{DATASET.lower()}}_{{CONFIG}}_seed{seed}.parquet'")
        cell["source"] = text.splitlines(keepends=True)
    folder = ROOT / "kaggle" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"id": f"nguynthanhquang/{name}", "title": f"Tiki Matrix {label} {dataset} s{seed}",
            "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["nguynthanhquang/tiki-matrix-data",
                                "nguynthanhquang/tiki-matrix-controls",
                                "metric/gemma-4-developer-agent-wheelhouse"],
            "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [],
            "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
            "machine_shape": "NvidiaL4"}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build("tiki-m4b-a-s7", "A", "natural", 7, "qwen4b")
    build("tiki-m4b-b-s7", "B", "balanced", 7, "qwen4b")
