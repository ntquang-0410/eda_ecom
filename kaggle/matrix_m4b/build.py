#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build kernel Qwen3-4B cho matrix A/B (cùng recipe 4M token, 500 step)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "kaggle" / "tiki-matrix-m8b-b" / "tiki-matrix-m8b-b.ipynb"


def build(dataset: str, config: str, label: str) -> Path:
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        # Qwen3-4B: hidden 2560, 36 layer
        text = text.replace("HIDDEN = 4096", "HIDDEN = 2560")
        text = text.replace("d.get('num_hidden_layers') in (28, 36)",
                            "d.get('num_hidden_layers') in (36,)")
        if "DATASET = 'B'" in text:
            text = text.replace("DATASET = 'B'", f"DATASET = '{dataset}'")
            text = text.replace("CONFIG = 'balanced'", f"CONFIG = '{config}'")
            text = text.replace("MODEL_LABEL = 'm8b'", f"MODEL_LABEL = '{label}'")
            text = text.replace("OUT = Path('/kaggle/working/out_B_m8b_' + CONFIG)",
                                f"OUT = Path('/kaggle/working/out_{dataset}_{label}_' + CONFIG)")
            text = text.replace("'--micro-batch', str(2)", "'--micro-batch', str(4)")
        cell["source"] = text.splitlines(keepends=True)

    folder = ROOT / "kaggle" / f"tiki-matrix-{label}-{dataset.lower()}"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"tiki-matrix-{label}-{dataset.lower()}"
    (folder / f"{name}.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": f"ntquang0410/{name}",
        "title": f"Tiki Matrix {label} {dataset}",
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["ntquang0410/tiki-matrix-data"],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
        "machine_shape": "NvidiaRtxPro6000",
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)
    return folder


if __name__ == "__main__":
    build("A", "natural", "m4b")
    build("B", "balanced", "m4b")
