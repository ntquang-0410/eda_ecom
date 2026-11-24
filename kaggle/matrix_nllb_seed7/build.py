#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLLB-200-distilled-600M seed 7 (A-natural, B-balanced) — Kaggle L4x4."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "matrix_nllb" / "nllb.ipynb"


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        text = text.replace("/kaggle/input/tiki-matrix-controls'", "/kaggle/input/tiki-matrix-controls-seed7'")
        text = text.replace("sample_A_natural_seed42.parquet", "sample_A_natural_seed7.parquet")
        text = text.replace("sample_B_balanced_seed42.parquet", "sample_B_balanced_seed7.parquet")
        text = text.replace("hyp_{cfg}_seed42.parquet", "hyp_{cfg}_seed7.parquet")
        text = text.replace("run_{cfg}_seed42.json", "run_{cfg}_seed7.json")
        text = text.replace("hyp['seed'] = 42", "hyp['seed'] = 7")
        cell["source"] = text.splitlines(keepends=True)

    folder = ROOT / "kaggle" / "matrix_nllb_seed7"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tiki-nllb-s7.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": "nguynthanhquang/tiki-nllb-s7",
        "title": "Tiki Nllb S7",
        "code_file": "tiki-nllb-s7.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "nguynthanhquang/tiki-matrix-data",
            "nguynthanhquang/tiki-matrix-controls-seed7",
            "metric/gemma-4-developer-agent-wheelhouse",
        ],
        "competition_sources": ["gemma-4-developer-agent"],
        "kernel_sources": [],
        "model_sources": ["amritha18/nllb-200-distilled-600m/Transformers/600m/1"],
        "machine_shape": "NvidiaL4",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    main()
