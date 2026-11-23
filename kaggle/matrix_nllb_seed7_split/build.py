#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLLB seed 7 — tách A và B thành 2 kernel song song trên account 2."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "matrix_nllb_seed7" / "tiki-nllb-s7.ipynb"


def build(ds: str, name: str, title: str) -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    keep = f"train_one('{ds}'"
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        lines = text.splitlines(keepends=True)
        lines = [ln for ln in lines
                 if not (ln.startswith("train_one(") and keep not in ln)]
        text = "".join(lines)
        text = text.replace("/kaggle/input/tiki-matrix-data", "/kaggle/input/tiki-matrix-data")
        cell["source"] = text.splitlines(keepends=True)

    folder = ROOT / "kaggle" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": f"anhtamxinhdep/{name}",
        "title": title,
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "anhtamxinhdep/tiki-matrix-data",
            "anhtamxinhdep/tiki-matrix-controls-seed7",
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
    build("A", "tiki-nllb-a-s7", "Tiki Nllb A S7")
    build("B", "tiki-nllb-b-s7", "Tiki Nllb B S7")
