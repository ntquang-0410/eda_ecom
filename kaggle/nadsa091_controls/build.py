#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Qwen3-4B control kernels cho account nadsa091 (L4x4, offline).

Nguồn notebook: bản đã pull từ nadsa091/tiki-m4b-abal (micro-batch ladder,
load model theo hidden_size=2560/layers=36).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path("/tmp/nad_m4b/tiki-m4b-abal.ipynb")
OUT = ROOT / "kaggle" / "nadsa091"
DATASETS = ["metric/gemma-4-developer-agent-wheelhouse",
            "nadsa091/tiki-matrix-controls",
            "nadsa091/tiki-matrix-data"]
MODEL = "qwen-lm/qwen-3/Transformers/4b/1"


def build(name: str, title: str, dataset: str, config: str, seed: int, label: str,
          max_seq: int = 960) -> None:
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        text = text.replace("print('gpu', p.name, round(p.total_memory / 1e9, 1), 'GB')",
                            "print('gpu', p.name, round(p.total_memory / 1e9, 1), 'GB')\n"
                            "print('n_gpu', torch.cuda.device_count())")
        text = text.replace("DATASET = 'A'", f"DATASET = '{dataset}'")
        text = text.replace("CONFIG = 'balanced'", f"CONFIG = '{config}'")
        text = text.replace("MODEL_LABEL = 'qwen4b'", f"MODEL_LABEL = '{label}'")
        text = text.replace("OUT = Path('/kaggle/working/out_A_qwen4b_' + CONFIG)",
                            f"OUT = Path('/kaggle/working/out_{dataset}_{label}_' + CONFIG)")
        text = text.replace("'--seed', '42'", f"'--seed', '{seed}'")
        text = text.replace("'--max-seq', '640'", f"'--max-seq', '{max_seq}'")
        text = text.replace("OUT / f'hyp_{CONFIG}_seed42.parquet'",
                            f"OUT / f'hyp_{{CONFIG}}_seed{seed}.parquet'")
        text = text.replace("OUT / f'hyp_{MODEL_LABEL}_{DATASET.lower()}_{CONFIG}_seed42.parquet'",
                            f"OUT / f'hyp_{{MODEL_LABEL}}_{{DATASET.lower()}}_{{CONFIG}}_seed{seed}.parquet'")
        cell["source"] = text.splitlines(keepends=True)

    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    meta = {
        "id": f"nadsa091/{name}",
        "title": title,
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": DATASETS,
        "competition_sources": ["gemma-4-developer-agent"],
        "kernel_sources": [],
        "model_sources": [MODEL],
        "machine_shape": "NvidiaL4",
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build("tiki-m4b-abal", "Tiki M4b Abal", "A", "balanced", 42, "qwen4b")
    build("tiki-m4b-bnat", "Tiki M4b Bnat", "B", "natural", 42, "qwen4b")
    build("tiki-m4b-a-s123", "Tiki M4b A S123", "A", "natural", 123, "qwen4b")
    build("tiki-m4b-b-s123", "Tiki M4b B S123", "B", "balanced", 123, "qwen4b")
