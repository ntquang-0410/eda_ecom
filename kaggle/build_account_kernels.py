#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Qwen3-4B kernels cho một account bất kỳ (L4x4, offline).

Dùng notebook template đã pull từ nadsa091/tiki-m4b-abal.
Cách dùng: python build_account.py --owner anhtamxinhdep --jobs a-s123,b-s123
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path("/tmp/nad_m4b/tiki-m4b-abal.ipynb")
MODEL = "qwen-lm/qwen-3/Transformers/4b/1"

JOBS = {
    "abal": dict(name="tiki-m4b-abal", title="Tiki M4b Abal", dataset="A",
                 config="balanced", seed=42, label="qwen4b"),
    "bnat": dict(name="tiki-m4b-bnat", title="Tiki M4b Bnat", dataset="B",
                 config="natural", seed=42, label="qwen4b"),
    "a-s123": dict(name="tiki-m4b-a-s123", title="Tiki M4b A S123", dataset="A",
                   config="natural", seed=123, label="qwen4b"),
    "b-s123": dict(name="tiki-m4b-b-s123", title="Tiki M4b B S123", dataset="B",
                   config="balanced", seed=123, label="qwen4b"),
}


def build(owner: str, job: dict, max_seq: int = 960) -> Path:
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        text = text.replace("""for name in ('train_weight_quota_gpu.py', 'quota_sampling.py', 'tpu_weight_train.py'):
    shutil.copy2(DATA_SRC / name, CODE / name)""", """for name in ('train_weight_quota_gpu.py', 'quota_sampling.py', 'tpu_weight_train.py'):
    src = DATA_SRC / name
    if src.exists():
        shutil.copy2(src, CODE / name)""")
        text = text.replace("print('gpu', p.name, round(p.total_memory / 1e9, 1), 'GB')",
                            "print('gpu', p.name, round(p.total_memory / 1e9, 1), 'GB')\n"
                            "print('n_gpu', torch.cuda.device_count())")
        text = text.replace("DATASET = 'A'", f"DATASET = '{job['dataset']}'")
        text = text.replace("CONFIG = 'balanced'", f"CONFIG = '{job['config']}'")
        text = text.replace("MODEL_LABEL = 'qwen4b'", f"MODEL_LABEL = '{job['label']}'")
        text = text.replace("OUT = Path('/kaggle/working/out_A_qwen4b_' + CONFIG)",
                            f"OUT = Path('/kaggle/working/out_{job['dataset']}_{job['label']}_' + CONFIG)")
        text = text.replace("'--seed', '42'", f"'--seed', '{job['seed']}'")
        text = text.replace("'--max-seq', '640'", f"'--max-seq', '{max_seq}'")
        text = text.replace("OUT / f'hyp_{CONFIG}_seed42.parquet'",
                            f"OUT / f'hyp_{{CONFIG}}_seed{job['seed']}.parquet'")
        text = text.replace("OUT / f'hyp_{MODEL_LABEL}_{DATASET.lower()}_{CONFIG}_seed42.parquet'",
                            f"OUT / f'hyp_{{MODEL_LABEL}}_{{DATASET.lower()}}_{{CONFIG}}_seed{job['seed']}.parquet'")
        cell["source"] = text.splitlines(keepends=True)

    folder = ROOT / "kaggle" / owner / job["name"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{job['name']}.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": f"{owner}/{job['name']}",
        "title": job["title"],
        "code_file": f"{job['name']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse",
                            f"{owner}/tiki-matrix-controls",
                            f"{owner}/tiki-matrix-data"],
        "competition_sources": ["gemma-4-developer-agent"],
        "kernel_sources": [],
        "model_sources": [MODEL],
        "machine_shape": "NvidiaL4",
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--jobs", default="abal,bnat,a-s123,b-s123")
    ap.add_argument("--max-seq", type=int, default=960)
    args = ap.parse_args()
    for key in [x.strip() for x in args.jobs.split(",") if x.strip()]:
        build(args.owner, JOBS[key], max_seq=args.max_seq)
