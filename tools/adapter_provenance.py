#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ghi adapter provenance vào manifest canonical (3.4).

Provenance gồm: nguồn (kernel output / dataset), step, md5 file adapter, base model,
và cờ 'recorded' (False nếu run cũ không lưu adapter).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"

# (cell, seed) -> provenance
KNOWN = {
    ("qwen4b_a", 42): {"source": "kernel_output", "dataset_ref": "nguynthanhquang/tiki-matrix-m4b-a",
                        "adapter_name": "adapter_natural_seed42", "step": 500,
                        "local_file": "/tmp/k42/out_A_m4b/adapter_natural_seed42/adapter_model.safetensors",
                        "base_model": "Qwen/Qwen3-4B"},
    ("qwen4b_a", 123): {"source": "kernel_output", "dataset_ref": "anhtamxinhdep/tiki-m4b-a-s123",
                         "adapter_name": "adapter_natural_seed123", "step": 500,
                         "local_file": "/tmp/k123/out_A_qwen4b_natural/adapter_natural_seed123/adapter_model.safetensors",
                         "base_model": "Qwen/Qwen3-4B"},
    ("qwen4b_a", 7): {"source": "dataset", "dataset_ref": "nguynthanhquang/qwen4b-a-s7-ckpt",
                       "adapter_name": "adapter_natural_seed7_ckpt", "step": 500,
                       "local_file": "/tmp/s7ad/adapter_model.safetensors",
                       "base_model": "Qwen/Qwen3-4B"},
    ("qwen8b_b_natural", 42): {"source": "dataset", "dataset_ref": "nguynthanhquang/qwen8b-b-ckpt",
                                "adapter_name": "adapter", "step": 500, "local_file": None,
                                "base_model": "Qwen/Qwen3-8B"},
    ("ecomgpt_a_balanced", 42): {"source": "dataset", "dataset_ref": "nguynthanhquang/ecomgpt-ckpt",
                                  "adapter_name": "adapter_balanced_seed42", "step": 500, "local_file": None,
                                  "base_model": "damo/nlp_ecomgpt_multilingual-7B-ecom"},
    ("qwen4b_b", 123): {"source": "kernel_output", "dataset_ref": "anhtamxinhdep/tiki-m4b-b-s123",
                         "adapter_name": "adapter_balanced_seed123", "step": 500, "local_file": None,
                         "base_model": "Qwen/Qwen3-4B"},
    ("nllb_a", 7): {"source": "dataset", "dataset_ref": "nguynthanhquang/tiki-matrix-checkpoints",
                     "adapter_name": "adapters/nllb_a_s7_adapter", "step": 500, "local_file": None,
                     "base_model": "facebook/nllb-200-distilled-600M"},
    ("nllb_b", 7): {"source": "dataset", "dataset_ref": "nguynthanhquang/tiki-matrix-checkpoints",
                     "adapter_name": "adapters/nllb_b_s7_adapter", "step": 500, "local_file": None,
                     "base_model": "facebook/nllb-200-distilled-600M"},
}


def md5(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    n = 0
    for mp in CANON.glob("*/seed*.manifest.json"):
        cell = mp.parent.name
        seed = int(mp.name.replace("seed", "").split(".")[0])
        man = json.loads(mp.read_text(encoding="utf-8"))
        prov = KNOWN.get((cell, seed), {
            "source": "unknown", "dataset_ref": None, "adapter_name": None,
            "step": None, "local_file": None, "base_model": man.get("base_model"),
        })
        prov = dict(prov)
        prov["md5"] = md5(prov.pop("local_file"))
        prov["recorded"] = prov["source"] != "unknown"
        man["adapter_provenance"] = prov
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
    print(f"cập nhật {n} manifest")


if __name__ == "__main__":
    main()
