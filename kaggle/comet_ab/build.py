#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh kernel COMET offline cho survey A/B (mỗi suite một kernel).

Dùng lại notebook COMET offline đã có; chỉ lọc hypothesis theo suite
(`/A/` hoặc `/B/` trong đường dẫn) để score file không ghi đè nhau.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "comet_weight_offline" / "comet.ipynb"
HYP_DATASET = "nadsa091/tiki-weight-ab-hyps"


def build(suite: str) -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "import os, sys, subprocess" in text:
            cell["source"] = (f"SUITE = '{suite}'\n" + text).splitlines(keepends=True)
        if "rglob('hyp_*.parquet')" in text:
            text = text.replace(
                "hyp_files = sorted(Path('/kaggle/input').rglob('hyp_*.parquet'))",
                "hyp_files = sorted(p for p in Path('/kaggle/input').rglob('hyp_*.parquet')\n"
                "                   if f'/{SUITE}/' in str(p))")
            cell["source"] = text.splitlines(keepends=True)

    folder = ROOT / "kaggle" / f"comet_ab_{suite.lower()}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "comet.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    metadata = json.loads(
        (ROOT / "kaggle" / "comet_weight_offline" / "kernel-metadata.json").read_text())
    metadata["id"] = f"nadsa091/tiki-weight-comet-ab-{suite.lower()}"
    metadata["title"] = f"Tiki Weight COMET AB {suite}"
    metadata["dataset_sources"] = [
        "nadsa091/tiki-4model-eval",
        "nadsa091/tiki-offline-wheels",
        "nadsa091/tiki-comet-wheels",
        "nadsa091/comet-offline-shards",
        HYP_DATASET,
    ]
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", folder)


if __name__ == "__main__":
    build("A")
    build("B")
