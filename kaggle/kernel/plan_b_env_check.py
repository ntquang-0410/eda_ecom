#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle environment check: dataset mount + TPU/GPU + parquet shapes.

Không load model, không gọi API.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd


def find_file(name: str) -> str | None:
    patterns = [
        f"/kaggle/input/**/{name}",
        f"data/cleaned/{name}",
        f"**/{name}",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def main() -> None:
    report: dict = {
        "env": {
            key: os.getenv(key)
            for key in (
                "KAGGLE_KERNEL_RUN_TYPE",
                "KAGGLE_URL_BASE",
                "TPU_NAME",
                "COLAB_TPU_ADDR",
                "PJRT_DEVICE",
            )
            if os.getenv(key) is not None
        },
        "files": {},
    }

    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as error:  # noqa: BLE001
        report["torch_error"] = str(error)

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        report["torch_xla"] = getattr(torch_xla, "__version__", "installed")
        report["xla_device"] = str(xm.xla_device())
    except Exception as error:  # noqa: BLE001
        report["torch_xla_error"] = str(error)

    for name in (
        "tiki_bt_input.parquet",
        "tiki_translation_cache.parquet",
        "tiki_bt_units_plan_a.parquet",
        "tiki_products.parquet",
        "tiki_products_bt_state.parquet",
    ):
        path = find_file(name)
        if not path:
            report["files"][name] = {"found": False}
            continue
        frame = pd.read_parquet(path)
        info = {"found": True, "path": path, "shape": list(frame.shape)}
        if name == "tiki_translation_cache.parquet":
            if "bt_route" in frame.columns:
                info["routes"] = {
                    str(k): int(v)
                    for k, v in frame["bt_route"].value_counts().items()
                }
        if name == "tiki_bt_input.parquet":
            info["columns"] = len(frame.columns)
            if "bt_route" in frame.columns:
                info["routes"] = {
                    str(k): int(v)
                    for k, v in frame["bt_route"].value_counts().items()
                }
        report["files"][name] = info
        print(name, info)

    out = Path("/kaggle/working/env_report.json") if Path("/kaggle/working").exists() else Path("env_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
