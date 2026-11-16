#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ghi dataset/content hash vào run JSON + manifest cho 10 cell matrix."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".artifacts/matrix_out/m4b_a/matrix_data"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).casefold()).strip()


def content_hash(values) -> str:
    h = hashlib.sha256()
    for v in sorted(values):
        h.update((v + "\n").encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    ds_hash = {ds: sha_file(BASE / f"processed_{ds}/units.parquet") for ds in ("A", "B")}
    eval_hash = {"v1": sha_file(BASE / "eval_A.parquet")}
    for name in ("eval_F_v2", "eval_F_v2_common"):
        p = ROOT / "data/cleaned" / f"{name}.parquet"
        eval_hash[name] = sha_file(p) if p.exists() else None
    ev = pd.read_parquet(BASE / "eval_A.parquet")
    eval_content = content_hash(norm(x) for x in ev.source_text)
    samples = {
        "A": ROOT / ".artifacts/matrix_out/m4b_a/out_A_m4b_natural/sample_natural_seed42.parquet",
        "B": ROOT / ".artifacts/matrix_out/m4b_b/out_B_m4b_balanced/sample_balanced_seed42.parquet",
    }
    sample_meta = {}
    for ds, p in samples.items():
        s = pd.read_parquet(p)
        sample_meta[ds] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": sha_file(p),
            "pair_content_hash": content_hash(s.pair_id.astype(str)),
            "draws": int(len(s)),
            "unique": int(s.row_index.nunique()),
        }

    cells = []
    for lab in ("qwen_a", "qwen_b", "qwen4b_a", "qwen4b_b", "ecomgpt_a", "ecomgpt_b",
                "moses_a", "moses_b", "onmt_a", "onmt_b"):
        ds = lab.rsplit("_", 1)[-1].upper()
        cells.append({"cell": lab, "dataset": ds, "dataset_hash": ds_hash[ds],
                      "sample_hash": sample_meta[ds]["sha256"],
                      "eval_v1_hash": eval_hash["v1"],
                      "eval_v2_hash": eval_hash["eval_F_v2"],
                      "eval_v2_common_hash": eval_hash["eval_F_v2_common"],
                      "eval_content_hash": eval_content})

    audit = json.loads((ROOT / "review_audit.json").read_text(encoding="utf-8"))
    exposure = {ds: audit["quota"][ds]["groups"] for ds in ("A", "B")}
    manifest = {"dataset_hashes": ds_hash, "eval_hashes": eval_hash,
                "eval_content_hash_v1": eval_content, "samples": sample_meta,
                "exposure": exposure, "cells": cells}
    out = ROOT / "reports/RUN_HASH_MANIFEST.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(cells).to_csv(ROOT / "reports/RUN_HASH_MANIFEST.csv", index=False)

    patched = 0
    for p in (ROOT / ".artifacts/matrix_out").rglob("run_*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            continue
        ds = "A" if "/out_A" in str(p) or "m4b_a" in str(p) else "B"
        d.update({"dataset_hash": ds_hash.get(ds), "eval_hash": eval_hash["v1"],
                  "eval_v2_hash": eval_hash["eval_F_v2"],
                  "sample_hash": sample_meta[ds]["sha256"],
                  "content_hash": eval_content,
                  "exposure": exposure.get(ds)})
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        patched += 1
    print("manifest", out, "patched run json", patched)
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
