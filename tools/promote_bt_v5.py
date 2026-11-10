#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuyển BT v5 thành bộ chính, archive v3 vào <model>/v3/.

Trước:
    data/cleaned/bt_tiki/<model>/{full,train,units}.parquet        (v3)
    data/cleaned/bt_tiki_v5/<model>/{full,train,units}.parquet     (v5)

Sau:
    data/cleaned/bt_tiki/<model>/{full,train,units}.parquet        (v5, bộ chính)
    data/cleaned/bt_tiki/<model>/v3/{full,train,units}.parquet     (v3, audit)

Manifest ghi cả 2 phiên bản + sha256; README nói rõ bộ nào là chính.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
KINDS = ("full", "train", "units")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path: Path) -> dict:
    return {
        "rows": int(pq.ParquetFile(path).metadata.num_rows),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    ap.add_argument("--v5-root", default="data/cleaned/bt_tiki_v5")
    args = ap.parse_args()

    root = Path(args.root)
    v5_root = Path(args.v5_root)

    manifest = {
        "dataset": "tiki-bt",
        "version": "2026-11-01",
        "main": "v5 (prompt brand-aware)",
        "archive": ["v3 (prompt plan_b_rtx_v3 + 2 vòng repair)"],
        "source": {
            "products": 13200,
            "units_total": 120884,
            "units_mt": 45709,
            "cache_keys_mt": 29446,
            "split_units": {"train": 36434, "dev": 4824, "test": 4451},
        },
        "models": {},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for model in MODELS:
        target_dir = root / model
        v3_dir = target_dir / "v3"
        v3_dir.mkdir(parents=True, exist_ok=True)

        for kind in KINDS:
            current = target_dir / f"{kind}.parquet"
            if current.exists():
                shutil.move(str(current), str(v3_dir / f"{kind}.parquet"))
            source = v5_root / model / f"{kind}.parquet"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, target_dir / f"{kind}.parquet")

        entry_model = {"version": "v5"}
        for kind in KINDS:
            entry_model[kind] = entry(target_dir / f"{kind}.parquet")
        entry_model["v3"] = {}
        for kind in KINDS:
            path = v3_dir / f"{kind}.parquet"
            if path.exists():
                entry_model["v3"][kind] = entry(path)
        manifest["models"][model] = entry_model

        print(f"{model}: main v5 full={entry_model['full']['rows']:,} "
              f"train={entry_model['train']['rows']:,} units={entry_model['units']['rows']:,} "
              f"| v3 full={entry_model['v3'].get('full', {}).get('rows', 0):,}")

    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    (root / "README.md").write_text(
        """---
language:
- vi
- zh
multilinguality:
- translation
task_categories:
- translation
tags:
- ecommerce
- tiki
- vietnamese
- chinese
- machine-translation
- qwen3
pretty_name: Tiki BT (v5 main, v3 archive)
---

# Tiki BT — v5 là bộ chính

Mỗi model Qwen3 tự sinh bản dịch cho cùng 29.446 cache MT của Tiki.
Bốn bộ **chỉ khác `target_text_bt`**.

## Phiên bản

| phiên bản | prompt | ghi chú |
|---|---|---|
| **v5 (chính)** | `plan_b_rtx_v5` brand-aware | gate loại `target == key_zh`; tốt nhất |
| v3 (archive) | `plan_b_rtx_v3` + 2 vòng repair | giữ để đối chiếu |

## Cấu trúc

```text
qwen3-0.6b/
  full.parquet        v5, 29.446 dòng (cache-level) + qc_status
  train.parquet       v5, chỉ dòng qc_status=ok
  units.parquet       v5, 45.709 dòng (unit-level, có product_id + split)
  v3/                 bản cũ: full/train/units
qwen3-1.7b/  qwen3-4b/  qwen3-8b/   (cùng cấu trúc)
reports/              báo cáo repair/audit
manifest.json         số dòng + sha256 từng file
```

## Chất lượng v5 (gate chặt)

| model | good | unit usable |
|---|---|---|
| 0.6B | 93,5% | 43.598 / 45.709 |
| 1.7B | 98,6% | 45.219 |
| 4B | 99,7% | 45.612 |
| 8B | 99,8% | 45.652 |

Lỗi `target == key_zh` từng chiếm 31% ở 0.6B (v3) đã giảm còn 4 dòng.

## Nguồn

13.200 sản phẩm Tiki → 120.884 unit; 45.709 unit route MT
(13.200 title + 32.509 attribute). Split theo `product_id`:
train 36.434 / dev 4.824 / test 4.451.

Bản dịch là **synthetic**, không phải gold.
""",
        encoding="utf-8")

    print("wrote", root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
