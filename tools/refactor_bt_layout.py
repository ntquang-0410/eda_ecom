#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refactor data/cleaned/bt_tiki thành layout theo model + upload HF.

Trước:
    data/cleaned/bt_tiki/tiki_bt_qwen3-0.6b_full.parquet
    data/cleaned/bt_tiki/tiki_bt_qwen3-0.6b_train.parquet
    data/cleaned/bt_tiki/tiki_bt_qwen3-0.6b_units.parquet
    data/cleaned/bt_tiki/bt_*.json

Sau:
    data/cleaned/bt_tiki/
      README.md            dataset card
      manifest.json        rows + sha256 + schema
      reports/             bt_repair_report.json, bt_units_report.json, bt_audit.json
      qwen3-0.6b/          full.parquet | train.parquet | units.parquet
      qwen3-1.7b/
      qwen3-4b/
      qwen3-8b/

Upload (khi có token):
    python tools/refactor_bt_layout.py --upload \
        --repo-id ntquang0410/zh-vie_ecom --prefix data/cleaned/tiki/bt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
KINDS = ("full", "train", "units")
REPORT_FILES = ("bt_repair_report.json", "bt_units_report.json", "bt_audit.json")

README = """---
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
pretty_name: Tiki BT (Qwen3 0.6B / 1.7B / 4B / 8B)
---

# Tiki BT — bản dịch máy Việt → Trung theo từng model

Mỗi model Qwen3 tự sinh bản dịch cho **cùng một tập unit** của Tiki, sau đó được
repair 2 vòng. Bốn bộ **chỉ khác cột `target_text_bt`**, mọi thứ khác dùng chung.

## Cấu trúc

```text
qwen3-0.6b/
  full.parquet    29.446 dòng (cache-level) + audit qc_status
  train.parquet   chỉ dòng dùng được để fine-tune
  units.parquet   45.709 dòng (unit-level, có product_id + split)
qwen3-1.7b/  qwen3-4b/  qwen3-8b/   (cùng cấu trúc)
reports/          bt_repair_report.json, bt_units_report.json, bt_audit.json
manifest.json     số dòng + sha256 từng file
```

## Nguồn

- 13.200 sản phẩm Tiki → 120.884 unit, trong đó **45.709 unit route MT**
  (13.200 title + 32.509 attribute).
- Split theo `product_id`: train 36.434 / dev 4.824 / test 4.451.
- `cache_key` là sha1 của (source, key, value_class, route) — 29.446 khóa MT.

## Cột chính

| cột | ý nghĩa |
|---|---|
| `pair_id` | id unit (`tiki:<product>:title` / `tiki:<product>:attr:<i>:<j>`) |
| `product_id`, `split`, `unit_type` | ngữ cảnh sản phẩm |
| `source_text_bt` | câu nguồn tiếng Việt |
| `target_text_bt` | bản dịch tiếng Trung của model tương ứng |
| `key_vi`, `key_zh_effective` | tên thuộc tính (chỉ attribute) |
| `qc_status` | `ok` / `repaired` / `repaired_v2` / `recovered_json` / `still_untranslated` |
| `repair_pass`, `prompt_version` | vòng sinh và prompt đã dùng |

## Chất lượng

Đã audit: 29.446/29.446 khóa khớp Plan A, 45.709/45.709 unit map được,
0 `pair_id` trùng, 0 lỗi encoding.

Còn 233 unit gắn cờ `still_untranslated` (mã màu, model code, đơn vị thuần).

Lưu ý: bản dịch là **synthetic**, không phải gold. Xem `reports/bt_audit.json`
và `../../../../reports/tiki_bt_eda.md` trong repo code để biết chi tiết EDA.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def move(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def refactor(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        (root / model).mkdir(exist_ok=True)
        for kind in KINDS:
            move(root / f"tiki_bt_{model}_{kind}.parquet", root / model / f"{kind}.parquet")

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    for name in REPORT_FILES:
        move(root / name, reports / name)

    manifest = {
        "dataset": "tiki-bt",
        "version": "2026-10-29",
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
        entry = {}
        for kind in KINDS:
            path = root / model / f"{kind}.parquet"
            if not path.exists():
                continue
            rows = int(pq.ParquetFile(path).metadata.num_rows)
            entry[kind] = {
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest["models"][model] = entry

    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "README.md").write_text(README, encoding="utf-8")

    # sửa lại đường dẫn file trong report JSON cho khớp layout mới
    for name in ("bt_units_report.json",):
        path = reports / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for model in MODELS:
                text = text.replace(
                    f"data/cleaned/bt_tiki/tiki_bt_{model}_units.parquet",
                    f"data/cleaned/bt_tiki/{model}/units.parquet")
            path.write_text(text, encoding="utf-8")
    return manifest


def upload(root: Path, repo_id: str, prefix: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"HF user: @{who}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(root),
        path_in_repo=prefix,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Refactor Tiki BT layout: one folder per model",
    )
    print(f"uploaded -> https://huggingface.co/datasets/{repo_id}/tree/main/{prefix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", "ntquang0410/zh-vie_ecom"))
    ap.add_argument("--prefix", default="data/cleaned/tiki/bt")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = refactor(root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:1200])
    print("layout:")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(" ", path.relative_to(root), f"{path.stat().st_size/1e6:.2f}MB")

    if args.upload:
        token = os.getenv("HF_TOKEN", "").strip() or None
        if not token:
            env = Path(".env")
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
        if not token:
            print("❌ Không có HF_TOKEN (env hoặc .env) — chưa upload.")
            return 2
        upload(root, args.repo_id, args.prefix, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
