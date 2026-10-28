#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đóng gói Plan A artifacts + configs thành Kaggle Dataset staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    "data/cleaned/tiki_bt_input.parquet",
    "data/cleaned/tiki_translation_cache.parquet",
    "data/cleaned/tiki_bt_units_plan_a.parquet",
    "data/cleaned/tiki_products_bt_state.parquet",
    "data/cleaned/tiki_bt_units_length_v3.parquet",
    "data/cleaned/tiki_products.parquet",
    "data/cleaned/tiki_bt_plan_a_stats.json",
]
CSV_FILES = [
    "csv/tiki/tiki_key_map.csv",
    "csv/tiki/tiki_key_map_review.csv",
    "csv/tiki/tiki_bt_routing_summary.csv",
]
CODE_FILES = [
    "cado_core/__init__.py",
    "cado_core/config.py",
    "cado_core/text.py",
    "cado_core/tiki.py",
    "cado_core/tiki_units.py",
    "cado_core/tiki_bt_prepare.py",
    "cado_core/key_map_review.py",
    "cado_core/length_prior.py",
    "cado_core/length_prior_v2.py",
    "cado_core/length_prior_v3.py",
]


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def kaggle_username() -> str:
    env = os.getenv("KAGGLE_USERNAME")
    if env:
        return env
    try:
        result = subprocess.run(
            ["kaggle", "config", "view"],
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return ""
    for line in (result.stdout or "").splitlines():
        stripped = line.strip().lstrip("-").strip()
        if stripped.lower().startswith("username:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(relative: str, staging: Path) -> dict:
    source = PROJECT_ROOT / relative
    if not source.exists():
        raise FileNotFoundError(f"Thiếu file: {source}")
    # Flat layout: Kaggle CLI bỏ qua thư mục con nếu không dùng dir-mode.
    target = staging / Path(relative).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "path": target.name,
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=None)
    parser.add_argument("--dataset-slug", default="tiki-plan-a-bt-input")
    parser.add_argument(
        "--staging-dir",
        default="/tmp/opencode/kaggle_plan_a",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    owner = args.owner or kaggle_username()
    if not owner:
        raise SystemExit(
            "Không đọc được Kaggle username. Set KAGGLE_USERNAME hoặc "
            "truyền --owner."
        )
    args.owner = owner
    staging = Path(args.staging_dir)
    if staging.exists():
        if not args.force:
            raise SystemExit(f"{staging} đã tồn tại; dùng --force để ghi đè.")
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for relative in DATA_FILES + CSV_FILES:
        manifest.append(copy_file(relative, staging))

    config_src = PROJECT_ROOT / "kaggle/configs"
    for path in sorted(config_src.glob("*.json")):
        target = staging / path.name
        shutil.copy2(path, target)
        manifest.append({
            "path": target.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    code_zip = staging / "code.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in CODE_FILES:
            source = PROJECT_ROOT / relative
            if source.exists():
                archive.write(source, arcname=relative)
    manifest.append({
        "path": "code.zip",
        "bytes": code_zip.stat().st_size,
        "sha256": sha256(code_zip),
    })

    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        shutil.copy2(requirements, staging / "requirements.txt")

    metadata = {
        "title": "Tiki Plan A BT Input",
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Tiki Plan A value-only BT input, key map, cache and product state",
        "description": (
            "Plan A artifacts for Tiki back-translation. "
            "Synthetic outputs are not included; only pre-BT inputs."
        ),
        "isPrivate": True,
        "keywords": ["ecommerce", "vietnamese", "chinese", "translation", "tiki"],
    }
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (staging / "manifest.json").write_text(
        json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": metadata["id"],
            "files": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"staging: {staging}")
    print(f"dataset id: {metadata['id']}")
    print(f"files: {len(manifest)}")
    print("\nUpload:")
    print(f"  kaggle datasets create -p {staging}")
    print(f"  # lần sau:\n  kaggle datasets version -p {staging} -m 'Plan A update'")


if __name__ == "__main__":
    main()
