#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload data/cleaned/bt_tiki lên Hugging Face, giữ layout theo model.

    python tools/upload_bt_hf.py                 # đọc HF_TOKEN từ env hoặc .env
    python tools/upload_bt_hf.py --token hf_...  # hoặc truyền trực tiếp

Repo mặc định: ntquang0410/zh-vie_ecom
Prefix mặc định: data/cleaned/tiki/bt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def read_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit.strip()
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    # token do huggingface-cli lưu
    for path in (Path.home() / ".cache/huggingface/token",
                 Path.home() / ".huggingface/token"):
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cleaned/bt_tiki")
    ap.add_argument("--repo-id", default="ntquang0410/zh-vie_ecom")
    ap.add_argument("--prefix", default="data/cleaned/tiki/bt")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = read_token(args.token)
    if not token:
        print("❌ Không tìm thấy HF_TOKEN (env, .env, ~/.cache/huggingface/token).")
        print("   Dán token vào /content/eda_ecom/.env:  HF_TOKEN=hf_...")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"✅ HF user: @{who}")

    root = Path(args.root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"📦 {len(files)} file, {sum(p.stat().st_size for p in files)/1e6:.1f} MB")

    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(root),
        path_in_repo=args.prefix,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Tiki BT: v5 main + v3 archive, one folder per model",
    )
    print(f"✅ uploaded -> https://huggingface.co/datasets/{args.repo_id}/tree/main/{args.prefix}")

    remote = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    prefix = args.prefix.rstrip("/") + "/"
    uploaded = [f for f in remote if f.startswith(prefix)]
    print(f"remote files under prefix: {len(uploaded)}")
    for name in uploaded[:20]:
        print("  ", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
