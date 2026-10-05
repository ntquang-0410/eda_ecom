#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script upload file dataset lên Hugging Face Hub.
Đọc cấu hình HF_TOKEN và HF_REPO_ID từ file .env.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Tải biến môi trường từ .env
load_dotenv()

DEFAULT_PARQUET_FILE = (
    "data/cleaned/bilingual_zh_vi_cleaned(huy).parquet"
    if Path("data/cleaned/bilingual_zh_vi_cleaned(huy).parquet").is_file()
    else "bilingual_zh_vi_cleaned(huy).parquet"
)


def create_dataset_card_content(repo_id: str, filename: str) -> str:
    """Tạo nội dung Dataset Card (README.md) chuẩn cho Hugging Face Dataset."""
    return f"""---
language:
- zh
- vi
multilinguality:
- translation
task_categories:
- translation
tags:
- ecommerce
- 1688
- bilingual
- chinese-vietnamese
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: "{filename}"
---

# {repo_id.split('/')[-1]}

Bilingual Chinese - Vietnamese dataset for E-commerce product titles & descriptions (cleaned).

- **Source:** E-commerce (1688 / Taobao)
- **Primary file:** `{filename}`
- **Languages:** Chinese (zh) <-> Vietnamese (vi)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Upload file Parquet lên Hugging Face Dataset Hub."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_PARQUET_FILE,
        help=f"Đường dẫn file cần upload (mặc định: {DEFAULT_PARQUET_FILE})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("HF_REPO_ID", "").strip(),
        help="HF Dataset Repo ID (VD: username/bilingual_zh_vi_cleaned). Mặc định lấy từ .env HF_REPO_ID",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Đường dẫn file khi lưu trên repo HF (mặc định theo đường dẫn file gốc)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=os.getenv("HF_PRIVATE", "false").lower() in ("true", "1", "yes"),
        help="Đặt repo là Private (mặc định lấy từ .env HF_PRIVATE hoặc False)",
    )
    parser.add_argument(
        "--skip-card",
        action="store_true",
        help="Bỏ qua việc tạo Dataset Card (README.md) nếu repo mới được tạo",
    )

    args = parser.parse_args()

    # 1. Kiểm tra file tồn tại
    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"❌ LỖI: Không tìm thấy file '{file_path.resolve()}'.")
        sys.exit(1)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"📦 File: {file_path.name} ({file_size_mb:.2f} MB)")

    # 2. Kiểm tra token
    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token.startswith("hf_xxx"):
        print("❌ LỖI: Chưa cấu hình HF_TOKEN hợp lệ trong file .env!")
        print("👉 Vui lòng mở file .env và điền Token của bạn:")
        print("   1. Truy cập: https://huggingface.co/settings/tokens")
        print("   2. Tạo một Token với quyền WRITE (hoặc full access)")
        print("   3. Dán vào dòng: HF_TOKEN=hf_... trong file .env")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Thư viện 'huggingface_hub' chưa được cài đặt.")
        print("👉 Chạy: pip install -r requirements.txt")
        sys.exit(1)

    api = HfApi(token=token)

    # 3. Xác thực Token & lấy thông tin user
    print("🔑 Đang kiểm tra token Hugging Face...")
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"✅ Đăng nhập thành công với tài khoản: @{username}")
    except Exception as e:
        print(f"❌ Xác thực token thất bại: {e}")
        print("👉 Vui lòng kiểm tra lại giá trị HF_TOKEN trong .env!")
        sys.exit(1)

    # 4. Xác định Repo ID
    repo_id = args.repo_id
    if not repo_id:
        # Nếu chưa đặt repo_id, dùng default username/bilingual_zh_vi_cleaned
        repo_name = "bilingual_zh_vi_cleaned"
        repo_id = f"{username}/{repo_name}"
        print(f"ℹ️ Không có HF_REPO_ID trong .env, tự động dùng: {repo_id}")
    elif "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    path_in_repo = args.path_in_repo or str(file_path).replace("\\", "/")

    # 5. Tạo Repo nếu chưa tồn tại
    print(f"🔍 Kiểm tra repository: https://huggingface.co/datasets/{repo_id} ...")
    repo_existed = False
    try:
        repo_existed = api.repo_exists(repo_id=repo_id, repo_type="dataset")
    except Exception:
        repo_existed = False

    if not repo_existed:
        print(f"🚀 Đang tạo dataset repo mới '{repo_id}' (private={args.private})...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=args.private,
                exist_ok=True,
            )
            print(f"✅ Đã tạo repository '{repo_id}'.")
        except Exception as e:
            print(f"❌ Không thể tạo repo '{repo_id}': {e}")
            sys.exit(1)
    else:
        print(f"✅ Repository '{repo_id}' đã tồn tại.")

    # Xoá file cũ 'data/cleaned.parquet' nếu có để đồng bộ với thư mục data/cleaned/
    try:
        if api.file_exists(repo_id=repo_id, filename="data/cleaned.parquet", repo_type="dataset"):
            print("🗑️ Đang xoá file cũ 'data/cleaned.parquet' trên repo...")
            api.delete_file(
                path_in_repo="data/cleaned.parquet",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Remove data/cleaned.parquet in favor of data/cleaned/ directory",
            )
            print("✅ Đã xoá 'data/cleaned.parquet' trên repo.")
    except Exception:
        pass

    # 6. Upload file parquet
    print(f"\n📤 Đang upload '{file_path.name}' lên '{repo_id}'...")
    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload dataset {path_in_repo}",
        )
        print(f"✅ Đã upload thành công '{path_in_repo}'!")
    except Exception as e:
        print(f"❌ Upload thất bại: {e}")
        sys.exit(1)

    # 7. Tạo Dataset Card README.md nếu chưa có và không bị skip
    if not args.skip_card:
        try:
            card_exists = False
            try:
                card_exists = api.file_exists(
                    repo_id=repo_id, filename="README.md", repo_type="dataset"
                )
            except Exception:
                card_exists = False

            if not card_exists:
                print("📝 Đang tạo Dataset Card (README.md) tự động...")
                card_content = create_dataset_card_content(repo_id, path_in_repo)
                api.upload_file(
                    path_or_fileobj=card_content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Add initial dataset card and YAML configuration",
                )
                print("✅ Đã tạo Dataset Card (README.md)!")
        except Exception as e:
            print(f"⚠️ Không thể tạo README.md (không nghiêm trọng): {e}")

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ ĐÃ HOÀN TẤT!")
    print(f"🔗 Xem dataset của bạn tại: {dataset_url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
