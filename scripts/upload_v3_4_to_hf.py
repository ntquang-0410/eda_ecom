# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(r"D:\download\NCKH\ecom_crawler-main\ecom_crawler-main-feature-1688\.env")
token = os.getenv("HF_TOKEN")
repo_id = os.getenv("HF_REPO_ID")

file_path = r"D:\download\NCKH\eda_ecom\data\tiki_vi_cleaned.parquet"
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Uploading {file_path} ({file_size_mb:.2f} MB) to Hugging Face {repo_id}...")

api = HfApi(token=token)

# 1. Upload as data/cleaned/tiki_vi_cleaned.parquet
commit_url1 = api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="data/cleaned/tiki_vi_cleaned.parquet",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload polished Tiki dataset v3.4 (forensic clean, zero-defect specs & titles, BGE-M3 aligned)"
)
print("Uploaded primary dataset:", commit_url1)

# 2. Upload as data/cleaned/tiki_vi_cleaned(huy).parquet
commit_url2 = api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="data/cleaned/tiki_vi_cleaned(huy).parquet",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload polished Tiki dataset v3.4 (huy) (forensic clean, zero-defect specs & titles)"
)
print("Uploaded huy alias dataset:", commit_url2)

print("Hugging Face upload completed successfully!")
