# -*- coding: utf-8 -*-
"""Đẩy bộ song ngữ 1 (description, dịch lại bằng LLM) lên HF dataset ntquang0410/zh-vie_ecom, thư mục data/description_zh_to_vi/.

Chỉ THÊM file mới, không ghi đè: dừng nếu thư mục đích đã có bất kỳ file nào; commit gắn parent_commit = phiên bản vừa kiểm
(có người đẩy chen giữa thì HF từ chối); đẩy xong kiểm lại 4 file mới đúng kích thước và mọi file cũ giữ nguyên.
Chạy bằng Python của ecom_crawler/.venv (đã có huggingface_hub 1.31). Dựng thư mục trước bằng build_description_zh_to_vi.py.

    <python> scripts/release/push_description_zh_to_vi.py kiem   # chỉ kiểm, không đẩy, không cần token
    <python> scripts/release/push_description_zh_to_vi.py day    # đẩy; token ghi lấy từ HF_TOKEN hoặc gõ ẩn khi được hỏi
"""
import getpass
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ntquang0410/zh-vie_ecom"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "release" / "description_zh_to_vi"
DICH = "data/description_zh_to_vi/"
FILES = {DICH + f: SRC / f for f in ("README.md", "description_pairs_zh_vi.parquet", "provenance.json", "SHA256SUMS.txt")}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kiem(api):
    tong = {ln.split()[1]: ln.split()[0] for ln in (SRC / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
    for ten, h in tong.items():
        assert sha(SRC / ten) == h, f"SHA256 lệch: {ten}"
    for p in FILES.values():
        assert p.exists(), p
    info = api.dataset_info(REPO)
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    trong_dich = sorted(p for p in co_san if p.startswith(DICH) or p == DICH.rstrip("/"))
    print(f"Repo {REPO} @ {info.sha[:10]}: {len(co_san)} mục; trong {DICH}: {trong_dich}")
    assert not trong_dich, f"DỪNG: thư mục đích đã có trên HF, không ghi đè: {trong_dich}"
    print(f"OK: {DICH} chưa có trên HF; SHA256 trên máy khớp; sẽ thêm {len(FILES)} file:")
    for d, p in FILES.items():
        print(f"   {d}  ({p.stat().st_size:,} byte)")
    return info.sha, co_san


def day(_):
    token = (os.environ.get("HF_TOKEN") or getpass.getpass("Dán token HF có quyền GHI vào repo (gõ ẩn, Enter): ")).strip().strip("\"'")
    if token.startswith("HF_TOKEN="):
        token = token[len("HF_TOKEN="):].strip().strip("\"'")
    if not token.startswith("hf_") or len(token) < 30:
        sys.exit(f"DỪNG: chuỗi nhận được dài {len(token)} ký tự và không giống token HF (phải bắt đầu bằng hf_). Copy lại rồi chạy lại.")
    api = HfApi(token=token)
    print("Đăng nhập HF:", api.whoami()["name"])
    head, truoc = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(p)) for d, p in FILES.items()]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head,
                          commit_message="Add data/description_zh_to_vi: description pairs re-translated by LLMs (nhatanh)",
                          commit_description="394,662 zh-vi attribute pairs, 16,342 products; Gemma 4 26B-A4B + Qwen3-14B, "
                                             "rule checks + CometKiwi selection. See data/description_zh_to_vi/README.md.")
    print("Commit:", c.commit_url)
    sau = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)}
    for d, p in FILES.items():
        assert d in sau and sau[d].size == p.stat().st_size, f"thiếu/lệch kích thước: {d}"
    for d, f in truoc.items():
        if getattr(f, "size", None) is not None:
            assert d in sau and sau[d].size == f.size, f"file cũ bị đổi: {d}"
            if getattr(f, "lfs", None) and getattr(sau[d], "lfs", None):
                assert f.lfs.sha256 == sau[d].lfs.sha256, f"file cũ bị đổi nội dung: {d}"
    print(f"OK: {len(FILES)} file mới có mặt; {len(truoc)} mục cũ giữ nguyên.")


def pr(api):
    """Mở Pull Request thêm 4 file (dùng khi token không có quyền ghi thẳng vào repo của Quang). Dùng phiên đăng nhập `hf auth login`."""
    print("Đăng nhập HF:", api.whoami()["name"])
    head, truoc = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(p)) for d, p in FILES.items()]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, create_pr=True,
                          commit_message="Add data/description_zh_to_vi: description pairs re-translated by LLMs (nhatanh)",
                          commit_description="394,662 zh-vi attribute pairs, 16,342 products; Gemma 4 26B-A4B + Qwen3-14B, "
                                             "rule checks + CometKiwi selection. Adds only new files under data/description_zh_to_vi/; "
                                             "see its README.md.")
    print("Pull Request:", c.pr_url)
    nhanh = api.dataset_info(REPO).sha
    print(f"main vẫn ở {nhanh[:10]} (không đổi: {nhanh == head}); file nằm ở nhánh PR, chờ chủ repo Merge.")


if __name__ == "__main__":
    {"kiem": kiem, "day": day, "pr": pr}[sys.argv[1]](HfApi())
