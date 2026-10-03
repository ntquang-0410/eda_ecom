# -*- coding: utf-8 -*-
"""Đẩy bản công bố description (desc_v1) lên HF dataset ntquang0410/zh-vie_ecom, thư mục data/cleaned/.

Chỉ THÊM file mới, không ghi đè: dừng nếu đường dẫn đích đã có trên repo; commit gắn parent_commit = phiên bản vừa kiểm
(có người đẩy chen giữa thì HF từ chối). Chạy bằng Python của ecom_crawler/.venv (đã có huggingface_hub 1.31).

    <python> scripts/release/push_desc_hf.py kiem     # chỉ kiểm, không đẩy (không cần đăng nhập)
    <python> scripts/release/push_desc_hf.py day      # đẩy (cần đã `hf auth login` bằng token có quyền ghi)
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ntquang0410/zh-vie_ecom"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "release" / "desc_v1"
DICH = "data/cleaned/"
FILES = {  # đường dẫn trên HF → file trên máy
    DICH + "bilingual_zh_vi_desc_cleaned(nhatanh).parquet": SRC / "bilingual_zh_vi_desc_cleaned(nhatanh).parquet",
    DICH + "bilingual_zh_vi_desc_pairs(nhatanh).parquet": SRC / "bilingual_zh_vi_desc_pairs(nhatanh).parquet",
    DICH + "README_desc(nhatanh).md": SRC / "README.md",
    DICH + "SHA256SUMS_desc(nhatanh).txt": SRC / "SHA256SUMS.txt",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kiem(api):
    # 1. file trên máy khớp SHA256SUMS
    tong = {ln.split()[1]: ln.split()[0] for ln in (SRC / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
    for ten, h in tong.items():
        assert sha(SRC / ten) == h, f"SHA256 lệch: {ten}"
    for p in FILES.values():
        assert p.exists(), p
    # 2. trên HF chưa có đường dẫn nào trùng
    info = api.dataset_info(REPO)
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    trung = [d for d in FILES if d in co_san]
    print(f"Repo {REPO} @ {info.sha[:10]}: {len(co_san)} mục; trong {DICH}:",
          sorted(p for p in co_san if p.startswith(DICH)))
    assert not trung, f"DỪNG: đã có trên HF, không ghi đè: {trung}"
    print("OK: 4 đường dẫn đích đều chưa có; SHA256 trên máy khớp.")
    return info.sha, co_san


def day(api):
    me = api.whoami()
    print("Đăng nhập HF:", me["name"])
    head, truoc = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(p)) for d, p in FILES.items()]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head,
                          commit_message="Add cleaned description data desc_v1 (nhatanh)",
                          commit_description="16.342 sản phẩm / 366.016 cặp dùng cho train; xem README_desc(nhatanh).md.")
    print("Commit:", c.commit_url)
    # 3. kiểm lại sau khi đẩy
    sau = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)}
    for d, p in FILES.items():
        assert d in sau and sau[d].size == p.stat().st_size, f"thiếu/lệch kích thước: {d}"
    for d, f in truoc.items():
        if getattr(f, "size", None) is not None:
            assert d in sau and sau[d].size == f.size, f"file cũ bị đổi: {d}"
            if getattr(f, "lfs", None) and getattr(sau[d], "lfs", None):
                assert f.lfs.sha256 == sau[d].lfs.sha256, f"file cũ bị đổi nội dung: {d}"
    print(f"OK: 4 file mới có mặt; {len(truoc)} mục cũ giữ nguyên.")


if __name__ == "__main__":
    # Token: lấy từ phiên đăng nhập `hf auth login` hoặc biến môi trường HF_TOKEN do người chạy tự đặt
    api = HfApi()
    {"kiem": kiem, "day": day}[sys.argv[1]](api)
