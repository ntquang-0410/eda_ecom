# -*- coding: utf-8 -*-
"""Đẩy bộ song ngữ 4 (1688 Trung -> Việt, hiệu đính cả cụm theo sản phẩm, có ngữ cảnh: bảng cặp và bảng sản phẩm) lên HF dataset ntquang0410/zh-vie_ecom,
thư mục data/1688/descriptions/zh_to_vi_context/ (bố cục repo sau khi ntquang0410 gom theo nguồn 25/10/2026). Bản gương của push_description_vi_to_zh_cum.py.

Chỉ THÊM 5 file mới trong thư mục của bộ 4, KHÔNG đụng bất kỳ mục nào khác của repo (README gốc, data/1688/README.md, data/tiki/README.md, các thư mục của
người khác): dừng nếu thư mục đích đã có bất kỳ file nào; mọi đường dẫn thao tác phải nằm trong thư mục đích; commit gắn parent_commit = phiên bản vừa kiểm
(có người đẩy chen giữa thì HF từ chối); đẩy xong kiểm lại 5 file mới đúng kích thước và MỌI mục cũ của repo giữ nguyên (kích thước, SHA256).
Chạy bằng Python của ecom_crawler/.venv (đã có huggingface_hub). Dựng thư mục trước bằng build_description_zh_to_vi_cum.py.

    <python> scripts/release/push_description_zh_to_vi_cum.py kiem   # chỉ kiểm, không đẩy, không cần token
    <python> scripts/release/push_description_zh_to_vi_cum.py day    # đẩy thẳng; token ghi lấy từ HF_TOKEN hoặc gõ ẩn khi được hỏi
    <python> scripts/release/push_description_zh_to_vi_cum.py pr     # mở Pull Request (khi token không ghi thẳng được); dùng phiên `hf auth login`
"""
import getpass
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ntquang0410/zh-vie_ecom"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "release" / "description_zh_to_vi_cum"
DICH = "data/1688/descriptions/zh_to_vi_context/"
FILES = {DICH + f: SRC / f for f in ("README.md", "description_pairs_zh_vi_ctx.parquet", "description_products_zh_vi.parquet", "provenance.json", "SHA256SUMS.txt")}
SHA_BO_1 = "e8c5cfed878268e8f63cd8760f2dbd91f4c420f21107b76082d0378a42ffb9ba"      # bảng cặp bộ 1 trên HF, bản nháp của bộ 4
# file mà README dẫn tới phải có sẵn trên HF (bố cục mới)
PHAI_CO = ("data/1688/descriptions/zh_to_vi/description_pairs_zh_vi.parquet", "data/1688/snapshot/bilingual_zh_vi.parquet",
           "data/tiki/descriptions/vi_to_zh_context/description_products_vi_zh.parquet")
TIEU_DE = "Add data/1688/descriptions/zh_to_vi_context: 1688 attributes zh->vi reviewed per product with context (nhatanh)"
MO_TA = ("401,692 zh-vi attribute pairs and 16,342 product descriptions (same translation, two tables); Gemma 4 26B-A4B (Qwen3-14B as fallback) review "
         "the unit-level draft of data/1688/descriptions/zh_to_vi with the whole product in view. Adds only new files under "
         "data/1688/descriptions/zh_to_vi_context/; see its README.md.")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kiem(api):
    tong = {ln.split()[1]: ln.split()[0] for ln in (SRC / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
    assert len(tong) == 3, tong
    for ten, h in tong.items():
        assert sha(SRC / ten) == h, f"SHA256 lệch: {ten}"
    for p in FILES.values():
        assert p.exists(), p
    assert all(d.startswith(DICH) for d in FILES), "có đường dẫn ngoài thư mục của bộ 4"
    readme = (SRC / "README.md").read_text(encoding="utf-8")
    for ten, h in tong.items():
        assert h in readme, f"README chưa ghi đúng SHA256 của {ten}"
        assert f"{(SRC / ten).stat().st_size:,}" in readme, f"README chưa ghi đúng kích thước của {ten}"
    assert "@@" not in readme and DICH in readme
    info = api.dataset_info(REPO)
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    trong_dich = sorted(p for p in co_san if p.startswith(DICH) or p == DICH.rstrip("/"))
    print(f"Repo {REPO} @ {info.sha[:10]}: {len(co_san)} mục; trong {DICH}: {trong_dich}")
    assert not trong_dich, f"DỪNG: thư mục đích đã có trên HF, không ghi đè: {trong_dich}"
    for ten in PHAI_CO:
        assert ten in co_san, f"thiếu file được README dẫn tới trên HF: {ten}"
    bo1 = api.get_paths_info(REPO, [PHAI_CO[0]], repo_type="dataset", revision=info.sha, expand=True)[0]
    assert bo1.lfs.sha256 == SHA_BO_1, "bảng cặp bộ 1 trên HF khác bản mà bộ 4 dùng làm nháp: có người đã sửa? DỪNG"
    print(f"OK: {DICH} chưa có trên HF; SHA256 và kích thước trên máy khớp README; các file README dẫn tới có trên HF; bảng cặp bộ 1 trên HF đúng bản nháp; "
          f"sẽ thêm {len(FILES)} file:")
    for d, p in FILES.items():
        print(f"   {d}  ({p.stat().st_size:,} byte)")
    return info.sha, co_san


def kiem_sau(api, head, truoc, revision=None):
    sau = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=revision, recursive=True)}
    for d, p in FILES.items():
        assert d in sau and sau[d].size == p.stat().st_size, f"thiếu/lệch kích thước: {d}"
    for d, f in truoc.items():
        if getattr(f, "size", None) is not None:
            assert d in sau and sau[d].size == f.size, f"mục cũ bị đổi: {d}"
            if getattr(f, "lfs", None) and getattr(sau[d], "lfs", None):
                assert f.lfs.sha256 == sau[d].lfs.sha256, f"mục cũ bị đổi nội dung: {d}"
    moi = sorted(set(sau) - set(truoc))
    assert all(d.startswith(DICH) for d in moi), f"có mục mới ngoài thư mục của bộ 4: {moi}"
    print(f"OK: {len(FILES)} file mới có mặt; {len(truoc)} mục cũ giữ nguyên (kích thước và SHA256); mọi mục mới nằm trong {DICH}")


def lay_token():
    token = (os.environ.get("HF_TOKEN") or getpass.getpass("Dán token HF có quyền GHI vào repo (gõ ẩn, Enter): ")).strip().strip("\"'")
    if token.startswith("HF_TOKEN="):
        token = token[len("HF_TOKEN="):].strip().strip("\"'")
    if not token.startswith("hf_") or len(token) < 30:
        sys.exit(f"DỪNG: chuỗi nhận được dài {len(token)} ký tự và không giống token HF (phải bắt đầu bằng hf_). Copy lại rồi chạy lại.")
    return token


def day(_):
    api = HfApi(token=lay_token())
    print("Đăng nhập HF:", api.whoami()["name"])
    head, truoc = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(p)) for d, p in FILES.items()]
    assert all(o.path_in_repo.startswith(DICH) for o in ops)
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head, commit_message=TIEU_DE, commit_description=MO_TA)
    print("Commit:", c.commit_url)
    kiem_sau(api, head, truoc)


def pr(api):
    print("Đăng nhập HF:", api.whoami()["name"])
    head, truoc = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(p)) for d, p in FILES.items()]
    assert all(o.path_in_repo.startswith(DICH) for o in ops)
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, create_pr=True, commit_message=TIEU_DE, commit_description=MO_TA)
    print("Pull Request:", c.pr_url)
    nhanh = api.dataset_info(REPO).sha
    print(f"main vẫn ở {nhanh[:10]} (không đổi: {nhanh == head}); file nằm ở nhánh PR, chờ chủ repo Merge.")


if __name__ == "__main__":
    {"kiem": kiem, "day": day, "pr": pr}[sys.argv[1]](HfApi())
