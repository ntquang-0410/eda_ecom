# -*- coding: utf-8 -*-
"""Đẩy cột `desc_nguon_dich` của 4 bộ description (bộ 1–4, của anh) lên HF dataset ntquang0410/zh-vie_ecom trong MỘT commit (Prompt 523/524).

Chỉ chạm 4 thư mục của anh, tổng 12 file:
    data/1688/descriptions/zh_to_vi_context/   GHI ĐÈ description_products_zh_vi.parquet, README.md, SHA256SUMS.txt
    data/tiki/descriptions/vi_to_zh_context/   GHI ĐÈ description_products_vi_zh.parquet, README.md, SHA256SUMS.txt
    data/1688/descriptions/zh_to_vi/           THÊM description_products_zh_vi_pair.parquet; GHI ĐÈ README.md, SHA256SUMS.txt
    data/tiki/descriptions/vi_to_zh/           THÊM description_products_vi_zh_pair.parquet; GHI ĐÈ README.md, SHA256SUMS.txt
Không đụng root README, data/1688/README.md, data/tiki/README.md, title_translations/, full_dataset/, cleaned/, raw/ (của người khác).
Từ chối chạy nếu trên HF một file định ghi đè khác bản gốc trên máy (có người đã sửa) hoặc file định thêm đã tồn tại; commit gắn parent_commit nên có
người đẩy chen giữa thì HF từ chối. Dựng thư mục nguồn trước bằng them_cot_desc_nguon_dich.py rồi doc_desc_nguon_dich.py.

    <python> scripts/release/push_desc_nguon_dich.py kiem   # chỉ kiểm, không đẩy, không cần token
    <python> scripts/release/push_desc_nguon_dich.py day    # đẩy; token ghi lấy từ HF_TOKEN hoặc gõ ẩn khi được hỏi
    <python> scripts/release/push_desc_nguon_dich.py pr     # mở Pull Request (khi token không ghi thẳng được); dùng phiên `hf auth login`
Chạy bằng Python của ecom_crawler/.venv (đã có huggingface_hub).
"""
import getpass
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ntquang0410/zh-vie_ecom"
ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "data" / "release"
SRC = REL / "desc_nguon_dich_20261028"
# thư mục HF -> (thư mục release gốc trên máy, file ghi đè, file thêm mới)
BO = {
    "data/1688/descriptions/zh_to_vi_context/": (REL / "description_zh_to_vi_cum", ["description_products_zh_vi.parquet", "README.md", "SHA256SUMS.txt"], []),
    "data/tiki/descriptions/vi_to_zh_context/": (REL / "description_vi_to_zh_cum", ["description_products_vi_zh.parquet", "README.md", "SHA256SUMS.txt"], []),
    "data/1688/descriptions/zh_to_vi/": (REL / "description_zh_to_vi", ["README.md", "SHA256SUMS.txt"], ["description_products_zh_vi_pair.parquet"]),
    "data/tiki/descriptions/vi_to_zh/": (REL / "description_vi_to_zh", ["README.md", "SHA256SUMS.txt"], ["description_products_vi_zh_pair.parquet"]),
}
TIEU_DE = "Add desc_nguon_dich to the four description sets and product tables for the two pair sets (nhatanh)"
MO_TA = ("New last column desc_nguon_dich (gemma4 / qwen3_14b_awq / gemma4+qwen3_14b_awq / tu_dien_quy_tac) in the products tables of "
         "data/1688/descriptions/zh_to_vi_context/ and data/tiki/descriptions/vi_to_zh_context/ (other columns unchanged), and new products tables "
         "description_products_zh_vi_pair.parquet / description_products_vi_zh_pair.parquet in the pair folders. Updates the READMEs and SHA256SUMS of these "
         "four folders only.")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p):
    d = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()


def hf_khop(f, p):
    """File HF f có đúng nội dung file máy p không (LFS: SHA256; file thường: git blob sha1)."""
    if getattr(f, "lfs", None):
        return f.lfs.sha256 == sha(p)
    return f.blob_id == git_blob_sha1(p)


def danh_sach():
    """[(đường dẫn HF, file mới trên máy, file gốc trên máy hoặc None nếu thêm mới)]"""
    out = []
    for d, (goc, ghi_de, them) in BO.items():
        out += [(d + f, SRC / d / f, goc / f) for f in ghi_de]
        out += [(d + f, SRC / d / f, None) for f in them]
    return out


def kiem_may():
    for d, (goc, ghi_de, them) in BO.items():
        sums = {ln.split()[1]: ln.split()[0] for ln in (SRC / d / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
        assert len(sums) == 3, (d, sums)
        readme = (SRC / d / "README.md").read_text(encoding="utf-8")
        for ten, h in sums.items():
            p = SRC / d / ten if (SRC / d / ten).exists() else goc / ten
            assert sha(p) == h, f"SHA256 lệch: {d}{ten}"
            assert h in readme and f"{p.stat().st_size:,}" in readme, f"README chưa ghi đúng SHA256 hoặc kích thước: {d}{ten}"
    import pyarrow.parquet as pq
    for hf, moi, _ in danh_sach():
        if hf.endswith(".parquet"):
            cot = pq.ParquetFile(moi).schema_arrow.names
            assert cot[-1] == "desc_nguon_dich" and len(cot) == 9, (hf, cot)


def kiem(api):
    kiem_may()
    info = api.dataset_info(REPO)
    ds = danh_sach()
    tren = {f.path: f for f in api.get_paths_info(REPO, [h for h, _, _ in ds], repo_type="dataset", revision=info.sha, expand=True)}
    for hf, moi, goc in ds:
        if goc is None:
            assert hf not in tren, f"DỪNG: {hf} đã có trên HF, không ghi đè file thêm mới"
        else:
            assert hf in tren, f"DỪNG: thiếu trên HF file định ghi đè: {hf}"
            assert hf_khop(tren[hf], goc), f"DỪNG: {hf} trên HF khác bản gốc trên máy ({goc.name}): có người đã sửa? Đối chiếu trước khi đẩy"
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    print(f"Repo {REPO} @ {info.sha[:10]}: {len(co_san)} mục. Mọi file định ghi đè trên HF khớp từng byte với bản gốc trên máy; mọi file thêm mới chưa tồn tại.")
    print(f"Sẽ đẩy {len(ds)} file trong MỘT commit:")
    for hf, moi, goc in ds:
        print(f"   {'THÊM  ' if goc is None else 'GHI ĐÈ'} {hf}  ({tren[hf].size if hf in tren else 0:,} -> {moi.stat().st_size:,} byte)")
    return info.sha, co_san


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
    ds = danh_sach()
    ops = [CommitOperationAdd(path_in_repo=hf, path_or_fileobj=str(moi)) for hf, moi, _ in ds]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head, commit_message=TIEU_DE, commit_description=MO_TA)
    print("Commit:", c.commit_url)
    sau = {f.path: f for f in api.get_paths_info(REPO, [h for h, _, _ in ds], repo_type="dataset", expand=True)}
    for hf, moi, _ in ds:
        assert hf in sau and hf_khop(sau[hf], moi), f"sau khi đẩy, {hf} trên HF khác bản máy"
    toan = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)}
    doi = {h for h, _, _ in ds}
    for d, f in truoc.items():
        if d in doi or getattr(f, "size", None) is None:
            continue
        assert d in toan and toan[d].size == f.size, f"file khác bị đổi: {d}"
        if getattr(f, "lfs", None) and getattr(toan[d], "lfs", None):
            assert f.lfs.sha256 == toan[d].lfs.sha256, f"file khác bị đổi nội dung: {d}"
    print(f"OK: {len(ds)} file khớp bản máy; {len([d for d in truoc if d not in doi])} mục khác của repo giữ nguyên.")


def pr(api):
    print("Đăng nhập HF:", api.whoami()["name"])
    head, _ = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=hf, path_or_fileobj=str(moi)) for hf, moi, _ in danh_sach()]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, create_pr=True, commit_message=TIEU_DE, commit_description=MO_TA)
    print("Pull Request:", c.pr_url)
    nhanh = api.dataset_info(REPO).sha
    print(f"main vẫn ở {nhanh[:10]} (không đổi: {nhanh == head}); file nằm ở nhánh PR, chờ chủ repo Merge.")


if __name__ == "__main__":
    {"kiem": kiem, "day": day, "pr": pr}[sys.argv[1]](HfApi())
