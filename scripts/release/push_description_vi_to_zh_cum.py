# -*- coding: utf-8 -*-
"""Đẩy bộ song ngữ 3 (Tiki Việt -> Trung, dịch theo cụm có ngữ cảnh: bảng cặp và bảng sản phẩm) lên HF dataset ntquang0410/zh-vie_ecom,
thư mục data/description_vi_to_zh_cum/. Bản gương của push_description_vi_to_zh.py.

Chỉ THÊM file mới, không ghi đè: dừng nếu thư mục đích đã có bất kỳ file nào; commit gắn parent_commit = phiên bản vừa kiểm (có người đẩy chen
giữa thì HF từ chối); đẩy xong kiểm lại 5 file mới đúng kích thước và mọi file cũ giữ nguyên.
Chạy bằng Python của ecom_crawler/.venv (đã có huggingface_hub). Dựng thư mục trước bằng build_description_vi_to_zh_cum.py.

    <python> scripts/release/push_description_vi_to_zh_cum.py kiem   # chỉ kiểm, không đẩy, không cần token
    <python> scripts/release/push_description_vi_to_zh_cum.py day    # đẩy thẳng; token ghi lấy từ HF_TOKEN hoặc gõ ẩn khi được hỏi
    <python> scripts/release/push_description_vi_to_zh_cum.py pr     # mở Pull Request (khi token không ghi thẳng được); dùng phiên `hf auth login`
    <python> scripts/release/push_description_vi_to_zh_cum.py kiem_cap_nhat   # kiểm lần cập nhật 2 (thêm so_cap_thap), không cần token
    <python> scripts/release/push_description_vi_to_zh_cum.py cap_nhat        # đẩy lần cập nhật 2: ghi đè README, bảng sản phẩm, SHA256SUMS
"""
import getpass
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ntquang0410/zh-vie_ecom"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "release" / "description_vi_to_zh_cum"
# 25/10/2026 ntquang0410 gom repo theo nguồn (data/1688, data/tiki): thư mục này đã được CHUYỂN từ data/description_vi_to_zh_cum/ sang đường dẫn dưới
# đây (nội dung giữ nguyên từng byte, đã đối chiếu 26/10). Đẩy vào đường dẫn cũ sẽ tạo thư mục thừa.
DICH = "data/tiki/descriptions/vi_to_zh_context/"
FILES = {DICH + f: SRC / f for f in ("README.md", "description_pairs_vi_zh_ctx.parquet", "description_products_vi_zh.parquet", "provenance.json",
                                      "SHA256SUMS.txt")}
TIEU_DE = "Add data/description_vi_to_zh_cum: Tiki attributes translated vi->zh per product with context (nhatanh)"
MO_TA = ("87,112 vi-zh attribute pairs and 13,187 product descriptions (same translation, two tables); Gemma 4 26B-A4B + Qwen3-14B review "
         "the unit-level draft of data/description_vi_to_zh with the whole product in view. Adds only new files under "
         "data/description_vi_to_zh_cum/; see its README.md.")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kiem(api):
    tong = {ln.split()[1]: ln.split()[0] for ln in (SRC / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
    assert len(tong) == 3, tong
    for ten, h in tong.items():
        assert sha(SRC / ten) == h, f"SHA256 lệch: {ten}"
    for p in FILES.values():
        assert p.exists(), p
    readme = (SRC / "README.md").read_text(encoding="utf-8")
    for ten, h in tong.items():
        assert h in readme, f"README chưa ghi đúng SHA256 của {ten}"
        assert f"{(SRC / ten).stat().st_size:,}" in readme, f"README chưa ghi đúng kích thước của {ten}"
    info = api.dataset_info(REPO)
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    trong_dich = sorted(p for p in co_san if p.startswith(DICH) or p == DICH.rstrip("/"))
    print(f"Repo {REPO} @ {info.sha[:10]}: {len(co_san)} mục; trong {DICH}: {trong_dich}")
    assert not trong_dich, f"DỪNG: thư mục đích đã có trên HF, không ghi đè: {trong_dich}"
    for ten in ("data/tiki/descriptions/vi_to_zh/description_pairs_vi_zh.parquet", "data/tiki/cleaned/tiki_eda.parquet"):
        assert ten in co_san, f"thiếu file được README dẫn tới trên HF: {ten}"
    print(f"OK: {DICH} chưa có trên HF; SHA256 và kích thước trên máy khớp README; hai file README dẫn tới có trên HF; sẽ thêm {len(FILES)} file:")
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
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head, commit_message=TIEU_DE, commit_description=MO_TA)
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
    print("Đăng nhập HF:", api.whoami()["name"])
    head, truoc = kiem(api)
    ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(p)) for d, p in FILES.items()]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, create_pr=True, commit_message=TIEU_DE, commit_description=MO_TA)
    print("Pull Request:", c.pr_url)
    nhanh = api.dataset_info(REPO).sha
    print(f"main vẫn ở {nhanh[:10]} (không đổi: {nhanh == head}); file nằm ở nhánh PR, chờ chủ repo Merge.")


# ---------------- cập nhật 24/10/2026: bỏ hai cột JSON khỏi bảng sản phẩm (anh duyệt) ----------------
# Chỉ ghi đè 3 file (README, bảng sản phẩm, SHA256SUMS); bảng cặp và provenance không đổi. Từ chối chạy nếu trên HF có gì khác với bản đã đẩy lúc đầu.
CAP_NHAT = ("README.md", "description_products_vi_zh.parquet", "SHA256SUMS.txt")
KHONG_DOI = ("description_pairs_vi_zh_ctx.parquet", "provenance.json")
# Mỗi lần cập nhật: bảng sản phẩm phải đang là bản của lần trước trên HF; cột bảng mới phải đúng như mong đợi.
LAN = {
    "cap_nhat": {"sha_cu": "a3298cf0509028fb9d625902de13081c7d827a97007c4f4337d6a721acc4be0c", "size_cu": 5319430,   # lúc đầu, có hai cột JSON
                 "cot": ["product_id", "category", "so_cap", "do_tin_cay", "so_cap_khong_cao", "description_vi", "description_zh"],
                 "tieu_de": "Update data/description_vi_to_zh_cum: drop the two JSON columns from the products table (nhatanh)",
                 "mo_ta": "The products table keeps description_vi/zh; the attribute list is in the pairs table (group by product_id, sort by pair_id). "
                          "Updates README, products parquet and SHA256SUMS only."},
    # 25/10/2026 (anh chốt): thêm so_cap_thap ở cuối, các cột cũ giữ nguyên; cùng cột với bảng sản phẩm bộ 4
    "cap_nhat_2": {"sha_cu": "429697aca2d79e1adf510a188c8d73097079acd95bb0926e13b1108932a8b20b", "size_cu": 2529153,  # sau lần cập nhật 1
                   "cot": ["product_id", "category", "so_cap", "do_tin_cay", "so_cap_khong_cao", "description_vi", "description_zh", "so_cap_thap"],
                   "tieu_de": "Update data/description_vi_to_zh_cum: add so_cap_thap to the products table (nhatanh)",
                   "mo_ta": "New last column so_cap_thap = number of pairs labelled thap in the product (the other columns are unchanged), so the "
                            "products tables of data/tiki/descriptions/vi_to_zh_context/ and data/1688/descriptions/zh_to_vi_context/ have the same columns. "
                            "Updates README, products parquet and SHA256SUMS only."}}


def git_blob_sha1(p):
    d = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()


def hf_khop(f, p):
    """File HF f có đúng nội dung file máy p không (LFS: SHA256; file thường: git blob sha1)."""
    if getattr(f, "lfs", None):
        return f.lfs.sha256 == sha(p)
    return f.blob_id == git_blob_sha1(p)


def kiem_cap_nhat(api, lan="cap_nhat_2"):
    L = LAN[lan]
    tong = {ln.split()[1]: ln.split()[0] for ln in (SRC / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
    for ten, h in tong.items():
        assert sha(SRC / ten) == h, f"SHA256 lệch: {ten}"
    readme = (SRC / "README.md").read_text(encoding="utf-8")
    for ten, h in tong.items():
        assert h in readme and f"{(SRC / ten).stat().st_size:,}" in readme, f"README chưa ghi đúng SHA256 hoặc kích thước của {ten}"
    assert sha(SRC / "description_products_vi_zh.parquet") != L["sha_cu"], "bảng sản phẩm trên máy vẫn là bản cũ"
    import pyarrow.parquet as pq
    cot = pq.ParquetFile(SRC / "description_products_vi_zh.parquet").schema_arrow.names
    assert cot == L["cot"], cot
    SHA_SP_CU, SIZE_SP_CU = L["sha_cu"], L["size_cu"]
    info = api.dataset_info(REPO)
    paths = [DICH + f for f in (*CAP_NHAT, *KHONG_DOI)]
    tren = {f.path: f for f in api.get_paths_info(REPO, paths, repo_type="dataset", revision=info.sha, expand=True)}
    trong = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, path_in_repo=DICH.rstrip("/"), recursive=True))
    assert trong == sorted(paths), f"thư mục trên HF không đúng 5 file đã đẩy: {trong}"
    for ten in KHONG_DOI:     # hai file không đổi phải khớp bản máy (không ai sửa trên HF)
        assert hf_khop(tren[DICH + ten], SRC / ten), f"{ten} trên HF khác bản máy: có người đã sửa? DỪNG"
    sp = tren[DICH + "description_products_vi_zh.parquet"]
    assert sp.lfs.sha256 == SHA_SP_CU and sp.size == SIZE_SP_CU, "bảng sản phẩm trên HF không phải bản em đã đẩy lúc đầu: có người đã sửa? DỪNG"
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    print(f"Repo {REPO} @ {info.sha[:10]}: thư mục {DICH} có đúng 5 file đã đẩy; bảng cặp và provenance khớp bản máy; bảng sản phẩm trên HF đúng bản cũ.")
    print("Sẽ GHI ĐÈ 3 file:")
    for ten in CAP_NHAT:
        print(f"   {DICH}{ten}  ({tren[DICH + ten].size:,} -> {(SRC / ten).stat().st_size:,} byte)")
    return info.sha, co_san


def cap_nhat(_, lan="cap_nhat_2"):
    L = LAN[lan]
    token = (os.environ.get("HF_TOKEN") or getpass.getpass("Dán token HF có quyền GHI vào repo (gõ ẩn, Enter): ")).strip().strip("\"'")
    if token.startswith("HF_TOKEN="):
        token = token[len("HF_TOKEN="):].strip().strip("\"'")
    if not token.startswith("hf_") or len(token) < 30:
        sys.exit(f"DỪNG: chuỗi nhận được dài {len(token)} ký tự và không giống token HF (phải bắt đầu bằng hf_). Copy lại rồi chạy lại.")
    api = HfApi(token=token)
    print("Đăng nhập HF:", api.whoami()["name"])
    head, truoc = kiem_cap_nhat(api, lan)
    ops = [CommitOperationAdd(path_in_repo=DICH + f, path_or_fileobj=str(SRC / f)) for f in CAP_NHAT]
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head,
                          commit_message=L["tieu_de"], commit_description=L["mo_ta"])
    print("Commit:", c.commit_url)
    paths = [DICH + f for f in (*CAP_NHAT, *KHONG_DOI)]
    sau = {f.path: f for f in api.get_paths_info(REPO, paths, repo_type="dataset", expand=True)}
    for f in (*CAP_NHAT, *KHONG_DOI):
        assert hf_khop(sau[DICH + f], SRC / f), f"sau khi đẩy, {f} trên HF khác bản máy"
    toan = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)}
    for d, f in truoc.items():
        if d in paths or getattr(f, "size", None) is None:
            continue
        assert d in toan and toan[d].size == f.size, f"file khác bị đổi: {d}"
        if getattr(f, "lfs", None) and getattr(toan[d], "lfs", None):
            assert f.lfs.sha256 == toan[d].lfs.sha256, f"file khác bị đổi nội dung: {d}"
    print(f"OK: 3 file đã cập nhật và khớp bản máy; 2 file không đổi; {len(truoc) - len(paths)} mục khác của repo giữ nguyên.")


if __name__ == "__main__":
    # 25/10/2026: kiem_cap_nhat / cap_nhat mặc định là lần cập nhật 2 (thêm so_cap_thap); lần 1 đã đẩy xong (main 51540c9e18)
    {"kiem": kiem, "day": day, "pr": pr, "kiem_cap_nhat": kiem_cap_nhat, "cap_nhat": cap_nhat}[sys.argv[1]](HfApi())
