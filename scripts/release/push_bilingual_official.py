# -*- coding: utf-8 -*-
"""Đẩy bộ song ngữ chính thức (thư mục do gop_tiki_vao_1688.py ghi) lên HF dataset ntquang0410/zh-vie_ecom, thư mục data/bilingual_official/ (cổng G5).

Chỉ THÊM file mới trong MỘT commit, không ghi đè: dừng nếu thư mục đích đã có bất kỳ file nào; commit gắn parent_commit = phiên bản vừa kiểm (có người đẩy chen
giữa thì HF từ chối, chạy lại); đẩy xong đối chiếu từng file với bản trên máy và kiểm mọi mục khác của repo giữ nguyên. Không đụng root README, data/1688, data/tiki.

    <python> scripts/release/push_bilingual_official.py kiem [--thu-muc DIR]   # chỉ kiểm, không đẩy, không cần token
    <python> scripts/release/push_bilingual_official.py day  [--thu-muc DIR]   # đẩy; token ghi lấy từ HF_TOKEN hoặc gõ ẩn khi được hỏi (không dán token vào chat)
    <python> scripts/release/push_bilingual_official.py pr   [--thu-muc DIR]   # mở Pull Request nếu token không ghi thẳng được
Mặc định DIR = data/release/bilingual_official (cạnh thư mục scripts). Chạy bằng Python có huggingface_hub, pandas, pyarrow; máy ít RAM đặt OPENBLAS_NUM_THREADS=1.
"""
import argparse
import getpass
import hashlib
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi

REPO = "ntquang0410/zh-vie_ecom"
DICH = "data/bilingual_official/"
_AQ = Path(__file__).resolve()
ROOT = _AQ.parents[2] if len(_AQ.parents) > 2 else _AQ.parent          # chạy được ở bất kỳ thư mục nào
COT = ["product_id", "source", "category", "title_zh", "title_vi", "description_zh", "description_vi", "title_nguon_dich", "desc_nguon_dich"]
FILES = ["bilingual_official_context.parquet", "bilingual_official_pair.parquet", "excluded_products.parquet", "selection_report.json", "manifest.json", "README.md",
         "SHA256SUMS.txt"]
TIEU_DE = "Add data/bilingual_official: 1688 and Tiki merged into one nine-column Chinese-Vietnamese set (title and description), with the list of excluded products"
MO_TA = ("Two files with the same products (description translated per product with context, and attribute by attribute), plus excluded_products.parquet with the reason for "
         "every product left out, selection_report.json, manifest.json, SHA256SUMS.txt and README.md. Adds only new files under data/bilingual_official/.")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kiem_may(src):
    for n in FILES:
        assert (src / n).exists(), f"thiếu file: {src / n}"
    ghi = {ln.split()[1]: ln.split()[0] for ln in (src / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln.strip()}
    assert set(ghi) == set(FILES) - {"SHA256SUMS.txt"}, f"SHA256SUMS ghi sai danh sách file: {sorted(ghi)}"
    for n, h in ghi.items():
        assert sha(src / n) == h, f"SHA256 lệch: {n}"
    for n in FILES[:2]:
        assert pq.ParquetFile(src / n).schema_arrow.names == COT, f"{n}: cột sai"
    a, b = (pq.ParquetFile(src / n).metadata.num_rows for n in FILES[:2])
    assert a == b > 0, f"hai bản khác số dòng: {a} và {b}"
    import json
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    assert man["cot"] == COT and man["file"][FILES[0]]["dong"] == a, "manifest không khớp file"
    dv = json.loads((src / "selection_report.json").read_text(encoding="utf-8"))["dau_van_tay"]
    return a, dv


def kiem(api, src):
    dong, dv = kiem_may(src)
    info = api.dataset_info(REPO)
    co_san = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, recursive=True)}
    trong = sorted(p for p in co_san if p.startswith(DICH) or p == DICH.rstrip("/"))
    assert not trong, f"DỪNG: {DICH} đã có trên HF, không ghi đè: {trong}"
    print(f"Repo {REPO} @ {info.sha[:10]}: {len(co_san)} mục; {DICH} chưa có. Trên máy: 2 bản × {dong:,} dòng, SHA256SUMS khớp, cột đúng 9 cột.")
    print(f"Dấu vân tay danh sách giữ: {dv['giu_sha256']}\n(so với kết quả của người kia chạy cùng đầu vào: phải giống hệt)")
    print(f"Sẽ THÊM {len(FILES)} file trong MỘT commit:")
    for n in FILES:
        print(f"   {DICH}{n}  ({(src / n).stat().st_size:,} byte)")
    return info.sha, co_san


def lay_token():
    token = (os.environ.get("HF_TOKEN") or getpass.getpass("Dán token HF có quyền GHI vào repo (gõ ẩn, Enter): ")).strip().strip("\"'")
    if token.startswith("HF_TOKEN="):
        token = token[len("HF_TOKEN="):].strip().strip("\"'")
    if not token.startswith("hf_") or len(token) < 30:
        sys.exit(f"DỪNG: chuỗi nhận được dài {len(token)} ký tự và không giống token HF (phải bắt đầu bằng hf_). Copy lại rồi chạy lại.")
    return token


def day(api, src, pr=False):
    head, truoc = kiem(api, src)
    ops = [CommitOperationAdd(path_in_repo=DICH + n, path_or_fileobj=str(src / n)) for n in FILES]
    if pr:
        c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, create_pr=True, commit_message=TIEU_DE, commit_description=MO_TA)
        print("Pull Request:", c.pr_url)
        return
    c = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, parent_commit=head, commit_message=TIEU_DE, commit_description=MO_TA)
    print("Commit:", c.commit_url)
    sau = {f.path: f for f in api.get_paths_info(REPO, [DICH + n for n in FILES], repo_type="dataset", expand=True)}
    for n in FILES:
        f = sau[DICH + n]
        khop = f.lfs.sha256 == sha(src / n) if getattr(f, "lfs", None) else f.blob_id == hashlib.sha1(b"blob %d\0" % (src / n).stat().st_size + (src / n).read_bytes()).hexdigest()
        assert khop, f"sau khi đẩy, {n} trên HF khác bản máy"
    toan = {f.path: f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)}
    for d, f in truoc.items():
        if getattr(f, "size", None) is None:
            continue
        assert d in toan and toan[d].size == f.size, f"file khác bị đổi: {d}"
        if getattr(f, "lfs", None) and getattr(toan[d], "lfs", None):
            assert f.lfs.sha256 == toan[d].lfs.sha256, f"file khác bị đổi nội dung: {d}"
    print(f"OK: {len(FILES)} file mới có mặt và khớp bản máy; {len(truoc)} mục cũ của repo giữ nguyên.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("che_do", choices=["kiem", "day", "pr"])
    ap.add_argument("--thu-muc", type=Path, default=ROOT / "data" / "release" / "bilingual_official")
    a = ap.parse_args()
    if a.che_do == "kiem":
        kiem(HfApi(), a.thu_muc)
    else:
        api = HfApi(token=lay_token())
        print("Đăng nhập HF:", api.whoami()["name"])
        day(api, a.thu_muc, pr=a.che_do == "pr")


if __name__ == "__main__":
    main()
