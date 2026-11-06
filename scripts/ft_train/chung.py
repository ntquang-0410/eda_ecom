"""Hằng số và hàm dùng chung cho scripts/ft_train (fine-tune Qwen3-4B + LoRA trên hai bộ sau Judge).

Quy tắc khóa tập test (mọi script phải theo):
  - Chỉ đọc phép chia qua `doc_split()`: hàm này kiểm SHA256 của split.parquet và băm danh sách mã test với
    DU_LIEU_DONG_BANG.json, lệch là dừng.
  - Mẫu huấn luyện chỉ lấy từ `la_test == False`; mẫu test ghi riêng vào data/processed/ft_run/test_khoa/.
  - Không script nào được ghi lại split.parquet ngoài chia_tap.py.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parents[2]
DU_LIEU = GOC / "data" / "processed" / "bo_chon_hf"
FT_RUN = GOC / "data" / "processed" / "ft_run"
FILE_SPLIT = FT_RUN / "split.parquet"
FILE_DONG_BANG = FT_RUN / "DU_LIEU_DONG_BANG.json"
THU_MUC_TEST = FT_RUN / "test_khoa"            # mọi thứ dẫn xuất từ tập test (không đưa vào gói huấn luyện)

HF_REPO = "ntquang0410/zh-vie_ecom"
HF_THU_MUC = "data/LLM/bilingual_judged"
HF_COMMIT = "82aa9c9"
FILE_BO = {"cum": "bo_cum_chon_deepseek.parquet", "pair": "bo_pair_chon_deepseek.parquet"}
FILE_DANH_DAU = {"cum": "bo_cum_danh_dau.parquet", "pair": "bo_pair_danh_dau.parquet"}
SHA256_HF = {          # lấy từ manifest.json của HF (commit 82aa9c9)
    "bo_cum_chon_deepseek.parquet": "e0a625b2c1159133af24a8ae43c3ec4af4911ada68e10cf392920b8e49a20def",
    "bo_pair_chon_deepseek.parquet": "d4535c62ed1e3692f1563143d10c5c612aaec30d5e574eb9969ef003c627230d",
    "bo_cum_danh_dau.parquet": "945c441f213c0442c788110b836d8aca51f0b36fa7b1e70b87fc2957c38840c6",
    "bo_pair_danh_dau.parquet": "12b66f24e4bc3718e6c784a927c50c4f0a02c84c1079791bb96734a984bddc24",
}
TACH_DONG = {"cum": "; ", "pair": "\n"}        # ký tự nối các thuộc tính trong mô tả (cả nguồn lẫn đích)
HUONG = {"1688": "zh->vi", "tiki": "vi->zh"}
KY_HIEU_XUONG_DONG = " ⏎ "                      # thay "\n" khi đưa vào CSV gửi Gemini


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest()


def bam_ds_test(ids):
    """Băm danh sách mã test (không phụ thuộc thứ tự)."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def doc_bo(ten, kiem_hash=True):
    """Đọc bộ 'cum' hoặc 'pair', thêm cột huong, nguon_title, dich_title, nguon_mo_ta, dich_mo_ta."""
    path = DU_LIEU / FILE_BO[ten]
    if kiem_hash and sha256_file(path) != SHA256_HF[FILE_BO[ten]]:
        raise RuntimeError(f"sai_hash_du_lieu:{path}")
    df = pd.read_parquet(path)
    z = df.source.eq("1688")
    df["huong"] = df.source.map(HUONG)
    df["nguon_title"] = df.title_zh.where(z, df.title_vi)
    df["dich_title"] = df.title_vi.where(z, df.title_zh)
    df["nguon_mo_ta"] = df.description_zh.where(z, df.description_vi)
    df["dich_mo_ta"] = df.description_vi.where(z, df.description_zh)
    return df


def doc_split():
    """Đọc phép chia đã khóa. Dừng nếu file bị sửa hoặc danh sách test không khớp bản ghi đóng băng."""
    if not FILE_SPLIT.exists() or not FILE_DONG_BANG.exists():
        raise RuntimeError("chua_co_split: chạy scripts/ft_train/chia_tap.py trước")
    ghi = json.loads(FILE_DONG_BANG.read_text(encoding="utf-8"))["split"]
    if sha256_file(FILE_SPLIT) != ghi["sha256_file"]:
        raise RuntimeError("split.parquet_bi_thay_doi")
    sp = pd.read_parquet(FILE_SPLIT)
    if bam_ds_test(sp.product_id[sp.la_test]) != ghi["sha256_ds_test"]:
        raise RuntimeError("ds_test_khong_khop")
    assert sp.product_id.is_unique
    assert (sp.fold5[sp.la_test] == -1).all() and sp.fold5[~sp.la_test].between(0, 4).all()
    return sp


def kiem_khong_giao(ids_huan_luyen, ids_test):
    """Dừng nếu một mã sản phẩm nằm ở cả hai phía."""
    giao = set(map(str, ids_huan_luyen)) & set(map(str, ids_test))
    if giao:
        raise RuntimeError(f"RO_RI_TEST: {len(giao)} mã test nằm trong dữ liệu huấn luyện, ví dụ {sorted(giao)[:5]}")
