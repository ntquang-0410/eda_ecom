# -*- coding: utf-8 -*-
"""Dựng thư mục công bố bộ song ngữ 3 (Tiki Việt -> Trung, dịch theo cụm, có ngữ cảnh) để đẩy lên HF: data/description_vi_to_zh_cum/.

Hai bảng dữ liệu cùng một bản dịch: bảng CẶP (một dòng một thuộc tính) và bảng SẢN PHẨM (một dòng một sản phẩm, cả description). Chọn cột công bố,
kiểm chặt (trùng, rỗng, nhãn, tổng cặp, riêng tư), làm sạch thẻ nguồn gốc (bỏ tên khoá API, câu gọi thử, đường dẫn Kaggle), ghi SHA256SUMS.
README.md (thẻ dữ liệu tiếng Anh) viết tay, nằm sẵn trong thư mục đích.

    .venv\\Scripts\\python.exe scripts/release/build_description_vi_to_zh_cum.py [nhãn bản, mặc định bản mới nhất]
"""
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_1" / "kaggle"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_3" / "kaggle"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_2.rieng_tu import rieng_tu  # noqa: E402
import dich_cum  # noqa: E402

SRC = ROOT / "data" / "processed" / "song_ngu_3"
DST = ROOT / "data" / "release" / "description_vi_to_zh_cum"
PAIRS, PRODUCTS, PROV = "description_pairs_vi_zh_ctx.parquet", "description_products_vi_zh.parquet", "provenance.json"
SHA_NGUON = "7af2a2620b8ca2def3b80762cb8f99196bfe7c64baeaeaa8b99d974df570d3bd"
SHA_BO_2 = "0177e2b233d260dc024e0abe3797e91f11ddab714557368eb46e0a7ee30e78fc"
COT_CAP = ["pair_id", "product_id", "category", "name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh", "do_tin_cay", "nguon_ban_dich"]
# 24/10/2026: bỏ hai cột JSON thuoc_tinh_vi/zh khỏi bảng sản phẩm (dư thừa với bảng cặp: nhóm theo product_id, sắp theo pair_id; chiếm 52,5% dung lượng;
# khóa "ten"/"gia_tri" tiếng Việt lọt vào dữ liệu fine-tune dạng JSON)
# 25/10/2026 (anh chốt): thêm so_cap_thap ở CUỐI (số cặp nhãn thấp, tính từ bảng cặp) cho cùng cột với bảng sản phẩm của bộ 4 (1688 cụm);
# các cột cũ giữ nguyên thứ tự, bảng cặp không đổi.
COT_SP = ["product_id", "category", "so_cap", "do_tin_cay", "so_cap_khong_cao", "description_vi", "description_zh", "so_cap_thap"]
NHAN = {"cao", "vua", "thap"}
NGUON = {"tu_dien_quy_tac", "giu_ban_dich_don_vi", "gemma_sua", "qwen_sua", "dich_moi"}
CAM = ("GEMINI_API_KEY_", "/kaggle/", "trinhnhatanh", "KGAT_", "theo_khoa", "goi_thu", "AIza")
RIENG_TU = [re.compile(p) for p in (r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){9,10}(?!\d)", r"(?<!\d)1[3-9]\d{9}(?!\d)", r"[\w.+-]+@[\w-]+\.[\w.]+",
                                    r"(?i)https?://|www\.|\b[\w-]+\.(?:com|vn|net|org)\b", r"(?i)\bzalo\b|facebook|\bhotline\b", "_x000D_")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lam_sach(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "theo_khoa":
                out["so_khoa"] = len(v)
            elif k in ("goi_thu", "dau_ra_cu", "loi_cuoi"):
                continue
            else:
                out[k] = lam_sach(v)
        return out
    if isinstance(x, list):
        return [lam_sach(v) for v in x]
    return x


def moi_nhat(mau):
    f = sorted(glob.glob(str(SRC / mau)))
    assert f, mau
    return Path(f[-1])


def main():
    DST.mkdir(parents=True, exist_ok=True)
    f_cap, f_sp = moi_nhat("cap_cum_*.parquet"), moi_nhat("san_pham_cum_*.parquet")
    f_tk = moi_nhat("tom_tat_cum_*.json")
    nhan = f_cap.stem.replace("cap_cum_", "")
    assert nhan in f_sp.stem and nhan in f_tk.stem, "ba file không cùng nhãn bản"
    C, S = pd.read_parquet(f_cap), pd.read_parquet(f_sp)
    if "so_cap_thap" not in S.columns:
        S["so_cap_thap"] = S.product_id.map(C[C.do_tin_cay == "thap"].groupby("product_id").size()).fillna(0).astype("int64")
    C, S = C[COT_CAP].copy(), S[COT_SP].copy()
    # ---- kiểm chặt
    assert C.pair_id.is_unique and S.product_id.is_unique
    assert not C.isna().any().any() and not S.isna().any().any()
    assert not (C[["name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh"]].apply(lambda s: s.str.strip().eq("")).any().any())
    assert set(C.do_tin_cay) <= NHAN and set(S.do_tin_cay) <= NHAN and set(C.nguon_ban_dich) <= NGUON
    assert int(S.so_cap.sum()) == len(C) and set(C.product_id) == set(S.product_id)
    g = C.groupby("product_id").do_tin_cay.apply(lambda s: {3: "cao", 2: "vua", 1: "thap"}[min({"cao": 3, "vua": 2, "thap": 1}[x] for x in s)])
    assert (S.set_index("product_id").do_tin_cay == g.reindex(S.product_id).values).all(), "nhãn sản phẩm khác mức thấp nhất của cặp"
    assert (S.set_index("product_id").so_cap_khong_cao == C[C.do_tin_cay != "cao"].groupby("product_id").size().reindex(S.product_id, fill_value=0).values).all()
    assert (S.set_index("product_id").so_cap_thap == C[C.do_tin_cay == "thap"].groupby("product_id").size().reindex(S.product_id, fill_value=0).values).all()
    assert ((S.so_cap_thap > 0) == (S.do_tin_cay == "thap")).all() and (S.so_cap_thap <= S.so_cap_khong_cao).all()
    assert sum(rieng_tu(n, v) is not None for n, v in zip(C.name_vi, C.gia_tri_vi)) == 0, "còn cặp bị bộ lọc riêng tư bắt"
    for p in RIENG_TU:      # quét TỪNG cột riêng (nối cột lại làm chữ số hai cột liền nhau ghép thành số điện thoại giả)
        for ten, s in (("gia_tri_vi", C.gia_tri_vi), ("gia_tri_zh", C.gia_tri_zh), ("name_vi", C.name_vi), ("name_zh", C.name_zh),
                       ("description_vi", S.description_vi), ("description_zh", S.description_zh)):
            n = int(s.str.contains(p.pattern, regex=True).sum())
            assert n == 0, f"quét riêng tư: {n} dòng cột {ten} khớp {p.pattern}"
    assert not C.name_zh.str.contains("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", case=False, regex=True).any(), "name_zh còn chữ Việt"
    # ---- ghi hai bảng
    C.to_parquet(DST / PAIRS, index=False)
    S.to_parquet(DST / PRODUCTS, index=False)
    # ---- thẻ nguồn gốc làm sạch
    tk = json.loads(f_tk.read_text(encoding="utf-8"))
    cb = json.loads((SRC / "the_chuan_bi_v2.json").read_text(encoding="utf-8"))
    gem = json.loads((SRC / "tu_kaggle_v2_that_gemma" / "ra_cum_v2" / "the_api_tuy_chon.json").read_text(encoding="utf-8"))
    qw = json.loads((SRC / "tu_kaggle_v2_that_qwen" / "ra_cum_v2" / "the_qwen3_14b.json").read_text(encoding="utf-8"))
    the = {"_note": "Provenance card of the build that produced the two tables of this folder. Keys are Vietnamese; see README.md, section 'Provenance file'. "
                    "API key names, test calls and Kaggle paths were removed before publishing.",
           "ban_dung": {"nhan": nhan, "bang_cap": PAIRS, "bang_san_pham": PRODUCTS},
           "nguon": {"file": "data/cleaned/tiki/tiki_eda.parquet", "sha256": SHA_NGUON,
                     "ban_dich_don_vi_dung_lam_nhap": {"thu_muc": "data/description_vi_to_zh/", "nhan_ban": "20261023_1656", "sha256_file_cap": SHA_BO_2}},
           "chuan_bi": {k: cb[k] for k in ("ngay_chay", "cap", "cum", "san_pham", "ten_khoa", "ten_nhap", "ten_dich_moi", "gia_tri_khoa", "gia_tri_nhap",
                                           "gia_tri_dich_moi", "gia_tri_khoa_doi_gr", "san_pham_khong_co_tieu_de", "mau_thu") if k in cb},
           "cau_lenh_he_thong_v2": dich_cum.HE_THONG_CUM_V2,
           "phien_ban_cau_lenh": dich_cum.PHIEN_BAN_CAU_LENH_V2,
           "chay": {"gemma_4": lam_sach(gem), "qwen3_14b": lam_sach(qw)},
           "ghep": lam_sach(tk)}
    txt = json.dumps(the, ensure_ascii=False, indent=1)
    for cam in CAM:
        assert cam not in txt, f"thẻ nguồn gốc còn chuỗi cấm: {cam}"
    (DST / PROV).write_text(txt + "\n", encoding="utf-8")
    tong = "".join(f"{sha(DST / f)}  {f}\n" for f in (PAIRS, PRODUCTS, PROV))
    (DST / "SHA256SUMS.txt").write_text(tong, encoding="ascii", newline="\n")
    sys.stdout.write(tong)
    print({f.name: f.stat().st_size for f in DST.iterdir()}, "| nhãn bản", nhan)


if __name__ == "__main__":
    main()
