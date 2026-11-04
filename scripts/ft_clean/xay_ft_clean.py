"""Bước 0-4 chuẩn bị dữ liệu fine-tune từ bộ chính thức v2 (bản cụm). Kết quả ra data/processed/ft_clean/.

Không sửa nội dung, không xóa dòng khỏi bảng ứng viên; bộ chính thức trên HF và bản trong data/snapshot giữ nguyên.
Chạy:  python -I -X utf8 scripts/ft_clean/xay_ft_clean.py            (bản thật)
       python -I -X utf8 scripts/ft_clean/xay_ft_clean.py --vao X.parquet --ra DIR --bo-qua-sha   (thử)"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import luat_ft as L  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
FILE_V2 = GOC / "data" / "snapshot" / "bilingual_official_v2" / "bilingual_official_context.parquet"
REVIEW_V2 = GOC / "data" / "snapshot" / "bilingual_official_v2" / "review_queue.parquet"
RA = GOC / "data" / "processed" / "ft_clean"
SHA_V2 = "8fd10827c9e21ab2cffc645365bf5bfd22ab387c1896f5ad2cfdb90ed1129975"   # manifest.json của commit 4a23574
SO_DONG_V2 = 25956
TRAN_KHUON = 200        # chọn 30/10/2026 từ phân bố: cắt 279 dòng Tiki ở các khuôn > 200 dòng; 1688 không bị ảnh hưởng
THU_TU_LY_DO = ["ro_loi_nhac_title", "so_doan_lech", "han_sot_ban_viet", "vuot_tran_khuon"]
COT_GOC = ["product_id", "source", "category", "title_zh", "title_vi", "description_zh", "description_vi",
           "title_nguon_dich", "desc_nguon_dich"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for kh in iter(lambda: f.read(1 << 20), b""):
            h.update(kh)
    return h.hexdigest()


def gan_co(df, review_ids=()):
    """Cột cờ cứng + mềm + khuôn. Không đổi nội dung."""
    d = df.copy()
    d["huong_goc"] = d.source.map({"1688": "zh->vi", "tiki": "vi->zh"})
    d["co_ro_loi_nhac_title"] = d.title_vi.map(L.ro_loi_nhac)
    lech = [L.can_hang(z, v) for z, v in zip(d.description_zh, d.description_vi)]
    d["co_so_doan_lech"] = [x[0] for x in lech]
    d["so_doan_khac_so"] = [x[1] for x in lech]                      # None khi số đoạn lệch
    d["can_hang_chac"] = [(not a) and b == 0 for a, b in lech]
    d["han_sot_ban_viet"] = [L.han_sot(s, v) for s, v in zip(d.source, d.description_vi)]
    d["mem_latin_la_ban_trung"] = [L.latin_la(s, z, v) for s, z, v in zip(d.source, d.description_zh, d.description_vi)]
    d["mem_viet_sot_title_zh"] = [L.viet_sot_title_zh(s, t) for s, t in zip(d.source, d.title_zh)]
    d["mem_v2_review"] = d.product_id.isin(set(review_ids))
    d["khuon_id"] = [L.khuon_id(s, z, v) for s, z, v in zip(d.source, d.description_zh, d.description_vi)]
    d["nhom_mo_ta_id"] = [L.nhom_mo_ta_id(z, v) for z, v in zip(d.description_zh, d.description_vi)]
    d["nhom_mo_ta_size"] = d.groupby("nhom_mo_ta_id").product_id.transform("size")
    return d


def loai_tuan_tu(d, tran=TRAN_KHUON):
    """Loại theo thứ tự THU_TU_LY_DO; lý do = lý do đầu tiên. Trần khuôn chỉ tính trên các dòng còn lại sau 3 luật đầu."""
    d = d.copy()
    ly_do = pd.Series([None] * len(d), index=d.index, dtype=object)
    for ten, mask in [("ro_loi_nhac_title", d.co_ro_loi_nhac_title), ("so_doan_lech", d.co_so_doan_lech),
                      ("han_sot_ban_viet", d.han_sot_ban_viet > 0)]:
        ly_do[ly_do.isna() & mask] = ten
    con = d[ly_do.isna()]
    thu_tu = con.assign(_k=con.product_id.map(L.khoa_thu_tu)).sort_values("_k").groupby("khuon_id").cumcount()
    d["khuon_thu_tu"] = thu_tu.reindex(d.index)                    # NaN với dòng đã bị loại sớm hơn
    d["khuon_size_con_lai"] = con.groupby("khuon_id").product_id.transform("size").reindex(d.index)
    ly_do[ly_do.isna() & (d.khuon_thu_tu >= tran)] = "vuot_tran_khuon"
    d["ly_do_loai_ft"] = ly_do
    d["giu_ft"] = d.ly_do_loai_ft.isna()
    return d


def pheu(d):
    out = {"vao": len(d), "theo_buoc": []}
    con = d
    for ten in THU_TU_LY_DO:
        n = int((con.ly_do_loai_ft == ten).sum())
        out["theo_buoc"].append({"ly_do": ten, "loai": n,
                                 "theo_nguon": {k: int(v) for k, v in con[con.ly_do_loai_ft == ten].source.value_counts().items()}})
    giu = d[d.giu_ft]
    out["con_lai"] = len(giu)
    out["con_lai_theo_nguon"] = {k: int(v) for k, v in giu.source.value_counts().items()}
    out["con_lai_theo_huong"] = {k: int(v) for k, v in giu.huong_goc.value_counts().items()}
    out["con_lai_theo_nhom_hang"] = {k: int(v) for k, v in giu.category.value_counts().items()}
    out["can_hang_chac_trong_con_lai"] = {"so_dong": int(giu.can_hang_chac.sum()),
                                           "ti_le": round(float(giu.can_hang_chac.mean()), 4)}
    out["co_mem_trong_con_lai"] = {c: int((giu[c] > 0).sum()) for c in
                                   ["mem_latin_la_ban_trung", "mem_viet_sot_title_zh", "mem_v2_review"]}
    out["nhom_mo_ta_trung_trong_con_lai"] = {"so_nhom_co_tu_2_dong": int((giu.groupby("nhom_mo_ta_id").size() >= 2).sum()),
                                              "so_dong_thuoc_nhom_trung": int((giu.nhom_mo_ta_size >= 2).sum())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--vao", default=str(FILE_V2))
    p.add_argument("--ra", default=str(RA))
    p.add_argument("--bo-qua-sha", action="store_true", help="chỉ để thử với dữ liệu giả")
    a = p.parse_args(argv)
    vao, ra = Path(a.vao), Path(a.ra)
    if not a.bo_qua_sha:
        h = sha256(vao)
        assert h == SHA_V2, f"SHA256 không khớp manifest v2: {h}"
    df = pd.read_parquet(vao)
    assert list(df.columns) == COT_GOC, df.columns.tolist()
    assert df.product_id.is_unique
    if not a.bo_qua_sha:
        assert len(df) == SO_DONG_V2, len(df)
    review = pd.read_parquet(REVIEW_V2).product_id if (REVIEW_V2.exists() and not a.bo_qua_sha) else []
    d = loai_tuan_tu(gan_co(df, review))
    ra.mkdir(parents=True, exist_ok=True)
    d.to_parquet(ra / "ft_candidate_v2.parquet", index=False)
    cot_giu = COT_GOC + ["huong_goc", "khuon_id", "nhom_mo_ta_id", "can_hang_chac"] + \
        [c for c in d.columns if c.startswith("mem_")]
    d[d.giu_ft][cot_giu].to_parquet(ra / "ft_buoc4_giu.parquet", index=False)
    d[~d.giu_ft][COT_GOC + ["ly_do_loai_ft"]].to_parquet(ra / "ft_buoc4_loai.parquet", index=False)
    f = pheu(d)
    f["tieu_chi"] = {"tran_khuon": TRAN_KHUON, "thu_tu_ly_do": THU_TU_LY_DO, "nguon": str(vao.name), "sha256_nguon": SHA_V2,
                     "ghi_chu": "co cung (co_*) loai dong; co mem (mem_*) chi ghi lai. Khong sua noi dung."}
    (ra / "pheu_buoc0_4.json").write_text(json.dumps(f, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(f, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
