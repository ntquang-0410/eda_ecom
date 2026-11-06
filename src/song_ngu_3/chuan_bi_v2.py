# Bước 1 của bộ 3, bản v2 (anh chọn 23/10/2026: gộp 3 cách). Mỗi cụm (một sản phẩm, hoặc một phần nếu dài) gửi cho mô hình gồm:
#   1. TÊN SẢN PHẨM làm ngữ cảnh (không dịch, không công bố): title_vi_cleaned của tiki_eda, đã xoá tên shop và số điện thoại;
#   2. mục KHOÁ: tên trong từ điển tên đã duyệt (P6), giá trị lấy từ từ điển giá trị (P7), bảng có/không (P8), quy tắc thời hạn,
#      khối lượng, và các tuyến chép nguyên (thương hiệu, số đo, số, mã model, cỡ chữ) của bộ 2: ghi sẵn bản Trung (n_zh, v_zh),
#      mô hình KHÔNG dịch, KHÔNG xuất, chỉ dùng làm ngữ cảnh; lúc ghép bản cuối, mục khoá luôn lấy bản khoá;
#   3. BẢN NHÁP: các trường còn lại mang bản dịch đơn vị của bộ 2 (draft_n, draft_v, bản 20261023_1656); mô hình đọc cả sản phẩm, giữ
#      nguyên bản nháp nếu đúng, chỉ sửa khi ngữ cảnh cho thấy sai, sót tiếng Anh/tiếng Việt, hoặc đơn vị viết 'gr'. Cặp bộ 2 không có
#      bản dịch (cất vì không chọn được) thì không có nháp: mô hình dịch mới.
# Lọc riêng tư và chia phần y như v1 (cùng cụm, cùng mẫu 200 sản phẩm: kiểm bằng assert ở bước chạy).
import json
import re
import time

import pandas as pd

from song_ngu_2.rieng_tu import rieng_tu

from . import cau_hinh as C
from .chuan_bi import chia_phan, rut_mau, sha, ten_khac

GIA_TRI_KHOA = {"tu_dien_p7", "quy_tac_co_khong", "quy_tac_thoi_han", "quy_tac_khoi_luong", "quy_tac_ten_hang", "quy_tac_do_luong",
                "quy_tac_chep_nguyen", "quy_tac_ma_model", "quy_tac_co_chu"}
TEN_KHOA = {"tu_dien_p6"}
GR = re.compile(r"(?<=\d)\s*(?:gr|gram|grams)\b", re.I)     # '30gr', '500 GRAM', '100-110 gr' -> 'g' (người đọc Trung viết g hoặc 克)
SDT = re.compile(r"(?<!\d)(?:\+?84|0)\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}(?!\d)")   # 0912345678, 0912 345 678, +84 912.345.678
MAX_TIEU_DE = 200


def chuan_gr(s):
    """Đơn vị gam viết 'gr', 'gram' sau số đổi thành 'g' (chỉ áp cho giá trị KHOÁ thuộc tuyến số đo/chép nguyên)."""
    return GR.sub("g", s) if isinstance(s, str) else s


def lam_tieu_de(tieu_de, shop):
    """Tên sản phẩm làm ngữ cảnh: xoá tên shop (không phân biệt hoa thường, kể cả trong ngoặc vuông) và số điện thoại, gọn khoảng
    trắng, cắt ở MAX_TIEU_DE ký tự. Bộ lọc riêng tư bắt được gì thì bỏ hẳn tên (trả rỗng)."""
    t = tieu_de if isinstance(tieu_de, str) else ""
    if isinstance(shop, str) and len(shop.strip()) >= 4:
        t = re.sub(r"\[?\s*" + re.escape(shop.strip()) + r"\s*\]?", " ", t, flags=re.I)
    t = SDT.sub(" ", t)
    t = " ".join(t.split()).strip(" -|,")
    if rieng_tu("Tên sản phẩm", t):
        return ""
    return t if len(t) <= MAX_TIEU_DE else t[:MAX_TIEU_DE].rsplit(" ", 1)[0] + " ..."


def muc_v2(r):
    """Một mục của cụm v2 từ một dòng cặp đã ghép thông tin bộ 2. Trả về (mục JSON, danh sách trường mô hình phải xuất)."""
    m = {"i": None, "n": r.name_vi, "v": r.gia_tri_vi}
    can = []
    if r.cach_ten in TEN_KHOA and isinstance(r.ten_zh_don_vi, str):
        m["n_zh"] = r.ten_zh_don_vi
    else:
        if isinstance(r.ten_zh_don_vi, str):
            m["draft_n"] = r.ten_zh_don_vi
        can.append("n")
    if r.cach_chon in GIA_TRI_KHOA and isinstance(r.gia_tri_zh, str):
        m["v_zh"] = chuan_gr(r.gia_tri_zh) if r.cach_chon in ("quy_tac_do_luong", "quy_tac_chep_nguyen") else r.gia_tri_zh
    else:
        if isinstance(r.gia_tri_zh, str):
            m["draft_v"] = r.gia_tri_zh
        can.append("v")
    return m, can


def lap_cum_v2(B, tieu_de):
    """B: cặp đã lọc, đã ghép cach_ten, ten_zh_don_vi, gia_tri_zh, cach_chon của bộ 2. Một dòng một cụm."""
    rows = []
    for (pid, cat), g in B.sort_values(["product_id", "vi_tri"]).groupby(["product_id", "category"], sort=True):
        dong = list(g.itertuples(index=False))
        phan = chia_phan([(r.pair_id, r.name_vi, r.gia_tri_vi) for r in dong])
        k0 = 0
        for k, p in enumerate(phan):
            muc, can = [], {}
            for i, r in enumerate(dong[k0:k0 + len(p)], 1):
                m, c = muc_v2(r)
                m["i"] = i
                muc.append(m)
                if c:
                    can[str(i)] = c
            k0 += len(p)
            rows.append({"ma_cum": f"{pid}#{k}", "product_id": pid, "category": cat, "phan": k, "so_phan": len(phan),
                         "tieu_de": tieu_de.get(pid, ""), "muc_json": json.dumps(muc, ensure_ascii=False),
                         "can_json": json.dumps(can), "pair_ids": json.dumps([int(x) for x, _, _ in p]),
                         "ten_khac": ten_khac(phan, k), "so_muc": len(p), "so_truong_can": sum(len(c) for c in can.values()),
                         "so_ky_tu": sum(len(t) + len(v) for _, t, v in p),
                         "so_ky_tu_nhap": sum(len(m.get("draft_n", "")) + len(m.get("draft_v", "")) for m in muc)})
    return pd.DataFrame(rows)


def chay(thu_muc_ra=C.THU_MUC_RA, file_bang_cap=C.FILE_BANG_CAP, file_bo_2=C.FILE_BO_2_DAY_DU, file_don_vi=None,
         file_tiki=C.ROOT / "data" / "cleaned" / "tiki" / "tiki_eda.parquet", so_mau=None, in_ra=print):
    t0 = time.time()
    file_don_vi = file_don_vi or C.THU_MUC_BO_2 / f"don_vi_da_chon_{C.NHAN_BO_2}.parquet"
    B0 = pd.read_parquet(file_bang_cap)
    B = B0[B0.trang_thai_vao == "dung"]
    B = B[[rieng_tu(n, v) is None for n, v in zip(B.name_vi, B.gia_tri_vi)]]
    U = pd.read_parquet(file_don_vi, columns=["ma_don_vi", "loai", "cach_chon", "ban"])
    UT = U[U.loai == "ten"].set_index("ma_don_vi")
    D2 = pd.read_parquet(file_bo_2, columns=["pair_id", "gia_tri_zh", "cach_chon"])
    B = B.merge(D2, on="pair_id", how="left")
    B["cach_ten"] = B.ma_don_vi_ten.map(UT.cach_chon)
    B["ten_zh_don_vi"] = B.ma_don_vi_ten.map(UT.ban)
    T = pd.read_parquet(file_tiki, columns=["product_id", "title_vi_cleaned", "shop"])
    T["product_id"] = T.product_id.astype(str)
    tieu_de = {p: lam_tieu_de(t, s) for p, t, s in zip(T.product_id, T.title_vi_cleaned, T.shop)}
    B["product_id"] = B.product_id.astype(str)
    CUM = lap_cum_v2(B, tieu_de)
    mau = rut_mau(CUM.drop_duplicates("product_id")[["product_id", "category"]], so_mau)
    CUM["mau_thu"] = CUM.product_id.isin(mau)
    vao = thu_muc_ra / "vao_kaggle"
    vao.mkdir(parents=True, exist_ok=True)
    CUM.to_parquet(vao / "cum_can_dich_v2.parquet", index=False)
    m = CUM[CUM.mau_thu]
    muc = [x for s in CUM.muc_json for x in json.loads(s)]
    the = {"buoc": "song_ngu_3.chuan_bi_v2", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256_bang_cap": sha(file_bang_cap),
           "bo_2": file_bo_2.name, "cap": len(muc), "cum": int(len(CUM)), "san_pham": int(CUM.product_id.nunique()),
           "ten_khoa": sum("n_zh" in x for x in muc), "ten_nhap": sum("draft_n" in x for x in muc),
           "ten_dich_moi": sum("n_zh" not in x and "draft_n" not in x for x in muc),
           "gia_tri_khoa": sum("v_zh" in x for x in muc), "gia_tri_nhap": sum("draft_v" in x for x in muc),
           "gia_tri_dich_moi": sum("v_zh" not in x and "draft_v" not in x for x in muc),
           "gia_tri_khoa_doi_gr": int(sum(1 for c, z in zip(B.cach_chon, B.gia_tri_zh)
                                          if c in ("quy_tac_do_luong", "quy_tac_chep_nguyen") and isinstance(z, str) and GR.search(z))),
           "san_pham_khong_co_tieu_de": int(sum(1 for p in CUM.product_id.unique() if not tieu_de.get(p))),
           "mau_thu": {"san_pham": int(m.product_id.nunique()), "cum": int(len(m)), "cap": int(m.so_muc.sum()),
                       "truong_can_xuat": int(m.so_truong_can.sum())}}
    (thu_muc_ra / "the_chuan_bi_v2.json").write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    in_ra(f"v2: {the['cap']:,} cặp, {the['cum']:,} cụm | tên: khoá {the['ten_khoa']:,}, nháp {the['ten_nhap']:,}, dịch mới {the['ten_dich_moi']} | "
          f"giá trị: khoá {the['gia_tri_khoa']:,}, nháp {the['gia_tri_nhap']:,}, dịch mới {the['gia_tri_dich_moi']} | "
          f"không có tên sản phẩm {the['san_pham_khong_co_tieu_de']} ({time.time() - t0:.0f} giây)")
    in_ra(f"Mẫu thử: {the['mau_thu']}")
    return CUM, the
