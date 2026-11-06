# Bước 1 của bộ 4 (laptop): gom các cặp thuộc tính 1688 theo sản phẩm (đúng thứ tự trên trang, vi_tri) thành cụm để hiệu đính một lần.
# Mỗi mục của cụm gửi cho mô hình gồm:
#   - "n", "v": tên và giá trị TIẾNG TRUNG (giá trị đã đổi dấu che sang dạng Việt [SĐT], [LIÊN HỆ]... như lúc dịch bộ 1);
#   - "n_vi" / "v_vi": bản KHOÁ (tên trong từ điển P6/P6b, tên không có chữ Hán, giá trị theo bảng có/không, từ điển P7 và quy tắc của bộ 1,
#     và giá trị DÀI hơn MAX_ZH_XEM chữ: giữ bản bộ 1, anh chốt 24/10/2026): mô hình chỉ đọc làm ngữ cảnh, không xuất, không sửa;
#   - "draft_n" / "draft_v": bản NHÁP (bản dịch đơn vị của bộ 1): mô hình giữ nguyên nếu đúng, chỉ sửa khi ngữ cảnh cho thấy sai;
#   - không có cả hai: cặp bộ 1 cất riêng vì không chọn được bản dịch: mô hình dịch mới.
# Cặp đưa vào: mọi cặp trang_thai_vao = "dung" của bộ 1 trừ thương hiệu vô nghĩa ("是"). Cặp bị loại/cất ở bước làm sạch (riêng tư, rỗng,
# tên hãng dài chờ xem) KHÔNG đưa vào. Giá trị dài mà không có bản bộ 1 thì không có bản nào dùng được: bỏ, đếm ở the_chuan_bi.
# Kiểm chặt: bản khoá và bản nháp dựng lại từ bảng đơn vị phải KHỚP TỪNG CHỮ với bản công bố của bộ 1 (bo_song_ngu_1_day_du).
import hashlib
import json
import re
import shutil
import time

import pandas as pd

from song_ngu_1.dinh_tuyen import van_ban_gui
from song_ngu_1.van_ban import co_han
from song_ngu_3.chuan_bi import rut_mau

from . import cau_hinh as C

SDT_ZH = re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d{9}(?!\d)|(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)|\d{7,}")
URL_EMAIL = re.compile(r"(?i)https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lam_tieu_de(t, shop):
    """Tên sản phẩm làm ngữ cảnh (không dịch, không công bố): xoá tên shop, số điện thoại/số dài, URL, email; cắt ở MAX_TIEU_DE ký tự."""
    t = t if isinstance(t, str) else ""
    if isinstance(shop, str) and len(shop.strip()) >= 2:
        t = t.replace(shop.strip(), " ")
    t = URL_EMAIL.sub(" ", SDT_ZH.sub(" ", t))
    t = " ".join(t.split()).strip(" -|,，")
    return t[:C.MAX_TIEU_DE]


def trong_so(m):
    """Trọng số của một mục để chia phần: chữ Trung (tên + giá trị) + nửa số ký tự Việt (bản khoá, bản nháp)."""
    zh = len(m["n"]) + len(m["v"])
    vi = sum(len(m.get(k, "")) for k in ("n_vi", "v_vi", "draft_n", "draft_v"))
    return zh + vi / 2


def chia_phan(muc, toi_da=C.MAX_TRONG_SO_PHAN):
    """muc: danh sách mục theo thứ tự trên trang. Chia thành các phần liên tiếp có tổng trọng số không quá toi_da; một mục một mình đã quá
    thì đi riêng một phần. Không đổi thứ tự, không bỏ mục."""
    phan, cur, n = [], [], 0.0
    for m in muc:
        d = trong_so(m)
        if cur and n + d > toi_da:
            phan.append(cur)
            cur, n = [], 0.0
        cur.append(m)
        n += d
    if cur:
        phan.append(cur)
    return phan


def ten_khac(phan, k, toi_da=C.MAX_KY_TU_TEN_KHAC):
    if len(phan) == 1:
        return ""
    ten = list(dict.fromkeys(m["n"] for j, p in enumerate(phan) if j != k for m in p))
    s = "; ".join(ten)
    return s if len(s) <= toi_da else s[:toi_da].rsplit("; ", 1)[0] + "; ..."


def lap_cap(B, U):
    """Một dòng một cặp đưa vào: chữ Trung gửi đi, bản khoá hoặc bản nháp của tên và giá trị, nguồn của từng trường."""
    UT = U[U.loai == "ten"].set_index("ma_don_vi")
    UG = U[U.loai == "gia_tri"].set_index("ma_don_vi")
    rows = []
    for r in B.itertuples(index=False):
        d = {"pair_id": int(r.pair_id), "product_id": str(r.product_id), "category": r.category, "vi_tri": int(r.vi_tri),
             "nhom_trung": int(r.nhom_trung), "name_zh": r.name_zh, "gia_tri_zh": r.gia_tri_zh, "n_gui": r.name_zh,
             "v_gui": van_ban_gui(r.gia_tri_zh), "n_khoa": None, "n_nhap": None, "v_khoa": None, "v_nhap": None}
        # tên
        if not co_han(r.name_zh):
            d["n_khoa"], d["nguon_ten_vao"] = r.name_zh, "chep_nguyen"
        else:
            t = UT.loc[r.ma_don_vi_ten]
            if t.cach_chon in C.TEN_KHOA:
                d["n_khoa"], d["nguon_ten_vao"] = t.ban, t.cach_chon
            elif isinstance(t.ban, str):
                d["n_nhap"], d["nguon_ten_vao"] = t.ban, "nhap"
            else:
                d["nguon_ten_vao"] = "moi"
        # giá trị
        if r.dinh_tuyen != "mo_hinh":
            d["v_khoa"], d["nguon_gt_vao"] = r.gia_tri_vi_quy_tac, f"quy_tac_{r.dinh_tuyen}"
        else:
            g = UG.loc[r.ma_don_vi_gia_tri]
            if g.cach_chon in C.GIA_TRI_KHOA:
                d["v_khoa"], d["nguon_gt_vao"] = g.ban, g.cach_chon
            elif isinstance(g.ban, str) and len(r.gia_tri_zh) > C.MAX_ZH_XEM:
                d["v_khoa"], d["nguon_gt_vao"] = g.ban, "dai_giu_bo_1"
            elif isinstance(g.ban, str):
                d["v_nhap"], d["nguon_gt_vao"] = g.ban, "nhap"
            elif len(r.gia_tri_zh) > C.MAX_ZH_XEM:
                d["nguon_gt_vao"] = "dai_khong_co_ban"
            else:
                d["nguon_gt_vao"] = "moi"
        rows.append(d)
    return pd.DataFrame(rows)


def kiem_khop_bo_1(P, D):
    """Bản khoá/nháp dựng lại phải khớp từng chữ với bản công bố của bộ 1 trên mọi cặp bộ 1 đã công bố; cặp không công bố phải đúng là
    cặp không có bản (tên hoặc giá trị 'moi'). Sai là dừng."""
    X = P.merge(D[["pair_id", "name_vi", "gia_tri_vi", "cach_chon"]], on="pair_id", how="left")
    co = X.name_vi.notna()
    ten = X.n_khoa.fillna(X.n_nhap)
    gt = X.v_khoa.fillna(X.v_nhap)
    sai_ten = int((co & (ten != X.name_vi)).sum())
    sai_gt = int((co & (gt != X.gia_tri_vi)).sum())
    assert sai_ten == 0 and sai_gt == 0, f"bản dựng lại khác bản công bố bộ 1: {sai_ten} tên, {sai_gt} giá trị"
    khoa_sai = int((co & X.v_khoa.notna() & X.nguon_gt_vao.isin(C.GIA_TRI_KHOA) & (X.nguon_gt_vao != X.cach_chon)).sum())
    assert khoa_sai == 0, f"{khoa_sai} giá trị khoá có cach_chon khác bộ 1"
    khong_cb = X[~co]
    assert ((khong_cb.nguon_ten_vao == "moi") | khong_cb.nguon_gt_vao.isin(["moi", "dai_khong_co_ban"])).all(), \
        "có cặp không công bố ở bộ 1 mà vẫn đủ bản tên và giá trị"
    assert set(D.pair_id) <= set(P.pair_id), "thiếu cặp bộ 1 đã công bố"
    return {"cap_bo_1_cong_bo": int(co.sum()), "cap_bo_1_cat_rieng_dua_vao": int((~co).sum())}


def muc_cua(r, i):
    m = {"i": i, "n": r.n_gui, "v": r.v_gui}
    can = []
    if isinstance(r.n_khoa, str):
        m["n_vi"] = r.n_khoa
    else:
        if isinstance(r.n_nhap, str):
            m["draft_n"] = r.n_nhap
        can.append("n")
    if isinstance(r.v_khoa, str):
        m["v_vi"] = r.v_khoa
        if len(m["v"]) > C.MAX_KHOA_HIEN or len(r.v_khoa) > 2 * C.MAX_KHOA_HIEN:     # chỉ ngữ cảnh: hiện phần đầu
            m["v"] = m["v"][:C.MAX_KHOA_HIEN] + (" …" if len(m["v"]) > C.MAX_KHOA_HIEN else "")
            m["v_vi"] = r.v_khoa[:2 * C.MAX_KHOA_HIEN] + (" …" if len(r.v_khoa) > 2 * C.MAX_KHOA_HIEN else "")
    else:
        if isinstance(r.v_nhap, str):
            m["draft_v"] = r.v_nhap
        can.append("v")
    return m, can


def lap_cum(P, tieu_de):
    """P: bảng cặp đã bỏ cặp không dùng được. Một dòng một cụm (phần)."""
    rows, vi_tri_muc = [], {}
    for (pid, cat), g in P.sort_values(["product_id", "vi_tri", "pair_id"]).groupby(["product_id", "category"], sort=True):
        dong = list(g.itertuples(index=False))
        phan = chia_phan([muc_cua(r, 0)[0] for r in dong])
        k0 = 0
        for k, p in enumerate(phan):
            muc, can, pids = [], {}, []
            for i, r in enumerate(dong[k0:k0 + len(p)], 1):
                m, c = muc_cua(r, i)
                muc.append(m)
                pids.append(int(r.pair_id))
                vi_tri_muc[int(r.pair_id)] = (f"{pid}#{k}", i)
                if c:
                    can[str(i)] = c
            k0 += len(p)
            rows.append({"ma_cum": f"{pid}#{k}", "product_id": pid, "category": cat, "phan": k, "so_phan": len(phan),
                         "tieu_de": tieu_de.get(pid, ""), "muc_json": json.dumps(muc, ensure_ascii=False), "can_json": json.dumps(can),
                         "pair_ids": json.dumps(pids), "ten_khac": ten_khac(phan, k), "so_muc": len(muc),
                         "so_truong_can": sum(len(c) for c in can.values()),
                         "so_ky_tu": sum(len(m["n"]) + len(m["v"]) for m in muc),
                         "so_ky_tu_nhap": sum(len(m.get("draft_n", "")) + len(m.get("draft_v", "")) for m in muc),
                         "so_ky_tu_can": sum(len(muc[int(i) - 1]["n" if f == "n" else "v"]) for i, fs in can.items() for f in fs),
                         "trong_so": round(sum(trong_so(m) for m in muc), 1)})
    return pd.DataFrame(rows), vi_tri_muc


def chay(thu_muc_ra=C.THU_MUC_RA, so_mau=None, in_ra=print):
    t0 = time.time()
    B0 = pd.read_parquet(C.FILE_BANG_CAP)
    B = B0[(B0.trang_thai_vao == "dung") & ~((B0.dinh_tuyen == "ten_hang") & B0.gia_tri_zh.str.strip().isin(C.TEN_HANG_VO_NGHIA))]
    assert not (B.dinh_tuyen == "ten_hang_dai").any(), "tên hãng dài (chờ xem) lẽ ra đã bị loại ở trang_thai_vao"
    U = pd.read_parquet(C.FILE_DON_VI, columns=["ma_don_vi", "loai", "ban", "cach_chon"])
    D = pd.read_parquet(C.FILE_BO_1_DAY_DU, columns=["pair_id", "name_vi", "gia_tri_vi", "cach_chon"])
    P = lap_cap(B, U)
    khop = kiem_khop_bo_1(P, D)
    bo = P.nguon_gt_vao == "dai_khong_co_ban"
    P_bo, P = P[bo], P[~bo].copy()
    T = pd.read_parquet(C.FILE_SAN_PHAM, columns=["product_id", "title_zh", "shop"])
    T["product_id"] = T.product_id.astype(str)
    tieu_de = {p: lam_tieu_de(t, s) for p, t, s in zip(T.product_id, T.title_zh, T.shop)}
    CUM, vi_tri_muc = lap_cum(P, tieu_de)
    P["ma_cum"] = [vi_tri_muc[p][0] for p in P.pair_id]
    P["i"] = [vi_tri_muc[p][1] for p in P.pair_id]
    mau = rut_mau(CUM.drop_duplicates("product_id")[["product_id", "category"]], so_mau or C.SO_SAN_PHAM_THU, C.HAT_GIONG)
    CUM["mau_thu"] = CUM.product_id.isin(mau)
    # ---- kiểm chặt
    assert CUM.ma_cum.is_unique and P.pair_id.is_unique
    tat_ca = [x for s in CUM.pair_ids for x in json.loads(s)]
    assert sorted(tat_ca) == sorted(P.pair_id.tolist()), "cụm không phủ đúng mỗi cặp một lần"
    for s, c in zip(CUM.muc_json, CUM.can_json):
        muc, can = json.loads(s), json.loads(c)
        assert [m["i"] for m in muc] == list(range(1, len(muc) + 1))
        for m in muc:
            assert ("n_vi" in m) != ("n" in can.get(str(m["i"]), [])) and ("v_vi" in m) != ("v" in can.get(str(m["i"]), []))
            assert not ("n_vi" in m and "draft_n" in m) and not ("v_vi" in m and "draft_v" in m)
    # ---- ghi
    vao = thu_muc_ra / "vao_kaggle"
    vao.mkdir(parents=True, exist_ok=True)
    CUM.to_parquet(vao / "cum_can_dich_zv.parquet", index=False)
    (vao / "cau_hinh_kaggle.json").write_text(json.dumps({"huong": "zh_vi", "loai": "cum_zv"}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    goc = C.ROOT / "src"
    shutil.copyfile(goc / "song_ngu_1" / "kaggle" / "dich.py", vao / "dich.py")
    shutil.copyfile(goc / "song_ngu_4" / "kaggle" / "dich_cum_zv.py", vao / "dich_cum_zv.py")
    P.to_parquet(thu_muc_ra / "cap_vao.parquet", index=False)
    m = CUM[CUM.mau_thu]
    can_goi = CUM[CUM.so_truong_can > 0]
    the = {"buoc": "song_ngu_4.chuan_bi", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"),
           "nguon": {"bang_cap": {"file": C.FILE_BANG_CAP.name, "sha256": sha(C.FILE_BANG_CAP)},
                     "bo_1": {"file": C.FILE_BO_1_DAY_DU.name, "sha256": sha(C.FILE_BO_1_DAY_DU)},
                     "don_vi": {"file": C.FILE_DON_VI.name, "sha256": sha(C.FILE_DON_VI)},
                     "san_pham": {"file": C.FILE_SAN_PHAM.name, "sha256": sha(C.FILE_SAN_PHAM)}},
           "max_zh_xem": C.MAX_ZH_XEM, "max_trong_so_phan": C.MAX_TRONG_SO_PHAN,
           "cap_dung_bo_1": int((B0.trang_thai_vao == "dung").sum()), "cap_ten_hang_vo_nghia": int((B0.trang_thai_vao == "dung").sum() - len(B)),
           **khop, "cap_bo_dai_khong_co_ban": int(len(P_bo)), "cap": int(len(P)), "san_pham": int(CUM.product_id.nunique()),
           "cum": int(len(CUM)), "cum_can_goi": int(len(can_goi)), "san_pham_nhieu_phan": int((CUM.drop_duplicates("product_id").so_phan > 1).sum()),
           "nguon_ten_vao": P.nguon_ten_vao.value_counts().to_dict(), "nguon_gt_vao": P.nguon_gt_vao.value_counts().to_dict(),
           "truong_can_xuat": int(CUM.so_truong_can.sum()), "ky_tu_trung_can_xem": int(CUM.so_ky_tu_can.sum()),
           "san_pham_khong_co_tieu_de": int(sum(1 for p in CUM.product_id.unique() if not tieu_de.get(p))),
           "mau_thu": {"san_pham": int(m.product_id.nunique()), "cum": int(len(m)), "cum_can_goi": int((m.so_truong_can > 0).sum()),
                       "cap": int(m.so_muc.sum()), "truong_can_xuat": int(m.so_truong_can.sum()),
                       "theo_danh_muc": m.drop_duplicates("product_id").category.value_counts().sort_index().to_dict(), "hat_giong": C.HAT_GIONG}}
    (thu_muc_ra / "the_chuan_bi.json").write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    in_ra(json.dumps({k: v for k, v in the.items() if k != "nguon"}, ensure_ascii=False, indent=1))
    in_ra(f"({time.time() - t0:.0f} giây)")
    return CUM, P, the
