# Bước 3 của bộ 3 bản v2 (laptop): ghép kết quả hiệu đính cả cụm về từng cặp, theo luật chặt:
#   - mục KHOÁ luôn lấy bản khoá (mô hình lỡ xuất lại cũng bỏ qua);
#   - mỗi trường mô hình xuất ra phải qua luật bộ 2 (kiem_mot); trượt thì LÙI VỀ BẢN NHÁP (bản nháp đã qua luật ở bộ 2), không có nháp
#     thì để trống (cặp đó không xuất);
#   - cụm sai cấu trúc (sau một lần hỏi lại): mọi trường lùi về bản nháp;
#   - cờ kiểm tay: sửa mà không ghi lý do (sua_khong_ly_do), cụm sửa quá nhiều (sua_nhieu: >= 3 trường và > 60% trường có nháp);
#   - chọn hai mô hình: lấy bản của A (Gemma) nếu qua luật, không thì bản của B (Qwen) nếu qua luật, không thì bản nháp.
import json

import pandas as pd

from song_ngu_2.luat_kiem import HAN, kiem_mot
from song_ngu_2.van_ban import chuan_hoa_zh

from . import cau_hinh as C
from .ket_qua import doc_lo

MO_HINH = {"A": "api_tuy_chon", "B": "qwen3_14b"}
NGUONG_SUA_NHIEU = (3, 0.6)


def giong(a, b):
    return isinstance(a, str) and isinstance(b, str) and chuan_hoa_zh(a) == chuan_hoa_zh(b)


def trai_phang_v2(CUM):
    """Một dòng một cặp: nguồn, bản khoá, bản nháp của tên và giá trị."""
    rows = []
    for r in CUM.itertuples(index=False):
        for m, pid in zip(json.loads(r.muc_json), json.loads(r.pair_ids)):
            rows.append({"pair_id": pid, "product_id": r.product_id, "category": r.category, "ma_cum": r.ma_cum, "i": m["i"],
                         "name_vi": m["n"], "gia_tri_vi": m["v"], "n_khoa": m.get("n_zh"), "v_khoa": m.get("v_zh"),
                         "n_nhap": m.get("draft_n"), "v_nhap": m.get("draft_v")})
    return pd.DataFrame(rows)


def gan_mo_hinh(P, KQ, k):
    """Thêm cột bản mô hình k (n_k, v_k, c_k) và trạng thái cụm (tt_k)."""
    kq = KQ.set_index("ma_cum")
    out_n, out_v, out_c, tt = [], [], [], []
    cache = {}
    for ma, i in zip(P.ma_cum, P.i):
        if ma not in cache:
            if ma not in kq.index:
                cache[ma] = ("chua_dich", {})
            elif kq.at[ma, "trang_thai"] != "ok":
                cache[ma] = ("loi_cau_truc", {})
            else:
                cache[ma] = ("ok", json.loads(kq.at[ma, "ket_qua"]))
        t, d = cache[ma]
        o = d.get(str(i), {})
        out_n.append(o.get("n"))
        out_v.append(o.get("v"))
        out_c.append(o.get("c", "") if o else None)
        tt.append(t)
    P[f"n_{k}"], P[f"v_{k}"], P[f"c_{k}"], P[f"tt_{k}"] = out_n, out_v, out_c, tt
    return P


def qua_luat(vi, zh, loai):
    return isinstance(zh, str) and not kiem_mot(vi, vi, zh, None, loai)[0]


def dem_han(s):
    return len(HAN.findall(s)) if isinstance(s, str) else 0


def bo_chu_han(nhap, z):
    """Luật thêm sau lần thử v2 (23/10/2026): bản sửa làm mất quá nửa chữ Hán so với bản nháp là mô hình đã QUAY LẠI chữ tiếng Anh của
    nguồn (danh sách thành phần mỹ phẩm chép nguyên tiếng Anh, 'Silver' thay cho 银色): loại, lùi về nháp. Tác dụng phụ đã biết: nếu
    bản nháp lỡ phiên âm tên hãng (三星) thì bản sửa về 'Samsung' cũng bị loại."""
    n = dem_han(nhap)
    return n >= 2 and dem_han(z) * 2 < n


def chon_truong(khoa, nhap, ung_vien, vi, loai):
    """ung_vien: danh sách (tên nguồn, bản) theo thứ tự ưu tiên. Trả về (bản cuối, nguồn, danh sách luật trượt của ứng viên đầu)."""
    if isinstance(khoa, str):
        return khoa, "khoa", []
    truot_dau = None
    for ten, z in ung_vien:
        if not isinstance(z, str):
            continue
        tr = kiem_mot(vi, vi, z, None, loai)[0] + (["bo_chu_han"] if bo_chu_han(nhap, z) else [])
        if truot_dau is None:
            truot_dau = tr
        if not tr:
            if isinstance(nhap, str):
                return z, ("giu_nhap" if giong(z, nhap) else f"sua_{ten}"), truot_dau
            return z, f"dich_moi_{ten}", truot_dau
    if isinstance(nhap, str):
        return nhap, ("lui_nhap" if truot_dau else "nhap_khong_co_ban_mo_hinh"), truot_dau or []
    return None, "khong_co", truot_dau or []


def lap_bang_v2(CUM, thu_muc_kaggle, file_bo_2=C.FILE_BO_2_DAY_DU):
    """thu_muc_kaggle: một thư mục (chứa cả hai mô hình) hoặc dict {"A": thư mục Gemma, "B": thư mục Qwen} (chạy thật: hai notebook riêng)."""
    P = trai_phang_v2(CUM)
    for k, m in MO_HINH.items():
        goc = thu_muc_kaggle[k] if isinstance(thu_muc_kaggle, dict) else thu_muc_kaggle
        P = gan_mo_hinh(P, doc_lo(goc, m), k)
    for cach, uv in (("A", ["A"]), ("B", ["B"]), ("AB", ["A", "B"])):
        for f, loai, cot_vi in (("n", "ten", "name_vi"), ("v", "gia_tri", "gia_tri_vi")):
            kq = [chon_truong(kh, nh, [(x, r[f"{f}_{x}"]) for x in uv], vi, loai)
                  for kh, nh, vi, r in zip(P[f"{f}_khoa"], P[f"{f}_nhap"], P[cot_vi], P[[f"{f}_{x}" for x in uv]].to_dict("records"))]
            P[f"{f}_cuoi_{cach}"] = [x[0] for x in kq]
            P[f"{f}_nguon_{cach}"] = [x[1] for x in kq]
            P[f"{f}_truot_{cach}"] = [x[2] for x in kq]
    # cờ kiểm tay theo bản A
    sua = P.v_nguon_A.str.startswith("sua") | P.n_nguon_A.str.startswith("sua")
    P["sua_khong_ly_do"] = sua & (P.c_A.fillna("").str.strip() == "")
    co_nhap = P.groupby("ma_cum").apply(lambda g: int(g.n_nhap.notna().sum() + g.v_nhap.notna().sum()), include_groups=False)
    so_sua = P.assign(s=sua.astype(int)).groupby("ma_cum").s.sum()
    nhieu = {m for m in co_nhap.index if so_sua.get(m, 0) >= NGUONG_SUA_NHIEU[0] and so_sua.get(m, 0) > NGUONG_SUA_NHIEU[1] * co_nhap[m]}
    P["sua_nhieu"] = P.ma_cum.isin(nhieu)
    B2 = pd.read_parquet(file_bo_2, columns=["pair_id", "name_zh", "gia_tri_zh", "cach_chon", "do_tin_cay"]).rename(
        columns={"name_zh": "n_don_vi", "gia_tri_zh": "v_don_vi", "cach_chon": "cach_chon_bo_2", "do_tin_cay": "do_tin_cay_bo_2"})
    return P.merge(B2, on="pair_id", how="left")


def tom_tat_v2(CUM, P):
    out = {"cum": int(len(CUM)), "san_pham": int(CUM.product_id.nunique()), "cap": int(len(P))}
    for k in MO_HINH:
        out[k] = {"cum_theo_trang_thai": P.drop_duplicates("ma_cum")[f"tt_{k}"].value_counts().to_dict()}
    for cach in ("A", "B", "AB"):
        out[f"chon_{cach}"] = {"ten_nguon": P[f"n_nguon_{cach}"].value_counts().to_dict(),
                               "gia_tri_nguon": P[f"v_nguon_{cach}"].value_counts().to_dict(),
                               "gia_tri_khac_don_vi": int(sum(not giong(a, b) for a, b in zip(P[f"v_cuoi_{cach}"], P.v_don_vi))),
                               "ten_khac_don_vi": int(sum(not giong(a, b) for a, b in zip(P[f"n_cuoi_{cach}"], P.n_don_vi)))}
    out["sua_khong_ly_do"] = int(P.sua_khong_ly_do.sum())
    out["cum_sua_nhieu"] = int(P[P.sua_nhieu].ma_cum.nunique())
    return out
