# Bước cuối của bộ 3 (laptop): từ kết quả hiệu đính của hai mô hình dựng hai bảng dùng cho fine-tune, cùng một nguồn nên khớp từng chữ:
#   - bảng CẶP (một dòng một thuộc tính): bản cuối theo luật chặt của ket_qua_v2 (khoá > Gemma qua luật > Qwen qua luật > bản nháp bộ 2),
#     kèm nguồn gốc bản dịch và mức đồng thuận hai mô hình;
#   - bảng SẢN PHẨM (một dòng một sản phẩm): cả danh sách thuộc tính dạng JSON và dạng văn bản liền theo cùng khuôn với bt_attributes_text của
#     bước EDA ("Tên: giá trị; Tên: giá trị"), cả tiếng Việt và tiếng Trung.
# Cặp không có bản cuối (không có nháp và mô hình trượt luật) bị bỏ khỏi cả hai bảng; sản phẩm ghi số cặp bị bỏ (so_cap_bo).
import json

import pandas as pd

from song_ngu_2.hoan_tat import bo_rac

from .ket_qua_v2 import giong, lap_bang_v2


KHONG_XEM = "nhap_khong_co_ban_mo_hinh"      # cụm của mô hình đó sai cấu trúc: mô hình không xem trường này, bản cuối là bản nháp
HANG = {"cao": 3, "vua": 2, "thap": 1}
TEN_HANG = {3: "cao", 2: "vua", 1: "thap"}
NGUON_BAN_DICH = {"khoa": "tu_dien_quy_tac", "giu_nhap": "giu_ban_dich_don_vi", "lui_nhap": "giu_ban_dich_don_vi",
                  KHONG_XEM: "giu_ban_dich_don_vi", "sua_A": "gemma_sua", "sua_B": "qwen_sua", "dich_moi_A": "dich_moi", "dich_moi_B": "dich_moi"}


def dong_thuan(r):
    """Mức đồng thuận hai mô hình trên giá trị: khoá; cả hai giữ nháp; cả hai sửa giống nhau; cả hai sửa khác nhau; chỉ A sửa; chỉ B sửa;
    chỉ một mô hình xem và giữ nháp (mô hình kia sai cấu trúc); không mô hình nào xem."""
    a, b = r.v_nguon_A, r.v_nguon_B
    if a == "khoa":
        return "khoa"
    xa, xb = a != KHONG_XEM, b != KHONG_XEM
    if not xa and not xb:
        return "khong_duoc_xem"
    sa, sb = a.startswith(("sua", "dich_moi")), b.startswith(("sua", "dich_moi"))
    if xa and xb:
        if not sa and not sb:
            return "ca_hai_giu_nhap"
        if sa and sb:
            return "ca_hai_sua_giong" if giong(r.v_cuoi_A, r.v_cuoi_B) else "ca_hai_sua_khac"
        return "chi_A_sua" if sa else "chi_B_sua"
    if xa:       # chỉ A xem
        return "chi_A_sua" if sa else "chi_mot_mo_hinh_giu_nhap"
    return "chi_B_sua" if sb else "chi_mot_mo_hinh_giu_nhap"


def muc_gia_tri(dt, nguon):
    """Nhãn của giá trị theo mức đồng thuận (đo ở mẫu chấm mù 24/10/2026: cả hai giữ nháp 95% đúng, cả hai sửa giống 100%, chỉ Gemma sửa 87%,
    cả hai sửa khác 72%, chỉ Qwen sửa 50%). Dịch mới và bản sửa bị loại phải lùi về nháp luôn là thấp."""
    if str(nguon).startswith(("lui", "dich_moi")):
        return "thap"
    if dt in ("khoa", "ca_hai_sua_giong", "ca_hai_giu_nhap"):
        return "cao"
    if dt in ("chi_A_sua", "chi_mot_mo_hinh_giu_nhap"):
        return "vua"
    return "thap"


def muc_ten(nguon):
    return "cao" if nguon in ("khoa", "giu_nhap", "lui_nhap", KHONG_XEM) else "vua"


def gan_do_tin_cay(P):
    """Nhãn của CẶP = mức thấp nhất của (giá trị, tên, nhãn bộ 2 đã công bố; cặp bộ 2 đã cất, không có nhãn, coi là thấp). Phương án bảo thủ anh
    duyệt 24/10/2026 để khớp nhãn của bộ 2."""
    mg = [muc_gia_tri(d, n) for d, n in zip(P.dong_thuan, P.nguon_gia_tri)]
    mt = [muc_ten(n) for n in P.nguon_ten]
    b2 = P.do_tin_cay_bo_2.fillna("thap")
    P["muc_gia_tri"], P["muc_ten"] = mg, mt
    P["do_tin_cay"] = [TEN_HANG[min(HANG[x], HANG[y], HANG[z])] for x, y, z in zip(mg, mt, b2)]
    P["nguon_ban_dich"] = [NGUON_BAN_DICH.get(n) for n in P.nguon_gia_tri]      # 'khong_co' (27 cặp không có bản cuối) là None, bị bỏ ở bước xuất
    return P


def ghep(CUM, thu_muc_kaggle, file_bo_2=None):
    """Trả về (bảng cặp đầy đủ, bảng sản phẩm, thống kê). Bản cuối = chọn AB: A qua luật, không thì B qua luật, không thì nháp."""
    P = lap_bang_v2(CUM, thu_muc_kaggle) if file_bo_2 is None else lap_bang_v2(CUM, thu_muc_kaggle, file_bo_2)
    P["dong_thuan"] = [dong_thuan(r) for r in P.itertuples(index=False)]
    for c in ("name_vi", "gia_tri_vi", "n_cuoi_AB", "v_cuoi_AB"):      # ký tự rác _x000D_ của Excel còn trong 4 giá trị nguồn (như bộ 2)
        P[c] = [bo_rac(x) for x in P[c]]
    P["name_zh"], P["gia_tri_zh"] = P.n_cuoi_AB, P.v_cuoi_AB
    P["nguon_ten"], P["nguon_gia_tri"] = P.n_nguon_AB, P.v_nguon_AB
    P = gan_do_tin_cay(P)
    co = P.name_zh.notna() & P.gia_tri_zh.notna()
    tk = {"cap_tong": int(len(P)), "cap_co_ban_cuoi": int(co.sum()), "cap_khong_co_ban_cuoi": int((~co).sum()),
          "nguon_gia_tri": P.nguon_gia_tri.value_counts().to_dict(), "nguon_ten": P.nguon_ten.value_counts().to_dict(),
          "dong_thuan": P.dong_thuan.value_counts().to_dict(), "sua_khong_ly_do": int(P.sua_khong_ly_do.sum()),
          "cum_sua_nhieu": int(P[P.sua_nhieu].ma_cum.nunique()),
          "gia_tri_khac_don_vi": int(sum(not giong(a, b) for a, b in zip(P.gia_tri_zh, P.v_don_vi))),
          "ten_khac_don_vi": int(sum(not giong(a, b) for a, b in zip(P.name_zh, P.n_don_vi)))}
    for k in ("A", "B"):
        tk[f"cum_{k}"] = P.drop_duplicates("ma_cum")[f"tt_{k}"].value_counts().to_dict()
    cap = P[co].sort_values(["product_id", "pair_id"])
    tong_cap = P.groupby("product_id").size()
    sp = []
    for pid, g in cap.groupby("product_id", sort=True):
        sp.append({"product_id": pid, "category": g.category.iloc[0], "so_cap": int(len(g)), "so_cap_bo": int(tong_cap[pid] - len(g)),
                   "do_tin_cay": TEN_HANG[min(HANG[x] for x in g.do_tin_cay)], "so_cap_khong_cao": int((g.do_tin_cay != "cao").sum()),
                   "thuoc_tinh_vi": json.dumps([{"ten": n, "gia_tri": v} for n, v in zip(g.name_vi, g.gia_tri_vi)], ensure_ascii=False),
                   "thuoc_tinh_zh": json.dumps([{"ten": n, "gia_tri": v} for n, v in zip(g.name_zh, g.gia_tri_zh)], ensure_ascii=False),
                   "description_vi": "; ".join(f"{n}: {v}" for n, v in zip(g.name_vi, g.gia_tri_vi)),
                   "description_zh": "; ".join(f"{n}: {v}" for n, v in zip(g.name_zh, g.gia_tri_zh))})
    SP = pd.DataFrame(sp)
    tk["san_pham"] = int(len(SP))
    tk["san_pham_mat_cap"] = int((SP.so_cap_bo > 0).sum())
    tk["san_pham_khong_con_cap_nao"] = int(len(tong_cap) - len(SP))
    return P, SP, tk
