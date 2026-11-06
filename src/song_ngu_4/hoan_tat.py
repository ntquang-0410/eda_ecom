# Bước cuối của bộ 4 (laptop): từ kết quả hiệu đính cả cụm của Gemma dựng hai bảng cùng một nguồn nên khớp từng chữ (như bộ 3):
#   - bảng CẶP (một dòng một thuộc tính): bản cuối theo luật chặt của ket_qua (khoá > Gemma qua luật > bản nháp bộ 1), kèm nguồn bản dịch
#     và nhãn độ tin cậy;
#   - bảng SẢN PHẨM (một dòng một sản phẩm): description_zh và description_vi nối "Tên: giá trị; Tên: giá trị" theo thứ tự trên trang,
#     dựng từ CÙNG các cặp được giữ.
# Anh chốt 25/10/2026: cả bộ CHỈ chạy Gemma (lần thử: Qwen chỉ làm khác bản cuối 1/4.802 cặp, dịch mới của Qwen sai nghĩa 8/20).
# Bản dịch mới (cặp bộ 1 không có bản) chỉ lấy của Gemma khi qua luật (lần thử: đúng 15/20), nhãn thấp. Cặp không có bản cuối bị bỏ.
# Nhãn độ tin cậy (bảo thủ như bộ 3): nhãn CẶP = mức thấp nhất của (mức tên, mức giá trị, nhãn bộ 1 của cặp; cặp bộ 1 không công bố coi
# là thấp). Mức một trường: khoá, giữ nháp, chưa xem (cụm Gemma hỏng) = cao (rồi bị chặn bởi nhãn bộ 1); Gemma sửa = vừa (lần thử: đúng
# 53/60); Gemma sửa mà trượt luật, lùi về nháp = vừa; dịch mới = thấp. Giá trị dài giữ bản bộ 1 nên nhãn đi theo bộ 1.
import pandas as pd

from song_ngu_2.hoan_tat import bo_rac

from . import cau_hinh as C
from .ket_qua import doc_khoang, giong, lap_bang, luat_truong

HANG = {"cao": 3, "vua": 2, "thap": 1}
TEN_HANG = {3: "cao", 2: "vua", 1: "thap"}
# Qwen (anh chốt 26/10/2026, phương án B) chỉ là DỰ PHÒNG khi Gemma hỏng, và CHỈ cho chỗ có bản nháp: bản Qwen qua luật chặt thay bản nháp
# (sua_B, nhãn thấp vì chỉ một mô hình yếu hơn đồng ý). Bản dịch MỚI của Qwen (không có nháp) KHÔNG dùng: chấm mù 26/10 trên 20 mục Gemma hơn 14,
# Qwen hơn 0, lần thử trước Qwen đúng 5/20 so với Gemma 15/20; cặp đó vẫn bị bỏ như khi chỉ có Gemma.
MUC_TRUONG = {"khoa": "cao", "giu_nhap": "cao", "nhap_khong_co_ban_mo_hinh": "cao", "sua_A": "vua", "lui_nhap": "vua",
              "dich_moi_A": "thap", "sua_B": "thap"}
DAI = "dai_giu_bo_1"
# Cột công bố (anh chốt 25/10/2026, đồng bộ các bộ trước): bảng cặp = cột bộ 1 + nguon_ban_dich (như bộ 3 = cột bộ 2 + nguon_ban_dich);
# bảng sản phẩm = cột bộ 3, bên nguồn (zh) trước, + so_cap_thap ở cuối (bộ 3 cũng được thêm cột này). Bảng nội bộ của ghep() có thêm cột.
COT_CAP_CONG_BO = ["pair_id", "product_id", "category", "nhom_trung", "name_zh", "gia_tri_zh", "name_vi", "gia_tri_vi", "do_tin_cay",
                   "nguon_ban_dich"]
COT_SP_CONG_BO = ["product_id", "category", "so_cap", "do_tin_cay", "so_cap_khong_cao", "description_zh", "description_vi", "so_cap_thap"]


# Kiểm toàn bộ bản dịch cuối 26/10/2026 (anh yêu cầu "check toàn bộ rồi xử lý luôn"): vài lỗi dịch hệ thống còn sót ở ô giữ nháp hoặc do Gemma áp
# bảng thuật ngữ sai chỗ. Mỗi dòng: (kiểu khớp chữ Trung "co" = có chứa | "bang" = đúng bằng, chữ Trung, cách dịch SAI, cách dịch ĐÚNG). Bản thay phải
# qua luật chặt của ket_qua rồi mới được dùng, nguồn ghi là tu_dien_quy_tac.
SUA_LOI_GIA_TRI = [
    ("co", "独立站", "Trạm độc lập", "Website độc lập"),             # nháp bộ 1 dịch sát chữ; bảng thuật ngữ v4
    ("co", "镜头", "thấu kính", "ống kính"), ("co", "镜头", "Thấu kính", "Ống kính"),
    ("co", "无印风", "Phong cách không nhãn hiệu", "Phong cách tối giản Nhật"),
    ("co", "拉链", "Ngăn kéo", "Ngăn khóa kéo"),                      # 拉链袋 là ngăn khóa kéo, không phải ngăn kéo (抽屉)
    ("bang", "厚底", "Đế độn mũi", "Đế dày"),                          # Gemma áp nhầm 防水台 (đế độn mũi) cho 厚底 (đế dày)
    ("bang", "套头", "Xỏ chân", "Chui đầu"),                            # 套头 là áo chui đầu
    ("bang", "日式", "Phong cách tối giản Nhật", "Kiểu Nhật"),          # Gemma áp nhầm 无印风 cho 日式
    ("bang", "中筒", "Thân trung", "Cổ vừa"), ("bang", "短筒", "Thân ngắn", "Cổ ngắn"),   # cùng họ lỗi "Thân thấp" của 低帮 (筒 là cổ giày, không phải thân)
]
NGUON_SUA_LOI = "quy_tac_sua_he_thong"
TEN_TOI_THIEU, TEN_DONG_THUAN = 50, 0.98     # tên thuộc tính có >= 50 cặp và >= 98% cùng một cách dịch thì là chuẩn


def nhat_quan(X, cach, khoang):
    """Chạy SAU khi chọn bản (Gemma / Qwen / nháp), TRƯỚC khi gắn nhãn. Sửa tại chỗ X, trả về thống kê.
    (1) Giá trị: thay các cách dịch sai ĐÃ BIẾT ở SUA_LOI_GIA_TRI (bản thay qua luật chặt, nguồn ghi quy tắc, nhãn theo bản khoá).
    (2) Tên thuộc tính: tên Trung phổ biến mà bản dịch lệch khỏi cách dịch chuẩn (do Gemma đổi tên theo giá trị: 颜色 -> "Kích thước",
        储藏方法 -> "Dịp sử dụng"...) thì trả về cách dịch chuẩn: tên phải dịch sát tên nguồn, không theo nội dung giá trị."""
    vc, vn, nc, nn = f"v_cuoi_{cach}", f"v_nguon_{cach}", f"n_cuoi_{cach}", f"n_nguon_{cach}"
    tk = {"gia_tri": {}, "gia_tri_bi_luat_chan": 0, "ten": 0}
    for kieu, zh, sai, dung in SUA_LOI_GIA_TRI:
        khop = X.gia_tri_zh.str.contains(zh, regex=False) if kieu == "co" else (X.gia_tri_zh == zh)
        co_sai = X[vc].map(lambda s: isinstance(s, str) and (sai in s if kieu == "co" else s == sai))
        # giá trị khoá bằng từ điển/quy tắc không đụng; giá trị DÀI giữ nguyên bản bộ 1 (cũng đánh dấu khoá) thì vẫn sửa được lỗi dịch sát chữ trong đó
        for i in X.index[khop & co_sai & ((X[vn] != "khoa") | (X.nguon_gt_vao == DAI))]:
            if zh == "拉链" and "抽屉" in X.at[i, "gia_tri_zh"]:
                continue
            moi = X.at[i, vc].replace(sai, dung) if kieu == "co" else dung
            # chỉ chặn nếu bản thay gây ra lỗi luật MỚI (giá trị dài giữ bộ 1 có thể đã vướng luật khác từ trước, không liên quan chỗ sửa)
            cu = luat_truong(X.at[i, "gia_tri_zh"], X.at[i, "v_gui"], X.at[i, vc], X.at[i, "v_nhap"], khoang)
            if set(luat_truong(X.at[i, "gia_tri_zh"], X.at[i, "v_gui"], moi, X.at[i, "v_nhap"], khoang)) - set(cu):
                tk["gia_tri_bi_luat_chan"] += 1
                continue
            X.at[i, vc], X.at[i, vn], X.at[i, "nguon_gt_vao"] = moi, "khoa", NGUON_SUA_LOI
            tk["gia_tri"][f"{zh}:{sai}"] = tk["gia_tri"].get(f"{zh}:{sai}", 0) + 1
    dem = X[X[nc].notna()].groupby(["name_zh", nc]).size().reset_index(name="n")
    tong = dem.groupby("name_zh").n.sum()
    tot = dem.sort_values("n", ascending=False, kind="stable").drop_duplicates("name_zh").set_index("name_zh")
    chuan = {z: tot.at[z, nc] for z in tot.index if tong[z] >= TEN_TOI_THIEU and tot.at[z, "n"] / tong[z] >= TEN_DONG_THUAN}
    c = X.name_zh.map(chuan)
    lech = c.notna() & X[nc].notna() & (X[nn] != "khoa") & (X[nc] != c)
    for i in X.index[lech]:
        X.at[i, nc] = c[i]
        X.at[i, nn] = "giu_nhap" if X.at[i, "n_nhap"] == c[i] else "khoa"
        tk["ten"] += 1
    return tk


def nguon_ban_dich(nguon_gt, nguon_gt_vao):
    """Nguồn của giá trị Việt cuối: tu_dien_quy_tac (từ điển đã duyệt, bảng có/không, quy tắc của bộ 1) | giu_ban_dich_don_vi (bản bộ 1:
    giữ nháp, lùi về nháp, chưa xem, giá trị dài) | gemma_sua | dich_moi."""
    if nguon_gt == "khoa":
        return "giu_ban_dich_don_vi" if nguon_gt_vao == DAI else "tu_dien_quy_tac"
    if nguon_gt in ("giu_nhap", "lui_nhap", "nhap_khong_co_ban_mo_hinh"):
        return "giu_ban_dich_don_vi"
    return {"sua_A": "gemma_sua", "dich_moi_A": "dich_moi", "sua_B": "qwen_sua"}.get(nguon_gt)


def gan_do_tin_cay(P, cach="A"):
    mt = [MUC_TRUONG[n] for n in P[f"n_nguon_{cach}"]]
    mg = [MUC_TRUONG[n] for n in P[f"v_nguon_{cach}"]]
    b1 = P.do_tin_cay_bo_1.fillna("thap")
    P["muc_ten"], P["muc_gia_tri"] = mt, mg
    P["do_tin_cay"] = [TEN_HANG[min(HANG[x], HANG[y], HANG[z])] for x, y, z in zip(mt, mg, b1)]
    return P


def ghep(P, thu_muc_gemma, cho_phep_thieu=False, khoang=None, thu_muc_qwen=None):
    """P: cap_vao.parquet (cả bộ, hoặc lọc theo mẫu thử). Trả về (bảng cặp đầy đủ, bảng sản phẩm, thống kê).
    cho_phep_thieu=False: dừng nếu còn cụm Gemma chưa dịch (chạy thật phải đủ mọi cụm).
    thu_muc_qwen: thư mục kết quả Qwen dự phòng (chỉ chạy một phần số cụm): Gemma qua luật thì lấy Gemma, không thì Qwen nếu qua luật VÀ có
    bản nháp, không thì lùi nháp."""
    cach = "AB" if thu_muc_qwen else "A"
    X = lap_bang(P, {"A": thu_muc_gemma, **({"B": thu_muc_qwen} if thu_muc_qwen else {})}, khoang)
    tt = X.drop_duplicates("ma_cum").tt_A.value_counts().to_dict()
    if not cho_phep_thieu:
        thieu = tt.get("chua_dich", 0)
        if cach == "AB":     # cụm Gemma chưa dịch nhưng Qwen dự phòng đã dịch xong thì chấp nhận
            c1 = X.drop_duplicates("ma_cum")
            thieu = int(((c1.tt_A == "chua_dich") & (c1.tt_B != "ok")).sum())
        assert thieu == 0, f"còn {thieu} cụm chưa dịch (Gemma chưa dịch và Qwen cũng chưa): chạy tiếp phiên sau rồi mới ghép"
    if cach == "AB":     # bản dịch mới của Qwen (không có nháp) không dùng: coi như không có bản (xem ghi chú ở MUC_TRUONG)
        for f in ("n", "v"):
            moi = X[f"{f}_nguon_AB"] == "dich_moi_B"
            X.loc[moi, f"{f}_cuoi_AB"] = None
            X.loc[moi, f"{f}_nguon_AB"] = "khong_co"
            # Gemma có viết ra bản khác nháp nhưng trượt luật, Qwen lại giữ nháp: đó là BẤT ĐỒNG giữa hai mô hình, không phải "hai bên đồng ý nháp"
            # -> nhãn như khi Gemma trượt và lùi về nháp (lui_nhap = vừa), không nâng lên cao (giu_nhap)
            lui = (X[f"{f}_nguon_AB"] == "giu_nhap") & (X[f"{f}_luat_A"] != "") & X[f"{f}_A"].notna()
            X.loc[lui, f"{f}_nguon_AB"] = "lui_nhap"
    kc = nhat_quan(X, cach, khoang or doc_khoang())
    D1 = pd.read_parquet(C.FILE_BO_1_DAY_DU, columns=["pair_id", "do_tin_cay"]).rename(columns={"do_tin_cay": "do_tin_cay_bo_1"})
    X = X.merge(D1, on="pair_id", how="left")
    X["name_zh"] = [bo_rac(x) for x in X.name_zh]
    X["gia_tri_zh"] = [bo_rac(x) for x in X.gia_tri_zh]
    X["name_vi"] = [bo_rac(x) if isinstance(x, str) else None for x in X[f"n_cuoi_{cach}"]]
    X["gia_tri_vi"] = [bo_rac(x) if isinstance(x, str) else None for x in X[f"v_cuoi_{cach}"]]
    co = X.name_vi.notna() & X.gia_tri_vi.notna()
    X["nguon_ban_dich"] = [nguon_ban_dich(g, v) if c else None for g, v, c in zip(X[f"v_nguon_{cach}"], X.nguon_gt_vao, co)]
    if cach == "AB":     # Qwen thay TÊN thuộc tính (giá trị vẫn của nguồn khác) cũng phải ghi nguồn qwen_sua để người dùng lọc được
        X["nguon_ban_dich"] = ["qwen_sua" if (x is not None and n == "sua_B") else x for x, n in zip(X.nguon_ban_dich, X.n_nguon_AB)]
    G = gan_do_tin_cay(X[co].copy(), cach)
    tk = {"cap_vao": int(len(X)), "cap_giu": int(co.sum()), "cap_bo_khong_co_ban": int((~co).sum()), "cach_chon": cach, "nhat_quan": kc,
          "cum_gemma": tt, "nguon_ten": X[f"n_nguon_{cach}"].value_counts().to_dict(), "nguon_gia_tri": X[f"v_nguon_{cach}"].value_counts().to_dict(),
          "nguon_ban_dich": G.nguon_ban_dich.value_counts().to_dict(), "do_tin_cay_cap": G.do_tin_cay.value_counts().to_dict(),
          "luat_truot_gia_tri_gemma": pd.Series([x for s in X.v_luat_A if s for x in s.split("|")]).value_counts().to_dict(),
          "luat_truot_ten_gemma": pd.Series([x for s in X.n_luat_A if s for x in s.split("|")]).value_counts().to_dict(),
          "sua_khong_ly_do": int(X.sua_khong_ly_do.sum()), "cum_sua_nhieu": int(X[X.sua_nhieu].ma_cum.nunique()),
          "gia_tri_khac_bo_1": int(sum(1 for a, b, c in zip(G.gia_tri_vi, G.v_nhap, G.v_khoa) if not giong(a, b if isinstance(b, str) else c))),
          "ten_khac_bo_1": int(sum(1 for a, b, c in zip(G.name_vi, G.n_nhap, G.n_khoa) if not giong(a, b if isinstance(b, str) else c)))}
    if cach == "AB":
        tk["cum_qwen"] = X.drop_duplicates("ma_cum").tt_B.value_counts().to_dict()
        tk["qwen_thay_gemma_gia_tri"] = int((X.v_nguon_AB == "sua_B").sum())
        tk["qwen_thay_gemma_ten"] = int((X.n_nguon_AB == "sua_B").sum())
        tk["qwen_dich_moi_khong_dung"] = int(((X.v_A.isna()) & X.v_B.notna() & X.v_khoa.isna() & X.v_nhap.isna()).sum())
    tong_cap = X.groupby("product_id").size()
    sp = []
    for pid, g in G.sort_values(["product_id", "vi_tri", "pair_id"]).groupby("product_id", sort=True):
        sp.append({"product_id": pid, "category": g.category.iloc[0], "nhom_trung": int(g.nhom_trung.iloc[0]), "so_cap": int(len(g)),
                   "so_cap_bo": int(tong_cap[pid] - len(g)), "do_tin_cay": TEN_HANG[min(HANG[x] for x in g.do_tin_cay)],
                   "so_cap_khong_cao": int((g.do_tin_cay != "cao").sum()), "so_cap_thap": int((g.do_tin_cay == "thap").sum()),
                   "description_zh": "; ".join(f"{n}: {v}" for n, v in zip(g.name_zh, g.gia_tri_zh)),
                   "description_vi": "; ".join(f"{n}: {v}" for n, v in zip(g.name_vi, g.gia_tri_vi))})
    SP = pd.DataFrame(sp)
    tk["san_pham"] = int(len(SP))
    tk["san_pham_mat_cap"] = int((SP.so_cap_bo > 0).sum())
    tk["san_pham_khong_con_cap_nao"] = int(X.product_id.nunique() - len(SP))
    tk["do_tin_cay_san_pham"] = SP.do_tin_cay.value_counts().to_dict()
    return G, SP, tk
