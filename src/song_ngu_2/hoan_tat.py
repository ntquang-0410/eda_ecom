# Bước 5, 6b, 6c, 7 của bộ 2 (laptop, CPU): đọc kết quả tải về từ Kaggle (bản A Gemma, bản B Qwen, điểm CometKiwi), gỡ nhãn thừa,
# kiểm luật, chọn một bản Trung cho mỗi đơn vị, đè bản cố định đã duyệt, ghép về từng cặp, xuất bản đầy đủ, bản công bố, file cất riêng
# và thẻ nguồn gốc. Bản gương của song_ngu_1.hoan_tat. Chạy lại bao nhiêu lần cũng được mà không phải chạy lại GPU; mỗi lần ghi file mới
# có dấu thời gian, không ghi đè lần trước.
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from song_ngu_1.chon_ban import chon
from song_ngu_1.luat_kiem import do_dai_bat_thuong, doc_tu_dien, khoang_do_dai

from . import cau_hinh as C
from .luat_kiem import go_nhan, kiem_mot, mau_anh_chua_dich
from .quy_tac_gia_tri import ban_quy_tac
from .rieng_tu import rieng_tu
from .van_ban import chrf, chuan_hoa_zh

COT_DAY_DU = ["pair_id", "product_id", "category", "split_nguon", "row_hash", "name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh",
              "gia_tri_zh_A", "gia_tri_zh_B", "cach_chon", "do_tin_cay", "chrf_ab", "diem_kiwi_A", "diem_kiwi_B", "luat_truot", "co_nhac"]
HANG_TIN = {"cao": 3, "vua": 2, "thap": 1}
ROUTE_CHEP = ("chep_nguyen", "ten_hang", "ma_model", "do_luong", "co_chu")
RAC_EXCEL = "_x000D_"     # dấu xuống dòng \r của Excel bị ghi thành chữ trong 4 giá trị nguồn; mô hình chép theo sang bản Trung


def bo_rac(s):
    """Bỏ chữ rác _x000D_ ở bước ghép cặp (không đổi bước chuẩn bị nên không lệch mã đơn vị đã dịch). Chuỗi không có rác giữ nguyên."""
    if not isinstance(s, str) or RAC_EXCEL not in s:
        return s
    return " ".join(s.replace(RAC_EXCEL, " ").split())


def doc_lo(thu_muc, ten):
    """Gộp mọi lo_*.parquet trong các thư mục con tên `ten` (tải về từ Kaggle, có thể nhiều phiên bản chạy nối tiếp)."""
    files = sorted(Path(thu_muc).rglob(f"{ten}/lo_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["ma_don_vi"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("ma_don_vi", keep="last")


def doc_the(*thu_muc):
    out = {}
    for tm in thu_muc:
        if not tm:
            continue
        for f in sorted(Path(tm).rglob("the_*.json")):
            try:
                out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:  # thẻ hỏng không làm hỏng cả lần chạy
                out[f.stem] = {"loi_doc": str(e)}
    return out


def nhap(a, b):
    return ";".join(x for x in [a, b] if x)


def ma_bam(file):
    p = Path(file)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def doc_bang_cap(file):
    """Bảng duyệt khoá theo CẶP (ten_vi, gia_tri_vi): P7 giá trị hay gặp và P8 có/không. Giá trị cùng chữ nhưng khác thuộc tính có thể
    dịch khác (Có/Không, Vàng, Điện tử) nên khoá theo cả tên thuộc tính; mỗi đơn vị gửi mô hình chính là một cặp như vậy. Cùng quy ước anh_duyet
    như doc_tu_dien: mục anh đã duyệt dùng anh_duyet nếu anh ghi bản khác OK, không thì goi_y_claude, không có nữa thì bản chuẩn tạm."""
    if not Path(file).exists():
        return {}
    t = pd.read_csv(file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    out = {}
    for r in t[t.anh_duyet.str.strip() != ""].itertuples():
        d = r.anh_duyet.strip()
        out[(r.ten_vi, r.gia_tri_vi)] = d if d.upper() != "OK" else (r.goi_y_claude.strip() or r.ban_chuan_tam)
    return out


def khoa_chuan(s):
    """Khoá tra bảng: bỏ khoảng trắng thừa, chữ thường. 'Dòng Máy', 'DÒNG MÁY', 'dòng máy' cùng một mục (lần kiểm 23/10/2026: biến thể chỉ khác
    chữ hoa thường nằm ngoài top 120 làm lệch bản dịch với tên chính, vd 'Đời Xe' vẫn là 年份 khi 'Đời xe' đã thành 年款)."""
    return " ".join(str(s).split()).lower()


def chuan_bang(d):
    """Chuẩn hoá khoá bảng duyệt (chuỗi hoặc cặp). Hai khoá chỉ khác chữ hoa thường mà bản dịch khác nhau thì giữ mục đầu và đếm là xung đột."""
    out, xung_dot = {}, 0
    for k, v in d.items():
        kk = khoa_chuan(k) if isinstance(k, str) else tuple(khoa_chuan(x) for x in k)
        if kk in out and out[kk] != v:
            xung_dot += 1
            continue
        out[kk] = v
    return out, xung_dot


def ban_co_dinh(loai, vi, ten_vi, tu_dien_ten, tu_dien_gt, co_khong):
    """Bước 6c: bản dịch cố định của một đơn vị, trả (bản, cách chọn) hoặc None. Tên thuộc tính: từ điển đã duyệt. Giá trị: bảng
    có/không theo thuộc tính trước, rồi từ điển giá trị đã duyệt, rồi quy tắc thời hạn. So khớp nguyên chuỗi Việt (bỏ khoảng trắng hai đầu)."""
    if not isinstance(vi, str):
        return None
    z, ten = khoa_chuan(vi), khoa_chuan(ten_vi or "")
    if loai == "ten":
        if tu_dien_ten.get(z):
            return tu_dien_ten[z], "tu_dien_p6"
    else:
        if co_khong.get((ten, z)):
            return co_khong[(ten, z)], "quy_tac_co_khong"
        if tu_dien_gt.get((ten, z)):
            return tu_dien_gt[(ten, z)], "tu_dien_p7"
    if loai != "ten":
        return ban_quy_tac(vi)
    return None


def chay(thu_muc_kaggle, thu_muc_ra=C.THU_MUC_RA, tau=C.TAU, kappa=C.KAPPA, delta=C.DELTA, in_ra=print, kappa_ngan=C.KAPPA_NGAN,
         uu_tien=C.UU_TIEN, tu_dien=None, chi_doi_khi_truot=C.CHI_DOI_KHI_TRUOT, thu_muc_the_phu=None,
         kappa_cat=C.KAPPA_CAT, kappa_cat_ngan=C.KAPPA_CAT_NGAN):
    """tu_dien: None thì đọc file duyệt thật (C.FILE_TU_DIEN_TEN, FILE_TU_DIEN_GIA_TRI, FILE_CO_KHONG; file chưa có thì rỗng);
    bộ thử truyền {"ten": {...}, "gia_tri": {...}, "co_khong": {(ten, gt): ...}}. thu_muc_the_phu: nơi có thêm thẻ (thẻ API của Gemma)."""
    t0 = time.time()
    thu_muc_ra = Path(thu_muc_ra)
    B = pd.read_parquet(thu_muc_ra / "bang_cap.parquet")
    DV = pd.read_parquet(thu_muc_ra / "vao_kaggle" / "don_vi_can_dich.parquet")
    ten_a, ten_b = C.MO_HINH["A"], C.MO_HINH["B"]
    A = doc_lo(thu_muc_kaggle, ten_a).rename(columns={"ban_dich": "A", "ly_do_dung": "dung_A"})
    Bm = doc_lo(thu_muc_kaggle, ten_b).rename(columns={"ban_dich": "B", "ly_do_dung": "dung_B"})
    KA = doc_lo(thu_muc_kaggle, f"kiwi_{ten_a}").rename(columns={"diem": "kiwi_A"})
    KB = doc_lo(thu_muc_kaggle, f"kiwi_{ten_b}").rename(columns={"diem": "kiwi_B"})
    U = DV.copy()
    for df, cot in [(A, "A"), (Bm, "B"), (A, "dung_A"), (Bm, "dung_B"), (KA, "kiwi_A"), (KB, "kiwi_B")]:
        U = U.merge(df[["ma_don_vi", cot]] if cot in df.columns else pd.DataFrame(columns=["ma_don_vi", cot]), on="ma_don_vi", how="left")
    in_ra(f"Đơn vị: {len(U):,}; có bản A {U.A.notna().sum():,}, bản B {U.B.notna().sum():,}, "
          f"điểm A {U.kiwi_A.notna().sum():,}, điểm B {U.kiwi_B.notna().sum():,}")

    # Bước 5a: gỡ nhãn tên thuộc tính mô hình tự thêm vào đầu bản dịch giá trị (go_nhan: chỉ khi nhãn trùng bản dịch tên của CHÍNH mô hình đó)
    ten_theo_mo_hinh = {x: dict(zip(U.loc[U.loai == "ten", "van_ban_nguon"], U.loc[U.loai == "ten", x])) for x in ["A", "B"]}
    go = {}
    for x in ["A", "B"]:
        moi = [go_nhan(v, g, ten_theo_mo_hinh[x].get(n), n) if (l == "gia_tri" and isinstance(v, str)) else None
               for v, g, n, l in zip(U[x], U.van_ban_gui, U.name_nguon, U.loai)]
        U[f"go_nhan_{x}"] = [m is not None for m in moi]
        U[x] = [m if m is not None else v for m, v in zip(moi, U[x])]
        go[x] = int(sum(m is not None for m in moi))
    in_ra(f"Gỡ nhãn tên thuộc tính lọt vào bản dịch: A {go['A']}, B {go['B']} đơn vị")

    # Bước 5: luật kiểm
    if tu_dien is None:
        tu_dien_ten = doc_tu_dien(C.FILE_TU_DIEN_TEN, "ten_vi", "ban_chuan_tam")
        tu_dien_gt = doc_bang_cap(C.FILE_TU_DIEN_GIA_TRI)
        co_khong = doc_bang_cap(C.FILE_CO_KHONG)
    else:
        tu_dien_ten, tu_dien_gt, co_khong = tu_dien.get("ten", {}), tu_dien.get("gia_tri", {}), tu_dien.get("co_khong", {})
    tu_dien_ten_goc = tu_dien_ten
    tu_dien_ten, xd_ten = chuan_bang(tu_dien_ten)
    tu_dien_gt, xd_gt = chuan_bang(tu_dien_gt)
    co_khong, xd_ck = chuan_bang(co_khong)
    if xd_ten or xd_gt or xd_ck:
        in_ra(f"CẢNH BÁO: khoá chỉ khác chữ hoa thường mà bản dịch khác nhau (giữ mục đầu): tên {xd_ten}, giá trị {xd_gt}, có/không {xd_ck}")
    for x in ["A", "B"]:
        kq = [kiem_mot(z, g, v, tu_dien_ten_goc if l == "ten" else None, l) if isinstance(v, str) else (["thieu"], [])
              for z, g, v, l in zip(U.van_ban_nguon, U.van_ban_gui, U[x], U.loai)]
        U[f"truot_{x}"] = [k[0] + ["bi_cat"] if d == "length" else k[0] for k, d in zip(kq, U[f"dung_{x}"])]     # hết token: bản dịch bị cắt
        U[f"nhac_{x}"] = [k[1] + (["go_nhan"] if f else []) for k, f in zip(kq, U[f"go_nhan_{x}"])]
    # luật mau_chua_dich (23/10/2026): bản còn nguyên tên màu tiếng Anh trượt NẾU bản kia đã dịch và qua luật
    for x, y in (("A", "B"), ("B", "A")):
        U[f"truot_{x}"] = [tr + ["mau_chua_dich"] if l == "gia_tri" and isinstance(vx, str) and isinstance(vy, str) and not try_
                           and mau_anh_chua_dich(g, vx) and not mau_anh_chua_dich(g, vy) and "mau_chua_dich" not in tr else tr
                           for tr, try_, vx, vy, g, l in zip(U[f"truot_{x}"], U[f"truot_{y}"], U[x], U[y], U.van_ban_gui, U.loai)]
    co = [(len(z), len(v)) for x in ["A", "B"] for z, v in zip(U.van_ban_gui, U[x]) if isinstance(v, str) and v.strip()]
    khoang = khoang_do_dai([a for a, _ in co], [b for _, b in co], *C.Q_DO_DAI) if co else {}
    for x in ["A", "B"]:
        U[f"truot_{x}"] = [t + ["do_dai"] if isinstance(v, str) and v.strip() and do_dai_bat_thuong(len(g), len(v), khoang)
                           and "do_dai" not in t else t for t, g, v in zip(U[f"truot_{x}"], U.van_ban_gui, U[x])]

    # Bước 6b: chọn một bản (so hai bản bằng chuan_hoa_zh: bỏ khác biệt dấu câu toàn/nửa góc và khoảng trắng)
    U["chrf_ab"] = [round(chrf(chuan_hoa_zh(a), chuan_hoa_zh(b)), 1) if isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()
                    else None for a, b in zip(U.A, U.B)]
    co_dinh = [ban_co_dinh(l, z, n, tu_dien_ten, tu_dien_gt, co_khong) for l, z, n in zip(U.loai, U.van_ban_nguon, U.name_nguon)]

    def chon_het(t, k, d, kc=None):
        kn = min(kappa_ngan, k)
        kc = kappa_cat if kc is None else kc
        kcn = min(kappa_cat_ngan, kc)
        kq = [chon(a if isinstance(a, str) else None, b if isinstance(b, str) else None, ta, tb, na, nb, ka, kb, t, k, d,
                   C.DO_DAI_NGAN, cf, dai_zh=len(g), kappa_ngan=kn, uu_tien=uu_tien, chi_doi_khi_truot=chi_doi_khi_truot,
                   norm=chuan_hoa_zh, dai_ngan_nguon=C.DO_DAI_NGUON_NGAN, kappa_cat=kc, kappa_cat_ngan=kcn)
              for a, b, ta, tb, na, nb, ka, kb, cf, g in zip(U.A, U.B, U.truot_A, U.truot_B, U.nhac_A, U.nhac_B,
                                                            U.kiwi_A, U.kiwi_B, U.chrf_ab, U.van_ban_gui)]
        return [{**x, "ban": cd[0], "tu": "co_dinh", "cach_chon": cd[1], "do_tin_cay": "cao", "ly_do_cat": None} if cd else x
                for x, cd in zip(kq, co_dinh)]
    kq = chon_het(tau, kappa, delta)
    for cot in ["ban", "tu", "cach_chon", "do_tin_cay", "ly_do_cat"]:
        U[cot] = [k[cot] for k in kq]
    U["co_nhac"] = [";".join(na if t == "A" else nb) if t in ("A", "B") else "" for t, na, nb in zip(U.tu, U.nhac_A, U.nhac_B)]
    dem_co_dinh = U[U.tu == "co_dinh"].groupby("cach_chon").agg(don_vi=("ma_don_vi", "size"), cap=("so_cap", "sum"))
    in_ra("Bản cố định (6c): " + (", ".join(f"{c} {int(r.don_vi):,} đơn vị ({int(r.cap):,} cặp)" for c, r in dem_co_dinh.iterrows()) or "chưa có"))
    U["luat_truot"] = [nhap(f"A:{'|'.join(ta)}" if ta else "", f"B:{'|'.join(tb)}" if tb else "") for ta, tb in zip(U.truot_A, U.truot_B)]

    # bảng độ nhạy để chọn ngưỡng
    nhay = []
    for kc in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        for t in [40, 60, 80]:
            g = chon_het(t, kappa, delta, kc)
            nhay.append({"kappa_cat": kc, "tau": t, "ti_le_giu": round(sum(x["ban"] is not None for x in g) / max(len(g), 1), 4)})

    # Bước 7: ghép về từng cặp
    Ut = U.set_index("ma_don_vi")
    rows, cat = [], []
    for r in B.itertuples(index=False):
        goc = {"pair_id": r.pair_id, "product_id": r.product_id, "category": r.category, "split_nguon": r.split_nguon,
               "row_hash": r.row_hash, "name_vi": r.name_vi, "gia_tri_vi": bo_rac(r.gia_tri_vi)}
        if r.trang_thai_vao != "dung":
            cat.append({**goc, "ly_do_cat": f"{r.trang_thai_vao}:{r.ly_do_van_loai}"})
            continue
        ly_rieng = rieng_tu(r.name_vi, r.gia_tri_vi)       # tên cửa hàng, địa chỉ đường phố còn sót (hiến chương mục 6)
        if ly_rieng:
            cat.append({**goc, "ly_do_cat": f"rieng_tu:{ly_rieng}"})
            continue
        t = Ut.loc[r.ma_don_vi_ten]
        if pd.isna(t.ban):
            cat.append({**goc, "ly_do_cat": f"ten:{t.ly_do_cat}"})
            continue
        if r.dinh_tuyen == "mo_hinh":
            g = Ut.loc[r.ma_don_vi_gia_tri]
            tt = {"gia_tri_zh_A": bo_rac(g.A), "gia_tri_zh_B": bo_rac(g.B), "chrf_ab": g.chrf_ab, "diem_kiwi_A": g.kiwi_A, "diem_kiwi_B": g.kiwi_B,
                  "luat_truot": g.luat_truot}
            if pd.isna(g.ban):
                cat.append({**goc, **tt, "ly_do_cat": f"gia_tri:{g.ly_do_cat}"})
                continue
            gia_tri_zh, cach, tin_gt, nhac_gt = bo_rac(g.ban), g.cach_chon, g.do_tin_cay, g.co_nhac
        else:
            tt = {"gia_tri_zh_A": None, "gia_tri_zh_B": None, "chrf_ab": None, "diem_kiwi_A": None, "diem_kiwi_B": None, "luat_truot": ""}
            gia_tri_zh, cach, tin_gt, nhac_gt = r.gia_tri_zh_quy_tac, f"quy_tac_{r.dinh_tuyen}", "cao", ""
        tin = min([t.do_tin_cay, tin_gt], key=lambda x: HANG_TIN[x])
        rows.append({**goc, **tt, "name_zh": t.ban, "gia_tri_zh": gia_tri_zh, "cach_chon": cach, "do_tin_cay": tin,
                     "co_nhac": nhap(nhac_gt, ";".join(f"ten:{x}" for x in t.co_nhac.split(";") if x))})
    D = pd.DataFrame(rows, columns=COT_DAY_DU)
    CR = pd.DataFrame(cat)
    nhan = time.strftime("%Y%m%d_%H%M")
    f_du = thu_muc_ra / f"bo_song_ngu_2_day_du_{nhan}.parquet"
    f_cb = thu_muc_ra / f"bo_song_ngu_2_cong_bo_{nhan}.parquet"
    f_cr = thu_muc_ra / f"cat_rieng_{nhan}.parquet"
    D.to_parquet(f_du, index=False)
    D[C.COT_CONG_BO].to_parquet(f_cb, index=False)      # bản công bố: hàm chọn cột cố định, không chọn tay
    CR.to_parquet(f_cr, index=False)
    U.to_parquet(thu_muc_ra / f"don_vi_da_chon_{nhan}.parquet", index=False)
    pd.DataFrame(nhay).to_csv(thu_muc_ra / f"do_nhay_nguong_{nhan}.csv", index=False, encoding="utf-8-sig")
    phan_bo = U[["loai", "kiwi_A", "kiwi_B", "chrf_ab"]].groupby("loai").describe(percentiles=[.05, .1, .25, .5, .75, .9]).T
    phan_bo.to_csv(thu_muc_ra / f"phan_bo_diem_{nhan}.csv", encoding="utf-8-sig")

    chuan_bi = json.loads((thu_muc_ra / "the_chuan_bi.json").read_text(encoding="utf-8"))
    the = {
        "buoc": "song_ngu_2.hoan_tat (5, 6b, 6c, 7)", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nguong": {"tau": tau, "kappa": kappa, "kappa_ngan": kappa_ngan, "kappa_cat": kappa_cat, "kappa_cat_ngan": kappa_cat_ngan, "delta": delta, "uu_tien": uu_tien,
                   "chi_doi_khi_truot": chi_doi_khi_truot, "do_dai_ngan": C.DO_DAI_NGAN, "do_dai_nguon_ngan": C.DO_DAI_NGUON_NGAN,
                   "q_do_dai": list(C.Q_DO_DAI),
                   "ghi_chu": "chọn từ phân bố điểm thật của bộ Tiki, không có nhãn đúng sai; Kiwi chỉ quyết định độ tin cậy và việc cất riêng khi "
                              "cả hai bản dưới sàn; bản B chỉ dùng khi bản A trượt luật; so hai bản bằng chuan_hoa_zh"},
        "khoang_do_dai_trung_tren_viet": {str(k): v for k, v in khoang.items()},
        "mo_hinh": C.MO_HINH, "huong": C.HUONG,
        "go_nhan": {"ghi_chu": "bước 5a: gỡ nhãn tên thuộc tính mô hình tự thêm vào đầu bản dịch giá trị (chỉ khi nhãn trùng bản dịch tên của chính mô hình đó)",
                    **go},
        "co_dinh": {"ghi_chu": "bước 6c: tên trong từ điển đã duyệt dùng bản từ điển; giá trị theo thứ tự: bảng có/không theo thuộc tính, từ điển giá trị "
                               "đã duyệt, quy tắc thời hạn; độ tin cậy cao",
                    "tu_dien_ten": {"file": C.FILE_TU_DIEN_TEN.name, "sha256": ma_bam(C.FILE_TU_DIEN_TEN) if tu_dien is None else "truyen_vao",
                                    "so_muc_da_duyet": len(tu_dien_ten), "xung_dot_hoa_thuong": xd_ten},
                    "tu_dien_gia_tri": {"file": C.FILE_TU_DIEN_GIA_TRI.name, "khoa": "(ten_vi, gia_tri_vi)", "sha256": ma_bam(C.FILE_TU_DIEN_GIA_TRI) if tu_dien is None else "truyen_vao",
                                        "so_muc_da_duyet": len(tu_dien_gt), "xung_dot_hoa_thuong": xd_gt},
                    "co_khong": {"file": C.FILE_CO_KHONG.name, "sha256": ma_bam(C.FILE_CO_KHONG) if tu_dien is None else "truyen_vao",
                                 "so_muc_da_duyet": len(co_khong), "xung_dot_hoa_thuong": xd_ck},
                    "don_vi": {c: {"don_vi": int(r.don_vi), "cap": int(r.cap)} for c, r in dem_co_dinh.iterrows()}},
        "chuan_bi": chuan_bi, "kaggle": doc_the(thu_muc_kaggle, thu_muc_the_phu),
        "don_vi": {"tong": int(len(U)), "giu": int(U.ban.notna().sum()), "cach_chon": U.cach_chon.value_counts().to_dict(),
                   "ly_do_cat": U.ly_do_cat.value_counts().to_dict()},
        "cap": {"giu": int(len(D)), "cat_rieng": int(len(CR)), "do_tin_cay": D.do_tin_cay.value_counts().to_dict(),
                "cach_chon": D.cach_chon.value_counts().to_dict()},
        "file": {"day_du": f_du.name, "cong_bo": f_cb.name, "cat_rieng": f_cr.name},
    }
    (thu_muc_ra / f"the_nguon_goc_{nhan}.json").write_text(json.dumps(the, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    in_ra(f"Ngưỡng: tau {tau}, độ tin cậy kappa {kappa} (nguồn ngắn {kappa_ngan}), cất riêng khi cả hai dưới {kappa_cat} (nguồn ngắn {kappa_cat_ngan}), ưu tiên {uu_tien}, "
          f"{'chỉ đổi sang bản kia khi bản ưu tiên trượt luật' if chi_doi_khi_truot else 'luật cũ (đổi theo điểm)'}")
    in_ra(f"Đơn vị giữ {the['don_vi']['giu']:,}/{the['don_vi']['tong']:,}; lý do cất riêng: {the['don_vi']['ly_do_cat']}")
    in_ra(f"Cặp giữ {len(D):,}, cất riêng {len(CR):,}; độ tin cậy {the['cap']['do_tin_cay']}")
    in_ra(f"Đã ghi {f_du.name}, {f_cb.name}, {f_cr.name} và thẻ nguồn gốc ({time.time() - t0:.0f} giây)")
    return D, CR, U, the
