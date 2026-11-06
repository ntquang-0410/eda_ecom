# Bước 5, 6b, 7 (laptop, CPU): đọc kết quả tải về từ Kaggle (bản A, bản B, điểm CometKiwi), kiểm luật, chọn một bản Việt
# cho mỗi đơn vị, ghép về từng cặp, xuất bản đầy đủ (18 cột), bản công bố (9 cột), file cất riêng và thẻ nguồn gốc.
# Chạy lại bao nhiêu lần cũng được (chỉnh ngưỡng tau, kappa, delta) mà không phải chạy lại GPU. Mỗi lần ghi file mới có dấu thời gian.
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from . import cau_hinh as C
from .chon_ban import DO_DAI_ZH_NGAN, chon
from .luat_kiem import do_dai_bat_thuong, doc_tu_dien, khoang_do_dai, kiem_mot
from .van_ban import chrf, chuan_hoa, co_han
from doi_chieu import doi_can  # noqa: E402  (src/doi_chieu.py; luat_kiem đã thêm src vào sys.path)
from .quy_tac_gia_tri import ban_quy_tac

COT_DAY_DU = ["pair_id", "product_id", "category", "nhom_trung", "ly_do_goc", "name_zh", "gia_tri_zh", "name_vi", "gia_tri_vi",
              "gia_tri_vi_A", "gia_tri_vi_B", "cach_chon", "do_tin_cay", "chrf_ab", "diem_kiwi_A", "diem_kiwi_B", "luat_truot", "co_nhac"]
HANG_TIN = {"cao": 3, "vua": 2, "thap": 1}


def doc_lo(thu_muc, ten):
    """Gộp mọi lo_*.parquet trong các thư mục con tên `ten` (tải về từ Kaggle, có thể nhiều phiên bản chạy nối tiếp)."""
    files = sorted(Path(thu_muc).rglob(f"{ten}/lo_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["ma_don_vi"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("ma_don_vi", keep="last")


def doc_the(thu_muc):
    out = {}
    for f in sorted(Path(thu_muc).rglob("the_*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # thẻ hỏng không làm hỏng cả lần chạy
            out[f.stem] = {"loi_doc": str(e)}
    return out


def nhap(a, b):
    return ";".join(x for x in [a, b] if x)


def ban_co_dinh(loai, zh, tu_dien_ten, tu_dien_gt, co_khong=None, tu_dien_ten_bs=None, ten_zh=None):
    """Bước 6c: bản dịch cố định của một đơn vị, trả (bản, cách chọn) hoặc None. Tên thuộc tính: từ điển P6 đã duyệt.
    Giá trị: bảng có/không (C.CO_KHONG) trước, rồi từ điển P7 đã duyệt (anh chốt 21/10/2026). So khớp nguyên chuỗi Trung
    (bỏ khoảng trắng hai đầu)."""
    co_khong = C.CO_KHONG if co_khong is None else co_khong
    tu_dien_ten_bs = tu_dien_ten_bs or {}
    if not isinstance(zh, str):
        return None
    for z in dict.fromkeys([zh, zh.strip()]):
        if loai == "ten":
            if tu_dien_ten.get(z):
                return tu_dien_ten[z], "tu_dien_p6"
            if tu_dien_ten_bs.get(z):          # tên bổ sung anh duyệt 22/10/2026 (P6b); P6 đứng trước khi trùng
                return tu_dien_ten_bs[z], "tu_dien_p6b"
        else:
            if z in co_khong:
                return co_khong[z], "quy_tac_co_khong"
            if tu_dien_gt.get(z):
                return tu_dien_gt[z], "tu_dien_p7"
    if loai != "ten":
        return ban_quy_tac(ten_zh, zh)     # quy tắc mẫu (khoảng có 含, 像素 + 万), xếp sau từ điển đã duyệt
    return None


def ma_bam(file):
    p = Path(file)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def chay(thu_muc_kaggle, thu_muc_ra=C.THU_MUC_RA, tau=C.TAU, kappa=C.KAPPA, delta=C.DELTA, in_ra=print,
         kappa_ngan=C.KAPPA_NGAN, uu_tien=C.UU_TIEN, tu_dien=None, chi_doi_khi_truot=C.CHI_DOI_KHI_TRUOT):
    """tu_dien: None thì đọc P6, P7 thật (C.FILE_TU_DIEN_TEN, C.FILE_TU_DIEN_GIA_TRI); bộ thử truyền {"ten": {...}, "gia_tri": {...}}."""
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
        U = U.merge(df[["ma_don_vi", cot]] if cot in df.columns else pd.DataFrame(columns=["ma_don_vi", cot]),
                    on="ma_don_vi", how="left")
    in_ra(f"Đơn vị: {len(U):,}; có bản A {U.A.notna().sum():,}, bản B {U.B.notna().sum():,}, "
          f"điểm A {U.kiwi_A.notna().sum():,}, điểm B {U.kiwi_B.notna().sum():,}")

    # Bước 5a (anh duyệt 22/10/2026): bản Việt giữ số 斤 mà viết "cân" thì tự đổi ra kg (chia đôi) khi chắc chắn (doi_chieu.doi_can),
    # gắn cờ nhắc jin_tu_doi. Không đổi được (vd giá tính theo 斤, số không khớp một đối một) thì để luật jin_can đánh trượt.
    sua_jin = {}
    for x in ["A", "B"]:
        moi = [doi_can(g, v) if isinstance(v, str) else None for g, v in zip(U.van_ban_gui, U[x])]
        U[f"jin_tu_doi_{x}"] = [m is not None for m in moi]
        U[x] = [m if m is not None else v for m, v in zip(moi, U[x])]
        sua_jin[x] = int(sum(m is not None for m in moi))
    in_ra(f"Tự đổi 斤 viết 'cân' sang kg: A {sua_jin['A']}, B {sua_jin['B']} đơn vị")

    # Bước 5: luật kiểm
    if tu_dien is None:
        tu_dien_ten = doc_tu_dien(C.FILE_TU_DIEN_TEN, "ten_zh", "ten_chuan_tam")
        tu_dien_gt = doc_tu_dien(C.FILE_TU_DIEN_GIA_TRI, "gia_tri_zh", "gia_tri_chuan_tam")
        tu_dien_ten_bs = doc_tu_dien(C.FILE_TU_DIEN_TEN_BO_SUNG, "ten_zh", "ten_chuan_tam")
    else:
        tu_dien_ten, tu_dien_gt = tu_dien.get("ten", {}), tu_dien.get("gia_tri", {})
        tu_dien_ten_bs = tu_dien.get("ten_bo_sung", {})
    for x in ["A", "B"]:
        kq = [kiem_mot(z, g, v, tu_dien_ten if l == "ten" else tu_dien_gt) if isinstance(v, str) else (["thieu"], [])
              for z, g, v, l in zip(U.van_ban_zh, U.van_ban_gui, U[x], U.loai)]
        # mô hình dừng vì hết số token cho phép: bản dịch bị cắt ngang
        U[f"truot_{x}"] = [k[0] + ["bi_cat"] if d == "length" else k[0] for k, d in zip(kq, U[f"dung_{x}"])]
        U[f"nhac_{x}"] = [k[1] + (["jin_tu_doi"] if f else []) for k, f in zip(kq, U[f"jin_tu_doi_{x}"])]
    co = [(len(z), len(v)) for x in ["A", "B"] for z, v in zip(U.van_ban_gui, U[x]) if isinstance(v, str) and v.strip()]
    khoang = khoang_do_dai([a for a, _ in co], [b for _, b in co]) if co else {}
    for x in ["A", "B"]:
        U[f"truot_{x}"] = [t + ["do_dai"] if isinstance(v, str) and v.strip() and do_dai_bat_thuong(len(g), len(v), khoang)
                           and "do_dai" not in t else t for t, g, v in zip(U[f"truot_{x}"], U.van_ban_gui, U[x])]

    # Bước 6b: chọn một bản
    U["chrf_ab"] = [round(chrf(chuan_hoa(a), chuan_hoa(b)), 1) if isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()
                    else None for a, b in zip(U.A, U.B)]

    # Bước 6c (anh duyệt 21/10/2026): đơn vị có bản cố định (từ điển P6/P7 đã duyệt, bảng có/không) dùng bản đó, độ tin cậy cao,
    # không phụ thuộc bản máy hay ngưỡng (nên cũng đè trong bảng độ nhạy)
    co_dinh = [ban_co_dinh(l, z, tu_dien_ten, tu_dien_gt, tu_dien_ten_bs=tu_dien_ten_bs, ten_zh=n)
               for l, z, n in zip(U.loai, U.van_ban_zh, U.name_zh)]

    def chon_het(t, k, d):
        kn = min(kappa_ngan, k)      # sàn cho chuỗi ngắn không cao hơn sàn chung (bảng độ nhạy đổi k)
        kq = [chon(a if isinstance(a, str) else None, b if isinstance(b, str) else None, ta, tb, na, nb, ka, kb, t, k, d,
                   C.DO_DAI_NGAN, cf, dai_zh=len(g), kappa_ngan=kn, uu_tien=uu_tien, chi_doi_khi_truot=chi_doi_khi_truot)
              for a, b, ta, tb, na, nb, ka, kb, cf, g in zip(U.A, U.B, U.truot_A, U.truot_B, U.nhac_A, U.nhac_B,
                                                            U.kiwi_A, U.kiwi_B, U.chrf_ab, U.van_ban_gui)]
        return [{**x, "ban": cd[0], "tu": "co_dinh", "cach_chon": cd[1], "do_tin_cay": "cao", "ly_do_cat": None} if cd else x
                for x, cd in zip(kq, co_dinh)]
    kq = chon_het(tau, kappa, delta)
    for cot in ["ban", "tu", "cach_chon", "do_tin_cay", "ly_do_cat"]:
        U[cot] = [k[cot] for k in kq]
    # cờ nhắc lấy theo bản được chọn; bản cố định không mang cờ nhắc của bản máy
    U["co_nhac"] = [";".join(na if t == "A" else nb) if t in ("A", "B") else "" for t, na, nb in zip(U.tu, U.nhac_A, U.nhac_B)]
    dem_co_dinh = U[U.tu == "co_dinh"].groupby("cach_chon").agg(don_vi=("ma_don_vi", "size"), cap=("so_cap", "sum"))
    in_ra("Bản cố định (6c): " + ", ".join(f"{c} {int(r.don_vi):,} đơn vị ({int(r.cap):,} cặp)" for c, r in dem_co_dinh.iterrows()))
    U["luat_truot"] = [nhap(f"A:{'|'.join(ta)}" if ta else "", f"B:{'|'.join(tb)}" if tb else "") for ta, tb in zip(U.truot_A, U.truot_B)]

    # Bảng độ nhạy để chọn ngưỡng: tỉ lệ đơn vị được giữ theo kappa và tau
    nhay = []
    for k in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for t in [40, 50, 60, 70, 80]:
            g = chon_het(t, k, delta)
            nhay.append({"kappa": k, "tau": t, "ti_le_giu": round(sum(x["ban"] is not None for x in g) / max(len(g), 1), 4)})

    # Bước 7: ghép về từng cặp
    Ut = U.set_index("ma_don_vi")
    rows, cat = [], []
    for r in B.itertuples(index=False):
        goc = {"pair_id": r.pair_id, "product_id": r.product_id, "category": r.category, "nhom_trung": r.nhom_trung,
               "ly_do_goc": r.ly_do_goc, "name_zh": r.name_zh, "gia_tri_zh": r.gia_tri_zh}
        if r.trang_thai_vao != "dung":
            cat.append({**goc, "ly_do_cat": (f"{r.trang_thai_vao}:{r.ly_do_goc}" + (":PII_SOT" if r.pii_sot else ""))})
            continue
        # tên thuộc tính
        if co_han(r.name_zh):
            t = Ut.loc[r.ma_don_vi_ten]
            if pd.isna(t.ban):
                cat.append({**goc, "ly_do_cat": f"ten:{t.ly_do_cat}"})
                continue
            name_vi, tin_ten, nhac_ten = t.ban, t.do_tin_cay, t.co_nhac
        else:
            name_vi, tin_ten, nhac_ten = r.name_zh, "cao", ""
        # giá trị
        if r.dinh_tuyen == "ten_hang_dai":
            cat.append({**goc, "ly_do_cat": "ten_hang_dai"})
            continue
        if r.dinh_tuyen == "ten_hang" and isinstance(r.gia_tri_zh, str) and r.gia_tri_zh.strip() in C.TEN_HANG_VO_NGHIA:
            cat.append({**goc, "ly_do_cat": "ten_hang_vo_nghia"})      # vd thương hiệu "是" (anh chốt 21/10/2026)
            continue
        if r.dinh_tuyen == "mo_hinh":
            g = Ut.loc[r.ma_don_vi_gia_tri]
            tt = {"gia_tri_vi_A": g.A, "gia_tri_vi_B": g.B, "chrf_ab": g.chrf_ab, "diem_kiwi_A": g.kiwi_A,
                  "diem_kiwi_B": g.kiwi_B, "luat_truot": g.luat_truot}
            if pd.isna(g.ban):
                cat.append({**goc, **tt, "ly_do_cat": f"gia_tri:{g.ly_do_cat}"})
                continue
            gia_tri_vi, cach, tin_gt, nhac_gt = g.ban, g.cach_chon, g.do_tin_cay, g.co_nhac
        else:
            tt = {"gia_tri_vi_A": None, "gia_tri_vi_B": None, "chrf_ab": None, "diem_kiwi_A": None, "diem_kiwi_B": None, "luat_truot": ""}
            gia_tri_vi, cach, tin_gt, nhac_gt = r.gia_tri_vi_quy_tac, f"quy_tac_{r.dinh_tuyen}", "cao", ""
        tin = min([tin_ten, tin_gt], key=lambda x: HANG_TIN[x])
        rows.append({**goc, **tt, "name_vi": name_vi, "gia_tri_vi": gia_tri_vi, "cach_chon": cach, "do_tin_cay": tin,
                     "co_nhac": nhap(nhac_gt, ";".join(f"ten:{x}" for x in nhac_ten.split(";") if x))})
    D = pd.DataFrame(rows, columns=COT_DAY_DU)
    CR = pd.DataFrame(cat)
    nhan = time.strftime("%Y%m%d_%H%M")
    f_du = thu_muc_ra / f"bo_song_ngu_1_day_du_{nhan}.parquet"
    f_cb = thu_muc_ra / f"bo_song_ngu_1_cong_bo_{nhan}.parquet"
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
        "buoc": "hoan_tat (5, 6b, 7)", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nguong": {"tau": tau, "kappa": kappa, "kappa_ngan": kappa_ngan, "delta": delta, "uu_tien": uu_tien,
                   "chi_doi_khi_truot": chi_doi_khi_truot,
                   "do_dai_ngan": C.DO_DAI_NGAN, "do_dai_zh_ngan": DO_DAI_ZH_NGAN,
                   "ghi_chu": "tạm; chọn từ phân bố điểm, không có nhãn đúng sai. delta là chênh điểm CometKiwi để bỏ mô hình ưu tiên (từ 15/10/2026). "
                              "Luật lap_tu từ 21/10/2026: phép tỉ lệ từ không trùng bỏ qua danh sách có số mục khớp nguồn"},
        "khoang_do_dai_viet_tren_trung": {str(k): v for k, v in khoang.items()},
        "mo_hinh": C.MO_HINH,
        "jin_tu_doi": {"ghi_chu": "bước 5a: bản Việt giữ số 斤 mà viết 'cân' được đổi ra kg (chia đôi) khi chắc chắn; số đơn vị đã đổi",
                       **sua_jin},
        "co_dinh": {"ghi_chu": "bước 6c, anh duyệt 21/10/2026: tên trong P6 (mục anh đã duyệt) dùng bản từ điển; giá trị trong bảng "
                               "co_khong dùng bảng, không có thì giá trị trong P7 đã duyệt dùng P7; độ tin cậy cao. Thương hiệu "
                               "trong ten_hang_vo_nghia bị cất riêng",
                    "ten_hang_vo_nghia": sorted(C.TEN_HANG_VO_NGHIA),
                    "tu_dien_ten": {"file": C.FILE_TU_DIEN_TEN.name, "sha256": ma_bam(C.FILE_TU_DIEN_TEN) if tu_dien is None else "truyen_vao",
                                    "so_muc_da_duyet": len(tu_dien_ten)},
                    "tu_dien_ten_bo_sung": {"file": C.FILE_TU_DIEN_TEN_BO_SUNG.name,
                                            "sha256": ma_bam(C.FILE_TU_DIEN_TEN_BO_SUNG) if tu_dien is None else "truyen_vao",
                                            "so_muc_da_duyet": len(tu_dien_ten_bs)},
                    "tu_dien_gia_tri": {"file": C.FILE_TU_DIEN_GIA_TRI.name,
                                        "sha256": ma_bam(C.FILE_TU_DIEN_GIA_TRI) if tu_dien is None else "truyen_vao",
                                        "so_muc_da_duyet": len(tu_dien_gt)},
                    "co_khong": C.CO_KHONG,
                    "don_vi": {c: {"don_vi": int(r.don_vi), "cap": int(r.cap)} for c, r in dem_co_dinh.iterrows()}},
        "chuan_bi": chuan_bi, "kaggle": doc_the(thu_muc_kaggle),
        "don_vi": {"tong": int(len(U)), "giu": int(U.ban.notna().sum()), "cach_chon": U.cach_chon.value_counts().to_dict(),
                   "ly_do_cat": U.ly_do_cat.value_counts().to_dict()},
        "cap": {"giu": int(len(D)), "cat_rieng": int(len(CR)), "do_tin_cay": D.do_tin_cay.value_counts().to_dict(),
                "cach_chon": D.cach_chon.value_counts().to_dict()},
        "file": {"day_du": f_du.name, "cong_bo": f_cb.name, "cat_rieng": f_cr.name},
    }
    (thu_muc_ra / f"the_nguon_goc_{nhan}.json").write_text(json.dumps(the, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    in_ra(f"Ngưỡng: tau {tau}, kappa {kappa} (chuỗi Trung ngắn {kappa_ngan}), delta {delta}, ưu tiên {uu_tien}, "
          f"{'chỉ đổi sang bản kia khi bản ưu tiên trượt luật' if chi_doi_khi_truot else 'luật cũ (đổi theo điểm)'}")
    in_ra(f"Đơn vị giữ {the['don_vi']['giu']:,}/{the['don_vi']['tong']:,}; lý do cất riêng: {the['don_vi']['ly_do_cat']}")
    in_ra(f"Cặp giữ {len(D):,}, cất riêng {len(CR):,}; độ tin cậy {the['cap']['do_tin_cay']}")
    in_ra(f"Đã ghi {f_du.name}, {f_cb.name}, {f_cr.name} và thẻ nguồn gốc ({time.time() - t0:.0f} giây)")
    return D, CR, U, the
