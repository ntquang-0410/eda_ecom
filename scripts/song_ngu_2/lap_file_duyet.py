# -*- coding: utf-8 -*-
"""Lập ba file để anh duyệt bản dịch cố định của bộ Tiki (cùng kiểu P6/P7 của bộ 1), từ lần chạy hoàn tất mới nhất (don_vi_da_chon_*.parquet):

    P6_ten_thuoc_tinh_tiki.csv   120 tên thuộc tính nhiều cặp nhất (phủ ~98% số cặp), khoá ten_vi
    P7_gia_tri_tiki.csv          300 giá trị nhiều cặp nhất (không kể có/không và thời hạn đã có quy tắc), khoá (ten_vi, gia_tri_vi)
    P8_co_khong_tiki.csv         mọi giá trị có/không theo từng thuộc tính, khoá (ten_vi, gia_tri_vi)

Cột anh_duyet để TRỐNG: trống là chưa duyệt (không áp dụng); OK là đồng ý bản gợi ý (goi_y_claude nếu có, không thì ban_chuan_tam); chữ khác là bản anh sửa.
Không ghi đè file đã có (anh có thể đã duyệt dở): file có rồi thì ghi thêm hậu tố _moi. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\lap_file_duyet.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_2 import cau_hinh as C  # noqa: E402

N_TEN, N_GIA_TRI = 120, 300
CO_KHONG = {"có", "không", "không có", "có bảo hành", "không bảo hành"}
COT_CUOI = ["goi_y_claude", "ly_do_claude", "anh_duyet", "anh_ghi_chu"]


def ghi(df, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        file = file.with_name(file.stem + "_moi" + file.suffix)
    df.to_csv(file, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {file.name}: {len(df)} dòng")


def kiwi(x):
    return "" if pd.isna(x) else f"{x:.2f}"


def main():
    U = pd.read_parquet(sorted(glob.glob(str(C.THU_MUC_RA / "don_vi_da_chon_*.parquet")))[-1])
    U["ban_tam"] = U.ban.where(U.ban.notna(), U.A)
    U["chuan_tam_cach"] = U.cach_chon.fillna("")
    # --- P6: tên ---
    T = U[U.loai == "ten"].sort_values("so_cap", ascending=False).head(N_TEN)
    P6 = pd.DataFrame({"ten_vi": T.van_ban_nguon, "so_cap": T.so_cap, "ban_A": T.A, "kiwi_A": T.kiwi_A.map(kiwi), "ban_B": T.B,
                       "kiwi_B": T.kiwi_B.map(kiwi), "cach_chon_hien_tai": T.chuan_tam_cach, "ban_chuan_tam": T.ban_tam})
    # --- P8: có/không theo thuộc tính ---
    G = U[(U.loai == "gia_tri") & (U.tu != "co_dinh")]
    yn = G[G.van_ban_nguon.str.strip().str.lower().isin(CO_KHONG)].sort_values("so_cap", ascending=False)
    P8 = pd.DataFrame({"ten_vi": yn.name_nguon, "gia_tri_vi": yn.van_ban_nguon, "so_cap": yn.so_cap, "ban_A": yn.A, "ban_B": yn.B,
                       "cach_chon_hien_tai": yn.chuan_tam_cach, "ban_chuan_tam": yn.ban_tam})
    # --- P7: giá trị hay gặp, đã bỏ có/không và phần có quy tắc thời hạn ---
    v = G[~G.van_ban_nguon.str.strip().str.lower().isin(CO_KHONG)].sort_values("so_cap", ascending=False).head(N_GIA_TRI)
    P7 = pd.DataFrame({"ten_vi": v.name_nguon, "gia_tri_vi": v.van_ban_nguon, "so_cap": v.so_cap, "ban_A": v.A, "kiwi_A": v.kiwi_A.map(kiwi),
                       "ban_B": v.B, "kiwi_B": v.kiwi_B.map(kiwi), "cach_chon_hien_tai": v.chuan_tam_cach, "ban_chuan_tam": v.ban_tam})
    for df in (P6, P7, P8):
        for c in COT_CUOI:
            df[c] = ""
    tong_ten = U[U.loai == "ten"].so_cap.sum()
    tong_gt = G.so_cap.sum()
    print(f"P6: {len(P6)} tên phủ {P6.so_cap.sum() / tong_ten * 100:.1f}% cặp tên; P7: {len(P7)} giá trị phủ {P7.so_cap.sum() / tong_gt * 100:.1f}% "
          f"cặp giá trị chưa cố định; P8: {len(P8)} cặp (tên, có/không) phủ {P8.so_cap.sum():,} cặp")
    ghi(P6, C.FILE_TU_DIEN_TEN)
    ghi(P7, C.FILE_TU_DIEN_GIA_TRI)
    ghi(P8, C.FILE_CO_KHONG)


if __name__ == "__main__":
    main()
