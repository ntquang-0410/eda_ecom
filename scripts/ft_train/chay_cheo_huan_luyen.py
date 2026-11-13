# -*- coding: utf-8 -*-
"""Thử chéo (12/11/2026, Nhật Anh duyệt, Prompt 956): bước 1 = huấn luyện lại hai adapter fold 0 ~11% title TRÊN TÀI KHOẢN 2.

Vì sao phải huấn luyện lại: adapter bản cuối (cum_cuoi, pair_cuoi) đã học cả mẫu eval, nên không dùng được cho phép thử chéo; adapter fold 0 chưa từng học mẫu eval
(cum_f0_p11, pair_f0_p11) có ở tài khoản 1 nhưng tài khoản 2 chỉ dùng cấu hình dưới đây, cùng cấu hình khảo sát tỉ lệ (Prompt 794): lr 2e-4, rank 64, 2 epoch, p_title 0,1074 (cụm) và 0,1086 (pair).

  set KAGGLE_NGUOI_DUNG=honghuynanh   (PowerShell: $env:KAGGLE_NGUOI_DUNG='honghuynanh')
  python -X utf8 chay_cheo_huan_luyen.py khoi    # khói ngắn trên nlp-group3, cả hai việc nối nhau
  python -X utf8 chay_cheo_huan_luyen.py that    # chạy thật: cụm ở nlp-group3, pair ở nlp-group3-pair, song song (mỗi cái một máy GPU)

Không chờ xong ở đây: đẩy lên rồi thoát; xem trạng thái bằng `kaggle kernels status`. Máy tắt thì phiên vẫn chạy.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dong_goi_kaggle as G  # noqa: E402

VIEC_CUM = {"ten": "cum_f0_p11", "nhanh": "cum", "fold": 0, "p_title": 0.1074, "ca_fold": False}
VIEC_PAIR = {"ten": "pair_f0_p11", "nhanh": "pair", "fold": 0, "p_title": 0.1086, "ca_fold": False}
KHOI = {"che_do": "b78", "ten_trang_thai": "thu_cheo", "viec": [VIEC_CUM, VIEC_PAIR], "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250,
        "rut_gon_sinh": 3, "so_train": 320, "gio_toi_da": 0.75}


def that(viec, ten_tt, gio=3.0):
    # trần 3 giờ cho một việc ~1,4 giờ chỉ để lần chạy lỗi không đốt quota; ngưỡng chrF chỉ để bắt lần chạy hỏng (mốc khảo sát 75-76), không dùng chọn gì
    return {"che_do": "b78", "ten_trang_thai": ten_tt, "viec": [viec], "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": gio,
            "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05}


def main():
    if G.NGUOI_DUNG != "honghuynanh":
        raise SystemExit(f"Phải đặt KAGGLE_NGUOI_DUNG=honghuynanh (đang là {G.NGUOI_DUNG}); thử chéo chỉ chạy ở tài khoản 2.")
    cd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cd == "khoi":
        G.chay_b5(KHOI, slug="nlp-group3")
    elif cd == "that":
        G.chay_b5(that(VIEC_CUM, "cheo_cum"), slug="nlp-group3")
        G.chay_b5(that(VIEC_PAIR, "cheo_pair"), slug="nlp-group3-pair")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
