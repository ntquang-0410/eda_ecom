# -*- coding: utf-8 -*-
"""Bước 5, 6b, 7 của bộ song ngữ thứ nhất (laptop, CPU), sau khi tải đầu ra Kaggle về. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_1\\chay_hoan_tat.py --kaggle data\\processed\\song_ngu_1\\tu_kaggle

Chỉnh ngưỡng thì thêm --tau 60 --kappa 0.5 --delta 0.05 (mặc định lấy ở src/song_ngu_1/cau_hinh.py) và chạy lại;
mỗi lần ghi file mới có dấu thời gian, không ghi đè lần trước. Xem do_nhay_nguong_*.csv và phan_bo_diem_*.csv để chọn ngưỡng.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from song_ngu_1 import cau_hinh as C  # noqa: E402
from song_ngu_1.hoan_tat import chay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kaggle", required=True, help="thư mục chứa đầu ra Kaggle đã tải về (giải nén)")
ap.add_argument("--ra", default=str(C.THU_MUC_RA))
ap.add_argument("--tau", type=float, default=C.TAU)
ap.add_argument("--kappa", type=float, default=C.KAPPA)
ap.add_argument("--delta", type=float, default=C.DELTA, help="chênh điểm CometKiwi để bỏ mô hình ưu tiên")
ap.add_argument("--kappa_ngan", type=float, default=C.KAPPA_NGAN, help="điểm sàn cho chuỗi Trung ngắn")
ap.add_argument("--uu_tien", default=C.UU_TIEN, choices=["A", "B"], help="mô hình ưu tiên khi hai bản cùng đạt")
ap.add_argument("--luat_cu", action="store_true",
                help="dùng luật chọn cũ (đổi sang bản kia theo điểm CometKiwi) thay cho C.CHI_DOI_KHI_TRUOT, để đối chiếu")
a = ap.parse_args()
chay(Path(a.kaggle), Path(a.ra), a.tau, a.kappa, a.delta, kappa_ngan=a.kappa_ngan, uu_tien=a.uu_tien,
     chi_doi_khi_truot=False if a.luat_cu else C.CHI_DOI_KHI_TRUOT)
