# -*- coding: utf-8 -*-
"""Bước 5, 6b, 6c, 7 của bộ song ngữ thứ hai (Tiki Việt -> Trung), laptop (CPU), sau khi tải đầu ra Kaggle về. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\chay_hoan_tat.py --kaggle data\\processed\\song_ngu_2\\tu_kaggle_that_qwen --the_phu data\\processed\\song_ngu_2\\tu_kaggle_that_gemma

--kaggle: thư mục chứa ra_tiki (api_tuy_chon, qwen3_14b, kiwi_*); --the_phu: nơi có thẻ API của Gemma (không bắt buộc).
Chỉnh ngưỡng thì thêm --kappa 0.4 --tau 60; mỗi lần ghi file mới có dấu thời gian, không ghi đè lần trước.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_2 import cau_hinh as C  # noqa: E402
from song_ngu_2.hoan_tat import chay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kaggle", required=True, help="thư mục chứa đầu ra Kaggle đã tải về (giải nén)")
ap.add_argument("--the_phu", default=None, help="thư mục có thêm thẻ (thẻ API của Gemma)")
ap.add_argument("--ra", default=str(C.THU_MUC_RA))
ap.add_argument("--tau", type=float, default=C.TAU)
ap.add_argument("--kappa", type=float, default=C.KAPPA)
ap.add_argument("--kappa_ngan", type=float, default=C.KAPPA_NGAN)
ap.add_argument("--delta", type=float, default=C.DELTA)
ap.add_argument("--uu_tien", default=C.UU_TIEN, choices=["A", "B"])
ap.add_argument("--luat_cu", action="store_true", help="luật chọn cũ (đổi sang bản kia theo điểm CometKiwi) để đối chiếu")
a = ap.parse_args()
chay(Path(a.kaggle), Path(a.ra), a.tau, a.kappa, a.delta, kappa_ngan=a.kappa_ngan, uu_tien=a.uu_tien,
     chi_doi_khi_truot=False if a.luat_cu else C.CHI_DOI_KHI_TRUOT, thu_muc_the_phu=a.the_phu)
