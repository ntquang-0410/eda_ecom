# -*- coding: utf-8 -*-
"""Bước 1-3 của bộ song ngữ thứ nhất (laptop, CPU). Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_1\\chay_chuan_bi.py

Ra: data/processed/song_ngu_1/bang_cap.parquet, the_chuan_bi.json và thư mục vao_kaggle/ (tải lên Kaggle thành Dataset riêng tư).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from song_ngu_1 import cau_hinh as C  # noqa: E402
from song_ngu_1.chuan_bi import chay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cap", default=str(C.FILE_CAP), help="file cặp desc_v1 (pairs)")
ap.add_argument("--san_pham", default=str(C.FILE_SAN_PHAM), help="file sản phẩm desc_v1 (chỉ lấy nhom_trung)")
ap.add_argument("--ra", default=str(C.THU_MUC_RA))
a = ap.parse_args()
chay(Path(a.cap), Path(a.san_pham), Path(a.ra))
