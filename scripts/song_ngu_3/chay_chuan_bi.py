# -*- coding: utf-8 -*-
"""Bước 1 của bộ song ngữ 3 (dịch cả cụm thuộc tính theo sản phẩm, Tiki Việt -> Trung), chạy trên laptop. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\chay_chuan_bi.py

Đọc data/processed/song_ngu_2/bang_cap.parquet, ghi data/processed/song_ngu_3/ (the_chuan_bi.json, vao_kaggle/ là thư mục tải lên
Kaggle: cum_can_dich.parquet, cau_hinh_kaggle.json, dich.py, dich_cum.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_3.chuan_bi import chay  # noqa: E402

chay()
