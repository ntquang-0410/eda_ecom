# -*- coding: utf-8 -*-
"""Bước 1-3 của bộ song ngữ thứ hai (Tiki Việt -> Trung), chạy trên laptop (CPU). Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\chay_chuan_bi.py

Đọc data/cleaned/tiki/tiki_vi_cado.parquet (tải từ HF, kiểm SHA256), ghi data/processed/song_ngu_2/ (bang_cap.parquet,
the_chuan_bi.json, vao_kaggle/ là thư mục tải lên Kaggle).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_2.chuan_bi import chay  # noqa: E402

chay()
