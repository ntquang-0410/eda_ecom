# -*- coding: utf-8 -*-
"""Bước 1 của bộ song ngữ 4 (1688 Trung -> Việt, hiệu đính cả cụm theo sản phẩm), chạy trên laptop. Chạy từ gốc repo:

    <python có pandas, pypinyin> scripts\\song_ngu_4\\chay_chuan_bi.py

Đọc bộ 1 (bang_cap, đơn vị đã chọn, bản công bố đầy đủ 20261022_2126) và tên sản phẩm (data/snapshot/bilingual_zh_vi.parquet), ghi
data/processed/song_ngu_4/: cap_vao.parquet (một dòng một cặp), the_chuan_bi.json, vao_kaggle/ (thư mục tải lên Kaggle:
cum_can_dich_zv.parquet, cau_hinh_kaggle.json, dich.py, dich_cum_zv.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_4.chuan_bi import chay  # noqa: E402

chay()
