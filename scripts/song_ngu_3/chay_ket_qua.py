# -*- coding: utf-8 -*-
"""Bước 3 của bộ song ngữ 3 (laptop): đọc kết quả dịch cả cụm tải về từ Kaggle, tách về từng cặp, kiểm luật bộ 2, ghép bản dịch đơn vị
của bộ 2 trên cùng các cặp, in số liệu so sánh và ghi bảng so sánh. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\chay_ket_qua.py [thư mục tải về, mặc định data/processed/song_ngu_3/tu_kaggle_thu]
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_3 import cau_hinh as C  # noqa: E402
from song_ngu_3.ket_qua import lap_bang, tom_tat  # noqa: E402

thu_muc = Path(sys.argv[1]) if len(sys.argv) > 1 else C.THU_MUC_RA / "tu_kaggle_thu"
CUM = pd.read_parquet(C.THU_MUC_RA / "vao_kaggle" / "cum_can_dich.parquet")
CUM = CUM[CUM.mau_thu]
P = lap_bang(CUM, thu_muc)
tt = tom_tat(CUM, P)
print(json.dumps(tt, ensure_ascii=False, indent=1))
nhan = time.strftime("%Y%m%d_%H%M")
f = C.THU_MUC_RA / f"so_sanh_mau_{nhan}.parquet"
P.to_parquet(f, index=False)
(C.THU_MUC_RA / f"tom_tat_mau_{nhan}.json").write_text(json.dumps(tt, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã ghi", f.name)
