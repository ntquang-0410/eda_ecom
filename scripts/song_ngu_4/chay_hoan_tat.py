# -*- coding: utf-8 -*-
"""Bước cuối của bộ 4 (laptop): ghép kết quả Gemma về cặp theo luật chặt, gắn nhãn độ tin cậy, dựng bảng cặp và bảng sản phẩm.

    .venv\\Scripts\\python.exe -X utf8 scripts\\song_ngu_4\\chay_hoan_tat.py <thư mục Gemma tải về> [--qwen <thư mục Qwen dự phòng>] [--mau_thu]

--mau_thu: chỉ ghép các cụm của mẫu thử 200 sản phẩm (kiểm code trên kết quả lần thử). Không có cờ này: cả bộ, dừng nếu còn cụm chưa dịch.
--qwen: kết quả Qwen chạy dự phòng trên một phần số cụm (anh chốt 26/10/2026): Qwen chỉ thay chỗ Gemma trượt luật và có bản nháp.
Ghi data/processed/song_ngu_4/: cap_cum_<nhãn>.parquet (bảng cặp đầy đủ cột), san_pham_cum_<nhãn>.parquet, tom_tat_cum_<nhãn>.json
(nhãn có hậu tố _thu khi --mau_thu).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_4 import cau_hinh as C  # noqa: E402
from song_ngu_4.hoan_tat import ghep  # noqa: E402

thu = "--mau_thu" in sys.argv
qw = Path(sys.argv[sys.argv.index("--qwen") + 1]).resolve() if "--qwen" in sys.argv else None
ga = Path([a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--qwen"][0]).resolve()
P = pd.read_parquet(C.THU_MUC_RA / "cap_vao.parquet")
if thu:
    CUM = pd.read_parquet(C.THU_MUC_RA / "vao_kaggle" / "cum_can_dich_zv.parquet", columns=["ma_cum", "mau_thu"])
    P = P[P.ma_cum.isin(CUM[CUM.mau_thu].ma_cum)]
t0 = time.time()
G, SP, tk = ghep(P, ga, thu_muc_qwen=qw)
nhan = time.strftime("%Y%m%d_%H%M") + ("_thu" if thu else "")
G.to_parquet(C.THU_MUC_RA / f"cap_cum_{nhan}.parquet", index=False)
SP.to_parquet(C.THU_MUC_RA / f"san_pham_cum_{nhan}.parquet", index=False)
tk.update(nhan=nhan, thu_muc_gemma=str(ga.relative_to(ROOT)) if ga.resolve().is_relative_to(ROOT) else ga.name,
          thu_muc_qwen=(str(qw.relative_to(ROOT)) if qw.is_relative_to(ROOT) else qw.name) if qw else None, giay=round(time.time() - t0))
(C.THU_MUC_RA / f"tom_tat_cum_{nhan}.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(tk, ensure_ascii=False, indent=1))
