# -*- coding: utf-8 -*-
"""Bước cuối của bộ song ngữ 3 (laptop): ghép kết quả dịch cả cụm của hai mô hình thành bảng cặp và bảng sản phẩm. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\chay_hoan_tat_cum.py [thư mục Gemma] [thư mục Qwen] [--thu]

Mặc định đọc chạy thật: data/processed/song_ngu_3/tu_kaggle_v2_that_gemma và tu_kaggle_v2_that_qwen, toàn bộ cụm. --thu: chỉ 201 cụm mẫu thử.
Ghi so_sanh_cum_<nhãn>.parquet (cặp đầy đủ, mọi cột để kiểm), cap_cum_<nhãn>.parquet, san_pham_cum_<nhãn>.parquet, tom_tat_cum_<nhãn>.json.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_3 import cau_hinh as C  # noqa: E402
from song_ngu_3.hoan_tat import ghep  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
thu = "--thu" in sys.argv
gem = Path(args[0]) if args else C.THU_MUC_RA / "tu_kaggle_v2_that_gemma"
qw = Path(args[1]) if len(args) > 1 else C.THU_MUC_RA / "tu_kaggle_v2_that_qwen"
CUM = pd.read_parquet(C.THU_MUC_RA / "vao_kaggle" / "cum_can_dich_v2.parquet")
if thu:
    CUM = CUM[CUM.mau_thu]
t0 = time.time()
P, SP, tk = ghep(CUM, {"A": gem, "B": qw})
nhan = time.strftime("%Y%m%d_%H%M")
P.to_parquet(C.THU_MUC_RA / f"so_sanh_cum_{nhan}.parquet", index=False)
P[P.name_zh.notna() & P.gia_tri_zh.notna()][["pair_id", "product_id", "category", "name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh", "do_tin_cay",
                                             "nguon_ban_dich", "nguon_ten", "nguon_gia_tri", "dong_thuan", "do_tin_cay_bo_2"]].to_parquet(
    C.THU_MUC_RA / f"cap_cum_{nhan}.parquet", index=False)
SP.to_parquet(C.THU_MUC_RA / f"san_pham_cum_{nhan}.parquet", index=False)
(C.THU_MUC_RA / f"tom_tat_cum_{nhan}.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1), encoding="utf-8")
tk["do_tin_cay_cap"] = P[P.name_zh.notna() & P.gia_tri_zh.notna()].do_tin_cay.value_counts().to_dict()
tk["do_tin_cay_san_pham"] = SP.do_tin_cay.value_counts().to_dict()
(C.THU_MUC_RA / f"tom_tat_cum_{nhan}.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(tk, ensure_ascii=False, indent=1))
print(f"Đã ghi *_cum_{nhan} ({time.time() - t0:.0f} giây)")
