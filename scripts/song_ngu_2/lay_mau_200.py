# -*- coding: utf-8 -*-
"""Lấy mẫu 200 cặp phân tầng của bộ song ngữ 2 (Tiki Việt -> Trung) để chấm tay. Ra hai file JSON (mẫu và mô tả nhóm) cho dung_excel_mau.py.
Chạy từ gốc repo bằng .venv của dự án:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\lay_mau_200.py <nhãn bản, ví dụ 20261023_1557> <thư mục ra>
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
NHAN = sys.argv[1]
RA = Path(sys.argv[2])
D = pd.read_parquet(ROOT / "data" / "processed" / "song_ngu_2" / f"bo_song_ngu_2_day_du_{NHAN}.parquet")
NHOM = [  # (mã, mô tả, các cách chọn, số mẫu)
    ("M1", "Máy dịch, hai mô hình ra giống hệt", ["giong_het"], 30),
    ("M2", "Máy dịch, hai bản gần giống hoặc khác chữ cùng nghĩa (lấy bản Gemma)", ["gan_giong_A", "khac_chu_cung_nghia_A"], 60),
    ("M3", "Máy dịch, chỉ một bản qua luật hoặc Gemma dưới sàn điểm (độ tin cậy thấp)", ["chi_A", "chi_B", "uu_tien_duoi_san_A"], 60),
    ("Q1", "Từ điển giá trị đã duyệt (P7)", ["tu_dien_p7"], 20),
    ("Q2", "Bảng có/không theo thuộc tính đã duyệt (P8)", ["quy_tac_co_khong"], 10),
    ("Q3", "Quy tắc thời hạn và khối lượng", ["quy_tac_thoi_han", "quy_tac_khoi_luong"], 10),
    ("Q4", "Chép nguyên (thương hiệu, số đo, số, mã model, cỡ chữ)",
     ["quy_tac_ten_hang", "quy_tac_do_luong", "quy_tac_chep_nguyen", "quy_tac_ma_model", "quy_tac_co_chu"], 10),
]
tat_ca = sum((n[2] for n in NHOM), [])
assert set(D.cach_chon.unique()) <= set(tat_ca), set(D.cach_chon.unique()) - set(tat_ca)
SEED = int(NHAN.replace("_", "")[:8])
ra, meta = [], []
for ma, mo_ta, cach, n in NHOM:
    p = D[D.cach_chon.isin(cach)]
    # lấy theo cặp (cặp lặp nhiều có cơ hội cao hơn, đúng tỉ lệ trong bộ) rồi bỏ trùng nội dung để anh khỏi đọc hai lần
    m = p.sample(frac=1, random_state=SEED).drop_duplicates(["name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh"]).head(n)
    meta.append({"nhom": ma, "mo_ta": mo_ta, "cach_chon": ", ".join(cach), "so_cap_trong_bo": int(len(p)), "so_mau": int(len(m))})
    ra.append(m.assign(nhom=ma))
M = pd.concat(ra).reset_index(drop=True)
M.insert(0, "stt", range(1, len(M) + 1))
cot = ["stt", "nhom", "category", "name_vi", "name_zh", "gia_tri_vi", "gia_tri_zh", "cach_chon", "do_tin_cay",
       "gia_tri_zh_A", "gia_tri_zh_B", "diem_kiwi_A", "diem_kiwi_B", "luat_truot", "pair_id"]
M = M[cot]
RA.mkdir(parents=True, exist_ok=True)
M.to_json(RA / "mau_200.json", orient="records", force_ascii=False)
json.dump({"nhan": NHAN, "seed": SEED, "tong_cap": int(len(D)), "nhom": meta, "do_tin_cay_trong_bo": D.do_tin_cay.value_counts().to_dict()},
          open(RA / "mau_200_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(M), M.nhom.value_counts().sort_index().to_dict(), M.do_tin_cay.value_counts().to_dict())
print(json.dumps(meta, ensure_ascii=False, indent=0))
