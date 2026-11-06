# -*- coding: utf-8 -*-
"""Bước 3 của bộ song ngữ 3 bản v2 (laptop): ghép kết quả hiệu đính cả cụm về từng cặp theo luật chặt (ket_qua_v2), in tóm tắt, ghi bảng
so sánh, và dựng file CHẤM MÙ cho các trường bản v2 (chọn A) khác bản dịch đơn vị bộ 2 (bỏ các chỗ chỉ khác do đổi 'gr' -> 'g' ở giá trị
khoá: đó là quy tắc cố định, không cần chấm). Lý do sửa của mô hình ghi ở file riêng để lúc chấm không biết bản nào là bản sửa.
Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\chay_ket_qua_v2.py [thư mục tải về, mặc định data/processed/song_ngu_3/tu_kaggle_v2_thu]
"""
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_3 import cau_hinh as C  # noqa: E402
from song_ngu_3.ket_qua_v2 import giong, lap_bang_v2, tom_tat_v2  # noqa: E402

thu_muc = Path(sys.argv[1]) if len(sys.argv) > 1 else C.THU_MUC_RA / "tu_kaggle_v2_thu"
CUM = pd.read_parquet(C.THU_MUC_RA / "vao_kaggle" / "cum_can_dich_v2.parquet")
CUM = CUM[CUM.mau_thu]
P = lap_bang_v2(CUM, thu_muc)
tt = tom_tat_v2(CUM, P)
print(json.dumps(tt, ensure_ascii=False, indent=1))
nhan = time.strftime("%Y%m%d_%H%M")
ra = C.THU_MUC_RA / f"cham_mau_v2_{nhan}"
ra.mkdir(parents=True, exist_ok=True)
P.to_parquet(C.THU_MUC_RA / f"so_sanh_mau_v2_{nhan}.parquet", index=False)
(C.THU_MUC_RA / f"tom_tat_mau_v2_{nhan}.json").write_text(json.dumps(tt, ensure_ascii=False, indent=1), encoding="utf-8")

# trường cần chấm: bản cuối (chọn A) khác bản đơn vị, trừ trường khoá (khoá = bản bộ 2, chỉ có thể khác do đổi gr)
viec = []
for r in P.itertuples(index=False):
    for f, cot_vi in (("n", "name_vi"), ("v", "gia_tri_vi")):
        cuoi, dv, nguon = getattr(r, f"{f}_cuoi_A"), getattr(r, f"{f}_don_vi"), getattr(r, f"{f}_nguon_A")
        if nguon == "khoa" or giong(cuoi, dv):
            continue
        viec.append({"pair_id": int(r.pair_id), "truong": f, "product_id": r.product_id, "category": r.category, "ten_vi": r.name_vi,
                     "nguon_vi": getattr(r, cot_vi), "v2": cuoi, "don_vi": dv if isinstance(dv, str) else None, "nguon_v2": nguon,
                     "ly_do_sua": getattr(r, "c_A") or ""})
print("Trường cần chấm (v2 khác bản đơn vị):", len(viec), "| theo nguồn:", pd.Series([v["nguon_v2"] for v in viec]).value_counts().to_dict())
tieu_de = dict(zip(CUM.product_id, CUM.tieu_de))
ngu_canh = {p: "; ".join(f"{n}: {v[:70]}" for n, v in zip(g.name_vi, g.gia_tri_vi)) for p, g in P.groupby("product_id", sort=False)}
rng = random.Random(11)
dong, khoa = [], {}
for k, v in enumerate(sorted(viec, key=lambda x: (x["product_id"], x["pair_id"], x["truong"])), 1):
    b1, b2, key = (v["v2"], v["don_vi"] or "(không có ở bộ 2)", "1=v2") if rng.random() < 0.5 else (v["don_vi"] or "(không có ở bộ 2)", v["v2"], "1=don_vi")
    khoa[k] = {**{x: v[x] for x in ("pair_id", "truong", "nguon_v2", "ly_do_sua")}, "khoa": key}
    dong.append(f"#{k} [{v['category']}] SP {v['product_id']} | TÊN SP: {tieu_de.get(v['product_id'], '')}\n  NGỮ CẢNH: {ngu_canh[v['product_id']][:700]}\n"
                f"  >> {'TÊN THUỘC TÍNH' if v['truong'] == 'n' else v['ten_vi']}: {v['nguon_vi']}\n  Bản 1: {b1}\n  Bản 2: {b2}\n")
(ra / "cham_mu.txt").write_text("\n".join(dong), encoding="utf-8")
(ra / "khoa_KHONG_XEM.json").write_text(json.dumps(khoa, ensure_ascii=False, indent=0), encoding="utf-8")
print("Đã ghi", ra.name, "(cham_mu.txt, khoa_KHONG_XEM.json)")
