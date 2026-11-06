# -*- coding: utf-8 -*-
"""Rút mẫu CHẤM MÙ mới cho lần chạy thật bộ 3 (24/10/2026). Mẫu mới, không trùng 200 sản phẩm thử. Các tầng (độ dài giá trị <= 250 ký tự để chấm được):
  S1 chỉ A sửa (so bản cuối với nháp), S2 cả hai sửa giống nhau, S3 cả hai sửa khác nhau (nháp, A, B), S4 chỉ B sửa (B có hơn nháp không,
  vì bản cuối vẫn là nháp), S5 cả hai giữ nháp ở giá trị không khoá (chấm riêng bản nháp để đo lỗi bị bỏ sót), S6 tên thuộc tính bị sửa.
Ra: cham_mu.txt (xáo thứ tự, chưa biết bản nào là bản nào) và khoa_KHONG_XEM.json. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\lay_mau_cham_that.py <so_sanh_cum_*.parquet> <thư mục ra>
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
P = pd.read_parquet(sys.argv[1])
ra = Path(sys.argv[2])
ra.mkdir(parents=True, exist_ok=True)
CUM = pd.read_parquet(ROOT / "data" / "processed" / "song_ngu_3" / "vao_kaggle" / "cum_can_dich_v2.parquet")
td = dict(zip(CUM.product_id, CUM.tieu_de))
thu = set(CUM[CUM.mau_thu].product_id)
P = P[~P.product_id.isin(thu)].copy()
P["L"] = P.gia_tri_vi.str.len()
rng = random.Random(20261024)
ngu_canh = {}


def nc(pid):
    if pid not in ngu_canh:
        g = P[P.product_id == pid]
        ngu_canh[pid] = "; ".join(f"{n}: {v[:60]}" for n, v in zip(g.name_vi, g.gia_tri_vi))[:600]
    return ngu_canh[pid]


def rut(df, n):
    df = df[df.L <= 250]
    return df.sample(n=min(n, len(df)), random_state=rng.randint(0, 10**6))


item = []
S1 = rut(P[(P.dong_thuan == "chi_A_sua")], 45)
S2 = rut(P[(P.dong_thuan == "ca_hai_sua_giong")], 15)
S3 = rut(P[(P.dong_thuan == "ca_hai_sua_khac")], 25)
S4 = rut(P[(P.dong_thuan == "chi_B_sua")], 20)
S5 = rut(P[(P.dong_thuan == "ca_hai_giu_nhap")], 40)
S6 = P[P.n_nguon_AB.astype(str).str.startswith("sua") & (P.name_vi.str.len() <= 60)]
S6 = S6.drop_duplicates("name_vi").sample(n=min(15, S6.name_vi.nunique()), random_state=rng.randint(0, 10**6))
for tang, df in (("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4), ("S5", S5), ("S6", S6)):
    for r in df.itertuples(index=False):
        if tang == "S6":
            opts = {"nhap": r.n_nhap, "A": r.n_A}
            tr = "ten"
        else:
            opts = {"nhap": r.v_nhap, "A": r.v_A} if tang in ("S1", "S2", "S3") else {"nhap": r.v_nhap, "B": r.v_B} if tang == "S4" else {"nhap": r.v_nhap}
            if tang == "S3":
                opts["B"] = r.v_B
            tr = "gia_tri"
        item.append({"tang": tang, "pair_id": int(r.pair_id), "truong": tr, "product_id": r.product_id, "category": r.category,
                     "ten_vi": r.name_vi, "nguon_vi": r.gia_tri_vi, "opts": opts,
                     "ly_do": {"A": r.c_A, "B": r.c_B}})
rng.shuffle(item)
dong, khoa = [], {}
for k, it in enumerate(item, 1):
    keys = list(it["opts"])
    rng.shuffle(keys)
    khoa[k] = {"tang": it["tang"], "pair_id": it["pair_id"], "truong": it["truong"], "thu_tu": keys, "ly_do": it["ly_do"]}
    hien = "\n".join(f"  Bản {i}: {it['opts'][x]}" for i, x in enumerate(keys, 1))
    muc = "TÊN THUỘC TÍNH" if it["truong"] == "ten" else it["ten_vi"]
    dong.append(f"#{k} [{it['category']}] SP {it['product_id']} | TÊN SP: {td.get(it['product_id'], '')}\n  NGỮ CẢNH: {nc(it['product_id'])}\n"
                f"  >> {muc}: {it['nguon_vi'] if it['truong'] == 'gia_tri' else it['ten_vi']}\n{hien}\n")
(ra / "cham_mu.txt").write_text("\n".join(dong), encoding="utf-8")
(ra / "khoa_KHONG_XEM.json").write_text(json.dumps(khoa, ensure_ascii=False, indent=0), encoding="utf-8")
from collections import Counter
print("Số mục:", len(item), Counter(i["tang"] for i in item), "| ký tự:", sum(len(x) for x in dong))
