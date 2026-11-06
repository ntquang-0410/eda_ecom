# -*- coding: utf-8 -*-
"""Đánh giá lần THỬ 200 sản phẩm của bộ 4: ghép kết quả hai mô hình về cặp theo luật chặt (song_ngu_4.ket_qua), in số liệu, ghi bảng so sánh
và rút mẫu CHẤM MÙ (xáo thứ tự, người chấm chưa biết bản nào là bản nào). Các tầng (giá trị dài <= 250 ký tự để chấm được):
  S1 chỉ A sửa (bản cuối là A, so với nháp), S2 cả hai sửa giống nhau, S3 cả hai sửa khác nhau (nháp, A, B), S4 chỉ B sửa (bản cuối là B
  nếu A giữ nháp? KHÔNG: A ưu tiên, A giữ nháp thì bản cuối là nháp; vẫn chấm để biết B có bắt được lỗi A bỏ sót), S5 cả hai giữ nháp
  (đo lỗi bỏ sót), S6 tên thuộc tính bị sửa, S7 dịch mới (không có nháp; A và B), S8 bản sửa bị luật loại (luật có bắt oan không).
Chạy từ gốc repo:

    .venv\\Scripts\\python.exe -X utf8 scripts\\song_ngu_4\\danh_gia_thu.py <thư mục Gemma> <thư mục Qwen> <thư mục ra>
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_4 import cau_hinh as C  # noqa: E402
from song_ngu_4 import ket_qua as KQ  # noqa: E402


def dong_thuan(a, b, va, vb):
    """Mức đồng thuận của hai mô hình trên một trường (theo nguồn khi chọn riêng từng mô hình)."""
    if a == "khoa":
        return "khoa"
    sa, sb = a.startswith(("sua", "dich_moi")), b.startswith(("sua", "dich_moi"))
    xa, xb = a != "nhap_khong_co_ban_mo_hinh", b != "nhap_khong_co_ban_mo_hinh"
    if not xa and not xb:
        return "khong_duoc_xem"
    if sa and sb:
        return "ca_hai_sua_giong" if KQ.giong(va, vb) else "ca_hai_sua_khac"
    if sa:
        return "chi_A_sua"
    if sb:
        return "chi_B_sua"
    return "ca_hai_giu_nhap" if xa and xb else "chi_mot_mo_hinh_giu_nhap"


def main():
    ga, qw, ra = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ra.mkdir(parents=True, exist_ok=True)
    CUM = pd.read_parquet(C.THU_MUC_RA / "vao_kaggle" / "cum_can_dich_zv.parquet")
    CUM = CUM[CUM.mau_thu]
    P = pd.read_parquet(C.THU_MUC_RA / "cap_vao.parquet")
    P = P[P.ma_cum.isin(CUM.ma_cum)]
    X = KQ.lap_bang(P, {"A": ga, "B": qw})
    for f in ("n", "v"):
        X[f"dt_{f}"] = [dong_thuan(a, b, va, vb) for a, b, va, vb in zip(X[f"{f}_nguon_A"], X[f"{f}_nguon_B"], X[f"{f}_cuoi_A"], X[f"{f}_cuoi_B"])]
    X.to_parquet(ra / "so_sanh_thu.parquet", index=False)
    tk = KQ.tom_tat(X)
    tk["dong_thuan_gia_tri"] = X.dt_v.value_counts().to_dict()
    tk["dong_thuan_ten"] = X.dt_n.value_counts().to_dict()
    (ra / "tom_tat_thu.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tk, ensure_ascii=False, indent=1))

    # ---- mẫu chấm mù
    td = dict(zip(CUM.product_id, CUM.tieu_de))
    rng = random.Random(20261024)
    X["L"] = X.gia_tri_zh.str.len()
    nc_cache = {}

    def nc(pid):
        if pid not in nc_cache:
            g = X[X.product_id == pid].sort_values("i")
            nc_cache[pid] = "; ".join(f"{n}: {v[:40]}" for n, v in zip(g.name_zh, g.gia_tri_zh))[:500]
        return nc_cache[pid]

    def rut(df, n):
        df = df[df.L <= 250]
        return df.sample(n=min(n, len(df)), random_state=rng.randint(0, 10 ** 6))

    co_nhap = X.v_nhap.notna()
    tang = {"S1": rut(X[co_nhap & (X.dt_v == "chi_A_sua")], 60), "S2": rut(X[co_nhap & (X.dt_v == "ca_hai_sua_giong")], 15),
            "S3": rut(X[co_nhap & (X.dt_v == "ca_hai_sua_khac")], 25), "S4": rut(X[co_nhap & (X.dt_v == "chi_B_sua")], 20),
            "S5": rut(X[co_nhap & (X.dt_v == "ca_hai_giu_nhap")], 30), "S7": rut(X[X.v_nhap.isna() & X.v_khoa.isna()], 20),
            "S8": rut(X[co_nhap & X.v_A.notna() & (X.v_luat_A != "") & ~X.v_A.combine(X.v_nhap, KQ.giong)], 20)}
    S6 = X[X.n_nguon_AB.astype(str).str.startswith("sua")]
    tang["S6"] = S6.drop_duplicates("name_zh").sample(n=min(15, S6.name_zh.nunique()), random_state=rng.randint(0, 10 ** 6))
    item = []
    for t, df in tang.items():
        for r in df.itertuples(index=False):
            if t == "S6":
                opts, truong = {"nhap": r.n_nhap, "A": r.n_A}, "ten"
                if isinstance(r.n_B, str) and not KQ.giong(r.n_B, r.n_A) and not KQ.giong(r.n_B, r.n_nhap):
                    opts["B"] = r.n_B
            elif t == "S7":
                opts, truong = {k: v for k, v in (("A", r.v_A), ("B", r.v_B)) if isinstance(v, str)}, "gia_tri"
            else:
                opts, truong = {"nhap": r.v_nhap}, "gia_tri"
                if t in ("S1", "S2", "S3", "S8"):
                    opts["A"] = r.v_A
                if t in ("S3", "S4"):
                    opts["B"] = r.v_B
            if not opts:
                continue
            item.append({"tang": t, "pair_id": int(r.pair_id), "truong": truong, "product_id": r.product_id, "category": r.category,
                         "ten_zh": r.name_zh, "nguon_zh": r.gia_tri_zh, "ten_vi": r.n_cuoi_AB, "opts": opts,
                         "ly_do": {"A": r.c_A, "B": r.c_B}, "luat_A": r.v_luat_A, "luat_B": r.v_luat_B})
    rng.shuffle(item)
    dong, khoa = [], {}
    for k, it in enumerate(item, 1):
        keys = list(it["opts"])
        rng.shuffle(keys)
        khoa[k] = {"tang": it["tang"], "pair_id": it["pair_id"], "truong": it["truong"], "thu_tu": keys, "ly_do": it["ly_do"],
                   "luat_A": it["luat_A"], "luat_B": it["luat_B"]}
        hien = "\n".join(f"  Bản {i}: {it['opts'][x]}" for i, x in enumerate(keys, 1))
        muc = f"TÊN THUỘC TÍNH: {it['ten_zh']}" if it["truong"] == "ten" else f"{it['ten_zh']} ({it['ten_vi']}): {it['nguon_zh']}"
        dong.append(f"#{k} [{it['category']}] SP {it['product_id']} | TÊN SP: {td.get(it['product_id'], '')}\n  NGỮ CẢNH: {nc(it['product_id'])}\n"
                    f"  >> {muc}\n{hien}\n")
    (ra / "cham_mu.txt").write_text("\n".join(dong), encoding="utf-8")
    (ra / "khoa_KHONG_XEM.json").write_text(json.dumps(khoa, ensure_ascii=False, indent=0), encoding="utf-8")
    print("Mẫu chấm mù:", len(item), "mục", dict(Counter(i["tang"] for i in item)), "| ký tự:", sum(len(x) for x in dong))


if __name__ == "__main__":
    main()
