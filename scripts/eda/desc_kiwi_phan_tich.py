# -*- coding: utf-8 -*-
"""Phân tích điểm CometKiwi (wmt22-cometkiwi-da, chạy trên Colab T4) trên 4.291 cặp description đã gắn nhãn,
so với bge-m3 cosine (desc_bge_do_truoc.py). Chỉ đọc; ghi vào data/eda/description/kiwi_do_truoc/.

    python scripts/eda/desc_kiwi_phan_tich.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from desc_bge_do_truoc import auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D_KIWI = ROOT / "data" / "eda" / "description" / "kiwi_do_truoc"
D_BGE = ROOT / "data" / "eda" / "description" / "bge_do_truoc"


def bang_giu(d, col, phan=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)):
    """Giữ x% cặp điểm cao nhất: phần giữ còn sai bao nhiêu? (câu hỏi quyết định có cứu tầng 2 được không)"""
    d = d.sort_values(col, ascending=False)
    rows = []
    for p in phan:
        g = d.head(int(round(len(d) * p)))
        rows.append({"giu_top": f"{p:.0%}", "so_cap": len(g), "nguong_diem": round(g[col].min(), 3),
                     "ti_le_sai_trong_phan_giu": round((g.nhan == "SAI").mean(), 3),
                     "bat_duoc_SAI_trong_phan_bo": int((d.nhan == "SAI").sum() - (g.nhan == "SAI").sum())})
    return pd.DataFrame(rows)


def main():
    K = pd.read_csv(D_KIWI / "kiwi_ket_qua.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    B = pd.read_csv(D_BGE / "cap_da_gan_nhan_cosine.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert len(K) == len(B) == 4291
    assert (K[["tap", "gia_tri_zh", "gia_tri_vi", "nhan"]].values == B[["tap", "gia_tri_zh", "gia_tri_vi", "nhan"]].values).all()
    D = K.copy()
    for c in ["kiwi_gia_tri", "kiwi_ten_gia_tri"]:
        D[c] = D[c].astype(float)
    for c in ["cos_gia_tri", "cos_ten_gia_tri"]:
        D[c] = B[c].astype(float)
    # kết hợp: trung bình thứ hạng (không cần cùng thang điểm)
    D["ket_hop"] = (D.kiwi_ten_gia_tri.rank(pct=True) + D.cos_ten_gia_tri.rank(pct=True)) / 2
    print("Điểm kiwi:", D.kiwi_gia_tri.describe().round(3).to_dict())

    cot = ["kiwi_gia_tri", "kiwi_ten_gia_tri", "cos_gia_tri", "cos_ten_gia_tri", "ket_hop"]
    tom = []
    for tap in ["tang1", "tang2"]:
        d = D[D.tap == tap]
        for c in cot:
            s, g = d[d.nhan == "SAI"][c].to_numpy(), d[d.nhan == "DUNG"][c].to_numpy()
            tom.append({"tap": tap, "cach": c, "tb_dung": round(g.mean(), 3), "tb_sai": round(s.mean(), 3),
                        "auc": round(auc(s, g), 3)})
    tom = pd.DataFrame(tom)
    print("\nAUC (0,5 = đoán mò):\n" + tom.to_string(index=False))
    tom.to_csv(D_KIWI / "tom_tat_auc_kiwi_vs_bge.csv", index=False, encoding="utf-8-sig")

    for tap in ["tang2", "tang1"]:
        d = D[D.tap == tap]
        best = tom[(tom.tap == tap) & tom.cach.str.startswith("kiwi")].sort_values("auc").cach.iloc[-1]
        b = bang_giu(d, best)
        print(f"\n[{tap}] giữ x% cặp điểm cao nhất theo {best} (tỉ lệ sai nền {(d.nhan == 'SAI').mean():.1%}):\n" + b.to_string(index=False))
        b.to_csv(D_KIWI / f"bang_giu_{tap}.csv", index=False, encoding="utf-8-sig")
        if tap == "tang1":
            # gắn cờ cặp điểm thấp nhất: precision ở đáy
            for q in [0.02, 0.05, 0.10, 0.20]:
                low = d.nsmallest(int(len(d) * q), best)
                print(f"  đáy {q:.0%} ({len(low)} cặp): {(low.nhan == 'SAI').mean():.1%} là SAI")

    for tap in ["tang1", "tang2"]:
        d = D[D.tap == tap]
        print(f"\n[{tap}] SAI nhưng kiwi cao nhất:")
        print(d[d.nhan == "SAI"].nlargest(8, "kiwi_gia_tri")[["gia_tri_zh", "gia_tri_vi", "kiwi_gia_tri"]].to_string(index=False))
        print(f"[{tap}] DUNG nhưng kiwi thấp nhất:")
        print(d[d.nhan == "DUNG"].nsmallest(8, "kiwi_gia_tri")[["gia_tri_zh", "gia_tri_vi", "kiwi_gia_tri"]].to_string(index=False))
    D.to_csv(D_KIWI / "cap_da_gan_nhan_kiwi_bge.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
