# -*- coding: utf-8 -*-
"""Test cham_bleu.py. Chạy: python -X utf8 test_cham_bleu.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cham_bleu as B  # noqa: E402


def test_giong_het():
    d = B.diem_nhom("zh->vi", ["Áo thun nam, size XL (100% cotton)."], ["Áo thun nam, size XL (100% cotton)."])
    assert abs(d["sacrebleu"] - 100) < 1e-6 and abs(d["bleu_mteval"] - 100) < 1e-6 and abs(d["ter"]) < 1e-6
    z = B.diem_nhom("vi->zh", ["男士T恤 XL码 纯棉"], ["男士T恤 XL码 纯棉"])
    assert abs(z["sacrebleu"] - 100) < 1e-6 and abs(z["ter"]) < 1e-6


def test_tach_chu():
    assert B._tach("zh->vi") == "13a" and B._tach("vi->zh") == "zh"
    ck = B.chu_ky()
    assert "tok:13a" in ck["zh->vi/sacrebleu"] and "tok:zh" in ck["vi->zh/sacrebleu"], ck
    assert "smooth:exp" in ck["zh->vi/sacrebleu"] and "smooth:none" in ck["zh->vi/bleu_mteval"], ck
    assert "case:mixed" in ck["zh->vi/ter"] and "case:mixed" in ck["vi->zh/ter"], ck


def test_dau_cau_va_chu_han():
    # 13a tách dấu câu: "100 %" và "100%" cho cùng token nên không bị phạt oan khoảng trắng
    a = B.diem_nhom("zh->vi", ["size XL , 100 % cotton ."], ["size XL, 100% cotton."])
    assert a["sacrebleu"] > 99.9, a
    # zh tách từng chữ Hán: lệch một chữ thì điểm tụt nhưng không về 0, và TER tính theo chữ
    z = B.diem_nhom("vi->zh", ["男士T恤 XL码 纯棉"], ["男式T恤 XL号 纯棉"])
    assert 0 < z["sacrebleu"] < 100 and 0 < z["ter"] < 100, z


def test_ter_dich_trung_khong_phu_thuoc_khoang_trang():
    # ca thật: tham chiếu có khoảng trắng sau dấu hai chấm, bản dịch viết liền; nội dung giống hệt nên TER phải bằng 0
    assert B.cat_chu_han("保修？:否") == B.cat_chu_han("保修？: 否") == "保 修 ？: 否"
    assert B.cat_chu_han("颜色: Burgundy Chrome") == "颜 色 : Burgundy Chrome"          # chuỗi không phải chữ Hán giữ nguyên
    z = B.diem_nhom("vi->zh", ["产品是否有保修？:否"], ["产品是否有保修？: 否"])
    assert abs(z["ter"]) < 1e-9, z
    # đích Việt không bị đụng tới
    v = B.diem_nhom("zh->vi", ["màu: đỏ"], ["màu:  đỏ"])
    assert abs(v["ter"]) < 1e-9


def test_rong_va_nhom_trong():
    assert B.diem_nhom("zh->vi", [], []) is None
    d = B.diem_nhom("zh->vi", [""], ["Áo thun"])
    assert d["sacrebleu"] == 0.0 and d["ter"] >= 100.0 - 1e-9


def df_gia():
    rows = []
    for huong in ("zh->vi", "vi->zh"):
        for loai in ("title", "mo_ta"):
            t = "áo thun nam size XL chất liệu cotton thoáng mát" if huong == "zh->vi" else "男士纯棉短袖圆领T恤夏季宽松款"      # đủ dài để có 4-gram
            rows.append({"huong": huong, "loai": loai, "hyp_cham": t, "_ref": t})
    return pd.DataFrame(rows)


def test_diem_bleu_ter_bon_nhom():
    r = B.diem_bleu_ter(df_gia())
    assert list(r) == B.NHOM and all(abs(v["sacrebleu"] - 100) < 1e-6 for v in r.values())
    # thiếu nhóm: trả None, không đổ
    r2 = B.diem_bleu_ter(df_gia().iloc[:2])
    assert r2["zh->vi/title"] is not None and r2["vi->zh/title"] is None


def test_chinh_macro():
    tn = {"zh->vi/title": {"ter": 40.0}, "zh->vi/mo_ta": {"ter": 20.0}, "vi->zh/title": {"ter": 60.0}, "vi->zh/mo_ta": {"ter": 30.0}}
    c, m = B.chinh_macro(tn, "ter")
    assert abs(c - (0.1 * 40 + 0.9 * 20)) < 1e-9 and abs(m - ((30.0 + 45.0) / 2)) < 1e-9
    c2, m2 = B.chinh_macro({"zh->vi/title": {"ter": 1.0}, "zh->vi/mo_ta": {"ter": 2.0}}, "ter")
    assert c2 is not None and m2 is None


def df_bien_thien(so_sp=30, doi=0):
    """df giả có nhiều sản phẩm; doi > 0 thì cố tình sửa bớt bản dịch để thành 'mô hình kém hơn'."""
    import random
    rng = random.Random(1)
    tu_vi = "áo thun nam size XL chất liệu cotton thoáng mát mùa hè quần jean nữ".split()
    rows = []
    for p in range(so_sp):
        for huong in ("zh->vi", "vi->zh"):
            for loai in ("title", "mo_ta"):
                if huong == "zh->vi":
                    ref = " ".join(rng.choice(tu_vi) for _ in range(8))
                    hyp = " ".join((rng.choice(tu_vi) if rng.random() < doi else w) for w in ref.split())
                else:
                    ref = "".join(rng.choice("男士纯棉短袖圆领宽松夏季款式") for _ in range(10))
                    hyp = "".join((rng.choice("男士纯棉短袖圆领宽松夏季款式") if rng.random() < doi else c) for c in ref)
                rows.append({"id": f"{1000 + p}_{loai}_{-1}_{len(rows)}", "huong": huong, "loai": loai, "hyp_cham": hyp, "_ref": ref})
    return pd.DataFrame(rows)


def test_boot_khop_diem_chinh():
    import numpy as np
    df = df_bien_thien(30, doi=0.3)
    pids = sorted(df["id"].str.split("_").str[0].unique())
    tk = B.thong_ke_df(df, pids)
    ra = B.diem_boot([tk], np.ones((1, len(pids))))             # trọng số toàn 1 = bộ eval gốc
    goc = B.diem_bleu_ter(df)
    for nhom in B.NHOM:
        assert abs(ra[(nhom, "sacrebleu")][0] - goc[nhom]["sacrebleu"]) < 1e-6, nhom
        assert abs(ra[(nhom, "ter")][0] - goc[nhom]["ter"]) < 1e-6, nhom      # gồm cả TER đích Trung đã tách chữ Hán
    c, m = B.chinh_macro_boot(ra, "ter")
    c0, m0 = B.chinh_macro({k: v for k, v in goc.items()}, "ter")
    assert abs(c[0] - c0) < 1e-6 and abs(m[0] - m0) < 1e-6


def test_boot_khoang_tin_cay_va_cap():
    import numpy as np
    tot, kem = df_bien_thien(40, doi=0.05), df_bien_thien(40, doi=0.5)
    pids = sorted(tot["id"].str.split("_").str[0].unique())
    w = np.random.default_rng(7).multinomial(len(pids), np.ones(len(pids)) / len(pids), size=300).astype(float)
    bt, bk = B.diem_boot([B.thong_ke_df(tot, pids)], w), B.diem_boot([B.thong_ke_df(kem, pids)], w)
    ct, _ = B.chinh_macro_boot(bt, "sacrebleu")
    ck, _ = B.chinh_macro_boot(bk, "sacrebleu")
    lo, hi = B.khoang(ct)
    assert lo < ct.mean() < hi and hi - lo > 0                                   # khoảng có độ rộng, chứa trung bình
    chenh = ct - ck
    assert B.khoang(chenh)[0] > 0, "mô hình tốt hơn rõ phải có khoảng chênh hẳn trên 0"
    assert abs(B.khoang(ct - ct)[0]) < 1e-12 and abs(B.khoang(ct - ct)[1]) < 1e-12   # cùng mô hình thì chênh đúng 0
    ter_t, _ = B.chinh_macro_boot(bt, "ter")
    ter_k, _ = B.chinh_macro_boot(bk, "ter")
    assert B.khoang(ter_k - ter_t)[0] > 0                                         # TER của mô hình kém cao hơn hẳn


def test_trung_binh_lan():
    a = {n: {"sacrebleu": 10.0, "bleu_mteval": 10.0, "ter": 90.0} for n in B.NHOM}
    b = {n: {"sacrebleu": 20.0, "bleu_mteval": 20.0, "ter": 70.0} for n in B.NHOM}
    r = B.trung_binh_lan([a, b])
    assert r["zh->vi/title"] == {"sacrebleu": 15.0, "bleu_mteval": 15.0, "ter": 80.0}


if __name__ == "__main__":
    for ten, ham in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        ham()
        print("dat", ten)
    print("TAT CA DAT")
