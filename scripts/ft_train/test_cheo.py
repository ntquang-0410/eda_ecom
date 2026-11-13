# -*- coding: utf-8 -*-
"""Test kaggle_cheo.py (phần thuần) và cham_cheo.py. Chạy: python -X utf8 test_cheo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung  # noqa: E402
import dung_mau  # noqa: E402
import kaggle_cheo as KC  # noqa: E402
import cham_cheo as CC  # noqa: E402


def mau(pid, nhanh, loai, nguon, dich, fold=0, phan=0, huong="zh->vi", user=None):
    id_ = f"{pid}_title" if loai == "title" else f"{pid}_mo_ta"
    return {"id": id_, "product_id": pid, "category": "c", "huong": huong, "loai": loai, "nhanh": nhanh, "fold5": fold, "phan": phan,
            "nguon": nguon, "dich": dich, "messages": [{"role": "user", "content": user or ""}, {"role": "assistant", "content": dich}]}


def pool_gia(so_sp=10):
    cum, pair = [], []
    for i in range(so_sp):
        pid = f"P{i:03d}"
        dong_n = [f"颜色: 红{i}", "尺码: XL", f"重量: {i}0g"]
        dong_d = [f"Màu: đỏ{i}", "Kích cỡ: XL", f"Trọng lượng: {i}0g"]
        tit = f"商品{i}"
        u_title_c = dung_mau.chi_dan("title", "cum", "zh->vi", tit, None)
        cum.append(mau(pid, "cum", "title", tit, f"SP{i}", user=u_title_c))
        pair.append(mau(pid, "pair", "title", tit, f"SP{i}", user=u_title_c))
        nc, dc = "; ".join(dong_n), "; ".join(dong_d)
        if i == 3:      # nguồn hai nhánh khác nhau: phải bị loại
            nc = nc + "; 材质: 棉"
            dc = dc + "; Chất liệu: bông"
        cum.append(mau(pid, "cum", "mo_ta", nc, dc, user=dung_mau.chi_dan("mo_ta", "cum", "zh->vi", nc, tit)))
        pair.append(mau(pid, "pair", "mo_ta", "\n".join(dong_n), "\n".join(dong_d), user=dung_mau.chi_dan("mo_ta", "pair", "zh->vi", "\n".join(dong_n), tit)))
    # khác fold, khác phan, khác hướng: phải bị loại
    cum.append(mau("P900", "cum", "mo_ta", "a: b", "c: d", fold=1))
    cum.append(mau("P901", "cum", "mo_ta", "a: b", "c: d", phan=1))
    cum.append(mau("P902", "cum", "mo_ta", "a: b", "c: d", huong="vi->zh"))
    return cum, pair


def test_khuon_khop_dung_mau():
    assert KC.TACH == chung.TACH_DONG
    for d in KC.DANG:
        assert KC.loi_dan(d, "颜色: 红", "标题") == dung_mau.chi_dan("mo_ta", d, "zh->vi", "颜色: 红", "标题"), d


def test_chon_eval():
    cum, pair = pool_gia(10)
    items, tk = KC.chon_eval(cum, pair, so_toi_da=100, seed=43, so_co_dinh=250)
    ids = {x["id"] for x in items}
    assert "P003_mo_ta" not in ids, "nguon hai nhanh khac nhau phai bi loai"
    assert not any(i.startswith(("P900", "P901", "P902")) for i in ids)
    assert len(items) == 9 and tk["so_chon"] == 9 and tk["bo"]["nguon_hai_nhanh_khac_nhau"] == 1, tk
    assert items == sorted(items, key=lambda x: x["id"])
    x = items[0]
    assert x["nguon"]["cum"] == "; ".join(x["nguon"]["pair"].split("\n"))
    KC.kiem_loi_dan(items)
    nho, _ = KC.chon_eval(cum, pair, so_toi_da=4, seed=43, so_co_dinh=250)
    nho2, _ = KC.chon_eval(cum, pair, so_toi_da=4, seed=43, so_co_dinh=250)
    assert [x["id"] for x in nho] == [x["id"] for x in nho2] and len(nho) == 4, "chon phai tat dinh"


def test_kiem_loi_dan_bat_lech():
    cum, pair = pool_gia(3)
    items, _ = KC.chon_eval(cum, pair, 10)
    items[0]["loi_dan_goc"]["pair"] += " "
    try:
        KC.kiem_loi_dan(items)
    except RuntimeError as e:
        assert "LOI_DAN_LECH_KHUON" in str(e)
        return
    raise AssertionError("phai bao loi khi loi dan lech khuon")


def test_chuan_hoa_va_so_manh():
    assert CC.manh("pair", "a: 1\n\n b: 2 \n") == ["a: 1", "b: 2"]
    assert CC.manh("cum", "a: 1; b: 2") == ["a: 1", "b: 2"]
    assert CC.manh("cum", "  ") == []
    assert CC.chuan_hoa("pair", "a: 1\nb: 2") == CC.chuan_hoa("cum", "a: 1; b: 2") == "a: 1; b: 2"


def test_giu_so():
    assert CC.giu_so("颜色: 红", "Màu: đỏ") is None
    assert CC.giu_so("重量: 500g 2个", "Trọng lượng: 500g") == 0.5
    assert CC.giu_so("100 100", "100") == 0.5, "da tap: hai so 100, chi co mot"


def test_diem_mot_giong_het_va_sai_dang():
    cum, pair = pool_gia(4)
    items, _ = KC.chon_eval(cum, pair, 10)
    x = items[0]
    r = CC.diem_mot(x, "pair", x["ref"]["pair"])
    assert r["chrf"] == 100.0 and r["dung_so_manh"] and not r["rong"] and not r["ro_loi_dan"]
    r2 = CC.diem_mot(x, "cum", x["ref"]["cum"])
    assert r2["chrf"] == 100.0 and r2["dung_so_manh"]
    r3 = CC.diem_mot(x, "pair", "Màu: đỏ\nKích cỡ: XL")           # thiếu một dòng
    assert not r3["dung_so_manh"] and r3["chrf"] < 100
    r4 = CC.diem_mot(x, "pair", "")
    assert r4["rong"] and r4["chrf"] == 0.0
    r5 = CC.diem_mot(x, "cum", "Product title (context only) Màu: đỏ")
    assert r5["ro_loi_dan"]


def test_boot_chenh():
    a = [1.0, 2.0, 3.0, 4.0] * 10
    b = [0.0, 1.0, 2.0, 3.0] * 10
    r = CC.boot_chenh(a, b)
    assert abs(r["chenh"] - 1.0) < 1e-9 and r["thap"] == r["cao"] == 1.0
    r2 = CC.boot_chenh(a, a)
    assert r2["chenh"] == 0.0 and r2["thap"] == r2["cao"] == 0.0
    assert CC.boot_chenh(a, b, seed=1) == CC.boot_chenh(a, b, seed=1)


def test_chay_het_tren_du_lieu_gia(tmp):
    import json
    cum, pair = pool_gia(12)
    items, _ = KC.chon_eval(cum, pair, 100)
    KC.ghi_dong_json(tmp / "eval_items.jsonl", items)
    for ten in KC.ADAPTER:
        for d in KC.DANG:
            rows = [{"id": x["id"], "hyp": x["ref"][d] if ten != "zero_shot" else "Dịch tệ"} for x in items]
            KC.ghi_dong_json(tmp / f"sinh_{ten}_{d}.jsonl", rows)
    items2, ket = CC.cham_het(tmp)
    bang, cap = CC.tong_hop(items2, ket)
    assert len(bang) == 6 and len(cap) == 7
    gd = {(r["adapter"], r["dang_dau_vao"]): r for r in bang}
    assert gd[("pair_f0_p11", "pair")]["chrf_cau_tb"] == 100.0 and gd[("zero_shot", "pair")]["chrf_cau_tb"] < 100
    md = CC.viet_md(bang, cap, {}, {"so_chon": len(items), "so_co_dinh_trong_chon": 0})
    assert "chênh" in md.lower() or "Chênh" in md
    # id lệch phải bị bắt
    KC.ghi_dong_json(tmp / "sinh_zero_shot_cum.jsonl", [{"id": "X", "hyp": "a"}])
    try:
        CC.cham_het(tmp)
    except RuntimeError:
        return
    raise AssertionError("phai bao loi khi id lech")


if __name__ == "__main__":
    import tempfile
    for ten, ham in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        if ten == "test_chay_het_tren_du_lieu_gia":
            with tempfile.TemporaryDirectory() as t:
                ham(Path(t))
        else:
            ham()
        print("dat", ten)
    print("TAT CA DAT")
