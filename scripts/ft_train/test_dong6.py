# -*- coding: utf-8 -*-
"""Test phần thuần của kaggle_dong6.py. Chạy: python -X utf8 test_dong6.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_dong6 as KD  # noqa: E402


def pool_gia():
    p = {}
    for pid, nguon, dich in (("P1", "a: 1\nb: 2\nc: 3", "A: 1\nB: 2\nC: 3"), ("P2", "x: 9\ny: 8", "X: 9\nY: 8")):
        p[(pid, "title")] = {"nguon": f"t{pid}", "dich": f"T{pid}", "messages": [{"role": "user", "content": f"TITLE {pid}"}, {"role": "assistant", "content": f"T{pid}"}]}
        p[(pid, "mo_ta")] = {"nguon": nguon, "dich": dich, "messages": [{"role": "user", "content": f"MOTA {pid}"}, {"role": "assistant", "content": dich}]}
    return p


def mau_gia():
    def m(pid, loai_ref, dong, j, loai):
        return {"id": f"{pid}_{loai_ref}_{dong}_{j}", "product_id": pid, "huong": "zh->vi", "loai": loai, "group": "1688_x", "nguon": "n", "dich": "d",
                "category": "c", "fold5": 0, "nhanh": "pair", "messages": [{"role": "user", "content": "QUOTA"}, {"role": "assistant", "content": "d"}]}
    return [m("P1", "title", -1, 0, "title"), m("P1", "attribute", 0, 1, "mo_ta"), m("P1", "attribute", 2, 2, "mo_ta"), m("P2", "attribute", 1, 3, "mo_ta")]


def test_tach_khoi():
    assert KD.tach_khoi("a\nb\nc", 3) == ["a", "b", "c"]
    assert KD.tach_khoi("a\nb\nc\n", 3) == ["a", "b", "c"]
    assert KD.tach_khoi("a\nb", 3) is None
    assert KD.tach_khoi("a\nb\nc\nd", 3) is None


def test_dong_cua_id():
    assert KD.dong_cua_id("123456_attribute_7_99") == 7
    assert KD.dong_cua_id("123456_title_-1_5") == -1


def test_dung_gen_pair_gop_khoi():
    gen, hang = KD.dung_gen("pair", mau_gia(), pool_gia())
    ids = [g["id"] for g in gen]
    assert ids == ["P1_title_-1_0", "P1_khoi", "P2_khoi"], ids      # P1 có hai dòng nhưng chỉ một lượt gọi
    assert gen[0]["messages"][0]["content"] == "TITLE P1" and gen[1]["messages"][0]["content"] == "MOTA P1"      # lời dẫn dung_mau, không phải lời dẫn quota
    assert [(h[2], h[3]) for h in hang] == [(None, None), (0, 3), (2, 3), (1, 2)]


def test_dung_gen_cum_mot_hang_mot_luot():
    mau = [{**m, "loai": m["loai"]} for m in mau_gia()]
    gen, hang = KD.dung_gen("cum", mau, pool_gia())
    assert len(gen) == len(mau) == len(hang)
    assert all(g["messages"][0]["content"] != "QUOTA" for g in gen)


def test_ghep_pair_va_lech_dong():
    gen, hang = KD.dung_gen("pair", mau_gia(), pool_gia())
    sinh = [{"id": "P1_title_-1_0", "hyp_tho": "TT", "hyp": "TT"}, {"id": "P1_khoi", "hyp_tho": "A\nB\nC", "hyp": "A\nB\nC"}, {"id": "P2_khoi", "hyp_tho": "X", "hyp": "X"}]
    du, lech = KD.ghep(sinh, hang)
    assert [d["hyp"] for d in du] == ["TT", "A", "C", ""] and lech == 1       # P2 thiếu một dòng: mọi dòng của P2 rỗng, đếm 1 khối lệch
    assert [d["id"] for d in du] == [h[0] for h in hang]


if __name__ == "__main__":
    for ten, ham in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        ham()
        print("dat", ten)
    print("TAT CA DAT")
