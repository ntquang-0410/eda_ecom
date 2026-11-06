"""Thử kiem_title trên dữ liệu tự dựng. Chạy: python -I -X utf8 scripts/ft_train/test_kiem_title.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import kiem_title as T  # noqa: E402


def _px(muc, ma=""):
    return {"muc": muc, "loai_loi": ma, "ghi_chu": ""}


def test_tang():
    assert T.tang(_px("dat")) is None and T.tang(_px("nhe", "thuat_ngu")) is None
    assert T.tang(_px("nang", "thieu_thua")) == "thieu"
    assert T.tang(_px("nang", "thieu_thua|sai_nghia")) == "loi"
    assert T.tang(_px("nang", "sot_chu_nguon")) == "loi"
    assert T.tang(_px("nang", "")) == "khac"


def test_chon_mau():
    meta = pd.DataFrame({"id": [f"K{i:05d}" for i in range(700)], "huong": ["zh->vi"] * 400 + ["vi->zh"] * 300,
                         "loai": ["title"] * 650 + ["mo_ta"] * 50, "mau": ["M1"] * 700})
    da = {"K00000", "K00001", "K00650"}
    lo, cl = T.chon_mau(meta, da)
    tat = [i for x in lo for i in x]
    h = meta.set_index("id").huong
    assert [len(x) for x in lo] == [50, 50, 50, 50] and len(set(tat)) == 200
    assert not set(tat) & da and h[tat].value_counts().to_dict() == {"zh->vi": 100, "vi->zh": 100}
    assert len(cl) == 40 and set(cl) <= set(tat) and h[cl].value_counts().to_dict() == {"zh->vi": 20, "vi->zh": 20}
    assert (lo, cl) == T.chon_mau(meta, da), "cùng seed phải ra cùng mẫu"
    try:
        T.chon_mau(meta[:50], set())
    except RuntimeError:
        return
    raise AssertionError("thiếu title phải báo lỗi")


def test_bang_va_doi_chieu():
    meta = pd.DataFrame({"huong": ["zh->vi"] * 3 + ["vi->zh"]}, index=["a", "b", "c", "d"])
    a = {"a": _px("nang", "sai_nghia"), "b": _px("nang", "thieu_thua"), "c": _px("dat"), "d": _px("nhe", "van_phong")}
    b = {"a": _px("nang", "sai_nghia"), "b": _px("nhe", "thieu_thua"), "c": _px("dat"), "d": _px("nhe", "van_phong")}
    t = T.bang_hai_tang(a, meta)
    assert t["zh->vi"]["n"] == 3 and t["zh->vi"]["nang"] == 2 and t["zh->vi"]["nang_loi"] == 1 and t["zh->vi"]["nang_thieu"] == 1
    assert t["chung"]["n"] == 4 and t["vi->zh"]["nang"] == 0
    d = T.doi_chieu(a, b)
    assert d["n"] == 4 and d["lech"] == ["b"]
    assert (d["nang"]["ca_hai"], d["nang"]["chi_a"], d["nang"]["chi_b"], d["nang"]["khong"]) == (1, 1, 0, 2)
    assert (d["nang_loi"]["ca_hai"], d["nang_loi"]["chi_a"], d["nang_loi"]["kappa"]) == (1, 0, 1.0)


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_"):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
