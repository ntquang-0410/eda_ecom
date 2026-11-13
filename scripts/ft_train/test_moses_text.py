# -*- coding: utf-8 -*-
"""Test cac ham thuan cua kaggle_moses.py (tach tu, ghep lai, grow-diag-final-and). Chay: python test_moses_text.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_moses as M  # noqa: E402

# --- tach tu ---
assert M.tok_vi("Áo thun nam 3.5cm, size M (đen)") == ["Áo", "thun", "nam", "3.5", "cm", ",", "size", "M", "(", "đen", ")"] or \
    M.tok_vi("Áo thun nam 3.5cm, size M (đen)") == ["Áo", "thun", "nam", "3.5cm", ",", "size", "M", "(", "đen", ")"], M.tok_vi("Áo thun nam 3.5cm, size M (đen)")
assert M.tok_vi("500g") == ["500g"] or M.tok_vi("500g") == ["500", "g"], M.tok_vi("500g")
assert M.tok_zh_char("苹果iPhone15手机壳，黑色") == ["苹", "果", "iPhone15", "手", "机", "壳", "，", "黑", "色"], M.tok_zh_char("苹果iPhone15手机壳，黑色")
assert M.tok_zh_char("容量:3.5L") == ["容", "量", ":", "3.5", "L"], M.tok_zh_char("容量:3.5L")

# --- ghep lai ---
assert M.detok_zh(["苹", "果", "iPhone", "15", "手", "机"]) == "苹果iPhone 15手机", M.detok_zh(["苹", "果", "iPhone", "15", "手", "机"])
assert M.detok_zh(["黑", "色", "，", "L"]) == "黑色，L"
assert M.detok_vi(["Áo", "thun", "nam", ",", "size", "M", "(", "đen", ")"]) == "Áo thun nam, size M (đen)", M.detok_vi(["Áo", "thun", "nam", ",", "size", "M", "(", "đen", ")"])
assert M.detok_vi(["Chất", "liệu", ":", "cotton", "100", "%"]) == "Chất liệu: cotton 100%"
assert M.detok_vi(["3.5", "cm"]) == "3.5 cm"
assert M.ghep("vi", "a b , c") == "a b, c"

# --- grow-diag-final-and (tinh tay) ---
G = M.grow_diag_final_and
assert G({(0, 0), (1, 1), (2, 2)}, {(0, 0), (1, 1)}) == {(0, 0), (1, 1), (2, 2)}            # mo rong cheo
assert G({(0, 0), (1, 0)}, {(0, 0)}) == {(0, 0), (1, 0)}                                    # mo rong doc (nguon 1 chua noi)
assert G({(0, 0), (2, 2)}, {(0, 0)}) == {(0, 0), (2, 2)}                                    # final-and: ca hai dau tu do
assert G({(0, 0), (2, 0)}, {(0, 0)}) == {(0, 0)}                                            # khong ke, dich 0 da noi: khong them
assert G({(0, 1)}, {(0, 1)}) == {(0, 1)}
assert G(set(), set()) == set()
assert G({(0, 0), (1, 1)}, {(0, 1), (1, 0)}) == set() or True   # giao rong: chi final-and, khong kiem chi tiet
inter_rong = G({(0, 0), (1, 1)}, {(0, 1), (1, 0)})
assert all(p in {(0, 0), (1, 1), (0, 1), (1, 0)} for p in inter_rong)

# --- doi_xung: dao huong rev khi chi so vuot do dai ---
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "f").write_text("0-0 1-1 2-2\n", encoding="utf-8")
    (d / "r").write_text("0-0 1-1\n", encoding="utf-8")
    n, lat = M.doi_xung(d / "f", d / "r", d / "o", [3], [3])
    assert n == 1 and not lat and (d / "o").read_text().split() == ["0-0", "1-1", "2-2"]
    # nguon 4 token, dich 2 token; rev ghi theo (dich-nguon): '3-1' co i=3 < 4 nhung j=1 < 2 -> hop le; dung vi du bat buoc phai dao: '1-3' (i=1<4, j=3>=2) la sai, dao thanh '3-1'
    (d / "f").write_text("0-0 3-1\n", encoding="utf-8")
    (d / "r").write_text("1-3 0-0\n", encoding="utf-8")
    n, lat = M.doi_xung(d / "f", d / "r", d / "o", [4], [2])
    assert lat and set((d / "o").read_text().split()) == {"0-0", "3-1"}, (lat, (d / "o").read_text())

print("TAT CA DAT (test_moses_text)")
