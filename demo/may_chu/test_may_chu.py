# -*- coding: utf-8 -*-
"""Test máy chủ dịch bằng mô hình giả (không GPU). Chạy: python -X utf8 test_may_chu.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dich as D  # noqa: E402
import loi_dan as L  # noqa: E402
import app as A  # noqa: E402


def doc_pool_mau():
    f = Path(__file__).resolve().parents[2] / "data" / "processed" / "ft_run" / "mau" / "pool_pair.jsonl"
    if not f.exists():
        return []
    ra = []
    with open(f, encoding="utf-8") as h:
        for x in h:
            m = json.loads(x)
            if m["huong"] == "zh->vi" and m["so_phan"] == 1:
                ra.append(m)
            if len(ra) >= 400:
                break
    return ra


def test_loi_dan_trung_mau_pool_that():
    ms = doc_pool_mau()
    if not ms:
        print("  (bỏ qua: không có pool_pair.jsonl)")
        return
    n = 0
    for m in ms:
        if m["loai"] == "title":
            u = L.loi_dan_title(m["nguon"])
        else:
            cap = [tuple(d.split(": ", 1)) for d in m["nguon"].split("\n") if ": " in d]
            if len(cap) != len(m["nguon"].split("\n")):
                continue
            u, _ = L.loi_dan_khoi(m["nguon_title"] if "nguon_title" in m else _title_cua(ms, m), cap)
        assert u == m["messages"][0]["content"], m["id"]
        n += 1
    assert n >= 50, n


def _title_cua(ms, m):
    for x in ms:
        if x["product_id"] == m["product_id"] and x["loai"] == "title":
            return x["nguon"]
    # title không nằm trong 400 mẫu đầu: lấy từ lời dẫn (dòng sau 'do not translate): ')
    t = m["messages"][0]["content"].split("Product title (context only, do not translate): ", 1)[1].split("\n\n", 1)[0]
    return t


class Gia:
    """Mô hình giả: dịch 'zh:X' -> 'vi:X'; có thể cố tình làm rơi một dòng."""
    ten = "gia"

    def __init__(self, rot_dong=False):
        self.rot, self.goi = rot_dong, 0

    def sinh(self, prompt, tran):
        self.goi += 1
        assert prompt.startswith("<|im_start|>user\n") and prompt.endswith("<think>\n\n</think>\n\n")
        nguon = prompt.split("<|im_end|>")[0].split("\n\n", 1)[1] if "attribute list" in prompt.split("\n")[1] else prompt.split("<|im_end|>")[0].split("\n\n", 1)[-1]
        dong = [("vi:" + d) for d in nguon.split("\n")]
        if self.rot and len(dong) > 1 and "\nProduct title" not in nguon and self.goi == 1:
            dong = dong[:-1]
        return "\n".join(dong)

    def suc_khoe(self):
        return {"status": "ok"}


def test_dich_khoi_binh_thuong_va_lech_dong():
    cap = [("材质", "棉"), ("颜色", "红")]
    ra, tk = D.dich_khoi(Gia().sinh, "T恤", cap)
    assert len(ra) == 2 and not tk["lech_dong"], (ra, tk)
    g = Gia(rot_dong=True)
    ra, tk = D.dich_khoi(g.sinh, "T恤", cap)
    assert tk["lech_dong"] and len(ra) == 2 and g.goi == 3, (ra, tk, g.goi)       # 1 lượt khối + 2 lượt từng dòng


def test_tach_cap():
    assert D.tach_cap("Chất liệu: Cotton: lanh") == ("Chất liệu", "Cotton: lanh")
    assert D.tach_cap("không có dấu") == ("không có dấu", None)


def test_khoi_qua_dai():
    try:
        D.dich_khoi(Gia().sinh, "T", [("a", "b")] * (D.KHOI_TOI_DA + 1))
        assert False
    except ValueError:
        pass


def test_app_bo_dem_va_khoa():
    with tempfile.TemporaryDirectory() as d:
        g = Gia()
        bd = A.BoDem(str(Path(d) / "x.sqlite"))
        app = A.tao_app(g, bd, khoa_truy_cap="bi-mat")
        c = app.test_client()
        assert c.post("/dich", json={"loai": "title", "zh": "T恤"}).status_code == 401
        h = {"X-Demo-Key": "bi-mat"}
        r1 = c.post("/dich", json={"loai": "title", "zh": "T恤"}, headers=h).get_json()
        r2 = c.post("/dich", json={"loai": "title", "zh": "T恤"}, headers=h).get_json()
        assert r1["vi"] == r2["vi"] and not r1["dem"] and r2["dem"] and g.goi == 1
        k = c.post("/dich", json={"loai": "khoi", "title": "T恤", "cap": [["材质", "棉"]]}, headers=h).get_json()
        assert k["cap"] == [["vi:材质", "棉"]], k          # mô hình giả viết 'vi:' trước cả dòng 'tên: giá trị'
        assert c.post("/dich", json={"loai": "khoi"}, headers=h).status_code == 400
        assert c.get("/suc_khoe", headers=h).status_code == 200
        bd.dong()


def test_chi_bo_dem_trung_khoa_va_bao_chua_dich():
    from backend import ChiBoDem
    with tempfile.TemporaryDirectory() as d:
        duong = str(Path(d) / "x.sqlite")
        bd = A.BoDem(duong)
        g = Gia()
        g.ten = "mo-hinh-x"
        h = {"X-Demo-Key": "k"}
        c1 = A.tao_app(g, bd, "k").test_client()
        r1 = c1.post("/dich", json={"loai": "title", "zh": "T恤"}, headers=h).get_json()
        k1 = c1.post("/dich", json={"loai": "khoi", "title": "T恤", "cap": [["材质", "棉"]]}, headers=h).get_json()
        c2 = A.tao_app(ChiBoDem("mo-hinh-x"), bd, "k").test_client()        # cùng bộ đệm, không mô hình
        r2 = c2.post("/dich", json={"loai": "title", "zh": "T恤"}, headers=h).get_json()
        k2 = c2.post("/dich", json={"loai": "khoi", "title": "T恤", "cap": [["材质", "棉"]]}, headers=h).get_json()
        assert r2["vi"] == r1["vi"] and r2["dem"] and k2["cap"] == k1["cap"] and k2["dem"]
        miss = c2.post("/dich", json={"loai": "title", "zh": "chưa có"}, headers=h)
        assert miss.status_code == 503 and "chua_dich_san" in miss.get_json()["loi"]
        assert c2.get("/suc_khoe", headers=h).get_json()["che_do"] == "chi_bo_dem"
        bd.dong()


if __name__ == "__main__":
    for ten, ham in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        ham()
        print("dat", ten)
    print("TAT CA DAT")
