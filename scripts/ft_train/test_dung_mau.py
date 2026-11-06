"""Thử dung_mau.py trên dữ liệu nhỏ tự dựng (không đụng dữ liệu thật). Chạy: python -I -X utf8 scripts/ft_train/test_dung_mau.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import chung  # noqa: E402
import dung_mau as M  # noqa: E402


def dem_gia(cap):
    """Đếm token giả: số ký tự của user + assistant (chỉ để thử nhanh)."""
    return [len(u) + len(a) for u, a in cap]


def hang(pid, source="1688", nhanh="pair", n_attr=3, title="标题", dich_title="Tiêu đề", gia_tri="giá trị"):
    tach = chung.TACH_DONG[nhanh]
    nguon = tach.join(f"属性{i}: 值{i}" for i in range(n_attr))
    dich = tach.join(f"Thuộc tính {i}: {gia_tri}{i}" for i in range(n_attr))
    z = source == "1688"
    return {"product_id": pid, "source": source, "category": "home", "huong": "zh->vi" if z else "vi->zh",
            "nguon_title": title if z else dich_title, "dich_title": dich_title if z else title,
            "nguon_mo_ta": nguon if z else dich, "dich_mo_ta": dich if z else nguon}


def split_gia(pool_ids, test_ids):
    rows = [{"product_id": p, "la_test": False, "fold5": i % 5} for i, p in enumerate(pool_ids)]
    rows += [{"product_id": p, "la_test": True, "fold5": -1} for p in test_ids]
    return pd.DataFrame(rows)


def test_chi_dan():
    assert M.chi_dan("title", "cum", "zh->vi", "标题", "标题") == \
        "Translate this product title from Chinese to Vietnamese. Output only the translation.\n\n标题"
    assert M.chi_dan("mo_ta", "pair", "vi->zh", "Hãng: A\nMàu: Đỏ", "Áo") == (
        "Translate this product's attribute list from Vietnamese to Chinese. Each line is one attribute written as "
        '"name: value". Keep the same number and order of lines, and output only the translation.\n'
        "Product title (context only, do not translate): Áo\n\nHãng: A\nMàu: Đỏ")
    assert M.chi_dan("mo_ta", "cum", "zh->vi", "品牌: A; 色: 红", "标题") == (
        "Translate this product's attribute list from Chinese to Vietnamese. The attributes are separated by \"; \" "
        'and each is written as "name: value". Keep the same number and order of attributes, separated by "; ", '
        "and output only the translation.\nProduct title (context only, do not translate): 标题\n\n品牌: A; 色: 红")


def test_khuon_dem_va_token_that():
    assert M.KHUON_DEM.format(u="U", a="A") == \
        "<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n"


def test_mau_co_ban_va_test_khong_bi_dong_cham():
    df = pd.DataFrame([hang("a1"), hang("a2", "tiki"), hang("t1"), hang("t2", "tiki")])
    sp = split_gia(["a1", "a2"], ["t1", "t2"])
    kq = M.dung_mau_nhanh(df, sp, "pair", 10000, dem_gia)
    assert len(kq["pool"]) == 4 and len(kq["test"]) == 4 and not kq["loai"]
    m = next(x for x in kq["pool"] if x["id"] == "a1_title")
    assert m["huong"] == "zh->vi" and m["loai"] == "title" and m["nhanh"] == "pair" and m["phan"] == 0 and m["so_phan"] == 1
    assert m["messages"][0]["content"] == M.chi_dan("title", "pair", "zh->vi", "标题", "标题")
    assert m["messages"][1] == {"role": "assistant", "content": "Tiêu đề"}
    assert m["nguon"] == "标题" and m["dich"] == "Tiêu đề" and m["so_token"] == len(m["messages"][0]["content"]) + len("Tiêu đề")
    m2 = next(x for x in kq["pool"] if x["id"] == "a2_mo_ta")
    assert m2["huong"] == "vi->zh" and "from Vietnamese to Chinese" in m2["messages"][0]["content"]
    assert all(x["fold5"] == -1 and x["qua_dai"] is False for x in kq["test"])
    assert all(0 <= x["fold5"] <= 4 and "qua_dai" not in x for x in kq["pool"])
    M.kiem_khoa(kq, sp, "pair")


def test_chia_phan_dung_ranh_gioi(nhanh="pair"):
    tach = chung.TACH_DONG[nhanh]
    df = pd.DataFrame([hang("p1", nhanh=nhanh, n_attr=40), hang("p2", nhanh=nhanh, n_attr=2)])
    sp = split_gia(["p1", "p2"], [])
    max_len = 600
    kq = M.dung_mau_nhanh(df, sp, nhanh, max_len, dem_gia)
    phan = sorted((x for x in kq["pool"] if x["product_id"] == "p1" and x["loai"] == "mo_ta"), key=lambda x: x["phan"])
    assert len(phan) > 1 and [x["phan"] for x in phan] == list(range(1, len(phan) + 1))
    assert all(x["so_phan"] == len(phan) for x in phan)
    assert [x["id"] for x in phan] == [f"p1_mo_ta_p{k}" for k in range(1, len(phan) + 1)]
    assert all(x["so_token"] <= max_len for x in phan)
    assert all(dem_gia([(x["messages"][0]["content"], x["messages"][1]["content"])])[0] == x["so_token"] for x in phan)
    goc = df.iloc[0]
    assert tach.join(x["nguon"] for x in phan) == goc.nguon_mo_ta                   # ghép lại đúng nguyên văn
    assert tach.join(x["dich"] for x in phan) == goc.dich_mo_ta
    for x in phan:                                                                   # nguồn và đích cắt cùng chỉ số
        assert len(x["nguon"].split(tach)) == len(x["dich"].split(tach))
    # tham lam: thêm thuộc tính đầu của phần kế tiếp vào phần trước thì vượt
    for a, b in zip(phan, phan[1:]):
        n = a["nguon"] + tach + b["nguon"].split(tach)[0]
        d = a["dich"] + tach + b["dich"].split(tach)[0]
        u = M.chi_dan("mo_ta", nhanh, "zh->vi", n, goc.nguon_title)
        assert dem_gia([(u, d)])[0] > max_len
    # mô tả ngắn của p2 vẫn nguyên, tiêu đề lặp lại ở mọi phần
    assert next(x for x in kq["pool"] if x["id"] == "p2_mo_ta")["so_phan"] == 1
    assert all("Product title (context only, do not translate): 标题" in x["messages"][0]["content"] for x in phan)
    M.kiem_khoa(kq, sp, nhanh)


def test_chia_phan_cum():
    test_chia_phan_dung_ranh_gioi("cum")


def test_cum_khong_can_duoc_thi_loai():
    h = hang("c1", nhanh="cum", n_attr=40)
    h["dich_mo_ta"] = h["dich_mo_ta"] + "; thừa: 1"                      # lệch số thuộc tính sau khi tách "; "
    h2 = hang("c2", nhanh="cum", n_attr=2)
    h2["dich_mo_ta"] = "Một: a; b; Hai: c"                               # lệch nhưng mẫu ngắn -> vẫn giữ
    df = pd.DataFrame([h, h2])
    sp = split_gia(["c1", "c2"], [])
    kq = M.dung_mau_nhanh(df, sp, "cum", 600, dem_gia)
    assert [(x["product_id"], x["loai"], x["ly_do"]) for x in kq["loai"]] == [("c1", "mo_ta", "khong_can_so_thuoc_tinh")]
    assert not any(x["product_id"] == "c1" and x["loai"] == "mo_ta" for x in kq["pool"])
    assert any(x["id"] == "c1_title" for x in kq["pool"]) and any(x["id"] == "c2_mo_ta" for x in kq["pool"])
    M.kiem_khoa(kq, sp, "cum")                                            # có lý do loại thì không bị coi là thiếu mẫu


def test_thuoc_tinh_don_le_va_title_vuot():
    h = hang("d1", n_attr=3)
    h["nguon_mo_ta"] = "a: 1\nb: " + "字" * 2000 + "\nc: 3"
    h["dich_mo_ta"] = "A: 1\nB: " + "x" * 2000 + "\nC: 3"
    h2 = hang("d2", title="标" * 1000, dich_title="T" * 1000)
    df = pd.DataFrame([h, h2])
    sp = split_gia(["d1", "d2"], [])
    kq = M.dung_mau_nhanh(df, sp, "pair", 1000, dem_gia)
    ly = {(x["product_id"], x["loai"]): x["ly_do"] for x in kq["loai"]}
    assert ly[("d1", "mo_ta")] == "thuoc_tinh_don_le_vuot_max_len" and ly[("d2", "title")] == "title_vuot_max_len"
    # d2 có title dài lặp lại trong chỉ dẫn mo_ta nên mo_ta của d2 cũng không còn chỗ cho thuộc tính nào
    assert ly[("d2", "mo_ta")] == "thuoc_tinh_don_le_vuot_max_len"
    assert len(kq["loai"]) == 3 and "d1_mo_ta" not in {x["id"] for x in kq["pool"]}


def test_mau_test_khong_chia_khong_loai():
    df = pd.DataFrame([hang("p1", n_attr=40), hang("t1", n_attr=40, title="标" * 1000, dich_title="T" * 1000), hang("t2", n_attr=2)])
    sp = split_gia(["p1"], ["t1", "t2"])
    kq = M.dung_mau_nhanh(df, sp, "pair", 600, dem_gia)
    t = {x["id"]: x for x in kq["test"]}
    assert set(t) == {"t1_title", "t1_mo_ta", "t2_title", "t2_mo_ta"}
    assert t["t1_title"]["qua_dai"] and t["t1_mo_ta"]["qua_dai"] and not t["t2_mo_ta"]["qua_dai"]
    assert all(x["phan"] == 0 and x["so_phan"] == 1 for x in kq["test"])
    assert t["t1_mo_ta"]["nguon"] == df.iloc[1].nguon_mo_ta                  # nguyên văn
    assert not any(x["product_id"] in ("t1", "t2") for x in kq["pool"])
    assert not kq["loai"]
    M.kiem_khoa(kq, sp, "pair")


def _loi(ham, *a):
    try:
        ham(*a)
    except RuntimeError as e:
        return str(e)
    raise AssertionError("đáng lẽ phải raise")


def test_khoa_test():
    df = pd.DataFrame([hang("a1"), hang("a2"), hang("t1"), hang("t2")])
    sp = split_gia(["a1", "a2"], ["t1", "t2"])
    kq = M.dung_mau_nhanh(df, sp, "pair", 10000, dem_gia)
    M.kiem_khoa(kq, sp, "pair")
    # cố tình đưa mẫu test vào pool
    xau = dict(kq, pool=kq["pool"] + [dict(kq["test"][0])])
    assert "RO_RI_TEST" in _loi(M.kiem_khoa, xau, sp, "pair")
    # pool có mẫu mang fold5 = -1
    m = dict(kq["pool"][0], fold5=-1)
    assert _loi(M.kiem_khoa, dict(kq, pool=[m] + kq["pool"][1:]), sp, "pair")
    # test thiếu một mẫu
    assert _loi(M.kiem_khoa, dict(kq, test=kq["test"][1:]), sp, "pair")
    # thiếu mẫu pool mà không có lý do loại
    assert "thiếu mẫu" in _loi(M.kiem_khoa, dict(kq, pool=kq["pool"][1:]), sp, "pair")
    # mẫu test mang fold5 khác -1
    m = dict(kq["test"][0], fold5=2)
    assert _loi(M.kiem_khoa, dict(kq, test=[m] + kq["test"][1:]), sp, "pair")
    # bộ dữ liệu và split khác tập sản phẩm
    assert _loi(M.dung_mau_nhanh, df.iloc[:3], sp, "pair", 10000, dem_gia)
    # hàm gốc của chung cũng phải chặn
    assert _loi(chung.kiem_khong_giao, ["a1", "t1"], ["t1"])


def test_ghi_file_atomic():
    df = pd.DataFrame([hang("a1"), hang("t1")])
    sp = split_gia(["a1"], ["t1"])
    kq = M.dung_mau_nhanh(df, sp, "pair", 10000, dem_gia)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "mau" / "pool_pair.jsonl"
        M.ghi_jsonl(f, kq["pool"])
        assert f.exists() and not list(f.parent.glob("*.tmp"))
        import json
        doc = [json.loads(s) for s in f.read_text(encoding="utf-8").splitlines()]
        assert doc == kq["pool"] and "标题" in f.read_text(encoding="utf-8")
        bc = M.lam_bao_cao(kq, "pair", 10000)
        assert bc["pool"]["so_mau"] == 2 and bc["test_chi_dem"]["so_mau"] == 2
        md = M.lam_md({"pair": kq})
        assert "a1_title" in md and "t1_title" not in md                   # báo cáo mẫu chỉ có pool


if __name__ == "__main__":
    test_chi_dan()
    test_khuon_dem_va_token_that()
    test_mau_co_ban_va_test_khong_bi_dong_cham()
    test_chia_phan_dung_ranh_gioi()
    test_chia_phan_cum()
    test_cum_khong_can_duoc_thi_loai()
    test_thuoc_tinh_don_le_va_title_vuot()
    test_mau_test_khong_chia_khong_loai()
    test_khoa_test()
    test_ghi_file_atomic()
    print("OK: mọi test dung_mau đều qua")
