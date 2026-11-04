# -*- coding: utf-8 -*-
"""Test đơn vị cho gop_tiki_vao_1688.py (không cần mạng, không cần dữ liệu thật).   python scripts/release/test_gop_tiki_vao_1688.py"""
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gop_tiki_vao_1688 as g  # noqa: E402

CFG = {"phien_ban": "test", "ghi_chu": "", "so_dong_dau_vao": {"1688": 10, "tiki": 10},
       "tham_so": {"tran_moi_nhom": 3, "nguong_cap_yeu": 0.25, "toi_da_mo_ta_trung": 2},
       "bac_A": [{"nguon": "tiki", "nhom": "food", "mo_ta": "thú cưng", "la": ["Thức ăn cho mèo"]}, {"nguon": "1688", "nhom": None, "mo_ta": "lạc nhóm", "la": ["U盘"]}],
       "bac_B": [{"nguon": "tiki", "nhom": "auto", "mo_ta": "xe", "la": ["Xe tay ga"]}],
       "quy_tac_title": [{"ly_do": "title_con_chu_han", "nguon": "1688", "cot": "title_vi", "regex": "[一-鿿]", "mo_ta": "Hán trong title_vi"}]}
_n = [0]


def sp(source="tiki", category="home", la="L", desc=None, so_cap=10, thap=0, title_vi="tiêu đề", title_zh="标题", pid=None):
    _n[0] += 1
    return {"product_id": pid or f"p{_n[0]:04d}", "source": source, "category": category, "la": la, "title_zh": title_zh, "title_vi": title_vi,
            "description_zh": desc if desc is not None else f"mô tả riêng {_n[0]}", "so_cap": so_cap, "so_cap_thap": thap, "title_van_de": ""}


def chay(*dong, cfg=CFG):
    return g.ap_dung_quy_tac(pd.DataFrame(list(dong)), cfg)[0]


def test_ly_do_dau_tien():
    d = chay(sp(category="food", la="Thức ăn cho mèo", thap=9))      # vi phạm cả R1 lẫn R2: chỉ ghi R1
    assert d.ly_do_loai.tolist() == ["sai_nhom"]


def test_khong_co_description():
    d = chay({**sp(), "description_zh": None, "so_cap": None, "so_cap_thap": None})
    assert d.ly_do_loai.tolist() == ["khong_co_description"]


def test_sai_nhom_theo_nguon_va_nhom():
    d = chay(sp(category="food", la="Thức ăn cho mèo"), sp(category="home", la="Thức ăn cho mèo"), sp(source="1688", category="auto", la="U盘"), sp(source="tiki", category="auto", la="U盘"))
    assert d.ly_do_loai.tolist() == ["sai_nhom", None, "sai_nhom", None]       # đúng nhóm mới loại; 1688 không ràng nhóm nên loại ở mọi nhóm; U盘 của Tiki không bị


def test_nguong_cap_yeu():
    d = chay(sp(so_cap=4, thap=1), sp(so_cap=100, thap=24), sp(so_cap=4, thap=0))      # 0,25 loại; 0,24 giữ
    assert d.ly_do_loai.tolist() == ["cap_yeu_cao", None, None]


def test_mo_ta_trung_giu_dung_so_luong_va_chon_theo_ma_bam():
    ids = [f"a{i}" for i in range(5)]
    d = chay(*[sp(desc="GIỐNG", pid=i) for i in ids], sp(desc="GIỐNG", category="auto", pid="khac_nhom"), sp(desc="GIỐNG", source="1688", pid="khac_nguon"))
    giu = set(d[d.ly_do_loai.isna() & (d.category == "home") & (d.source == "tiki")].product_id)
    mong_doi = set(sorted(ids, key=lambda i: (zlib.crc32(i.encode()), i))[:2])
    assert giu == mong_doi, (giu, mong_doi)
    assert d[d.ly_do_loai == "mo_ta_trung"].shape[0] == 3
    assert d[d.product_id.isin(["khac_nhom", "khac_nguon"])].ly_do_loai.isna().all()      # khác nhóm hoặc khác nguồn không tính là trùng


def test_title_chi_ap_dung_dung_nguon():
    d = chay(sp(source="1688", category="auto", title_vi="Giày 拼豆"), sp(source="tiki", title_vi="Giày 拼豆"), sp(source="1688", category="auto", title_vi="Giày sạch"))
    assert d.ly_do_loai.tolist() == ["title_con_chu_han", None, None]


def test_tran_bo_bac_b_truoc_roi_cap_yeu_roi_ma_bam():
    cfg = {**CFG, "tham_so": {**CFG["tham_so"], "tran_moi_nhom": 3}}
    d = chay(sp(category="auto", la="Xe tay ga", pid="B1"), sp(category="auto", pid="x1", thap=2), sp(category="auto", pid="x2", thap=1), sp(category="auto", pid="x3", thap=0),
             sp(category="auto", pid="x4", thap=0), cfg=cfg)
    giu = set(d[d.ly_do_loai.isna()].product_id)
    assert "B1" not in giu and len(giu) == 3                                  # bỏ bậc B trước
    assert d.set_index("product_id").ly_do_loai["B1"] == "vuot_tran_nhom"
    d2 = chay(*[sp(category="auto", pid=f"y{i}", thap=i) for i in range(5)], cfg={**CFG, "tham_so": {**CFG["tham_so"], "tran_moi_nhom": 2}})
    assert set(d2[d2.ly_do_loai.isna()].product_id) == {"y0", "y1"}           # không còn bậc B: tỷ lệ cặp yếu thấp được giữ


def test_nhom_khong_vuot_tran_giu_het_ke_ca_bac_b():
    d = chay(sp(category="auto", la="Xe tay ga"), sp(category="auto", la="Xe tay ga"))
    assert d.ly_do_loai.isna().all() and d.bac_b.all()


def test_ke_toan_va_tat_dinh():
    dong = [sp(category="food", la="Thức ăn cho mèo"), sp(desc="Z"), sp(desc="Z"), sp(desc="Z"), sp(so_cap=4, thap=2), sp(source="1688", category="auto", title_vi="拼")] + [sp() for _ in range(6)]
    d1, buoc = g.ap_dung_quy_tac(pd.DataFrame(dong), CFG)
    d2, _ = g.ap_dung_quy_tac(pd.DataFrame(dong), CFG)
    assert d1.ly_do_loai.fillna("").tolist() == d2.ly_do_loai.fillna("").tolist()
    assert sum(b["loai"] for b in buoc) == int(d1.ly_do_loai.notna().sum()) and buoc[-1]["con_lai"] == int(d1.ly_do_loai.isna().sum())
    assert set(d1.ly_do_loai.dropna()) <= set(g.danh_sach_ly_do(CFG))


def test_cau_hinh_that():
    cfg = g.doc_cau_hinh()
    assert cfg["tham_so"] == {"tran_moi_nhom": 3500, "nguong_cap_yeu": 0.25, "toi_da_mo_ta_trung": 3}
    assert sum(len(e["la"]) for e in cfg["bac_A"]) == 56 and sum(len(e["la"]) for e in cfg["bac_B"]) == 175
    assert [r["ly_do"] for r in cfg["quy_tac_title"]] == ["title_con_chu_han"]
    assert g.danh_sach_ly_do(cfg) == ["khong_co_description", "sai_nhom", "cap_yeu_cao", "mo_ta_trung", "title_con_chu_han", "vuot_tran_nhom"]


def test_cau_hinh_sai_bi_chan():
    for sai in ({"ly_do": "sai_nhom", "nguon": "1688", "cot": "title_vi", "regex": "x"}, {"ly_do": "ok", "nguon": "1688", "cot": "title_vi", "regex": "("}, {"ly_do": "ok2", "nguon": "khac", "cot": "title_vi", "regex": "x"}):
        import json
        import tempfile
        p = Path(tempfile.mkdtemp()) / "c.json"
        p.write_text(json.dumps({**CFG, "quy_tac_title": [sai]}), encoding="utf-8")
        try:
            g.doc_cau_hinh(p)
        except (AssertionError, KeyError, Exception) as e:  # noqa: BLE001
            assert not isinstance(e, SystemExit)
        else:
            raise AssertionError(f"cấu hình sai không bị chặn: {sai}")


def test_cong_g0_bat_ten_cot_sai():
    sai = [c if c != "title_nguon_dich" else "title_ngon_dich" for c in g.COT]
    try:
        g.kiem_ten_cot("x.parquet", sai)
    except SystemExit as e:
        assert "title_ngon_dich" in str(e)
    else:
        raise AssertionError("không chặn tên cột sai")
    assert g.kiem_ten_cot("x.parquet", g.COT) == g.COT
    assert g.kiem_ten_cot("x.parquet", sai, chap_nhan_ten_cu=True) == g.COT               # chạy thử: tên cũ được đổi
    assert g.kiem_ten_cot("x.parquet", g.COT + ["title_van_de"]) == g.COT + ["title_van_de"]   # cột phụ của Huy được nhận
    for sai2 in (g.COT[:-1], g.COT + ["cot_la"], ["title_zh"] + g.COT[1:], g.COT[::-1]):
        try:
            g.kiem_ten_cot("x.parquet", sai2, chap_nhan_ten_cu=True)
        except SystemExit:
            continue
        raise AssertionError(f"không chặn danh sách cột sai: {sai2}")


def test_chot_chong_mat_san_pham_am_tham():
    ids = [f"p{i}" for i in range(10)]
    g.kiem_dong_dau_vao("tiki", ids, ids[:8], 10)                                  # đủ: không sao (description có thể thiếu)
    for ids_full, ids_sp, du_kien in ((ids[:9], ids[:9], 10),                      # Huy xóa 1 dòng title: số dòng lệch
                                      (ids[:9], ids, 9),                           # đúng số dòng nhưng description còn giữ sản phẩm đã bị xóa title
                                      (ids + ["p10"], ids, 10)):                   # thêm dòng lạ
        try:
            g.kiem_dong_dau_vao("tiki", ids_full, ids_sp, du_kien)
        except SystemExit as e:
            assert "DỪNG" in str(e)
        else:
            raise AssertionError(f"không chặn trường hợp lệch: {len(ids_full)} dòng full, {len(ids_sp)} dòng description, dự kiến {du_kien}")
    g.kiem_dong_dau_vao("tiki", ids[:9], ids, 10, chap_nhan=True)                  # cố ý: chỉ cảnh báo


def test_quy_tac_title_dung_nhan_title_van_de():
    cfg = {**CFG, "quy_tac_title": CFG["quy_tac_title"] + [{"ly_do": "title_nghi_cat_cut", "nguon": "tiki", "cot": "title_van_de", "regex": "nghi_cat_cut_hoac_lap", "mo_ta": "x"}]}
    d = chay(sp(title_vi="a") | {"title_van_de": "nghi_lap_tu;nghi_cat_cut_hoac_lap"}, sp(title_vi="b") | {"title_van_de": "nghi_lap_tu"}, sp(title_vi="c") | {"title_van_de": ""}, cfg=cfg)
    assert d.ly_do_loai.tolist() == ["title_nghi_cat_cut", None, None]


def _bo_ra_hop_le():
    d = chay(sp(pid="a1"), sp(pid="a2"), sp(source="1688", category="auto", pid="b1"), sp(category="food", la="Thức ăn cho mèo", pid="c1"))
    giu = d[d.ly_do_loai.isna()]
    mot = pd.DataFrame({"product_id": giu.product_id, "source": giu.source, "category": giu.category, "title_zh": "标", "title_vi": "tiêu", "description_zh": "d", "description_vi": "m",
                        "title_nguon_dich": "gemma4", "desc_nguon_dich": "tu_dien_quy_tac"})[g.COT].reset_index(drop=True)
    loai = d[d.ly_do_loai.notna()][["product_id", "source", "category", "title_zh", "title_vi", "ly_do_loai"]].reset_index(drop=True)
    return d, mot, mot.copy(), loai


def test_nghiem_thu_chap_nhan_dung():
    d, cum, pair, loai = _bo_ra_hop_le()
    g.nghiem_thu(d, CFG, cum, pair, loai)


def test_nghiem_thu_bat_loi():
    def phai_loi(sua, nhan):
        d, cum, pair, loai = _bo_ra_hop_le()
        cum, pair, loai = sua(cum, pair, loai)
        try:
            g.nghiem_thu(d, CFG, cum, pair, loai)
        except AssertionError:
            return
        raise AssertionError(f"nghiệm thu không bắt lỗi: {nhan}")
    phai_loi(lambda c, p, ls: (c.assign(title_vi=None), p, ls), "ô null")
    phai_loi(lambda c, p, ls: (c.assign(title_nguon_dich="mo_hinh_la"), p, ls), "title_nguon_dich lạ")
    phai_loi(lambda c, p, ls: (c.assign(desc_nguon_dich="la"), p, ls), "desc_nguon_dich lạ")
    phai_loi(lambda c, p, ls: (c, p.iloc[::-1].reset_index(drop=True), ls), "hai bản khác thứ tự")
    phai_loi(lambda c, p, ls: (c.iloc[:-1], p.iloc[:-1], ls), "giữ + loại thiếu một sản phẩm")
    phai_loi(lambda c, p, ls: (c, p, ls.assign(ly_do_loai="ly_do_la")), "ly_do_loai lạ")
    phai_loi(lambda c, p, ls: (c.assign(category="shoes"), p.assign(category="shoes"), ls), "category lạ ở Tiki")


def main():
    ten = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in ten:
        globals()[n]()
        print("OK", n)
    print(f"{len(ten)}/{len(ten)} test đạt")


if __name__ == "__main__":
    main()
