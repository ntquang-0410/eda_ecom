"""Bài thử cho ghep_bo.py (ghép hai bộ cuối từ quyết định của judge). Chạy: python -I -X utf8 scripts/judge/test_ghep_bo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import ghep_bo as G  # noqa: E402
import so_cap_du_lieu as DL  # noqa: E402


def base_1688():
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3"], "source": "1688", "category": "home",
        "title_zh": ["原1", "原2", "原3"], "title_vi": ["Cu1", "Cu2", "Cu3"],
        "description_zh": ["甲: 1; 乙: 2", "甲: 3", "丙: 4"], "description_vi": ["A: 1; B: 2", "A: 3", "C: 4"],
        "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})


def test_title_thay_theo_quyet_dinh_va_chan():
    b = base_1688()
    t = {"1688": pd.DataFrame({"product_id": ["p1", "p2", "p3"],
                               "gemma4_translation": ["G1", "G2", "G3"],
                               "qwen3_14b_awq_translation": ["Q1", "Q2 chữ 汉字", "Cu3"]})}
    qd = {"title:1688:p1": "qwen", "title:1688:p2": "qwen", "title:1688:p3": "qwen"}      # p2 bị chặn (Hán sót), p3 trùng hiện trạng
    d, tk = G.ghep_title(b, t, qd, "giu_hien_tai")
    assert d.title_vi.tolist() == ["Q1", "Cu2", "Cu3"] and d.title_nguon_dich.tolist() == ["qwen3_14b_awq", "gemma4", "gemma4"]
    assert tk == {"doi_sang_qwen": 1, "chan_do_chu_han_hoac_chu_viet": 1, "trung_hien_trang": 1}
    # mục không có quyết định = giữ nguyên; không quyết định theo chính sách
    d, _ = G.ghep_title(b, t, {"title:1688:p1": "khong_quyet"}, "giu_hien_tai")
    assert d.title_vi.tolist() == ["Cu1", "Cu2", "Cu3"]
    d, _ = G.ghep_title(b, t, {"title:1688:p1": "khong_quyet"}, "gemma")
    assert d.title_vi.tolist() == ["G1", "Cu2", "Cu3"]


def test_tiki_chan_chu_viet_co_dau():
    b = pd.DataFrame({"product_id": ["t1"], "source": "tiki", "category": "auto", "title_zh": ["中文"], "title_vi": ["Gốc"],
                      "description_zh": ["x"], "description_vi": ["y"], "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
    t = {"tiki": pd.DataFrame({"product_id": ["t1"], "gemma4_translation": ["中文 Chữ Việt"],
                               "qwen3_14b_awq_translation": ["中文乙"]})}
    d, tk = G.ghep_title(b, t, {"title:tiki:t1": "gemma"}, "giu_hien_tai")
    assert d.title_zh.tolist() == ["中文"] and tk == {"chan_do_chu_han_hoac_chu_viet": 1}
    d, tk = G.ghep_title(b, t, {"title:tiki:t1": "qwen"}, "giu_hien_tai")
    assert d.title_zh.tolist() == ["中文乙"] and tk == {"doi_sang_qwen": 1}


def cap_cum():
    # p1: cặp 1 hiện trạng = A (Gemma), cặp 2 bị khóa (không bằng A hay B); p2: hiện trạng = B
    return pd.DataFrame({
        "pair_id": [1, 2, 3, 4], "product_id": ["p1", "p1", "p2", "p3"],
        "nguon_ten": ["甲", "乙", "甲", "丙"], "nguon_gt": ["1", "2", "3", "4"],
        "cuoi_ten": ["A", "KHOA", "A", "C"], "cuoi_gt": ["1-g", "2-quy-tac", "3-q", "4-g"],
        "nA": ["A", "B", "A", "C"], "vA": ["1-g", "2-g", "3-g", "4-g"],
        "nB": ["A", "B", "A", "C"], "vB": ["1-q", "2-q", "3-q", "4-q"]})


def id_c(r, cat="home"):
    k = DL.khoa_cum(r.nguon_ten + ": " + r.nguon_gt, r.nA + ": " + r.vA, r.nB + ": " + r.vB, cat)
    return DL.id_cum("1688", k)


def test_cum_thay_dung_cap_va_khoa_cung():
    b, p = base_1688(), cap_cum()
    ids = [id_c(r) for r in p.itertuples()]
    # p1 cặp 1: judge chọn Qwen (hiện trạng là A) -> đổi; p1 cặp 2: bị khóa -> giữ; p2 cặp 3: chọn Gemma (hiện trạng là B) -> đổi
    qd = {ids[0]: "qwen", ids[1]: "qwen", ids[2]: "gemma"}
    moi, tk = G.ghep_mota_cum(b, p, qd, "giu_hien_tai")
    assert moi["p1"] == "A: 1-q; KHOA: 2-quy-tac" and moi["p2"] == "A: 3-g" and moi["p3"] == "C: 4-g"
    assert tk == {"doi_sang_qwen": 1, "khoa_giu_nguyen": 1, "doi_sang_gemma": 1}
    moi, tk = G.ghep_mota_cum(b, p, {}, "giu_hien_tai")                      # không quyết định: dựng lại giống hệt
    assert moi["p1"] == "A: 1-g; KHOA: 2-quy-tac" and tk == {}


def test_cum_ngu_canh_quyet_dinh_rieng_cho_tung_san_pham():
    """Có ngữ cảnh: cùng một cặp ở hai sản phẩm có thể ra hai quyết định khác nhau; id không ngữ cảnh bị bỏ qua."""
    b = pd.DataFrame({"product_id": ["p1", "p2"], "source": "1688", "category": "home", "title_zh": ["原1", "原2"],
                      "title_vi": ["Cu1", "Cu2"], "description_zh": ["甲: 1", "甲: 1"], "description_vi": ["A: 1-g", "A: 1-g"],
                      "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
    p = pd.DataFrame({"pair_id": [1, 2], "product_id": ["p1", "p2"], "nguon_ten": "甲", "nguon_gt": "1", "cuoi_ten": "A",
                      "cuoi_gt": "1-g", "nA": "A", "vA": "1-g", "nB": "A", "vB": "1-q"})
    qd = {DL.id_cum_ctx("1688", "p1", 1): "qwen", DL.id_cum_ctx("1688", "p2", 2): "khong_quyet"}
    moi, tk = G.ghep_mota_cum(b, p, qd, "giu_hien_tai", ngu_canh=True)
    assert moi["p1"] == "A: 1-q" and moi["p2"] == "A: 1-g" and tk == {"doi_sang_qwen": 1}
    # khóa cũ (không ngữ cảnh) không có tác dụng ở chế độ ngữ cảnh, và ngược lại
    k = DL.id_cum("1688", DL.khoa_cum("甲: 1", "A: 1-g", "A: 1-q", "home"))
    moi, tk = G.ghep_mota_cum(b, p, {k: "qwen"}, "giu_hien_tai", ngu_canh=True)
    assert moi["p1"] == "A: 1-g" and tk == {}
    moi, tk = G.ghep_mota_cum(b, p, {k: "qwen"}, "giu_hien_tai", ngu_canh=False)
    assert moi["p1"] == "A: 1-q" and moi["p2"] == "A: 1-q" and tk == {"doi_sang_qwen": 2}
    moi, tk = G.ghep_mota_cum(b, p, {}, "giu_hien_tai", ngu_canh=True)                 # không quyết định: giống hệt bộ gốc
    assert moi["p1"] == "A: 1-g" and moi["p2"] == "A: 1-g" and tk == {}


def test_cum_lai_cap_1_san_pham_dung_cumctx_cap_nhieu_san_pham_dung_pair():
    b = pd.DataFrame({"product_id": ["p1", "p2", "p3"], "source": "1688", "category": "home", "title_zh": "原", "title_vi": "Cu",
                      "description_zh": "x", "description_vi": "y", "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
    # cặp 甲:1 dùng ở p1 và p2 (dùng chung, id pair:...); cặp 乙:2 chỉ ở p3 (id cumctx:...)
    p = pd.DataFrame({"pair_id": [1, 2, 3], "product_id": ["p1", "p2", "p3"], "nguon_ten": ["甲", "甲", "乙"], "nguon_gt": ["1", "1", "2"],
                      "cuoi_ten": ["A", "A", "B"], "cuoi_gt": ["1-g", "1-g", "2-g"], "nA": ["A", "A", "B"], "vA": ["1-g", "1-g", "2-g"],
                      "nB": ["A", "A", "B"], "vB": ["1-q", "1-q", "2-q"]})
    khoa_chung = DL.id_cum("1688", DL.khoa_cum("甲: 1", "A: 1-g", "A: 1-q", "home"))
    khoa_rieng = DL.id_cum("1688", DL.khoa_cum("乙: 2", "B: 2-g", "B: 2-q", "home"))
    ctx_p3 = DL.id_cum_ctx("1688", "p3", 3)
    qd = {khoa_chung: "qwen", ctx_p3: "qwen", khoa_rieng: "gemma"}      # khoa_rieng (id không ngữ cảnh của cặp 1 sản phẩm) phải BỊ BỎ QUA ở chế độ lai
    moi, tk = G.ghep_mota_cum(b, p, qd, "giu_hien_tai", ngu_canh="lai")
    assert moi["p1"] == "A: 1-q" and moi["p2"] == "A: 1-q" and moi["p3"] == "B: 2-q" and tk == {"doi_sang_qwen": 3}
    moi, tk = G.ghep_mota_cum(b, p, {khoa_rieng: "qwen"}, "giu_hien_tai", ngu_canh="lai")
    assert moi["p3"] == "B: 2-g" and tk == {}                          # chế độ lai không đọc id không ngữ cảnh cho cặp 1 sản phẩm
    moi, tk = G.ghep_mota_cum(b, p, {}, "giu_hien_tai", ngu_canh="lai")
    assert moi["p1"] == "A: 1-g" and moi["p3"] == "B: 2-g" and tk == {}


def test_tuong_duong_chinh_sach_va_nhan():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ghi_kq(tmp, "m", [
            {"item_id": "title:1688:p1", "ket_qua": "qwen", "che_do": "mot_luot", "luot": [{"tuong_duong": False}]},
            {"item_id": "title:1688:p2", "ket_qua": "qwen", "che_do": "mot_luot", "luot": [{"tuong_duong": True}]},
            {"item_id": "title:1688:p3", "ket_qua": "gemma", "che_do": "mot_luot", "luot": [{"tuong_duong": True}]}])
        kqdd, co = G.doc_ket_qua_va_tuong_duong("m", thu_muc=tmp)
        assert co == {"title:1688:p2", "title:1688:p3"}
        b = pd.DataFrame({"product_id": ["p1", "p2", "p3", "p4"], "source": "1688", "category": "home", "title_zh": "原",
                          "title_vi": ["Cu1", "Cu2", "Cu3", "Cu4"], "description_zh": "x", "description_vi": "y",
                          "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
        t = {"1688": pd.DataFrame({"product_id": ["p1", "p2", "p3", "p4"], "gemma4_translation": ["G1", "G2", "G3", "G4"],
                                   "qwen3_14b_awq_translation": ["Q1", "Q2", "Q3", "Q4"]})}
        for chinh_sach, mong_doi in (("giu_hien_tai", ["Q1", "Cu2", "Cu3", "Cu4"]), ("theo_judge", ["Q1", "Q2", "G3", "Cu4"])):
            qd, kq2 = G.ap_chinh_sach_tuong_duong(kqdd, co, chinh_sach)
            dau = G.Dau()
            d, _ = G.ghep_title(b, t, qd, "giu_hien_tai", kq2, dau)
            assert d.title_vi.tolist() == mong_doi, chinh_sach
            assert dau.dem() == {"title:judge_t1": 1, "title:judge_tuong_duong": 2, "title:chua_cham": 1}, chinh_sach
        # ngay cả khi chính sách "không quyết" là gemma, mục tương đương giữ hiện trạng (không bị ép về gemma)
        qd, kq2 = G.ap_chinh_sach_tuong_duong(kqdd, co, "giu_hien_tai")
        d, _ = G.ghep_title(b, t, qd, "gemma", kq2)
        assert d.title_vi.tolist()[1:3] == ["Cu2", "Cu3"]


def test_pair_thay_rieng_ten_va_gia_tri():
    b = base_1688()
    p = pd.DataFrame({"pair_id": [1, 2], "product_id": ["p1", "p2"], "nguon_ten": ["甲", "甲"], "nguon_gt": ["红", "蓝"],
                      "cuoi_ten": ["Màu", "Màu"], "cuoi_gt": ["Đỏ", "Khóa-cứng"]})
    u = pd.DataFrame({"ma_don_vi": ["T1", "V1", "V2"], "loai": ["ten", "gia_tri", "gia_tri"], "van_ban": ["甲", "红", "蓝"],
                      "A": ["Màu", "Đỏ", "Xanh"], "B": ["Màu sắc", "Đỏ tươi", "Lam"]})
    qd = {"unit:1688:T1": "qwen", "unit:1688:V1": "qwen", "unit:1688:V2": "qwen"}
    moi, tk = G.ghep_mota_pair(b, p, u, qd, "giu_hien_tai")
    assert moi["p1"] == "Màu sắc: Đỏ tươi"
    assert moi["p2"] == "Màu sắc: Khóa-cứng"                                     # giá trị bị khóa giữ nguyên, tên vẫn đổi
    assert tk["doi_sang_qwen"] == 3 and tk["khoa_giu_nguyen"] == 1
    moi, _ = G.ghep_mota_pair(b, p, u, {}, "giu_hien_tai")
    assert moi["p1"] == "Màu: Đỏ" and moi["p2"] == "Màu: Khóa-cứng"             # không quyết định: dựng lại giống hệt


def ghi_kq(thu_muc, ten, dong):
    import json
    (Path(thu_muc) / f"ket_qua_{ten}.jsonl").write_text(
        "\n".join(json.dumps({"trang_thai": "ok", **d}, ensure_ascii=False) for d in dong), encoding="utf-8")


def test_tang2_thang_tang1_bat_ke_thu_tu_dot():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ghi_kq(tmp, "t1", [{"item_id": "a", "ket_qua": "khong_nhat_quan"}, {"item_id": "b", "ket_qua": "gemma"},
                           {"item_id": "c", "ket_qua": "hoa_that"}])
        ghi_kq(tmp, "t2", [{"item_id": "a", "ket_qua": "qwen", "tang": 2}, {"item_id": "c", "ket_qua": "khong_nhat_quan", "tang": 2}])
        for ten in (("t1", "t2"), ("t2", "t1")):                      # thứ tự đợt trong dòng lệnh KHÔNG làm tầng 1 đè tầng 2
            kq = G.doc_ket_qua_day_du(*ten, thu_muc=tmp)
            assert kq == {"a": ("qwen", 2), "b": ("gemma", 1), "c": ("khong_nhat_quan", 2)}, ten
            assert G.doc_quyet_dinh(*ten, thu_muc=tmp) == {"a": "qwen", "b": "gemma", "c": "khong_quyet"}
        assert [G.nhan_nguon(i, {}, kq) for i in "abcd"] == ["judge_t2", "judge_t1", "khong_nhat_quan", "chua_cham"]
        ghi_kq(tmp, "t3", [{"item_id": "a", "ket_qua": "gemma"}, {"item_id": "x", "ket_qua": "hoa_that"}])
        kq = G.doc_ket_qua_day_du("t1", "t2", "t3", thu_muc=tmp)                 # cùng tầng 1: đợt sau ghi đè đợt trước; tầng 2 vẫn thắng
        assert kq["a"] == ("qwen", 2) and G.nhan_nguon("x", {}, kq) == "hoa_that"


def test_danh_dau_title_chi_cho_muc_tranh_chap():
    b = pd.DataFrame({"product_id": [f"p{i}" for i in range(1, 6)], "source": "1688", "category": "home",
                      "title_zh": "原", "title_vi": [f"Cu{i}" for i in range(1, 6)], "description_zh": "x", "description_vi": "y",
                      "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
    t = {"1688": pd.DataFrame({"product_id": [f"p{i}" for i in range(1, 6)],
                               "gemma4_translation": ["G1", "G2", "G3", "Giống", "G5"],
                               "qwen3_14b_awq_translation": ["Q1", "Q2", "Q3", "giống ", "Q5"]})}       # p4 giống nhau (chuẩn hóa): không tranh chấp
    kq = {"title:1688:p1": ("qwen", 2), "title:1688:p2": ("gemma", 1), "title:1688:p3": ("hoa_that", 1)}   # p5: chưa chấm
    qd = G.tu_ket_qua_day_du(kq)
    dau = G.Dau()
    d, tk = G.ghep_title(b, t, qd, "giu_hien_tai", kq, dau)
    assert d.title_vi.tolist() == ["Q1", "G2", "Cu3", "Cu4", "Cu5"] and tk == {"doi_sang_qwen": 1, "doi_sang_gemma": 1}
    assert dau.dem() == {"title:judge_t1": 1, "title:judge_t2": 1, "title:hoa_that": 1, "title:chua_cham": 1}
    bang = dau.bang(b.product_id).set_index("product_id")
    assert bang.title_nhan.to_dict() == {"p1": "judge_t2", "p2": "judge_t1", "p3": "hoa_that", "p4": "khong_tranh_chap", "p5": "chua_cham"}
    assert bang.title_da_doi.to_dict() == {"p1": True, "p2": True, "p3": False, "p4": False, "p5": False}
    # đánh dấu không ảnh hưởng kết quả ghép
    d2, tk2 = G.ghep_title(b, t, qd, "giu_hien_tai")
    assert d2.equals(d) and tk2 == tk


def test_danh_dau_mo_ta_cum_va_pair_khong_dem_muc_khoa_cung():
    b, p = base_1688(), cap_cum()
    ids = [id_c(r) for r in p.itertuples()]
    kq = {ids[0]: ("qwen", 1), ids[1]: ("khong_nhat_quan", 1)}                  # cặp 3, 4: chưa chấm
    dau = G.Dau()
    moi, tk = G.ghep_mota_cum(b, p, G.tu_ket_qua_day_du(kq), "giu_hien_tai", kqdd=kq, dau=dau)
    assert moi["p1"] == "A: 1-q; KHOA: 2-quy-tac" and tk == {"doi_sang_qwen": 1}
    assert dau.dem() == {"mo_ta:judge_t1": 1, "mo_ta:khong_nhat_quan": 1, "mo_ta:chua_cham": 2}
    bang = dau.bang(b.product_id).set_index("product_id")
    assert bang.loc["p1", ["mo_ta_so_manh_tranh_chap", "mo_ta_so_manh_judge_t1", "mo_ta_so_manh_khong_nhat_quan", "mo_ta_so_manh_da_doi"]].tolist() == [2, 1, 1, 1]
    assert bang.loc["p2", ["mo_ta_so_manh_tranh_chap", "mo_ta_so_manh_chua_cham", "mo_ta_so_manh_da_doi"]].tolist() == [1, 1, 0]
    # bộ PAIR: đơn vị tu == 'co_dinh' không phải tranh chấp
    pr = pd.DataFrame({"pair_id": [1, 2], "product_id": ["p1", "p2"], "nguon_ten": ["甲", "甲"], "nguon_gt": ["红", "蓝"],
                       "cuoi_ten": ["Màu", "Màu"], "cuoi_gt": ["Đỏ", "Xanh"]})
    u = pd.DataFrame({"ma_don_vi": ["T1", "V1", "V2"], "loai": ["ten", "gia_tri", "gia_tri"], "van_ban": ["甲", "红", "蓝"],
                      "A": ["Màu", "Đỏ", "Xanh"], "B": ["Màu sắc", "Đỏ tươi", "Lam"], "tu": ["A", "A", "co_dinh"]})
    kq2 = {"unit:1688:T1": ("qwen", 1), "unit:1688:V1": ("gemma", 2)}
    dau = G.Dau()
    moi, tk = G.ghep_mota_pair(b, pr, u, G.tu_ket_qua_day_du(kq2), "giu_hien_tai", kq2, dau)
    assert moi["p1"] == "Màu sắc: Đỏ" and moi["p2"] == "Màu sắc: Xanh"
    assert dau.dem() == {"mo_ta:judge_t1": 2, "mo_ta:judge_t2": 1}          # V2 là co_dinh: không đếm; T1 xuất hiện ở p1 và p2


def test_pair_don_vi_theo_nhom_hang_khong_dung_nham_bien_the():
    """Cùng chữ nguồn có thể có nhiều ma_don_vi (mỗi nhóm hàng một đơn vị): mỗi sản phẩm phải dùng đơn vị của NHÓM HÀNG của nó (lỗi 31/10:
    trước đây chỉ lấy biến thể đầu tiên, mảnh của nhóm hàng khác bị gán nhầm/không có quyết định)."""
    b = pd.DataFrame({"product_id": ["p1", "p2"], "source": "1688", "category": ["home", "bags"], "title_zh": "原", "title_vi": "Cu",
                      "description_zh": "x", "description_vi": "y", "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
    p = pd.DataFrame({"pair_id": [1, 2], "product_id": ["p1", "p2"], "nguon_ten": ["材质", "材质"], "nguon_gt": ["其他", "其他"],
                      "cuoi_ten": ["Chất liệu", "Chất liệu"], "cuoi_gt": ["Khác", "Loại khác"]})      # hiện trạng: home = A của biến thể home, bags = A của biến thể bags
    u = pd.DataFrame({"ma_don_vi": ["H1", "B1", "HT", "BT"], "loai": ["gia_tri", "gia_tri", "ten", "ten"], "van_ban": ["其他", "其他", "材质", "材质"],
                      "A": ["Khác", "Loại khác", "Chất liệu", "Chất liệu"], "B": ["Cái khác", "Khác nữa", "Chất liệu", "Chất liệu"],
                      "category": ["home", "bags", "home", "bags"]})
    qd = {"unit:1688:H1": "qwen", "unit:1688:B1": "qwen"}
    dau = G.Dau()
    moi, tk = G.ghep_mota_pair(b, p, u, qd, "giu_hien_tai", {k: (v, 1) for k, v in qd.items()}, dau)
    assert moi["p1"] == "Chất liệu: Cái khác" and moi["p2"] == "Chất liệu: Khác nữa"               # mỗi sản phẩm dùng đơn vị đúng nhóm hàng
    assert tk == {"doi_sang_qwen": 2} and dau.dem() == {"mo_ta:judge_t1": 2}
    qd2 = {"unit:1688:H1": "qwen"}                                                                 # chỉ biến thể home có quyết định: bags giữ nguyên
    moi, _ = G.ghep_mota_pair(b, p, u, qd2, "giu_hien_tai")
    assert moi["p1"] == "Chất liệu: Cái khác" and moi["p2"] == "Chất liệu: Loại khác"


def test_cach_1_nhan_goc_va_so_manh_da_doi():
    b = base_1688()
    p = cap_cum()
    ids = [id_c(r) for r in p.itertuples()]
    kq = {ids[0]: ("qwen", 1), ids[2]: ("gemma", 1)}
    dau = G.Dau()
    moi, _ = G.ghep_mota_cum(b, p, G.tu_ket_qua_day_du(kq), "giu_hien_tai", kqdd=kq, dau=dau)
    d = b.copy()
    d["description_vi"] = [moi.get(pid, cu) if pid in moi.index else cu for pid, cu in zip(d.product_id, d.description_vi)]
    r = G.hoan_thien_cot(d, dau.bang(d.product_id))
    assert "desc_nguon_dich" not in r.columns and r.desc_nguon_dich_goc.tolist() == b.desc_nguon_dich.tolist()      # nhãn cũ giữ nguyên, đổi tên
    assert "title_nguon_dich" in r.columns                                                                         # tiêu đề giữ tên cũ
    assert r.desc_so_manh_da_doi.tolist() == [1, 1, 0]                                                              # p1 đổi 1 mảnh, p2 đổi 1 mảnh, p3 không
    try:
        G.hoan_thien_cot(d, dau.bang(list(reversed(d.product_id.tolist()))))
        raise AssertionError("sai thứ tự phải báo lỗi")
    except AssertionError as e:
        assert "thứ tự" in str(e)


def test_cung_chinh_sach_cho_hai_bo_va_khong_dong_den_san_pham_khac():
    b = base_1688()
    p = cap_cum()
    du = {"title": {"1688": pd.DataFrame({"product_id": ["p1", "p2", "p3"], "gemma4_translation": ["G1", "G2", "G3"],
                                          "qwen3_14b_awq_translation": ["Q1", "Q2", "Q3"]})},
          "cum": {"1688": p, "tiki": p.iloc[0:0]}}
    ids = [id_c(r) for r in p.itertuples()]
    d, tk = G.ghep_mot_bo(b, "cum", {"title:1688:p1": "gemma"}, {ids[3]: "qwen"}, "giu_hien_tai", du)
    assert d.title_vi.tolist() == ["G1", "Cu2", "Cu3"]
    assert d.description_vi.tolist() == ["A: 1-g; KHOA: 2-quy-tac", "A: 3-q", "C: 4-q"]      # p3 được Qwen, p2 dựng lại giống hệt
    assert d.product_id.tolist() == b.product_id.tolist() and d.description_zh.tolist() == b.description_zh.tolist()


if __name__ == "__main__":
    n = 0
    for ten, f in list(globals().items()):
        if ten.startswith("test_"):
            f()
            n += 1
            print("ok", ten)
    print(f"{n}/{n} bài thử đạt")
