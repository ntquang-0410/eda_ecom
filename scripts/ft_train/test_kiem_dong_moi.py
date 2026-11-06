"""Test kiem_dong_moi.py. Chạy: python -I -X utf8 scripts/ft_train/test_kiem_dong_moi.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cham_diem as D  # noqa: E402
import kaggle_train as K  # noqa: E402
import kiem_dong_moi as M  # noqa: E402


def _hang(id_, huong, dong, can=True, loai="mo_ta"):
    return {"id": id_, "huong": huong, "loai": loai, "can_dong": can, "_dong": dong}


def test_lay_dong_moi_chi_lay_dong_moi_dung_huong():
    hang = [_hang("a", "zh->vi", [(True, "x", "x"), (False, "y", "z")]),
            _hang("b", "zh->vi", [(True, "x", "x")]),                      # toàn dòng đã thấy
            _hang("c", "zh->vi", None, can=False),                         # không khớp dòng
            _hang("d", "vi->zh", [(False, "m", "n")]),                     # khác hướng
            _hang("e", "zh->vi", None, can=float("nan")),                  # NaN không được coi là khớp
            _hang("f", "zh->vi", None, can=False, loai="title")]           # không phải mô tả
    ra, khong_can, khong_moi = M.lay_dong_moi(hang, "zh->vi")
    assert ra == {"a": [("y", "z")]}
    assert (khong_can, khong_moi) == (2, 1)


def test_ghep_cap_chi_giu_san_pham_co_o_ca_hai_ben():
    a = {"1": [("h1", "r1")], "2": [("h2", "r2")]}
    b = {"2": [("k2", "r2")], "3": [("k3", "r3")]}
    cap = M.ghep_cap(a, b, ["1", "2", "3"])
    assert cap == [{"id": "2", "ref": ["r2"], "hyp_a": ["h2"], "hyp_b": ["k2"]}]


def test_ghep_cap_khac_tham_chieu_thi_loi():
    try:
        M.ghep_cap({"1": [("h", "r1")]}, {"1": [("k", "r2")]}, ["1"])
    except RuntimeError as e:
        assert "tham_chieu_khac_nhau" in str(e)
    else:
        raise AssertionError("phải báo lỗi")


def test_ket_luan_bien():
    assert M.ket_luan(0.01) == M.QUA
    assert M.ket_luan(0.0) == M.DUNG       # đúng bằng 0 là chứa 0, không qua
    assert M.ket_luan(-0.5) == M.DUNG
    assert M.ket_luan(None) == M.CHUA_DU
    assert M.ket_luan(float("nan")) == M.CHUA_DU


def test_mau_thuan():
    assert M.mau_thuan({"thap": 0.1}, {"thap": -0.1}) is True
    assert M.mau_thuan({"thap": -0.2}, {"thap": -0.1}) is False


def test_tap_trung_tinh_tay():
    x = [-2.0, -1.0, 0.0] + [1.0] * 8 + [10.0, 20.0, 30.0, 40.0, 50.0]
    t = M.tap_trung(x, so_cat=5)
    assert (t["n"], t["tang"], t["giam"], t["bang"]) == (16, 13, 2, 1)
    assert t["trung_vi"] == 1.0
    assert abs(t["tb_bo_tang_nhat"] - 5.0 / 11) < 1e-9     # bỏ 10..50, còn -2,-1,0 và tám số 1
    assert abs(t["tb_cat_hai_dau"] - 1.0) < 1e-9           # còn sáu số 1


def test_cong_thong_ke_ra_dung_chrf_corpus_cua_sacrebleu():
    """Phép bootstrap cộng thống kê phải khớp chrf_corpus, nếu không khoảng tin cậy nói về một số khác."""
    cap = [(["Áo thun cotton màu đen", "Chất liệu: vải"], ["Áo phông cotton đen", "Chất liệu: vải bông"]),
           (["Kích thước 10cm"], ["Kích thước: 10 cm"]),
           (["Giày da nam", "Xuất xứ: Trung Quốc"], ["Giày da cho nam", "Nơi sản xuất: Trung Quốc"])]
    for huong in ("zh->vi", "vi->zh"):
        m = D.tao_chrf(huong)
        tk = np.array([M.thong_ke(m, h, r) for h, r in cap])
        tu_cong = float(m._aggregate_and_compute([[int(v) for v in tk.sum(axis=0)]]).score)
        goc = D.chrf_corpus(huong, [x for h, _ in cap for x in h], [x for _, r in cap for x in r])
        assert abs(tu_cong - goc) < 1e-9, (huong, tu_cong, goc)


def test_bootstrap_corpus_hai_ben_giong_nhau_thi_chenh_0():
    m = D.tao_chrf("zh->vi")
    tk = np.array([M.thong_ke(m, ["Áo thun đen"], ["Áo phông đen"]), M.thong_ke(m, ["Giày da"], ["Giày da nam"])])
    kq = M.bootstrap_corpus([(m, tk, tk.copy())], so_lan=50)
    assert kq == {"chenh": 0.0, "thap": 0.0, "cao": 0.0}


def test_bootstrap_corpus_ben_a_dung_het_thi_cao_hon_ro():
    m = D.tao_chrf("zh->vi")
    ref = ["Áo phông cotton màu đen", "Giày da cho nam giới", "Túi xách nữ thời trang", "Mũ lưỡi trai vải kaki"]
    sai = ["Bàn học gỗ", "Đèn ngủ nhỏ", "Quạt điện mini", "Nồi cơm điện"]
    a = np.array([M.thong_ke(m, [r], [r]) for r in ref])
    b = np.array([M.thong_ke(m, [s], [r]) for s, r in zip(sai, ref)])
    kq = M.bootstrap_corpus([(m, a, b)], so_lan=200)
    assert kq["thap"] > 0 and kq["chenh"] > 0
    assert kq["thap"] <= kq["chenh"] <= kq["cao"]


def test_bootstrap_chenh_mot_nhom_la_trung_binh_chenh():
    """Chỉ số chính gọi bootstrap_chenh với một nhóm: chenh phải đúng trung bình chênh, không bị macro làm khác đi."""
    a, b = [3.0, 5.0, 7.0], [1.0, 1.0, 1.0]
    kq = K.bootstrap_chenh(a, b, ["zh->vi|mo_ta"] * 3, so_lan=200, seed=42)
    assert abs(kq["chenh"] - 4.0) < 1e-12
    assert 2.0 <= kq["thap"] <= kq["cao"] <= 6.0


def test_cum_chung_dong_gom_ca_qua_trung_gian():
    cum = M.cum_chung_dong([["A", "b"], ["c"], ["B", "d"], ["D"], ["e"]])
    assert cum == [0, 1, 0, 0, 2]        # 0-2 chung "b" (khác hoa), 2-3 chung "d"


def test_bootstrap_cum_chenh_khop_trung_binh_va_rong_hon_khi_trung():
    chenh = [5.0] * 10 + [-5.0] * 10
    rieng = M.bootstrap_cum(chenh, list(range(20)), so_lan=500)
    gop = M.bootstrap_cum(chenh, [0] * 10 + [1] * 10, so_lan=500)     # hai cụm lớn: thực chất chỉ có 2 bằng chứng
    assert rieng["chenh"] == 0.0 and gop["chenh"] == 0.0
    assert (rieng["so_cum"], gop["so_cum"]) == (20, 2)
    assert gop["cao"] - gop["thap"] > rieng["cao"] - rieng["thap"]


def test_ti_le_chi_khac_hoa():
    cap = [{"hyp_a": ["Trắng", "Đen", "Mũ"], "hyp_b": ["trắng", "Xám", "Mũ"]}]
    assert M.ti_le_chi_khac_hoa(cap) == {"dong_khac": 2, "chi_khac_hoa": 1, "ti_le": 0.5}
    assert M.ti_le_chi_khac_hoa([{"hyp_a": ["x"], "hyp_b": ["x"]}])["ti_le"] is None


def test_chrf_khac_hoa_sau_khi_ha_chu_thi_bang_nhau():
    """Xác nhận cách đọc mẫu: chrF phân biệt hoa thường, hạ chữ xong thì hai bản này bằng nhau."""
    ref = "Màu sắc: 7007 trắng tinh, 7007 trắng đen"
    hoa = "Màu sắc: 7007 Trắng tinh, 7007 Trắng đen"
    assert D.chrf_corpus("zh->vi", [hoa], [ref]) < 90
    assert D.chrf_corpus("zh->vi", [hoa.lower()], [ref.lower()]) == D.chrf_corpus("zh->vi", [ref.lower()], [ref.lower()])


def test_quyet_fold_luat_khoa():
    import kiem_fold_ti_le as F
    assert F.quyet(0.1, 60.0, 59.5)[0] == F.DAT
    assert F.quyet(0.1, 60.0, 59.0)[0] == F.DAT            # tụt đúng 1 điểm vẫn đạt
    assert F.quyet(0.1, 60.0, 58.9)[0] == F.KHONG_DAT      # tụt quá 1 điểm
    assert F.quyet(0.0, 60.0, 61.0)[0] == F.KHONG_DAT      # cận dưới đúng bằng 0 là chứa 0
    assert F.quyet(-0.3, 60.0, 61.0)[0] == F.KHONG_DAT
    assert F.quyet(None, 60.0, 60.0)[0] == F.CHUA_DU
    assert F.quyet(0.1, None, 60.0)[0] == F.CHUA_DU
    assert F.quyet(0.1, float("nan"), 60.0)[0] == F.CHUA_DU
    assert F.quyet(-0.3, None, 60.0)[0] == F.KHONG_DAT     # đã có bằng chứng không đạt thì không chờ số còn thiếu
    assert F.quyet(None, 60.0, 58.0)[0] == F.KHONG_DAT


if __name__ == "__main__":
    ten = [k for k in list(globals()) if k.startswith("test_")]
    for k in ten:
        globals()[k]()
        print("đạt", k)
    print(f"ĐẠT {len(ten)}/{len(ten)}")
