"""Thử tong_hop_b78 bằng dữ liệu tự dựng nhỏ (không đọc dữ liệu thật). Chạy: python -I -X utf8 scripts/ft_train/test_tong_hop_b78.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tong_hop_b78 as T  # noqa: E402


def _cham(macro, sinh_lap=0, so_mau=1000, zv=None, vz=None):
    return {"epoch": "2", "giay_moi_mau": 0.5, "so_mau": so_mau, "sinh_lap": sinh_lap, "chrf": {"chrf_macro_huong": macro},
            "theo_huong": {"zh->vi": {"chrf_corpus": macro if zv is None else zv}, "vi->zh": {"chrf_corpus": macro if vz is None else vz}},
            "theo_nhom": {"zh->vi/title": {"chrf_corpus": macro, "ti_le_giu_so_dat": 0.9, "ti_le_giu_latin_dat": 0.8,
                                           "ti_le_sai_chu_viet": 0.01, "ti_le_sinh_lap": 0.0}},
            "theo_category": {"zh->vi/title/home": {"n": 10, "chrf_corpus": macro}},
            "dong_da_thay_moi": {"zh->vi": {"da_thay": {"n_dong": 5, "chrf_corpus": macro}, "moi": {"n_dong": 0, "chrf_corpus": None}}}}


def _loss(macro, cat=None):
    return {"diem_macro": macro, "theo_nhom": {"zh->vi|title": 1.0, "vi->zh|title": 1.0, "zh->vi|desc": 1.0, "vi->zh|desc": 1.0},
            "theo_category": cat or {}}


def _viec(nhanh, fold, macro=50.0, loss=1.0, cat=None, **kw):
    return {"nhanh": nhanh, "fold": fold, "cau_hinh": "lr2e-4_r64", "epoch_xong": 2, "diem": {"1": 1.1, "2": 1.0},
            "token_moi_giay": {"1": 3000, "2": 3100}, "loss_ca_fold": _loss(loss, cat), "cham": _cham(macro, **kw)}


def test_c1_bien_chrf():
    cum = _viec("cum", 0, 52.0)
    assert T.kiem_c1(cum, _viec("pair", 0, 50.0))["trang_thai"] == "khong_bat", "đúng 2,0 thì không bật"
    kq = T.kiem_c1(cum, _viec("pair", 0, 49.99))
    assert kq["trang_thai"] == "bat" and kq["hanh_dong"] == T.HANH_DONG_C1
    assert T.kiem_c1(cum, _viec("pair", 0, 60.0))["trang_thai"] == "khong_bat", "pair hơn cum thì không bật"


def test_c1_bien_sinh_lap():
    cum = _viec("cum", 0, 50.0, sinh_lap=10)  # 0,010
    assert T.kiem_c1(cum, _viec("pair", 0, 50.0, sinh_lap=20))["trang_thai"] == "khong_bat", "chênh đúng 0,01 thì không bật"
    kq = T.kiem_c1(cum, _viec("pair", 0, 50.0, sinh_lap=21))
    assert kq["trang_thai"] == "bat" and len(kq["ly_do"]) == 1


def test_c1_thieu():
    cum = _viec("cum", 0)
    for a, b in [(None, None), (cum, None), (None, cum), (cum, {"cham": None}), (cum, {"cham": {"chrf": {}, "so_mau": 0, "sinh_lap": 0}})]:
        assert T.kiem_c1(a, b)["trang_thai"] == "chua_du_du_lieu", (a, b)
    # sinh_lap thiếu thì không được coi là đạt dù chrF ổn
    p = _viec("pair", 0)
    del p["cham"]["sinh_lap"]
    assert T.kiem_c1(cum, p)["trang_thai"] == "chua_du_du_lieu"
    # có bằng chứng vượt ngưỡng thì vẫn bật dù phần khác thiếu
    p = _viec("pair", 0, 40.0)
    del p["cham"]["sinh_lap"]
    kq = T.kiem_c1(cum, p)
    assert kq["trang_thai"] == "bat" and kq["thieu"]


def test_c2_bien_lech_giua_fold():
    kq = T.kiem_c2(_viec("cum", 0, 50.0), _viec("cum", 1, 51.0), _viec("cum", 2, 52.0))
    assert kq["trang_thai"] == "khong_bat", "max trừ min đúng 2,0 thì không bật"
    kq = T.kiem_c2(_viec("cum", 0, 50.0), _viec("cum", 1, 51.0), _viec("cum", 2, 52.01))
    assert kq["trang_thai"] == "bat" and kq["ma_ly_do"] == ["lech_giua_fold"] and kq["hanh_dong"] == T.HANH_DONG_C2_LECH


def test_c2_f1f2_kem_f0():
    # f0 = 52, f1 = f2 = 50 -> max trừ min = 2,0 (không bật), f0 hơn trung bình f1, f2 đúng 2,0 (không bật)
    assert T.kiem_c2(_viec("cum", 0, 52.0), _viec("cum", 1, 50.0), _viec("cum", 2, 50.0))["trang_thai"] == "khong_bat"
    # f0 = 52, f1 = 50,5, f2 = 49,6: lệch 2,4 (bật lý do 1), trung bình 50,05 kém f0 1,95 (không bật lý do 2)
    kq = T.kiem_c2(_viec("cum", 0, 52.0), _viec("cum", 1, 50.5), _viec("cum", 2, 49.6))
    assert kq["ma_ly_do"] == ["lech_giua_fold"]
    # lý do 2 bật thì f0 trừ min cũng quá ngưỡng, nên luôn đi kèm lý do 1
    kq = T.kiem_c2(_viec("cum", 0, 53.0), _viec("cum", 1, 50.5), _viec("cum", 2, 50.4))
    assert kq["ma_ly_do"] == ["lech_giua_fold", "f1f2_kem_f0"] and kq["trang_thai"] == "bat"
    assert T.HANH_DONG_C2_F0 in kq["hanh_dong"] and T.HANH_DONG_C2_LECH in kq["hanh_dong"]
    assert any("may rủi" in x for x in kq["ly_do"]), "lý do thứ hai phải nêu riêng"


def test_c2_do_lech_chuan_mau():
    kq = T.kiem_c2(_viec("cum", 0, 50.0), _viec("cum", 1, 51.0), _viec("cum", 2, 52.0))
    assert abs(kq["trung_binh"] - 51.0) < 1e-12 and abs(kq["do_lech_chuan"] - 1.0) < 1e-12, "n-1: sqrt((1+0+1)/2) = 1"
    kq = T.kiem_c2(_viec("cum", 0, 10.0), _viec("cum", 1, 12.0), _viec("cum", 2, 14.0))
    assert abs(kq["do_lech_chuan"] - 2.0) < 1e-12


def test_c2_thieu():
    ba = [_viec("cum", i, 50.0) for i in range(3)]
    assert T.kiem_c2(ba[0], None, None)["trang_thai"] == "chua_du_du_lieu"
    assert T.kiem_c2(None, None, None)["trang_thai"] == "chua_du_du_lieu"
    kq = T.kiem_c2(ba[0], ba[1], {"cham": None})
    assert kq["trang_thai"] == "chua_du_du_lieu" and kq["so_fold"] == 2 and kq["do_lech_chuan"] == 0.0
    assert T.kiem_c2(ba[0], ba[1], ba[2])["trang_thai"] == "khong_bat"
    # hai fold đã lệch quá ngưỡng thì bật dù fold còn lại chưa có
    kq = T.kiem_c2(_viec("cum", 0, 50.0), _viec("cum", 1, 53.0), None)
    assert kq["trang_thai"] == "bat" and kq["thieu"]


def test_c3_bien_chrf_macro():
    cum = _viec("cum", 0, 50.0)
    assert T.kiem_c3(cum, _viec("pair", 0, 52.0))["trang_thai"] == "khong_bat"
    assert T.kiem_c3(cum, _viec("pair", 0, 48.0))["trang_thai"] == "khong_bat"
    kq = T.kiem_c3(cum, _viec("pair", 0, 52.01))
    assert kq["trang_thai"] == "bat" and kq["hanh_dong"] == T.HANH_DONG_C3
    assert T.kiem_c3(cum, _viec("pair", 0, 47.99))["trang_thai"] == "bat", "lệch hai phía đều tính (trị tuyệt đối)"


def test_c3_bien_loss():
    cum = _viec("cum", 0, 50.0, loss=1.0)
    assert T.kiem_c3(cum, _viec("pair", 0, 50.0, loss=1.05))["trang_thai"] == "khong_bat", "đúng 5% thì không bật"
    assert T.kiem_c3(cum, _viec("pair", 0, 50.0, loss=0.95))["trang_thai"] == "khong_bat"
    assert T.kiem_c3(cum, _viec("pair", 0, 50.0, loss=1.0501))["trang_thai"] == "bat"
    assert T.kiem_c3(cum, _viec("pair", 0, 50.0, loss=0.9499))["trang_thai"] == "bat"


def test_c3_bien_theo_huong():
    cum = _viec("cum", 0, 50.0)
    # macro chênh 1,0 (không bật) nhưng hướng vi->zh chênh đúng 2,0 (không bật)
    assert T.kiem_c3(cum, _viec("pair", 0, 50.0, zv=50.0, vz=52.0))["trang_thai"] == "khong_bat"
    kq = T.kiem_c3(cum, _viec("pair", 0, 50.0, zv=50.0, vz=52.01))
    assert kq["trang_thai"] == "bat" and any("vi->zh" in x for x in kq["ly_do"])
    kq = T.kiem_c3(cum, _viec("pair", 0, 50.0, zv=47.0, vz=53.0))
    assert len(kq["ly_do"]) == 2, "hai hướng cùng lệch thì báo cả hai"


def test_c3_thieu():
    cum, pair = _viec("cum", 0), _viec("pair", 0)
    assert T.kiem_c3(None, pair)["trang_thai"] == "chua_du_du_lieu"
    assert T.kiem_c3(cum, None)["trang_thai"] == "chua_du_du_lieu"
    p = _viec("pair", 0)
    p["loss_ca_fold"] = None
    assert T.kiem_c3(cum, p)["trang_thai"] == "chua_du_du_lieu", "thiếu loss cả fold thì không tính là đạt"
    p = _viec("pair", 0)
    p["cham"]["theo_huong"]["vi->zh"] = {}
    assert T.kiem_c3(cum, p)["trang_thai"] == "chua_du_du_lieu"
    p = _viec("pair", 0)
    del p["cham"]["theo_huong"]
    assert T.kiem_c3(cum, p)["trang_thai"] == "chua_du_du_lieu"
    assert T.kiem_c3(cum, pair)["trang_thai"] == "khong_bat"


def _cat(**kv):
    return {k: {"loss": l, "so_token": t} for k, (l, t) in kv.items()}


def test_tui_loi_chi_bao_khi_vuot_moi_fold():
    cat = {"zh->vi|title|home": {"loss": 1.30, "so_token": 5000}, "zh->vi|title|toy": {"loss": 1.30, "so_token": 5000},
           "zh->vi|title|pet": {"loss": 1.25, "so_token": 5000}}
    cat2 = {"zh->vi|title|home": {"loss": 1.40, "so_token": 6000}, "zh->vi|title|toy": {"loss": 1.10, "so_token": 6000},
            "zh->vi|title|pet": {"loss": 2.0, "so_token": 6000}}
    ra = T.tui_loi_category([_viec("cum", 0, cat=cat), _viec("cum", 1, cat=cat2)])
    assert [x["khoa"] for x in ra] == ["zh->vi|title|home"], "toy chỉ vượt ở 1 fold; pet đúng 1,25 ở fold 0 thì không vượt"
    assert abs(ra[0]["ti_so_trung_binh"] - 1.35) < 1e-9 and ra[0]["so_fold"] == 2 and ra[0]["so_token_nho_nhat"] == 5000


def test_tui_loi_bo_khoa_it_token_va_sap_xep():
    c1 = {"zh->vi|title|a": {"loss": 1.5, "so_token": 5000}, "zh->vi|title|b": {"loss": 2.0, "so_token": 5000},
          "zh->vi|title|c": {"loss": 3.0, "so_token": 1999}, "vi->zh|desc|d": {"loss": 1.4, "so_token": 2000}}
    c2 = {"zh->vi|title|a": {"loss": 1.6, "so_token": 5000}, "zh->vi|title|b": {"loss": 2.2, "so_token": 5000},
          "zh->vi|title|c": {"loss": 3.0, "so_token": 9000}, "vi->zh|desc|d": {"loss": 1.4, "so_token": 2000}}
    ra = T.tui_loi_category([_viec("cum", 0, cat=c1), _viec("cum", 1, cat=c2)])
    assert [x["khoa"] for x in ra] == ["zh->vi|title|b", "zh->vi|title|a", "vi->zh|desc|d"], "c bị bỏ vì có fold dưới 2000 token; xếp giảm dần"
    # mỗi khóa phải có ở mọi fold có dữ liệu
    c3 = {"zh->vi|title|a": {"loss": 1.6, "so_token": 5000}}
    ra = T.tui_loi_category([_viec("cum", 0, cat=c1), _viec("cum", 1, cat=c3)])
    assert [x["khoa"] for x in ra] == ["zh->vi|title|a"]


def test_tui_loi_thieu_va_tham_so():
    assert T.tui_loi_category([]) == [] and T.tui_loi_category(None) == []
    assert T.tui_loi_category([None, {"loss_ca_fold": None}]) == []
    cat = _cat(**{"x": (1, 1)})
    assert T.tui_loi_category([_viec("pair", 0, cat=cat)]) == [], "khóa sai dạng thì bỏ qua, không lỗi"
    v = _viec("cum", 0, cat={"zh->vi|title|a": {"loss": 1.3, "so_token": 100}})
    assert T.tui_loi_category([v]) == []
    assert len(T.tui_loi_category([v], toi_thieu_token=100)) == 1
    assert T.tui_loi_category([v], he_so=1.3, toi_thieu_token=100) == [], "đúng bằng hệ số thì không báo"
    # nhóm cha thiếu thì không chia được
    w = _viec("cum", 0, cat={"zh->vi|title|a": {"loss": 9.0, "so_token": 9999}})
    w["loss_ca_fold"]["theo_nhom"] = {}
    assert T.tui_loi_category([w]) == []


def test_ket_luan_attn_ba_truong_hop():
    def a(chenh, thap, cao):
        return {"lr1e-4_r16_attn": _cham(50.0), "lr1e-4_r16": _cham(51.0), "so_cap": {"chenh": chenh, "thap": thap, "cao": cao}}
    kem = T.ket_luan_attn(a(-2.0, -3.0, -0.5))
    assert kem["ket_luan"] == "kem", "cận trên âm"
    chua_0 = T.ket_luan_attn(a(-0.5, -1.5, 0.4))
    assert chua_0["ket_luan"] == "khong_kem", "khoảng chứa 0"
    duong = T.ket_luan_attn(a(1.0, 0.2, 1.8))
    assert duong["ket_luan"] == "khong_kem", "cận dưới dương"
    assert T.ket_luan_attn(a(-1.0, -2.0, 0.0))["ket_luan"] == "khong_kem", "cận trên đúng 0 thì không kém"
    for r in (kem, chua_0, duong):
        assert "epoch 1" in r["ghi_chu"] and "1 fold" in r["ghi_chu"] and "epoch 2" in r["ghi_chu"], "luôn kèm ghi chú"
    assert abs(duong["chrf_attn"] - 50.0) < 1e-12 and abs(duong["chrf_moi_lop"] - 51.0) < 1e-12


def test_ket_luan_attn_thieu():
    for attn in ({}, None, {"so_cap": None}, {"so_cap": {"chenh": 1.0}}, {"so_cap": {"chenh": 1.0, "thap": 0.0, "cao": None}},
                 {"so_cap": {"chenh": "x", "thap": 0.0, "cao": 1.0}}):
        r = T.ket_luan_attn(attn)
        assert r["ket_luan"] == "chua_du_du_lieu" and "epoch 1" in r["ghi_chu"], attn
    assert T.ket_luan_attn({"so_cap": {"chenh": 1, "thap": 0, "cao": True}})["ket_luan"] == "chua_du_du_lieu", "bool không phải số"


def test_doc_trang_thai_khong_nem_loi():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        r = T.doc_trang_thai(d / "khong_co.json")
        assert r["loi_doc"] and r["viec"]["cum_f0"] is None and r["xong"] is False
        (d / "hong.json").write_text("{khong phai json", encoding="utf-8")
        assert T.doc_trang_thai(d / "hong.json")["loi_doc"]
        (d / "mang.json").write_text("[1, 2]", encoding="utf-8")
        assert T.doc_trang_thai(d / "mang.json")["loi_doc"]
        (d / "la.json").write_text(json.dumps({"xong": "co", "attn": 5, "viec": {"cum_f0": 3, "pair_f0": None, "cum_f1": {"nhanh": "cum"}}}), encoding="utf-8")
        r = T.doc_trang_thai(d / "la.json")
        assert r["xong"] is False and r["attn"] == {} and r["viec"]["cum_f0"] is None and r["viec"]["cum_f1"] == {"nhanh": "cum"}
        kq = T.tinh_tat_ca(r)
        assert kq["c1"]["trang_thai"] == kq["c2"]["trang_thai"] == kq["c3"]["trang_thai"] == "chua_du_du_lieu"
        assert kq["attn"]["ket_luan"] == "chua_du_du_lieu"


def _trang_thai_day_du():
    cat = {"zh->vi|title|home": {"loss": 1.4, "so_token": 5000}}
    return {"xong": True,
            "attn": {"lr1e-4_r16_attn": _cham(49.0), "lr1e-4_r16": _cham(50.0), "so_cap": {"chenh": -1.0, "thap": -2.0, "cao": 0.1}},
            "viec": {"cum_f0": _viec("cum", 0, 52.0, cat=cat), "pair_f0": _viec("pair", 0, 48.0, 1.2, cat=cat, sinh_lap=30),
                     "cum_f1": _viec("cum", 1, 50.0, cat=cat), "cum_f2": _viec("cum", 2, 49.0, cat=cat)}}


def test_main_day_du():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "tt.json").write_text(json.dumps(_trang_thai_day_du()), encoding="utf-8")
        assert T.main(["--trang-thai", str(d / "tt.json"), "--ra", str(d / "ra" / "con")]) == 0
        md = (d / "ra" / "con" / "bao_cao_b78.md").read_text(encoding="utf-8")
        js = json.loads((d / "ra" / "con" / "bao_cao_b78.json").read_text(encoding="utf-8"))
        assert js["c1"]["trang_thai"] == "bat" and js["c3"]["trang_thai"] == "bat" and js["c2"]["trang_thai"] == "bat"
        assert js["attn"]["ket_luan"] == "khong_kem" and js["tui_loi"]["cum"][0]["khoa"] == "zh->vi|title|home"
        for can in ("C1", "C2", "C3", "CHỜ số zero-shot", "[MẶC ĐỊNH]", "cum_f2", "pair_f0", "epoch 1", T.HANH_DONG_C1, T.HANH_DONG_C3, "zh->vi / title"):
            assert can in md, can
        assert "—" not in md and "–" not in md, "không dùng gạch nối dài"
        assert md.count("|---|") >= 3
        phan1 = md.split("## 2.")[0].splitlines()
        so_cot = [x.count("|") for x in phan1 if x.startswith("|")]
        assert len(set(so_cot)) == 1, "bảng việc phải đều cột"


def test_main_gan_nhu_rong():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "rong.json").write_text("{}", encoding="utf-8")
        (d / "gan.json").write_text(json.dumps({"xong": None, "viec": {"cum_f0": {"nhanh": "cum", "fold": 0, "diem": {"1": 1.0}, "cham": None,
                                                                              "loss_ca_fold": None}}}), encoding="utf-8")
        for ten in ("rong", "gan", "khong_co"):
            ra = d / f"ra_{ten}"
            assert T.main(["--trang-thai", str(d / f"{ten}.json"), "--ra", str(ra)]) == 0
            md = (ra / "bao_cao_b78.md").read_text(encoding="utf-8")
            js = json.loads((ra / "bao_cao_b78.json").read_text(encoding="utf-8"))
            assert js["c1"]["trang_thai"] == js["c2"]["trang_thai"] == js["c3"]["trang_thai"] == "chua_du_du_lieu"
            assert js["attn"]["ket_luan"] == "chua_du_du_lieu" and "khong_bat" not in md.split("## 5.")[0].split("## 4.")[1]
            assert "CHỜ số zero-shot" in md and "chua_du_du_lieu" in md


def test_bao_cao_khong_ket_luan_tot():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "tt.json").write_text(json.dumps(_trang_thai_day_du()), encoding="utf-8")
        T.main(["--trang-thai", str(d / "tt.json"), "--ra", str(d)])
        md = (d / "bao_cao_b78.md").read_text(encoding="utf-8").lower()
        for cam in (" tốt", "mô hình ổn", "đạt yêu cầu"):
            assert cam not in md, cam


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_"):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
