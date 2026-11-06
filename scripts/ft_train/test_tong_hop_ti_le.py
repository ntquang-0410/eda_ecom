"""Thử tong_hop_ti_le bằng dữ liệu tự dựng nhỏ (không đọc dữ liệu thật). Chạy: python -I -X utf8 scripts/ft_train/test_tong_hop_ti_le.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tong_hop_ti_le as T  # noqa: E402


# ---------------------------------------------------------------- hàm thuần
def _s(title, mo_ta, zh_vi):
    return {"title": title, "mo_ta": mo_ta, "zh_vi": zh_vi}


def test_khong_hoc_lech_bien():
    moc = _s(70.0, 80.0, 75.0)
    assert T.NGUONG_LECH == 1.0
    r = T.khong_hoc_lech(moc, _s(69.0, 79.0, 74.0))
    assert r["dat"] is True, "tụt đúng 1,0 ở cả ba thì đạt"
    assert all(abs(r["chenh"][k] + 1.0) < 1e-12 for k in T.KHOA_LECH), "chenh = muc trừ moc"
    for k, ds in (("title", 69.0), ("mo_ta", 79.0), ("zh_vi", 74.0)):
        muc = _s(70.0, 80.0, 75.0)
        muc[k] = ds - 0.01
        r = T.khong_hoc_lech(moc, muc)
        assert r["dat"] is False and len(r["ly_do"]) == 1, (k, r)
    assert T.khong_hoc_lech(moc, _s(90.0, 90.0, 90.0))["dat"] is True, "hơn mốc thì đạt"
    # số thật, trừ 1,0 có sai số làm tròn: vẫn đạt
    x = 75.76131971214022
    assert T.khong_hoc_lech(_s(x, x, x), _s(x - 1.0, x - 1.0, x - 1.0))["dat"] is True
    assert T.khong_hoc_lech(_s(x, x, x), _s(x - 1.01, x, x))["dat"] is False


def test_khong_hoc_lech_thieu():
    moc = _s(70.0, 80.0, 75.0)
    for muc in (_s(None, 80.0, 75.0), _s(70.0, None, 75.0), _s(70.0, 80.0, None), {}, None, _s("x", 80.0, 75.0), _s(True, 80.0, 75.0)):
        r = T.khong_hoc_lech(moc, muc)
        assert r["dat"] is None and any("thiếu" in x for x in r["ly_do"]), muc
    assert T.khong_hoc_lech(None, _s(70.0, 80.0, 75.0))["dat"] is None
    assert T.khong_hoc_lech(None, None)["dat"] is None
    # đã có bằng chứng tụt quá ngưỡng thì không đạt dù còn thiếu số khác (không bao giờ thành None hoặc đạt)
    r = T.khong_hoc_lech(moc, _s(60.0, None, 75.0))
    assert r["dat"] is False and r["chenh"]["mo_ta"] is None


def test_qua_cong_bien():
    tot = {"dat": True, "ly_do": ["ok"], "chenh": {}}
    r = T.qua_cong_fold0({"thap": 0.01}, tot)
    assert r["trang_thai"] == "qua" and r["ly_do"]
    for thap in (0.0, -0.01, -3.0):
        r = T.qua_cong_fold0({"thap": thap}, tot)
        assert r["trang_thai"] == "khong_qua" and r["ly_do"], thap
    lech = T.khong_hoc_lech(_s(70.0, 80.0, 75.0), _s(60.0, 80.0, 75.0))
    r = T.qua_cong_fold0({"thap": 1.0}, lech)
    assert r["trang_thai"] == "khong_qua" and any("title" in x for x in r["ly_do"])
    r = T.qua_cong_fold0({"thap": -1.0}, lech)
    assert r["trang_thai"] == "khong_qua" and len(r["ly_do"]) >= 2, "báo cả hai lý do"


def test_qua_cong_thieu():
    tot = {"dat": True, "ly_do": [], "chenh": {}}
    cho = {"dat": None, "ly_do": [], "chenh": {}}
    for so_cap, lech in ((None, tot), ({}, tot), ({"thap": None}, tot), ({"thap": "x"}, tot), ({"thap": 1.0}, cho), ({"thap": 1.0}, None),
                         ({"thap": 1.0}, {}), (None, None), ({"thap": 1.0}, {"dat": "co"})):
        r = T.qua_cong_fold0(so_cap, lech)
        assert r["trang_thai"] == "chua_du_du_lieu" and r["ly_do"], (so_cap, lech)
    # thiếu một phần nhưng đã có bằng chứng không qua thì không qua
    assert T.qua_cong_fold0({"thap": -1.0}, cho)["trang_thai"] == "khong_qua"
    assert T.qua_cong_fold0(None, {"dat": False, "ly_do": ["x"], "chenh": {}})["trang_thai"] == "khong_qua"


def _m(ten, tt, chenh=None):
    return {"ten": ten, "trang_thai": tt, "chenh": chenh}


def test_chon_ung_vien():
    r = T.chon_ung_vien([_m("p11", "qua", 0.5), _m("p75", "qua", 1.2)])
    assert r["ung_vien"] == "p75" and r["trang_thai"] == "chon"
    r = T.chon_ung_vien([_m("p11", "qua", 1.2), _m("p75", "qua", 0.5)])
    assert r["ung_vien"] == "p11"
    r = T.chon_ung_vien([_m("p11", "khong_qua", 9.0), _m("p75", "qua", 0.1)])
    assert r["ung_vien"] == "p75", "mức không qua không được chọn dù chênh lớn"
    r = T.chon_ung_vien([_m("p11", "khong_qua", 9.0), _m("p75", "khong_qua", 1.0)])
    assert r["ung_vien"] is None and r["trang_thai"] == "giu_muc_hien_tai"
    r = T.chon_ung_vien([])
    assert r["ung_vien"] is None and r["trang_thai"] == "giu_muc_hien_tai"
    assert T.chon_ung_vien(None)["ung_vien"] is None
    # bằng nhau thì theo tên, ghi chú
    r = T.chon_ung_vien([_m("b", "qua", 1.0), _m("a", "qua", 1.0)])
    assert r["ung_vien"] == "a" and len(r["ly_do"]) == 2
    # nhận cả dict
    assert T.chon_ung_vien({"p11": {"trang_thai": "qua", "chenh": 2.0}, "p75": {"trang_thai": "qua", "chenh": 1.0}})["ung_vien"] == "p11"


def test_chon_ung_vien_chua_du():
    r = T.chon_ung_vien([_m("p11", "qua", 2.0), _m("p75", "chua_du_du_lieu")])
    assert r["ung_vien"] is None and r["trang_thai"] == "chua_ket_luan" and "p75" in r["ly_do"][0] and "chưa kết luận" in r["ly_do"][0]
    r = T.chon_ung_vien([_m("p11", "khong_qua"), _m("p75", "chua_du_du_lieu")])
    assert r["ung_vien"] is None and r["trang_thai"] == "chua_ket_luan"
    r = T.chon_ung_vien([_m("p11", "qua", None)])
    assert r["ung_vien"] is None and r["trang_thai"] == "chua_ket_luan", "qua mà thiếu chênh thì không so được"


def test_ket_luan_kfold():
    tot = {"dat": True, "ly_do": [], "chenh": {}}
    r = T.ket_luan_kfold([2.0, 2.5, 3.0], tot)
    assert r["doi_ti_le"] is True and r["ly_do"] and set(r) == {"doi_ti_le", "ly_do"}
    # trung bình đúng bằng độ lệch chuẩn thì không đổi: [1, 3, 5] tb 3 sd 2 -> đổi; [0.5, 0.5, 0.5] sd 0 -> đổi
    assert T.ket_luan_kfold([0.5, 0.5, 0.5], tot)["doi_ti_le"] is True
    assert T.ket_luan_kfold([1.0, 3.0, 5.0], tot)["doi_ti_le"] is True
    # tb 1, sd 1: [0.0001...]: [0.5, 1.0, 1.5] tb 1 sd 0,5 -> đổi ; [0.1, 0.2, 1.2] tb 0,5 sd 0,6 -> không
    r = T.ket_luan_kfold([0.1, 0.2, 1.2], tot)
    assert r["doi_ti_le"] is False and any("độ lệch chuẩn" in x for x in r["ly_do"])
    # n-1: [1, 2, 3] sd = 1, tb = 2 > 1; nếu dùng n sẽ là 0,816; kiểm biên đúng bằng: [0.0, ...] không dùng được vì phải dương, nên dựng tb = sd
    # [a, a, b]: dựng tb = sd: chọn [1, 1, x]: tb=(2+x)/3, sd=|x-1|/sqrt(3); x = 4 -> tb 2, sd 1,732
    r = T.ket_luan_kfold([0.01, 0.01, 4.0], tot)
    assert r["doi_ti_le"] is False
    # có fold không dương
    for xs in ([1.0, 2.0, 0.0], [1.0, 2.0, -0.5], [-1.0, -1.0, -1.0]):
        r = T.ket_luan_kfold(xs, tot)
        assert r["doi_ti_le"] is False and any("dương" in x for x in r["ly_do"]), xs
    # học lệch
    assert T.ket_luan_kfold([2.0, 2.5, 3.0], {"dat": False, "ly_do": [], "chenh": {}})["doi_ti_le"] is False
    # chưa đủ 3 fold, thiếu số, lệch chưa rõ
    for xs in ([], [1.0], [1.0, 2.0], [1.0, 2.0, None], [1.0, 2.0, "x"], None, [1.0, 2.0, 3.0, 4.0]):
        assert T.ket_luan_kfold(xs, tot)["doi_ti_le"] is None, xs
    assert T.ket_luan_kfold([2.0, 2.5, 3.0], {"dat": None})["doi_ti_le"] is None
    assert T.ket_luan_kfold([2.0, 2.5, 3.0], None)["doi_ti_le"] is None
    # chênh đã không đạt thì không cần biết lệch
    assert T.ket_luan_kfold([-1.0, 2.5, 3.0], {"dat": None})["doi_ti_le"] is False


def test_ket_luan_kfold_do_lech_n_tru_1():
    # [1, 1, b]: tìm b có trung bình = độ lệch chuẩn mẫu (n-1) bằng chia đôi, kiểm hai bên biên
    from statistics import fmean, stdev
    lo, hi = 1.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if fmean([1.0, 1.0, mid]) > stdev([1.0, 1.0, mid]):   # b nhỏ thì sd nhỏ nên đổi; b lớn thì không
            lo = mid
        else:
            hi = mid
    tot = {"dat": True}
    assert T.ket_luan_kfold([1.0, 1.0, lo - 1e-6], tot)["doi_ti_le"] is True
    assert T.ket_luan_kfold([1.0, 1.0, hi + 1e-6], tot)["doi_ti_le"] is False


def test_gom_so():
    cham = {"so_mau": 1000, "sinh_lap": 3, "chrf": {"chrf_macro_huong": 70.0},
            "theo_huong": {"zh->vi": {"chrf_corpus": 71.0}, "vi->zh": {"chrf_corpus": 69.0}},
            "theo_nhom": {"zh->vi/title": {"chrf_corpus": 60.0}, "vi->zh/title": {"chrf_corpus": 64.0},
                          "zh->vi/mo_ta": {"chrf_corpus": 80.0}, "vi->zh/mo_ta": {"chrf_corpus": 90.0}},
            "dong_da_thay_moi": {"zh->vi": {"da_thay": {"n_dong": 5, "chrf_corpus": 99.0}, "moi": {"n_dong": 7, "chrf_corpus": 75.5}},
                                 "vi->zh": {"moi": {"n_dong": 0, "chrf_corpus": None}}}}
    s = T.gom_so(cham)
    assert s["title"] == 62.0 and s["mo_ta"] == 85.0 and s["zh_vi"] == 71.0 and s["vi_zh"] == 69.0 and s["chrf"] == 70.0
    assert s["moi_zh_vi"] == {"n": 7.0, "chrf": 75.5} and s["moi_vi_zh"]["chrf"] is None and s["sinh_lap"] == 3.0 and s["so_mau"] == 1000.0
    del cham["theo_nhom"]["vi->zh/title"]
    assert T.gom_so(cham)["title"] is None, "thiếu một hướng thì không tính title"
    assert T.gom_so(None) is None and T.gom_so({})["title"] is None and T.gom_so({})["moi_zh_vi"]["n"] is None


# ---------------------------------------------------------------- dựng dữ liệu giả cho main
ZH = ["夏季纯棉短袖T恤男宽松圆领上衣", "冬季加厚保暖羽绒服女中长款外套", "儿童卡通图案益智拼装积木玩具", "不锈钢保温杯大容量便携车载水杯"]
VI = ["Áo thun cotton ngắn tay nam mùa hè cổ tròn rộng", "Áo khoác lông vũ nữ dày giữ ấm mùa đông dáng dài",
      "Đồ chơi xếp hình giáo dục trẻ em họa tiết hoạt hình", "Cốc giữ nhiệt inox dung tích lớn mang theo ô tô tiện lợi"]
ZH_MO = ["品牌: 美的; 材质: 纯棉面料; 产地: 中国广东省佛山市", "品牌: 小米; 材质: 优质涤纶面料; 产地: 中国浙江省杭州", "品牌: 华为; 材质: 食品级硅胶; 产地: 中国江苏省苏州市", "品牌: 苏泊尔; 材质: 不锈钢; 产地: 中国广东省"]
VI_MO = ["Thương hiệu: Midea; Chất liệu: vải cotton nguyên chất; Xuất xứ: Phật Sơn Quảng Đông", "Thương hiệu: Xiaomi; Chất liệu: vải polyester cao cấp; Xuất xứ: Hàng Châu Chiết Giang",
         "Thương hiệu: Huawei; Chất liệu: silicone cấp thực phẩm; Xuất xứ: Tô Châu Giang Tô", "Thương hiệu: Supor; Chất liệu: thép không gỉ; Xuất xứ: Quảng Đông Trung Quốc"]
NHOM = [("zh->vi", "title"), ("zh->vi", "mo_ta"), ("vi->zh", "title"), ("vi->zh", "mo_ta")]


def _hang(nhanh, huong, loai, i, fold, ten):
    ch, vi = (ZH, VI) if loai == "title" else (ZH_MO, VI_MO)
    nguon, dich = (ch[i % 4], vi[i % 4]) if huong == "zh->vi" else (vi[i % 4], ch[i % 4])
    if loai == "mo_ta" and nhanh == "pair":
        nguon, dich = nguon.replace("; ", "\n"), dich.replace("; ", "\n")
    return {"id": f"{ten}", "huong": huong, "loai": loai, "nhanh": nhanh, "fold5": fold, "phan": 0, "category": "home", "nguon": nguon, "dich": dich}


def _pool(nhanh):
    ra = []
    for gi, (h, l) in enumerate(NHOM):
        for k in range(3):
            ra.append(_hang(nhanh, h, l, k, 0, f"f0_{gi}_{k}"))
        for k in range(2):
            ra.append(_hang(nhanh, h, l, k + 1, 1, f"f1_{gi}_{k}"))
    return ra


def _hyp(dich, loai, kieu):
    if kieu == "p11":
        return dich
    if kieu == "p75":
        return dich[:-2]
    if kieu == "moc":
        return dich[:-6]
    if kieu == "rac_title":
        return "xyz" if loai == "title" else dich[:-2]
    raise ValueError(kieu)


def _viet_jsonl(duong_dan, nhanh, kieu):
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    dong = [json.dumps({"id": m["id"], "hyp_tho": _hyp(m["dich"], m["loai"], kieu), "hyp": _hyp(m["dich"], m["loai"], kieu)}, ensure_ascii=False)
            for m in _pool(nhanh) if m["fold5"] == 0]
    duong_dan.write_text("\n".join(dong) + "\n", encoding="utf-8")


def _loss(v):
    return {"zh->vi|title": v, "zh->vi|mo_ta": v / 10, "vi->zh|title": v, "vi->zh|mo_ta": v / 10}


def _viec(nhanh, p, w, **kw):
    v = {"nhanh": nhanh, "fold": 0, "p_muc_tieu": p, "p_thuc": p + 0.0001, "w_title": w, "lr": 2e-4, "epoch_xong": 2,
         "diem": {"1": 0.4, "2": 0.35}, "loss": {"1": _loss(0.7), "2": _loss(0.6)}}
    v.update(kw)
    return v


def _dung(d, kieu_p11=("p11", "p11"), kieu_p75=("rac_title", "p75"), **tt_them):
    """Dựng thư mục giả. kieu_p11, kieu_p75: kiểu bản dịch (cum, pair). Trả (đường dẫn trạng thái, dd)."""
    d = Path(d)
    for n in ("cum", "pair"):
        (d / f"pool_{n}.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in _pool(n)) + "\n", encoding="utf-8")
        _viet_jsonl(d / f"moc_{n}.jsonl", n, "moc")
    (d / "moc_tt.json").write_text(json.dumps({"viec": {"cum_f0": {"loss": {"2": _loss(0.65)}, "diem": {"2": 0.36}},
                                                      "pair_f0": {"loss": {"2": _loss(0.66)}, "diem": {"2": 0.37}}}}), encoding="utf-8")
    tl = d / "tl"
    tl.mkdir()
    viec = {}
    for i, n in enumerate(("cum", "pair")):
        for ma, p, w, kieu in (("p11", 0.1074, 0.12, kieu_p11[i]), ("p75", 0.75, 0.8, kieu_p75[i])):
            ten = f"{n}_f0_{ma}"
            _viet_jsonl(tl / ten / "sinh_ep2.jsonl", n, kieu)
            viec[ten] = _viec(n, p, w)
    tt = {"xong": True, "viec": viec}
    tt.update(tt_them)
    (tl / "tl_trang_thai.json").write_text(json.dumps(tt), encoding="utf-8")
    dd = {"moc_cum": d / "moc_cum.jsonl", "moc_pair": d / "moc_pair.jsonl", "moc_trang_thai": d / "moc_tt.json",
          "pool_cum": d / "pool_cum.jsonl", "pool_pair": d / "pool_pair.jsonl"}
    return tl / "tl_trang_thai.json", dd


def _chay(d, tt_path, dd, ky_vong="tu_cham", ten_ra="ra"):
    """Chạy main, trả (md, js). ky_vong 'tu_cham': lấy số mốc tự chấm làm kỳ vọng (để khớp); None: dùng số thật (sẽ không khớp)."""
    if ky_vong == "tu_cham":
        ky_vong = {n: r["chrf_tu_cham"] for n, r in T.kiem_moc(dd, 2, 8).items()}
    ra = Path(d) / ten_ra
    assert T.main(["--trang-thai", str(tt_path), "--ra", str(ra)], duong_dan=dd, so_moi_nhom=2, so_mau=8, moc_ky_vong=ky_vong) == 0
    return (ra / "bao_cao_ti_le.md").read_text(encoding="utf-8"), json.loads((ra / "bao_cao_ti_le.json").read_text(encoding="utf-8"))


def _cong(js, nhanh, ma):
    return js["nhanh"][nhanh]["muc"][ma]["cong"]["trang_thai"]


def test_kiem_moc_du_lieu_gia():
    with tempfile.TemporaryDirectory() as d:
        _, dd = _dung(d)
        r = T.kiem_moc(dd, 2, 8)
        for n in ("cum", "pair"):
            assert r[n]["khop"] is False and r[n]["chrf_tu_cham"] is not None and not r[n]["loi"], "số thật không khớp dữ liệu giả"
        ky = {n: r[n]["chrf_tu_cham"] for n in r}
        assert all(v["khop"] for v in T.kiem_moc(dd, 2, 8, ky).values())
        # lệch dưới 1e-6 vẫn khớp, trên thì không
        ky2 = {n: ky[n] + 5e-7 for n in ky}
        assert all(v["khop"] for v in T.kiem_moc(dd, 2, 8, ky2).values())
        ky3 = {n: ky[n] + 5e-6 for n in ky}
        assert not any(v["khop"] for v in T.kiem_moc(dd, 2, 8, ky3).values())
        # so_mau không đúng thì báo lỗi, không tự chọn lại
        r = T.kiem_moc(dd, 2, 9, ky)
        assert r["cum"]["khop"] is None and r["cum"]["loi"]
        assert T.main(["--chi-kiem-moc"], duong_dan=dd, so_moi_nhom=2, so_mau=8, moc_ky_vong=ky) == 0
        assert T.main(["--chi-kiem-moc"], duong_dan=dd, so_moi_nhom=2, so_mau=8) == 1


def test_main_day_du():
    with tempfile.TemporaryDirectory() as d:
        tt_path, dd = _dung(d)
        md, js = _chay(d, tt_path, dd)
        # cum: ~11% bản dịch đúng hết nên qua; 75% title rác nên học lệch
        assert _cong(js, "cum", "p11") == "qua" and _cong(js, "cum", "p75") == "khong_qua"
        assert js["nhanh"]["cum"]["ung_vien"]["ung_vien"] == "~11%" and js["nhanh"]["cum"]["ung_vien"]["trang_thai"] == "chon"
        assert js["nhanh"]["cum"]["muc"]["p75"]["lech"]["dat"] is False
        # pair: cả hai qua, ~11% chênh lớn hơn nên được chọn
        assert _cong(js, "pair", "p11") == "qua" and _cong(js, "pair", "p75") == "qua"
        assert js["nhanh"]["pair"]["ung_vien"]["ung_vien"] == "~11%"
        assert all(js["nhanh"][n]["moc"]["khop"] for n in ("cum", "pair"))
        so_cap = js["nhanh"]["cum"]["muc"]["p11"]["so_cap"]
        assert so_cap["thap"] > 0 and so_cap["thap"] <= so_cap["chenh"] <= so_cap["cao"]
        assert js["nhanh"]["cum"]["muc"]["p11"]["w_title"] == 0.12 and abs(js["nhanh"]["cum"]["muc"]["p11"]["p_thuc"] - 0.1075) < 1e-12
        assert js["nhanh"]["cum"]["muc"]["p11"]["nguon_so"] == "chấm lại trên máy"
        for can in ("[MẶC ĐỊNH]", "chứa 0", "không phải đánh giá của người", "Bản cum", "Bản pair", "mốc ~50%", "~11%", "75%", "% title thực tế",
                    "Mô tả dòng mới", "Sinh lặp", "Ứng viên", "khong_qua", "w_title", "zh→vi title"):
            assert can in md, can
        assert "—" not in md and "–" not in md, "không dùng gạch nối dài"
        low = md.lower()
        for cam in (" tốt", "mô hình ổn", "đạt yêu cầu"):
            assert cam not in low, cam
        # các bảng đều cột
        khoi = []
        for dong in md.splitlines() + [""]:
            if dong.startswith("|"):
                khoi.append(dong.count("|"))
            elif khoi:
                assert len(set(khoi)) == 1, khoi
                khoi = []


def test_main_hai_muc_qua_chon_lon_hon():
    with tempfile.TemporaryDirectory() as d:
        tt_path, dd = _dung(d, kieu_p75=("p75", "p75"))
        md, js = _chay(d, tt_path, dd)
        for n in ("cum", "pair"):
            assert _cong(js, n, "p11") == "qua" and _cong(js, n, "p75") == "qua"
            assert js["nhanh"][n]["ung_vien"]["ung_vien"] == "~11%", "p11 bản dịch đúng hết nên chênh lớn hơn"
        # đảo: bản dịch 75% sát hơn thì chọn 75%
        (Path(d) / "x").mkdir()
        tt_path, dd = _dung(Path(d) / "x", kieu_p11=("p75", "p75"), kieu_p75=("p11", "p11"))
        md, js = _chay(Path(d) / "x", tt_path, dd)
        assert js["nhanh"]["cum"]["ung_vien"]["ung_vien"] == "75%"


def test_main_hong_bat_thuong_thieu_viec():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a").mkdir()
        tt_path, dd = _dung(d / "a")
        tt = json.loads(tt_path.read_text(encoding="utf-8"))
        tt["viec"]["pair_f0_p11"]["hong"] = True
        del tt["viec"]["cum_f0_p75"]
        tt["viec"]["pair_f0_p75"]["epoch_xong"] = 1
        tt_path.write_text(json.dumps(tt), encoding="utf-8")
        md, js = _chay(d / "a", tt_path, dd)
        assert _cong(js, "pair", "p11") == "chua_du_du_lieu" and _cong(js, "cum", "p75") == "chua_du_du_lieu"
        assert _cong(js, "pair", "p75") == "chua_du_du_lieu"
        assert _cong(js, "cum", "p11") == "qua", "việc còn lại vẫn kết luận được"
        # cum còn một mức chưa đủ nên chưa chọn ứng viên bừa
        assert js["nhanh"]["cum"]["ung_vien"]["ung_vien"] is None and js["nhanh"]["cum"]["ung_vien"]["trang_thai"] == "chua_ket_luan"
        assert js["nhanh"]["pair"]["ung_vien"]["trang_thai"] == "chua_ket_luan"
        assert "hỏng" in md and "không có trong trạng thái" in md and "mới xong epoch 1" in md
        # bất thường thì không kết luận cổng nào
        (d / "b").mkdir()
        tt_path, dd = _dung(d / "b", bat_thuong="loss tăng ở bước 120")
        md, js = _chay(d / "b", tt_path, dd)
        assert js["bat_thuong"] == "loss tăng ở bước 120" and "loss tăng ở bước 120" in md
        for n in ("cum", "pair"):
            for ma in ("p11", "p75"):
                assert _cong(js, n, ma) == "chua_du_du_lieu"
            assert js["nhanh"][n]["ung_vien"]["ung_vien"] is None
        assert js["nhanh"]["cum"]["muc"]["p11"]["so"] is not None, "vẫn đọc số, chỉ không kết luận"


def test_main_thieu_file_sinh_hoac_hong():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tt_path, dd = _dung(d)
        (tt_path.parent / "cum_f0_p75" / "sinh_ep2.jsonl").unlink()
        (tt_path.parent / "pair_f0_p11" / "sinh_ep2.jsonl").write_text("{khong phai json\n", encoding="utf-8")
        # pair_f0_p75: thiếu bản dịch của một mẫu
        p = tt_path.parent / "pair_f0_p75" / "sinh_ep2.jsonl"
        p.write_text("\n".join(p.read_text(encoding="utf-8").splitlines()[:5]) + "\n", encoding="utf-8")
        md, js = _chay(d, tt_path, dd)
        assert _cong(js, "cum", "p75") == "chua_du_du_lieu" and _cong(js, "pair", "p11") == "chua_du_du_lieu" and _cong(js, "pair", "p75") == "chua_du_du_lieu"
        assert _cong(js, "cum", "p11") == "qua"
        assert js["nhanh"]["pair"]["muc"]["p75"]["so"] is None
        assert "thiếu" in md and "chưa có số" in md


def test_main_moc_khong_khop_khong_ket_luan():
    with tempfile.TemporaryDirectory() as d:
        tt_path, dd = _dung(d)
        md, js = _chay(d, tt_path, dd, ky_vong=None)   # số mốc thật không khớp dữ liệu giả
        assert "KHÔNG KHỚP" in md and "CẢNH BÁO LỚN" in md
        for n in ("cum", "pair"):
            assert js["nhanh"][n]["moc"]["khop"] is False
            for ma in ("p11", "p75"):
                assert _cong(js, n, ma) == "chua_du_du_lieu", "mốc lệch thì không kết luận cổng"
            assert js["nhanh"][n]["muc"]["p11"]["so"] is not None


def test_main_canh_bao_lech_so_kaggle():
    with tempfile.TemporaryDirectory() as d:
        tt_path, dd = _dung(d)
        tt = json.loads(tt_path.read_text(encoding="utf-8"))
        tt["viec"]["cum_f0_p11"]["cham"] = {"chrf": {"chrf_macro_huong": 12.5}}
        tt_path.write_text(json.dumps(tt), encoding="utf-8")
        md, js = _chay(d, tt_path, dd)
        assert js["nhanh"]["cum"]["muc"]["p11"]["canh_bao"] and "khác số Kaggle ghi" in md
        assert _cong(js, "cum", "p11") == "qua", "chỉ cảnh báo, không chặn"


def test_main_thieu_pool_hoac_cong_cu():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tt_path, dd = _dung(d)
        dd2 = {**dd, "pool_pair": d / "khong_co.jsonl"}
        md, js = _chay(d, tt_path, dd2, ky_vong=None)
        assert _cong(js, "pair", "p11") == "chua_du_du_lieu" and "không đọc được pool" in md
        dd3 = {**dd, "pool_cum": d / "khong_co.jsonl", "moc_trang_thai": d / "khong_co.json"}
        md, js = _chay(d, tt_path, dd3, ky_vong=None, ten_ra="ra3")
        assert all(_cong(js, n, ma) == "chua_du_du_lieu" for n in ("cum", "pair") for ma in ("p11", "p75"))
        assert js["loi_chung"] and "CẢNH BÁO" in md


def test_doc_trang_thai_khong_nem_loi():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        r = T.doc_trang_thai(d / "khong_co.json")
        assert r["loi_doc"] and r["xong"] is False and all(v is None for v in r["viec"].values()) and len(r["viec"]) == 4
        (d / "hong.json").write_text("{khong phai json", encoding="utf-8")
        assert T.doc_trang_thai(d / "hong.json")["loi_doc"]
        (d / "mang.json").write_text("[1, 2]", encoding="utf-8")
        assert T.doc_trang_thai(d / "mang.json")["loi_doc"]
        (d / "la.json").write_text(json.dumps({"xong": "co", "bat_thuong": {"a": 1}, "viec": {"cum_f0_p11": 3, "pair_f0_p11": {"nhanh": "pair"}}}), encoding="utf-8")
        r = T.doc_trang_thai(d / "la.json")
        assert r["xong"] is False and r["viec"]["cum_f0_p11"] is None and r["viec"]["pair_f0_p11"] == {"nhanh": "pair"}
        assert r["bat_thuong"] == '{"a": 1}'
        (d / "bt.json").write_text(json.dumps({"bat_thuong": "loi abc", "viec": 5}), encoding="utf-8")
        r = T.doc_trang_thai(d / "bt.json")
        assert r["bat_thuong"] == "loi abc" and r["viec"]["cum_f0_p75"] is None and r["loi_doc"] is None


def test_main_gan_nhu_rong_va_hong():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _, dd = _dung(d)
        (d / "rong.json").write_text("{}", encoding="utf-8")
        (d / "gan.json").write_text(json.dumps({"xong": None, "viec": {"cum_f0_p11": {"nhanh": "cum", "fold": 0, "diem": {"1": 1.0}, "cham": None,
                                                                                    "loss": None, "epoch_xong": None}}}), encoding="utf-8")
        (d / "hong.json").write_text("{khong phai json", encoding="utf-8")
        (d / "mang.json").write_text("[]", encoding="utf-8")
        for ten in ("rong", "gan", "hong", "mang", "khong_co"):
            md, js = _chay(d, d / f"{ten}.json", dd, ten_ra=f"ra_{ten}")
            for n in ("cum", "pair"):
                for ma in ("p11", "p75"):
                    assert _cong(js, n, ma) == "chua_du_du_lieu", (ten, n, ma)
                assert js["nhanh"][n]["ung_vien"]["ung_vien"] is None and js["nhanh"][n]["ung_vien"]["trang_thai"] == "chua_ket_luan"
            assert "**~11%: chua_du_du_lieu**" in md and "**Chưa kết luận được ứng viên**" in md
            assert "**~11%: qua**" not in md, "thiếu dữ liệu không được báo qua"
            if ten in ("hong", "mang", "khong_co"):
                assert js["loi_doc"] and "CẢNH BÁO" in md
        # trạng thái mốc hỏng và pool hỏng cũng không ném lỗi
        (d / "pool_hong.jsonl").write_text("{x\n", encoding="utf-8")
        (d / "moc_tt_hong.json").write_text("[1]", encoding="utf-8")
        dd2 = {**dd, "pool_cum": d / "pool_hong.jsonl", "moc_trang_thai": d / "moc_tt_hong.json"}
        md, js = _chay(d, d / "rong.json", dd2, ky_vong=None, ten_ra="ra_hong2")
        assert js["loi_chung"]


def test_main_doi_so_dong_lenh():
    with tempfile.TemporaryDirectory() as d:
        try:
            T.main(["--ra", str(Path(d) / "x")])
        except SystemExit as e:
            assert e.code != 0
        else:
            raise AssertionError("thiếu --trang-thai phải báo lỗi dòng lệnh")


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_"):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
