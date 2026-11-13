"""Thử bộ chấm trên dữ liệu tự dựng (cả hai hướng). Chạy: python -I -X utf8 scripts/ft_train/test_cham_diem.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from sacrebleu.metrics import CHRF  # noqa: E402

import cham_diem as C  # noqa: E402

# (huong, loai, nhanh, nguon, dich)
ZH_TITLE = ("zh->vi", "title", "cum", "Elmich 不锈钢保温杯 500ml 黑色 2024款", "Cốc giữ nhiệt inox Elmich 500ml màu đen mẫu 2024")
ZH_MOTA = ("zh->vi", "mo_ta", "cum", "品牌: Elmich; 容量: 500克; 颜色: 黑色",
           "Thương hiệu: Elmich; Dung lượng: 500g; Màu sắc: Đen")
VI_TITLE = ("vi->zh", "title", "pair", "Cốc giữ nhiệt Elmich 500ml màu đen", "Elmich保温杯 500ml 黑色")
VI_MOTA = ("vi->zh", "mo_ta", "pair", "Thương hiệu: Elmich\nDung lượng: 500ml\nMàu sắc: Đen",
           "品牌: Elmich\n容量: 500ml\n颜色: 黑色")
TAT_CA = [ZH_TITLE, ZH_MOTA, VI_TITLE, VI_MOTA]


def tc(mau, category="home"):
    return pd.DataFrame([{"id": f"m{i}", "huong": h, "loai": l, "nhanh": nh, "category": category, "nguon": n, "dich": d}
                         for i, (h, l, nh, n, d) in enumerate(mau)])


def dd(df_tc, hyps=None):
    """hyps: dict id -> hyp; mặc định hyp == dich."""
    hyps = hyps or {}
    return pd.DataFrame([{"id": r.id, "hyp": hyps.get(r.id, r.dich)} for r in df_tc.itertuples() if r.id not in hyps or hyps[r.id] is not None])


def mot(m, hyp):
    """Chấm một mục với hyp cho trước, trả dòng kết quả."""
    t = tc([m])
    df, _ = C.cham(t, dd(t, {"m0": hyp}))
    return df.iloc[0]


def test_hyp_bang_ref():
    t = tc(TAT_CA)
    df, tm = C.cham(t, dd(t))
    assert (df.chrf == 100).all(), df.chrf.tolist()
    for c in ("giu_so_dat", "giu_latin_dat"):
        assert df[c].all(), (c, df[df[c] == False][["id", "so_mat", "so_la", "latin_mat"]])        # noqa: E712
    assert not df.sai_chu_viet.any() and not df.sinh_lap.any() and not df.da_cat_loi_dan.any()
    assert df[df.loai == "mo_ta"].dung_so_dong.all()
    assert tm["diem_chinh"]["chrf_macro_huong"] == 100 and tm["thieu_hyp"] == 0
    assert tm["phien_ban"]["sacrebleu"] and tm["cau_hinh"]["min_doan"] == 8


def test_chrf_dich_dung_loai():
    # đích Việt dùng chrF++ (word_order=2), đích Trung dùng chrF ký tự
    a = C.chrf_cau("zh->vi", "cốc giữ nhiệt đen", "Cốc giữ nhiệt inox màu đen")
    assert abs(a - CHRF(word_order=2).sentence_score("cốc giữ nhiệt đen", ["Cốc giữ nhiệt inox màu đen"]).score) < 1e-9
    b = C.chrf_cau("vi->zh", "保温杯黑色", "不锈钢保温杯黑色")
    assert abs(b - CHRF(word_order=0).sentence_score("保温杯黑色", ["不锈钢保温杯黑色"]).score) < 1e-9
    assert 0 < a < 100 and 0 < b < 100


def test_giu_so():
    r = C.giu_so("保温杯 500g", "Cốc 500g")
    assert r["dat"]
    r = C.giu_so("保温杯 500g", "Cốc 300g")
    assert not r["dat"] and r["mat"] == ["500"] and r["la"] == ["300"]
    assert not mot(ZH_TITLE, "Cốc giữ nhiệt inox Elmich 300ml màu đen mẫu 2024").giu_so_dat
    # chuẩn hóa: toàn chiều, phẩy thập phân, dấu phân nghìn
    assert C.giu_so("容量５００ml", "Dung tích 500ml")["dat"]
    assert C.giu_so("重量2.5kg", "Nặng 2,5kg")["dat"]
    assert C.giu_so("价格1,000元", "Giá 1.000 tệ")["dat"]
    assert C.giu_so("0.50升", "0,5 lít")["dat"]
    assert C.giu_so("尺寸10x20cm", "Kích thước 10x20cm")["dat"]
    assert C.giu_so("1.5L", "1,50 lít")["dat"]
    # 25款 <-> 2025 không báo oan; thêm số lạ vẫn bị bắt
    assert C.giu_so("25款 保温杯", "Cốc giữ nhiệt mẫu 2025")["dat"]
    assert C.giu_so("2025款 保温杯", "Cốc giữ nhiệt mẫu 25")["dat"]
    r = C.giu_so("保温杯", "Cốc 2 lớp")
    assert not r["dat"] and r["la"] == ["2"]
    # vi->zh
    assert not mot(VI_TITLE, "Elmich保温杯 600ml 黑色").giu_so_dat


def test_giu_latin():
    assert C.token_latin("Elmich 保温杯 500ml 黑色 SM-G991B", "zh->vi") == ["elmich", "sm-g991b"]
    assert C.giu_latin("Elmich 保温杯", "Cốc Elmich")["dat"]
    assert C.giu_latin("Elmich 保温杯", "Cốc ELMICH")["dat"]                       # không phân biệt hoa thường
    r = C.giu_latin("Elmich 保温杯", "Cốc Elmik")
    assert not r["dat"] and r["mat"] == ["elmich"]
    assert not C.giu_latin("Elmich 保温杯", "Cốc giữ nhiệt")["dat"]                 # xóa thương hiệu
    # Việt nguồn: không bắt từ tiếng Việt không dấu như nhãn "Dung", "Nam"; vẫn bắt thương hiệu
    assert C.token_latin("Thương hiệu: Elmich\nDung lượng: 500ml\nGiới tính: Nam", "vi->zh", "mo_ta", "pair") == ["elmich"]
    assert C.token_latin("Cốc giữ nhiệt Elmich 500ml màu đen", "vi->zh") == ["elmich"]
    assert C.token_latin("Dung dịch rửa tay Lifebuoy", "vi->zh") == ["lifebuoy"]
    assert not mot(VI_TITLE, "Elmik保温杯 500ml 黑色").giu_latin_dat
    assert not mot(VI_MOTA, "品牌: Elmik\n容量: 500ml\n颜色: 黑色").giu_latin_dat
    assert not mot(ZH_MOTA, "Thương hiệu: Elmik; Dung lượng: 500g; Màu sắc: Đen").giu_latin_dat


def test_sai_chu_viet():
    r = mot(ZH_TITLE, "Cốc giữ nhiệt Elmich 保温杯 500ml màu đen mẫu 2024")
    assert r.sai_chu_viet and r.so_han == 3 and r.sai_chu_viet_ly_do == "co_chu_han"
    assert mot(ZH_MOTA, "Thương hiệu: Elmich; Dung lượng: 500g; Màu sắc: 黑色").sai_chu_viet
    r = mot(VI_TITLE, "Elmich保温杯 500ml màu đen")
    assert r.sai_chu_viet and r.so_viet_dau > 0
    r = mot(VI_TITLE, "Elmich black thermos cup 500ml")                            # dịch không ra chữ Trung
    assert r.sai_chu_viet and "nhieu_latin" in r.sai_chu_viet_ly_do
    # Latin có trong nguồn (Elmich, ml) không bị tính
    assert not mot(VI_TITLE, "Elmich 500ml 保温杯 黑色").sai_chu_viet


def test_dung_so_dong():
    assert not mot(ZH_MOTA, "Thương hiệu: Elmich; Dung lượng: 500g").dung_so_dong            # thiếu một dòng
    assert not mot(VI_MOTA, "品牌: Elmich\n容量: 500ml").dung_so_dong
    assert not mot(ZH_MOTA, "Thương hiệu: Elmich; Dung lượng: 500g; Màu sắc: Đen; Thêm: x").dung_so_dong
    assert C.dung_so_dong("a: 1; b: 2", "A: 1; B: 2", "cum") and C.dung_so_dong("a\nb\n", "A\nB", "pair")
    assert mot(ZH_TITLE, "x").dung_so_dong is None or pd.isna(mot(ZH_TITLE, "x").dung_so_dong)        # title: không áp dụng


def test_sinh_lap():
    ref = "Màu đen"
    r = C.sinh_lap("Màu " + "đen " * 30, ref)
    assert r["co"] and "doan_lap" in r["ly_do"]
    assert C.sinh_lap("黑色" * 30, "黑色")["co"]
    assert mot(ZH_TITLE, "Cốc " + "đen " * 40).sinh_lap and mot(VI_TITLE, "黑色" * 40).sinh_lap
    # quá dài so với ref
    r = C.sinh_lap("abcdefghij klmnopqrst uvwxyz" * 3, "ab")
    assert r["co"] and r["ly_do"] == "qua_dai"
    # văn bình thường và lặp ngắn không bị cờ
    assert not C.sinh_lap("Cốc giữ nhiệt inox Elmich 500ml màu đen", "Cốc giữ nhiệt inox Elmich 500ml màu đen mẫu 2024")["co"]
    assert not C.sinh_lap("đen đen đen", "Màu đen đen đen")["co"]
    # ngưỡng là tham số
    assert C.sinh_lap("đen " * 12, "x" * 100, min_doan=8, min_lan=4)["co"]
    assert not C.sinh_lap("đen " * 12, "x" * 100, min_doan=12, min_lan=4)["co"]
    assert abs(C.ti_le_do_dai("abcd", "ab") - 2.0) < 1e-9


def test_cat_loi_dan():
    f = C.cat_loi_dan
    assert f("<think>\nsuy nghĩ\n</think>\n\nÁo thun") == ("Áo thun", True, "think")
    assert f("<think></think>Áo thun") == ("Áo thun", True, "think")
    assert f("```\nÁo thun\n```") == ("Áo thun", True, "rao_code")
    assert f("```text\nA: 1; B: 2\n```") == ("A: 1; B: 2", True, "rao_code")
    assert f("Here is the translation:\nÁo thun") == ("Áo thun", True, "dong_dan")
    assert f("here is the translation\nÁo thun")[0] == "Áo thun"
    assert f("Translation:\nÁo thun")[0] == "Áo thun"
    assert f("TRANSLATION:\nÁo thun")[0] == "Áo thun"
    assert f("Bản dịch:\nÁo thun")[0] == "Áo thun" and f("Dịch:\nÁo thun")[0] == "Áo thun"
    assert f("翻译：\n纯棉T恤")[0] == "纯棉T恤" and f("译文\n纯棉T恤")[0] == "纯棉T恤"
    assert f("Bản dịch: Áo thun") == ("Áo thun", True, "tien_to_dan")
    assert f('"Áo thun"') == ("Áo thun", True, "ngoac_kep")
    assert f("“Áo thun”")[0] == "Áo thun"
    assert f('"Áo" và "thun"') == ('"Áo" và "thun"', False, "")                    # không bao toàn bộ
    assert f("  Áo thun \n") == ("Áo thun", False, "")                              # chỉ strip: không tính là cắt
    s, c, q = f("<think>\nx\n</think>\n```\nTranslation:\n\"Áo thun\"\n```")
    assert (s, c, q) == ("Áo thun", True, "think,rao_code,dong_dan,ngoac_kep"), (s, c, q)
    # KHÔNG cắt nội dung bắt đầu bằng 'Dịch'/'Translation' là một phần câu dịch
    for giu in ["Dịch vụ bảo hành: Có", "Dịch vụ bảo hành: Có; Màu sắc: Đen", "Translation memory", "Dịch vụ", "翻译机 便携式",
                "Áo thun nam\nDịch vụ bảo hành: Có", "Bản dịch hay"]:
        assert f(giu) == (giu, False, ""), (giu, f(giu))
    assert f(None) == ("", False, "") and f("") == ("", False, "")


def test_cat_loi_dan_trong_cham():
    # zero-shot kèm lời dẫn không bị thiệt oan; tắt cắt thì điểm tụt và tỉ lệ cắt được báo
    t = tc([ZH_TITLE, VI_TITLE])
    hy = {"m0": "Here is the translation:\n" + ZH_TITLE[4], "m1": "<think>\n</think>\n" + VI_TITLE[4]}
    df, tm = C.cham(t, dd(t, hy))
    assert (df.chrf == 100).all() and df.da_cat_loi_dan.all()
    assert tm["cat_loi_dan"]["zh->vi/title"]["ti_le"] == 1.0
    assert tm["cat_loi_dan"]["zh->vi/title"]["theo_quy_tac"] == {"dong_dan": 1}
    df2, _ = C.cham(t, dd(t, hy), cat_loi_dan_bat=False)
    assert (df2.chrf < 100).all()


def test_thieu_hyp():
    t = tc(TAT_CA)
    d = dd(t, {"m1": None})                                                          # bỏ hẳn m1
    d = pd.concat([d, pd.DataFrame([{"id": "lạ", "hyp": "x"}])], ignore_index=True)
    df, tm = C.cham(t, d)
    assert tm["thieu_hyp"] == 1 and tm["hyp_thua_khong_co_trong_tham_chieu"] == 1
    r = df[df.id == "m1"].iloc[0]
    assert r.thieu_hyp and r.chrf == 0 and r.hyp_cham == ""
    assert tm["thieu_hyp_theo_nhom"]["zh->vi/mo_ta"] == 1
    assert tm["theo_nhom"]["zh->vi/mo_ta"]["chrf_corpus"] == 0


def test_macro_theo_huong():
    # 1688 nhiều mục (20 title + 20 mo_ta, hoàn hảo), Tiki ít mục (1 + 1) toàn sai: macro = (100 + thấp)/2, không theo số mục
    mau = [("zh->vi", "title", "cum", f"品牌{i} 保温杯 {i}", f"Thương hiệu {i} cốc giữ nhiệt {i}") for i in range(20)]
    mau += [("zh->vi", "mo_ta", "cum", f"品牌: A{i}; 颜色: 黑色", f"Thương hiệu: A{i}; Màu sắc: Đen") for i in range(20)]
    mau += [VI_TITLE, VI_MOTA]
    t = tc(mau)
    hy = {"m40": "xyz", "m41": "xyz\nxyz\nxyz"}
    df, tm = C.cham(t, dd(t, hy))
    ch = CHRF(word_order=0)
    tiki_title = ch.corpus_score(["xyz"], [[VI_TITLE[4]]]).score
    tiki_mota = ch.corpus_score(["xyz\nxyz\nxyz"], [[VI_MOTA[4]]]).score
    ky_vong = (100 + (tiki_title + tiki_mota) / 2) / 2
    dc = tm["diem_chinh"]
    assert abs(dc["chrf_macro_huong"] - ky_vong) < 1e-6, (dc["chrf_macro_huong"], ky_vong)
    assert tm["theo_huong"]["zh->vi"]["chrf_corpus"] == 100 and tm["theo_huong"]["zh->vi"]["loai_gom"] == ["title", "mo_ta"]
    assert dc["chrf_macro_huong"] < 60                                               # trung bình theo mục sẽ ~ 95
    assert abs(dc["chrf_macro_huong"] - df.chrf.mean()) > 30
    assert tm["theo_nhom"]["zh->vi/title"]["n"] == 20 and "zh->vi/mo_ta/home" in tm["theo_category"]


def test_dong_da_thay_moi():
    # tập huấn luyện: fold5 = 1 chứa 'Thương hiệu', fold5 = 2 chứa 'màu', fold5 = -1 (test) chứa 'dung lượng', khác category chứa 'trọng lượng'
    mau = pd.DataFrame([
        {"id": "a", "category": "home", "loai": "mo_ta", "nhanh": "cum", "fold5": 1, "nguon": "品牌: Elmich; 材质: 不锈钢"},
        {"id": "b", "category": "home", "loai": "mo_ta", "nhanh": "cum", "fold5": 2, "nguon": "颜色: 黑色"},
        {"id": "c", "category": "home", "loai": "mo_ta", "nhanh": "cum", "fold5": -1, "nguon": "容量: 500克"},
        {"id": "d", "category": "toys", "loai": "mo_ta", "nhanh": "cum", "fold5": 3, "nguon": "容量: 500克"},
        {"id": "e", "category": "home", "loai": "title", "nhanh": "cum", "fold5": 3, "nguon": "容量: 500克"}])
    tap1 = C.xay_tap_da_thay(mau, fold_kiem=1)
    assert ("home", "品牌: Elmich") not in tap1 and ("home", "颜色: 黑色") in tap1
    assert ("home", "容量: 500克") not in tap1 and ("toys", "容量: 500克") in tap1
    tap_all = C.xay_tap_da_thay(mau)
    assert ("home", "品牌: Elmich") in tap_all and ("home", "容量: 500克") not in tap_all
    t = tc([ZH_MOTA])                                      # 品牌 / 容量 / 颜色
    hy = {"m0": "Thương hiệu: Elmich; Dung lượng: 500 g; Màu sắc: Đen"}               # dòng 'dung lượng' sai nhẹ
    df, tm = C.cham(t, dd(t, hy), tap_da_thay=tap1)
    r = df.iloc[0]
    assert r.can_dong and r.n_dong_da_thay == 1 and r.n_dong_moi == 2
    dm = tm["dong_da_thay_moi"]["zh->vi"]
    assert dm["da_thay"]["n_dong"] == 1 and dm["da_thay"]["chrf_corpus"] == 100          # 'Màu sắc: Đen' đúng
    assert dm["moi"]["n_dong"] == 2 and dm["moi"]["chrf_corpus"] < 100
    ch = CHRF(word_order=2)
    mv = ch.corpus_score(["Thương hiệu: Elmich", "Dung lượng: 500 g"], [["Thương hiệu: Elmich", "Dung lượng: 500g"]]).score
    assert abs(dm["moi"]["chrf_corpus"] - mv) < 1e-6
    # fold_kiem khác: 'Thương hiệu' thành đã thấy
    tap2 = C.xay_tap_da_thay(mau, fold_kiem=2)
    df, tm = C.cham(t, dd(t, hy), tap_da_thay=tap2)
    assert df.iloc[0].n_dong_da_thay == 1 and df.iloc[0].n_dong_moi == 2
    assert tm["dong_da_thay_moi"]["zh->vi"]["da_thay"]["chrf_corpus"] == 100               # 'Thương hiệu: Elmich' đúng
    # số dòng hyp lệch -> không căn được, để trống và đếm
    df, tm = C.cham(t, dd(t, {"m0": "Thương hiệu: Elmich"}), tap_da_thay=tap1)
    assert df.iloc[0].can_dong == False and pd.isna(df.iloc[0].n_dong_da_thay)              # noqa: E712
    dm = tm["dong_da_thay_moi"]["zh->vi"]
    assert dm["so_muc_khong_can_duoc"] == 1 and dm["da_thay"]["n_dong"] == 0 and dm["da_thay"]["chrf_corpus"] is None
    # không truyền mẫu huấn luyện thì bỏ phần này
    _, tm = C.cham(t, dd(t))
    assert "dong_da_thay_moi" not in tm


def test_comet_thieu_thu_vien():
    # không có comet/mô hình thì trả None, không làm hỏng
    assert C.cham_comet(["a"], ["b"], ["c"], "khong/co/mo/hinh.ckpt") is None
    assert C.cham_kiwi(["a"], ["b"], "khong/co/mo/hinh.ckpt") is None
    assert C.cham_comet(["a"], ["b"], ["c"], None) is None
    t = tc([ZH_TITLE])
    df, tm = C.cham(t, dd(t), comet="khong/co.ckpt")
    assert "comet" not in df.columns and tm["comet"]["comet"].startswith("khong_chay_duoc")


def test_dong_lenh_va_file_ra():
    t = tc(TAT_CA)
    t["fold5"] = -1
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with open(td / "ref.jsonl", "w", encoding="utf-8") as f:
            for r in t.to_dict("records"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        hy = {"m0": "Here is the translation:\n" + ZH_TITLE[4]}
        with open(td / "hyp.jsonl", "w", encoding="utf-8") as f:
            for r in dd(t, hy).to_dict("records"):
                r["hyp_tho"] = r["hyp"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(td / "train.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "x", "category": "home", "loai": "mo_ta", "nhanh": "cum", "fold5": 0,
                                "nguon": "品牌: Elmich", "messages": []}, ensure_ascii=False) + "\n")
        C.main(["--tham-chieu", str(td / "ref.jsonl"), "--du-doan", str(td / "hyp.jsonl"), "--ra-thu-muc", str(td / "ra"),
                "--nhanh", "cum", "--mau-huan-luyen", str(td / "train.jsonl"), "--fold-kiem", "3"])
        dong = [json.loads(x) for x in (td / "ra" / "diem_tung_muc.jsonl").read_text(encoding="utf-8").splitlines()]
        tm = json.loads((td / "ra" / "tong_hop.json").read_text(encoding="utf-8"))
        assert len(dong) == 2 and all("_ref" not in d and "_dong" not in d for d in dong)       # --nhanh cum lọc bỏ mục pair
        assert tm["cau_hinh"]["fold_kiem"] == 3 and tm["dong_da_thay_moi"]["zh->vi"]["da_thay"]["n_dong"] == 1
        assert tm["diem_chinh"]["chrf_macro_huong"] == 100 and "sacrebleu" in tm["phien_ban"]


def test_dong_ngu_canh():
    """Sửa 07/11/2026: dòng chép lại lời dẫn 'Product title (context only...)' đã dịch bị cắt; dòng thuộc tính bình thường không bị đụng."""
    sach, cat, qt = C.cat_loi_dan("产品标题（仅上下文，不翻译）：Cốc giữ nhiệt\n品牌: Elmich\n容量: 500ml")
    assert sach == "品牌: Elmich\n容量: 500ml" and cat and qt == "dong_ngu_canh", (sach, qt)
    sach, cat, qt = C.cat_loi_dan("Tiêu đề sản phẩm (chỉ để làm ngữ cảnh, không dịch): 保温杯\nThương hiệu: Elmich")
    assert sach == "Thương hiệu: Elmich" and qt == "dong_ngu_canh", (sach, qt)
    sach, cat, qt = C.cat_loi_dan("Product title (context only, do not translate): x\nThương hiệu: Elmich")
    assert sach == "Thương hiệu: Elmich" and qt == "dong_ngu_canh"
    for ok in ("品牌: Elmich\n产品标题: 保温杯", "Thương hiệu: Elmich\nTên sản phẩm: Cốc", "产品: 保温杯 (仅限室内)"):
        s, c, q = C.cat_loi_dan(ok)
        assert s == ok and not c, (ok, s, q)
    # kết hợp với lời dẫn đầu dòng: cả hai quy tắc cùng chạy
    s, c, q = C.cat_loi_dan("Here is the translation:\n产品标题（仅上下文）：x\nA: 1")
    assert s == "A: 1" and q == "dong_ngu_canh,dong_dan", (s, q)


def test_moc_tham_chieu():
    """Sửa 07/11/2026: giu_so, giu_latin, sai_chu_viet cũng chạy trên bản tham chiếu; bản dịch hỏng thì mốc vẫn nguyên, bản dịch đúng thì hai bên bằng nhau."""
    t = tc(TAT_CA)
    df, tm = C.cham(t, dd(t))
    for c in ("ref_giu_so_dat", "ref_giu_latin_dat", "ref_sai_chu_viet"):
        assert c in df.columns and "_ref" not in c
    assert (df["ref_giu_so_dat"] == df["giu_so_dat"]).all() and (df["ref_sai_chu_viet"] == df["sai_chu_viet"]).all()
    nhom = tm["theo_nhom"]["zh->vi/title"]
    assert nhom["ref_ti_le_giu_so_dat"] == nhom["ti_le_giu_so_dat"] and "ref_ti_le_sai_chu_viet" in nhom and "ref_ti_le_giu_latin_dat" in nhom
    hong = dd(t, {"m0": "Cốc giữ nhiệt inox màu đen"})                # mất 500ml và Elmich: hyp kém mốc, mốc không đổi
    df2, tm2 = C.cham(t, hong)
    assert df2.loc[df2.id == "m0", "giu_so_dat"].iloc[0] in (False, 0) and bool(df2.loc[df2.id == "m0", "ref_giu_so_dat"].iloc[0]) == bool(df.loc[df.id == "m0", "ref_giu_so_dat"].iloc[0])


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_") and callable(ham):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
