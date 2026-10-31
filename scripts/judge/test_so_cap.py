"""Bài thử cho bộ so cặp mù: chủ yếu kiểm tra KHÔNG THIÊN VỊ (mù, đối xứng, không mặc định). Chạy:
python -I -X utf8 scripts/judge/test_so_cap.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import so_cap_chon as SC  # noqa: E402
import so_cap_du_lieu as DL  # noqa: E402
import so_cap_rubric as SR  # noqa: E402


def muc(g="GGG", q="QQQ", **them):
    d = {"item_id": "title:1688:1", "kind": "title", "source": "1688", "huong": "zh->vi", "category": "home",
         "nguon": "SRC", "cand_gemma": g, "cand_qwen": q, "kiwi_g": 0.7777, "kiwi_q": 0.3333, "margin": 0.4444,
         "bin_margin": ">=0.1", "san_pham_n": 1, "vi_du_product_id": "SECRET123"}
    d.update(them)
    return d


def vi_tri(prompt):
    m = SC._MAU.search(prompt)
    return m.group(1), m.group(2)


def goi_theo(quy_tac):
    """Tạo backend giả: quy_tac(t1, t2) -> 1 | 2 | 'tie'. Đếm số lời gọi và lưu thứ tự đã thấy."""
    thay = []

    def goi(khoa, tn, max_token=None):
        t1, t2 = vi_tri(tn[-1]["content"])
        thay.append((t1, t2))
        w = quy_tac(t1, t2)
        return json.dumps({"winner": w, "criterion": "meaning", "reason": "x"}), 10, 5, "stop", {}
    goi.thay = thay
    return goi


def test_mu_khong_ro_ri():
    """Lời nhắc không chứa tên mô hình, điểm Kiwi, id hay trường nào ngoài danh sách cho phép."""
    ban_cam = ["0.7777", "0.3333", "0.4444", "SECRET123", "title:1688:1"]
    for thu_tu in ("gq", "qg"):
        p, _ = SR.dung_nguoi_dung(muc(ten_mo_hinh="GEMMA-SECRET"), thu_tu)
        for s in ban_cam + ["GEMMA-SECRET"]:
            assert s not in p, (thu_tu, s)
        khung = p.replace("GGG", "").replace("QQQ", "").replace("SRC", "").lower()
        for tu in SR.TU_CAM:
            assert tu not in khung, tu
    for tu in SR.TU_CAM:
        assert tu not in SR.HE_THONG.lower(), tu

    class Ghi(dict):
        da_doc = set()

        def __getitem__(self, k):
            Ghi.da_doc.add(k)
            return dict.__getitem__(self, k)
    SR.dung_nguoi_dung(Ghi(muc()), "gq")
    assert Ghi.da_doc <= set(SR.DANH_SACH_TRUONG), Ghi.da_doc - set(SR.DANH_SACH_TRUONG)


def test_doi_xung_hai_thu_tu():
    p1, x1 = SR.dung_nguoi_dung(muc(), "gq")
    p2, x2 = SR.dung_nguoi_dung(muc(), "qg")
    assert x1 == {1: "gemma", 2: "qwen"} and x2 == {1: "qwen", 2: "gemma"}
    hoan = p1.replace("GGG", "@@").replace("QQQ", "GGG").replace("@@", "QQQ")
    assert hoan == p2                                      # hai lời nhắc chỉ khác nhau chỗ hoán đổi hai bản dịch


def test_luon_hoi_du_hai_thu_tu():
    goi = goi_theo(lambda a, b: "tie")
    SC.cham_muc(muc(), "k", goi)
    assert goi.thay == [("GGG", "QQQ"), ("QQQ", "GGG")]    # đúng 2 lời gọi, Gemma ở vị trí 1 rồi Qwen ở vị trí 1


def test_thien_vi_vi_tri_khong_tao_nguoi_thang():
    for vt in (1, 2):                                      # judge chỉ biết chọn vị trí 1 (hoặc luôn vị trí 2)
        kq = SC.cham_muc(muc(), "k", goi_theo(lambda a, b, vt=vt: vt))
        assert kq["trang_thai"] == "ok" and kq["ket_qua"] == "khong_nhat_quan"


def test_chon_theo_noi_dung_khong_theo_thu_tu_va_khong_theo_ten():
    tot = goi_theo(lambda a, b: 1 if "GOOD" in a else (2 if "GOOD" in b else "tie"))
    assert SC.cham_muc(muc(g="GOOD g", q="bad"), "k", tot)["ket_qua"] == "gemma"
    assert SC.cham_muc(muc(g="bad", q="GOOD q"), "k", tot)["ket_qua"] == "qwen"      # đối xứng: Qwen thắng được
    assert SC.cham_muc(muc(g="bad", q="meh"), "k", tot)["ket_qua"] == "hoa_that"


def test_hoa_mot_ben_la_khong_nhat_quan_va_khong_co_mac_dinh():
    dem = {"n": 0}

    def goi(khoa, tn, max_token=None):
        dem["n"] += 1
        w = "tie" if dem["n"] == 1 else 1
        return json.dumps({"winner": w, "criterion": "none", "reason": ""}), 1, 1, "stop", {}
    kq = SC.cham_muc(muc(), "k", goi)
    assert kq["ket_qua"] == "khong_nhat_quan"
    assert SR.gop("gemma", "qwen") == "khong_nhat_quan" and SR.gop("hoa_that", "gemma") == "khong_nhat_quan"


def test_doc_ket_qua_va_loi():
    assert SR.doc_ket_qua('{"winner": 1, "criterion": "fluency", "reason": "a"}')["winner"] == 1
    assert SR.doc_ket_qua('```json\n{"winner": "2", "criterion": "none", "reason": ""}\n```')["winner"] == 2
    assert SR.doc_ket_qua('{"winner": "Tie", "criterion": "none"}')["winner"] == "tie"
    for xau in ['{"winner": 3, "criterion": "none"}', '{"winner": 1, "criterion": "tot hon"}', "khong co json", ""]:
        try:
            SR.doc_ket_qua(xau)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError(xau)
    kq = SC.cham_muc(muc(), "k", lambda k, t, m=None: ("rác", 1, 1, "stop", {}))
    assert kq["trang_thai"] == "loi"
    kq = SC.cham_muc(muc(), "k", lambda k, t, m=None: ("{}", 1, 1, "length", {}))
    assert kq["ly_do"] == "bi_cat_het_token"


def tao_du_lieu(tmp):
    giu = pd.DataFrame({"product_id": ["1", "2", "3", "t1"], "source": ["1688", "1688", "1688", "tiki"],
                        "category": ["home", "home", "bags", "auto"]})
    t1688 = pd.DataFrame({"product_id": ["1", "2", "3", "9"], "title_zh": ["z1", "z2", "z3", "z9"],
                          "title_vi": ["x"] * 4, "gemma4_translation": ["Cái  Ly", "Giống", "Túi", "Khác"],
                          "qwen3_14b_awq_translation": ["Chén", "giống ", "Ba lô", "Khác2"],
                          "cometkiwi_gemma4": [0.70, 0.5, 0.6, 0.5], "cometkiwi_qwen3_14b_awq": [0.60, 0.5, 0.58, 0.4]})
    ttiki = pd.DataFrame({"product_id": ["t1"], "title_vi": ["nguồn việt"], "title_zh": ["x"],
                          "gemma4_translation": ["中文甲"], "qwen3_14b_awq_translation": ["中文乙"],
                          "cometkiwi_gemma4": [0.9], "cometkiwi_qwen3_14b_awq": [0.5]})
    a, b = Path(tmp) / "a.parquet", Path(tmp) / "b.parquet"
    t1688.to_parquet(a), ttiki.to_parquet(b)
    cap = pd.DataFrame({"product_id": ["1", "2", "3"], "name_zh": ["颜色", "颜色", "材质"], "gia_tri_zh": ["红", "红", "牛皮"],
                        "n_cuoi_A": ["Màu", "Màu", "Chất liệu"], "v_cuoi_A": ["Đỏ", "Đỏ", "Da bò"],
                        "n_cuoi_B": ["Màu sắc", "Màu sắc", "Chất liệu"], "v_cuoi_B": ["Đỏ", "Đỏ", "Da bò"]})
    c = Path(tmp) / "c.parquet"
    cap.to_parquet(c)
    return giu, {"1688": a, "tiki": b}, {"1688": (c, "name_zh", "gia_tri_zh")}


def test_dung_muc_bo_giong_nhau_va_gop_trung():
    with tempfile.TemporaryDirectory() as tmp:
        giu, tp, cp = tao_du_lieu(tmp)
        t = DL.xay_title(giu, tp)
        assert set(t.item_id) == {"title:1688:1", "title:1688:3", "title:tiki:t1"}    # "2" giống hệt (chuẩn hóa), "9" ngoài tập giữ
        assert abs(t.set_index("item_id").loc["title:1688:1", "margin"] - 0.10) < 1e-9
        assert t.set_index("item_id").loc["title:tiki:t1", "nguon"] == "nguồn việt"       # nguồn Tiki là title_vi
        assert t.set_index("item_id").loc["title:1688:1", "nguon"] == "z1"                # nguồn 1688 là title_zh
        c = DL.xay_cum(giu, cp)
        assert len(c) == 1 and c.san_pham_n.iloc[0] == 2                                  # cặp của sản phẩm 1 và 2 gộp làm một
        assert c[c.cand_gemma == "Chất liệu: Da bò"].empty                                # sản phẩm 3: A = B thì bị bỏ


def test_khac_nhau_that_su():
    assert DL.chuan_khac("柔软超细纤维布/ 玻璃纤维") == DL.chuan_khac("柔软超细纤维布/玻璃纤维")        # chỉ khác khoảng trắng
    assert DL.chuan_khac("尺寸: Size M") == DL.chuan_khac("尺寸：Size  M")                            # dấu hai chấm toàn góc, 2 khoảng trắng
    assert DL.chuan_khac("Quần áo mùa xuân cho trẻ sơ sinh") != DL.chuan_khac("Quần áo cho trẻ em mùa xuân")
    assert DL.chuan_khac("Phổ thông") != DL.chuan_khac("Tổng hợp") and DL.chuan_khac("100") != DL.chuan_khac("1000")


def muc_ctx(**them):
    return muc(kind="cum_pair", nguon="材质: 棉", ngu_canh="Title: T恤\nAll attributes of this product:\n- 材质: 棉\n- 颜色: 红",
               id_khong_ngu_canh="pair:1688:abc", **them)


def test_ngu_canh_chi_co_nguon_va_van_doi_xung():
    for thu_tu in ("gq", "qg"):
        p, _ = SR.dung_nguoi_dung(muc_ctx(), thu_tu)
        assert "PRODUCT CONTEXT" in p and "Title: T恤" in p and p.index("PRODUCT CONTEXT") < p.index("SOURCE (")
        for cam in ("0.7777", "0.3333", "SECRET123", "title:1688:1", "pair:1688:abc"):
            assert cam not in p, cam                                   # ngữ cảnh không kéo theo id / điểm / khóa
        khung = p.replace("GGG", "").replace("QQQ", "").replace("SRC", "").lower()
        for tu in SR.TU_CAM:
            assert tu not in khung, tu
    p1, _ = SR.dung_nguoi_dung(muc_ctx(), "gq")
    p2, _ = SR.dung_nguoi_dung(muc_ctx(), "qg")
    assert p1.replace("GGG", "@@").replace("QQQ", "GGG").replace("@@", "QQQ") == p2
    for tu in SR.TU_CAM:
        assert tu not in SR.HE_THONG_NGU_CANH.lower(), tu
    assert "PRODUCT CONTEXT" in SR.HE_THONG_NGU_CANH and SR.he_thong(muc_ctx()) == SR.HE_THONG_NGU_CANH
    # mục KHÔNG ngữ cảnh: lời nhắc và hệ thống y như cũ (không đổi giao thức của title / pair / cụm không ngữ cảnh)
    p0, _ = SR.dung_nguoi_dung(muc(), "gq")
    assert "PRODUCT CONTEXT" not in p0 and SR.he_thong(muc()) == SR.HE_THONG and SR.phien_ban_muc(muc()) == SR.PHIEN_BAN
    assert SR.phien_ban_muc(muc_ctx()) == SR.PHIEN_BAN_NGU_CANH
    assert SR.HE_THONG_NGU_CANH.startswith(SR.HE_THONG.split("Reply with ONE JSON")[0])      # cùng tiêu chí, chỉ thêm một đoạn


def test_dung_ngu_canh_cat_giu_cap_dang_xet():
    cap = [f"tên{i}: giá trị {i}" for i in range(50)]
    ngan = DL.dung_ngu_canh("Tiêu đề", cap[:3], 1)
    assert "Title: Tiêu đề" in ngan and "omitted" not in ngan and all(c in ngan for c in cap[:3])
    dai = DL.dung_ngu_canh("Tiêu đề", cap, 25, toi_da=300)
    assert "tên25: giá trị 25" in dai and "tên24:" in dai and "tên26:" in dai and "more attributes omitted" in dai
    assert "tên0:" not in dai and "tên49:" not in dai and len(dai) < 450
    dau = DL.dung_ngu_canh("T", cap, 0, toi_da=200)
    assert "tên0:" in dau and "tên1:" in dau


def test_xay_cum_ctx_moi_san_pham_mot_muc_va_ngu_canh_dung():
    with tempfile.TemporaryDirectory() as tmp:
        giu = pd.DataFrame({"product_id": ["1", "2"], "source": "1688", "category": ["home", "home"],
                            "title_zh": ["标题一", "标题二"], "title_vi": ["x", "y"]})
        cap = pd.DataFrame({"pair_id": [11, 12, 13, 21], "product_id": ["1", "1", "1", "2"],
                            "name_zh": ["颜色", "材质", "尺寸", "颜色"], "gia_tri_zh": ["红", "棉", "M", "红"],
                            "n_cuoi_A": ["Màu", "Chất liệu", "Cỡ", "Màu"], "v_cuoi_A": ["Đỏ", "Cotton", "M", "Đỏ"],
                            "n_cuoi_B": ["Màu", "Chất liệu", "Cỡ", "Màu"], "v_cuoi_B": ["Đỏ", "Bông", "M", "Đỏ"]})
        cap.loc[3, "n_cuoi_B"] = "Màu sắc"                                    # product 2: tên khác
        c = Path(tmp) / "c.parquet"
        cap.to_parquet(c)
        x = DL.xay_cum_ctx(giu, {"1688": (c, "name_zh", "gia_tri_zh")})
        assert sorted(x.item_id) == ["cumctx:1688:1:12", "cumctx:1688:2:21"]             # 2 mục: cặp khác nhau của từng sản phẩm
        a = x.set_index("item_id").loc["cumctx:1688:1:12"]
        assert a.nguon == "材质: 棉" and a.cand_gemma == "Chất liệu: Cotton" and a.cand_qwen == "Chất liệu: Bông"
        assert "Title: 标题一" in a.ngu_canh and "- 颜色: 红" in a.ngu_canh and "- 尺寸: M" in a.ngu_canh and "- 材质: 棉" in a.ngu_canh
        assert "标题二" not in a.ngu_canh and "Cotton" not in a.ngu_canh and "Bông" not in a.ngu_canh   # chỉ nguồn, không có bản dịch
        assert a.id_khong_ngu_canh == DL.id_cum("1688", DL.khoa_cum("材质: 棉", "Chất liệu: Cotton", "Chất liệu: Bông", "home"))


def test_ban_ghi_ngu_canh_va_chay_tiep_theo_phien_ban():
    m = muc_ctx(item_id="cumctx:1688:1:1")
    kq = SC.cham_muc(m, "k", goi_theo(lambda a, b: 1))
    b = SC.ban_ghi(m, kq)
    assert b["phien_ban"] == SR.PHIEN_BAN_NGU_CANH and b["id_khong_ngu_canh"] == "pair:1688:abc" and b["dai_ngu_canh"] == len(m["ngu_canh"])
    assert "phien_ban" in SC.ban_ghi(muc(), kq) and SC.ban_ghi(muc(), kq)["phien_ban"] == SR.PHIEN_BAN
    assert "id_khong_ngu_canh" not in SC.ban_ghi(muc(), kq)
    with tempfile.TemporaryDirectory() as tmp:
        goi = goi_theo(lambda a, b: 1 if "GOOD" in a else 2)
        it = pd.DataFrame([muc_ctx(item_id=f"cumctx:1688:{i}:1", cand_gemma="GOOD g", cand_qwen="bad") for i in range(6)])
        assert SC.chay(it, "cx", "k", goi, tmp, 2, in_tien_do=False) == 0 and len(goi.thay) == 12
        assert SC.chay(it, "cx", "k", goi, tmp, 2, in_tien_do=False) == 0 and len(goi.thay) == 12      # chạy tiếp: không hỏi lại
        import so_sanh_ngu_canh as SN
        assert len(SN.doc(Path(tmp) / "ket_qua_cx.jsonl")) == 6


def test_so_sanh_ngu_canh_dem_dung():
    import so_sanh_ngu_canh as SN

    def b(i, kq, tv=100, nc=None):
        d = {"item_id": i, "trang_thai": "ok", "ket_qua": kq, "tokens_vao": tv, "tokens_ra": 10, "giay_goi": 1.0,
             "luot": [{"thu_tu": "gq", "vi_tri_chon": 1, "chon": "gemma", "tieu_chi": "meaning", "ly_do": ""},
                      {"thu_tu": "qg", "vi_tri_chon": 2, "chon": "gemma", "tieu_chi": "meaning", "ly_do": ""}]}
        if nc:
            d["id_khong_ngu_canh"] = nc
        return d
    khong = {"k1": b("k1", "khong_nhat_quan"), "k2": b("k2", "gemma"), "k3": b("k3", "gemma"), "k4": b("k4", "gemma")}
    ctx = {"c1": b("c1", "qwen", 300, "k1"), "c2": b("c2", "gemma", 300, "k2"), "c3": b("c3", "qwen", 300, "k3"),
           "c4": b("c4", "hoa_that", 300, "k4"), "c5": b("c5", "gemma", 300, "k_khong_co")}
    r = SN.so_sanh(ctx, khong, tong_muc_ctx=1000)
    assert r["so_cap_so_duoc"] == 4                                                      # c5 không có mục đối chiếu
    assert r["chua_quyet_nhung_nay_quyet_duoc"] == 1 and r["quyet_duoc_nhung_nay_khong_quyet"] == 1
    assert r["ca_hai_deu_quyet"] == {"so_muc": 2, "dao_nguoc": 1, "dao_sang_gemma": 0, "dao_sang_qwen": 1}
    assert r["ti_le_co_nguoi_thang"] == {"khong_ngu_canh": 0.75, "ngu_canh": 0.75}
    assert r["moi_muc_trung_binh"]["tokens_vao"] == {"khong_ngu_canh": 100, "ngu_canh": 300, "ti_so": 3.0}
    assert r["uoc_tinh_toan_bo_cum_ngu_canh"]["tokens_vao_trieu"] == 0.3 and r["uoc_tinh_toan_bo_cum_ngu_canh"]["so_loi_goi"] == 2000


def test_so_sanh_lap_do_nhieu_cua_judge():
    import so_sanh_ngu_canh as SN
    a = {"i1": {"ket_qua": "gemma"}, "i2": {"ket_qua": "khong_nhat_quan"}, "i3": {"ket_qua": "gemma"}, "i4": {"ket_qua": "qwen"},
         "i5": {"ket_qua": "hoa_that"}, "ngoai": {"ket_qua": "gemma"}}
    b = {"i1": {"ket_qua": "gemma"}, "i2": {"ket_qua": "gemma"}, "i3": {"ket_qua": "khong_nhat_quan"}, "i4": {"ket_qua": "gemma"},
         "i5": {"ket_qua": "hoa_that"}}
    r = SN.so_sanh_lap(a, b)
    assert r["so_muc_chung"] == 5 and r["giong_het_ket_qua"] == 0.4
    assert (r["lan1_co_nguoi_thang"], r["trong_do_lan2_van_cung_nguoi_thang"]) == (3, 1)
    assert r["lan1_chua_quyet_lan2_quyet_duoc"] == 1 and r["lan1_quyet_lan2_chua_quyet"] == 1 and r["dao_nguoc_gemma_qwen"] == 1


def tao_du_lieu_tang2(tmp):
    """Dữ liệu giả cho tầng 2: 3 sản phẩm 1688 (1, 2, 3) và 1 Tiki (t1, không có thuộc tính)."""
    tmp = Path(tmp)
    giu = pd.DataFrame({"product_id": ["1", "2", "3", "t1"], "source": ["1688", "1688", "1688", "tiki"],
                        "category": ["home", "home", "bags", "auto"], "title_zh": ["标题1", "标题2", "标题3", "x"],
                        "title_vi": ["x", "x", "x", "tiêu đề tiki"]})
    t1688 = pd.DataFrame({"product_id": ["1", "3"], "title_zh": ["标题1", "标题3"], "title_vi": ["x", "x"],
                          "gemma4_translation": ["Cái Ly", "Túi"], "qwen3_14b_awq_translation": ["Chén", "Ba lô"],
                          "cometkiwi_gemma4": [0.7, 0.6], "cometkiwi_qwen3_14b_awq": [0.6, 0.5]})
    ttiki = pd.DataFrame({"product_id": ["t1"], "title_vi": ["tiêu đề tiki"], "title_zh": ["x"],
                          "gemma4_translation": ["中文甲"], "qwen3_14b_awq_translation": ["中文乙"],
                          "cometkiwi_gemma4": [0.9], "cometkiwi_qwen3_14b_awq": [0.5]})
    a, b = tmp / "ta.parquet", tmp / "tb.parquet"
    t1688.to_parquet(a), ttiki.to_parquet(b)
    cap = pd.DataFrame({"pair_id": [11, 12, 21, 31], "product_id": ["1", "1", "2", "3"],
                        "name_zh": ["颜色", "材质", "颜色", "材质"], "gia_tri_zh": ["红", "棉", "蓝", "牛皮"],
                        "n_cuoi_A": ["Màu"] * 3 + ["Chất liệu"], "v_cuoi_A": ["Đỏ", "Cotton", "Xanh", "Da bò"],
                        "n_cuoi_B": ["Màu"] * 3 + ["Chất liệu"], "v_cuoi_B": ["Đỏ", "Cotton", "Xanh", "Da bò"]})
    c = tmp / "cap.parquet"
    cap.to_parquet(c)
    # bản PAIR: file đơn vị + file từng sản phẩm
    pdv = pd.DataFrame({"pair_id": [1, 2, 3], "product_id": ["1", "1", "2"], "name_zh": ["颜色", "材质", "颜色"],
                        "gia_tri_zh": ["红", "棉", "蓝"]})
    pc = tmp / "dv_sp.parquet"
    pdv.to_parquet(pc)
    dv = pd.DataFrame({"ma_don_vi": ["T1", "V1", "V2", "T2", "C1"], "loai": ["ten", "gia_tri", "gia_tri", "ten", "gia_tri"],
                       "van_ban_zh": ["颜色", "红", "蓝", "材质", "棉"], "A": ["Màu", "Đỏ", "Xanh", "Chất liệu", "Cotton"],
                       "B": ["Màu sắc", "Đỏ tươi", "Lam", "Chất liệu", "Bông"], "tu": ["A", "A", "B", "A", "co_dinh"],
                       "category": "home", "kiwi_A": 0.5, "kiwi_B": 0.4})
    pu = tmp / "dv.parquet"
    dv.to_parquet(pu)
    return (giu, {"1688": a, "tiki": b}, {"1688": (c, "name_zh", "gia_tri_zh")},
            {"1688": (pu, pc, "name_zh", "gia_tri_zh", "van_ban_zh")})


def test_xay_title_ctx_chi_thuoc_tinh_va_cung_id_voi_tang1():
    with tempfile.TemporaryDirectory() as tmp:
        giu, tp, cp, _ = tao_du_lieu_tang2(tmp)
        t1 = DL.xay_title(giu, tp)
        tx, bo = DL.xay_title_ctx(giu, tp, cp)
        assert set(tx.item_id) == {"title:1688:1", "title:1688:3"} and bo == 1               # Tiki t1 không có thuộc tính: bỏ, đếm 1
        assert set(tx.item_id) <= set(t1.item_id)                                            # CÙNG id với tầng 1 để ghi đè đúng mục
        x = tx.set_index("item_id")
        assert x.loc["title:1688:1", "ngu_canh"] == "All attributes of this product:\n- 颜色: 红\n- 材质: 棉"
        assert "Title:" not in x.loc["title:1688:1", "ngu_canh"]                              # văn bản đang xét CHÍNH LÀ tiêu đề
        assert "Cái Ly" not in x.loc["title:1688:1", "ngu_canh"] and "Chén" not in x.loc["title:1688:1", "ngu_canh"]
        assert x.loc["title:1688:3", "ngu_canh"] == "All attributes of this product:\n- 材质: 牛皮"
        assert (tx.cand_gemma == t1.set_index("item_id").loc[tx.item_id, "cand_gemma"].values).all()   # ứng viên y hệt tầng 1


def test_xay_pair_ctx_san_pham_dai_dien_co_dinh_va_cung_id_voi_tang1():
    with tempfile.TemporaryDirectory() as tmp:
        giu, _, _, dvp = tao_du_lieu_tang2(tmp)
        u1 = DL.xay_don_vi(giu, dvp)
        ux = DL.xay_pair_ctx(giu, dvp)
        assert sorted(u1.item_id) == sorted(ux.item_id) == ["unit:1688:T1", "unit:1688:V1", "unit:1688:V2"]    # T2 (A=B) và C1 (co_dinh) không có
        x = ux.set_index("item_id")
        assert x.loc["unit:1688:T1", "vi_du_product_id"] == "1" and x.loc["unit:1688:V2", "vi_du_product_id"] == "2"
        assert x.loc["unit:1688:T1", "ngu_canh"].startswith("Title: 标题1\nAll attributes of this product:\n- 颜色: 红")
        assert "标题2" in x.loc["unit:1688:V2", "ngu_canh"] and "- 颜色: 蓝" in x.loc["unit:1688:V2", "ngu_canh"]
        assert "Màu" not in x.loc["unit:1688:T1", "ngu_canh"] and "Đỏ" not in x.loc["unit:1688:T1", "ngu_canh"]       # không có bản dịch
        # giữ cặp chứa đơn vị khi bị cắt
        dai = DL.dung_ngu_canh("T", [f"n{i}: v{i}" for i in range(30)], 20, toi_da=120)
        assert "n20: v20" in dai and "omitted" in dai


def test_chi_chay_tang2_cho_muc_khong_nhat_quan_va_danh_dau_tang():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tx = pd.DataFrame([muc(item_id=f"title:1688:{i}", ngu_canh="All attributes of this product:\n- a: b") for i in range(5)])
        tx.to_parquet(t / "muc_title_ctx.parquet", index=False)
        tang1 = [("title:1688:0", "gemma"), ("title:1688:1", "khong_nhat_quan"), ("title:1688:2", "hoa_that"),
                 ("title:1688:3", "khong_nhat_quan"), ("title:1688:9", "khong_nhat_quan")]        # :9 không có trong bảng tầng 2
        (t / "ket_qua_t1.jsonl").write_text("\n".join(json.dumps({"item_id": i, "trang_thai": "ok", "ket_qua": k}) for i, k in tang1) +
                                            "\n" + json.dumps({"item_id": "title:1688:3", "trang_thai": "loi"}), encoding="utf-8")
        assert SC.ids_khong_nhat_quan(t, ["t1"]) == {"title:1688:1", "title:1688:3", "title:1688:9"}
        assert SC.main(["--loai", "title_ctx", "--chi-khong-nhat-quan", "t1", "--tat-ca", "--gia", "--ra", tmp, "--ten", "t2",
                        "--vao", str(t / "muc_title_ctx.parquet")]) == 0
        rows = [json.loads(x) for x in open(t / "ket_qua_gia_t2.jsonl", encoding="utf-8")]
        assert {r["item_id"] for r in rows} == {"title:1688:1", "title:1688:3"}                    # chỉ mục không nhất quán và có trong bảng
        assert all(r["tang"] == 2 and r["phien_ban"] == SR.PHIEN_BAN_NGU_CANH for r in rows)
        # tầng 2 không được chạy tràn: thiếu --chi-khong-nhat-quan là lỗi (trừ khi chạy thử giả)
        try:
            SC.main(["--loai", "title_ctx", "--tat-ca", "--ra", tmp, "--ten", "x", "--vao", str(t / "muc_title_ctx.parquet")])
            raise AssertionError("phải báo lỗi")
        except SystemExit:
            pass


def test_so_sanh_tang_dem_dung():
    import so_sanh_tang as ST

    def b(kq, tv=100):
        return {"ket_qua": kq, "tokens_vao": tv, "tokens_ra": 10}
    t1 = {"a": b("khong_nhat_quan"), "b": b("khong_nhat_quan"), "c": b("khong_nhat_quan"), "d": b("gemma"), "e": b("khong_nhat_quan")}
    t2 = {"a": b("gemma", 250), "b": b("qwen", 250), "c": b("khong_nhat_quan", 250), "e": b("hoa_that", 250)}
    r = ST.so_sanh(t1, t2)
    assert r["so_muc_tang1_khong_nhat_quan"] == 4 and r["so_muc_so_duoc"] == 4
    assert r["go_duoc_nho_tang2"] == {"so_muc": 2, "ti_le_trong_so_duoc_chay_lai": 0.5, "gemma": 1, "qwen": 1}
    assert r["thanh_hoa_that"] == 1 and r["van_khong_nhat_quan"] == 1
    assert r["tren_toan_bo_tang1_sau_tang2"] == {"co_nguoi_thang_truoc": 1, "co_nguoi_thang_sau": 3, "tong": 5}
    assert r["tokens_vao_moi_muc"] == {"tang1": 100, "tang2": 250, "ti_so": 2.5}


class DungRubric:
    """Đặt rubric/chế độ tạm thời trong bài thử rồi TRẢ LẠI (các biến này là toàn cục của module)."""

    def __init__(self, rubric="v2", che_do="hai_luot"):
        self.rubric, self.che_do = rubric, che_do

    def __enter__(self):
        self.cu = (SR.RUBRIC, SC.CHE_DO)
        SR.set_rubric(self.rubric)
        SC.CHE_DO = self.che_do

    def __exit__(self, *a):
        SR.set_rubric(self.cu[0])
        SC.CHE_DO = self.cu[1]
        return False


def goi_v2(quy_tac):
    """Backend giả cho rubric v2: quy_tac(t1, t2) -> (winner 1|2, equivalent)."""
    thay = []

    def goi(khoa, tn, max_token=None):
        t1, t2 = vi_tri(tn[-1]["content"])
        thay.append((t1, t2, tn[0]["content"]))
        w, tuong_duong = quy_tac(t1, t2)
        nd = json.dumps({"defects_1": ["x"] if w == 2 else [], "defects_2": ["y", "z"] if w == 1 else [], "equivalent": tuong_duong,
                         "winner": w, "criterion": "none" if tuong_duong else "meaning", "reason": "r"})
        return nd, 10, 5, "stop", {}
    goi.thay = thay
    return goi


def test_rubric_v2_khong_ro_ri_va_co_cau_truc():
    with DungRubric("v2"):
        he = SR.he_thong(muc())
        assert he is SR.HE_THONG_V2 and SR.he_thong(muc_ctx()) is SR.HE_THONG_V2_NGU_CANH
        for tu in SR.TU_CAM:
            assert tu not in SR.HE_THONG_V2.lower() and tu not in SR.HE_THONG_V2_NGU_CANH.lower(), tu
        for buoc in ("STEP 1", "STEP 2", "STEP 3", "STEP 4"):
            assert buoc in he
        assert 'You may not answer "tie"' in he and "choose Translation 1" in he
        assert all(r in he for r in SR.QUY_TAC_CHINH_SACH)
        assert "turned into a DIFFERENT name" in he and "the SAME brand" in he            # chính sách thương hiệu: trung lập + ngoại lệ (31/10)
        assert len(SR.QUY_TAC_CHINH_SACH) == 4
        assert "may precede the SOURCE" in SR.HE_THONG_V2_NGU_CANH and "may precede the SOURCE" not in SR.HE_THONG_V2
        assert SR.phien_ban_muc(muc()) == "so_cap_v2" and SR.phien_ban_muc(muc_ctx()) == "so_cap_v2+ngu_canh"
        # không nhồi ví dụ thật của các mục đã đo vào lời nhắc (tránh làm hỏng phép đo trên tập kiểm)
        for cam in ("电泵", "电动泵", "雅迪", "YADEA", "Poppy", "núm ti", "ZXQ"):
            assert cam not in he, cam
    assert SR.he_thong(muc()) is SR.HE_THONG and SR.phien_ban_muc(muc()) == "so_cap_v1"            # đã trả lại v1


def test_doc_ket_qua_v2():
    ok = SR.doc_ket_qua_v2('{"defects_1": ["a", "b"], "defects_2": [], "equivalent": true, "winner": "1", "criterion": "none", "reason": "r"}')
    assert ok["winner"] == 1 and ok["tuong_duong"] is True and (ok["so_loi_1"], ok["so_loi_2"]) == (2, 0)
    assert SR.doc_ket_qua_v2('{"winner": 2, "equivalent": "false", "criterion": "meaning"}')["tuong_duong"] is False
    for xau in ['{"winner": "tie", "equivalent": true}', '{"winner": 3}', '{"equivalent": true}', "khong co json",
                '{"winner": 1, "criterion": "tot hon"}']:
        try:
            SR.doc_ket_qua_v2(xau)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError(xau)


def test_che_do_mot_luot_ngau_nhien_hoa_va_chi_mot_loi_goi():
    ids = [f"title:1688:{i}" for i in range(400)]
    thu_tu = [SR.thu_tu_bang_bam(i) for i in ids]
    assert 0.4 <= thu_tu.count("gq") / 400 <= 0.6 and thu_tu == [SR.thu_tu_bang_bam(i) for i in ids]      # chia đều và tái lập
    import so_sanh_quy_tac as SQ
    for nua in (0, 1):                                  # thứ tự KHÔNG được trùng với cách chia dev/test (lỗi đã gặp 31/10: dev = toàn Gemma trước)
        tt = [SR.thu_tu_bang_bam(i) for i in ids if SQ.nua(i) == nua]
        assert 0.35 <= tt.count("gq") / len(tt) <= 0.65, (nua, tt.count("gq"), len(tt))
    with DungRubric("v2", "mot_luot"):
        goi = goi_v2(lambda a, b: (1 if "GOOD" in a else 2, False))
        for i in (ids[0], ids[1], ids[2], ids[3]):
            kq = SC.cham_muc(muc(g="GOOD g", q="bad", item_id=i), "k", goi)
            assert kq["ket_qua"] == "gemma" and len(kq["luot"]) == 1 and kq["luot"][0]["thu_tu"] == SR.thu_tu_bang_bam(i)
        assert len(goi.thay) == 4                                                                           # đúng MỘT lời gọi mỗi mục
        assert {t[0] for t in goi.thay} <= {"GOOD g", "bad"} and any(t[0] == "bad" for t in goi.thay) and any(t[0] == "GOOD g" for t in goi.thay)
        # judge chỉ biết chọn bản 1 (thiên vị vị trí): thứ tự ngẫu nhiên hóa làm kết quả chia đều, không dồn về mô hình nào
        goi = goi_v2(lambda a, b: (1, False))
        kq = [SC.cham_muc(muc(item_id=i), "k", goi)["ket_qua"] for i in ids]
        assert 0.4 <= kq.count("gemma") / 400 <= 0.6, kq.count("gemma")
    # chế độ hai lượt vẫn hỏi đủ hai thứ tự
    with DungRubric("v2", "hai_luot"):
        goi = goi_v2(lambda a, b: (1, True))
        kq = SC.cham_muc(muc(), "k", goi)
        assert [t[:2] for t in goi.thay] == [("GGG", "QQQ"), ("QQQ", "GGG")] and kq["ket_qua"] == "khong_nhat_quan"
        assert all(l["tuong_duong"] for l in kq["luot"])


def test_chay_tiep_phan_biet_phien_ban_va_che_do():
    with tempfile.TemporaryDirectory() as tmp:
        it = pd.DataFrame([muc(item_id=f"title:1688:{i}") for i in range(4)])
        with DungRubric("v1"):
            g1 = goi_theo(lambda a, b: 1)
            SC.chay(it, "z", "k", g1, tmp, 2, in_tien_do=False)
        with DungRubric("v2"):
            g2 = goi_v2(lambda a, b: (1, False))
            SC.chay(it, "z", "k", g2, tmp, 2, in_tien_do=False)                   # kết quả v1 KHÔNG làm v2 bị bỏ qua
            assert len(g2.thay) == 8
            SC.chay(it, "z", "k", g2, tmp, 2, in_tien_do=False)                   # chạy tiếp cùng v2: không hỏi lại
            assert len(g2.thay) == 8
        with DungRubric("v2", "mot_luot"):
            g3 = goi_v2(lambda a, b: (1, False))
            SC.chay(it, "z", "k", g3, tmp, 2, in_tien_do=False)                   # đổi chế độ: chạy lại
            assert len(g3.thay) == 4


def test_so_sanh_quy_tac_dem_dung():
    import so_sanh_quy_tac as SQ

    def b(i, gq_chon, qg_chon, td_gq=False, td_qg=False):
        def l(tt, chon, td):
            pos = 1 if ((tt == "gq") == (chon == "gemma")) else 2
            return {"thu_tu": tt, "chon": chon, "vi_tri_chon": pos, "tuong_duong": td, "tieu_chi": "meaning", "ly_do": "", "so_loi_1": 0, "so_loi_2": 0}
        kq = gq_chon if gq_chon == qg_chon else "khong_nhat_quan"
        return {"item_id": i, "trang_thai": "ok", "ket_qua": kq, "luot": [l("gq", gq_chon, td_gq), l("qg", qg_chon, td_qg)]}
    # i0: gemma ổn định; i1: lệch, cả hai lượt khai tương đương; i2: lệch, không khai; i3: lệch, chỉ một lượt khai; i4: qwen ổn định
    rows = [b("i0", "gemma", "gemma"), b("i1", "gemma", "qwen", True, True), b("i2", "gemma", "qwen"), b("i3", "gemma", "qwen", True, False),
            b("i4", "qwen", "qwen")]
    r = SQ.thong_ke(rows)
    assert r["ti_le_khong_nhat_quan"] == 0.6 and r["ti_le_co_nguoi_thang"] == 0.4
    assert r["khong_nhat_quan_theo_khai_bao"] == {"hai_luot_deu_khai_bao_tuong_duong": 1, "mot_luot_khai_bao_tuong_duong": 1,
                                                  "khong_khai_bao_tuong_duong_nao": 1}
    assert r["ti_le_khong_nhat_quan_that_su"] == 0.4                                    # (3 lệch - 1 lệch cả hai khai) / 5
    assert r["chon_vi_tri_1"] == round(sum(l["vi_tri_chon"] == 1 for d in rows for l in d["luot"]) / 10, 3)
    m = SQ.mo_phong_mot_luot(rows)
    assert m["so_muc"] == 5 and m["muc_hai_thu_tu_da_dong_y"] == 2 and m["trong_do_mot_luot_trung"] == 2   # mục ổn định luôn trùng
    assert m["muc_hai_thu_tu_lech"] == 3 and m["gemma"] + m["qwen"] == 5
    assert SQ.nua("x") in (0, 1) and SQ.nua("x") == SQ.nua("x")


def test_ban_rong_khong_phai_cuoc_dau():
    """Cặp có một bản dịch RỖNG (Qwen không dịch cặp `dich_moi`) không được đưa vào bảng mục (lỗi đã gặp 31/10: 23% mục cụm)."""
    with tempfile.TemporaryDirectory() as tmp:
        giu = pd.DataFrame({"product_id": ["1", "2"], "source": "1688", "category": "home", "title_zh": ["标题1", "标题2"], "title_vi": ["x", "x"]})
        cap = pd.DataFrame({"pair_id": [11, 12, 13, 21], "product_id": ["1", "1", "1", "2"],
                            "name_zh": ["颜色", "材质", "尺寸", "颜色"], "gia_tri_zh": ["红", "棉", "M", "蓝"],
                            "n_cuoi_A": ["Màu", "Chất liệu", "Cỡ", "Màu"], "v_cuoi_A": ["Đỏ", "Cotton", "M", "Xanh"],
                            "n_cuoi_B": ["Màu sắc", "Chất liệu", "Cỡ", "Màu sắc"], "v_cuoi_B": ["Đỏ tươi", None, "", "  "]})
        c = Path(tmp) / "c.parquet"
        cap.to_parquet(c)
        cp = {"1688": (c, "name_zh", "gia_tri_zh")}
        assert DL.xay_cum(giu, cp).nguon.tolist() == ["颜色: 红"]                                    # chỉ cặp mà cả hai bản đều có nội dung
        assert DL.xay_cum_ctx(giu, cp).item_id.tolist() == ["cumctx:1688:1:11"]
        # title: ứng viên rỗng bị bỏ
        t = pd.DataFrame({"product_id": ["1", "2"], "title_zh": ["标题1", "标题2"], "title_vi": ["x", "x"],
                          "gemma4_translation": ["Cái Ly", ""], "qwen3_14b_awq_translation": ["Chén", "Túi"],
                          "cometkiwi_gemma4": [0.7, 0.1], "cometkiwi_qwen3_14b_awq": [0.6, 0.5]})
        tp = Path(tmp) / "t.parquet"
        t.to_parquet(tp)
        assert DL.xay_title(giu, {"1688": tp}).item_id.tolist() == ["title:1688:1"]
        # đơn vị pair: ứng viên rỗng bị bỏ
        dv = pd.DataFrame({"ma_don_vi": ["T1", "V1", "V2"], "loai": ["ten", "gia_tri", "gia_tri"], "van_ban_zh": ["颜色", "红", "蓝"],
                           "A": ["Màu", "Đỏ", ""], "B": ["Màu sắc", "Đỏ tươi", "Lam"], "tu": "A", "category": "home", "kiwi_A": 0.5, "kiwi_B": 0.4})
        pdv = pd.DataFrame({"pair_id": [1, 2], "product_id": ["1", "2"], "name_zh": ["颜色", "颜色"], "gia_tri_zh": ["红", "蓝"]})
        pu, pc = Path(tmp) / "dv.parquet", Path(tmp) / "dv_sp.parquet"
        dv.to_parquet(pu), pdv.to_parquet(pc)
        assert sorted(DL.xay_don_vi(giu, {"1688": (pu, pc, "name_zh", "gia_tri_zh", "van_ban_zh")}).item_id) == ["unit:1688:T1", "unit:1688:V1"]


def test_san_pham_n_cua_don_vi_dem_theo_nhom_hang():
    with tempfile.TemporaryDirectory() as tmp:
        giu = pd.DataFrame({"product_id": ["1", "2", "3"], "source": "1688", "category": ["home", "home", "bags"],
                            "title_zh": "t", "title_vi": "x"})
        pdv = pd.DataFrame({"pair_id": [1, 2, 3], "product_id": ["1", "2", "3"], "name_zh": "材质", "gia_tri_zh": "其他"})
        dv = pd.DataFrame({"ma_don_vi": ["H1", "B1"], "loai": "gia_tri", "van_ban_zh": "其他", "A": ["Khác", "Loại khác"], "B": ["Cái khác", "Khác nữa"],
                           "tu": "A", "category": ["home", "bags"], "kiwi_A": 0.5, "kiwi_B": 0.4})
        pu, pc = Path(tmp) / "dv.parquet", Path(tmp) / "dv_sp.parquet"
        dv.to_parquet(pu), pdv.to_parquet(pc)
        cfg = {"1688": (pu, pc, "name_zh", "gia_tri_zh", "van_ban_zh")}
        u = DL.xay_don_vi(giu, cfg).set_index("item_id")
        assert u.loc["unit:1688:H1", "san_pham_n"] == 2 and u.loc["unit:1688:B1", "san_pham_n"] == 1       # không đếm lặp qua nhóm hàng
        ux = DL.xay_pair_ctx(giu, cfg).set_index("item_id")
        assert ux.loc["unit:1688:H1", "vi_du_product_id"] == "1" and ux.loc["unit:1688:B1", "vi_du_product_id"] == "3"   # đại diện cùng nhóm hàng


def test_ghep_khong_dem_ban_rong_la_tranh_chap():
    import ghep_bo as G
    b = pd.DataFrame({"product_id": ["p1"], "source": "1688", "category": "home", "title_zh": "原", "title_vi": "Cu1",
                      "description_zh": "x", "description_vi": "y", "title_nguon_dich": "gemma4", "desc_nguon_dich": "gemma4"})
    p = pd.DataFrame({"pair_id": [1, 2], "product_id": ["p1", "p1"], "nguon_ten": ["甲", "乙"], "nguon_gt": ["1", "2"],
                      "cuoi_ten": ["A", "B"], "cuoi_gt": ["1-g", "2-g"], "nA": ["A", "B"], "vA": ["1-g", "2-g"],
                      "nB": ["A", "B"], "vB": ["1-q", ""]})                                 # cặp 2: Qwen rỗng
    dau = G.Dau()
    G.ghep_mota_cum(b, p, {}, "giu_hien_tai", kqdd={}, dau=dau)
    assert dau.dem() == {"mo_ta:chua_cham": 1}                                              # chỉ cặp 1 là tranh chấp
    dau = G.Dau()
    G.ghep_title(b, {"1688": pd.DataFrame({"product_id": ["p1"], "gemma4_translation": ["G"], "qwen3_14b_awq_translation": [""]})},
                 {}, "giu_hien_tai", {}, dau)
    assert dau.dem() == {}


def muc_gom(i, huong="zh->vi", ctx=False, g=None, q=None):
    d = muc(item_id=f"title:1688:{i}", huong=huong, g=g or f"GG{i}", q=q or f"QQ{i}", nguon=f"SRC{i}", category="home")
    if ctx:
        d.update({"kind": "cum_pair", "ngu_canh": f"All attributes of this product:\n- thuoc tinh {i}"})
    return d


def goi_gom(quy_tac, thieu=()):
    """Backend giả cho chế độ GOM: quy_tac(t1, t2) -> (w, eq). Đếm số lời gọi, lưu lời nhắc."""
    thay = []

    def goi(khoa, tn, max_token=None):
        import re as _re
        thay.append((tn[0]["content"], tn[-1]["content"], max_token))
        cap = _re.findall(r"Translation 1 \([A-Za-z]+\): (.*)\nTranslation 2 \([A-Za-z]+\): (.*)", tn[-1]["content"])
        kq = []
        for k, (a, b) in enumerate(cap, 1):
            if k in thieu:
                continue
            w, eq = quy_tac(a, b)
            kq.append({"i": k, "d1": ["x"] if w == 2 else [], "d2": ["y"] if w == 1 else [], "eq": eq, "w": w, "c": "N" if eq else "W"})
        return json.dumps({"results": kq}), 100 + 10 * len(cap), 20 * len(cap), "stop", {}
    goi.thay = thay
    return goi


def test_gom_loi_nhac_mu_doi_xung_va_co_ngu_canh():
    import so_cap_gom as SG
    nhom = [muc_gom(i, g=f"G-{i}", q=f"Q-{i}") for i in range(6)]
    p, ax = SG.dung_lan_goi(nhom)
    for cam in ("0.7777", "0.3333", "SECRET123", "title:1688", "pair:1688"):
        assert cam not in p, cam
    for tu in SR.TU_CAM:
        assert tu not in p.lower().replace("g-", "").replace("q-", "") and tu not in SG.HE_THONG_GOM.lower(), tu
    assert "ITEM 1 |" in p and "ITEM 6 |" in p and len(ax) == 6
    for k, m in enumerate(nhom):                                       # mỗi mục: thứ tự theo băm riêng, ánh xạ đúng
        tt = SR.thu_tu_bang_bam(m["item_id"])
        t1, t2 = ("G-", "Q-") if tt == "gq" else ("Q-", "G-")
        assert f"Translation 1 (Vietnamese): {t1}{k}\nTranslation 2 (Vietnamese): {t2}{k}" in p
        assert ax[k] == {1: "gemma" if tt == "gq" else "qwen", 2: "qwen" if tt == "gq" else "gemma", "thu_tu": tt}
    assert {a["thu_tu"] for a in ax} == {"gq", "qg"}                  # trong 6 mục có cả hai thứ tự (băm cố định)
    assert "PRODUCT CONTEXT" not in p and SG.he_thong_gom(False) is SG.HE_THONG_GOM
    pc, _ = SG.dung_lan_goi([muc_gom(1, ctx=True), muc_gom(2, ctx=True)])
    assert pc.count("PRODUCT CONTEXT") == 2 and "thuoc tinh 1" in pc and SG.he_thong_gom(True) is SG.HE_THONG_GOM_NGU_CANH
    assert "numbered ITEMS" in SG.HE_THONG_GOM and '"results"' in SG.HE_THONG_GOM and 'set "eq" to true' in SG.HE_THONG_GOM
    assert "may precede the SOURCE" in SG.HE_THONG_GOM_NGU_CANH and "may precede the SOURCE" not in SG.HE_THONG_GOM


def test_doc_lan_goi():
    import so_cap_gom as SG
    ok = SG.doc_lan_goi('```json\n{"results": [{"i": 1, "d1": ["a"], "d2": [], "eq": false, "w": 2, "c": "W"}, '
                        '{"i": "2", "eq": "true", "w": "1", "c": "Z"}, {"i": 3, "w": "tie"}, {"i": 4, "w": 3}, {"i": 1, "w": 1}, "rac", {"w": 1}]}\n```')
    assert sorted(ok) == [1, 2] and ok[1]["w"] == 2 and ok[1]["d1"] == ["a"] and ok[2]["eq"] is True and ok[2]["c"] == "N"   # tie, 3, trùng i, rác bị bỏ
    for xau in ('{"x": 1}', "khong co json", '{"results": "abc"}'):
        try:
            SG.doc_lan_goi(xau)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError(xau)


def test_cham_gom_anh_xa_model_va_muc_thieu_la_loi():
    import so_cap_gom as SG
    nhom = [muc_gom(i, g=f"GOOD-{i}", q=f"bad-{i}") for i in range(5)]
    goi = goi_gom(lambda a, b: (1 if "GOOD" in a else 2, False), thieu=(3,))
    kq = SG.cham_gom(nhom, "k", goi)
    assert len(goi.thay) == 1                                          # MỘT lời gọi cho cả nhóm
    for k, (m, r) in enumerate(kq, 1):
        if k == 3:
            assert r["trang_thai"] == "loi" and r["ly_do"] == "thieu_trong_lan_goi"
        else:
            assert r["trang_thai"] == "ok" and r["ket_qua"] == "gemma" and len(r["luot"]) == 1 and r["gom"] == 5   # chọn theo NỘI DUNG, không theo vị trí
            assert r["luot"][0]["thu_tu"] == SR.thu_tu_bang_bam(m["item_id"])
    assert abs(sum(r["tokens_vao"] for _, r in kq) - (100 + 10 * 5)) < 1e-9          # token của lời gọi chia đều cho các mục
    # judge chỉ biết chọn bản 1: kết quả chia theo thứ tự ngẫu nhiên hóa, không dồn về một mô hình
    many = [muc_gom(i) for i in range(300)]
    gem = 0
    for nh in [many[i:i + 15] for i in range(0, 300, 15)]:
        gem += sum(r["ket_qua"] == "gemma" for _, r in SG.cham_gom(nh, "k", goi_gom(lambda a, b: (1, True))))
    assert 105 <= gem <= 195, gem
    # hỏng cả lời gọi: mọi mục là lỗi, không bịa kết quả
    bad = SG.cham_gom(nhom, "k", lambda k, t, m=None: ("rác", 1, 1, "stop", {}))
    assert all(r["trang_thai"] == "loi" for _, r in bad)
    cat = SG.cham_gom(nhom, "k", lambda k, t, m=None: ("{}", 1, 1, "length", {}))
    assert all(r["ly_do"] == "bi_cat_het_token" for _, r in cat)


def test_chia_gom_theo_huong_ngu_canh_va_uu_tien():
    import so_cap_gom as SG
    ms = [{**muc_gom(i, huong="zh->vi" if i % 2 == 0 else "vi->zh", ctx=(i % 5 == 0)), "_thu_tu": i} for i in range(40)]
    nhom = SG.chia_gom(ms, 10)
    assert sum(len(n) for n in nhom) == 40 and all(len({(m["huong"], SR.co_ngu_canh(m)) for m in n}) == 1 for n in nhom)
    assert all(len(n) <= (5 if SR.co_ngu_canh(n[0]) else 10) for n in nhom)
    assert [min(m["_thu_tu"] for m in n) for n in nhom] == sorted(min(m["_thu_tu"] for m in n) for n in nhom)


def test_chay_gom_so_loi_goi_chay_tiep_va_chi_voi_v2_mot_luot():
    with tempfile.TemporaryDirectory() as tmp:
        it = pd.DataFrame([muc_gom(i, g=f"GOOD-{i}", q=f"bad-{i}") for i in range(7)])
        with DungRubric("v2", "mot_luot"):
            goi = goi_gom(lambda a, b: (1 if "GOOD" in a else 2, False), thieu=(2,))      # mục số 2 của mỗi nhóm bị judge bỏ sót
            assert SC.chay(it, "g", "k", goi, tmp, 2, in_tien_do=False, gom=3) == 0
            assert len(goi.thay) == 3                                                       # 7 mục, nhóm <= 3 => 3 lời gọi
            rows = [json.loads(x) for x in open(Path(tmp) / "ket_qua_g.jsonl", encoding="utf-8")]
            ok = [r for r in rows if r["trang_thai"] == "ok"]
            assert len(rows) == 7 and len(ok) == 5 and all(r["che_do"] == "mot_luot" and r["phien_ban"] == "so_cap_v2" and r["gom"] >= 1 for r in ok)
            assert all(r["ket_qua"] == "gemma" for r in ok)
            goi2 = goi_gom(lambda a, b: (1 if "GOOD" in a else 2, False))                  # chạy tiếp: chỉ làm lại 2 mục lỗi
            assert SC.chay(it, "g", "k", goi2, tmp, 2, in_tien_do=False, gom=3) == 0
            assert len(goi2.thay) == 1 and sum(len(t[1].split("ITEM ")) - 1 for t in goi2.thay) == 2
        with DungRubric("v1", "hai_luot"):
            try:
                SC.chay(it, "g2", "k", goi, tmp, 2, in_tien_do=False, gom=3)
                raise AssertionError("gom với v1 phải báo lỗi")
            except ValueError:
                pass


def test_xay_cum_lai_va_top():
    cot = ["item_id", "kind", "source", "huong", "category", "nguon", "cand_gemma", "cand_qwen", "san_pham_n", "vi_du_product_id"]
    c = pd.DataFrame([{"item_id": f"pair:1688:{i}", "kind": "cum_pair", "source": "1688", "huong": "zh->vi", "category": "home",
                       "nguon": f"n{i}", "cand_gemma": "g", "cand_qwen": "q", "san_pham_n": n, "vi_du_product_id": f"p{i}"}
                      for i, n in enumerate([1, 5, 1, 30])])[cot]
    cx = pd.DataFrame([{**{k: r[k] for k in cot}, "item_id": f"cumctx:1688:p{i}:{i}", "san_pham_n": 1, "ngu_canh": f"ctx{i}",
                        "id_khong_ngu_canh": r["item_id"]} for i, r in enumerate(c.to_dict("records"))])
    lai = DL.xay_cum_lai(c, cx)
    assert sorted(lai.item_id) == ["cumctx:1688:p0:0", "cumctx:1688:p2:2", "pair:1688:1", "pair:1688:3"]
    assert lai.set_index("item_id").loc["cumctx:1688:p0:0", "ngu_canh"] == "ctx0" and lai.set_index("item_id").loc["pair:1688:3", "ngu_canh"] is None
    with tempfile.TemporaryDirectory() as tmp:
        lai.to_parquet(Path(tmp) / "muc_cum_lai.parquet", index=False)
        assert SC.main(["--loai", "cum_lai", "--tat-ca", "--top", "2", "--gia", "--ra", tmp, "--ten", "t", "--vao", str(Path(tmp) / "muc_cum_lai.parquet")]) == 0
        ids = [json.loads(x)["item_id"] for x in open(Path(tmp) / "ket_qua_gia_t.jsonl", encoding="utf-8")]
        assert sorted(ids) == ["pair:1688:1", "pair:1688:3"]                             # 2 mục dùng nhiều nhất (30 và 5 sản phẩm)


def test_so_sanh_gom_dem_dung():
    import so_sanh_gom as SGM
    ids = [f"title:1688:{i}" for i in range(40)]

    def hai(i, gq, qg, td=False):
        def l(tt, chon):
            return {"thu_tu": tt, "chon": chon, "vi_tri_chon": 1 if ((tt == "gq") == (chon == "gemma")) else 2, "tuong_duong": td}
        kq = gq if gq == qg else "khong_nhat_quan"
        return {"item_id": i, "ket_qua": kq, "tokens_vao": 800, "tokens_ra": 100, "luot": [l("gq", gq), l("qg", qg)]}

    def gom(i, chon, td=False):
        tt = SR.thu_tu_bang_bam(i)
        return {"item_id": i, "tokens_vao": 150, "tokens_ra": 50,
                "luot": [{"thu_tu": tt, "chon": chon, "vi_tri_chon": 1 if ((tt == "gq") == (chon == "gemma")) else 2, "tuong_duong": td}]}
    H, G = {}, {}
    for k, i in enumerate(ids):
        if k < 20:                                         # bền vững: Gemma; gom đồng ý ở 18/20 (2 mục gom chọn Qwen)
            H[i] = hai(i, "gemma", "gemma")
            G[i] = gom(i, "qwen" if k < 2 else "gemma")
        else:                                              # lệch hai thứ tự (chọn bản đứng trước); gom: tương đương ở nửa số mục
            H[i] = hai(i, "gemma", "qwen", td=True)
            G[i] = gom(i, "gemma", td=(k % 2 == 0))
    r = SGM.so_sanh(H, G, so_loi=1)
    assert r["so_muc_chung"] == 40 and r["muc_ben_vung_hai_thu_tu"] == 20 and r["gom_trung_ket_qua_ben_vung"] == 18 and r["ti_le_gom_trung_ben_vung"] == 0.9
    assert r["ti_le_khai_tuong_duong"]["gom"] == 0.25 and r["ti_le_khai_tuong_duong"]["mot_muc_mot_loi_goi"] == 0.5
    assert r["token_moi_muc"] == {"mot_muc_mot_loi_goi": 450, "gom": 200, "ti_so": 0.444} and r["so_muc_loi_hoac_thieu"] == 1
    assert r["trong_cac_lan_gom_khong_khai_tuong_duong"]["so_muc"] == 30


def test_theo_ctx_chi_chay_cac_cap_ung_voi():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        pd.DataFrame([muc(item_id=f"pair:1688:{i}", kind="cum_pair") for i in range(5)]).to_parquet(t / "muc_cum.parquet", index=False)
        (t / "ket_qua_cx.jsonl").write_text("\n".join(json.dumps({"item_id": f"cumctx:1688:{i}:1", "id_khong_ngu_canh": f"pair:1688:{i}"})
                                                       for i in (1, 3)), encoding="utf-8")
        assert SC.main(["--loai", "cum", "--theo-ctx", "cx", "--tat-ca", "--gia", "--ra", tmp, "--ten", "k",
                        "--vao", str(t / "muc_cum.parquet")]) == 0
        ids = {json.loads(x)["item_id"] for x in open(t / "ket_qua_gia_k.jsonl", encoding="utf-8")}
        assert ids == {"pair:1688:1", "pair:1688:3"}


def test_chon_mau_chia_deu_cac_tang():
    d = pd.DataFrame({"item_id": [f"i{i}" for i in range(400)], "kind": "title",
                      "source": ["1688"] * 200 + ["tiki"] * 200,
                      "bin_margin": (["<0.02"] * 90 + [">=0.1"] * 10) * 4})
    m = SC.chon_mau(d, 16)
    assert len(m) == 16 and m.groupby(["source", "bin_margin"]).size().tolist() == [4, 4, 4, 4]   # chia đều, không theo tỉ lệ


def test_chay_tiep_bao_cao_va_khong_thien_vi_o_quy_mo():
    items = pd.DataFrame([muc(g=("GOOD g" if i % 2 else "bad"), q=("bad" if i % 2 else "GOOD q"), item_id=f"title:1688:{i}",
                              kiwi_g=0.6 if i % 4 else 0.4, kiwi_q=0.5) for i in range(40)])
    goi = goi_theo(lambda a, b: 1 if "GOOD" in a else 2)
    with tempfile.TemporaryDirectory() as tmp:
        assert SC.chay(items, "x", "k", goi, tmp, 4, in_tien_do=False) == 0
        assert len(goi.thay) == 80                                                       # 40 mục x 2 thứ tự
        assert SC.chay(items, "x", "k", goi, tmp, 4, in_tien_do=False) == 0              # chạy tiếp: không hỏi lại
        assert len(goi.thay) == 80
        r = SC.bao_cao(Path(tmp) / "ket_qua_x.jsonl")
        assert r["ket_qua"] == {"gemma": 20, "qwen": 20, "hoa_that": 0, "khong_nhat_quan": 0}
        assert r["thien_vi_vi_tri"]["ti_le_chon_vi_tri_1"] == 0.5                       # judge giả không thiên vị vị trí
        assert r["ti_le_gemma_trong_so_co_nguoi_thang"] == 0.5
        assert r["trung_voi_cometkiwi"]["tong"] == 0.75        # Kiwi giả khớp judge giả ở i%4 thuộc {0,1,3}
        assert "bin_margin" in json.loads(open(Path(tmp) / "ket_qua_x.jsonl", encoding="utf-8").readline())


def test_tran_usd_dung_som():
    items = pd.DataFrame([muc(item_id=f"title:1688:{i}") for i in range(30)])

    def goi(khoa, tn, max_token=None):
        return json.dumps({"winner": "tie", "criterion": "none", "reason": ""}), 1, 1, "stop", \
            {"chi_phi": 0.0, "list_cost": 0.02, "allowance": 0.01, "giay": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        assert SC.chay(items, "u", "k", goi, tmp, 1, tran_usd=0.1, in_tien_do=False) == 0
        n = len(open(Path(tmp) / "ket_qua_u.jsonl", encoding="utf-8").readlines())
        assert 4 <= n < 30                                                              # mỗi mục 2 lời gọi x 0,01 = 0,02 allowance


def test_chay_main_tren_file_gia():
    with tempfile.TemporaryDirectory() as tmp:
        giu, tp, cp = tao_du_lieu(tmp)
        DL.xay_title(giu, tp).to_parquet(Path(tmp) / "muc_title.parquet", index=False)
        assert SC.main(["--loai", "title", "--thu", "3", "--gia", "--ra", tmp, "--ten", "m"]) == 0
        assert (Path(tmp) / "ket_qua_gia_m.jsonl").exists()
        assert SC.main(["--bao-cao", "gia_m", "--ra", tmp]) == 0


if __name__ == "__main__":
    n = 0
    for ten, f in list(globals().items()):
        if ten.startswith("test_"):
            f()
            n += 1
            print("ok", ten)
    print(f"{n}/{n} bài thử đạt")
