"""Thử máy móc của judge bằng bản giả (không gọi API, không cần khóa). Chạy: python -I -X utf8 test_judge.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import chay_judge as C  # noqa: E402
import luat_kiem  # noqa: E402
import rubric  # noqa: E402


def hang(nguon="1688", zh="材质: PU; 重量: 2斤; 颜色: 黑色", vi="Chất liệu: PU; Trọng lượng: 1kg; Màu sắc: Đen", pid="p1"):
    return {"product_id": pid, "source": nguon, "category": "bags", "title_zh": "包", "title_vi": "Túi",
            "description_zh": zh, "description_vi": vi, "desc_nguon_dich": "gemma4"}


def test_doc_ket_qua():
    tot = {"sai_nghia": 5, "thieu_y": 5, "them_y": 4, "thuat_ngu": 5, "so_don_vi": 5, "troi_chay": 5,
           "ket_luan": "dat", "loi": []}
    assert rubric.doc_ket_qua(json.dumps(tot))["them_y"] == 4
    assert rubric.doc_ket_qua("```json\n" + json.dumps(tot) + "\n```")["ket_luan"] == "dat"
    assert rubric.doc_ket_qua("Đây là kết quả: " + json.dumps(tot) + " hết")["troi_chay"] == 5
    chuoi_co_ngoac = dict(tot, ket_luan="loi", loi=[{"loai": "sai_nghia", "doan_nguon": "a}b{", "doan_dich": "x", "goi_y": "y"}])
    assert rubric.doc_ket_qua(json.dumps(chuoi_co_ngoac))["loi"][0]["doan_nguon"] == "a}b{"
    for xau in [dict(tot, sai_nghia=6), dict(tot, sai_nghia=True), dict(tot, ket_luan="tot"), dict(tot, loi="x")]:
        try:
            rubric.doc_ket_qua(json.dumps(xau))
        except ValueError:
            continue
        raise AssertionError(f"phải từ chối: {xau}")
    for xau in ["", "không có json", '{"sai_nghia": 5']:
        try:
            rubric.doc_ket_qua(xau)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError(f"phải từ chối: {xau!r}")


def test_huong_dich():
    u1688 = rubric.dung_nguoi_dung(hang("1688"))
    assert "SOURCE is Chinese" in u1688 and "SOURCE (Chinese):\n材质" in u1688
    utiki = rubric.dung_nguoi_dung(hang("tiki", zh="品牌: A", vi="Thương hiệu: A"))
    assert "SOURCE is Vietnamese" in utiki and "SOURCE (Vietnamese):\nThương hiệu" in utiki
    assert "TRANSLATION (Chinese):\n品牌" in utiki


def test_luat():
    ok = luat_kiem.kiem(hang())
    assert ok["so_doan_lech"] is False and ok["han_sot"] == 0 and ok["lech_so"] is True  # 2 斤 -> 1 kg
    xau = luat_kiem.kiem(hang(vi="Chất liệu: PU 材质"))
    assert xau["han_sot"] == 2 and xau["so_doan_lech"] is True and xau["co_luat"] is True
    assert luat_kiem.kiem(hang(vi=""))["rong"] is True
    mp = luat_kiem.kiem(hang(vi="Số đăng ký: 粤妆网备字2023123456; 材质: PU; x"))   # số đăng ký mỹ phẩm là hợp lệ
    assert mp["han_sot"] == 2   # chỉ tính "材质"


def test_chon_mau_va_chay_tiep():
    df = pd.DataFrame([hang(s, pid=f"{s}{c}{i}") | {"category": c, "desc_nguon_dich": "gemma4"}
                       for s in ("1688", "tiki") for c in ("bags", "home") for i in range(30)]).reset_index(drop=True)
    m = C.chon_mau(df, 20)
    assert len(m) == 20 and set(m.source) == {"1688", "tiki"} and set(m.category) == {"bags", "home"}
    assert m.product_id.tolist() == C.chon_mau(df, 20).product_id.tolist()      # lặp lại được
    with tempfile.TemporaryDirectory() as tmp:
        assert C.chay(m, "t", "gia", C.goi_gia, tmp, 3, in_tien_do=False) == 0
        dong = [json.loads(x) for x in open(Path(tmp) / "ket_qua_t.jsonl", encoding="utf-8")]
        assert len(dong) == 20 and all(d["trang_thai"] == "ok" and d["mo_hinh"] == C.CH.MO_HINH for d in dong)
        assert C.chay(m, "t", "gia", C.goi_gia, tmp, 3, in_tien_do=False) == 0   # chạy tiếp: không chấm lại
        assert len(open(Path(tmp) / "ket_qua_t.jsonl", encoding="utf-8").readlines()) == 20
        # trần token: dừng sớm, rồi chạy tiếp được
        assert C.chay(m, "u", "gia", C.goi_gia, tmp, 1, tran_token=1, in_tien_do=False) == 0
        so_u = len(open(Path(tmp) / "ket_qua_u.jsonl", encoding="utf-8").readlines())
        assert 1 <= so_u < 20
        assert C.chay(m, "u", "gia", C.goi_gia, tmp, 3, in_tien_do=False) == 0
        assert len(C.da_xong(Path(tmp) / "ket_qua_u.jsonl")) == 20


def test_loi_nang_dung_dot_chay():
    def goi_sai_khoa(khoa, tn, mt=None, url=None):
        raise C.LoiNang("HTTP 401")
    df = pd.DataFrame([hang(pid=f"p{i}") for i in range(10)])
    with tempfile.TemporaryDirectory() as tmp:
        assert C.chay(df, "x", "k", goi_sai_khoa, tmp, 2, in_tien_do=False) == 2


def test_tran_usd_theo_usage_cost():
    def goi_co_gia(khoa, tn, mt=None, url=None):
        nd, tv, tr, kt, _ = C.goi_gia(khoa, tn, mt)
        return nd, tv, tr, kt, {"chi_phi": 0.01, "nguon": "balance"}
    df = pd.DataFrame([hang(pid=f"p{i}") for i in range(30)])
    with tempfile.TemporaryDirectory() as tmp:
        assert C.chay(df, "usd", "k", goi_co_gia, tmp, 1, tran_usd=0.05, in_tien_do=False) == 0
        dong = [json.loads(x) for x in open(Path(tmp) / "ket_qua_usd.jsonl", encoding="utf-8")]
        assert 5 <= len(dong) < 30 and dong[0]["chi_phi_usd"] == 0.01 and dong[0]["nguon_tinh_phi"] == "balance"
        tom = json.load(open(Path(tmp) / "tom_tat_usd.json", encoding="utf-8"))
        assert abs(tom["chi_phi_usd"] - 0.01 * len(dong)) < 1e-9


def test_tran_usd_khi_go_tra_cost_bang_0():
    """Gói Go trả usage.cost = 0 khi còn hạn mức; trần phải dựa vào giá niêm yết ước từ token."""
    def goi_go(khoa, tn, mt=None, url=None):
        nd, tv, tr, kt, _ = C.goi_gia(khoa, tn, mt)
        return nd, 100000, 100000, kt, {"chi_phi": 0, "nguon": None}   # 0,015 + 0,06 = 0,075 USD mỗi lời gọi (giá 0,15/0,60)
    df = pd.DataFrame([hang(pid=f"p{i}") for i in range(30)])
    with tempfile.TemporaryDirectory() as tmp:
        assert C.chay(df, "go", "k", goi_go, tmp, 1, tran_usd=0.5, in_tien_do=False) == 0
        dong = [json.loads(x) for x in open(Path(tmp) / "ket_qua_go.jsonl", encoding="utf-8")]
        assert len(dong) == 7 and abs(dong[0]["chi_phi_niem_yet_uoc_usd"] - 0.075) < 1e-9     # 7 x 0,075 = 0,525 >= 0,5
        tom = json.load(open(Path(tmp) / "tom_tat_go.json", encoding="utf-8"))
        assert tom["chi_phi_usd"] == 0 and abs(tom["chi_phi_dung_cho_tran_usd"] - 0.525) < 1e-9


def test_doc_usage_opencode_va_tran_theo_allowance():
    import urllib.request
    tot = {"sai_nghia": 5, "thieu_y": 5, "them_y": 5, "thuat_ngu": 5, "so_don_vi": 5, "troi_chay": 5,
           "ket_luan": "dat", "loi": []}

    class Tra:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):      # đúng cấu trúc thấy ở phản hồi thật của cổng Go (30/10/2026)
            return json.dumps({"choices": [{"message": {"content": json.dumps(tot), "reasoning_content": "nghĩ"},
                                            "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0,
                                         "opencode": {"billing_source": "go", "list_cost": 0.02, "allowance": 0.015}}}).encode()
    goc = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: Tra()
    df = pd.DataFrame([hang(pid=f"p{i}") for i in range(30)])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            assert C.chay(df, "al", "k", C.goi_api, tmp, 1, tran_usd=0.05, in_tien_do=False) == 0
            dong = [json.loads(x) for x in open(Path(tmp) / "ket_qua_al.jsonl", encoding="utf-8")]
            assert len(dong) == 4 and dong[0]["nguon_tinh_phi"] == "go"                      # 4 x 0,015 = 0,06 >= 0,05
            assert dong[0]["list_cost_usd"] == 0.02 and dong[0]["allowance_usd"] == 0.015
            tom = json.load(open(Path(tmp) / "tom_tat_al.json", encoding="utf-8"))
            assert abs(tom["allowance_usd"] - 0.06) < 1e-9 and abs(tom["list_cost_usd"] - 0.08) < 1e-9
    finally:
        urllib.request.urlopen = goc


def test_boc_lop_vo_noi_dung_cua_cong_go():
    tot = {"sai_nghia": 5, "thieu_y": 5, "them_y": 5, "thuat_ngu": 5, "so_don_vi": 5, "troi_chay": 5,
           "ket_luan": "dat", "loi": []}
    vo = json.dumps({"content": json.dumps(tot), "refusal": None, "role": "assistant"})
    assert rubric.doc_ket_qua(vo)["ket_luan"] == "dat"


def test_extra_body_va_suy_luan_tokens():
    import urllib.request

    class Tra:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 900,
                                         "completion_tokens_details": {"reasoning_tokens": 700}}}).encode()
    thay = {}

    def gia_urlopen(req, timeout=None):
        thay["body"] = json.loads(req.data)
        thay["hdr"] = dict(req.header_items())
        return Tra()
    goc = urllib.request.urlopen
    urllib.request.urlopen = gia_urlopen
    C._them.update({"thinking": {"type": "disabled"}})
    try:
        r = C.goi_api("k", [{"role": "user", "content": "x"}])
    finally:
        urllib.request.urlopen = goc
        C._them.clear()
    assert thay["body"]["thinking"] == {"type": "disabled"} and thay["body"]["model"] == C.CH.MO_HINH
    assert r[4]["suy_luan"] == 700 and r[2] == 900
    assert "X-opencode-session" in thay["hdr"] and thay["hdr"]["Authorization"] == "Bearer k"


def test_co_tat_suy_luan_va_so_sanh():
    import so_sanh_hai_lan as S
    with tempfile.TemporaryDirectory() as tmp:
        try:
            assert C.main(["--thu", "6", "--gia", "--ra", tmp, "--ten", "a"]) == 0
            assert C.main(["--thu", "6", "--gia", "--ra", tmp, "--ten", "b", "--tat-suy-luan"]) == 0
        finally:
            tham_so = dict(C._them)
            C._them.clear()
        assert tham_so == {"thinking": {"type": "disabled"}}
        b = S.doc("gia_b", tmp)
        assert all(x["tham_so_them"] == {"thinking": {"type": "disabled"}} for x in b.values())
        kq = S.so_sanh(S.doc("gia_a", tmp), b)
        assert kq["chung_ok"] == 6 and kq["giong_ket_luan"] == 1.0 and kq["chi_a_co"] == 0


def test_nha_cung_cap_nvidia_va_gioi_han_rpm():
    import time
    import urllib.request

    class Tra:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
    thay = {}

    def gia_urlopen(req, timeout=None):
        thay["url"], thay["body"], thay["hdr"] = req.full_url, json.loads(req.data), dict(req.header_items())
        return Tra()
    goc = urllib.request.urlopen
    urllib.request.urlopen = gia_urlopen
    try:
        C.CH.chon_nha_cung_cap("nvidia")
        C.CH.RPM = 600                      # 0,1 giây/lời gọi
        t0 = time.monotonic()
        for _ in range(5):
            C.goi_api("khoa-nvidia", [{"role": "user", "content": "x"}])
        assert time.monotonic() - t0 >= 0.35                      # 5 lời gọi cách nhau >= 0,1 giây
        assert thay["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert thay["body"]["model"] == "deepseek-ai/deepseek-v4.1-flash"
        assert "X-opencode-session" not in thay["hdr"] and thay["hdr"]["Authorization"] == "Bearer khoa-nvidia"
        assert C.CH.TEN_BIEN_KHOA == "NVIDIA_API_KEY"
    finally:
        urllib.request.urlopen = goc
        C.CH.chon_nha_cung_cap("opencode")
        C.CH.RPM = None
        C._tiep_theo[0] = 0.0
    assert C.CH.MO_HINH == "deepseek-v4.1-flash" and C.CH.NHA_CUNG_CAP == "opencode"


def test_nha_cung_cap_alibaba_dia_chi_theo_bien_va_tat_suy_luan():
    import os
    import urllib.request

    class Tra:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
    thay = {}

    def gia_urlopen(req, timeout=None):
        thay["url"], thay["body"], thay["hdr"] = req.full_url, json.loads(req.data), dict(req.header_items())
        return Tra()
    goc, cu = urllib.request.urlopen, os.environ.pop("ALIBABA_BASE_URL", None)
    urllib.request.urlopen = gia_urlopen
    try:
        try:
            C.CH.chon_nha_cung_cap("alibaba")
            assert False, "thiếu ALIBABA_BASE_URL phải báo lỗi"
        except SystemExit:
            pass
        os.environ["ALIBABA_BASE_URL"] = "https://ws1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/"
        C.CH.chon_nha_cung_cap("alibaba")
        C._them.update(C.CH.THAM_SO_TAT_SUY_LUAN)
        C.goi_api("khoa-ali", [{"role": "user", "content": "x"}])
        assert thay["url"] == "https://ws1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert thay["body"]["model"] == "deepseek-v4.1-flash" and thay["body"]["enable_thinking"] is False
        assert "thinking" not in thay["body"] and "X-opencode-session" not in thay["hdr"]
        assert C.CH.TEN_BIEN_KHOA == "DASHSCOPE_API_KEY"
    finally:
        urllib.request.urlopen = goc
        C._them.clear()
        if cu is None:
            os.environ.pop("ALIBABA_BASE_URL", None)
        else:
            os.environ["ALIBABA_BASE_URL"] = cu
        C.CH.chon_nha_cung_cap("opencode")
    assert C.CH.THAM_SO_TAT_SUY_LUAN == {"thinking": {"type": "disabled"}}


def test_mau_tu_cung_mau():
    df = pd.DataFrame([hang(pid=f"p{i}") for i in range(12)])
    with tempfile.TemporaryDirectory() as tmp:
        vao = Path(tmp) / "vao.parquet"
        df.to_parquet(vao)
        C.main(["--thu", "6", "--gia", "--ra", tmp, "--vao", str(vao), "--ten", "a"])
        cu = C.CH.THU_MUC_RA
        C.CH.THU_MUC_RA = Path(tmp)                       # --mau-tu đọc kết quả cũ trong thư mục ra mặc định
        try:
            assert C.main(["--mau-tu", "gia_a", "--gia", "--ra", tmp, "--vao", str(vao), "--ten", "b"]) == 0
        finally:
            C.CH.THU_MUC_RA = cu
        a, b = (sorted(json.loads(x)["product_id"] for x in open(Path(tmp) / f"ket_qua_{n}.jsonl", encoding="utf-8"))
                for n in ("gia_a", "gia_b"))
        assert a == b and len(a) == 6


def test_content_null_lay_reasoning_content():
    import urllib.request
    tot = {"sai_nghia": 5, "thieu_y": 5, "them_y": 5, "thuat_ngu": 5, "so_don_vi": 5, "troi_chay": 5,
           "ket_luan": "dat", "loi": []}

    class Tra:
        def __init__(self, msg):
            self.msg = msg

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": self.msg, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
    goc = urllib.request.urlopen
    try:
        urllib.request.urlopen = lambda req, timeout=None: Tra({"content": None, "reasoning_content": json.dumps(tot)})
        nd, _, _, _, phu = C.goi_api("k", [{"role": "user", "content": "x"}])
        assert phu["tu"] == "reasoning_content" and rubric.doc_ket_qua(nd)["ket_luan"] == "dat"
        urllib.request.urlopen = lambda req, timeout=None: Tra({"content": json.dumps(tot), "reasoning_content": "nghĩ..."})
        nd, _, _, _, phu = C.goi_api("k", [{"role": "user", "content": "x"}])
        assert phu["tu"] == "content"
        urllib.request.urlopen = lambda req, timeout=None: Tra({"content": None, "reasoning_content": "chỉ là lời suy nghĩ"})
        nd, _, _, _, phu = C.goi_api("k", [{"role": "user", "content": "x"}])
        try:
            rubric.doc_ket_qua(nd)
        except ValueError:
            pass
        else:
            raise AssertionError("lời suy nghĩ không có JSON phải bị từ chối")
    finally:
        urllib.request.urlopen = goc


def test_doi_tuong_title_pair_va_so_sanh_hai_bo():
    import so_sanh_hai_bo as SB
    h1688 = {"product_id": "a", "source": "1688", "category": "home", "title_zh": "标题ZH", "title_vi": "Tiêu đề VI",
             "description_zh": "x: 1", "description_vi": "x: 1", "desc_nguon_dich": "gemma4"}
    htiki = dict(h1688, source="tiki", title_zh="标题BAN-DICH", title_vi="Tiêu đề GỐC")
    p = rubric.dung_nguoi_dung_title(h1688)
    assert "SOURCE title (Chinese):\n标题ZH" in p and "TRANSLATION title (Vietnamese):\nTiêu đề VI" in p and "description" not in p.lower()
    p = rubric.dung_nguoi_dung_title(htiki)
    assert "SOURCE title (Vietnamese):\nTiêu đề GỐC" in p and "TRANSLATION title (Chinese):\n标题BAN-DICH" in p
    assert rubric.HE_THONG_TITLE != rubric.HE_THONG and "TITLE" in rubric.HE_THONG_TITLE
    tot = {"sai_nghia": 5, "thieu_y": 5, "them_y": 5, "thuat_ngu": 5, "so_don_vi": 5, "troi_chay": 5, "ket_luan": "dat", "loi": []}
    goi = lambda k, t, m=None: (json.dumps(tot), 10, 5, "stop", {})          # noqa: E731
    assert C.cham_mot(h1688, "k", goi, None, "mo_ta")["luat"] is not None      # luật cứng chỉ cho mô tả định dạng "; "
    for dt in ("mo_ta_pair", "title"):
        r = C.cham_mot(h1688, "k", goi, None, dt)
        assert r["luat"] is None and r["doi_tuong"] == dt
    # so sánh hai bộ: A hơn B đúng 1 điểm ở so_don_vi cho mọi sản phẩm
    def kq(pid, so, dat="dat"):
        return {"product_id": pid, "source": "1688", "trang_thai": "ok", "sai_nghia": 5, "thieu_y": 5, "them_y": 5, "thuat_ngu": 5,
                "so_don_vi": so, "troi_chay": 5, "ket_luan": dat, "co_cam_co": so <= 3, "loi": []}
    a = {str(i): kq(str(i), 5) for i in range(40)}
    b = {str(i): kq(str(i), 4) for i in range(40)}
    r = SB.so_sanh(a, b)
    sd = r["tat_ca"]["so_don_vi"]
    assert r["so_san_pham_chung"] == 40 and sd["a_tru_b"] == 1.0 and sd["ci95"] == [1.0, 1.0] and sd["a_cao_hon"] == 40
    assert r["tat_ca"]["sai_nghia"]["bang"] == 40 and r["tat_ca"]["diem_thap_nhat"]["a_tru_b"] == 1.0
    assert SB.thong_ke_title(a)["so_title_ok"] == 40
    with tempfile.TemporaryDirectory() as tmp:
        d = pd.DataFrame([dict(h1688, product_id=f"p{i}") for i in range(6)])
        vao = Path(tmp) / "vao.parquet"
        d.to_parquet(vao)
        assert C.main(["--doi-tuong", "title", "--thu", "4", "--gia", "--ra", tmp, "--vao", str(vao)]) == 0
        assert (Path(tmp) / "ket_qua_gia_title_thu_4.jsonl").exists()
        assert json.loads(open(Path(tmp) / "ket_qua_gia_title_thu_4.jsonl", encoding="utf-8").readline())["doi_tuong"] == "title"


def test_het_so_du_402_la_loi_nang():
    import io
    import urllib.error
    import urllib.request

    def gia_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", {}, io.BytesIO(b'{"error":"funds"}'))
    goc = urllib.request.urlopen
    urllib.request.urlopen = gia_urlopen
    try:
        C.goi_api("k", [{"role": "user", "content": "x"}])
    except C.LoiNang as e:
        assert "402" in str(e)
    else:
        raise AssertionError("402 phải là LoiNang")
    finally:
        urllib.request.urlopen = goc


def test_tra_loi_hong_duoc_ghi_la_loi():
    df = pd.DataFrame([hang(pid="a")])
    with tempfile.TemporaryDirectory() as tmp:
        C.chay(df, "h", "k", lambda k, t, m=None, u=None: ("không phải json", 10, 5, "stop", {}), tmp, 1, in_tien_do=False)
        d = json.loads(open(Path(tmp) / "ket_qua_h.jsonl", encoding="utf-8").readline())
        assert d["trang_thai"] == "loi" and d["product_id"] == "a"
        assert C.da_xong(Path(tmp) / "ket_qua_h.jsonl") == set()   # lỗi không tính là xong → lần sau thử lại


if __name__ == "__main__":
    n = 0
    for ten, f in list(globals().items()):
        if ten.startswith("test_"):
            f()
            n += 1
            print("ok", ten)
    print(f"{n}/{n} bài thử đạt")
