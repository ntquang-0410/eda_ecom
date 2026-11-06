"""Thử kiem_ket_qua.py trên dữ liệu giả. Chạy: python -I -X utf8 scripts/ft_train/test_kiem_ket_qua.py
Không đụng dữ liệu thật; mọi đầu ra ghi vào thư mục tạm."""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import kiem_ket_qua as K  # noqa: E402
import test_lay_mau_kiem as TL  # noqa: E402


def gan(v, d):
    assert abs(v - d) < 1e-3, (v, d)


# ---------- thống kê ----------
def test_wilson():
    lo, hi = K.wilson(5, 100)
    gan(lo, 0.0215)
    gan(hi, 0.1118)
    lo, hi = K.wilson(10, 20)
    gan(lo, 0.2993)
    gan(hi, 0.7007)
    lo, hi = K.wilson(0, 50)
    assert lo == 0.0 and 0.06 < hi < 0.08
    assert K.wilson(0, 0) == (None, None)


def test_kappa():
    gan(K.kappa_cohen(20, 5, 10, 65), 0.625)                                          # tính tay: po .85, pe .60
    gan(K.kappa_cohen(10, 0, 0, 10), 1.0)
    gan(K.kappa_cohen(0, 10, 10, 0), -1.0)
    assert K.kappa_cohen(0, 0, 0, 50) is None                                          # không có nặng nào: không xác định
    assert K.kappa_cohen(0, 0, 0, 0) is None


def test_kiem_dinh_dau():
    gan(K.kiem_dinh_dau(9, 1)["p_hai_phia"], 0.021484)                                 # 2 * 11/1024
    gan(K.kiem_dinh_dau(1, 9)["p_hai_phia"], 0.021484)
    gan(K.kiem_dinh_dau(10, 0)["p_hai_phia"], 2 / 1024)
    assert K.kiem_dinh_dau(5, 5)["p_hai_phia"] == 1.0
    assert K.kiem_dinh_dau(0, 0)["p_hai_phia"] == 1.0
    gan(K.kiem_dinh_dau(60, 40)["p_hai_phia"], 0.0568)                                 # biết trước: 100 ván, 60-40
    assert K.kiem_dinh_dau(9, 1)["n"] == 10


def test_cong_thuc_pair():
    gan(K.uoc_luong_pair(0.03, 0.20, 0.10), 0.03 + 0.048 * 0.10)
    gan(K.uoc_luong_pair(0.03, 0.05, 0.15, he_so=0.1), 0.02)
    bang = [{"huong": "zh->vi", "loai": "mo_ta", "bo": "cum", "ty_le": 0.04, "don_vi": 1000},
            {"huong": "zh->vi", "loai": "manh", "bo": "cum", "ty_le": 0.10, "don_vi": 150},
            {"huong": "zh->vi", "loai": "manh", "bo": "pair", "ty_le": 0.30, "don_vi": 150}]
    p = K.uoc_pair_theo_huong(bang)
    assert len(p) == 1 and p[0]["huong"] == "zh->vi"
    gan(p[0]["ty_le"], 0.04 + 0.048 * 0.2)
    p2 = K.uoc_pair_theo_huong(bang, he_so=0.1)
    gan(p2[0]["ty_le"], 0.06)


def test_cong():
    assert K.xet_cong({"a": 0.01, "b": 0.05})["ket_luan"] == "QUA"                    # đúng ngưỡng vẫn QUA
    assert K.xet_cong({"a": 0.01, "b": 0.0501})["ket_luan"] == "VUNG XAM"
    assert K.xet_cong({"a": 0.01, "b": 0.065})["ket_luan"] == "VUNG XAM"
    assert K.xet_cong({"a": 0.01, "b": 0.0651})["ket_luan"] == "KHONG QUA"
    assert K.xet_cong({"a": 0.01, "b": 0.07}, nguong_qua=0.08)["ket_luan"] == "QUA"
    assert K.xet_cong({})["ket_luan"] == "CHUA_CO_DU_LIEU"


# ---------- parser ----------
META = pd.DataFrame({"loai": ["title", "manh", "mo_ta", "mo_ta"], "so_manh": [0, 1, 5, 3],
                     "huong": "zh->vi", "bo": ["chung", "cum", "cum", "cum"], "thuoc_M1c": True, "thuoc_M2c": False},
                    index=["K00001", "K00002", "K00003", "K00004"])
IDS = list(META.index)
TEXT_TOT = '''Dưới đây là kết quả:
```csv
id,muc,loai_loi,ghi_chu
K00001, "Nang" ,L1;L2,"sai nghĩa, mất số"
K00002,nhẹ,L3,
```
```csv
id,so_manh_nang,ds_manh_nang,loai_loi,ghi_chu
"K00003","2","2; 5",L1,"hai mảnh sai"
K00004,0,,,
id,so_manh_nang,ds_manh_nang,loai_loi,ghi_chu
```
'''


def test_parser_chiu_loi_dinh_dang():
    rows, bo_qua = K.tach_dong_gemini(TEXT_TOT)
    assert [r[0] for r in rows] == IDS and len(bo_qua) == 1                           # 1 dòng lời dẫn bị bỏ qua
    kq, loi = K.kiem_lo("lo_001", IDS, META, rows)
    assert not loi, loi
    assert kq["K00001"]["muc"] == "nang" and "mất số" in kq["K00001"]["ghi_chu"]
    assert kq["K00002"]["muc"] == "nhe"
    assert kq["K00003"]["ds"] == frozenset({2, 5}) and kq["K00003"]["so_nang"] == 2
    assert kq["K00004"]["ds"] == frozenset() and kq["K00004"]["so_nang"] == 0
    # bảng markdown, id viết thường, tiêu đề lặp giữa chừng
    md = "| id | muc | loai_loi |\n|---|---|---|\n| k00001 | Nặng | L1 |\n| K00002 | Đạt | |\n"
    rows, _ = K.tach_dong_gemini(md)
    assert [r[0] for r in rows] == ["K00001", "K00002"]
    assert K.chuan_muc("Nặng") == "nang" and K.chuan_muc("Đạt") == "dat" and K.chuan_muc("Nhẹ") == "nhe" and K.chuan_muc("xx") is None


def test_parser_bat_loi():
    rows, _ = K.tach_dong_gemini(TEXT_TOT)
    # thiếu id
    kq, loi = K.kiem_lo("lo_001", IDS, META, [r for r in rows if r[0] != "K00002"])
    assert kq == {} and any("thieu" in x and "K00002" in x for x in loi)
    # id lạ (Gemini bịa)
    kq, loi = K.kiem_lo("lo_001", IDS, META, rows + [("K09999", ["Nang"], 99)])
    assert kq == {} and any("la" in x and "K09999" in x for x in loi)
    # id trùng
    kq, loi = K.kiem_lo("lo_001", IDS, META, rows + [rows[0]])
    assert kq == {} and any("trung" in x and "K00001" in x for x in loi)
    # ds_manh_nang > so_manh (mục K00004 chỉ có 3 mảnh)
    xau = [(i, c, n) if i != "K00004" else (i, ["1", "4"], n) for i, c, n in rows]
    kq, loi = K.kiem_lo("lo_001", IDS, META, xau)
    assert kq == {} and any("K00004" in x and "ngoai" in x for x in loi)          # mục chỉ có 3 mảnh mà ds chứa 4
    so_lech = [(i, c, n) if i != "K00004" else (i, ["2", "1"], n) for i, c, n in rows]
    assert K.kiem_lo("lo_001", IDS, META, so_lech)[0] == {}                         # so_manh_nang 2 nhưng ds chỉ 1 số
    hop_le = [(i, c, n) if i != "K00004" else (i, ["1", "3"], n) for i, c, n in rows]
    assert K.kiem_lo("lo_001", IDS, META, hop_le)[1] == []
    px, e = K.dung_phan_xet("mo_ta", 3, so_nang="1", ds="4")
    assert px is None and "ngoai" in e
    px, e = K.dung_phan_xet("mo_ta", 3, so_nang="2", ds="1;2;3")
    assert px is None and "khong khop" in e
    px, e = K.dung_phan_xet("mo_ta", 3, so_nang="2", ds="2;2")
    assert px is None
    px, e = K.dung_phan_xet("title", 0, muc="tệ")
    assert px is None
    # muc sai chữ trong dòng
    sai = [(i, c, n) if i != "K00001" else (i, ["Rất nặng"], n) for i, c, n in rows]
    assert K.kiem_lo("lo_001", IDS, META, sai)[0] == {}


def test_parser_claude():
    txt = "\n".join(json.dumps(o, ensure_ascii=False) for o in [
        {"id": "K00001", "muc": "Dat", "loai_loi": "", "ghi_chu": ""},
        {"id": "K00002", "muc": "nang", "loai_loi": "L1", "ghi_chu": "x"},
        {"id": "K00003", "so_manh_nang": 1, "ds_manh_nang": [3], "loai_loi": "L2", "ghi_chu": ""},
        {"id": "K00004", "so_manh_nang": 0, "ds_manh_nang": "", "loai_loi": "", "ghi_chu": ""}])
    rows, loi_cp = K.tach_dong_claude("```json\n" + txt + "\n```\n")
    assert not loi_cp
    kq, loi = K.kiem_lo("lo_001", IDS, META, rows)
    assert not loi and kq["K00003"]["ds"] == frozenset({3}) and kq["K00002"]["muc"] == "nang"
    rows, loi_cp = K.tach_dong_claude(txt + "\nkhong phai json")
    assert len(loi_cp) == 1
    assert K.kiem_lo("lo_001", IDS, META, rows, loi_cp)[0] == {}


# ---------- tỉ lệ nhóm, hiệu chuẩn ----------
def dung_meta_manh(n):
    return pd.DataFrame({"loai": "manh", "so_manh": 1, "huong": "zh->vi", "bo": "cum", "thuoc_M1c": False, "thuoc_M2c": True,
                         "category": "x"}, index=[f"K{i:05d}" for i in range(1, n + 1)])


def test_hieu_chuan_tinh_tay():
    meta = dung_meta_manh(100)
    ids = list(meta.index)
    # a=20 cả hai nặng, b=5 chỉ Gemini, c=10 chỉ Claude, d=65 cả hai không
    g = {i: {"muc": "nang" if k < 25 else "dat"} for k, i in enumerate(ids)}
    c = {i: {"muc": "nang" if (k < 20 or 25 <= k < 35) else "dat"} for k, i in enumerate(ids)}
    h = K.hieu_chuan(g, c, meta)
    assert (h["ca_hai_nang"], h["chi_gemini_nang"], h["chi_claude_nang"], h["ca_hai_khong"]) == (20, 5, 10, 65)
    gan(h["kappa"], 0.625)
    gan(h["ty_le_bat_dong"], 0.15)
    gan(h["bo_sot"], 0.05)                                                             # max(0, 10 - 5) / 100
    gan(h["ty_le_claude_nang_gemini_qua"], 10 / 30)
    assert any("bo sot" in x for x in h["canh_bao"]) and not any("kappa" in x for x in h["canh_bao"])   # 33% > 30%
    # đổi vai: Gemini nặng nhiều hơn Claude -> bo_sot = 0 (không âm)
    h2 = K.hieu_chuan(c, g, meta)
    assert h2["bo_sot"] == 0.0
    # kappa thấp -> cảnh báo
    g3 = {i: {"muc": "nang" if k < 10 else "dat"} for k, i in enumerate(ids)}
    c3 = {i: {"muc": "nang" if 10 <= k < 20 else "dat"} for k, i in enumerate(ids)}
    h3 = K.hieu_chuan(g3, c3, meta)
    assert h3["kappa"] < 0.4 and any("kappa" in x for x in h3["canh_bao"])
    # chưa có dữ liệu chung
    h4 = K.hieu_chuan(g, {}, meta)
    assert h4["bo_sot"] == 0.0 and h4["so_don_vi"] == 0 and h4["canh_bao"]


def test_don_vi_mo_ta_theo_manh():
    meta = pd.DataFrame({"loai": ["mo_ta", "mo_ta"], "so_manh": [4, 6], "huong": "zh->vi", "bo": "cum",
                         "thuoc_M1c": True, "thuoc_M2c": False}, index=["K00001", "K00002"])
    g = {"K00001": {"so_nang": 1, "ds": frozenset({1})}, "K00002": {"so_nang": 0, "ds": frozenset()}}
    c = {"K00001": {"so_nang": 2, "ds": frozenset({1, 3})}, "K00002": {"so_nang": 1, "ds": frozenset({6})}}
    h = K.hieu_chuan(g, c, meta)
    assert h["so_don_vi"] == 10 and (h["ca_hai_nang"], h["chi_gemini_nang"], h["chi_claude_nang"], h["ca_hai_khong"]) == (1, 0, 2, 7)
    gan(h["bo_sot"], 0.2)
    # tỉ lệ nhóm mo_ta: Σ mảnh nặng / Σ mảnh
    meta["category"] = "x"
    r = K.ty_le_nhom(g, meta)
    assert len(r) == 1 and r[0]["don_vi"] == 10 and r[0]["nang"] == 1 and r[0]["so_muc"] == 2
    gan(r[0]["ty_le"], 0.1)
    lo, hi = K.wilson(1, 10)
    gan(r[0]["wilson_lo"], lo)


def test_ty_le_nhom_title_va_manh():
    meta = pd.DataFrame({"loai": ["title"] * 4 + ["manh"] * 2, "so_manh": [0] * 4 + [1] * 2,
                         "huong": "vi->zh", "bo": ["chung"] * 4 + ["cum", "pair"],
                         "thuoc_M1c": False, "thuoc_M2c": False, "category": "x"},
                        index=[f"K{i:05d}" for i in range(1, 7)])
    kq = {f"K0000{i}": {"muc": m} for i, m in zip(range(1, 7), ["nang", "dat", "nhe", "dat", "nang", "dat"])}
    r = {(x["loai"], x["bo"]): x for x in K.ty_le_nhom(kq, meta)}
    assert r[("title", "chung")]["nang"] == 1 and r[("title", "chung")]["don_vi"] == 4 and r[("title", "chung")]["nhe"] == 1
    assert r[("manh", "cum")]["ty_le"] == 1.0 and r[("manh", "pair")]["ty_le"] == 0.0


# ---------- tích hợp trên mẫu giả ----------
def sinh_ket_qua(ra, meta, seed=5, p=0.12, mat=0.2, thuc=None):
    """Sinh sự thật + kết quả Gemini (bỏ sót mat) và Claude (đúng sự thật), ghi file đúng định dạng."""
    rng = np.random.default_rng(seed)
    thuc = {}
    for i, m in meta.iterrows():
        dv = K.danh_sach_don_vi(m.loai, m.so_manh)
        thuc[i] = {u for u in dv if rng.random() < p}
    gem = {i: {u for u in s if rng.random() >= mat} for i, s in thuc.items()}
    (ra / "ket_qua_gemini").mkdir(exist_ok=True)
    (ra / "ket_qua_claude").mkdir(exist_ok=True)
    for f in sorted((ra / "gui_gemini").glob("lo_*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        a, b = ["id,muc,loai_loi,ghi_chu"], ["id,so_manh_nang,ds_manh_nang,loai_loi,ghi_chu"]
        for i in d.id:
            m, nang = meta.loc[i], gem[i]
            if m.loai == "mo_ta":
                b.append(f'{i},{len(nang)},{";".join(map(str, sorted(nang)))},L1,"ghi chú, có phẩy"')
            else:
                a.append(f'{i},{"Nang" if nang else "Dat"},,"ok"')
        txt = "```csv\n" + "\n".join(a) + "\n```\n```csv\n" + "\n".join(b) + "\n```\n" if len(b) > 1 else "```csv\n" + "\n".join(a) + "\n```\n"
        (ra / "ket_qua_gemini" / f"{f.stem}.txt").write_text(txt, encoding="utf-8")
    for f in sorted((ra / "gui_claude").glob("lo_*.jsonl")):
        out = []
        for x in f.read_text(encoding="utf-8").splitlines():
            i = json.loads(x)["id"]
            m, nang = meta.loc[i], thuc[i]
            out.append(json.dumps({"id": i, "so_manh_nang": len(nang), "ds_manh_nang": ";".join(map(str, sorted(nang))), "loai_loi": "", "ghi_chu": ""}
                                  if m.loai == "mo_ta" else {"id": i, "muc": "Nang" if nang else "Dat", "loai_loi": "", "ghi_chu": ""}))
        (ra / "ket_qua_claude" / f.name).write_text("\n".join(out) + "\n", encoding="utf-8")
    return thuc, gem


def test_tich_hop_bao_cao():
    with tempfile.TemporaryDirectory() as tmp:
        ra = Path(tmp)
        TL.chay_thu(ra)
        meta = pd.read_parquet(ra / "mau_kiem.parquet").set_index("id")
        thuc, gem = sinh_ket_qua(ra, meta)
        bc = K.tao_bao_cao(ra)
        assert all(t["trang_thai"] == "ok" for t in bc["lo_gemini"] + bc["lo_claude"]), bc["lo_gemini"]
        # tỉ lệ Gemini khớp tự tính độc lập
        for r in bc["ty_le_nhom"]:
            ids = meta.index[(meta.huong == r["huong"]) & (meta.loai == r["loai"]) & (meta.bo == r["bo"])]
            nang = sum(len(gem[i]) for i in ids)
            dv = sum(len(K.danh_sach_don_vi(meta.loc[i].loai, meta.loc[i].so_manh)) for i in ids)
            assert (r["nang"], r["don_vi"]) == (nang, dv), r
        # hiệu chuẩn: Claude = sự thật, Gemini bỏ sót => chỉ_gemini_nang = 0, kappa tính được
        h = bc["hieu_chuan"]
        chung = meta.index[meta.thuoc_M1c | meta.thuoc_M2c]
        assert h["so_muc_chung"] == len(chung) and h["chi_gemini_nang"] == 0
        assert h["chi_claude_nang"] == sum(len(thuc[i]) - len(gem[i]) for i in chung)
        gan(h["bo_sot"], h["chi_claude_nang"] / h["so_don_vi"])
        assert bc["cong"]["ket_luan"] in {"QUA", "VUNG XAM", "KHONG QUA"} and len(bc["uoc_luong_pair"]) == 2
        for r in bc["uoc_luong_pair"]:
            gan(r["ty_le_tho"], r["ty_le_cum_M1"] + 0.048 * (r["ty_le_M2_pair"] - r["ty_le_M2_cum"]))
        assert K.main(["--kiem", str(ra)]) == 0 and (ra / "bao_cao_kiem.json").exists() and (ra / "bao_cao_kiem.md").exists()
        json.loads((ra / "bao_cao_kiem.json").read_text(encoding="utf-8"))
        assert "18,3% sp, bản cụm có thêm thuộc tính" in (ra / "bao_cao_kiem.md").read_text(encoding="utf-8")

        # hỏng một lô Gemini: xóa một dòng + thêm id bịa -> báo gửi lại lô, loại cả lô khỏi thống kê
        f = ra / "ket_qua_gemini" / "lo_001.txt"
        dong = f.read_text(encoding="utf-8").splitlines()
        i_dong = next(k for k, x in enumerate(dong) if x.startswith("K"))
        f.write_text("\n".join(dong[:i_dong] + dong[i_dong + 1:] + ["K99999,Nang,,bia"]) + "\n", encoding="utf-8")
        bc2 = K.tao_bao_cao(ra)
        loi = [t for t in bc2["lo_gemini"] if t["trang_thai"] == "loi"]
        assert len(loi) == 1 and loi[0]["lo"] == "lo_001" and loi[0]["huong_dan"] == "gui lai lo lo_001"
        assert any("thieu" in x for x in loi[0]["loi"]) and any("la" in x and "K99999" in x for x in loi[0]["loi"])
        assert sum(r["so_muc"] for r in bc2["ty_le_nhom"]) < sum(r["so_muc"] for r in bc["ty_le_nhom"])
        assert any("gui lai lo lo_001" in x for x in bc2["canh_bao"])


def test_thu20():
    with tempfile.TemporaryDirectory() as tmp:
        ra = Path(tmp)
        TL.chay_thu(ra)
        meta = pd.read_parquet(ra / "mau_kiem.parquet").set_index("id")
        assert K.main(["--kiem", str(ra), "--thu20"]) == 1                              # chưa có kết quả
        sinh_ket_qua(ra, meta, mat=0.3, p=0.4)
        kq = K.so_sanh_thu20(ra)
        assert len(kq["muc"]) == 10 and kq["khop"] + kq["lech"] == 10 and "n quá nhỏ" in kq["ghi_chu"]
        assert K.main(["--kiem", str(ra), "--thu20"]) == 0 and (ra / "bao_cao_thu20.json").exists()
        assert not (ra / "bao_cao_kiem.json").exists()                                  # chế độ thử không xuất báo cáo đầy đủ


def test_dau_cli():
    assert K.main(["--dau", "9", "1"]) == 0


if __name__ == "__main__":
    n = 0
    for ten, f in list(globals().items()):
        if ten.startswith("test_"):
            f()
            n += 1
            print("ok", ten)
    print(f"{n}/{n} bài thử đạt")
