"""Thử lay_mau_kiem.py trên dữ liệu giả. Chạy: python -I -X utf8 scripts/ft_train/test_lay_mau_kiem.py
Không đụng dữ liệu thật; mọi đầu ra ghi vào thư mục tạm."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import chung as C  # noqa: E402
import lay_mau_kiem as L  # noqa: E402

CFG = dict(L.CAU_HINH)
CFG.update(m1={"zh->vi": 30, "vi->zh": 20}, m1c={"zh->vi": 6, "vi->zh": 4}, thu20={"zh->vi": 3, "vi->zh": 2},
           m2={"zh->vi": 15, "vi->zh": 15}, m2c={"zh->vi": 5, "vi->zh": 5}, b9={"zh->vi": 30, "vi->zh": 20},
           gemini_dong=10, gemini_ky_tu=3000, kc_gemini=12, claude_min=4, claude_max=6, kc_claude=6)
CATS = ["home", "beauty", "food", "toy"]
RUBRIC = "# Rubric thử\nDat / Nhe / Nang.\n\nDòng cuối.\n"


def tao_du_lieu(seed=1, n_zh=300, n_vi=200):
    """Hai bộ giả: cùng sp, cụm nối '; ', pair nối '\\n'; ~40% sp có 1-2 mảnh dịch khác nhau."""
    rng = np.random.default_rng(seed)
    cum, pair = [], []
    for k in range(n_zh + n_vi):
        zh = k < n_zh
        n = int(rng.integers(3, 7))
        nguon = [f"属性{j}: 值{k}_{j}" for j in range(n)]
        dich = [f"Thuộc tính {j}: giá trị {k}_{j}" for j in range(n)]
        dich_p = list(dich)
        if rng.random() < 0.4:
            for j in rng.choice(n, size=int(rng.integers(1, 3)), replace=False):
                dich_p[j] = dich[j] + " (khác)"
        base = dict(product_id=f"p{k:04d}", huong="zh->vi" if zh else "vi->zh",
                    category=CATS[int(rng.choice(4, p=[.4, .3, .2, .1]))],
                    nguon_title=f"标题{k}", dich_title=f"Tiêu đề {k}")
        nguon_c, dich_c = list(nguon), list(dich)
        if k % 5 == 0:                                       # bản cụm có thêm thuộc tính (như 颜色) mà pair không có
            nguon_c.insert(1, f"颜色: 红{k}")
            dich_c.insert(1, f"Màu sắc: đỏ {k}")
        elif k % 23 == 7:                                    # cùng số thuộc tính nhưng nội dung nguồn cụm khác pair
            nguon_c[0] = f"属性0: 别的{k}"
        cum.append(dict(base, nguon_mo_ta="; ".join(nguon_c), dich_mo_ta="; ".join(dich_c)))
        pair.append(dict(base, nguon_mo_ta="\n".join(nguon), dich_mo_ta="\n".join(dich_p)))
    return pd.DataFrame(cum), pd.DataFrame(pair)


def chay_thu(ra, cum=None, pair=None, ghi_de=False, seed=L.SEED):
    if cum is None:
        cum, pair = tao_du_lieu()
    # tập test giả: một nửa số sp (đủ lớn để chứa B9 không giao M1/M2)
    ids = cum.product_id[::1]
    bang = L.dung_bang_test(cum, pair, ids)
    man = L.chay(bang, ra, RUBRIC, "ab" * 32, CFG, seed, ghi_de)
    return bang, man


def test_phan_bo():
    dem = pd.Series({"a": 50, "b": 30, "c": 15, "d": 5})
    co = L.phan_bo(dem, 10)
    assert co.to_dict() == {"a": 5, "b": 3, "c": 2, "d": 0}, co.to_dict()       # 5, 3, 1.5, 0.5 -> phần dư lớn nhất: c lên 2, d lệ thuộc thứ tự
    assert L.phan_bo(dem, 7).sum() == 7 and (L.phan_bo(dem, 100) <= dem).all()
    try:
        L.phan_bo(dem, 101)
        assert False
    except ValueError:
        pass


def test_so_luong_va_phan_tang():
    with tempfile.TemporaryDirectory() as tmp:
        bang, man = chay_thu(tmp)
        ra = Path(tmp)
        mk = pd.read_parquet(ra / "mau_kiem.parquet")
        # M1: 30 + 20 sp, mỗi sp 2 mục; M2: 30 mảnh, mỗi mảnh 2 mục
        m1 = mk[mk.mau == "M1"]
        assert m1[m1.loai == "title"].groupby("huong").size().to_dict() == {"zh->vi": 30, "vi->zh": 20}
        assert m1[m1.loai == "mo_ta"].groupby("huong").size().to_dict() == {"zh->vi": 30, "vi->zh": 20}
        m2 = mk[mk.mau == "M2"]
        assert m2.groupby(["huong", "bo"]).size().to_dict() == {("vi->zh", "cum"): 15, ("vi->zh", "pair"): 15,
                                                                 ("zh->vi", "cum"): 15, ("zh->vi", "pair"): 15}
        # phân tầng: đúng phân bổ tỉ lệ trên tập đủ điều kiện
        for h, n in CFG["m1"].items():
            kỳ_vọng = L.phan_bo(bang[(bang.huong == h) & bang.du_m1].category.value_counts(), n).to_dict()
            thuc = m1[(m1.huong == h) & (m1.loai == "title")].category.value_counts().to_dict()
            assert {k: v for k, v in kỳ_vọng.items() if v} == thuc, (h, kỳ_vọng, thuc)
        # M1c, thu20 (thu20 là tập con M1c)
        c = mk[mk.thuoc_M1c & (mk.loai == "title")]
        assert c.groupby("huong").size().to_dict() == {"zh->vi": 6, "vi->zh": 4}
        t = mk[mk.thuoc_thu20]
        assert len(t) == 10 and t.loai.isin(["title", "mo_ta"]).all() and t.thuoc_M1c.all()
        assert t[t.loai == "title"].groupby("huong").size().to_dict() == {"zh->vi": 3, "vi->zh": 2}
        # M2c: 10 mảnh = 20 mục, cả hai bản đều thuộc M2c
        c2 = mk[mk.thuoc_M2c]
        assert len(c2) == 20 and c2.groupby("huong").size().to_dict() == {"zh->vi": 10, "vi->zh": 10}
        assert man["so_muc"]["tong"] == len(mk) == 2 * 50 + 2 * 30
        # so_manh = số thuộc tính nguồn BẢN CỤM, đánh số theo đúng các thuộc tính đó
        md = mk[mk.loai == "mo_ta"]
        for r in md.itertuples():
            ds = bang.loc[r.product_id, "l_nguon_cum"]
            assert r.so_manh == len(ds) == r.nguon.count("\n") + 1
            assert r.nguon == "\n".join(f"[{k}] {t}" for k, t in enumerate(ds, 1))
        # tất cả sp đều là sp test
        assert set(mk.product_id) <= set(bang.index)


def test_m2_chi_gom_manh_khac_nhau():
    with tempfile.TemporaryDirectory() as tmp:
        bang, _ = chay_thu(tmp)
        mk = pd.read_parquet(Path(tmp) / "mau_kiem.parquet")
        m2 = mk[mk.mau == "M2"]
        assert (m2.groupby("product_id").chi_so_manh.nunique() <= 2).all()           # tối đa 2 mảnh mỗi sp
        for (pid, i), g in m2.groupby(["product_id", "chi_so_manh"]):
            r = bang.loc[pid]
            assert r.can_duoc
            dc, dp = r.l_dich_cum[i - 1], r.l_dich_pair[i - 1]
            assert dc != dp
            assert set(g.bo) == {"cum", "pair"} and len(g) == 2
            assert g[g.bo == "cum"].ban_dich.iloc[0] == dc and g[g.bo == "pair"].ban_dich.iloc[0] == dp
            assert (g.nguon == r.l_nguon_pair[i - 1]).all()


def test_sp_khong_can_duoc_bi_loai():
    cum, pair = tao_du_lieu()
    pair.loc[pair.product_id == "p0006", "nguon_mo_ta"] = "chỉ một dòng"
    pair.loc[pair.product_id == "p0006", "dich_mo_ta"] = "một dòng khác"
    bang = L.dung_bang_test(cum, pair, cum.product_id)
    assert not bang.loc["p0006", "can_duoc"] and bang.loc["p0004", "can_duoc"]
    assert bang.loc["p0006", "du_m1"]                                                  # vẫn đủ điều kiện cho M1 (nguồn cụm hợp lệ)


def test_nguon_cum_them_thuoc_tinh():
    """(a) M1 đánh số theo nguồn bản cụm; (b) sp có nguồn cụm khác nguồn pair (thêm thuộc tính, hoặc cùng số lượng nhưng
    khác nội dung) không bao giờ vào M2."""
    cum, pair = tao_du_lieu()
    bang = L.dung_bang_test(cum, pair, cum.product_id)
    r = bang.loc["p0005"]                                                              # k % 5 == 0: cụm có thêm 颜色
    assert len(r.l_nguon_cum) == len(r.l_nguon_pair) + 1 and "颜色: 红5" in r.l_nguon_cum and not r.can_duoc
    muc = L.muc_sp(r, False, False)[1]
    assert muc["so_manh"] == len(r.l_nguon_cum) and "[2] 颜色: 红5" in muc["nguon"] and muc["nguon"].count("\n") + 1 == muc["so_manh"]
    assert muc["ban_dich"] == r.dich_mo_ta_cum.strip()
    r7 = bang.loc["p0007"]                                                             # 7 % 23 == 7: cùng số lượng, khác nội dung
    assert len(r7.l_nguon_cum) == len(r7.l_nguon_pair) and r7.l_nguon_cum != r7.l_nguon_pair and not r7.can_duoc
    assert bang.loc["p0004", "can_duoc"]
    ten = bang.index[~bang.can_duoc & bang.du_m1]
    # chạy đủ: không sp "không căn được" nào vào M2, nhưng có sp cụm-thêm-thuộc-tính vào M1
    with tempfile.TemporaryDirectory() as tmp:
        L.chay(bang, tmp, RUBRIC, "ab" * 32, CFG)
        mk = pd.read_parquet(Path(tmp) / "mau_kiem.parquet")
    assert not set(mk.product_id[mk.mau == "M2"]) & set(ten)
    them = mk[(mk.loai == "mo_ta") & mk.product_id.isin(bang.index[[len(a) == len(b) + 1 for a, b in zip(bang.l_nguon_cum, bang.l_nguon_pair)]])]
    assert len(them) > 0, "M1 phải có sp cụm-thêm-thuộc-tính"


def test_b9_khong_giao():
    with tempfile.TemporaryDirectory() as tmp:
        _, man = chay_thu(tmp)
        mk = pd.read_parquet(Path(tmp) / "mau_kiem.parquet")
        b9 = pd.read_parquet(Path(tmp) / "mau_b9.parquet")
        assert len(b9) == 50 and b9.groupby("huong").size().to_dict() == {"zh->vi": 30, "vi->zh": 20}
        assert not set(b9.product_id) & set(mk.product_id)
        assert man["b9_giao_m1_m2"] == 0


def test_id_mu_va_csv():
    with tempfile.TemporaryDirectory() as tmp:
        _, man = chay_thu(tmp)
        ra = Path(tmp)
        mk = pd.read_parquet(ra / "mau_kiem.parquet")
        assert list(mk.id) == [f"K{i:05d}" for i in range(1, len(mk) + 1)]
        assert mk.thuoc_thu20.iloc[:10].all() and not mk.thuoc_thu20.iloc[10:].any()
        lo_g = sorted((ra / "gui_gemini").glob("lo_*.csv"))
        assert lo_g[0].name == "lo_000_thu20.csv"
        tong, thay = 0, []
        for f in lo_g:
            raw = f.read_bytes()
            assert raw.startswith(b"\xef\xbb\xbf")                                       # UTF-8 có BOM
            d = pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False)
            assert list(d.columns) == ["id", "huong", "loai", "nguon", "ban_dich"]
            assert len(d) <= CFG["gemini_dong"] and len(raw.decode("utf-8-sig")) <= CFG["gemini_ky_tu"] + 400
            assert not d.nguon.str.contains("\n").any() and not d.ban_dich.str.contains("\n").any()
            text = raw.decode("utf-8-sig")
            for cam in ("cum", "pair", "chung", "M1", "M2", "p0001", "gemma", "deepseek"):  # không lộ bộ/mẫu/sp
                assert cam not in text.replace("Thuộc tính", ""), (f.name, cam)
            tong += len(d)
            thay.append(set(d.id))
        assert tong == len(mk)
        assert set(pd.read_csv(lo_g[0], encoding="utf-8-sig", dtype=str).id) == set(mk.id[mk.thuoc_thu20])
        assert any(" ⏎ [2] " in x for f in lo_g for x in pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False).nguon)
        # hai bản của cùng một mảnh M2 không cùng lô Gemini và không nằm cạnh nhau
        lo_cua = {i: n for n, s in enumerate(thay) for i in s}
        pos = {i: k for k, i in enumerate(mk.id)}
        for _, g in mk[mk.loai == "manh"].groupby(["product_id", "chi_so_manh"]):
            a, b = g.id
            assert lo_cua[a] != lo_cua[b] and abs(pos[a] - pos[b]) >= CFG["kc_gemini"]
        # lô Claude: chỉ M1c + M2c, lô đầu đúng thu20, 4..6 mục/lô, hai bản M2c khác lô
        lo_c = sorted((ra / "gui_claude").glob("lo_*.jsonl"))
        assert lo_c[0].name == "lo_000_thu20.jsonl"
        ids_c, lo_id = [], {}
        for n, f in enumerate(lo_c):
            dong = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
            assert set(dong[0]) == {"id", "huong", "loai", "nguon", "ban_dich", "so_manh"}
            if n == 0:
                assert len(dong) == 10
            else:
                assert CFG["claude_min"] <= len(dong) <= CFG["claude_max"], (f.name, len(dong))
            for x in dong:
                lo_id[x["id"]] = n
            ids_c += [x["id"] for x in dong]
        chung = mk[mk.thuoc_M1c | mk.thuoc_M2c]
        assert sorted(ids_c) == sorted(chung.id) and len(set(ids_c)) == len(ids_c)
        for _, g in chung[chung.loai == "manh"].groupby(["product_id", "chi_so_manh"]):
            a, b = g.id
            assert lo_id[a] != lo_id[b]
        # manifest
        assert [x["so_dong"] for x in man["lo"]["gemini"]][0] == 10
        assert sum(x["so_dong"] for x in man["lo"]["gemini"]) == len(mk)
        assert (ra / "manifest_kiem.json").exists()


def test_loi_nhac_co_rubric_nguyen_van():
    with tempfile.TemporaryDirectory() as tmp:
        chay_thu(tmp)
        t = (Path(tmp) / "gui_gemini" / "LOI_NHAC_GEMINI.md").read_text(encoding="utf-8")
        assert t.splitlines()[0].startswith("<!-- SHA256 rubric_kiem.md: " + "ab" * 32)
        assert RUBRIC.rstrip("\n") in t
        for tu in ("id,muc,loai_loi,ghi_chu", "id,so_manh_nang,ds_manh_nang,loai_loi,ghi_chu", "Dat", "Nhe", "Nang"):
            assert tu in t


def test_khong_ghi_de_va_xac_dinh():
    cum, pair = tao_du_lieu()
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        chay_thu(t1, cum, pair)
        try:
            chay_thu(t1, cum, pair)
            assert False, "phải từ chối ghi đè"
        except RuntimeError as e:
            assert "da_co_mau" in str(e)
        chay_thu(t1, cum, pair, ghi_de=True)
        chay_thu(t2, cum, pair)
        a = pd.read_parquet(Path(t1) / "mau_kiem.parquet")
        b = pd.read_parquet(Path(t2) / "mau_kiem.parquet")
        assert a.equals(b)                                                               # cùng seed -> cùng mẫu, cùng id


def test_thieu_rubric_dung_lai():
    try:
        L.doc_rubric(Path(tempfile.gettempdir()) / "khong_co_rubric_kiem.md")
        assert False
    except RuntimeError as e:
        assert "thieu_rubric" in str(e)


def test_b9_dao_ab():
    n = 400
    tc = [f"nguon{i}" for i in range(n)]
    a = [f"A{i}" for i in range(n)]
    b = [f"B{i}" for i in range(n)]
    gui, ax = L.dung_lo_b9(tc, a, b, seed=7)
    assert list(gui.columns) == ["id", "tham_chieu", "ban_1", "ban_2"] and "ban_1_la" in ax.columns
    assert 0.4 < (ax.ban_1_la == "a").mean() < 0.6                                       # đảo ngẫu nhiên
    for r, m in zip(gui.itertuples(), ax.itertuples()):
        assert {r.ban_1[0], r.ban_2[0]} == {"A", "B"}
        assert r.ban_1[0].lower() == m.ban_1_la and r.ban_2[0].lower() == m.ban_2_la
    g2, ax2 = L.dung_lo_b9(tc, a, b, seed=7)
    assert gui.equals(g2) and ax.equals(ax2)
    assert not gui.equals(L.dung_lo_b9(tc, a, b, seed=8)[0])
    # giải mã: chọn 'ban_1' mọi dòng -> số thắng của A = số dòng ban_1_la == 'a'
    chon = pd.Series(["ban_1"] * n, index=gui.id)
    sa, sb, hoa = L.giai_ma_b9(chon, ax)
    assert sa == int((ax.ban_1_la == "a").sum()) and sa + sb == n and hoa == 0


def test_sua_khoang_cach():
    rng = np.random.default_rng(3)
    doi = {}
    for i in range(0, 60, 2):
        doi[i], doi[i + 1] = i + 1, i
    kq = L.sua_khoang_cach(list(range(60)), doi, 10, rng)
    vt = {x: i for i, x in enumerate(kq)}
    assert all(abs(vt[i] - vt[i + 1]) >= 10 for i in range(0, 60, 2)) and sorted(kq) == list(range(60))


if __name__ == "__main__":
    n = 0
    for ten, f in list(globals().items()):
        if ten.startswith("test_"):
            f()
            n += 1
            print("ok", ten)
    print(f"{n}/{n} bài thử đạt")
