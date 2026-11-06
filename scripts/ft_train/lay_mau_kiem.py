"""Bước B2: lấy mẫu kiểm chất lượng dữ liệu bằng LLM chấm (Gemini web + Claude hiệu chuẩn), mọi mẫu từ TẬP TEST.

Mẫu: M1 (300 zh->vi + 200 vi->zh sp; title + mô tả bản cụm, nguồn đánh số theo thuộc tính nguồn CỦA BẢN CỤM), M1c (100 sp chung), thu20 (20 mục thử),
M2 (mảnh khác nhau giữa cụm và pair, 150+150), M2c (100 mảnh chung), B9 (500 sp để dành, không giao M1/M2).
Mọi thứ ghi vào data/processed/ft_run/test_khoa/kiem/ ; id mù K00001..., người chấm không biết bộ/mẫu/sp.

Chạy:  python -I -X utf8 scripts/ft_train/lay_mau_kiem.py            (cần split + rubric_kiem.md)
       python -I -X utf8 scripts/ft_train/lay_mau_kiem.py --ghi-de  (cố ý bỏ kết quả cũ; id sẽ đổi!)"""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung as C  # noqa: E402

SEED = 20261104
FILE_RUBRIC = Path(__file__).resolve().parent / "rubric_kiem.md"
HUONGS = ["zh->vi", "vi->zh"]
CAU_HINH = {
    "m1": {"zh->vi": 300, "vi->zh": 200},
    "m1c": {"zh->vi": 60, "vi->zh": 40},
    "thu20": {"zh->vi": 6, "vi->zh": 4},
    "m2": {"zh->vi": 150, "vi->zh": 150},
    "m2c": {"zh->vi": 50, "vi->zh": 50},
    "b9": {"zh->vi": 300, "vi->zh": 200},
    "toi_da_manh_moi_sp": 2,
    "kc_gemini": 60,          # cách tối thiểu (vị trí) giữa hai bản của cùng một mảnh M2 > 50 dòng/lô
    "kc_claude": 30,          # > 25 mục/lô
    "gemini_dong": 50,
    "gemini_ky_tu": 15000,
    "claude_min": 20,
    "claude_max": 25,
}
COT_GEMINI = ["id", "huong", "loai", "nguon", "ban_dich"]
COT_CAM = {"bo", "mau", "model", "judge", "product_id", "thuoc_M1c", "thuoc_M2c", "thuoc_thu20"}
COT_CLAUDE = ["id", "huong", "loai", "nguon", "ban_dich", "so_manh"]


# ---------- tiện ích văn bản ----------
def lam_phang(s):
    """'\\n' -> ' ⏎ ' để một mục nằm trọn một dòng CSV."""
    return str(s).replace("\r\n", "\n").replace("\r", "\n").replace("\n", C.KY_HIEU_XUONG_DONG)


def tach_manh(s, sep):
    """Tách thuộc tính theo sep, bỏ phần rỗng; None/NaN -> []."""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return []
    return [x.strip() for x in str(s).split(sep) if x.strip()]


def csv_text(df, cot):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cot)
    for r in df[cot].itertuples(index=False):
        w.writerow([lam_phang(x) if isinstance(x, str) else x for x in r])
    return buf.getvalue()


def do_dai_dong(r, cot):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow([lam_phang(r[c]) if isinstance(r[c], str) else r[c] for c in cot])
    return len(buf.getvalue())


# ---------- bảng test ----------
def dung_bang_test(cum, pair, ids_test):
    """Ghép hai bộ theo product_id (chỉ sp test); thêm các danh sách thuộc tính đã tách. Index = product_id."""
    ids = set(map(str, ids_test))
    c = cum.assign(product_id=cum.product_id.astype(str))
    c = c[c.product_id.isin(ids)]
    p = pair.assign(product_id=pair.product_id.astype(str))
    p = p[p.product_id.isin(ids)]
    assert c.product_id.is_unique and p.product_id.is_unique
    m = c[["product_id", "huong", "category", "nguon_title", "dich_title", "nguon_mo_ta", "dich_mo_ta"]].merge(
        p[["product_id", "nguon_mo_ta", "dich_mo_ta"]], on="product_id", suffixes=("_cum", "_pair"))
    m = m.sort_values("product_id").set_index("product_id", drop=False)
    m["category"] = m.category.fillna("(khong_ro)").astype(str)
    sc, sp = C.TACH_DONG["cum"], C.TACH_DONG["pair"]
    m["l_nguon_cum"] = [tach_manh(x, sc) for x in m.nguon_mo_ta_cum]
    m["l_dich_cum"] = [tach_manh(x, sc) for x in m.dich_mo_ta_cum]
    m["l_nguon_pair"] = [tach_manh(x, sp) for x in m.nguon_mo_ta_pair]
    m["l_dich_pair"] = [tach_manh(x, sp) for x in m.dich_mo_ta_pair]
    m["du_m1"] = [bool(str(a).strip()) and bool(str(b).strip()) and len(n) > 0 and bool(str(d).strip())
                  for a, b, n, d in zip(m.nguon_title.fillna(""), m.dich_title.fillna(""), m.l_nguon_cum, m.dich_mo_ta_cum.fillna(""))]
    # M2 chỉ nhận sp mà danh sách nguồn cụm TRÙNG NỘI DUNG, đúng thứ tự, với các dòng nguồn pair (bản cụm có thể có thêm
    # thuộc tính mà pair không có: 18,3% sp, gần hết là 1688), và số thuộc tính đích cụm = số dòng đích pair
    m["can_duoc"] = [len(a) > 0 and a == c_ and len(b) == len(d) == len(a)
                     for a, b, c_, d in zip(m.l_nguon_cum, m.l_dich_cum, m.l_nguon_pair, m.l_dich_pair)]
    return m


# ---------- lấy mẫu ----------
def phan_bo(dem, n):
    """Phân bổ n theo tỉ lệ dem (Series nhóm -> số có thể chọn), phần dư lớn nhất; không vượt dem."""
    dem = dem[dem > 0]
    tong = int(dem.sum())
    if n > tong:
        raise ValueError(f"can {n} nhung chi co {tong}")
    tho = dem * n / tong
    co = np.floor(tho).astype(int)
    du = tho - co
    thieu = int(n - co.sum())
    for c in sorted(dem.index, key=lambda k: (-du[k], str(k)))[:thieu]:
        co[c] += 1
    assert int(co.sum()) == n and (co <= dem).all()
    return co


def lay_phan_tang(bang, n, rng):
    """Lấy n sp, phân tầng theo category. Trả về danh sách product_id (đã sắp theo category rồi id)."""
    quota = phan_bo(bang.category.value_counts(), n)
    chon = []
    for cat in sorted(quota.index, key=str):
        k = int(quota[cat])
        if k:
            ids = bang.index[bang.category == cat].to_numpy()
            chon.extend(rng.choice(ids, size=k, replace=False).tolist())
    return chon


def lay_m1(bang, cfg, rng):
    """{huong: [product_id]}"""
    kq = {}
    for h in HUONGS:
        kq[h] = lay_phan_tang(bang[(bang.huong == h) & bang.du_m1], cfg["m1"][h], rng)
    return kq


def lay_con(ds_theo_huong, cfg_con, rng):
    """Chọn ngẫu nhiên tập con theo hướng."""
    return {h: sorted(rng.choice(np.array(ds_theo_huong[h], dtype=object), size=cfg_con[h], replace=False).tolist())
            for h in HUONGS}


def lay_m2(bang, cfg, rng):
    """{huong: [(product_id, chi_so_0based)]}: mảnh có đích cụm != đích pair, tối đa k mảnh mỗi sp.
    Chỉ xét sp can_duoc (nguồn cụm trùng nội dung, đúng thứ tự, với nguồn pair)."""
    kq = {}
    for h in HUONGS:
        pool = []
        for pid, r in bang[(bang.huong == h) & bang.can_duoc].iterrows():
            idx = [i for i, (a, b, s) in enumerate(zip(r.l_dich_cum, r.l_dich_pair, r.l_nguon_pair)) if a != b and s]
            if idx:
                k = min(cfg["toi_da_manh_moi_sp"], len(idx))
                pool += [(pid, int(i)) for i in rng.choice(idx, size=k, replace=False)]
        if len(pool) < cfg["m2"][h]:
            raise RuntimeError(f"M2 {h}: chi co {len(pool)} manh khac nhau, can {cfg['m2'][h]}")
        pos = rng.choice(len(pool), size=cfg["m2"][h], replace=False)
        kq[h] = sorted(pool[i] for i in pos)
    return kq


def lay_m2c(m2, cfg, rng):
    kq = {}
    for h in HUONGS:
        pos = rng.choice(len(m2[h]), size=cfg["m2c"][h], replace=False)
        kq[h] = sorted(m2[h][i] for i in pos)
    return kq


# ---------- dựng mục ----------
def muc_sp(r, thuoc_m1c, thuoc_thu20):
    dong = r.l_nguon_cum          # mảnh M1 = một thuộc tính nguồn của BẢN CỤM (tách "; "), không dùng dòng nguồn pair
    chung = dict(product_id=r.product_id, huong=r.huong, category=r.category, mau="M1", chi_so_manh=0,
                 thuoc_M1c=thuoc_m1c, thuoc_M2c=False, thuoc_thu20=thuoc_thu20)
    return [
        dict(chung, loai="title", bo="chung", so_manh=0, nguon=str(r.nguon_title).strip(), ban_dich=str(r.dich_title).strip()),
        dict(chung, loai="mo_ta", bo="cum", so_manh=len(dong),
             nguon="\n".join(f"[{k}] {t}" for k, t in enumerate(dong, 1)), ban_dich=str(r.dich_mo_ta_cum).strip()),
    ]


def muc_manh(r, i, thuoc_m2c):
    chung = dict(product_id=r.product_id, huong=r.huong, category=r.category, loai="manh", mau="M2", so_manh=1,
                 chi_so_manh=i + 1, thuoc_M1c=False, thuoc_M2c=thuoc_m2c, thuoc_thu20=False, nguon=r.l_nguon_pair[i])
    return [dict(chung, bo="cum", ban_dich=r.l_dich_cum[i]), dict(chung, bo="pair", ban_dich=r.l_dich_pair[i])]


def dung_muc(bang, m1, m1c, thu20, m2, m2c):
    muc = []
    set_m1c = {p for h in HUONGS for p in m1c[h]}
    set_t20 = {p for h in HUONGS for p in thu20[h]}
    for h in HUONGS:
        for pid in m1[h]:
            muc += muc_sp(bang.loc[pid], pid in set_m1c, pid in set_t20)
    set_m2c = {x for h in HUONGS for x in m2c[h]}
    for h in HUONGS:
        for pid, i in m2[h]:
            muc += muc_manh(bang.loc[pid], i, (pid, i) in set_m2c)
    return pd.DataFrame(muc)


# ---------- xáo, gán id mù ----------
def sua_khoang_cach(thu_tu, doi, kc, rng, toi_da=20000):
    """Hoán đổi ngẫu nhiên cho đến khi hai phần tử của mỗi cặp (doi: idx -> idx đối tác) cách nhau >= kc vị trí."""
    thu_tu = list(thu_tu)
    n = len(thu_tu)
    vt = {x: i for i, x in enumerate(thu_tu)}
    for _ in range(toi_da):
        xau = [x for x in thu_tu if x in doi and doi[x] in vt and abs(vt[x] - vt[doi[x]]) < kc]
        if not xau:
            return thu_tu
        x = xau[int(rng.integers(len(xau)))]
        y = thu_tu[int(rng.integers(n))]
        i, j = vt[x], vt[y]
        thu_tu[i], thu_tu[j] = y, x
        vt[x], vt[y] = j, i
    raise RuntimeError("khong xep duoc khoang cach giua hai ban cua cung manh M2")


def xao_va_gan_id(muc, cfg, rng):
    """Thu20 đứng đầu (lô thử), phần còn lại xáo trộn và giữ hai bản cùng mảnh M2 cách xa nhau. Cấp id sau khi xáo."""
    muc = muc.reset_index(drop=True)
    khoa = muc.product_id + "#" + muc.chi_so_manh.astype(str)
    doi = {}
    for k, ds in muc[muc.loai == "manh"].groupby(khoa[muc.loai == "manh"]).groups.items():
        a, b = list(ds)
        doi[a], doi[b] = b, a
    dau = rng.permutation(muc.index[muc.thuoc_thu20].to_numpy()).tolist()
    con = rng.permutation(muc.index[~muc.thuoc_thu20].to_numpy()).tolist()
    con = sua_khoang_cach(con, doi, min(cfg["kc_gemini"], max(2, len(con) // 4)), rng)
    thu_tu = dau + con
    out = muc.loc[thu_tu].reset_index(drop=True)
    out.insert(0, "id", [f"K{i:05d}" for i in range(1, len(out) + 1)])
    return out


# ---------- chia lô ----------
def chia_lo_gemini(df, cfg):
    """df đã có id, thu20 đứng đầu. Trả về [(ten_lo, df_con)]; lô thử đúng các mục thu20."""
    lo = []
    t20 = df[df.thuoc_thu20]
    lo.append(("lo_000_thu20", t20))
    rest = df[~df.thuoc_thu20]
    hien, ky_tu, bat_dau = [], len(csv_text(df.iloc[:0], COT_GEMINI)), 0
    for k, (_, r) in enumerate(rest.iterrows()):
        d = do_dai_dong(r, COT_GEMINI)
        if hien and (len(hien) >= cfg["gemini_dong"] or ky_tu + d > cfg["gemini_ky_tu"]):
            lo.append((f"lo_{len(lo):03d}", rest.iloc[bat_dau:k]))
            hien, ky_tu, bat_dau = [], len(csv_text(df.iloc[:0], COT_GEMINI)), k
        hien.append(k)
        ky_tu += d
    if hien:
        lo.append((f"lo_{len(lo):03d}", rest.iloc[bat_dau:]))
    return lo


def chia_deu(n, toi_thieu, toi_da):
    """Chia n mục thành các lô cỡ gần bằng nhau trong [toi_thieu, toi_da] (n nhỏ -> một lô)."""
    if n == 0:
        return []
    k = -(-n // toi_da)
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def chia_lo_claude(df, cfg, rng):
    """Chỉ M1c + M2c. Lô đầu = thu20. Phần còn lại theo thứ tự id rồi sửa khoảng cách hai bản của cùng mảnh."""
    chung = df[df.thuoc_M1c | df.thuoc_M2c]
    t20 = chung[chung.thuoc_thu20]
    rest = chung[~chung.thuoc_thu20]
    khoa = rest.product_id + "#" + rest.chi_so_manh.astype(str)
    doi = {}
    pos = {i: p for p, i in enumerate(rest.index)}
    for _, ds in rest[rest.loai == "manh"].groupby(khoa[rest.loai == "manh"]).groups.items():
        a, b = list(ds)
        doi[a], doi[b] = b, a
    thu_tu = sua_khoang_cach(list(rest.index), doi, min(cfg["kc_claude"], max(2, len(rest) // 4)), rng)
    rest = rest.loc[thu_tu]
    lo = [("lo_000_thu20", t20)]
    b = 0
    for s in chia_deu(len(rest), cfg["claude_min"], cfg["claude_max"]):
        lo.append((f"lo_{len(lo):03d}", rest.iloc[b:b + s]))
        b += s
    return lo


# ---------- lời nhắc ----------
HUONG_DAN_DINH_DANG = """
## Cách đọc file CSV

- Cột: `id, huong, loai, nguon, ban_dich`. `huong` = `zh->vi` (nguồn tiếng Trung, dịch sang tiếng Việt) hoặc `vi->zh`.
- Trong `nguon` và `ban_dich`, ký hiệu ` ⏎ ` thay cho xuống dòng.
- `loai = title`: tiêu đề sản phẩm. `loai = manh`: một thuộc tính dạng `tên: giá trị`.
- `loai = mo_ta`: cột `nguon` gồm các mảnh đánh số `[1] tên: giá trị ⏎ [2] ...`; cột `ban_dich` là bản dịch của cả khối
  (các thuộc tính nối nhau bằng `; `). Mảnh thứ k trong bản dịch ứng với `[k]` ở nguồn.
- Chấm theo rubric ở trên. Không suy đoán dữ liệu này do mô hình nào hay cách nào tạo ra.

## Định dạng trả lời

Trả lời ĐỦ mọi dòng của file, đúng `id` đã cho, không tóm tắt, không bỏ dòng, không thêm id lạ, không lặp id.
Chỉ xuất CSV (đặt trong khối ```csv), không thêm giải thích ngoài khối.

**Mục `loai` = `title` hoặc `manh`**, một dòng mỗi mục, tiêu đề cột đúng như sau:

```csv
id,muc,loai_loi,ghi_chu
```

- `muc` chỉ nhận một trong ba giá trị `Dat`, `Nhe`, `Nang` (không dấu).
- `loai_loi`: mã lỗi theo rubric (nhiều mã nối bằng `|`), để trống nếu `Dat`.
- `ghi_chu`: tối đa 25 từ theo rubric, đặt trong ngoặc kép, không xuống dòng.

**Mục `loai` = `mo_ta`**, một dòng mỗi mục, tiêu đề cột đúng như sau:

```csv
id,so_manh_nang,ds_manh_nang,loai_loi,ghi_chu
```

- `so_manh_nang`: số mảnh bị lỗi nặng trong khối.
- `ds_manh_nang`: số thứ tự các mảnh lỗi nặng, nối bằng `;` (ví dụ `2;5`), để trống nếu không có mảnh nặng.
  Số lượng phải bằng `so_manh_nang`, mỗi số nằm trong 1..số mảnh của mục.
- `loai_loi`, `ghi_chu`: như trên.

Nếu một file có cả hai loại mục, trả HAI bảng CSV riêng trong cùng câu trả lời (mỗi bảng một khối ```csv với tiêu đề riêng).
"""


def dung_loi_nhac(rubric_text, sha):
    return (f"<!-- SHA256 rubric_kiem.md: {sha} -->\n"
            f"SHA256 của rubric: `{sha}`\n\n"
            "# Lời nhắc chấm bản dịch Trung–Việt (thương mại điện tử)\n\n"
            "## Rubric\n\n" + rubric_text.rstrip("\n") + "\n\n---\n" + HUONG_DAN_DINH_DANG)


# ---------- B9 (để dành) ----------
def dung_lo_b9(tham_chieu, hyp_a, hyp_b, seed=SEED):
    """Đảo ngẫu nhiên vị trí A/B từng dòng. Trả (gui, anh_xa); anh_xa KHÔNG gửi đi.
    gui: id, tham_chieu, ban_1, ban_2 ; anh_xa: id, ban_1_la, ban_2_la (giá trị 'a' hoặc 'b')."""
    tc, a, b = list(tham_chieu), list(hyp_a), list(hyp_b)
    assert len(tc) == len(a) == len(b)
    dao = np.random.default_rng(seed).random(len(tc)) < 0.5
    ids = [f"B{i:04d}" for i in range(1, len(tc) + 1)]
    ban_1 = [y if d else x for x, y, d in zip(a, b, dao)]
    ban_2 = [x if d else y for x, y, d in zip(a, b, dao)]
    gui = pd.DataFrame({"id": ids, "tham_chieu": tc, "ban_1": ban_1, "ban_2": ban_2})
    anh_xa = pd.DataFrame({"id": ids, "ban_1_la": np.where(dao, "b", "a"), "ban_2_la": np.where(dao, "a", "b")})
    return gui, anh_xa


def giai_ma_b9(chon, anh_xa):
    """chon: Series theo id nhận 'ban_1'/'ban_2'/'hoa'. Trả (so_a_thang, so_b_thang, so_hoa) để đưa vào kiem_dinh_dau."""
    m = anh_xa.set_index("id")
    ket = {"a": 0, "b": 0, "hoa": 0}
    for i, v in chon.items():
        ket["hoa" if v == "hoa" else m.loc[i, "ban_1_la" if v == "ban_1" else "ban_2_la"]] += 1
    return ket["a"], ket["b"], ket["hoa"]


# ---------- ghi ----------
def dem_theo_huong_cat(df, cot_huong="huong"):
    return {h: {str(c): int(n) for c, n in g.category.value_counts().sort_index().items()} for h, g in df.groupby(cot_huong)}


def doc_rubric(path=FILE_RUBRIC):
    if not Path(path).exists():
        raise RuntimeError(f"thieu_rubric: chua co {path}; tao rubric truoc roi chay lai")
    return Path(path).read_bytes().decode("utf-8-sig"), C.sha256_file(path)


def chay(bang, ra, rubric_text, rubric_sha, cfg=CAU_HINH, seed=SEED, ghi_de=False):
    """Lõi: bang = dung_bang_test(...); ra = thư mục kiem. Trả manifest."""
    ra = Path(ra)
    if (ra / "mau_kiem.parquet").exists() and not ghi_de:
        raise RuntimeError(f"da_co_mau: {ra / 'mau_kiem.parquet'} da ton tai; dung --ghi-de neu that su muon doi id")
    rng = np.random.default_rng(seed)
    m1 = lay_m1(bang, cfg, rng)
    m1c = lay_con(m1, cfg["m1c"], rng)
    thu20 = lay_con(m1c, cfg["thu20"], rng)
    m2 = lay_m2(bang, cfg, rng)
    m2c = lay_m2c(m2, cfg, rng)
    ids_m1 = {p for h in HUONGS for p in m1[h]}
    ids_m2 = {p for h in HUONGS for p, _ in m2[h]}
    con = bang[~bang.index.isin(ids_m1 | ids_m2)]
    b9 = {h: lay_phan_tang(con[con.huong == h], cfg["b9"][h], rng) for h in HUONGS}
    ids_b9 = {p for h in HUONGS for p in b9[h]}
    assert ids_b9.isdisjoint(ids_m1 | ids_m2), "B9 giao M1/M2"

    muc = xao_va_gan_id(dung_muc(bang, m1, m1c, thu20, m2, m2c), cfg, rng)
    assert muc.id.is_unique and muc.thuoc_thu20.iloc[:muc.thuoc_thu20.sum()].all()
    lo_g = chia_lo_gemini(muc, cfg)
    lo_c = chia_lo_claude(muc, cfg, rng)
    assert set(lo_g[0][1].id) == set(muc.id[muc.thuoc_thu20]) == set(lo_c[0][1].id)
    assert sum(len(d) for _, d in lo_g) == len(muc)           # mỗi mục đúng một lô, không lặp thu20

    for sub in ("gui_gemini", "gui_claude", "ket_qua_gemini", "ket_qua_claude", "b9"):
        (ra / sub).mkdir(parents=True, exist_ok=True)
    for sub, hau in (("gui_gemini", "csv"), ("gui_claude", "jsonl")):
        for f in (ra / sub).glob(f"lo_*.{hau}"):
            f.unlink()                                       # --ghi-de: bỏ lô cũ
    muc.to_parquet(ra / "mau_kiem.parquet", index=False)
    pd.DataFrame([(p, h, bang.loc[p, "category"]) for h in HUONGS for p in b9[h]],
                 columns=["product_id", "huong", "category"]).to_parquet(ra / "mau_b9.parquet", index=False)
    lo_ra = {"gemini": [], "claude": []}
    for ten, d in lo_g:
        txt = csv_text(d, COT_GEMINI)
        assert not (set(txt.splitlines()[0].split(",")) & COT_CAM)
        (ra / "gui_gemini" / f"{ten}.csv").write_bytes(("﻿" + txt).encode("utf-8"))
        lo_ra["gemini"].append({"file": f"{ten}.csv", "so_dong": len(d), "so_ky_tu": len(txt)})
    for ten, d in lo_c:
        with open(ra / "gui_claude" / f"{ten}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in d.itertuples(index=False):
                f.write(json.dumps({"id": r.id, "huong": r.huong, "loai": r.loai, "nguon": lam_phang(r.nguon),
                                    "ban_dich": lam_phang(r.ban_dich), "so_manh": int(r.so_manh)}, ensure_ascii=False) + "\n")
        lo_ra["claude"].append({"file": f"{ten}.jsonl", "so_dong": len(d)})
    (ra / "gui_gemini" / "LOI_NHAC_GEMINI.md").write_text(dung_loi_nhac(rubric_text, rubric_sha), encoding="utf-8", newline="\n")

    man = {
        "seed": seed, "sha256_rubric": rubric_sha, "cau_hinh": {k: v for k, v in cfg.items()},
        "so_sp_test_dung_duoc": int(len(bang)), "so_sp_du_dieu_kien_m1": int(bang.du_m1.sum()), "so_sp_can_duoc": int(bang.can_duoc.sum()),
        "mau": {
            "M1_sp": {h: dem_theo_huong_cat(bang.loc[m1[h]].assign(huong=h)).get(h, {}) for h in HUONGS},
            "M1c_sp": {h: len(m1c[h]) for h in HUONGS},
            "thu20_sp": {h: len(thu20[h]) for h in HUONGS},
            "M2_manh": {h: dem_theo_huong_cat(bang.loc[[p for p, _ in m2[h]]].assign(huong=h)).get(h, {}) for h in HUONGS},
            "M2c_manh": {h: len(m2c[h]) for h in HUONGS},
            "B9_sp": {h: dem_theo_huong_cat(bang.loc[b9[h]].assign(huong=h)).get(h, {}) for h in HUONGS},
        },
        "so_muc": {"tong": int(len(muc)), "theo_loai": {k: int(v) for k, v in muc.loai.value_counts().items()},
                   "M1c": int((muc.thuoc_M1c).sum()), "M2c": int((muc.thuoc_M2c).sum()), "thu20": int(muc.thuoc_thu20.sum())},
        "so_sp_m1_cung_m2": len(ids_m1 & ids_m2),
        "dinh_nghia_manh": {
            "M1": "mảnh = một thuộc tính nguồn của BẢN CỤM (tách '; ', strip); so_manh = số thuộc tính đó. Giới hạn: một ít giá trị "
                  "có chứa '; ' nên có thể bị tách thừa. Nguồn cụm có thể có thêm thuộc tính mà bản pair không có (18,3% sp).",
            "M2": "mảnh = một dòng nguồn pair; chỉ lấy sp có danh sách nguồn cụm trùng nội dung, đúng thứ tự, với dòng nguồn pair "
                  "và số thuộc tính đích cụm = số dòng đích pair",
        },
        "b9_giao_m1_m2": 0, "ghi_chu_b9": "B9 ∩ (M1 ∪ M2) = ∅ (assert trong code); chi luu danh sach mau_b9.parquet",
        "lo": lo_ra,
    }
    (ra / "manifest_kiem.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ghi-de", action="store_true", help="ghi đè mau_kiem.parquet cũ (id mù sẽ đổi, kết quả Gemini cũ mất khớp)")
    p.add_argument("--ra", default=str(C.THU_MUC_TEST / "kiem"), help="chỉ để thử")
    a = p.parse_args(argv)
    try:
        rubric_text, sha = doc_rubric()                              # thiếu rubric: dừng ngay
        ra = Path(a.ra)
        if (ra / "mau_kiem.parquet").exists() and not a.ghi_de:
            raise RuntimeError(f"da_co_mau: {ra / 'mau_kiem.parquet'} da ton tai; dung --ghi-de neu that su muon doi id")
        sp = C.doc_split()
        ids_test = sp.product_id[sp.la_test].astype(str)
        bang = dung_bang_test(C.doc_bo("cum"), C.doc_bo("pair"), ids_test)
        man = chay(bang, ra, rubric_text, sha, ghi_de=a.ghi_de)
    except (RuntimeError, ValueError) as e:
        print("LOI:", e, file=sys.stderr)
        return 2
    print(json.dumps(man, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
