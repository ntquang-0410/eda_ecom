"""B1. Chia tập dùng chung cho hai bộ (cụm, pair): khóa 10% làm TEST, phần còn lại (pool) chia 5 fold.

Đơn vị chia là NHÓM GẦN-TRÙNG, không phải sản phẩm: hai sản phẩm cùng nhóm luôn nằm cùng một phía.
Hai sản phẩm bị nối vào một nhóm nếu một trong các "góc nhìn" sau giống nhau đủ mức (Jaccard >= NGUONG của góc nhìn đó):
  - title tiếng Trung, title tiếng Việt: tập 4-gram ký tự sau chuẩn hóa (NFKC, chữ thường, bỏ dấu câu/khoảng trắng);
  - mô tả tiếng Trung, mô tả tiếng Việt: tập thuộc tính chuẩn hóa, xét cả bản pair (tách "\n") lẫn bản cụm (tách "; ").
Xét CẢ HAI phía ngôn ngữ cho mọi sản phẩm (không chỉ phía nguồn) để chặn rò rỉ chéo hướng: một sp Tiki (vi->zh)
có phía Trung gần giống một sp 1688 (zh->vi) thì cũng bị gom chung.
Tìm cặp bằng phép nối tập chính xác (lọc tiền tố AllPairs, Bayardo 2007): KHÔNG xấp xỉ, không bỏ sót cặp nào >= ngưỡng.
Thêm (v2): hai sp CÙNG thương hiệu cụ thể mà title này gần như nằm gọn trong title kia (hệ số chứa 4-gram >= 0,7)
cũng bị nối, để bắt tin đăng anh em của cùng sản phẩm có title ngắn (ví dụ "Trà kỷ tử" và "Trà kỷ tử ... (500gr)").
Nhóm = thành phần liên thông (bắc cầu).

Phân tầng: hướng x category (tầng của nhóm = tầng chiếm đa số trong nhóm).
Chạy: python -I -X utf8 scripts/ft_train/chia_tap.py   (thêm --ghi-de để tạo lại; mặc định từ chối ghi đè)
Kiểm độc lập sau khi chia: scripts/ft_train/kiem_ro_ri.py
"""
import argparse
import collections
import datetime as dt
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung as C  # noqa: E402

PHIEN_BAN = "chia_tap_v2"   # v2: thêm luật cùng thương hiệu + title chứa trong nhau
SEED = 20261104
TI_LE_TEST = 0.10
SO_FOLD = 5
NGUONG_TITLE = 0.5    # Jaccard 4-gram title nối nhóm (kế hoạch ghi 0,7; hạ xuống 0,5 cho chặt hơn)
NGUONG_MO_TA = 0.7    # Jaccard tập dòng mô tả (0,6 làm mô tả khuôn mẫu ngắn nối dây chuyền thành nhóm 2.992 sp)
NGUONG_CHUA = 0.7    # hệ số chứa 4-gram title khi CÙNG thương hiệu cụ thể (v2, vá ca "Trà kỷ tử" Dược Liệu Tuệ Tâm)
NGUONG = {"title_zh": NGUONG_TITLE, "title_vi": NGUONG_TITLE, "mo_ta_zh": NGUONG_MO_TA, "mo_ta_vi": NGUONG_MO_TA,
          "mo_ta_zh_cum": NGUONG_MO_TA, "mo_ta_vi_cum": NGUONG_MO_TA, "chua_cung_hieu": NGUONG_CHUA}


def chuan_hoa(s):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s).lower())


def ngram4(s):
    s = chuan_hoa(s)
    if not s:
        return frozenset()
    return frozenset([s]) if len(s) < 4 else frozenset(s[i:i + 4] for i in range(len(s) - 3))


def tap_dong(s, sep="\n"):
    out = (re.sub(r"\s+", " ", unicodedata.normalize("NFKC", x)).strip().lower() for x in s.split(sep))
    return frozenset(x for x in out if x)


def cap_gan_trung(sets, nguong):
    """Mọi cặp (i, j, jaccard) có Jaccard >= nguong. Chính xác (lọc tiền tố), tập rỗng bỏ qua."""
    freq = collections.Counter(t for s in sets for t in s)
    hang = {t: k for k, t in enumerate(sorted(freq, key=lambda t: (freq[t], t)))}   # hiếm trước
    recs = sorted(((len(s), i, sorted(hang[t] for t in s)) for i, s in enumerate(sets) if s))
    chi_muc = collections.defaultdict(list)
    canh = []
    eps = 1e-9
    for pos, (n, i, toks) in enumerate(recs):
        p = n - math.ceil(nguong * n - eps) + 1
        ung_vien = set()
        for t in toks[:p]:
            for q in chi_muc[t]:
                if recs[q][0] >= nguong * n - eps:          # lọc kích thước (các bản ghi trước có cỡ <= n)
                    ung_vien.add(q)
        a = sets[i]
        for q in ung_vien:
            j = recs[q][1]
            b = sets[j]
            giao = len(a & b)
            jac = giao / (len(a) + len(b) - giao)
            if jac >= nguong - eps:
                canh.append((j, i, jac))
        for t in toks[:p]:
            chi_muc[t].append(pos)
    return canh


def thanh_phan(n, canh):
    cha = list(range(n))

    def goc(x):
        while cha[x] != x:
            cha[x] = cha[cha[x]]
            x = cha[x]
        return x

    for i, j, *_ in canh:
        a, b = goc(i), goc(j)
        if a != b:
            cha[max(a, b)] = min(a, b)
    return np.array([goc(i) for i in range(n)])


HIEU_CHUNG = {chuan_hoa(x) for x in (           # giá trị "thương hiệu" không chỉ một hãng cụ thể
    "", "khác", "không thương hiệu", "không nhãn hiệu", "oem", "no brand", "không", "không có", "chưa có",
    "tạm thời không có", "tạm thời chưa có", "không rõ", "other", "others", "none", "n/a", "generic", "no name",
    "khác (other)", "other/khác", "khác/other", "thương hiệu riêng")}


def thuong_hieu(mo_ta_vi):
    """Thương hiệu lấy từ dòng 'Thương hiệu:' / 'Nhãn hiệu:' của mô tả tiếng Việt (bản pair), đã chuẩn hóa; rỗng nếu chung chung."""
    for d in mo_ta_vi.split("\n"):
        m = re.match(r"\s*(?:thương hiệu|nhãn hiệu)\s*:\s*(.+)$", d, re.I)
        if m:
            h = chuan_hoa(m.group(1))
            return "" if h in HIEU_CHUNG or h.isdigit() else h
    return ""


def cap_cung_hieu(df, goc_nhin, nguong):
    """Cặp CÙNG thương hiệu cụ thể mà title này gần như nằm gọn trong title kia (hệ số chứa = giao / tập nhỏ hơn >= nguong,
    4-gram, xét title zh và vi). Bắt tin đăng anh em của cùng sản phẩm (khác gói, khác cỡ) mà Jaccard bỏ sót vì title ngắn."""
    hieu = df.description_vi.map(thuong_hieu).values
    canh = []
    for h, idx in pd.Series(range(len(df))).groupby(hieu).groups.items():
        if not h or len(idx) < 2:
            continue
        idx = list(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                for ten in ("title_zh", "title_vi"):
                    s, t = goc_nhin[ten][i], goc_nhin[ten][j]
                    if s and t and len(s & t) / min(len(s), len(t)) >= nguong - 1e-9:
                        canh.append((i, j, ten))
                        break
    return canh


def gom_nhom(df, nguong=NGUONG, df_cum=None):
    """df (bản pair) theo thứ tự cố định; df_cum cùng thứ tự (mô tả tách "; " khác cách tách của pair nên xét riêng).
    nguong = {góc nhìn: Jaccard}. Trả (nhom_id theo dòng, số cặp từng góc nhìn)."""
    goc_nhin = {
        "title_zh": [ngram4(s) for s in df.title_zh],
        "title_vi": [ngram4(s) for s in df.title_vi],
        "mo_ta_zh": [tap_dong(s) for s in df.description_zh],
        "mo_ta_vi": [tap_dong(s) for s in df.description_vi],
    }
    if df_cum is not None:
        assert (df_cum.product_id.values == df.product_id.values).all()
        goc_nhin["mo_ta_zh_cum"] = [tap_dong(s, "; ") for s in df_cum.description_zh]
        goc_nhin["mo_ta_vi_cum"] = [tap_dong(s, "; ") for s in df_cum.description_vi]
    tat_ca, tk = [], {}
    for ten, sets in goc_nhin.items():
        c = cap_gan_trung(sets, nguong[ten])
        tk[ten] = len(c)
        tat_ca += c
    for cot in ("title_zh", "title_vi"):                         # trùng title chính xác sau chuẩn hóa (tường minh)
        khoa = df[cot].map(chuan_hoa)
        for _, idx in khoa[khoa != ""].groupby(khoa).groups.items():
            idx = list(idx)
            tat_ca += [(idx[0], k) for k in idx[1:]]
    c = cap_cung_hieu(df, goc_nhin, nguong["chua_cung_hieu"])
    tk["chua_cung_hieu"] = len(c)
    tat_ca += c
    goc = thanh_phan(len(df), tat_ca)
    # nhom_id ổn định: theo product_id nhỏ nhất trong nhóm
    pid = df.product_id.astype(str).values
    dai_dien = pd.Series(pid).groupby(goc).transform("min").values
    return dai_dien, tk


def chon_test(nhom, rng, ti_le=TI_LE_TEST):
    """nhom: DataFrame (nhom_id, tang, co). Theo từng tầng, duyệt nhóm theo thứ tự ngẫu nhiên, nhận nhóm nếu đưa tổng gần đích hơn."""
    chon = set()
    for tang, g in nhom.groupby("tang", sort=True):
        dich = ti_le * g.co.sum()
        g = g.iloc[rng.permutation(len(g))]
        hien = 0
        for nid, co in zip(g.nhom_id, g.co):
            if abs(hien + co - dich) < abs(hien - dich):
                chon.add(nid)
                hien += co
    return chon


def chia_fold(nhom, rng, k=SO_FOLD):
    """Trong từng tầng: nhóm lớn trước (hòa thì ngẫu nhiên), gán vào fold đang ít sp nhất của tầng (hòa: ít sp toàn cục)."""
    fold = {}
    toan_cuc = np.zeros(k, dtype=int)
    for tang, g in nhom.groupby("tang", sort=True):
        g = g.iloc[rng.permutation(len(g))]
        g = g.sort_values("co", ascending=False, kind="stable")
        dem = np.zeros(k, dtype=int)
        for nid, co in zip(g.nhom_id, g.co):
            f = min(range(k), key=lambda x: (dem[x], toan_cuc[x], x))
            fold[nid] = f
            dem[f] += co
            toan_cuc[f] += co
    return fold


def chia(df, nguong=NGUONG, seed=SEED, df_cum=None):
    """df = bản pair, df_cum = bản cụm (cùng thứ tự). Trả DataFrame split (cùng thứ tự df) + thống kê."""
    df = df.reset_index(drop=True)
    nhom_id, tk = gom_nhom(df, nguong, None if df_cum is None else df_cum.reset_index(drop=True))
    sp = pd.DataFrame({"product_id": df.product_id.astype(str), "source": df.source, "category": df.category,
                       "huong": df.source.map(C.HUONG), "nhom_id": nhom_id})
    sp["tang"] = sp.huong + "|" + sp.category
    sp["kich_thuoc_nhom"] = sp.groupby("nhom_id").product_id.transform("size")
    # tầng của nhóm = tầng đa số (hòa: theo tên tầng)
    dem = sp.groupby(["nhom_id", "tang"]).size().reset_index(name="n")
    dem = dem.sort_values(["nhom_id", "n", "tang"], ascending=[True, False, True]).drop_duplicates("nhom_id")
    nhom = dem.rename(columns={"n": "_"}).drop(columns="_").merge(
        sp.groupby("nhom_id").size().rename("co").reset_index(), on="nhom_id").sort_values("nhom_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    test = chon_test(nhom, rng)
    fold = chia_fold(nhom[~nhom.nhom_id.isin(test)], rng)
    sp["la_test"] = sp.nhom_id.isin(test)
    sp["fold5"] = sp.nhom_id.map(fold).fillna(-1).astype(int)
    sp = sp.drop(columns="tang")
    tk["so_nhom"] = int(sp.nhom_id.nunique())
    tk["so_nhom_tu_2"] = int((sp.groupby("nhom_id").size() >= 2).sum())
    tk["nhom_lon_nhat"] = int(sp.kich_thuoc_nhom.max())
    return sp, tk


def kiem_split(sp):
    """Bất biến bắt buộc; sai là dừng."""
    assert sp.product_id.is_unique
    assert (sp.fold5[sp.la_test] == -1).all() and sp.fold5[~sp.la_test].between(0, SO_FOLD - 1).all()
    assert sp.groupby("nhom_id").la_test.nunique().max() == 1, "nhóm nằm hai phía test/pool"
    assert sp[~sp.la_test].groupby("nhom_id").fold5.nunique().max() == 1, "nhóm nằm hai fold"
    C.kiem_khong_giao(sp.product_id[~sp.la_test], sp.product_id[sp.la_test])


def bao_cao(sp, tk):
    goc = sp.groupby(["huong", "category"]).size().rename("tong")
    t = sp[sp.la_test].groupby(["huong", "category"]).size().rename("test")
    bang = pd.concat([goc, t], axis=1).fillna(0).astype(int)
    bang["ti_le_test"] = (bang.test / bang.tong).round(4)
    for f in range(SO_FOLD):
        bang[f"fold{f}"] = sp[sp.fold5 == f].groupby(["huong", "category"]).size().reindex(bang.index).fillna(0).astype(int)
    nhieu = sp.kich_thuoc_nhom >= 2
    return {
        "so_cap_theo_goc_nhin": {k: v for k, v in tk.items() if k.startswith(("title", "mo_ta", "chua"))},
        "so_nhom": tk["so_nhom"], "so_nhom_tu_2_sp": tk["so_nhom_tu_2"], "nhom_lon_nhat": tk["nhom_lon_nhat"],
        "n_test": int(sp.la_test.sum()), "n_pool": int((~sp.la_test).sum()),
        "test_theo_huong": sp[sp.la_test].huong.value_counts().to_dict(),
        "fold_co": sp[~sp.la_test].fold5.value_counts().sort_index().to_dict(),
        "ti_le_sp_thuoc_nhom_tu_2": {"toan_bo": round(float(nhieu.mean()), 4),
                                     "test": round(float(nhieu[sp.la_test].mean()), 4),
                                     "pool": round(float(nhieu[~sp.la_test].mean()), 4)},
        "bang_tang": bang.reset_index().to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghi-de", action="store_true", help="cho phép tạo lại phép chia (mặc định từ chối)")
    ap.add_argument("--nguong-title", type=float, default=NGUONG_TITLE)
    ap.add_argument("--nguong-mo-ta", type=float, default=NGUONG_MO_TA)
    ap.add_argument("--nguong-chua", type=float, default=NGUONG_CHUA)
    a = ap.parse_args()
    if (C.FILE_SPLIT.exists() or C.FILE_DONG_BANG.exists()) and not a.ghi_de:
        sys.exit("TỪ CHỐI: phép chia đã khóa (split.parquet / DU_LIEU_DONG_BANG.json đã có). Tạo lại sẽ làm lệch tập test.")

    hash_hf = {}
    for ten, h in C.SHA256_HF.items():
        thuc = C.sha256_file(C.DU_LIEU / ten)
        if thuc != h:
            sys.exit(f"SAI HASH {ten}: {thuc} != {h}")
        hash_hf[ten] = thuc
    pair = C.doc_bo("pair")
    cum = C.doc_bo("cum")
    assert (cum.product_id.values == pair.product_id.values).all() and pair.product_id.is_unique

    nguong = {"title_zh": a.nguong_title, "title_vi": a.nguong_title, "mo_ta_zh": a.nguong_mo_ta, "mo_ta_vi": a.nguong_mo_ta,
              "mo_ta_zh_cum": a.nguong_mo_ta, "mo_ta_vi_cum": a.nguong_mo_ta, "chua_cung_hieu": a.nguong_chua}
    sp, tk = chia(pair, nguong, df_cum=cum)
    kiem_split(sp)
    bc = bao_cao(sp, tk)

    C.FT_RUN.mkdir(parents=True, exist_ok=True)
    C.THU_MUC_TEST.mkdir(parents=True, exist_ok=True)
    tam = C.FILE_SPLIT.with_suffix(".tmp")
    sp.to_parquet(tam, index=False)
    tam.replace(C.FILE_SPLIT)
    ds_test = sorted(sp.product_id[sp.la_test])
    (C.THU_MUC_TEST / "ds_test.txt").write_text("\n".join(ds_test) + "\n", encoding="utf-8")
    dong_bang = {
        "tao_luc": dt.datetime.now().isoformat(timespec="seconds"),
        "nguon_du_lieu": {"hf_repo": C.HF_REPO, "thu_muc": C.HF_THU_MUC, "commit": C.HF_COMMIT, "sha256": hash_hf},
        "split": {"file": "split.parquet", "sha256_file": C.sha256_file(C.FILE_SPLIT),
                  "sha256_ds_test": C.bam_ds_test(ds_test), "n_test": len(ds_test), "n_pool": int((~sp.la_test).sum()),
                  "phien_ban": PHIEN_BAN, "seed": SEED, "nguong_jaccard": nguong, "ti_le_test": TI_LE_TEST,
                  "so_fold": SO_FOLD},
        "quy_tac": "Tập test (la_test=True) chỉ dùng để đánh giá. Không mẫu nào của test được đưa vào huấn luyện, "
                   "dò siêu tham số hay chọn cấu hình. Đọc phép chia qua chung.doc_split().",
    }
    C.FILE_DONG_BANG.write_text(json.dumps(dong_bang, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.FT_RUN / "bao_cao_chia_tap.json").write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in bc.items() if k != "bang_tang"}, ensure_ascii=False, indent=1))
    print(pd.DataFrame(bc["bang_tang"]).to_string(index=False))


if __name__ == "__main__":
    main()
