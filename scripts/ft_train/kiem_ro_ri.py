"""Kiểm rò rỉ ĐỘC LẬP cho phép chia B1 (không dùng lại mã gom nhóm của chia_tap.py).

Cách làm khác chia_tap: vét cạn MỌI cặp (sp test, sp pool) bằng chỉ mục ngược + đếm giao (numpy.bincount),
tính Jaccard chính xác cho từng góc nhìn, rồi kiểm không cặp nào đạt ngưỡng gom nhóm. Làm tương tự cho từng fold
so với 4 fold còn lại. Ngoài các góc nhìn dùng khi gom (title zh/vi, mô tả pair zh/vi), kiểm thêm mô tả bản CỤM
(tách "; ") vì phía đích của cụm khác pair ở ~4,8% mảnh.

Chạy: python -I -X utf8 scripts/ft_train/kiem_ro_ri.py   -> data/processed/ft_run/bao_cao_ro_ri.json (+ in tóm tắt)
Thoát mã 1 nếu phát hiện rò rỉ.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung as C  # noqa: E402


def _nfkc_thuong(s):
    return unicodedata.normalize("NFKC", s).lower()


def gram4(s):
    s = "".join(ch for ch in _nfkc_thuong(s) if ch.isalnum())
    if not s:
        return set()
    return {s} if len(s) < 4 else {s[i:i + 4] for i in range(len(s) - 3)}


def dong(s, sep):
    out = set()
    for x in s.split(sep):
        x = " ".join(_nfkc_thuong(x).split())
        if x:
            out.add(x)
    return out


class ChiMuc:
    """Chỉ mục ngược trên các tài liệu 'kho'; truy vấn trả Jaccard với mọi tài liệu kho."""

    def __init__(self, sets_kho):
        self.tu = {}
        tok, doc = [], []
        for d, s in enumerate(sets_kho):
            for t in s:
                tok.append(self.tu.setdefault(t, len(self.tu)))
                doc.append(d)
        tok, doc = np.array(tok, dtype=np.int64), np.array(doc, dtype=np.int64)
        thu_tu = np.argsort(tok, kind="stable")
        self.doc = doc[thu_tu]
        self.ptr = np.searchsorted(tok[thu_tu], np.arange(len(self.tu) + 1))
        self.co = np.array([len(s) for s in sets_kho], dtype=np.int64)
        self.n = len(sets_kho)

    def jaccard(self, s):
        if not s:
            return np.zeros(self.n)
        ids = [self.tu[t] for t in s if t in self.tu]
        if not ids:
            return np.zeros(self.n)
        giao = np.bincount(np.concatenate([self.doc[self.ptr[i]:self.ptr[i + 1]] for i in ids]), minlength=self.n)
        mau = len(s) + self.co - giao
        return np.where(mau > 0, giao / np.maximum(mau, 1), 0.0)


def so_hai_phia(sets, la_truy, la_kho, nguong, ten, pid, so_vi_du=5):
    """Jaccard lớn nhất của mỗi tài liệu truy vấn với kho. Trả thống kê + ví dụ gần ngưỡng."""
    kho_idx = np.where(la_kho)[0]
    cm = ChiMuc([sets[i] for i in kho_idx])
    ket = []
    for i in np.where(la_truy)[0]:
        j = cm.jaccard(sets[i])
        k = int(j.argmax()) if len(j) else 0
        ket.append((float(j[k]) if len(j) else 0.0, int(i), int(kho_idx[k])))
    m = np.array([x[0] for x in ket])
    vi_pham = [x for x in ket if x[0] >= nguong - 1e-9]
    gan = sorted((x for x in ket if x[0] < nguong - 1e-9), reverse=True)[:so_vi_du]
    return {
        "goc_nhin": ten, "nguong": nguong, "so_truy_van": len(ket),
        "so_vi_pham": len(vi_pham),
        "vi_pham": [(pid[a], pid[b], round(v, 4)) for v, a, b in vi_pham[:20]],
        "phan_vi_max_jaccard": {q: round(float(np.quantile(m, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)} if len(m) else {},
        "so_trong_dai_duoi_nguong_0.1": int(((m >= nguong - 0.1) & (m < nguong - 1e-9)).sum()),
        "gan_nguong_nhat": [(pid[a], pid[b], round(v, 4)) for v, a, b in gan],
    }


def so_chua_cung_hieu(sets, hieu, la_truy, la_kho, nguong, ten, pid):
    """Hệ số chứa (giao / tập nhỏ hơn) lớn nhất của mỗi tài liệu truy vấn với tài liệu kho CÙNG thương hiệu cụ thể."""
    kho_idx = np.where(la_kho)[0]
    cm = ChiMuc([sets[i] for i in kho_idx])
    hieu_kho = hieu[kho_idx]
    vi_pham, m = [], []
    for i in np.where(la_truy)[0]:
        if not hieu[i] or not sets[i]:
            continue
        cung = hieu_kho == hieu[i]
        ids = [cm.tu[t] for t in sets[i] if t in cm.tu]
        if not cung.any() or not ids:
            continue
        giao = np.bincount(np.concatenate([cm.doc[cm.ptr[k]:cm.ptr[k + 1]] for k in ids]), minlength=cm.n)
        oc = np.where(cung, giao / np.maximum(np.minimum(len(sets[i]), cm.co), 1), 0.0)
        k = int(oc.argmax())
        m.append(float(oc[k]))
        if oc[k] >= nguong - 1e-9:
            vi_pham.append((pid[i], pid[kho_idx[k]], round(float(oc[k]), 4)))
    return {"goc_nhin": ten, "nguong": nguong, "so_truy_van": len(m), "so_vi_pham": len(vi_pham), "vi_pham": vi_pham[:20],
            "phan_vi_max": {q: round(float(np.quantile(m, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)} if m else {}}


def kiem(sp, pair, cum, nguong):
    """sp: split (cùng thứ tự product_id với pair/cum). nguong: {title, mo_ta}. Trả (báo cáo, có_rò_rỉ)."""
    assert (sp.product_id.values == pair.product_id.astype(str).values).all()
    assert (cum.product_id.values == pair.product_id.values).all()
    pid = sp.product_id.values
    loi = []
    # 1) bất biến cấu trúc
    if not sp.product_id.is_unique:
        loi.append("product_id trùng")
    if set(pid[sp.la_test.values]) & set(pid[~sp.la_test.values]):
        loi.append("mã sp nằm cả test và pool")
    if (sp.groupby("nhom_id").la_test.nunique() > 1).any():
        loi.append("nhóm nằm hai phía test/pool")
    if (sp[~sp.la_test].groupby("nhom_id").fold5.nunique() > 1).any():
        loi.append("nhóm nằm hai fold")
    if not ((sp.fold5[sp.la_test] == -1).all() and sp.fold5[~sp.la_test].between(0, 4).all()):
        loi.append("fold5 sai miền giá trị")
    # 2) trùng title chính xác sau chuẩn hóa (zh và vi, xét chéo mọi hướng)
    trung_title = {}
    for cot in ("title_zh", "title_vi"):
        k = pair[cot].map(lambda s: "".join(ch for ch in _nfkc_thuong(s) if ch.isalnum())).values
        a, b = set(k[sp.la_test.values]) - {""}, set(k[~sp.la_test.values]) - {""}
        trung_title[cot] = len(a & b)
        if a & b:
            loi.append(f"{cot}: {len(a & b)} title test trùng chính xác với pool")
    # 3) vét cạn Jaccard test x pool, và từng fold x các fold còn lại
    goc = {
        "title_zh": ([gram4(s) for s in pair.title_zh], nguong["title"]),
        "title_vi": ([gram4(s) for s in pair.title_vi], nguong["title"]),
        "mo_ta_zh_pair": ([dong(s, "\n") for s in pair.description_zh], nguong["mo_ta"]),
        "mo_ta_vi_pair": ([dong(s, "\n") for s in pair.description_vi], nguong["mo_ta"]),
        "mo_ta_zh_cum": ([dong(s, "; ") for s in cum.description_zh], nguong["mo_ta"]),
        "mo_ta_vi_cum": ([dong(s, "; ") for s in cum.description_vi], nguong["mo_ta"]),
    }
    test = sp.la_test.values
    pool = ~test
    test_vs_pool, fold_vs_con_lai = [], []
    for ten, (sets, ng) in goc.items():
        r = so_hai_phia(sets, test, pool, ng, ten, pid)
        test_vs_pool.append(r)
        if r["so_vi_pham"]:
            loi.append(f"test x pool, {ten}: {r['so_vi_pham']} sp test có Jaccard >= {ng} với pool")
        for f in range(5):
            rf = so_hai_phia(sets, sp.fold5.values == f, pool & (sp.fold5.values != f), ng, ten, pid, so_vi_du=0)
            fold_vs_con_lai.append({"fold": f, **{k: rf[k] for k in ("goc_nhin", "so_truy_van", "so_vi_pham", "vi_pham")}})
            if rf["so_vi_pham"]:
                loi.append(f"fold {f} x các fold khác, {ten}: {rf['so_vi_pham']} vi phạm")
    # 3b) cùng thương hiệu cụ thể + title chứa trong nhau (luật v2 của chia_tap)
    if "chua" in nguong:
        from chia_tap import thuong_hieu        # định nghĩa "thương hiệu cụ thể" là một phần của luật, dùng chung
        hieu = pair.description_vi.map(thuong_hieu).values
        for ten in ("title_zh", "title_vi"):
            r = so_chua_cung_hieu(goc[ten][0], hieu, test, pool, nguong["chua"], ten + "_cung_hieu", pid)
            test_vs_pool.append(r)
            if r["so_vi_pham"]:
                loi.append(f"test x pool, {ten} cùng hiệu: {r['so_vi_pham']} sp test có hệ số chứa >= {nguong['chua']}")
            for f in range(5):
                rf = so_chua_cung_hieu(goc[ten][0], hieu, sp.fold5.values == f, pool & (sp.fold5.values != f),
                                       nguong["chua"], ten + "_cung_hieu", pid)
                fold_vs_con_lai.append({"fold": f, **{k: rf[k] for k in ("goc_nhin", "so_truy_van", "so_vi_pham", "vi_pham")}})
                if rf["so_vi_pham"]:
                    loi.append(f"fold {f} x các fold khác, {ten} cùng hiệu: {rf['so_vi_pham']} vi phạm")
    # 4) thông tin (không phải lỗi): tỉ lệ dòng mô tả nguồn của test đã xuất hiện nguyên dòng trong pool (cùng category)
    nguon = np.where(pair.source.eq("1688"), pair.description_zh, pair.description_vi)
    da_co = {(c, l) for c, s in zip(pair.category.values[pool], nguon[pool]) for l in s.split("\n")}
    dong_test = [(c, l) for c, s in zip(pair.category.values[test], nguon[test]) for l in s.split("\n")]
    bc = {
        "nguong": nguong, "n_test": int(test.sum()), "n_pool": int(pool.sum()),
        "trung_title_chinh_xac": trung_title,
        "test_vs_pool": test_vs_pool, "fold_vs_cac_fold_khac": fold_vs_con_lai,
        "thong_tin_dong_mo_ta_test_da_co_trong_pool": round(float(np.mean([x in da_co for x in dong_test])), 4),
        "loi": loi, "ket_luan": "KHÔNG RÒ RỈ" if not loi else "CÓ RÒ RỈ",
    }
    return bc, bool(loi)


def main():
    sp = C.doc_split()
    pair, cum = C.doc_bo("pair"), C.doc_bo("cum")
    ghi = json.loads(C.FILE_DONG_BANG.read_text(encoding="utf-8"))["split"]["nguong_jaccard"]
    ng = {"title": ghi["title_zh"], "mo_ta": ghi["mo_ta_zh"]}
    if "chua_cung_hieu" in ghi:
        ng["chua"] = ghi["chua_cung_hieu"]
    bc, co_loi = kiem(sp, pair, cum, ng)
    (C.FT_RUN / "bao_cao_ro_ri.json").write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in bc["test_vs_pool"]:
        print(f"{r['goc_nhin']:22s} ngưỡng {r['nguong']}: vi phạm {r['so_vi_pham']} | phân vị max "
              f"{r.get('phan_vi_max_jaccard', r.get('phan_vi_max'))}")
    print("fold x fold khác: tổng vi phạm", sum(r["so_vi_pham"] for r in bc["fold_vs_cac_fold_khac"]))
    print("trùng title chính xác:", bc["trung_title_chinh_xac"])
    print("dòng mô tả test đã có trong pool (thông tin):", bc["thong_tin_dong_mo_ta_test_da_co_trong_pool"])
    print("KẾT LUẬN:", bc["ket_luan"], bc["loi"])
    sys.exit(1 if co_loi else 0)


if __name__ == "__main__":
    main()
