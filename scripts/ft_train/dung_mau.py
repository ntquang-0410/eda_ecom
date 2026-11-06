"""B3: dựng mẫu huấn luyện (messages) từ hai bộ cụm/pair sau Judge, tách pool và test theo phép chia đã khóa.

Mỗi sản phẩm sinh 2 mẫu: title và mo_ta (cho từng nhánh). Mẫu pool -> data/processed/ft_run/mau/pool_{nhanh}.jsonl,
mẫu test -> data/processed/ft_run/test_khoa/test_{nhanh}.jsonl. Không có gì dẫn xuất từ test (ngoài các số đếm tổng) trong mau/.
Phép chia chỉ đọc qua chung.doc_split().

Chạy:  python -I -X utf8 scripts/ft_train/dung_mau.py --tokenizer <tokenizer.json> [--nhanh cum|pair|ca_hai] [--max-len 2048]
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung  # noqa: E402

# ---- Chỉ dẫn (B4/zero-shot import lại các hằng số và chi_dan để dùng đúng cùng chữ) ----
TEN_NGON_NGU = {"zh": "Chinese", "vi": "Vietnamese"}
CHI_DAN_TITLE = "Translate this product title from {SRC} to {TGT}. Output only the translation.\n\n{NGUON}"
_MO_TA_DAU = "Translate this product's attribute list from {SRC} to {TGT}. "
_MO_TA_CUOI = "\nProduct title (context only, do not translate): {TITLE}\n\n{NGUON}"
CHI_DAN_MO_TA = {
    "pair": _MO_TA_DAU + 'Each line is one attribute written as "name: value". '
            "Keep the same number and order of lines, and output only the translation." + _MO_TA_CUOI,
    "cum": _MO_TA_DAU + 'The attributes are separated by "; " and each is written as "name: value". '
           'Keep the same number and order of attributes, separated by "; ", and output only the translation.' + _MO_TA_CUOI,
}

# Khuôn chat Qwen3 khi tắt suy luận. CHỈ LÀ XẤP XỈ để đếm token; khuôn thật được áp bằng tokenizer của transformers trên Kaggle.
KHUON_DEM = "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{a}<|im_end|>\n"
NHANH = ("cum", "pair")
LOAI = ("title", "mo_ta")
SEED_MAU = 20261104
MAX_LEN_MAC_DINH = 2048


def ngon_ngu(huong):
    """'zh->vi' -> ('Chinese', 'Vietnamese')."""
    s, t = huong.split("->")
    return TEN_NGON_NGU[s], TEN_NGON_NGU[t]


def chi_dan(loai, nhanh, huong, nguon, title_nguon):
    """Chỉ dẫn (nội dung user) cho một mẫu. title_nguon chỉ dùng ở mo_ta."""
    src, tgt = ngon_ngu(huong)
    if loai == "title":
        return CHI_DAN_TITLE.format(SRC=src, TGT=tgt, NGUON=nguon)
    if loai == "mo_ta":
        return CHI_DAN_MO_TA[nhanh].format(SRC=src, TGT=tgt, TITLE=title_nguon, NGUON=nguon)
    raise ValueError(f"loai_la:{loai}")


def lam_dem_token(duong_dan_tokenizer):
    """Trả hàm dem_nhieu([(user, assistant), ...]) -> [số token xấp xỉ theo khuôn Qwen3]."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(duong_dan_tokenizer))

    def dem_nhieu(cap):
        if not cap:
            return []
        van_ban = [KHUON_DEM.format(u=u, a=a) for u, a in cap]
        return [len(e.ids) for e in tok.encode_batch(van_ban, add_special_tokens=False)]
    return dem_nhieu, tok


def _chac(dieu_kien, thong_bao):
    if not dieu_kien:
        raise RuntimeError(thong_bao)


def _cat_thuoc_tinh(text, nhanh):
    return text.split(chung.TACH_DONG[nhanh])


def _dong_mau(r, nhanh, loai, nguon, dich, user, so_token, phan, so_phan, la_test, max_len):
    pid = str(r.product_id)
    id_ = f"{pid}_title" if loai == "title" else (f"{pid}_mo_ta" if phan == 0 else f"{pid}_mo_ta_p{phan}")
    m = {"id": id_, "product_id": pid, "source": r.source, "category": r.category, "huong": r.huong, "loai": loai,
         "nhanh": nhanh, "fold5": int(r.fold5), "phan": phan, "so_phan": so_phan, "nguon": nguon, "dich": dich,
         "so_token": int(so_token),
         "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": dich}]}
    if la_test:
        m["qua_dai"] = bool(so_token > max_len)
    return m


def chia_phan(r, nhanh, nguon, dich, max_len, dem_nhieu):
    """Chia mô tả pool quá dài thành các phần liên tiếp theo ranh giới thuộc tính (tham lam, mỗi phần <= max_len).
    Trả (danh_sach_phan, ly_do). danh_sach_phan = [(nguon_phan, dich_phan, user, so_token), ...]; lỗi -> ([], (ly_do, chi_tiet))."""
    a_n, a_d = _cat_thuoc_tinh(nguon, nhanh), _cat_thuoc_tinh(dich, nhanh)
    if len(a_n) != len(a_d):
        return [], ("khong_can_so_thuoc_tinh", f"nguon={len(a_n)} dich={len(a_d)}")
    tach = chung.TACH_DONG[nhanh]

    def phan_tu(i, k):
        n, d = tach.join(a_n[i:i + k]), tach.join(a_d[i:i + k])
        u = chi_dan("mo_ta", nhanh, r.huong, n, r.nguon_title)
        return n, d, u

    def vua(i, k):
        n, d, u = phan_tu(i, k)
        t = dem_nhieu([(u, d)])[0]
        return t <= max_len, (n, d, u, t)

    ket, i = [], 0
    while i < len(a_n):
        ok1, p1 = vua(i, 1)
        if not ok1:
            return [], ("thuoc_tinh_don_le_vuot_max_len", f"thuoc_tinh_{i}: {p1[3]} token > {max_len}")
        lo, hi, tot = 1, len(a_n) - i, p1       # lo luôn là giá trị đã kiểm vừa
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ok, p = vua(i, mid)
            if ok:
                lo, tot = mid, p
            else:
                hi = mid - 1
        ket.append(tot)
        i += lo
    return ket, None


def dung_mau_nhanh(df, split, nhanh, max_len, dem_nhieu):
    """Lõi: nhận DataFrame bộ (cột như chung.doc_bo) + split (product_id, la_test, fold5). Không ghi gì ra đĩa.
    Trả dict: pool, test (list mẫu), loai (list lý do loại, chỉ pool), tong_token_nguyen (pool+test, trước khi chia phần/loại)."""
    _chac(nhanh in NHANH, f"nhanh_la:{nhanh}")
    _chac(split.product_id.is_unique and df.product_id.is_unique, "product_id_trung")
    _chac(set(df.product_id.astype(str)) == set(split.product_id.astype(str)), "tap_sp_bo_va_split_khong_khop")
    d = df.drop(columns=[c for c in ("la_test", "fold5") if c in df.columns]).copy()
    d["product_id"] = d.product_id.astype(str)
    sp = split[["product_id", "la_test", "fold5"]].copy()
    sp["product_id"] = sp.product_id.astype(str)
    d = d.merge(sp, on="product_id", how="inner", validate="one_to_one").sort_values("product_id", kind="stable")

    # bản đầy đủ của từng mẫu (chưa chia) và đếm token một lượt
    ung_vien = []
    for r in d.itertuples(index=False):
        ung_vien.append((r, "title", r.nguon_title, r.dich_title,
                         chi_dan("title", nhanh, r.huong, r.nguon_title, r.nguon_title)))
        ung_vien.append((r, "mo_ta", r.nguon_mo_ta, r.dich_mo_ta,
                         chi_dan("mo_ta", nhanh, r.huong, r.nguon_mo_ta, r.nguon_title)))
    tokens = dem_nhieu([(u, dich) for _, _, _, dich, u in ung_vien])

    pool, test, loai = [], [], []
    tong_nguyen = {"title": 0, "mo_ta": 0}
    for (r, lo, nguon, dich, user), t in zip(ung_vien, tokens):
        tong_nguyen[lo] += t
        if r.la_test:        # test: giữ nguyên, không chia, không loại; chỉ gắn cờ qua_dai
            test.append(_dong_mau(r, nhanh, lo, nguon, dich, user, t, 0, 1, True, max_len))
            continue
        ghi_loai = lambda ly_do, ct: loai.append({"product_id": str(r.product_id), "loai": lo, "ly_do": ly_do, "chi_tiet": ct})
        if not str(nguon).strip() or not str(dich).strip():
            ghi_loai("rong", "nguon hoặc đích rỗng")
        elif t <= max_len:
            pool.append(_dong_mau(r, nhanh, lo, nguon, dich, user, t, 0, 1, False, max_len))
        elif lo == "title":
            ghi_loai("title_vuot_max_len", f"{t} token > {max_len}")
        else:
            cac_phan, ly_do = chia_phan(r, nhanh, nguon, dich, max_len, dem_nhieu)
            if ly_do:
                ghi_loai(*ly_do)
                continue
            for k, (n, dd, u, tk) in enumerate(cac_phan, 1):
                pool.append(_dong_mau(r, nhanh, lo, n, dd, u, tk, k, len(cac_phan), False, max_len))
    return {"pool": pool, "test": test, "loai": loai, "tong_token_nguyen": tong_nguyen}


def kiem_khoa(ket_qua, split, nhanh=""):
    """Khóa test: dừng (RuntimeError) nếu có bất kỳ vi phạm nào. Gọi TRƯỚC khi ghi file."""
    pool, test, loai = ket_qua["pool"], ket_qua["test"], ket_qua["loai"]
    ids_test_split = set(split.product_id[split.la_test].astype(str))
    ids_pool_split = set(split.product_id[~split.la_test].astype(str))
    ids_pool = {m["product_id"] for m in pool}
    ids_test = {m["product_id"] for m in test}
    chung.kiem_khong_giao(ids_pool, ids_test)                 # không sp nào nằm ở cả hai phía
    chung.kiem_khong_giao(ids_pool, ids_test_split)           # không mẫu pool nào thuộc test của split
    _chac(ids_pool <= ids_pool_split, f"{nhanh}: mẫu pool có mã ngoài tập pool của split")
    _chac(ids_test == ids_test_split, f"{nhanh}: tập mã mẫu test khác tập test của split")
    _chac(all(0 <= m["fold5"] <= 4 for m in pool), f"{nhanh}: mẫu pool có fold5 ngoài 0..4")
    _chac(all(m["fold5"] == -1 for m in test), f"{nhanh}: mẫu test có fold5 != -1")
    # fold của từng mẫu phải đúng với split
    fold = dict(zip(split.product_id.astype(str), split.fold5))
    _chac(all(fold[m["product_id"]] == m["fold5"] for m in pool + test), f"{nhanh}: fold5 lệch split")
    # mọi sp có đủ title + mo_ta (hoặc có lý do loại)
    co = {(m["product_id"], m["loai"]) for m in pool}
    bi_loai = {(x["product_id"], x["loai"]) for x in loai}
    for pid in ids_pool_split:
        for lo in LOAI:
            _chac((pid, lo) in co or (pid, lo) in bi_loai, f"{nhanh}: thiếu mẫu {pid}/{lo} (không có lý do loại)")
    co_t = {(m["product_id"], m["loai"]) for m in test}
    _chac(co_t == {(p, lo) for p in ids_test_split for lo in LOAI}, f"{nhanh}: tập test thiếu/thừa mẫu")
    # mỗi mẫu pool chia phần: các phần liên tiếp 1..so_phan
    theo = {}
    for m in pool:
        if m["loai"] == "mo_ta":
            theo.setdefault(m["product_id"], []).append(m)
    for pid, ds in theo.items():
        _chac(sorted(x["phan"] for x in ds) == (list(range(1, len(ds) + 1)) if len(ds) > 1 else [0]),
              f"{nhanh}: {pid} phần không liên tục")
        _chac(all(x["so_phan"] == len(ds) for x in ds), f"{nhanh}: {pid} so_phan sai")


def ghi_jsonl(duong_dan, hang):
    """Ghi tạm rồi đổi tên (atomic)."""
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    tam = duong_dan.with_name(duong_dan.name + ".tmp")
    with open(tam, "w", encoding="utf-8", newline="\n") as f:
        for m in hang:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(tam, duong_dan)


def _ghi_text(duong_dan, noi_dung):
    tam = Path(str(duong_dan) + ".tmp")
    tam.write_text(noi_dung, encoding="utf-8", newline="\n")
    os.replace(tam, duong_dan)


def _tong(rows, khoa):
    out = {}
    for m in rows:
        k = str(m[khoa])
        out[k] = out.get(k, {"so_mau": 0, "so_token": 0})
        out[k]["so_mau"] += 1
        out[k]["so_token"] += m["so_token"]
    return dict(sorted(out.items()))


def lam_bao_cao(ket_qua, nhanh, max_len):
    """Số liệu cho pool + số đếm mẫu test (chỉ đếm, không nội dung)."""
    pool, test, loai = ket_qua["pool"], ket_qua["test"], ket_qua["loai"]
    nguyen = ket_qua["tong_token_nguyen"]
    return {
        "nhanh": nhanh, "max_len": max_len,
        "pool": {"so_mau": len(pool), "so_token": sum(m["so_token"] for m in pool),
                 "theo_huong": _tong(pool, "huong"), "theo_loai": _tong(pool, "loai"), "theo_fold": _tong(pool, "fold5"),
                 "so_sp_bi_chia_phan": len({m["product_id"] for m in pool if m["so_phan"] > 1}),
                 "so_mau_sau_chia_phan": sum(1 for m in pool if m["so_phan"] > 1),
                 "so_mau_bi_loai": len(loai),
                 "ly_do_loai": {k: sum(1 for x in loai if x["ly_do"] == k) for k in sorted({x["ly_do"] for x in loai})},
                 "chi_tiet_loai": loai},
        "test_chi_dem": {"so_mau": len(test), "so_mau_qua_dai": sum(1 for m in test if m["qua_dai"])},
        "kiem_token_toan_bo_truoc_chia": {"title": nguyen["title"], "mo_ta": nguyen["mo_ta"],
                                           "tong": nguyen["title"] + nguyen["mo_ta"],
                                           "ghi_chu": "pool + test, trước khi chia phần/loại; chỉ là tổng số"},
    }


def lam_md(ket_qua_theo_nhanh, so_mau=5, seed=SEED_MAU):
    """5 mẫu POOL ngẫu nhiên (seed cố định) mỗi nhánh x hướng x loại, in đầy đủ messages để đọc tay."""
    dong = ["# Mẫu pool để đọc tay (B3 dựng mẫu)", "",
            f"Chọn ngẫu nhiên seed={seed}, {so_mau} mẫu mỗi nhánh x hướng x loại; chỉ lấy từ pool, không có mẫu test.", ""]
    for nhanh, kq in ket_qua_theo_nhanh.items():
        for huong in ("zh->vi", "vi->zh"):
            for lo in LOAI:
                ds = sorted((m for m in kq["pool"] if m["huong"] == huong and m["loai"] == lo), key=lambda m: m["id"])
                chon = random.Random(f"{seed}|{nhanh}|{huong}|{lo}").sample(ds, min(so_mau, len(ds)))
                dong += [f"## nhánh {nhanh} | {huong} | {lo} ({len(ds)} mẫu trong pool)", ""]
                for m in chon:
                    dong += [f"### {m['id']}  (source={m['source']}, fold5={m['fold5']}, phan={m['phan']}/{m['so_phan']}, "
                             f"so_token={m['so_token']})", ""]
                    for ms in m["messages"]:
                        dong += [f"**{ms['role']}**", "", "~~~text", ms["content"], "~~~", ""]
    return "\n".join(dong) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nhanh", default="ca_hai", choices=["cum", "pair", "ca_hai"])
    p.add_argument("--max-len", type=int, default=MAX_LEN_MAC_DINH)
    p.add_argument("--tokenizer", required=True, help="đường dẫn tokenizer.json của Qwen3")
    p.add_argument("--ra-mau", default=str(chung.FT_RUN / "mau"))
    p.add_argument("--ra-test", default=str(chung.THU_MUC_TEST))
    a = p.parse_args(argv)
    ra_mau, ra_test = Path(a.ra_mau), Path(a.ra_test)
    dem_nhieu, tok = lam_dem_token(a.tokenizer)
    split = chung.doc_split()                       # khóa test: chỉ đọc phép chia qua đây
    ds_nhanh = list(NHANH) if a.nhanh == "ca_hai" else [a.nhanh]

    ket, bo = {}, {}
    for nh in ds_nhanh:
        bo[nh] = chung.doc_bo(nh)
        ket[nh] = dung_mau_nhanh(bo[nh], split, nh, a.max_len, dem_nhieu)
        kiem_khoa(ket[nh], split, nh)
    # chỉ ghi khi mọi nhánh đã qua kiểm khóa
    bao_cao = {}
    for nh in ds_nhanh:
        ghi_jsonl(ra_mau / f"pool_{nh}.jsonl", ket[nh]["pool"])
        ghi_jsonl(ra_test / f"test_{nh}.jsonl", ket[nh]["test"])
        bao_cao[nh] = lam_bao_cao(ket[nh], nh, a.max_len)
    _ghi_text(ra_mau / "bao_cao_dung_mau.json", json.dumps(bao_cao, ensure_ascii=False, indent=1))
    _ghi_text(ra_mau / "bao_cao_dung_mau.md", lam_md(ket))

    # so token toàn bộ 25.575 sp với số đo cũ (~17,0M cụm, ~16,3M pair)
    tham_chieu = {"cum": 17.0e6, "pair": 16.3e6}
    for nh in ds_nhanh:
        d = bo[nh]
        cu = 0
        for c in ("nguon_title", "dich_title", "nguon_mo_ta", "dich_mo_ta"):
            cu += sum(len(e.ids) for e in tok.encode_batch(d[c].tolist(), add_special_tokens=False))
        cu += 40 * 2 * len(d)            # cách đo cũ: token nội dung + ~40 khung mỗi mẫu
        moi = bao_cao[nh]["kiem_token_toan_bo_truoc_chia"]["tong"]
        print(f"[{nh}] pool={bao_cao[nh]['pool']['so_mau']} mẫu/{bao_cao[nh]['pool']['so_token']} token | "
              f"test={bao_cao[nh]['test_chi_dem']['so_mau']} mẫu | loại={bao_cao[nh]['pool']['so_mau_bi_loai']} | "
              f"chia phần={bao_cao[nh]['pool']['so_sp_bi_chia_phan']} sp")
        print(f"[{nh}] tổng token toàn bộ (khuôn+chỉ dẫn, trước chia): {moi:,}; "
              f"cách đo cũ (nội dung + 40/mẫu): {cu:,}; tham chiếu {tham_chieu[nh]:,.0f}; "
              f"lệch so tham chiếu {moi / tham_chieu[nh] - 1:+.1%} (mới), {cu / tham_chieu[nh] - 1:+.1%} (đo lại theo cách cũ)")
        bao_cao[nh]["kiem_token_toan_bo_truoc_chia"].update(
            {"do_lai_theo_cach_cu": cu, "tham_chieu": tham_chieu[nh]})
    _ghi_text(ra_mau / "bao_cao_dung_mau.json", json.dumps(bao_cao, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
