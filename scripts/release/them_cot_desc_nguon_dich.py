# -*- coding: utf-8 -*-
"""Thêm cột `desc_nguon_dich` vào bảng sản phẩm của 4 bộ description (bộ 1 và 4 của 1688, bộ 2 và 3 của Tiki) và tạo bảng sản phẩm cho hai bộ dịch
theo pair (bộ 1, bộ 2) để dựng được bộ full gồm 9 cột (Prompt 523/524).

Ý nghĩa `desc_nguon_dich` (theo sản phẩm, xét các cặp thuộc tính của sản phẩm):
    gemma4                  có >= 1 cặp do Gemma (mô hình A) dịch hoặc sửa, không có cặp Qwen
    qwen3_14b_awq           có >= 1 cặp do Qwen (mô hình B), không có cặp Gemma
    gemma4+qwen3_14b_awq    có cả hai
    tu_dien_quy_tac         không có cặp nào do mô hình: toàn bộ từ điển/quy tắc
Từ điển và quy tắc luôn đóng góp ở mọi sản phẩm; cột chỉ nêu mô hình đã dịch ít nhất một cặp. Cặp `giong_het` (hai mô hình ra cùng bản dịch) tính cho Gemma
vì pipeline ưu tiên A.

Nguồn mức cặp:
    bộ pair (bộ 1, bộ 2): cột `cach_chon` của bảng build cục bộ (song_ngu_1 20261022_2126, song_ngu_2 20261023_1656), khớp 100% (pair_id, product_id) với
                          bảng cặp đã đẩy lên HF; HF không có cột này.
    bộ cụm (bộ 3, bộ 4):  cột `nguon_ban_dich` của bảng cặp cụm; riêng `giu_ban_dich_don_vi` kế thừa từ bộ 1/2 nên tra ngược `cach_chon` theo (pair_id, product_id).

Ghi ra data/release/desc_nguon_dich_20261028/ theo đúng đường dẫn sẽ đẩy lên HF; KHÔNG sửa data/release/description_*. Chạy:
    python -X utf8 scripts/release/them_cot_desc_nguon_dich.py
"""
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "data" / "release"
PROC = ROOT / "data" / "processed"
OUT = REL / "desc_nguon_dich_20261028"
HF = Path(r"C:\Users\ADMIN\hf_fd\data")          # bản HF đã tải để đối chiếu (chỉ đọc)

GEMMA = {"giong_het", "khac_chu_cung_nghia_A", "gan_giong_A", "chi_A", "uu_tien_duoi_san_A"}   # mô hình A (api_tuy_chon = Gemma 4 26B-A4B)
QWEN = {"chi_B"}                                                                              # mô hình B (Qwen3-14B-AWQ)
CUM_GEMMA = {"gemma_sua", "dich_moi"}
CUM_QWEN = {"qwen_sua"}
BAC = {"thap": 0, "vua": 1, "cao": 2}

NGUON = {
    "1688": dict(
        build=PROC / "song_ngu_1" / "bo_song_ngu_1_day_du_20261022_2126.parquet",
        cum_dir=REL / "description_zh_to_vi_cum", cum_cap="description_pairs_zh_vi_ctx.parquet", cum_sp="description_products_zh_vi.parquet",
        pair_dir=REL / "description_zh_to_vi", pair_cap="description_pairs_zh_vi.parquet",
        out_cum="data/1688/descriptions/zh_to_vi_context", out_pair="data/1688/descriptions/zh_to_vi", sp_pair="description_products_zh_vi_pair.parquet",
        ten_zh="name_zh", gt_zh="gia_tri_zh", ten_vi="name_vi", gt_vi="gia_tri_vi",
        hf_full_ind=HF / "1688/full_dataset/1688_full_attributes_independent_zh_vi.parquet"),
    "tiki": dict(
        build=PROC / "song_ngu_2" / "bo_song_ngu_2_day_du_20261023_1656.parquet",
        cum_dir=REL / "description_vi_to_zh_cum", cum_cap="description_pairs_vi_zh_ctx.parquet", cum_sp="description_products_vi_zh.parquet",
        pair_dir=REL / "description_vi_to_zh", pair_cap="description_pairs_vi_zh.parquet",
        out_cum="data/tiki/descriptions/vi_to_zh_context", out_pair="data/tiki/descriptions/vi_to_zh", sp_pair="description_products_vi_zh_pair.parquet",
        ten_zh="name_zh", gt_zh="gia_tri_zh", ten_vi="name_vi", gt_vi="gia_tri_vi",
        hf_full_ind=HF / "tiki/full_dataset/tiki_full_attributes_independent_vi_zh.parquet"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kiem_tu_vung(gia_tri):
    """Mọi giá trị cach_chon phải thuộc nhóm mô hình A/B hoặc nhóm quy tắc/từ điển; giá trị lạ thì dừng (không lặng lẽ tính là quy tắc)."""
    la = {v for v in set(gia_tri) if v not in GEMMA and v not in QWEN and not str(v).startswith(("quy_tac_", "tu_dien_"))}
    assert not la, f"cach_chon có giá trị chưa phân loại: {sorted(la)}"


def nhan(co_gemma, co_qwen):
    """Nhãn một sản phẩm từ hai cờ: có cặp Gemma / có cặp Qwen."""
    if co_gemma and co_qwen:
        return "gemma4+qwen3_14b_awq"
    if co_gemma:
        return "gemma4"
    if co_qwen:
        return "qwen3_14b_awq"
    return "tu_dien_quy_tac"


def nhan_theo_san_pham(cap):
    """cap: DataFrame có product_id, g, q (bool). Trả về Series product_id -> nhãn."""
    t = cap.groupby("product_id")[["g", "q"]].any()
    return pd.Series([nhan(g, q) for g, q in zip(t.g, t.q)], index=t.index)


def doc(p, cols=None):
    d = pq.read_table(p, columns=cols).to_pandas()
    d["product_id"] = d["product_id"].astype(str)
    return d


def co_mo_hinh_doc_lap(cap_hf, build):
    """Cờ mô hình cho bảng cặp độc lập (bộ 1, 2): tra cach_chon theo (pair_id, product_id); phải khớp 100%."""
    b = doc(build, ["pair_id", "product_id", "cach_chon", "do_tin_cay"])
    kiem_tu_vung(b.cach_chon)
    m = cap_hf[["pair_id", "product_id", "do_tin_cay"]].merge(b, on=["pair_id", "product_id"], how="left", suffixes=("", "_build"), validate="one_to_one")
    assert m.cach_chon.notna().all(), f"{int(m.cach_chon.isna().sum())} cặp HF không có trong bảng build"
    assert (m.do_tin_cay == m.do_tin_cay_build).all(), "do_tin_cay bảng build khác bảng đã đẩy"
    assert len(b) == len(cap_hf), (len(b), len(cap_hf))
    out = cap_hf[["pair_id", "product_id"]].copy()
    out["g"], out["q"] = m.cach_chon.isin(GEMMA).values, m.cach_chon.isin(QWEN).values
    return out


def co_mo_hinh_cum(cap_cum, build):
    """Cờ mô hình cho bảng cặp cụm (bộ 3, 4). Trả về (bảng cờ, số cặp kế thừa không tra được)."""
    b = doc(build, ["pair_id", "product_id", "cach_chon"])
    kiem_tu_vung(b.cach_chon)
    m = cap_cum[["pair_id", "product_id", "nguon_ban_dich"]].merge(b, on=["pair_id", "product_id"], how="left", validate="many_to_one")
    ke_thua = m.nguon_ban_dich == "giu_ban_dich_don_vi"
    khong_tra = int((ke_thua & m.cach_chon.isna()).sum())
    out = m[["pair_id", "product_id"]].copy()
    out["g"] = (m.nguon_ban_dich.isin(CUM_GEMMA) | (ke_thua & m.cach_chon.isin(GEMMA))).values
    out["q"] = (m.nguon_ban_dich.isin(CUM_QWEN) | (ke_thua & m.cach_chon.isin(QWEN))).values
    assert set(m.nguon_ban_dich) <= (CUM_GEMMA | CUM_QWEN | {"tu_dien_quy_tac", "giu_ban_dich_don_vi"}), set(m.nguon_ban_dich)
    return out, khong_tra


def san_pham_tu_cap(cap, c, sep):
    """Dựng bảng sản phẩm từ bảng cặp (sắp theo pair_id): so_cap, do_tin_cay (mức thấp nhất), so_cap_khong_cao, so_cap_thap, description_zh/vi."""
    cap = cap.sort_values(["product_id", "pair_id"], kind="stable")
    cap["dz"] = cap[c["ten_zh"]] + ": " + cap[c["gt_zh"]]       # gán tại chỗ, không nhân bản bảng (máy thiếu RAM)
    cap["dv"] = cap[c["ten_vi"]] + ": " + cap[c["gt_vi"]]
    cap["b"] = cap.do_tin_cay.map(BAC)
    cap = cap[["product_id", "category", "do_tin_cay", "dz", "dv", "b"]]
    g = cap.groupby("product_id", sort=False)
    sp = pd.DataFrame({
        "category": g.category.first(),
        "so_cap": g.size(),
        "do_tin_cay": g.b.min().map({v: k for k, v in BAC.items()}),
        "so_cap_khong_cao": g.do_tin_cay.apply(lambda s: int((s != "cao").sum())),
        "description_zh": g.dz.apply(sep.join),
        "description_vi": g.dv.apply(sep.join),
        "so_cap_thap": g.do_tin_cay.apply(lambda s: int((s == "thap").sum())),
    })
    return sp.reset_index()


def luu(bang, dich):
    dich.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(bang, dich, compression="zstd")


def lam(ten, c, thong_ke):
    print(f"\n===== {ten} =====")
    cot = ["product_id", "pair_id", "category", c["ten_zh"], c["gt_zh"], c["ten_vi"], c["gt_vi"], "do_tin_cay"]
    # ---- bộ cụm: cập nhật bảng sản phẩm ----
    cap_cum = doc(c["cum_dir"] / c["cum_cap"])
    sp_cum_t = pq.read_table(c["cum_dir"] / c["cum_sp"])
    sp_cum = sp_cum_t.to_pandas()
    sp_cum["product_id"] = sp_cum.product_id.astype(str)
    flags_cum, khong_tra = co_mo_hinh_cum(cap_cum, c["build"])
    lab_cum = nhan_theo_san_pham(flags_cum)
    assert set(sp_cum.product_id) == set(lab_cum.index), "sản phẩm của bảng cụm lệch bảng cặp cụm"
    # kiểm: dựng bảng sản phẩm từ bảng cặp cụm phải trùng bảng sản phẩm đã đẩy ở mọi cột cũ
    dung = san_pham_tu_cap(cap_cum[cot], c, "; ").set_index("product_id")
    cu = sp_cum.set_index("product_id")
    for k in ("category", "so_cap", "do_tin_cay", "so_cap_khong_cao", "so_cap_thap", "description_zh", "description_vi"):
        assert (dung.loc[cu.index, k] == cu[k]).all(), f"dựng lại từ bảng cặp cụm KHÁC cột {k} của bảng sản phẩm đã đẩy"
    print(f"bộ cụm: {len(cu)} sp; dựng lại từ bảng cặp trùng 100% 7 cột cũ; cặp kế thừa không tra được nguồn: {khong_tra}")
    moi = sp_cum_t.append_column("desc_nguon_dich", pa.array(sp_cum.product_id.map(lab_cum).values, type=pa.large_string()))
    assert moi.column_names[:-1] == sp_cum_t.column_names
    luu(moi, OUT / c["out_cum"] / c["cum_sp"])
    thong_ke[ten]["cum"] = {"san_pham": len(cu), "cap_ke_thua_khong_tra_duoc": khong_tra, "phan_bo": lab_cum.value_counts().to_dict()}
    print("   desc_nguon_dich:", lab_cum.value_counts().to_dict())
    del cap_cum, flags_cum, dung, cu, sp_cum, sp_cum_t, moi
    gc.collect()
    # ---- bộ pair: bảng sản phẩm mới ----
    cap_pair = doc(c["pair_dir"] / c["pair_cap"])
    flags_p = co_mo_hinh_doc_lap(cap_pair, c["build"])
    lab_p = nhan_theo_san_pham(flags_p)
    sp = san_pham_tu_cap(cap_pair[cot], c, "\n")
    sp["desc_nguon_dich"] = sp.product_id.map(lab_p)
    assert sp.desc_nguon_dich.notna().all() and sp.product_id.is_unique
    # kiểm: description dựng ở đây phải trùng bản đã công bố trong bộ full độc lập trên HF (nếu có bản tải về)
    if c["hf_full_ind"].exists():
        f = pq.read_table(c["hf_full_ind"], columns=["product_id", "description_zh", "description_vi"]).to_pandas()
        f["product_id"] = f.product_id.astype(str)
        f = f[f.description_zh.notna()].set_index("product_id")
        s = sp.set_index("product_id")
        assert set(f.index) == set(s.index), (len(f), len(s))
        assert (s.loc[f.index, "description_zh"] == f.description_zh).all() and (s.loc[f.index, "description_vi"] == f.description_vi).all(), \
            "description dựng từ bảng cặp KHÁC bộ full độc lập trên HF"
        print(f"bộ pair: {len(s)} sp; description_zh/vi trùng 100% bộ full độc lập trên HF")
    ref = pq.ParquetFile(c["cum_dir"] / c["cum_sp"]).schema_arrow            # cùng kiểu và thứ tự cột với bảng sản phẩm cụm
    sp = sp[[n for n in ref.names if n in sp.columns] + ["desc_nguon_dich"]]
    schema = pa.schema([pa.field(n, ref.field(n).type) for n in ref.names if n in sp.columns] + [pa.field("desc_nguon_dich", pa.large_string())])
    luu(pa.Table.from_pandas(sp, schema=schema, preserve_index=False), OUT / c["out_pair"] / c["sp_pair"])
    thong_ke[ten]["pair"] = {"san_pham": len(sp), "phan_bo": lab_p.value_counts().to_dict()}
    print("   desc_nguon_dich (pair):", lab_p.value_counts().to_dict())
    # so sánh nhãn pair và cụm cho cùng sản phẩm
    j = pd.concat([lab_p.rename("pair"), lab_cum.rename("cum")], axis=1).dropna()
    thong_ke[ten]["pair_bang_cum"] = round(float((j.pair == j.cum).mean()), 4)
    print(f"   cùng sản phẩm: nhãn pair == nhãn cụm: {(j.pair == j.cum).mean():.1%}")


def kiem_dau_ra():
    """Kiểm các file vừa ghi: cột, kiểu, số dòng, giá trị hợp lệ; ghi SHA256SUMS và thống kê."""
    thong_ke = {ten: json.loads((OUT / f"thong_ke_{ten}.json").read_text(encoding="utf-8")) for ten in NGUON}
    hop_le = {"gemma4", "qwen3_14b_awq", "gemma4+qwen3_14b_awq", "tu_dien_quy_tac"}
    sums = []
    for p in sorted(OUT.rglob("*.parquet")):
        t = pq.read_table(p)
        d = t.to_pandas()
        assert d.product_id.is_unique and d.desc_nguon_dich.notna().all() and set(d.desc_nguon_dich) <= hop_le, p
        assert t.schema.field("desc_nguon_dich").type == pa.large_string()
        rel = p.relative_to(OUT).as_posix()
        print(f"OK {rel}: {len(d)} dòng, cột {t.column_names}")
        sums.append(f"{sha(p)}  {rel}")
    (OUT / "sha256_parquet_moi.txt").write_text("\n".join(sums) + "\n", encoding="ascii")    # không đặt tên SHA256SUMS.txt: tên đó dành cho từng thư mục đích
    (OUT / "thong_ke.json").write_text(json.dumps(thong_ke, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    # Mỗi nguồn chạy trong một tiến trình riêng để trả lại RAM (máy chỉ trống ~1,5 GB): `<script>` = cả hai nguồn rồi kiểm; `<script> 1688|tiki|kiem` = một bước.
    buoc = sys.argv[1] if len(sys.argv) > 1 else None
    if buoc in NGUON:
        tk = {"cum": {}, "pair": {}}
        lam(buoc, NGUON[buoc], {buoc: tk})
        (OUT / f"thong_ke_{buoc}.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1), encoding="utf-8")
    elif buoc == "kiem":
        kiem_dau_ra()
    else:
        import subprocess
        for b in (*NGUON, "kiem"):
            r = subprocess.run([sys.executable, "-X", "utf8", __file__, b])
            assert r.returncode == 0, f"bước {b} lỗi"
