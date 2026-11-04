# -*- coding: utf-8 -*-
"""Gộp Tiki vào 1688: bộ song ngữ chính thức 9 cột (kế hoạch docs/gop_tiki_1688/KE_HOACH_GOP_TIKI_VAO_1688.md).

Đọc 4 file full 9 cột (1688 và Tiki, mỗi nguồn 2 bản: cụm và pair) cùng 2 bảng sản phẩm description và 2 bảng `cado` (lấy danh mục lá) trực tiếp từ
HF dataset ntquang0410/zh-vie_ecom ở một revision cố định (hoặc từ thư mục tải sẵn), áp các quy tắc loại bỏ rồi trần theo nhóm, ghi:
    bilingual_official_context.parquet, bilingual_official_pair.parquet   cùng một danh sách sản phẩm, 9 cột
    excluded_products.parquet                                             sản phẩm bị loại kèm ly_do_loai
    selection_report.json, manifest.json, SHA256SUMS.txt, README.md

Quy tắc (theo thứ tự; mỗi sản phẩm bị loại ghi đúng một ly_do_loai, quy tắc đầu tiên nó vi phạm):
    R0 khong_co_description   description_zh trống
    R1 sai_nhom               danh mục lá thuộc bac_A (file cấu hình)
    R2 cap_yeu_cao            so_cap_thap / so_cap >= nguong_cap_yeu (bảng sản phẩm bộ cụm, cùng ngưỡng cho cả hai nguồn)
    R3 mo_ta_trung            cùng (source, category, description_zh) của bản cụm: giữ tối đa toi_da_mo_ta_trung, chọn theo mã băm crc32(product_id)
    R4 quy_tac_title          các quy tắc title trong file cấu hình (Huy duyệt, thêm quy tắc ở đó)
    R5 vuot_tran_nhom         nhóm vượt tran_moi_nhom: bỏ bậc B trước, rồi tỷ lệ cặp yếu cao, rồi mã băm
Quy tắc, danh sách lá, ngưỡng và trần nằm trong gop_tiki_vao_1688.quy_tac.json; sửa ở đó, không sửa trong code.

    <python> scripts/release/gop_tiki_vao_1688.py --che-do thu                       # chạy thử: in số sau từng quy tắc, ghi selection_report + file loại mẫu
    <python> scripts/release/gop_tiki_vao_1688.py --che-do chinh_thuc --ra DIR       # ghi đủ file, chạy bộ nghiệm thu
Tuỳ chọn: --revision <commit HF> (mặc định HEAD, được ghi vào báo cáo), --goc-cuc-bo <thư mục có data/...> để dùng bản tải sẵn.
Chạy bằng Python của ecom_crawler/.venv (có huggingface_hub); máy ít RAM thì đặt OPENBLAS_NUM_THREADS=1.
"""
import argparse
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "ntquang0410/zh-vie_ecom"
_AQ = Path(__file__).resolve()
ROOT = _AQ.parents[2] if len(_AQ.parents) > 2 else _AQ.parent          # chạy được ở bất kỳ thư mục nào (Huy đặt 4 file chung một chỗ)
CAU_HINH = Path(__file__).with_name("gop_tiki_vao_1688.quy_tac.json")
COT = ["product_id", "source", "category", "title_zh", "title_vi", "description_zh", "description_vi", "title_nguon_dich", "desc_nguon_dich"]
FULL = {
    ("1688", "cum"): "data/1688/full_dataset/1688_full_attributes_context_zh_vi.parquet",
    ("1688", "pair"): "data/1688/full_dataset/1688_full_attributes_independent_zh_vi.parquet",
    ("tiki", "cum"): "data/tiki/full_dataset/tiki_full_attributes_context_vi_zh.parquet",
    ("tiki", "pair"): "data/tiki/full_dataset/tiki_full_attributes_independent_vi_zh.parquet",
}
SAN_PHAM = {"1688": "data/1688/descriptions/zh_to_vi_context/description_products_zh_vi.parquet",
            "tiki": "data/tiki/descriptions/vi_to_zh_context/description_products_vi_zh.parquet"}
CADO = {"1688": ("data/1688/cleaned/bilingual_zh_vi_cado.parquet", "category_1688"), "tiki": ("data/tiki/cleaned/tiki_vi_cado.parquet", "category_tiki")}
NHOM = {"1688": ["auto", "bags", "beauty", "electronics", "fashion", "food", "home", "mother_baby", "shoes"],
        "tiki": ["auto", "beauty", "electronics", "food", "home", "mother_baby"]}
TITLE_NGUON = {"gemma4", "qwen3_14b_awq"}
DESC_NGUON = {"gemma4", "qwen3_14b_awq", "gemma4+qwen3_14b_awq", "tu_dien_quy_tac"}
LY_DO_CO_DINH = ["khong_co_description", "sai_nhom", "cap_yeu_cao", "mo_ta_trung", "vuot_tran_nhom"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ma_bam(pid):
    return zlib.crc32(str(pid).encode("utf-8"))


def doc_cau_hinh(p=CAU_HINH):
    cfg = json.loads(Path(p).read_text(encoding="utf-8"))
    tp = cfg["tham_so"]
    assert tp["tran_moi_nhom"] > 0 and 0 < tp["nguong_cap_yeu"] <= 1 and tp["toi_da_mo_ta_trung"] >= 1, tp
    assert set(cfg["so_dong_dau_vao"]) == {"1688", "tiki"} and all(isinstance(v, int) and v > 0 for v in cfg["so_dong_dau_vao"].values()), cfg.get("so_dong_dau_vao")
    for r in cfg["quy_tac_title"]:
        assert {"ly_do", "nguon", "cot", "regex"} <= set(r), r
        assert r["nguon"] in NHOM and r["cot"] in ("title_zh", "title_vi", COT_PHU), r
        re.compile(r["regex"])
    ly_do = [r["ly_do"] for r in cfg["quy_tac_title"]]
    assert len(set(ly_do)) == len(ly_do) and not set(ly_do) & set(LY_DO_CO_DINH), ly_do
    return cfg


def danh_sach_ly_do(cfg):
    return LY_DO_CO_DINH[:4] + [r["ly_do"] for r in cfg["quy_tac_title"]] + LY_DO_CO_DINH[4:]


def ap_dung_quy_tac(bang, cfg):
    """bang: mỗi dòng một sản phẩm, cột product_id, source, category, la, title_zh, title_vi, description_zh, so_cap, so_cap_thap.
    Trả về (bảng có thêm ly_do_loai, bac_b, ty_le_thap, ma_bam; danh sách số liệu từng bước)."""
    d = bang.copy().reset_index(drop=True)
    tp = cfg["tham_so"]
    d["ly_do_loai"] = None
    d["ma_bam"] = d.product_id.map(ma_bam)
    d["ty_le_thap"] = d.so_cap_thap / d.so_cap
    d["bac_b"] = False
    for e in cfg["bac_B"]:
        d.loc[(d.source == e["nguon"]) & (d.category == e["nhom"]) & d.la.isin(e["la"]), "bac_b"] = True
    buoc = [{"buoc": "dau_vao", "ly_do": None, "loai": 0, "con_lai": len(d)}]

    def danh_dau(mask, ly_do):
        m = mask & d.ly_do_loai.isna()
        d.loc[m, "ly_do_loai"] = ly_do
        buoc.append({"buoc": ly_do, "ly_do": ly_do, "loai": int(m.sum()), "con_lai": int(d.ly_do_loai.isna().sum())})

    danh_dau(d.description_zh.isna(), "khong_co_description")
    mask = pd.Series(False, index=d.index)
    for e in cfg["bac_A"]:
        mask |= (d.source == e["nguon"]) & ((d.category == e["nhom"]) if e.get("nhom") else True) & d.la.isin(e["la"])
    danh_dau(mask, "sai_nhom")
    danh_dau(d.ty_le_thap >= tp["nguong_cap_yeu"], "cap_yeu_cao")
    con = d[d.ly_do_loai.isna()].sort_values(["ma_bam", "product_id"], kind="stable")
    hang = con.groupby(["source", "category", "description_zh"], sort=False).cumcount()
    danh_dau(d.index.isin(con.index[(hang >= tp["toi_da_mo_ta_trung"]).values]), "mo_ta_trung")
    for r in cfg["quy_tac_title"]:
        danh_dau((d.source == r["nguon"]) & d[r["cot"]].fillna("").str.contains(r["regex"], regex=True), r["ly_do"])
    for nhom, con in d[d.ly_do_loai.isna()].groupby("category"):
        if len(con) > tp["tran_moi_nhom"]:
            giu = con.sort_values(["bac_b", "ty_le_thap", "ma_bam", "product_id"], kind="stable").index[: tp["tran_moi_nhom"]]
            d.loc[con.index.difference(giu), "ly_do_loai"] = "vuot_tran_nhom"
    da = int((d.ly_do_loai == "vuot_tran_nhom").sum())
    buoc.append({"buoc": "vuot_tran_nhom", "ly_do": "vuot_tran_nhom", "loai": da, "con_lai": int(d.ly_do_loai.isna().sum())})
    return d, buoc


# ------------------------------------------------------------------ nạp đầu vào
TEN_CU = {"title_ngon_dich": "title_nguon_dich"}          # lỗi chính tả trong file full của Huy (28/10)
COT_PHU = "title_van_de"                                  # cột nhãn nghi vấn của title Huy thêm ngày 28/10: nhận vào, không đưa ra


def kiem_ten_cot(rel, ten, chap_nhan_ten_cu=False):
    """Cổng G0: file full phải có đúng 9 cột theo đúng thứ tự chốt (được thêm cột phụ title_van_de ở cuối). Trả về tên cột đã chuẩn hoá."""
    ten = [TEN_CU.get(c, c) if chap_nhan_ten_cu else c for c in ten]
    if ten == COT + [COT_PHU]:
        return ten
    if ten != COT:
        sai = [(a, b) for a, b in zip(ten, COT) if a != b] if len(ten) == len(COT) else list(ten)
        sys.exit(f"DỪNG (cổng G0): {rel} có cột lệch bộ chốt: {sai}. Huy đổi tên cột rồi đẩy lại (hoặc chạy với --chap-nhan-ten-cu khi chạy thử).")
    return ten


def lay_file(rel, args, revision):
    if args.goc_cuc_bo and (Path(args.goc_cuc_bo) / rel).exists():
        return Path(args.goc_cuc_bo) / rel
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, rel, repo_type="dataset", revision=revision, local_dir=args.goc_cuc_bo or None))


def kiem_dong_dau_vao(nguon, ids_full, ids_san_pham, du_kien, chap_nhan=False):
    """Chốt chống mất sản phẩm âm thầm: file full (gốc của phép gộp) phải đúng số dòng dự kiến, và mọi sản phẩm của bảng description phải có trong file full.
    Nếu Huy xóa dòng title mà bên description vẫn giữ (hoặc ngược lại) thì dừng, không để sản phẩm biến mất không dấu vết."""
    loi = []
    if len(ids_full) != du_kien:
        loi.append(f"file full có {len(ids_full):,} dòng, cấu hình dự kiến {du_kien:,}")
    thua = set(ids_san_pham) - set(ids_full)
    if thua:
        loi.append(f"{len(thua):,} sản phẩm có trong bảng description nhưng KHÔNG có trong file full (title đã bị xóa?), ví dụ {sorted(thua)[:3]}")
    if loi and not chap_nhan:
        sys.exit(f"DỪNG (cổng G1): {nguon}: " + "; ".join(loi) + ". Muốn bỏ một sản phẩm thì thêm quy tắc title (để có ly_do_loai), đừng xóa dòng; "
                 "nếu thay đổi là cố ý thì cập nhật so_dong_dau_vao trong file quy tắc hoặc chạy với --chap-nhan-doi-so-dong.")
    if loi:
        print(f"CẢNH BÁO ({nguon}): " + "; ".join(loi))


def nap_dau_vao(args, cfg):
    revision = args.revision
    if not revision:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(REPO).sha
    duong_dan = {}
    for k, rel in FULL.items():
        duong_dan[rel] = lay_file(rel, args, revision)
        kiem_ten_cot(rel, pq.ParquetFile(duong_dan[rel]).schema_arrow.names, args.chap_nhan_ten_cu)
    cac = []
    for nguon in ("1688", "tiki"):
        ten_cot = pq.ParquetFile(duong_dan[FULL[(nguon, "cum")]]).schema_arrow.names
        cot_doc = ["product_id", "source", "category", "title_zh", "title_vi", "description_zh"] + ([COT_PHU] if COT_PHU in ten_cot else [])
        t = pq.read_table(duong_dan[FULL[(nguon, "cum")]], columns=cot_doc).to_pandas()
        if COT_PHU not in t:
            t[COT_PHU] = ""
        t[COT_PHU] = t[COT_PHU].fillna("")
        rel, cot_la = CADO[nguon]
        duong_dan[SAN_PHAM[nguon]], duong_dan[rel] = lay_file(SAN_PHAM[nguon], args, revision), lay_file(rel, args, revision)
        sp = pq.read_table(duong_dan[SAN_PHAM[nguon]], columns=["product_id", "so_cap", "so_cap_thap"]).to_pandas()
        cado = pq.read_table(duong_dan[rel], columns=["product_id", cot_la]).to_pandas().rename(columns={cot_la: "la"})
        for x in (t, sp, cado):
            x["product_id"] = x.product_id.astype(str)
        assert t.product_id.is_unique and sp.product_id.is_unique and cado.product_id.is_unique, nguon
        kiem_dong_dau_vao(nguon, t.product_id, sp.product_id, cfg["so_dong_dau_vao"][nguon], args.chap_nhan_doi_so_dong)
        assert (t.source == nguon).all() and set(t.category) <= set(NHOM[nguon]), f"{nguon}: source hoặc category lạ"
        m = t.merge(sp, on="product_id", how="left", validate="one_to_one").merge(cado, on="product_id", how="left", validate="one_to_one")
        assert m.la.notna().all(), f"{nguon}: {int(m.la.isna().sum())} sản phẩm không có danh mục lá"
        assert m.so_cap[m.description_zh.notna()].notna().all(), f"{nguon}: sản phẩm có description mà thiếu so_cap"
        cac.append(m)
    # hai bản (cụm, pair) phải cùng sản phẩm, cùng title, cùng category
    for nguon in ("1688", "tiki"):
        a = pq.read_table(duong_dan[FULL[(nguon, "pair")]], columns=["product_id", "category", "title_zh", "title_vi"]).to_pandas()
        b = cac[0 if nguon == "1688" else 1]
        assert (a.product_id.astype(str).values == b.product_id.values).all() and (a.category.values == b.category.values).all() \
            and (a.title_zh.values == b.title_zh.values).all() and (a.title_vi.values == b.title_vi.values).all(), f"{nguon}: bản pair khác bản cụm ở khoá, category hoặc title"
    bang = pd.concat(cac, ignore_index=True)
    return bang, duong_dan, revision


# ------------------------------------------------------------------ ghi đầu ra
def bang_arrow(df):
    return pa.Table.from_pandas(df[COT].astype("object").where(df[COT].notna(), None), schema=pa.schema([pa.field(c, pa.string()) for c in COT]), preserve_index=False)


def dung_dau_ra(d, duong_dan):
    """Trả về (bản cụm giữ, bản pair giữ, bảng loại), cùng thứ tự: 1688 trước, Tiki sau, mỗi nguồn theo thứ tự file cụm."""
    giu = d[d.ly_do_loai.isna()]
    kq = {}
    for bien in ("cum", "pair"):
        ph = []
        for nguon in ("1688", "tiki"):
            t = pq.read_table(duong_dan[FULL[(nguon, bien)]]).to_pandas().rename(columns=TEN_CU)
            t = t[COT]                                                  # bỏ cột phụ title_van_de; đúng 9 cột theo đúng thứ tự
            t["product_id"] = t.product_id.astype(str)
            ids = giu[giu.source == nguon].product_id.values
            ph.append(t.set_index(pd.Index(t.product_id.values, name="khoa")).loc[ids].reset_index(drop=True))
        kq[bien] = pd.concat(ph, ignore_index=True)
    loai = d[d.ly_do_loai.notna()][["product_id", "source", "category", "title_zh", "title_vi", "ly_do_loai"]].reset_index(drop=True)
    return kq["cum"], kq["pair"], loai


def nghiem_thu(d, cfg, cum, pair, loai):
    tp = cfg["tham_so"]
    ly_do_hop_le = set(danh_sach_ly_do(cfg))
    for ten, x in (("cum", cum), ("pair", pair)):
        assert list(x.columns) == COT, f"{ten}: cột sai {list(x.columns)}"
        assert x.notna().all().all(), f"{ten}: còn ô null ở cột {[c for c in COT if x[c].isna().any()]}"
        assert set(x.source) <= {"1688", "tiki"}, f"{ten}: source lạ {set(x.source)}"
        for nguon in ("1688", "tiki"):
            assert set(x[x.source == nguon].category) <= set(NHOM[nguon]), f"{ten}: category lạ ở {nguon}"
        assert set(x.title_nguon_dich) <= TITLE_NGUON, f"{ten}: title_nguon_dich lạ {set(x.title_nguon_dich) - TITLE_NGUON}"
        assert set(x.desc_nguon_dich) <= DESC_NGUON, f"{ten}: desc_nguon_dich lạ {set(x.desc_nguon_dich) - DESC_NGUON}"
        assert x.groupby("category").size().max() <= tp["tran_moi_nhom"], f"{ten}: có nhóm vượt trần"
        for r in cfg["quy_tac_title"]:
            sub = x[x.source == r["nguon"]]
            assert not sub[r["cot"]].str.contains(r["regex"], regex=True).any(), f"{ten}: còn sản phẩm vi phạm quy tắc title {r['ly_do']}"
    assert (cum.product_id.values == pair.product_id.values).all() and (cum.source.values == pair.source.values).all(), "hai bản khác tập hoặc thứ tự sản phẩm"
    assert (cum.title_zh.values == pair.title_zh.values).all() and (cum.category.values == pair.category.values).all(), "hai bản khác title hoặc category"
    assert cum.product_id.is_unique, "product_id trùng"
    assert set(loai.ly_do_loai) <= ly_do_hop_le, f"ly_do_loai lạ: {set(loai.ly_do_loai) - ly_do_hop_le}"
    khoa_giu = set(zip(cum.source, cum.product_id))
    khoa_loai = set(zip(loai.source, loai.product_id))
    khoa_vao = set(zip(d.source, d.product_id))
    assert not khoa_giu & khoa_loai and khoa_giu | khoa_loai == khoa_vao and len(khoa_giu) + len(khoa_loai) == len(d), "giữ + loại không khớp đầu vào"
    d2, _ = ap_dung_quy_tac(d.drop(columns=["ly_do_loai", "ma_bam", "ty_le_thap", "bac_b"]), cfg)
    assert d2.ly_do_loai.fillna("").tolist() == d.ly_do_loai.fillna("").tolist(), "chạy lại cho kết quả khác (không tất định)"


def dau_van_tay(d):
    """sha256 của danh sách giữ và danh sách loại (kèm lý do), không phụ thuộc phiên bản pandas/pyarrow: hai máy chạy cùng đầu vào, cùng quy tắc phải ra cùng dấu vân tay
    (file parquet thì có thể khác byte giữa các phiên bản thư viện, nên so dấu vân tay này chứ không so sha256 của parquet)."""
    giu = d[d.ly_do_loai.isna()]
    loai = d[d.ly_do_loai.notna()]
    k_giu = sorted(f"{s}|{p}" for s, p in zip(giu.source, giu.product_id))
    k_loai = sorted(f"{s}|{p}|{r}" for s, p, r in zip(loai.source, loai.product_id, loai.ly_do_loai))
    h = lambda ds: hashlib.sha256("\n".join(ds).encode("utf-8")).hexdigest()  # noqa: E731
    return {"giu_sha256": h(k_giu), "loai_sha256": h(k_loai), "so_giu": len(k_giu), "so_loai": len(k_loai)}


def thong_ke(d, buoc, cfg, revision, duong_dan):
    giu = d[d.ly_do_loai.isna()]
    return {
        "phien_ban_quy_tac": cfg["phien_ban"],
        "dau_vao": {"revision_hf": revision, "tong_san_pham": len(d), "theo_nguon": d.groupby("source").size().to_dict(),
                    "file": {rel: {"sha256": sha(p), "byte": Path(p).stat().st_size} for rel, p in sorted(duong_dan.items())}},
        "tham_so": cfg["tham_so"],
        "quy_tac_title": cfg["quy_tac_title"],
        "cac_buoc": buoc,
        "theo_ly_do": d.ly_do_loai.value_counts().to_dict(),
        "theo_ly_do_va_nguon": {f"{a}|{b}": int(v) for (a, b), v in d[d.ly_do_loai.notna()].groupby(["ly_do_loai", "source"]).size().items()},
        "giu": {"tong": len(giu), "theo_nguon": giu.groupby("source").size().to_dict(),
                "theo_nhom_va_nguon": {n: g.groupby("source").size().to_dict() for n, g in giu.groupby("category")}},
        "bac_B": {"truoc_tran": {n: int(c) for n, c in d[d.bac_b & (d.ly_do_loai.isna() | (d.ly_do_loai == "vuot_tran_nhom"))].groupby("category").size().items()},
                  "sau_tran": {n: int(c) for n, c in giu[giu.bac_b].groupby("category").size().items()}},
        "dau_van_tay": dau_van_tay(d),
        "danh_muc_la_bac_A": cfg["bac_A"], "danh_muc_la_bac_B": cfg["bac_B"],
    }


def viet_readme(tk, cfg):
    giu, vao = tk["giu"], tk["dau_vao"]
    dong_ly_do = "\n".join(f"| `{b['buoc']}` | {b['loai']:,} | {b['con_lai']:,} |" for b in tk["cac_buoc"][1:])
    dong_nhom = "\n".join(f"| {n} | {v.get('1688', 0):,} | {v.get('tiki', 0):,} | {sum(v.values()):,} |" for n, v in sorted(giu["theo_nhom_va_nguon"].items(), key=lambda kv: -sum(kv[1].values())))
    tiki_pc = {n: v.get("tiki", 0) / sum(v.values()) for n, v in giu["theo_nhom_va_nguon"].items() if v.get("tiki")}
    pc = ", ".join(f"{n} {tiki_pc[n]:.0%}" for n in sorted(tiki_pc, key=tiki_pc.get))
    b_sau = sum(tk["bac_B"]["sau_tran"].values())
    mo_ta_title = "; ".join(r.get("mo_ta_en") or "`%s`: rows of %s whose %s matches the pattern %s" % (r["ly_do"], r["nguon"], r["cot"], r["regex"]) for r in cfg["quy_tac_title"])
    return f"""# bilingual_official: Chinese-Vietnamese e-commerce products from 1688 and Tiki

Rule version `{tk['phien_ban_quy_tac']}` · Input: Hugging Face `{REPO}` at revision `{vao['revision_hf']}`

**{giu['tong']:,} products** ({giu['theo_nguon'].get('1688', 0):,} from 1688.com, {giu['theo_nguon'].get('tiki', 0):,} from Tiki.vn) with a title and an attribute-list description in Chinese and
Vietnamese, in 9 columns. The two files hold **the same products in the same order**; they differ only in how the description was translated.

It is **machine-translated, silver-quality data**: no one has verified the meaning (`semantic_verified` is `False` in the title sets), so it is suitable for training,
fine-tuning and analysis but **not a human reference**, and not a test set unless people review it first.

## Files

| File | Rows | Content |
|---|---:|---|
| `bilingual_official_context.parquet` | {giu['tong']:,} | description translated per product with its context (set `zh_to_vi_context` / `vi_to_zh_context`) |
| `bilingual_official_pair.parquet` | {giu['tong']:,} | description translated attribute by attribute (set `zh_to_vi` / `vi_to_zh`), same products |
| `excluded_products.parquet` | {vao['tong_san_pham'] - giu['tong']:,} | products left out, with the reason (`ly_do_loai`); restorable |
| `selection_report.json`, `manifest.json`, `SHA256SUMS.txt` | – | counts after every rule, columns and checksums, input files with sha256 |

**Do not train on both files for the same product.**

## Columns (same in both files)

| Column | Meaning |
|---|---|
| `product_id` | product id (string); unique |
| `source` | `1688` (Chinese to Vietnamese; the Chinese side is the original) or `tiki` (Vietnamese to Chinese; the Vietnamese side is the original) |
| `category` | product group: 1688 has 9 (auto, bags, beauty, electronics, fashion, food, home, mother_baby, shoes), Tiki has 6 (no bags, shoes, fashion) |
| `title_zh`, `title_vi` | product title in each language |
| `description_zh`, `description_vi` | attribute list as one text (`name: value`); separator `; ` in the context file, newline in the pair file |
| `title_nguon_dich` | model that produced the translated title: `gemma4` or `qwen3_14b_awq` (the better of the two by COMETKiwi, or the other when it repaired residual source-language text) |
| `desc_nguon_dich` | model(s) that translated at least one attribute pair: `gemma4`, `qwen3_14b_awq`, `gemma4+qwen3_14b_awq`; `tu_dien_quy_tac` = dictionary and rules only. Reviewed dictionaries and fixed rules contribute to every product |

## What was left out and why

Rules are applied in this order; a product is recorded with the first rule it breaks. The rule file is `scripts/release/gop_tiki_vao_1688.quy_tac.json`.

| Rule (`ly_do_loai`) | Removed | Remaining |
|---|---:|---:|
{dong_ly_do}

- `khong_co_description`: no attribute list.
- `sai_nhom`: leaf categories that do not belong to the group (pet food and supplies, adult care products, musical instruments, 1688 pet bowls, adult items, digital storage, medical and packaging).
- `cap_yeu_cao`: share of low-confidence attribute pairs {cfg['tham_so']['nguong_cap_yeu']} or more (computed the same way for both sources).
- `mo_ta_trung`: more than {cfg['tham_so']['toi_da_mo_ta_trung']} products with the same description in the same source and group; the kept ones are chosen by a fixed hash of `product_id`.
- title rules (applied before the group cap): {mo_ta_title}.
- `vuot_tran_nhom`: each group keeps at most {cfg['tham_so']['tran_moi_nhom']:,} products; off-topic products go first, then those with more low-confidence pairs, then a fixed hash.

## Products kept, by group and source

| Group | 1688 | Tiki | Total |
|---|---:|---:|---:|
{dong_nhom}

## Limitations

- Machine translation, not verified by people. Some 1688 titles use the lower-scored candidate when a residual Chinese string was repaired.
- The two sources translate in opposite directions; filter on `source`. Tiki makes up {pc} of the six shared groups; bags, shoes and fashion exist only for 1688, so compare groups by share rather than by count.
- Fashion is dominated by one leaf category (about 73 % of its products) and shoes by another (about 26 %).
- {b_sau:,} kept products are off-topic for their group (whole motorbikes and bicycles in auto, medical and health products in beauty, alcohol in food) but valid for the Tiki category tree; join the leaf category of `data/tiki/cleaned/tiki_vi_cado.parquet` to find them.
- Some Tiki titles keep Vietnamese brand or model names inside the Chinese title.

## Reading

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/{REPO}/data/bilingual_official/bilingual_official_context.parquet")
```
"""


def ghi_tat_ca(ra, d, buoc, cfg, revision, duong_dan, cum, pair, loai):
    ra.mkdir(parents=True, exist_ok=True)
    tk = thong_ke(d, buoc, cfg, revision, duong_dan)
    pq.write_table(bang_arrow(cum), ra / "bilingual_official_context.parquet", compression="zstd")
    pq.write_table(bang_arrow(pair), ra / "bilingual_official_pair.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(loai, preserve_index=False), ra / "excluded_products.parquet", compression="zstd")
    (ra / "selection_report.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1, default=int), encoding="utf-8")
    (ra / "README.md").write_text(viet_readme(tk, cfg), encoding="utf-8", newline="\n")
    ten = ["bilingual_official_context.parquet", "bilingual_official_pair.parquet", "excluded_products.parquet", "selection_report.json", "README.md"]
    man = {"cot": COT, "revision_hf_dau_vao": revision, "file": {n: {"byte": (ra / n).stat().st_size, "sha256": sha(ra / n),
                                                                   **({"dong": pq.ParquetFile(ra / n).metadata.num_rows} if n.endswith(".parquet") else {})} for n in ten}}
    (ra / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    ten.append("manifest.json")
    (ra / "SHA256SUMS.txt").write_text("".join(f"{sha(ra / n)}  {n}\n" for n in ten), encoding="ascii", newline="\n")
    return tk


def doc_lai_va_kiem(ra, d, cfg):
    cum = pq.read_table(ra / "bilingual_official_context.parquet").to_pandas()
    pair = pq.read_table(ra / "bilingual_official_pair.parquet").to_pandas()
    loai = pq.read_table(ra / "excluded_products.parquet").to_pandas()
    assert pq.read_schema(ra / "bilingual_official_context.parquet").names == COT and pq.read_schema(ra / "bilingual_official_pair.parquet").names == COT
    nghiem_thu(d, cfg, cum, pair, loai)
    ghi = {ln.split()[1]: ln.split()[0] for ln in (ra / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()}
    assert all(sha(ra / n) == h for n, h in ghi.items()), "SHA256SUMS không khớp file"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--che-do", choices=["thu", "chinh_thuc"], default="thu")
    ap.add_argument("--ra", type=Path, default=ROOT / "data" / "release" / "bilingual_official")
    ap.add_argument("--revision")
    ap.add_argument("--goc-cuc-bo")
    ap.add_argument("--chap-nhan-ten-cu", action="store_true", help="chạy thử khi file full của Huy còn cột title_ngon_dich: đổi thành title_nguon_dich khi ghi")
    ap.add_argument("--chap-nhan-doi-so-dong", action="store_true", help="cho chạy dù số dòng file full khác so_dong_dau_vao hoặc bảng description có sản phẩm thừa (chỉ khi thay đổi là cố ý)")
    ap.add_argument("--cau-hinh", type=Path, default=CAU_HINH)
    a = ap.parse_args()
    cfg = doc_cau_hinh(a.cau_hinh)
    bang, duong_dan, revision = nap_dau_vao(a, cfg)
    print(f"Đầu vào: revision {revision[:10]}, {len(bang):,} sản phẩm {bang.groupby('source').size().to_dict()}")
    d, buoc = ap_dung_quy_tac(bang, cfg)
    for b in buoc:
        print(f"  {b['buoc']:<22} loại {b['loai']:>6,} | còn {b['con_lai']:>7,}")
    giu = d[d.ly_do_loai.isna()]
    print("Giữ:", len(giu), giu.groupby("source").size().to_dict())
    print(pd.crosstab(giu.category, giu.source, margins=True).to_string())
    if a.che_do == "thu":
        a.ra.mkdir(parents=True, exist_ok=True)
        tk = thong_ke(d, buoc, cfg, revision, duong_dan)
        (a.ra / "selection_report.json").write_text(json.dumps(tk, ensure_ascii=False, indent=1, default=int), encoding="utf-8")
        loai = d[d.ly_do_loai.notna()][["product_id", "source", "category", "la", "title_zh", "title_vi", "ly_do_loai"]]
        mau = pd.concat([g.head(30) for _, g in loai.groupby("ly_do_loai")])
        mau.to_parquet(a.ra / "excluded_mau.parquet", index=False)
        print(f"Chạy thử xong (không ghi bộ chính thức). Đã ghi {a.ra / 'selection_report.json'} và {a.ra / 'excluded_mau.parquet'} ({len(mau)} dòng mẫu).")
        return
    cum, pair, loai = dung_dau_ra(d, duong_dan)
    nghiem_thu(d, cfg, cum, pair, loai)
    tk = ghi_tat_ca(a.ra, d, buoc, cfg, revision, duong_dan, cum, pair, loai)
    doc_lai_va_kiem(a.ra, d, cfg)
    dv = tk["dau_van_tay"]
    print(f"Đã ghi và nghiệm thu: {a.ra}")
    print(f"Dấu vân tay (so với máy kia, phải giống hệt): giữ {dv['so_giu']:,} sản phẩm {dv['giu_sha256']} | loại {dv['so_loai']:,} sản phẩm {dv['loai_sha256']}")


if __name__ == "__main__":
    main()
