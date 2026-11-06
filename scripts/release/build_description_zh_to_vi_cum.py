# -*- coding: utf-8 -*-
"""Dựng thư mục công bố bộ song ngữ 4 (1688 Trung -> Việt, hiệu đính cả cụm theo sản phẩm, có ngữ cảnh) để đẩy lên HF:
data/1688/descriptions/zh_to_vi_context/ (bố cục repo sau khi ntquang0410 gom theo nguồn 25/10/2026; thư mục anh em: data/tiki/descriptions/vi_to_zh_context/).

Hai bảng cùng một bản dịch: bảng CẶP (cột bộ 1 + nguon_ban_dich) và bảng SẢN PHẨM (cột bộ 3, tiếng Trung trước, có so_cap_thap). Kiểm chặt (trùng, rỗng, nhãn,
tổng cặp, đủ mọi cặp bộ 1 và nguyên văn tiếng Trung, riêng tư, chữ Hán còn lại, cách dịch sai đã biết, tên lệch chuẩn), làm sạch thẻ nguồn gốc (bỏ tên khoá API,
câu gọi thử, văn bản lỗi, đường dẫn máy), ghi SHA256SUMS và README.md (thẻ dữ liệu tiếng Anh, sinh từ readme_zh_to_vi_cum.tpl.md bằng số liệu thật).

    .venv\\Scripts\\python.exe -X utf8 scripts/release/build_description_zh_to_vi_cum.py <nhãn bản, ví dụ 20261026_2123>
"""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_1" / "kaggle"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_4" / "kaggle"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_4 import hoan_tat as HT  # noqa: E402
from song_ngu_4 import ket_qua as KQ  # noqa: E402
import dich_cum_zv  # noqa: E402

S4 = ROOT / "data" / "processed" / "song_ngu_4"
SET1 = ROOT / "data" / "release" / "description_zh_to_vi" / "description_pairs_zh_vi.parquet"
DST = ROOT / "data" / "release" / "description_zh_to_vi_cum"
TPL = Path(__file__).with_name("readme_zh_to_vi_cum.tpl.md")
PAIRS, PRODUCTS, PROV = "description_pairs_zh_vi_ctx.parquet", "description_products_zh_vi.parquet", "provenance.json"
SHA_SET1 = "e8c5cfed878268e8f63cd8760f2dbd91f4c420f21107b76082d0378a42ffb9ba"      # bảng cặp bộ 1 đã công bố (20261022_2126)
NHAN_BO_1 = "20261022_2126"
NHAN = {"cao", "vua", "thap"}
NGUON = {"tu_dien_quy_tac", "giu_ban_dich_don_vi", "gemma_sua", "qwen_sua", "dich_moi"}
HAN = re.compile("[一-鿿㐀-䶿]")
VIET = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
CAM = ("GEMINI_API_KEY_", "/kaggle/", "trinhnhatanh", "KGAT_", "theo_khoa", "goi_thu", "AIza", "loi_cuoi", "ADMIN", "data\\processed", "D:\\")
# quét riêng tư theo TỪNG cột (nối cột làm chữ số hai cột liền nhau ghép thành số điện thoại giả). Mã hàng/mã báo cáo có thể trông như số điện thoại nên mẫu
# số điện thoại Việt chỉ áp cho thuộc tính không phải loại mã; WeChat cần nhãn kèm dấu hai chấm (mã hàng WXY-... không bị bắt oan)
TEN_MA = re.compile(r"货号|编号|型号|条形码|证号|报告|批准|编码|款号|SKU|sku|批号|产品标准")
RIENG_TU = {"điện thoại di động Trung Quốc": r"(?<!\d)1[3-9]\d{9}(?!\d)", "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
            "URL/tên miền": r"(?i)https?://|www\.|\b[\w-]+\.(?:com|vn|net|org|cn)\b", "zalo/facebook/hotline": r"(?i)\bzalo\b|facebook|\bhotline\b", "_x000D_": "_x000D_",
            "WeChat sau nhãn": r"(?i)(?:微信号?|威信|wechat|vx|wx)\s*[:：]\s*[A-Za-z][\w\-]{5,19}", "QQ kèm số": r"(?i)\bQQ\s*[:：]?\s*\d{5,11}", "加V/加微": r"加\s*[Vv微]|加我",
            "điện thoại kèm nhãn (中)": r"(?:电话|手机|联系方式|热线|TEL|tel)\s*[:：]?\s*\+?\d[\d\s\-]{6,}"}
RIENG_TU_MA = {"điện thoại Việt (không áp cho thuộc tính mã)": r"(?<![A-Za-z\d])(?:\+?84|0)(?:[\s.\-]?\d){9,10}(?!\d)"}
# cách dịch sai ĐÃ BIẾT không được còn trong bản cuối (cùng bảng với hoan_tat.SUA_LOI_GIA_TRI)
SAI_BIET = [("独立站", "Trạm độc lập"), ("镜头", "thấu kính"), ("无印风", "Phong cách không nhãn hiệu"), ("低帮", "Thân thấp"), ("小白鞋", "Giày trắng nhỏ"),
            ("丝瓜络", "Sơ dừa"), ("纯色", "Màu thuần"), ("单排扣", "Đơn hàng"), ("防水台", "chống thấm"), ("揭阳", "Khiết Dương")]
MA_SO = {"tu_dien_quy_tac": "fixed entry: reviewed dictionary, yes/no table, copies of values without Chinese characters, brands, licence codes, dates, measurement ranges, and the values "
                            "corrected by the consistency step",
         "giu_ban_dich_don_vi": "the draft of `zh_to_vi` was kept", "gemma_sua": "Gemma 4 changed the draft",
         "dich_moi": "no draft existed and Gemma translated the value", "qwen_sua": "Qwen3-14B changed the draft (used only when the Gemma change failed the rules)"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lam_sach(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "theo_khoa":
                out["so_khoa"] = len(v)
            elif k in ("goi_thu", "dau_ra_cu", "loi_cuoi", "thu_muc_gemma", "thu_muc_qwen"):
                continue
            else:
                out[k] = lam_sach(v)
        return out
    if isinstance(x, list):
        return [lam_sach(v) for v in x]
    return x


def f(n):
    return f"{int(n):,}"


def p(x):
    return f"{x * 100:.1f} %"


def pt(s):
    return f"{f(s.median())} / {f(s.quantile(.99))} / {f(s.max())}"


def md(rows):
    return "\n".join("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows)


def main():
    nhan = sys.argv[1]
    f_cap, f_sp, f_tk = S4 / f"cap_cum_{nhan}.parquet", S4 / f"san_pham_cum_{nhan}.parquet", S4 / f"tom_tat_cum_{nhan}.json"
    for x in (f_cap, f_sp, f_tk):
        assert x.exists(), x
    assert "_thu" not in nhan, "đây là bản chạy thử"
    assert sha(SET1) == SHA_SET1, "bảng cặp bộ 1 trên máy không phải bản đã công bố"
    C, S = pd.read_parquet(f_cap), pd.read_parquet(f_sp)
    tk = json.loads(f_tk.read_text(encoding="utf-8"))
    cb = json.loads((S4 / "the_chuan_bi.json").read_text(encoding="utf-8"))
    assert tk["cach_chon"] == "AB" and tk["nhan"] == nhan
    B1 = pd.read_parquet(SET1)
    PAIRS_COLS = HT.COT_CAP_CONG_BO
    SP_COLS = HT.COT_SP_CONG_BO
    # ---- kiểm chặt (trên bảng đầy đủ cột)
    assert C.pair_id.is_unique and S.product_id.is_unique
    D = C[PAIRS_COLS].copy()
    E = S[SP_COLS].copy()
    assert not D.isna().any().any() and not E.isna().any().any()
    assert not D[["name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh"]].apply(lambda s: s.str.strip().eq("")).any().any()
    assert set(D.do_tin_cay) <= NHAN and set(E.do_tin_cay) <= NHAN and set(D.nguon_ban_dich) <= NGUON
    assert int(E.so_cap.sum()) == len(D) and set(D.product_id) == set(E.product_id)
    rk = {"cao": 3, "vua": 2, "thap": 1}
    g = D.groupby("product_id").do_tin_cay.apply(lambda s: {3: "cao", 2: "vua", 1: "thap"}[min(rk[x] for x in s)])
    assert (E.set_index("product_id").do_tin_cay == g.reindex(E.product_id).values).all(), "nhãn sản phẩm khác mức thấp nhất của cặp"
    assert (E.set_index("product_id").so_cap_khong_cao == D[D.do_tin_cay != "cao"].groupby("product_id").size().reindex(E.product_id, fill_value=0).values).all()
    assert (E.set_index("product_id").so_cap_thap == D[D.do_tin_cay == "thap"].groupby("product_id").size().reindex(E.product_id, fill_value=0).values).all()
    assert ((E.so_cap_thap > 0) == (E.do_tin_cay == "thap")).all() and (E.so_cap_thap <= E.so_cap_khong_cao).all()
    # đủ mọi cặp bộ 1 với đúng id, sản phẩm, danh mục, nhóm trùng, tên và giá trị tiếng Trung
    m = B1.merge(D, on="pair_id", how="left", suffixes=("_1", ""))
    assert m.name_zh.notna().all(), "thiếu cặp của bộ 1"
    for c in ("product_id", "category", "nhom_trung", "name_zh", "gia_tri_zh"):
        assert (m[f"{c}_1"] == m[c]).all(), f"cột {c} khác bộ 1"
    n_diff_v, n_diff_n = int((m.gia_tri_vi_1 != m.gia_tri_vi).sum()), int((m.name_vi_1 != m.name_vi).sum())
    n_new = len(D) - len(B1)
    assert (~D.pair_id.isin(B1.pair_id)).sum() == n_new
    # chữ Hán còn trong bản Việt chỉ là số giấy phép theo quy tắc mã thật
    han_v = D.gia_tri_vi.map(lambda s: bool(HAN.search(s)))
    assert set(C.loc[han_v, "nguon_gt_vao"]) <= {"quy_tac_ma_that"}, set(C.loc[han_v, "nguon_gt_vao"])
    assert not D.name_vi.map(lambda s: bool(HAN.search(s))).any(), "tên Việt còn chữ Hán"
    assert not D.name_zh.str.contains(f"[{VIET}]", case=False, regex=True).any(), "name_zh còn chữ Việt"
    # cách dịch sai đã biết không còn; tên không lệch chuẩn
    for zh, sai in SAI_BIET:
        con = D[D.gia_tri_zh.str.contains(zh, regex=False) & D.gia_tri_vi.str.contains(sai, case=False, regex=False)]
        assert con.empty, f"còn cách dịch sai '{sai}' với {zh}: {len(con)}"
    for zh, sai in (("厚底", "Đế độn mũi"), ("套头", "Xỏ chân"), ("日式", "Phong cách tối giản Nhật"), ("中筒", "Thân trung"), ("短筒", "Thân ngắn")):
        assert not ((D.gia_tri_zh == zh) & (D.gia_tri_vi == sai)).any(), f"còn {zh} = {sai}"
    dem = D.groupby(["name_zh", "name_vi"]).size().reset_index(name="n")
    tong = dem.groupby("name_zh").n.sum()
    top = dem.sort_values("n", ascending=False, kind="stable").drop_duplicates("name_zh").set_index("name_zh")
    chuan = {z: top.at[z, "name_vi"] for z in top.index if tong[z] >= HT.TEN_TOI_THIEU and top.at[z, "n"] / tong[z] >= HT.TEN_DONG_THUAN}
    assert not (D.name_zh.map(chuan).notna() & (D.name_vi != D.name_zh.map(chuan))).any(), "còn tên lệch chuẩn"
    # luật chặt chạy lại trên mọi giá trị do mô hình hoặc bước nhất quán viết
    KH = KQ.doc_khoang()
    sel = C[C.nguon_ban_dich.isin(["gemma_sua", "qwen_sua", "dich_moi"]) | (C.nguon_gt_vao == HT.NGUON_SUA_LOI)]
    co_co = 0
    for r in sel.itertuples():
        nh = r.v_nhap if isinstance(r.v_nhap, str) else None
        if KQ.luat_truong(r.gia_tri_zh, r.v_gui, r.gia_tri_vi, nh, KH):
            co_co += 1
    assert co_co <= 2, f"luật chặt: {co_co} giá trị còn bị luật bắt (mong đợi tối đa 2 danh sách dài giữ bộ 1)"
    # riêng tư: từng cột riêng
    for ten, pat in RIENG_TU.items():
        for col, s in (("gia_tri_vi", D.gia_tri_vi), ("gia_tri_zh", D.gia_tri_zh), ("name_vi", D.name_vi), ("name_zh", D.name_zh),
                       ("description_vi", E.description_vi), ("description_zh", E.description_zh)):
            n = int(s.str.contains(pat, regex=True).sum())
            assert n == 0, f"quét riêng tư '{ten}': {n} dòng cột {col}"
    # không áp cho thuộc tính mã (货号, 编号...) và cho giá trị chỉ gồm danh sách mã số từ 2 mã trở lên (ví dụ mã hoa văn "02601141009, 02601141001, ...")
    ds_ma = D.gia_tri_zh.map(lambda s: bool(re.fullmatch(r"[\d\s,;.\-/]+", s)) and len(re.findall(r"\d{6,}", s)) >= 2)
    khong_ma = ~D.name_zh.str.contains(TEN_MA) & ~ds_ma
    for ten, pat in RIENG_TU_MA.items():
        for col in ("gia_tri_vi", "gia_tri_zh"):
            n = int(D.loc[khong_ma, col].str.contains(pat, regex=True).sum())
            assert n == 0, f"quét riêng tư '{ten}': {n} dòng cột {col}"
    # ---- số liệu cho README
    DST.mkdir(parents=True, exist_ok=True)
    D.to_parquet(DST / PAIRS, index=False)
    E.to_parquet(DST / PRODUCTS, index=False)
    cards = {"nen": S4 / "tu_kaggle_that_gemma_p2", **{f"phan_{i}": S4 / f"tu_kaggle_that_gemma_phan{i}" for i in range(4)}}
    gem = {k: json.loads((d / "ra_cum_zv" / "the_api_tuy_chon.json").read_text(encoding="utf-8")) for k, d in cards.items()}
    qw = json.loads((S4 / "that_qwen_du_phong" / "ra_cum_zv" / "the_qwen3_14b.json").read_text(encoding="utf-8"))
    cbs = {k: v for k, v in cb.items() if k != "nguon"}
    the = {"_note": "Provenance card of the build that produced the two tables of this folder. Keys are Vietnamese; see README.md, section 'Provenance file'. "
                    "API key names, test calls, error texts and local paths were removed before publishing.",
           "ban_dung": {"nhan": nhan, "bang_cap": PAIRS, "bang_san_pham": PRODUCTS},
           "nguon": {"ban_dich_don_vi_dung_lam_nhap": {"thu_muc": "data/1688/descriptions/zh_to_vi/", "nhan_ban": NHAN_BO_1, "sha256_file_cap": SHA_SET1},
                     "san_pham": {"file": "data/1688/snapshot/bilingual_zh_vi.parquet", "sha256": cb["nguon"]["san_pham"]["sha256"]}},
           "chuan_bi": cbs, "cau_lenh_he_thong": dich_cum_zv.HE_THONG, "phien_ban_cau_lenh": dich_cum_zv.PHIEN_BAN_CAU_LENH,
           "chay": {"gemma_4": {k: lam_sach(v) for k, v in gem.items()}, "qwen3_14b_du_phong": lam_sach(qw)},
           "ghep": lam_sach(tk)}
    txt = json.dumps(the, ensure_ascii=False, indent=1)
    for cam in CAM:
        assert cam not in txt, f"thẻ nguồn gốc còn chuỗi cấm: {cam}"
    (DST / PROV).write_text(txt + "\n", encoding="utf-8")
    tong_sha = "".join(f"{sha(DST / x)}  {x}\n" for x in (PAIRS, PRODUCTS, PROV))
    (DST / "SHA256SUMS.txt").write_text(tong_sha, encoding="ascii", newline="\n")
    sys.stdout.write(tong_sha)

    nk, nh = tk["nhat_quan"]["gia_tri"], tk["nhat_quan"]
    lab = D.do_tin_cay.value_counts()
    prod = E.do_tin_cay.value_counts()
    gsum = lambda k: sum(c["api"][k] for c in gem.values())  # noqa: E731
    cap_g = D[D.category.notna()]
    cat_rows = [["Category", "Products", "Pairs", "`cao`", "`vua`", "`thap`"]]
    cat_rows.append(["---"] * 6)
    for cat, d in sorted(D.groupby("category"), key=lambda x: -len(x[1])):
        cat_rows.append([cat, f(d.product_id.nunique()), f(len(d)), f((d.do_tin_cay == "cao").sum()), f((d.do_tin_cay == "vua").sum()), f((d.do_tin_cay == "thap").sum())])
    cat_rows.append(["**Total**", f"**{f(len(E))}**", f"**{f(len(D))}**", f"**{f(lab.get('cao', 0))}**", f"**{f(lab.get('vua', 0))}**", f"**{f(lab.get('thap', 0))}**"])
    cat_table = md(cat_rows[:1]) + "\n|---|---:|---:|---:|---:|---:|\n" + md(cat_rows[2:])
    xt = [["Origin of the value (`nguon_ban_dich`)", "`cao`", "`vua`", "`thap`", "Total"]]
    for o in ["tu_dien_quy_tac", "giu_ban_dich_don_vi", "gemma_sua", "dich_moi", "qwen_sua"]:
        d = D[D.nguon_ban_dich == o].do_tin_cay.value_counts()
        xt.append([f"`{o}`", f(d.get("cao", 0)), f(d.get("vua", 0)), f(d.get("thap", 0)), f(d.sum())])
    xtab = md(xt[:1]) + "\n|---|---:|---:|---:|---:|\n" + md(xt[1:])
    src_table = md([[f"`{o}`", MA_SO[o], f((D.nguon_ban_dich == o).sum()), p((D.nguon_ban_dich == o).mean())]
                    for o in ["tu_dien_quy_tac", "giu_ban_dich_don_vi", "gemma_sua", "dich_moi", "qwen_sua"]])
    ex = [["category", "name_zh", "gia_tri_zh", "name_vi", "gia_tri_vi", "do_tin_cay", "nguon_ban_dich"], ["---"] * 7]
    for o in ["tu_dien_quy_tac", "giu_ban_dich_don_vi", "gemma_sua", "dich_moi", "qwen_sua"]:
        d = D[(D.nguon_ban_dich == o) & (D.gia_tri_zh.str.len() <= 30) & ~D.name_zh.str.fullmatch(r"\d+") & D.gia_tri_zh.map(lambda s: bool(HAN.search(s)))].sort_values("pair_id")
        r = d.iloc[min(300, len(d) - 1) if o != "qwen_sua" else 0]
        ex.append([r.category, r.name_zh, r.gia_tri_zh, r.name_vi, r.gia_tri_vi, r.do_tin_cay, r.nguon_ban_dich])
    examples = md(ex[:1]) + "\n|---|---|---|---|---|---|---|\n" + md(ex[2:])
    combos = D.groupby(["name_zh", "gia_tri_zh"]).gia_tri_vi.nunique()
    nombos = D.groupby("name_zh").name_vi.nunique()
    app = D.groupby("product_id").size()
    st = {
        "DATE": "2026-10-26", "NHAN": nhan, "N_PAIRS": f(len(D)), "N_PROD": f(len(E)), "N_SET1": f(len(B1)), "N_NEW": f(n_new), "N_NEW_KEPT": f(n_new),
        "N_CLUSTERS": f(cb["cum"]), "N_BACK": f(cb["cap_bo_1_cat_rieng_dua_vao"]), "N_DROP_PREP": f(cb["cap_bo_dai_khong_co_ban"]), "N_PAIRS_IN": f(cb["cap"]),
        "N_PAIRS_DROPPED": f(tk["cap_bo_khong_co_ban"]), "PAIRS_PER_PRODUCT": f"{len(D) / len(E):.0f}", "SIZE_PAIRS": f((DST / PAIRS).stat().st_size),
        "SHA_PAIRS": sha(DST / PAIRS), "SIZE_PROD": f((DST / PRODUCTS).stat().st_size), "SHA_PROD": sha(DST / PRODUCTS), "SIZE_PROV": f((DST / PROV).stat().st_size),
        "SHA_PROV": sha(DST / PROV), "SIZE_SUMS": f((DST / "SHA256SUMS.txt").stat().st_size), "SHA_SET1": SHA_SET1,
        "PAIRS_CAO": f(lab.get("cao", 0)), "PAIRS_CAO_P": p(lab.get("cao", 0) / len(D)), "PAIRS_VUA": f(lab.get("vua", 0)), "PAIRS_VUA_P": p(lab.get("vua", 0) / len(D)),
        "PAIRS_THAP": f(lab.get("thap", 0)), "PAIRS_THAP_P": p(lab.get("thap", 0) / len(D)), "PROD_VUA": f(prod.get("vua", 0)), "PROD_VUA_P": p(prod.get("vua", 0) / len(E)),
        "PROD_THAP": f(prod.get("thap", 0)), "PROD_THAP_P": p(prod.get("thap", 0) / len(E)), "XTAB": xtab, "SRC_TABLE": src_table, "CAT_TABLE": cat_table, "EXAMPLES": examples,
        "N_NAMES_GEMMA": f((C.n_nguon_AB == "sua_A").sum()), "N_COMBOS": f(len(combos)), "N_NAMES_DISTINCT": f(D.name_zh.nunique()), "N_VALUES_DISTINCT": f(D.gia_tri_zh.nunique()),
        "MULTI_COMBOS": f((combos > 1).sum()), "MULTI_NAMES": f((nombos > 1).sum()), "LEN_ZH": pt(D.gia_tri_zh.str.len()), "LEN_VI": pt(D.gia_tri_vi.str.len()),
        "ATTR_PER_PRODUCT": f"{app.mean():.1f} / {f(app.median())} / {f(app.max())}", "DESC_ZH": pt(E.description_zh.str.len()), "DESC_VI": pt(E.description_vi.str.len()),
        "N_HAN_VI": f(han_v.sum()), "PROMPT_VERSION": dich_cum_zv.PHIEN_BAN_CAU_LENH,
        "GEMMA_DATES": "25-26 Oct 2026", "GEMMA_RUNS": str(len(gem)), "GEMMA_HOURS": f"{sum(c['giay'] for c in gem.values()) / 3600:.1f}", "GEMMA_REQ": f(gsum("yeu_cau")),
        "GEMMA_TOK_IN": f"{gsum('token_vao') / 1e6:.1f}", "GEMMA_TOK_OUT": f"{gsum('token_ra') / 1e6:.1f}", "QWEN_DATE": "26 Oct 2026", "QWEN_SECONDS": f(qw["giay"]),
        "GEMMA_OK": f(tk["cum_gemma"]["ok"]), "GEMMA_BAD": f(tk["cum_gemma"]["loi_cau_truc"]), "GEMMA_MISSING": f(tk["cum_gemma"].get("chua_dich", 0)),
        "QWEN_N": f(sum(tk["cum_qwen"][k] for k in ("ok", "loi_cau_truc"))), "QWEN_OK": f(tk["cum_qwen"]["ok"]), "QWEN_BAD": f(tk["cum_qwen"]["loi_cau_truc"]),
        "QWEN_NEW_NOT_USED": f(tk["qwen_dich_moi_khong_dung"]), "QWEN_VALUES": f(tk["qwen_thay_gemma_gia_tri"]), "QWEN_NAMES": f(tk["qwen_thay_gemma_ten"]),
        "GEMMA_FAIL_VALUES": f(((C.v_luat_A != "") & C.v_A.notna()).sum()), "GEMMA_FAIL_NAMES": f(((C.n_luat_A != "") & C.n_A.notna()).sum()),
        "N_NAMES_RESTORED": f(nh["ten"]), "N_VALUE_FIXES": f(sum(nk.values())),
        "F_DOC_LAP": f(nk.get("独立站:Trạm độc lập", 0)), "F_ZHONG": f(nk.get("中筒:Thân trung", 0)), "F_DUAN": f(nk.get("短筒:Thân ngắn", 0)), "F_HOUDI": f(nk.get("厚底:Đế độn mũi", 0)),
        "F_RISHI": f(nk.get("日式:Phong cách tối giản Nhật", 0)), "F_WUYIN": f(nk.get("无印风:Phong cách không nhãn hiệu", 0)), "F_JINGTOU": f(nk.get("镜头:thấu kính", 0)),
        "F_LALIAN": f(nk.get("拉链:Ngăn kéo", 0)), "F_TAOTOU": f(nk.get("套头:Xỏ chân", 0)),
        "RULE_RERUN": f"{f(len(sel))} values re-checked; {co_co} still carry a rule flag, both inside very long lists kept from the draft",
        "N_DIFF_VALUES": f(n_diff_v), "N_DIFF_VALUES_P": p(n_diff_v / len(B1)), "N_DIFF_NAMES": f(n_diff_n),
        "PROD_THAP0": f((E.so_cap_thap == 0).sum()), "PROD_THAP1": f((E.so_cap_thap <= 1).sum()), "PROD_THAP2": f((E.so_cap_thap <= 2).sum()),
        "FIXED_P": p((D.nguon_ban_dich == "tu_dien_quy_tac").mean()), "N_FULLWIDTH": f((D.gia_tri_vi.str.contains("[，。；：（）、！？]") & ~han_v).sum()),
        "N_NO_REASON": f(tk["sua_khong_ly_do"]),
    }
    readme = TPL.read_text(encoding="utf-8")
    readme = re.sub(r"@@([A-Z0-9_]+)@@", lambda mm: str(st[mm.group(1)]), readme)
    assert "@@" not in readme
    (DST / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    for x in (PAIRS, PRODUCTS, PROV):
        assert sha(DST / x) in readme and f"{(DST / x).stat().st_size:,}" in readme, f"README chưa ghi đúng SHA256 hoặc kích thước của {x}"
    print({x.name: x.stat().st_size for x in DST.iterdir()}, "| nhãn bản", nhan)
    print("cặp", len(D), "| sản phẩm", len(E), "| cặp mới so với bộ 1:", n_new, "| giá trị khác bộ 1:", n_diff_v, "| tên khác bộ 1:", n_diff_n)


if __name__ == "__main__":
    main()
