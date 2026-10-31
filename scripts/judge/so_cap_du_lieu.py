"""Dựng bảng các mục cần so cặp từ dữ liệu có sẵn (chỉ đọc, không gọi API).

 - title: hai bản dịch title của Gemma/Qwen (HF best_cometkiwi, bản chép ở data/snapshot/title_candidates/) cho các sản phẩm
   còn lại sau bước làm sạch; điểm CometKiwi được GIỮ trong bảng để phân tích nhưng KHÔNG bao giờ đưa vào lời nhắc.
 - cum (bộ CỤM): từng cặp (tên: giá trị) mà bản Gemma (A) và bản Qwen (B) sau rà soát KHÁC nhau, gộp trùng theo
   (nguồn, A, B, nhóm hàng). A = Gemma, B = Qwen (ghi chú CLAUDE.md: 'always take the preferred model A (Gemma)').
 - pair (bộ PAIR): từng ĐƠN VỊ (tên hoặc giá trị thuộc tính, mỗi đơn vị dịch riêng) mà Gemma (A) và Qwen (B) dịch khác nhau."""
import hashlib
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
GIU = GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu.parquet"
TITLE = {"1688": GOC / "data" / "snapshot" / "title_candidates" / "1688_title_best_cometkiwi.parquet",
         "tiki": GOC / "data" / "snapshot" / "title_candidates" / "tiki_title_best_cometkiwi.parquet"}
# (file có ứng viên A/B, cột nguồn tên, cột nguồn giá trị, file các cặp ĐÃ PHÁT HÀNH hoặc None). Bộ 3 (Tiki): so_sanh_cum có thêm 27 cặp
# không phát hành nên phải lọc theo cap_cum (kiểm 30/10: ghép lại mô tả từ cap_cum khớp 100% bản chính thức v2).
CAP = {"1688": (GOC / "data" / "processed" / "song_ngu_4" / "cap_cum_20261026_2123.parquet", "name_zh", "gia_tri_zh", None),
       "tiki": (GOC / "data" / "processed" / "song_ngu_3" / "so_sanh_cum_20261024_1616.parquet", "name_vi", "gia_tri_vi",
                GOC / "data" / "processed" / "song_ngu_3" / "cap_cum_20261024_1616.parquet")}
HUONG = {"1688": "zh->vi", "tiki": "vi->zh"}
RA = GOC / "data" / "processed" / "so_cap"
BIN_MARGIN = [0.0, 0.02, 0.05, 0.1, 9.9]


def chuan(s):
    return " ".join(unicodedata.normalize("NFKC", "" if pd.isna(s) else str(s)).casefold().split())


def chuan_khac(s):
    """Dùng để quyết định hai ứng viên có THỰC SỰ khác nhau không: bỏ mọi khoảng trắng và gạch chéo/dấu chấm câu thường gặp, nên
    '/ 玻璃' với '/玻璃', ':' với '：' không tạo thành một mục (kiểm tay 30/10: nhiều mục chỉ khác khoảng trắng hoặc dấu câu)."""
    s = unicodedata.normalize("NFKC", "" if pd.isna(s) else str(s)).casefold()
    return "".join(ch for ch in s if not ch.isspace() and unicodedata.category(ch)[0] != "P")


def co_chu(s):
    """Series: True khi chuỗi KHÔNG rỗng (sau khi bỏ khoảng trắng). Ứng viên rỗng không phải là một BẢN DỊCH để so (xem ghi chú ở xay_cum)."""
    return s.fillna("").astype(str).str.strip() != ""


def co_noi_dung(s):
    return isinstance(s, str) and s.strip() != ""


def doc_giu(duong=GIU):
    return pd.read_parquet(duong, columns=["product_id", "source", "category", "title_zh", "title_vi"])


def xay_title(giu=None, duong=None):
    giu = doc_giu() if giu is None else giu
    out = []
    for src, p in (duong or TITLE).items():
        t = pd.read_parquet(p)
        nguon_col = "title_zh" if src == "1688" else "title_vi"       # cột chứa văn bản NGUỒN gốc
        g = giu[giu.source == src][["product_id", "category"]]
        d = t.merge(g, on="product_id", how="inner")
        khac = (d.gemma4_translation.map(chuan_khac) != d.qwen3_14b_awq_translation.map(chuan_khac)) \
            & co_chu(d.gemma4_translation) & co_chu(d.qwen3_14b_awq_translation)        # ứng viên rỗng: không phải cuộc đấu
        d = d[khac]
        out.append(pd.DataFrame({
            "item_id": "title:" + src + ":" + d.product_id.astype(str), "kind": "title", "source": src, "huong": HUONG[src],
            "category": d.category.values, "nguon": d[nguon_col].values,
            "cand_gemma": d.gemma4_translation.values, "cand_qwen": d.qwen3_14b_awq_translation.values,
            "kiwi_g": d.cometkiwi_gemma4.values, "kiwi_q": d.cometkiwi_qwen3_14b_awq.values,
            "san_pham_n": 1, "vi_du_product_id": d.product_id.astype(str).values}))
    r = pd.concat(out, ignore_index=True)
    r["margin"] = (r.kiwi_g - r.kiwi_q).abs()
    r["bin_margin"] = pd.cut(r.margin, BIN_MARGIN, right=False, labels=["<0.02", "0.02-0.05", "0.05-0.1", ">=0.1"]).astype(str)
    return r


# Bản PAIR: mỗi tên / giá trị thuộc tính được dịch RIÊNG theo "đơn vị" (bộ 1 và bộ 2); A = Gemma, B = Qwen.
DON_VI = {"1688": (GOC / "data" / "processed" / "song_ngu_1" / "don_vi_da_chon_20261022_2126.parquet",
                   GOC / "data" / "processed" / "song_ngu_1" / "bo_song_ngu_1_day_du_20261022_2126.parquet",
                   "name_zh", "gia_tri_zh", "van_ban_zh"),
          "tiki": (GOC / "data" / "processed" / "song_ngu_2" / "don_vi_da_chon_20261023_1656.parquet",
                   GOC / "data" / "processed" / "song_ngu_2" / "bo_song_ngu_2_day_du_20261023_1656.parquet",
                   "name_vi", "gia_tri_vi", "van_ban_nguon")}


def xay_don_vi(giu=None, duong=None):
    """Đơn vị của bản PAIR mà Gemma (A) và Qwen (B) dịch KHÁC nhau và được các sản phẩm còn lại dùng. Bỏ đơn vị `tu == 'co_dinh'`
    (từ điển đã duyệt khóa cứng, không phải cuộc đấu giữa hai mô hình)."""
    giu = doc_giu() if giu is None else giu
    out = []
    for src, (pu, pc, sn, sv, vn) in (duong or DON_VI).items():
        u = pd.read_parquet(pu, columns=["ma_don_vi", "loai", vn, "A", "B", "tu", "category", "kiwi_A", "kiwi_B"])
        c = pd.read_parquet(pc, columns=["product_id", sn, sv])
        c = c[c.product_id.astype(str).isin(set(giu[giu.source == src].product_id))].copy()
        # Đơn vị là THEO NHÓM HÀNG (cùng chữ nguồn có thể có nhiều ma_don_vi, mỗi nhóm hàng một đơn vị): đếm số sản phẩm THEO (chữ, nhóm hàng).
        # Trước 31/10 đếm theo chữ không kể nhóm hàng nên san_pham_n bị đếm lặp (vd "其他": 5.314 ở mọi biến thể nhóm hàng).
        c["nhom"] = c.product_id.astype(str).map(dict(zip(giu.product_id.astype(str), giu.category)))
        dem_ten = c.groupby([sn, "nhom"]).product_id.nunique()
        dem_gt = c.groupby([sv, "nhom"]).product_id.nunique()
        u["san_pham_n"] = [int((dem_ten if l == "ten" else dem_gt).get((t, cg), 0)) for l, t, cg in zip(u.loai, u[vn], u.category)]
        u = u[(u.san_pham_n > 0) & (u.tu != "co_dinh")]
        for col in ("A", "B"):
            u[col] = u[col].fillna("").astype(str)
        u = u[(u.A.map(chuan_khac) != u.B.map(chuan_khac)) & co_chu(u.A) & co_chu(u.B)]
        out.append(pd.DataFrame({
            "item_id": "unit:" + src + ":" + u.ma_don_vi.astype(str), "kind": np.where(u.loai == "ten", "unit_name", "unit_value"),
            "source": src, "huong": HUONG[src], "category": u.category.values, "nguon": u[vn].astype(str).values,
            "cand_gemma": u.A.values, "cand_qwen": u.B.values, "kiwi_g": u.kiwi_A.values, "kiwi_q": u.kiwi_B.values,
            "san_pham_n": u.san_pham_n.values, "ma_don_vi": u.ma_don_vi.astype(str).values}))
    r = pd.concat(out, ignore_index=True)
    r["margin"] = (r.kiwi_g - r.kiwi_q).abs()
    r["bin_margin"] = pd.cut(r.margin, BIN_MARGIN, right=False, labels=["<0.02", "0.02-0.05", "0.05-0.1", ">=0.1"]).astype(str)
    return r


def khoa_cum(nguon, cand_gemma, cand_qwen, category):
    """Khóa của một mục cặp bộ CỤM; dùng CHUNG cho dựng mục và ghép lại (ghep_bo.py) để hai nơi không lệch nhau."""
    return nguon + "\x1f" + cand_gemma + "\x1f" + cand_qwen + "\x1f" + category


def id_cum(src, khoa):
    return "pair:" + src + ":" + hashlib.md5(khoa.encode("utf-8")).hexdigest()[:16]


def xay_cum(giu=None, duong=None):
    giu = doc_giu() if giu is None else giu
    out = []
    for src, (p, sn, sv, *phat_hanh) in (duong or CAP).items():
        phat_hanh = phat_hanh[0] if phat_hanh else None
        d = pd.read_parquet(p, columns=(["pair_id"] if phat_hanh else []) +
                            ["product_id", sn, sv, "n_cuoi_A", "v_cuoi_A", "n_cuoi_B", "v_cuoi_B"])
        if phat_hanh:                                           # chỉ giữ các cặp đã phát hành
            d = d[d.pair_id.isin(set(pd.read_parquet(phat_hanh, columns=["pair_id"]).pair_id))]
        g = giu[giu.source == src][["product_id", "category"]]
        d = d.merge(g, on="product_id", how="inner")
        for c in ("n_cuoi_A", "v_cuoi_A", "n_cuoi_B", "v_cuoi_B", sn, sv):
            d[c] = d[c].fillna("").astype(str)
        # CẶP CÓ MỘT BẢN RỖNG KHÔNG PHẢI CUỘC ĐẤU: ở bộ 1688, 6.193 cặp `dich_moi` chỉ có bản của Gemma (v_cuoi_B rỗng), Qwen không dịch cặp đó.
        # Trước 31/10 các cặp này bị đưa vào bảng mục (23% mục cụm có ngữ cảnh) và judge "thắng" cho Gemma một cách tầm thường.
        khac = ((d.n_cuoi_A.map(chuan_khac) != d.n_cuoi_B.map(chuan_khac)) | (d.v_cuoi_A.map(chuan_khac) != d.v_cuoi_B.map(chuan_khac))) \
            & co_chu(d.v_cuoi_A) & co_chu(d.v_cuoi_B)
        d = d[khac].copy()
        d["nguon"] = d[sn] + ": " + d[sv]
        d["cand_gemma"] = d.n_cuoi_A + ": " + d.v_cuoi_A
        d["cand_qwen"] = d.n_cuoi_B + ": " + d.v_cuoi_B
        d["khoa"] = khoa_cum(d.nguon, d.cand_gemma, d.cand_qwen, d.category)
        gop = d.groupby("khoa", sort=False).agg(nguon=("nguon", "first"), cand_gemma=("cand_gemma", "first"),
                                                cand_qwen=("cand_qwen", "first"), category=("category", "first"),
                                                san_pham_n=("product_id", "nunique"),
                                                vi_du_product_id=("product_id", "first")).reset_index()
        gop["item_id"] = gop.khoa.map(lambda k: id_cum(src, k))
        gop["kind"], gop["source"], gop["huong"] = "cum_pair", src, HUONG[src]
        out.append(gop.drop(columns="khoa"))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- bộ CỤM CÓ NGỮ CẢNH (quyết định 31/10/2026 của Nhật Anh)
# Bản cụm được dịch khi mô hình thấy cả tiêu đề và danh sách thuộc tính của sản phẩm, nên judge cũng được xem đúng ngữ cảnh đó.
# Mỗi mục là MỘT (sản phẩm, cặp) chứ không gộp trùng theo cặp như muc_cum.parquet. Ngữ cảnh chỉ gồm văn bản NGUỒN (không có bản dịch nào).
NGU_CANH_TOI_DA = 3000      # ký tự; sản phẩm dài hơn thì giữ cặp đang xét và các cặp kề nó, ghi chú số cặp bị bỏ


def id_cum_ctx(src, product_id, pair_id):
    """Khóa mục cụm CÓ ngữ cảnh; dùng chung cho dựng mục và ghép lại (ghep_bo.py)."""
    return f"cumctx:{src}:{product_id}:{pair_id}"


def _than_ngu_canh(cac_cap, k, toi_da):
    """Phần danh sách thuộc tính: luôn giữ cặp thứ k, rồi thêm các cặp kề nó (xen kẽ hai phía) cho tới hết ngân sách ký tự."""
    toi_da = toi_da or NGU_CANH_TOI_DA
    dong = [f"- {c}" for c in cac_cap]
    chon = {k}
    dai = len(dong[k]) + 1
    lan, gan = 1, True
    while gan:
        gan = False
        for j in (k - lan, k + lan):
            if 0 <= j < len(dong) and dai + len(dong[j]) + 1 <= toi_da:
                chon.add(j)
                dai += len(dong[j]) + 1
                gan = True
        lan += 1
        if lan > len(dong):
            break
    giu = [dong[j] for j in sorted(chon)]
    if len(chon) < len(dong):
        giu.append(f"- (… {len(dong) - len(chon)} more attributes omitted)")
    return "All attributes of this product:\n" + "\n".join(giu)


def dung_ngu_canh(tieu_de, cac_cap, k, toi_da=None):
    """tieu_de: tiêu đề NGUỒN; cac_cap: các 'tên: giá trị' NGUỒN của sản phẩm theo thứ tự pair_id; k: chỉ số cặp đang xét."""
    return f"Title: {tieu_de}\n" + _than_ngu_canh(cac_cap, k, toi_da)


def dung_ngu_canh_thuoc_tinh(cac_cap, toi_da=None):
    """Ngữ cảnh chỉ gồm thuộc tính (dùng khi văn bản đang xét CHÍNH LÀ tiêu đề). Rỗng nếu sản phẩm không có thuộc tính."""
    return _than_ngu_canh(cac_cap, 0, toi_da) if len(cac_cap) else ""


def doc_cap_cum(src, p, sn, sv, phat_hanh, giu):
    """Các cặp (tên: giá trị) của bộ CỤM cho một nguồn: chỉ sản phẩm còn trong `giu`, Tiki chỉ cặp đã phát hành, xếp theo
    (product_id, pair_id). Thêm cột nguon, cand_gemma, cand_qwen, khac. Trả (bảng, {product_id: tiêu đề NGUỒN})."""
    d = pd.read_parquet(p, columns=["pair_id", "product_id", sn, sv, "n_cuoi_A", "v_cuoi_A", "n_cuoi_B", "v_cuoi_B"])
    if phat_hanh:
        d = d[d.pair_id.isin(set(pd.read_parquet(phat_hanh, columns=["pair_id"]).pair_id))]
    g = giu[giu.source == src]
    tcol = "title_zh" if src == "1688" else "title_vi"                # tiêu đề NGUỒN (1688 = Trung, Tiki = Việt)
    tieu_de = dict(zip(g.product_id.astype(str), g[tcol])) if tcol in g.columns else {}
    d = d.merge(g[["product_id", "category"]], on="product_id", how="inner")
    for c in ("n_cuoi_A", "v_cuoi_A", "n_cuoi_B", "v_cuoi_B", sn, sv):
        d[c] = d[c].fillna("").astype(str)
    d = d.sort_values(["product_id", "pair_id"], kind="stable").reset_index(drop=True)
    d["nguon"] = d[sn] + ": " + d[sv]
    d["cand_gemma"] = d.n_cuoi_A + ": " + d.v_cuoi_A
    d["cand_qwen"] = d.n_cuoi_B + ": " + d.v_cuoi_B
    d["khac"] = ((d.n_cuoi_A.map(chuan_khac) != d.n_cuoi_B.map(chuan_khac)) | (d.v_cuoi_A.map(chuan_khac) != d.v_cuoi_B.map(chuan_khac))) \
        & co_chu(d.v_cuoi_A) & co_chu(d.v_cuoi_B)                      # bản rỗng không phải cuộc đấu (xem xay_cum)
    return d, tieu_de


def xay_cum_ctx(giu=None, duong=None, toi_da=None):
    giu = doc_giu() if giu is None else giu
    out = []
    for src, (p, sn, sv, *phat_hanh) in (duong or CAP).items():
        phat_hanh = phat_hanh[0] if phat_hanh else None
        d, tieu_de = doc_cap_cum(src, p, sn, sv, phat_hanh, giu)
        dong = []
        for pid, nhom in d.groupby("product_id", sort=False):
            cac_cap, td = nhom.nguon.tolist(), tieu_de.get(str(pid), "")
            for k, (_, r) in enumerate(nhom.iterrows()):
                if not r.khac:
                    continue
                khoa = khoa_cum(r.nguon, r.cand_gemma, r.cand_qwen, r.category)
                dong.append({"item_id": id_cum_ctx(src, pid, r.pair_id), "kind": "cum_pair", "source": src, "huong": HUONG[src],
                             "category": r.category, "nguon": r.nguon, "cand_gemma": r.cand_gemma, "cand_qwen": r.cand_qwen,
                             "ngu_canh": dung_ngu_canh(td, cac_cap, k, toi_da), "san_pham_n": 1, "vi_du_product_id": str(pid),
                             "pair_id": int(r.pair_id), "id_khong_ngu_canh": id_cum(src, khoa)})
        out.append(pd.DataFrame(dong))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- TẦNG 2 của title và pair (31/10/2026)
# Tầng 1 hỏi KHÔNG ngữ cảnh. Judge gần như tất định (chạy lặp giống 97%) nên hỏi lại cùng lời nhắc không gỡ được mục 'không nhất quán';
# tầng 2 chạy lại CHỈ các mục đó với ngữ cảnh NGUỒN. Hai bảng dưới có CÙNG item_id với tầng 1 (title_ctx = muc_title, pair_ctx =
# muc_pair) nên kết quả tầng 2 ghi đè đúng mục của tầng 1; mục không có ngữ cảnh để thêm thì không đưa vào (hỏi lại cũng vô ích).
def xay_title_ctx(giu=None, duong_title=None, duong_cap=None, toi_da=None):
    """title_ctx: các mục của muc_title kèm ngữ cảnh = thuộc tính NGUỒN của chính sản phẩm đó (không có dòng Title vì văn bản đang
    xét CHÍNH LÀ tiêu đề). Trả (bảng, số mục bỏ vì sản phẩm không có thuộc tính nào)."""
    giu = doc_giu() if giu is None else giu
    t = xay_title(giu, duong_title)
    thuoc_tinh = {}
    for src, (p, sn, sv, *phat_hanh) in (duong_cap or CAP).items():
        d, _ = doc_cap_cum(src, p, sn, sv, phat_hanh[0] if phat_hanh else None, giu)
        for pid, nhom in d.groupby("product_id", sort=False):
            thuoc_tinh[(src, str(pid))] = nhom.nguon.tolist()
    t["ngu_canh"] = [dung_ngu_canh_thuoc_tinh(thuoc_tinh.get((s, str(p)), []), toi_da)
                     for s, p in zip(t.source, t.vi_du_product_id)]
    co = t.ngu_canh != ""
    return t[co].reset_index(drop=True), int((~co).sum())


def xay_pair_ctx(giu=None, duong=None, toi_da=None):
    """pair_ctx: các mục của muc_pair kèm ngữ cảnh từ MỘT sản phẩm đại diện (product_id nhỏ nhất theo thứ tự chuỗi trong số sản phẩm
    còn lại dùng đơn vị đó; chọn cố định nên tái lập được): tiêu đề NGUỒN + thuộc tính NGUỒN, giữ cặp chứa đơn vị và các cặp kề.
    Quyết định áp cho đơn vị ở MỌI sản phẩm dùng nó. `vi_du_product_id` ghi sản phẩm đại diện."""
    giu = doc_giu() if giu is None else giu
    u = xay_don_vi(giu, duong)
    out = []
    for src, (pu, pc, sn, sv, vn) in (duong or DON_VI).items():
        mu = u[u.source == src].copy()
        if mu.empty:
            continue
        g = giu[giu.source == src]
        tcol = "title_zh" if src == "1688" else "title_vi"
        tieu_de = dict(zip(g.product_id.astype(str), g[tcol])) if tcol in g.columns else {}
        c = pd.read_parquet(pc, columns=["pair_id", "product_id", sn, sv])
        c = c[c.product_id.astype(str).isin(set(g.product_id.astype(str)))]
        for col in (sn, sv):
            c[col] = c[col].fillna("").astype(str)
        c = c.sort_values(["product_id", "pair_id"], kind="stable")
        c["nhom"] = c.product_id.astype(str).map(dict(zip(giu.product_id.astype(str), giu.category)))
        rep_ten = c.groupby([sn, "nhom"], sort=False).product_id.min()          # sản phẩm đại diện phải CÙNG nhóm hàng với đơn vị
        rep_gt = c.groupby([sv, "nhom"], sort=False).product_id.min()
        mu["vi_du_product_id"] = [str((rep_ten if k == "unit_name" else rep_gt).get((t, cg), "")) for k, t, cg in zip(mu.kind, mu.nguon, mu.category)]
        mu = mu[mu.vi_du_product_id != ""]
        sp = {pid: (n.tolist(), v.tolist()) for pid, n, v in
              ((pid, grp[sn], grp[sv]) for pid, grp in c[c.product_id.astype(str).isin(set(mu.vi_du_product_id))].groupby("product_id"))}
        ctx = []
        for k, t, pid in zip(mu.kind, mu.nguon, mu.vi_du_product_id):
            ten, gt = sp[pid]
            cap = [f"{a}: {b}" for a, b in zip(ten, gt)]
            ds = ten if k == "unit_name" else gt
            vi_tri = ds.index(t) if t in ds else 0
            ctx.append(dung_ngu_canh(tieu_de.get(pid, ""), cap, vi_tri, toi_da))
        mu["ngu_canh"] = ctx
        out.append(mu)
    return pd.concat(out, ignore_index=True)


def xay_cum_lai(c, cx):
    """Bảng cụm LAI (chốt 31/10/2026): cặp duy nhất dùng bởi ĐÚNG 1 sản phẩm -> mục CÓ ngữ cảnh của sản phẩm đó (id cumctx:...);
    cặp dùng bởi >= 2 sản phẩm -> mục KHÔNG ngữ cảnh dùng chung (id pair:...). c = muc_cum (cặp duy nhất), cx = muc_cum_ctx."""
    mot = c[c.san_pham_n == 1]
    ctx = cx[cx.id_khong_ngu_canh.isin(set(mot.item_id))].copy()
    nhieu = c[c.san_pham_n >= 2].copy()
    nhieu["ngu_canh"] = None
    nhieu["id_khong_ngu_canh"] = None
    cot = ["item_id", "kind", "source", "huong", "category", "nguon", "cand_gemma", "cand_qwen", "san_pham_n", "vi_du_product_id",
           "ngu_canh", "id_khong_ngu_canh"]
    r = pd.concat([ctx[cot], nhieu[cot]], ignore_index=True)
    assert len(r) == len(c) and r.item_id.is_unique, (len(r), len(c))
    return r


def main():
    RA.mkdir(parents=True, exist_ok=True)
    giu = doc_giu()
    t, c, u = xay_title(giu), xay_cum(giu), xay_don_vi(giu)
    cx = xay_cum_ctx(giu)
    tx, bo_tx = xay_title_ctx(giu)
    ux = xay_pair_ctx(giu)
    t.to_parquet(RA / "muc_title.parquet", index=False)
    c.to_parquet(RA / "muc_cum.parquet", index=False)
    cx.to_parquet(RA / "muc_cum_ctx.parquet", index=False)
    u.to_parquet(RA / "muc_pair.parquet", index=False)
    tx.to_parquet(RA / "muc_title_ctx.parquet", index=False)
    ux.to_parquet(RA / "muc_pair_ctx.parquet", index=False)
    lai = xay_cum_lai(c, cx)
    lai.to_parquet(RA / "muc_cum_lai.parquet", index=False)
    print("CỤM LAI:", len(lai), "mục | có ngữ cảnh (cặp 1 sản phẩm):", int(lai.ngu_canh.notna().sum()), "| không ngữ cảnh (cặp >= 2 sản phẩm):", int(lai.ngu_canh.isna().sum()))
    for ten, x in (("TITLE tầng 2", tx), ("PAIR tầng 2", ux)):
        dai = x.ngu_canh.str.len()
        print(f"{ten}: {len(x)} mục có ngữ cảnh | độ dài ngữ cảnh (ký tự): trung vị {int(dai.median())}, p90 {int(dai.quantile(.9))}, "
              f"tối đa {int(dai.max())} | bị cắt: {int(x.ngu_canh.str.contains('more attributes omitted', regex=False).sum())}")
    print(f"  title: bỏ {bo_tx} mục vì sản phẩm không có thuộc tính | pair: {len(u) - len(ux)} mục không tìm được sản phẩm đại diện")
    dai = cx.ngu_canh.str.len()
    print("CỤM CÓ NGỮ CẢNH:", len(cx), "mục (sản phẩm, cặp) |", cx.source.value_counts().to_dict(),
          "| độ dài ngữ cảnh (ký tự): trung vị", int(dai.median()), "p90", int(dai.quantile(.9)), "tối đa", int(dai.max()),
          "| bị cắt:", int(cx.ngu_canh.str.contains("more attributes omitted", regex=False).sum()))
    print("title khác nhau:", len(t), t.source.value_counts().to_dict(), "| theo margin:", t.bin_margin.value_counts().to_dict())
    print("CỤM: cặp (tên: giá trị) khác nhau, đã gộp trùng:", len(c), c.source.value_counts().to_dict(),
          "| phủ", int(c.san_pham_n.sum()), "lượt sản phẩm-cặp")
    print("PAIR: đơn vị (tên / giá trị) khác nhau:", len(u), u.source.value_counts().to_dict(), "|", u.kind.value_counts().to_dict())
    print("TỔNG số mục:", len(t) + len(c) + len(u), "=> số lời gọi (2 thứ tự):", 2 * (len(t) + len(c) + len(u)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
