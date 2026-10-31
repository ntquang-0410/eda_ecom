"""GHÉP HAI BỘ DỮ LIỆU CUỐI (CỤM và PAIR; mỗi bộ có Title + Mô tả) từ quyết định chọn Gemma/Qwen của judge.

Bộ cuối có các cột của bộ official + `desc_nguon_dich_goc` (đổi tên từ `desc_nguon_dich`) + `desc_so_manh_da_doi` (Cách 1, 31/10/2026).

Quy tắc:
 - Chỉ thay khi judge CÓ NGƯỜI THẮNG (hai lượt cùng chọn một mô hình). Mục judge chưa chấm = giữ nguyên hiện trạng. Mục judge chấm
   nhưng không có người thắng (hòa / không nhất quán) xử lý theo --khi-khong-quyet (mặc định giu_hien_tai = giữ nguyên hiện trạng,
   không đưa thêm thiên vị mới; gemma / qwen chỉ để phân tích độ nhạy). CÙNG chính sách cho cả hai bộ.
 - TẦNG: kết quả có `tang: 2` (title_ctx / pair_ctx, chạy lại có ngữ cảnh các mục 'không nhất quán' của tầng 1) THẮNG kết quả tầng 1 của
   cùng mục, bất kể thứ tự tên đợt trong dòng lệnh; cùng tầng thì đợt đứng sau ghi đè đợt đứng trước.
 - Chỉ thay vào chỗ mà bản dịch hiện trạng đúng bằng ứng viên A hoặc B (nếu không là bản đã khóa bằng từ điển/quy tắc: giữ nguyên).
 - Bản được chọn bị CHẶN nếu làm xuất hiện chữ Hán sót (bản Việt của 1688) hoặc thêm chữ Việt có dấu (bản Trung của Tiki).
 - Không sửa gì ngoài các chỗ đó; chạy với 0 quyết định phải cho lại ĐÚNG bộ gốc (--kiem-vong).
 - ĐÁNH DẤU: mỗi mục TRANH CHẤP (hai ứng viên khác nhau thật sự) được gán nhãn nguồn quyết định (judge_t1, judge_t2, hoa_that,
   khong_nhat_quan, chua_cham) và ghi vào file RIÊNG bo_<cum|pair>_danh_dau.parquet (mỗi sản phẩm một dòng); bộ huấn luyện
   bo_<cum|pair>_chon_deepseek.parquet KHÔNG đổi cột.

  python -I -X utf8 ghep_bo.py --kiem-vong
  python -I -X utf8 ghep_bo.py --title t1,t2 --cum c1 --pair p1,p2 [--khi-khong-quyet giu_hien_tai]"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ft_clean"))
import luat_ft as LF  # noqa: E402
import so_cap_du_lieu as DL  # noqa: E402
from so_cap_du_lieu import chuan, chuan_khac  # noqa: E402

GOC = DL.GOC
BASE_CUM = GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu.parquet"
BASE_PAIR = GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu_pair.parquet"
RA_QD = DL.RA
RA_BO = GOC / "data" / "processed" / "bo_chon"
D3, D4 = GOC / "data" / "processed" / "song_ngu_3", GOC / "data" / "processed" / "song_ngu_4"
D1, D2 = GOC / "data" / "processed" / "song_ngu_1", GOC / "data" / "processed" / "song_ngu_2"
COT_GOC = ["product_id", "source", "category", "title_zh", "title_vi", "description_zh", "description_vi",
           "title_nguon_dich", "desc_nguon_dich"]
DICH = {"1688": ("title_vi", "description_vi"), "tiki": ("title_zh", "description_zh")}     # cột bản DỊCH theo nguồn
MO_HINH_TITLE = {"gemma": "gemma4", "qwen": "qwen3_14b_awq"}
NHAN = ("judge_t1", "judge_t2", "judge_tuong_duong", "hoa_that", "khong_nhat_quan", "chua_cham")      # nguồn quyết định của một mục TRANH CHẤP


# ---------------------------------------------------------------- quyết định của judge
def doc_ket_qua_day_du(*ten, thu_muc=None):
    """item_id -> (ket_qua_tho, tang). Chỉ bản ghi ok. Tầng cao hơn thắng; cùng tầng thì đợt đứng sau ghi đè đợt đứng trước."""
    kq = {}
    for t in ten:
        for dong in open(Path(thu_muc or RA_QD) / f"ket_qua_{t}.jsonl", encoding="utf-8"):
            d = json.loads(dong)
            if d.get("trang_thai") != "ok":
                continue
            tang = d.get("tang", 1)
            cu = kq.get(d["item_id"])
            if cu is None or tang >= cu[1]:
                kq[d["item_id"]] = (d["ket_qua"], tang)
    return kq


def doc_ket_qua_va_tuong_duong(*ten, thu_muc=None):
    """Như doc_ket_qua_day_du nhưng thêm tập item_id mà bản ghi CHỌN (tầng cao nhất, đợt sau cùng) có judge KHAI TƯƠNG ĐƯƠNG
    (chế độ một lượt, rubric v2: luot[].tuong_duong). Trả (kqdd, tập item_id tương đương)."""
    kq, co = {}, set()
    for t in ten:
        for dong in open(Path(thu_muc or RA_QD) / f"ket_qua_{t}.jsonl", encoding="utf-8"):
            d = json.loads(dong)
            if d.get("trang_thai") != "ok":
                continue
            tang = d.get("tang", 1)
            cu = kq.get(d["item_id"])
            if cu is None or tang >= cu[1]:
                kq[d["item_id"]] = (d["ket_qua"], tang)
                if any(l.get("tuong_duong") for l in d.get("luot", [])):
                    co.add(d["item_id"])
                else:
                    co.discard(d["item_id"])
    return kq, co


def ap_chinh_sach_tuong_duong(kqdd, co, chinh_sach_td):
    """Mục judge khai TƯƠNG ĐƯƠNG: nhãn "judge_tuong_duong". giu_hien_tai (mặc định) = giữ bản hiện tại (hai bản ngang nhau thì không có lý do
    đổi, tránh đổi ngẫu nhiên); theo_judge = lấy bản judge chọn (đồng xu công bằng vì thứ tự đã ngẫu nhiên hóa). Trả (qd, kqdd mới)."""
    qd = tu_ket_qua_day_du(kqdd)
    kq2 = dict(kqdd)
    for i in co:
        if i in kq2:
            kq2[i] = ("tuong_duong", kq2[i][1])
            if chinh_sach_td == "giu_hien_tai":
                qd[i] = "tuong_duong"
    return qd, kq2


def tu_ket_qua_day_du(kqdd):
    return {i: (k if k in ("gemma", "qwen") else "khong_quyet") for i, (k, _) in kqdd.items()}


def doc_quyet_dinh(*ten, thu_muc=None):
    """item_id -> 'gemma' | 'qwen' | 'khong_quyet'."""
    return tu_ket_qua_day_du(doc_ket_qua_day_du(*ten, thu_muc=thu_muc))


def chon(item_id, qd, chinh_sach):
    """Trả 'gemma' | 'qwen' | None (None = giữ hiện trạng)."""
    q = qd.get(item_id)
    if q in ("gemma", "qwen"):
        return q
    if q == "khong_quyet" and chinh_sach in ("gemma", "qwen"):
        return chinh_sach
    return None                                   # gồm q == "tuong_duong" (giữ hiện trạng bất kể chính sách khi không quyết)


def nhan_nguon(item_id, qd, kqdd=None):
    """Nhãn nguồn quyết định của một mục TRANH CHẤP. kqdd (từ doc_ket_qua_day_du) cho nhãn đủ chi tiết; không có thì suy từ qd."""
    if kqdd is not None:
        r = kqdd.get(item_id)
        if r is None:
            return "chua_cham"
        k, tang = r
        if k == "tuong_duong":
            return "judge_tuong_duong"
        if k in ("gemma", "qwen"):
            return "judge_t2" if tang == 2 else "judge_t1"
        return "hoa_that" if k == "hoa_that" else "khong_nhat_quan"
    q = qd.get(item_id)
    return "chua_cham" if q is None else ("judge_t1" if q in ("gemma", "qwen") else "khong_nhat_quan")


class Dau:
    """Thu nhãn nguồn quyết định cho từng mục tranh chấp trong lúc ghép (không ảnh hưởng kết quả ghép)."""

    def __init__(self):
        self.dong = []

    def them(self, phan, product_id, item_id, nhan, da_doi):
        self.dong.append((phan, str(product_id), item_id, nhan, bool(da_doi)))

    def dem(self):
        c = Counter((p, n) for p, _, _, n, _ in self.dong)
        return {f"{p}:{n}": c[(p, n)] for p in ("title", "mo_ta") for n in NHAN if c[(p, n)]}

    def bang(self, san_pham):
        """Mỗi sản phẩm một dòng: nhãn title và số mảnh mô tả theo từng nhãn."""
        ti = {}
        mo = {}
        for phan, pid, _, nhan, da_doi in self.dong:
            if phan == "title":
                ti[pid] = (nhan, da_doi)
            else:
                c = mo.setdefault(pid, Counter())
                c["tranh_chap"] += 1
                c[nhan] += 1
                c["da_doi"] += int(da_doi)
        pids = [str(p) for p in san_pham]
        out = {"product_id": pids,
               "title_nhan": [ti.get(p, ("khong_tranh_chap", False))[0] for p in pids],
               "title_da_doi": [ti.get(p, ("", False))[1] for p in pids],
               "mo_ta_so_manh_tranh_chap": [mo.get(p, Counter())["tranh_chap"] for p in pids]}
        for n in NHAN:
            out[f"mo_ta_so_manh_{n}"] = [mo.get(p, Counter())[n] for p in pids]
        out["mo_ta_so_manh_da_doi"] = [mo.get(p, Counter())["da_doi"] for p in pids]
        return pd.DataFrame(out)


def an_toan(src, moi, cu):
    """Chặn bản mới nếu làm xuất hiện chữ Hán sót (1688) hoặc thêm chữ Việt có dấu vào bản Trung (Tiki)."""
    if src == "1688":
        return LF.han_sot("1688", moi) <= LF.han_sot("1688", cu)
    return len(LF.DAU_VI.findall(moi)) <= len(LF.DAU_VI.findall(cu))


# ---------------------------------------------------------------- title
def ghep_title(base, title_dfs, qd, chinh_sach, kqdd=None, dau=None):
    """base: bảng 9 cột. title_dfs: {nguồn: bảng có product_id, gemma4_translation, qwen3_14b_awq_translation}."""
    d = base.copy()
    dem = Counter()
    cand = {}
    for src, t in title_dfs.items():
        for pid, g, q in zip(t.product_id.astype(str), t.gemma4_translation, t.qwen3_14b_awq_translation):
            cand[f"title:{src}:{pid}"] = (g, q)
    cot, chon_ct = {}, {}
    for i, (pid, src) in enumerate(zip(d.product_id, d.source)):
        iid = f"title:{src}:{pid}"
        c = chon(iid, qd, chinh_sach)
        tranh_chap = iid in cand and chuan_khac(cand[iid][0]) != chuan_khac(cand[iid][1]) \
            and DL.co_noi_dung(cand[iid][0]) and DL.co_noi_dung(cand[iid][1])        # bản rỗng không phải cuộc đấu
        da_doi = False
        if c is not None:
            g, q = cand[iid]
            moi, cu = (g if c == "gemma" else q), d.at[d.index[i], DICH[src][0]]
            if chuan(moi) == chuan(cu):
                dem["trung_hien_trang"] += 1
            elif not an_toan(src, moi, cu):
                dem["chan_do_chu_han_hoac_chu_viet"] += 1
            else:
                cot[d.index[i]] = moi
                chon_ct[d.index[i]] = c
                dem[f"doi_sang_{c}"] += 1
                da_doi = True
        if dau is not None and tranh_chap:
            dau.them("title", pid, iid, nhan_nguon(iid, qd, kqdd), da_doi)
    for idx, moi in cot.items():
        src = d.at[idx, "source"]
        d.at[idx, DICH[src][0]] = moi
        d.at[idx, "title_nguon_dich"] = MO_HINH_TITLE[chon_ct[idx]]
    return d, dict(dem)


# ---------------------------------------------------------------- mô tả bộ CỤM (theo cặp "tên: giá trị")
def ghep_mota_cum(base, pairs, qd, chinh_sach, ngu_canh=False, kqdd=None, dau=None):
    """pairs (một nguồn): pair_id, product_id, nguon_ten, nguon_gt, cuoi_ten, cuoi_gt, nA, vA, nB, vB.
    Trả {product_id: mô tả mới} CHỈ cho sản phẩm có ít nhất một cặp bị đổi, và bộ đếm.
    ngu_canh=True: quyết định theo (sản phẩm, cặp) từ lần chạy CÓ ngữ cảnh (id cumctx:...), mỗi cặp của mỗi sản phẩm có quyết định riêng;
    ngu_canh=False: quyết định theo cặp duy nhất, không ngữ cảnh (id pair:...)."""
    src = base.source.iloc[0]
    cat = dict(zip(base.product_id, base.category))
    p = pairs[pairs.product_id.isin(set(base.product_id))].sort_values(["product_id", "pair_id"]).copy()
    for c in ("nguon_ten", "nguon_gt", "cuoi_ten", "cuoi_gt", "nA", "vA", "nB", "vB"):
        p[c] = p[c].fillna("").astype(str)
    p["catg"] = p.product_id.map(cat)
    nguon, ca, cb = p.nguon_ten + ": " + p.nguon_gt, p.nA + ": " + p.vA, p.nB + ": " + p.vB
    if ngu_canh == "lai":                       # cặp 1 sản phẩm: quyết định CÓ ngữ cảnh (cumctx); cặp >= 2 sản phẩm: quyết định dùng chung (pair)
        ids_ctx = [DL.id_cum_ctx(src, pid, pair_id) for pid, pair_id in zip(p.product_id, p.pair_id)]
        khoa_s = DL.khoa_cum(nguon, ca, cb, p.catg)
        ids_khoa = [DL.id_cum(src, k) for k in khoa_s]
        dung_chung = p.groupby(khoa_s.values).product_id.transform("nunique").values   # số SẢN PHẨM dùng cùng một cặp (khóa = nguồn + hai ứng viên + nhóm hàng), như san_pham_n
        ids = [b if n >= 2 else a for a, b, n in zip(ids_ctx, ids_khoa, dung_chung)]
    elif ngu_canh:
        ids = [DL.id_cum_ctx(src, pid, pair_id) for pid, pair_id in zip(p.product_id, p.pair_id)]
    else:
        ids = [DL.id_cum(src, k) for k in DL.khoa_cum(nguon, ca, cb, p.catg)]
    dem, ten_moi, gt_moi = Counter(), [], []
    for i, (tcu, gcu, a, b, ni, vi_, nb, vb, pid) in enumerate(zip(p.cuoi_ten, p.cuoi_gt, ca, cb, p.nA, p.vA, p.nB, p.vB, p.product_id)):
        c = chon(ids[i], qd, chinh_sach)
        cu = tcu + ": " + gcu
        da_doi = False
        if c is None:
            ten_moi.append(tcu), gt_moi.append(gcu)
        else:
            la = "A" if chuan(cu) == chuan(a) else ("B" if chuan(cu) == chuan(b) else None)
            if la is None:
                dem["khoa_giu_nguyen"] += 1                  # bản hiện trạng không phải A/B: từ điển / quy tắc khóa
                ten_moi.append(tcu), gt_moi.append(gcu)
            else:
                moi_t, moi_g, moi = (ni, vi_, a) if c == "gemma" else (nb, vb, b)
                if chuan(moi) == chuan(cu):
                    dem["trung_hien_trang"] += 1
                    ten_moi.append(tcu), gt_moi.append(gcu)
                elif not an_toan(src, moi, cu):
                    dem["chan_do_chu_han_hoac_chu_viet"] += 1
                    ten_moi.append(tcu), gt_moi.append(gcu)
                else:
                    dem[f"doi_sang_{c}"] += 1
                    ten_moi.append(moi_t), gt_moi.append(moi_g)
                    da_doi = True
        if dau is not None and (chuan_khac(ni) != chuan_khac(nb) or chuan_khac(vi_) != chuan_khac(vb)) \
                and DL.co_noi_dung(vi_) and DL.co_noi_dung(vb):                   # bản rỗng (Qwen không dịch cặp dich_moi) không phải cuộc đấu
            dau.them("mo_ta", pid, ids[i], nhan_nguon(ids[i], qd, kqdd), da_doi)
    p["t"] = [a + ": " + b for a, b in zip(ten_moi, gt_moi)]
    return p.groupby("product_id").t.agg("; ".join), dict(dem)


# ---------------------------------------------------------------- mô tả bộ PAIR (theo đơn vị: tên / giá trị dịch riêng)
def ghep_mota_pair(base, pairs, units, qd, chinh_sach, kqdd=None, dau=None):
    """pairs: pair_id, product_id, nguon_ten, nguon_gt, cuoi_ten, cuoi_gt. units: ma_don_vi, loai ('ten'|'gia_tri'), van_ban, A, B
    (và, nếu có, tu: đơn vị tu == 'co_dinh' là từ điển khóa cứng, không tranh chấp)."""
    src = base.source.iloc[0]
    um = {}
    co_tu = "tu" in units.columns
    co_nhom = "category" in units.columns
    cat_sp = dict(zip(base.product_id, base.category))
    # Đơn vị là THEO NHÓM HÀNG: cùng chữ nguồn có thể có nhiều ma_don_vi (mỗi nhóm hàng một đơn vị); phải ghép đúng đơn vị của nhóm hàng của sản phẩm
    nhom_u = units.category.tolist() if co_nhom else [None] * len(units)
    for i, (ma, loai, vb, a, b) in enumerate(zip(units.ma_don_vi.astype(str), units.loai, units.van_ban, units.A, units.B)):
        co_dinh = bool(co_tu and units.tu.iloc[i] == "co_dinh")
        um.setdefault((loai, vb, nhom_u[i]), (ma, "" if pd.isna(a) else str(a), "" if pd.isna(b) else str(b), co_dinh))
    p = pairs[pairs.product_id.isin(set(base.product_id))].sort_values(["product_id", "pair_id"]).copy()
    for c in ("nguon_ten", "nguon_gt", "cuoi_ten", "cuoi_gt"):
        p[c] = p[c].fillna("").astype(str)
    dem = Counter()

    def doi(pid, loai, nguon, cu):
        u = um.get((loai, nguon, cat_sp.get(pid) if co_nhom else None))
        if u is None:
            return cu
        iid = f"unit:{src}:{u[0]}"
        c = chon(iid, qd, chinh_sach)
        da_doi = False
        ket_qua = cu
        if c is not None:
            if chuan(cu) not in (chuan(u[1]), chuan(u[2])):
                dem["khoa_giu_nguyen"] += 1
            else:
                moi = u[1] if c == "gemma" else u[2]
                if chuan(moi) == chuan(cu):
                    dem["trung_hien_trang"] += 1
                elif not an_toan(src, moi, cu):
                    dem["chan_do_chu_han_hoac_chu_viet"] += 1
                else:
                    dem[f"doi_sang_{c}"] += 1
                    ket_qua, da_doi = moi, True
        if dau is not None and not u[3] and chuan_khac(u[1]) != chuan_khac(u[2]) and DL.co_noi_dung(u[1]) and DL.co_noi_dung(u[2]):
            dau.them("mo_ta", pid, iid, nhan_nguon(iid, qd, kqdd), da_doi)
        return ket_qua
    t = [doi(pid, "ten", n, ct) + ": " + doi(pid, "gia_tri", g, cg)
         for pid, n, g, ct, cg in zip(p.product_id, p.nguon_ten, p.nguon_gt, p.cuoi_ten, p.cuoi_gt)]
    p["t"] = t
    return p.groupby("product_id").t.agg("\n".join), dict(dem)


# ---------------------------------------------------------------- nạp dữ liệu thật
def nap_cap_cum():
    a = pd.read_parquet(D4 / "cap_cum_20261026_2123.parquet", columns=[
        "pair_id", "product_id", "name_zh", "gia_tri_zh", "name_vi", "gia_tri_vi", "n_cuoi_A", "v_cuoi_A", "n_cuoi_B", "v_cuoi_B"])
    a = a.rename(columns={"name_zh": "nguon_ten", "gia_tri_zh": "nguon_gt", "name_vi": "cuoi_ten", "gia_tri_vi": "cuoi_gt",
                          "n_cuoi_A": "nA", "v_cuoi_A": "vA", "n_cuoi_B": "nB", "v_cuoi_B": "vB"})
    f = pd.read_parquet(D3 / "cap_cum_20261024_1616.parquet", columns=["pair_id", "product_id", "name_vi", "gia_tri_vi",
                                                                      "name_zh", "gia_tri_zh"])
    s = pd.read_parquet(D3 / "so_sanh_cum_20261024_1616.parquet", columns=["pair_id", "n_cuoi_A", "v_cuoi_A", "n_cuoi_B", "v_cuoi_B"])
    t = f.merge(s, on="pair_id", how="left").rename(columns={
        "name_vi": "nguon_ten", "gia_tri_vi": "nguon_gt", "name_zh": "cuoi_ten", "gia_tri_zh": "cuoi_gt",
        "n_cuoi_A": "nA", "v_cuoi_A": "vA", "n_cuoi_B": "nB", "v_cuoi_B": "vB"})
    return {"1688": a, "tiki": t}


def nap_pair_unit():
    out = {}
    for src, (d, uf, nguon_cols, cuoi_cols, vb) in {
            "1688": (D1, "don_vi_da_chon_20261022_2126.parquet", ("name_zh", "gia_tri_zh"), ("name_vi", "gia_tri_vi"), "van_ban_zh"),
            "tiki": (D2, "don_vi_da_chon_20261023_1656.parquet", ("name_vi", "gia_tri_vi"), ("name_zh", "gia_tri_zh"), "van_ban_nguon")}.items():
        pc = d / ("bo_song_ngu_1_day_du_20261022_2126.parquet" if src == "1688" else "bo_song_ngu_2_day_du_20261023_1656.parquet")
        p = pd.read_parquet(pc, columns=["pair_id", "product_id", *nguon_cols, *cuoi_cols])
        p = p.rename(columns=dict(zip(nguon_cols + cuoi_cols, ("nguon_ten", "nguon_gt", "cuoi_ten", "cuoi_gt"))))
        u = pd.read_parquet(d / uf, columns=["ma_don_vi", "loai", vb, "A", "B", "tu", "category"]).rename(columns={vb: "van_ban"})
        out[src] = (p, u)
    return out


def nap_title():
    return {s: pd.read_parquet(p) for s, p in DL.TITLE.items()}


# ---------------------------------------------------------------- ghép
def ghep_mot_bo(base, loai_bo, qd_title, qd_mota, chinh_sach, du_lieu, cum_ngu_canh=False, kqdd_title=None, kqdd_mota=None, dau=None):
    """loai_bo: 'cum' | 'pair'. Trả (bảng mới, thống kê). dau: đối tượng Dau để thu nhãn nguồn quyết định (tùy chọn)."""
    d, tk = ghep_title(base, du_lieu["title"], qd_title, chinh_sach, kqdd_title, dau)
    tk = {"title": tk}
    d = d.copy()
    for src in ("1688", "tiki"):
        b = d[d.source == src]
        if b.empty:
            continue
        if loai_bo == "cum":
            moi, t = ghep_mota_cum(b, du_lieu["cum"][src], qd_mota, chinh_sach, cum_ngu_canh, kqdd_mota, dau)
        else:
            p, u = du_lieu["pair"][src]
            moi, t = ghep_mota_pair(b, p, u, qd_mota, chinh_sach, kqdd_mota, dau)
        cot = DICH[src][1]
        d[cot] = [moi.get(pid, cu) if pid in moi.index else cu for pid, cu in zip(d.product_id, d[cot])]
        tk[f"mo_ta_{src}"] = t
    return d, tk


def hoan_thien_cot(moi, bang_dau):
    """CÁCH 1 (Nhật Anh chốt 31/10/2026) cho cột nhãn nguồn dịch của MÔ TẢ: sau khi judge đổi các mảnh, nhãn `desc_nguon_dich` của bộ gốc có thể
    không còn đúng (mô tả đã thành hỗn hợp), nên GIỮ nhãn cũ nhưng đổi tên thành `desc_nguon_dich_goc` (= nhãn của bản gốc) và thêm cột
    `desc_so_manh_da_doi` (số mảnh mô tả judge đã đổi). `title_nguon_dich` giữ nguyên tên (đã cập nhật đúng khi đổi tiêu đề)."""
    assert moi.product_id.astype(str).tolist() == bang_dau.product_id.astype(str).tolist(), "thứ tự sản phẩm của file đánh dấu phải khớp bộ cuối"
    r = moi.rename(columns={"desc_nguon_dich": "desc_nguon_dich_goc"}).copy()
    r["desc_so_manh_da_doi"] = bang_dau.mo_ta_so_manh_da_doi.astype(int).values
    return r


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--title", default="", help="tên các đợt kết quả title (cách nhau bằng dấu phẩy; gồm cả đợt tầng 2 title_ctx)")
    a.add_argument("--cum", default="", help="tên các đợt kết quả cặp bộ CỤM (mặc định là kết quả loại cum_ctx, CÓ ngữ cảnh)")
    a.add_argument("--cum-khong-ngu-canh", action="store_true",
                   help="(cũ) đọc --cum như kết quả loại cum (không ngữ cảnh, theo cặp duy nhất); tương đương --cum-che-do khong")
    a.add_argument("--cum-che-do", choices=["lai", "ctx", "khong"], default="lai",
                   help="cách đọc kết quả bộ CỤM: lai (mặc định, chốt 31/10: cặp 1 sản phẩm có ngữ cảnh, cặp >= 2 sản phẩm không), "
                        "ctx (mọi (sản phẩm, cặp) có ngữ cảnh), khong (cặp duy nhất không ngữ cảnh)")
    a.add_argument("--khi-tuong-duong", choices=["giu_hien_tai", "theo_judge"], default="giu_hien_tai",
                   help="mục judge khai TƯƠNG ĐƯƠNG (chế độ một lượt): giữ bản hiện tại (mặc định, đề xuất của Claude, chờ Nhật Anh xác nhận) hoặc lấy bản judge chọn")
    a.add_argument("--pair", default="", help="tên các đợt kết quả đơn vị bộ PAIR (gồm cả đợt tầng 2 pair_ctx)")
    a.add_argument("--khi-khong-quyet", choices=["giu_hien_tai", "gemma", "qwen"], default="giu_hien_tai")
    a.add_argument("--kiem-vong", action="store_true", help="không có quyết định nào: kết quả phải GIỐNG HỆT bộ gốc")
    a.add_argument("--ra", default=str(RA_BO))
    a.add_argument("--thu-muc-qd", default=None)
    x = a.parse_args(argv)
    ten = lambda s: [t for t in s.split(",") if t]      # noqa: E731
    kt, ct = doc_ket_qua_va_tuong_duong(*ten(x.title), thu_muc=x.thu_muc_qd)
    kc, cc = doc_ket_qua_va_tuong_duong(*ten(x.cum), thu_muc=x.thu_muc_qd)
    kp, cp = doc_ket_qua_va_tuong_duong(*ten(x.pair), thu_muc=x.thu_muc_qd)
    if x.kiem_vong:
        kt, kc, kp, ct, cc, cp = {}, {}, {}, set(), set(), set()
    qt, kt = ap_chinh_sach_tuong_duong(kt, ct, x.khi_tuong_duong)
    qc, kc = ap_chinh_sach_tuong_duong(kc, cc, x.khi_tuong_duong)
    qp, kp = ap_chinh_sach_tuong_duong(kp, cp, x.khi_tuong_duong)
    cum_che_do = "khong" if x.cum_khong_ngu_canh else x.cum_che_do
    cum_ngu_canh = {"lai": "lai", "ctx": True, "khong": False}[cum_che_do]
    du = {"title": nap_title(), "cum": nap_cap_cum(), "pair": nap_pair_unit()}
    ra = Path(x.ra)
    ra.mkdir(parents=True, exist_ok=True)
    tong = {}
    for loai, base_path, qd, kd in (("cum", BASE_CUM, qc, kc), ("pair", BASE_PAIR, qp, kp)):
        base = pd.read_parquet(base_path)
        dau = Dau()
        moi, tk = ghep_mot_bo(base, loai, qt, qd, x.khi_khong_quyet, du, cum_ngu_canh=cum_ngu_canh,
                              kqdd_title=kt, kqdd_mota=kd, dau=dau)
        tk["nhan_nguon_quyet_dinh"] = dau.dem()
        bang_dau = dau.bang(moi.product_id)
        moi = hoan_thien_cot(moi, bang_dau)
        if x.kiem_vong:
            ten_moi = {"desc_nguon_dich": "desc_nguon_dich_goc"}
            same = {c: bool((moi[ten_moi.get(c, c)].fillna("") == base[c].fillna("")).all()) for c in COT_GOC}
            same["desc_so_manh_da_doi_deu_bang_0"] = bool((moi.desc_so_manh_da_doi == 0).all())
            tk["kiem_vong_giong_het"] = same
            assert all(same.values()), same
        else:
            moi.to_parquet(ra / f"bo_{loai}_chon_deepseek.parquet", index=False)
            bang_dau.to_parquet(ra / f"bo_{loai}_danh_dau.parquet", index=False)
        tong[loai] = {"so_san_pham": len(moi), "thong_ke": tk}
    tong["chinh_sach_khi_khong_quyet"] = x.khi_khong_quyet
    tong["chinh_sach_khi_tuong_duong"] = x.khi_tuong_duong
    tong["cum_che_do"] = cum_che_do
    if not x.kiem_vong:
        (ra / "ghep_bo_thong_ke.json").write_text(json.dumps(tong, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tong, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
