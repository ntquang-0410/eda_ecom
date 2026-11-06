"""Kiểm tín hiệu "mức ~11% nâng mô tả dòng mới zh->vi" trên bản dịch đã có của fold 0 (bước 1, Prompt 791, 792). Không train, không dịch lại, không dùng GPU.

Chạy:  python -I -X utf8 scripts/ft_train/kiem_dong_moi.py [--thu-muc-tl <.../tl>] [--ra <thu_muc>]
Ghi <thu_muc>/kiem_dong_moi.md và kiem_dong_moi.json.

Luật khóa trước khi xem số, riêng từng bản, so mức ~11% với mốc 1 : 1:
  Chỉ số quyết định là chênh chrF trung bình theo sản phẩm trên dòng mới của mô tả zh->vi, bootstrap 2.000 lần, seed 42.
  Cận dưới khoảng 95% > 0 thì "qua" (đề xuất kiểm trên fold khác), còn lại là "dung".
  Chỉ so sản phẩm khớp dòng ở cả hai mức. Số phụ không quyết định; corpus mâu thuẫn với chỉ số chính thì báo, không tự chọn.
Fold 0 là nơi tín hiệu được phát hiện, nên "qua" chỉ có nghĩa là đáng kiểm tiếp, không đủ để đổi tỉ lệ.
KHÔNG đọc test_khoa, split.parquet, ds_test.txt: chỉ đọc pool_*.jsonl và các bản dịch của fold 0."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tong_hop_ti_le import DUONG_DAN_THAT, MOC_KY_VONG, _doc_jsonl, _FT, _nap_cong_cu, chuan_bi_mau  # noqa: E402

SO_LAN, SEED, SO_CAT = 2000, 42, 5
QUA, DUNG, CHUA_DU = "qua", "dung", "chua_du_du_lieu"
NHANH = ("cum", "pair")
MUC_THU = {"cum": "cum_f0_p11", "pair": "pair_f0_p11"}
THU_MUC_TL = _FT / "kaggle_tl" / "phien_1_20261105_231941" / "tl"
# Số đã công bố trong bao_cao_ti_le.md, chạy lại phải ra đúng, lệch là dừng (sai số do báo cáo làm tròn 2 chữ số)
KY_VONG_MOI = {("cum", "zh->vi"): {"moc": (77.17, 895), "muc": (79.05, 895)}, ("cum", "vi->zh"): {"moc": (77.79, 456), "muc": (78.20, 456)},
               ("pair", "zh->vi"): {"moc": (76.36, 849), "muc": (78.68, 849)}, ("pair", "vi->zh"): {"moc": (75.12, 454), "muc": (75.91, 454)}}
KY_VONG_TITLE = {"cum": {"moc": 60.22, "muc": 60.36}, "pair": {"moc": 60.35, "muc": 59.39}}
DUNG_SAI_BAO_CAO = 0.0051


# ---------------------------------------------------------------- hàm thuần
def lay_dong_moi(hang, huong):
    """hang: bản ghi của cham_diem.cham (có huong, loai, can_dong, _dong). Trả ({id: [(hyp, ref)] dòng mới}, số mô tả không khớp dòng, số mô tả không có dòng mới)."""
    ra, khong_can, khong_moi = {}, 0, 0
    for r in hang:
        if r["loai"] != "mo_ta" or r["huong"] != huong:
            continue
        if not r.get("can_dong") == True:  # noqa: E712  can_dong có thể là NaN
            khong_can += 1
            continue
        moi = [(hy, rf) for thay, hy, rf in r["_dong"] if not thay]
        if moi:
            ra[str(r["id"])] = moi
        else:
            khong_moi += 1
    return ra, khong_can, khong_moi


def ghep_cap(a, b, thu_tu):
    """Chỉ giữ sản phẩm có ở cả hai bên, theo thu_tu. Cùng sản phẩm phải cùng tham chiếu, không thì lỗi vì đang so hai thứ khác nhau."""
    ra = []
    for k in thu_tu:
        if k in a and k in b:
            ref_a, ref_b = [x[1] for x in a[k]], [x[1] for x in b[k]]
            if ref_a != ref_b:
                raise RuntimeError(f"tham_chieu_khac_nhau:{k}")
            ra.append({"id": k, "ref": ref_a, "hyp_a": [x[0] for x in a[k]], "hyp_b": [x[0] for x in b[k]]})
    return ra


def ket_luan(thap):
    """Luật khóa: cận dưới > 0 là qua, còn lại là dừng; thiếu số thì không kết luận."""
    if thap is None or not math.isfinite(thap):
        return CHUA_DU
    return QUA if thap > 0 else DUNG


def mau_thuan(chinh, corpus):
    """True khi hai khoảng cho kết luận khác nhau; báo cho người đọc, không đổi kết luận."""
    return ket_luan(chinh["thap"]) != ket_luan(corpus["thap"])


def tap_trung(chenh, so_cat=SO_CAT):
    """Độ tập trung của chênh theo sản phẩm: mức tăng có do vài sản phẩm kéo lên không."""
    x = sorted(chenh)
    return {"n": len(x), "tang": sum(v > 0 for v in x), "giam": sum(v < 0 for v in x), "bang": sum(v == 0 for v in x),
            "trung_binh": statistics.fmean(x), "trung_vi": statistics.median(x),
            "tb_bo_tang_nhat": statistics.fmean(x[:-so_cat]) if len(x) > so_cat else None,
            "tb_cat_hai_dau": statistics.fmean(x[so_cat:-so_cat]) if len(x) > 2 * so_cat else None}


def cum_chung_dong(cac_ref):
    """cac_ref: danh sách các dòng tham chiếu của từng sản phẩm. Sản phẩm chung ít nhất một dòng (không phân biệt hoa thường) thì cùng cụm, kể cả qua trung gian.
    Trả mã cụm từng sản phẩm, đánh số theo thứ tự xuất hiện."""
    cha = list(range(len(cac_ref)))

    def goc(i):
        while cha[i] != i:
            cha[i] = cha[cha[i]]
            i = cha[i]
        return i

    gap = {}
    for i, ds in enumerate(cac_ref):
        for dong in ds:
            k = dong.lower()
            if k in gap:
                cha[goc(i)] = goc(gap[k])
            else:
                gap[k] = i
    ma, ra = {}, []
    for i in range(len(cac_ref)):
        ra.append(ma.setdefault(goc(i), len(ma)))
    return ra


def bootstrap_cum(chenh, cum, so_lan=SO_LAN, seed=SEED):
    """Lấy mẫu lại theo cụm, chênh = tổng chênh các sản phẩm được bốc chia số sản phẩm được bốc. Sản phẩm trùng dòng không được tính như bằng chứng độc lập."""
    theo = {}
    for x, c in zip(chenh, cum):
        theo.setdefault(c, []).append(x)
    ds = [(sum(v), len(v)) for v in theo.values()]
    rng = np.random.default_rng(seed)
    kq = []
    for _ in range(so_lan):
        i = rng.integers(0, len(ds), len(ds))
        kq.append(sum(ds[k][0] for k in i) / sum(ds[k][1] for k in i))
    kq.sort()
    return {"chenh": sum(chenh) / len(chenh), "thap": kq[int(0.025 * so_lan)], "cao": kq[int(0.975 * so_lan) - 1], "so_cum": len(ds)}


def ti_le_chi_khac_hoa(cap):
    """Trong các dòng mà hai mức dịch khác nhau, bao nhiêu dòng chỉ khác chữ hoa, chữ thường."""
    khac = chi_hoa = 0
    for c in cap:
        for x, y in zip(c["hyp_a"], c["hyp_b"]):
            if x != y:
                khac += 1
                chi_hoa += x.lower() == y.lower()
    return {"dong_khac": khac, "chi_khac_hoa": chi_hoa, "ti_le": chi_hoa / khac if khac else None}


def bootstrap_corpus(nhom, so_lan=SO_LAN, seed=SEED):
    """nhom: danh sách (metric, A, B); A, B là mảng thống kê chrF cộng được, mỗi hàng một sản phẩm, cùng thứ tự.
    Lấy mẫu lại sản phẩm trong từng nhóm, chênh = trung bình đều theo nhóm của chrF corpus A trừ B."""
    def diem(m, hang):
        return float(m._aggregate_and_compute([[int(v) for v in hang]]).score)

    rng = np.random.default_rng(seed)
    kq = []
    for _ in range(so_lan):
        c = []
        for m, a, b in nhom:
            i = rng.integers(0, len(a), len(a))
            c.append(diem(m, a[i].sum(axis=0)) - diem(m, b[i].sum(axis=0)))
        kq.append(sum(c) / len(c))
    kq.sort()
    goc = [diem(m, a.sum(axis=0)) - diem(m, b.sum(axis=0)) for m, a, b in nhom]
    return {"chenh": sum(goc) / len(goc), "thap": kq[int(0.025 * so_lan)], "cao": kq[int(0.975 * so_lan) - 1]}


def thong_ke(metric, hyps, refs):
    """Tổng thống kê chrF của một sản phẩm (các dòng cộng lại), dùng được cho corpus."""
    s = metric._extract_corpus_statistics(list(hyps), [list(refs)])
    return np.array(s, dtype=np.int64).sum(axis=0)


# ---------------------------------------------------------------- tính trên bản dịch
def cham_file(cong_cu, nhanh, bo, duong_dan):
    K, D, pd = cong_cu
    co = set(bo["ids"])
    du = [r for r in _doc_jsonl(duong_dan) if isinstance(r, dict) and str(r.get("id")) in co]
    thieu = co - {str(r["id"]) for r in du}
    if thieu:
        raise RuntimeError(f"thiếu {len(thieu)}/{len(co)} bản dịch trong {duong_dan}")
    df, th = D.cham(pd.DataFrame(bo["mau"]), pd.DataFrame(du), nhanh=nhanh, tap_da_thay=bo["tap"])
    return df, th


def so_dong_moi(cong_cu, hang_moc, hang_muc, thu_tu, huong):
    """So mức với mốc trên dòng mới của một hướng, chỉ trên sản phẩm khớp dòng ở cả hai mức."""
    K, D, _ = cong_cu
    metric = D.tao_chrf(huong)
    moc, kc_moc, km_moc = lay_dong_moi(hang_moc, huong)
    muc, kc_muc, km_muc = lay_dong_moi(hang_muc, huong)
    cap = ghep_cap(muc, moc, thu_tu)
    if not cap:
        raise RuntimeError(f"không có sản phẩm chung: {huong}")
    a = [D.chrf_corpus(huong, c["hyp_a"], c["ref"]) for c in cap]
    b = [D.chrf_corpus(huong, c["hyp_b"], c["ref"]) for c in cap]
    if not all(math.isfinite(v) for v in a + b):
        raise RuntimeError("có chrF không hữu hạn")
    chinh = K.bootstrap_chenh(a, b, [f"{huong}|mo_ta"] * len(cap), so_lan=SO_LAN, seed=SEED)
    tk_a = np.array([thong_ke(metric, c["hyp_a"], c["ref"]) for c in cap])
    tk_b = np.array([thong_ke(metric, c["hyp_b"], c["ref"]) for c in cap])
    corpus = bootstrap_corpus([(metric, tk_a, tk_b)])
    hyp_a = [x for c in cap for x in c["hyp_a"]]
    hyp_b = [x for c in cap for x in c["hyp_b"]]
    ref = [x for c in cap for x in c["ref"]]
    corpus["muc"], corpus["moc"] = D.chrf_corpus(huong, hyp_a, ref), D.chrf_corpus(huong, hyp_b, ref)
    if abs(corpus["muc"] - corpus["moc"] - corpus["chenh"]) > 1e-6:
        raise RuntimeError("cộng thống kê không ra đúng chrF corpus của sacrebleu")
    chenh = [x - y for x, y in zip(a, b)]
    # Độ nhạy (thêm sau khi đọc mẫu, không thay luật khóa): bỏ khác biệt chữ hoa, và không coi sản phẩm trùng dòng là độc lập
    a_th = [D.chrf_corpus(huong, [x.lower() for x in c["hyp_a"]], [x.lower() for x in c["ref"]]) for c in cap]
    b_th = [D.chrf_corpus(huong, [x.lower() for x in c["hyp_b"]], [x.lower() for x in c["ref"]]) for c in cap]
    chenh_th = [x - y for x, y in zip(a_th, b_th)]
    cum = cum_chung_dong([c["ref"] for c in cap])
    tk_a_th = np.array([thong_ke(metric, [x.lower() for x in c["hyp_a"]], [x.lower() for x in c["ref"]]) for c in cap])
    tk_b_th = np.array([thong_ke(metric, [x.lower() for x in c["hyp_b"]], [x.lower() for x in c["ref"]]) for c in cap])
    do_nhay = {"khong_phan_biet_hoa": K.bootstrap_chenh(a_th, b_th, [f"{huong}|mo_ta"] * len(cap), so_lan=SO_LAN, seed=SEED),
               "khong_phan_biet_hoa_corpus": bootstrap_corpus([(metric, tk_a_th, tk_b_th)]),
               "theo_cum": bootstrap_cum(chenh, cum),
               "khong_phan_biet_hoa_theo_cum": bootstrap_cum(chenh_th, cum),
               "chi_khac_hoa": ti_le_chi_khac_hoa(cap)}
    xep = sorted(range(len(cap)), key=lambda i: chenh[i])
    def mau(i):
        return {"id": cap[i]["id"], "chenh": chenh[i], "muc": a[i], "moc": b[i], "n_dong": len(cap[i]["ref"]),
                "dong": [{"ref": r, "muc": x, "moc": y} for r, x, y in zip(cap[i]["ref"], cap[i]["hyp_a"], cap[i]["hyp_b"])
                         if x != y][:6]}
    return {"n_sp": len(cap), "n_dong": len(ref), "khong_khop_dong_moc": kc_moc, "khong_khop_dong_muc": kc_muc,
            "khong_co_dong_moi_moc": km_moc, "khong_co_dong_moi_muc": km_muc, "sp_loai_vi_chi_mot_ben_khop": len(set(muc) ^ set(moc)),
            "chinh": chinh, "corpus": corpus, "tap_trung": tap_trung(chenh), "do_nhay": do_nhay,
            "mau_giam_nhat": [mau(i) for i in xep[:5]], "mau_tang_nhat": [mau(i) for i in xep[-5:][::-1]]}


def so_ca_muc(cong_cu, df_moc, df_muc, loai, cac_huong):
    """So cả mục (không tách dòng): chênh theo từng mục và chrF corpus, trung bình đều theo hướng."""
    K, D, _ = cong_cu
    a_tat, b_tat, nhom_tat, nhom_c, muc_c, moc_c = [], [], [], [], [], []
    for h in cac_huong:
        x = df_moc[(df_moc.loai == loai) & (df_moc.huong == h)].set_index("id")
        y = df_muc[(df_muc.loai == loai) & (df_muc.huong == h)].set_index("id").reindex(x.index)
        a_tat += y["chrf"].tolist()
        b_tat += x["chrf"].tolist()
        nhom_tat += [f"{h}|{loai}"] * len(x)
        m = D.tao_chrf(h)
        nhom_c.append((m, np.array([thong_ke(m, [p], [r]) for p, r in zip(y["hyp_cham"], y["_ref"])]),
                       np.array([thong_ke(m, [p], [r]) for p, r in zip(x["hyp_cham"], x["_ref"])])))
        muc_c.append(D.chrf_corpus(h, y["hyp_cham"], y["_ref"]))
        moc_c.append(D.chrf_corpus(h, x["hyp_cham"], x["_ref"]))
    corpus = bootstrap_corpus(nhom_c)
    corpus["muc"], corpus["moc"] = statistics.fmean(muc_c), statistics.fmean(moc_c)
    if abs(corpus["muc"] - corpus["moc"] - corpus["chenh"]) > 1e-6:
        raise RuntimeError("cộng thống kê không ra đúng chrF corpus của sacrebleu")
    return {"n": len(a_tat), "theo_muc": K.bootstrap_chenh(a_tat, b_tat, nhom_tat, so_lan=SO_LAN, seed=SEED), "corpus": corpus}


def kiem_ky_vong(so, ky_vong, ten):
    if abs(so - ky_vong) > DUNG_SAI_BAO_CAO:
        raise RuntimeError(f"không khớp báo cáo đã công bố: {ten} ra {so:.4f}, báo cáo {ky_vong}")


def xu_ly_nhanh(cong_cu, nhanh, pool, thu_muc_tl):
    K, D, pd = cong_cu
    bo = chuan_bi_mau(cong_cu, nhanh, pool["cum"], pool[nhanh])
    df_moc, th_moc = cham_file(cong_cu, nhanh, bo, DUONG_DAN_THAT[f"moc_{nhanh}"])
    df_muc, th_muc = cham_file(cong_cu, nhanh, bo, Path(thu_muc_tl) / MUC_THU[nhanh] / "sinh_ep2.jsonl")
    # Dừng nếu không tái tạo được số đã công bố: nghĩa là đang đọc sai dữ liệu hoặc sai cách chấm
    if abs(th_moc["diem_chinh"]["chrf_macro_huong"] - MOC_KY_VONG[nhanh]) > 1e-6:
        raise RuntimeError(f"mốc {nhanh} không khớp: {th_moc['diem_chinh']['chrf_macro_huong']}")
    for h in ("zh->vi", "vi->zh"):
        for ten, th in (("moc", th_moc), ("muc", th_muc)):
            so, n = KY_VONG_MOI[(nhanh, h)][ten]
            kiem_ky_vong(th["dong_da_thay_moi"][h]["moi"]["chrf_corpus"], so, f"{nhanh} {h} {ten}")
            if th["dong_da_thay_moi"][h]["moi"]["n_dong"] != n:
                raise RuntimeError(f"số dòng lệch báo cáo: {nhanh} {h} {ten}")
    thu_tu = bo["ids"]
    hang_moc, hang_muc = df_moc.to_dict("records"), df_muc.to_dict("records")
    r = {"zh_vi": so_dong_moi(cong_cu, hang_moc, hang_muc, thu_tu, "zh->vi"),
         "vi_zh": so_dong_moi(cong_cu, hang_moc, hang_muc, thu_tu, "vi->zh"),
         "title": so_ca_muc(cong_cu, df_moc, df_muc, "title", ("zh->vi", "vi->zh")),
         "title_zh_vi": so_ca_muc(cong_cu, df_moc, df_muc, "title", ("zh->vi",)),
         "mo_ta_zh_vi": so_ca_muc(cong_cu, df_moc, df_muc, "mo_ta", ("zh->vi",)),
         "sinh_lap": {"moc": int(df_moc["sinh_lap"].sum()), "muc": int(df_muc["sinh_lap"].sum())}}
    kiem_ky_vong(r["title"]["corpus"]["moc"], KY_VONG_TITLE[nhanh]["moc"], f"{nhanh} title moc")
    kiem_ky_vong(r["title"]["corpus"]["muc"], KY_VONG_TITLE[nhanh]["muc"], f"{nhanh} title muc")
    r["ket_luan"] = ket_luan(r["zh_vi"]["chinh"]["thap"])
    r["corpus_mau_thuan"] = mau_thuan(r["zh_vi"]["chinh"], r["zh_vi"]["corpus"])
    return r


# ---------------------------------------------------------------- báo cáo
def _k(d):
    return f"{d['chenh']:+.2f} [{d['thap']:+.2f}; {d['cao']:+.2f}]"


def viet_md(kq):
    d = ["# Kiểm tín hiệu mô tả dòng mới, mức ~11% so với 1 : 1 (fold 0)", "",
         "Bước 1, không train, không dịch lại, không dùng GPU. Đây là phần đọc số chrF, không phải đánh giá của người.",
         "Fold 0 là nơi tín hiệu được phát hiện, nên \"qua\" chỉ có nghĩa là đáng kiểm tiếp, không đủ để đổi tỉ lệ.", ""]
    for nh in NHANH:
        r = kq[nh]
        z = r["zh_vi"]
        t = z["tap_trung"]
        d += [f"## Bản {nh}", "", f"**Kết luận theo luật khóa: {r['ket_luan']}**" + (" (chrF corpus cho kết luận khác, xem bảng)" if r["corpus_mau_thuan"] else ""), "",
              "| Chỉ số | n | Chênh ~11% trừ 1 : 1 [khoảng 95%] |", "|---|---|---|",
              f"| **Dòng mới zh→vi, theo sản phẩm (quyết định)** | {z['n_sp']} sp, {z['n_dong']} dòng | **{_k(z['chinh'])}** |",
              f"| Dòng mới zh→vi, chrF corpus ({z['corpus']['moc']:.2f} → {z['corpus']['muc']:.2f}) | như trên | {_k(z['corpus'])} |",
              f"| Dòng mới vi→zh, theo sản phẩm | {r['vi_zh']['n_sp']} sp, {r['vi_zh']['n_dong']} dòng | {_k(r['vi_zh']['chinh'])} |",
              f"| Dòng mới vi→zh, chrF corpus ({r['vi_zh']['corpus']['moc']:.2f} → {r['vi_zh']['corpus']['muc']:.2f}) | như trên | {_k(r['vi_zh']['corpus'])} |",
              f"| Mô tả zh→vi toàn bộ, theo mục | {r['mo_ta_zh_vi']['n']} | {_k(r['mo_ta_zh_vi']['theo_muc'])} |",
              f"| Mô tả zh→vi toàn bộ, chrF corpus ({r['mo_ta_zh_vi']['corpus']['moc']:.2f} → {r['mo_ta_zh_vi']['corpus']['muc']:.2f}) | như trên | {_k(r['mo_ta_zh_vi']['corpus'])} |",
              f"| Title hai hướng, theo mục | {r['title']['n']} | {_k(r['title']['theo_muc'])} |",
              f"| Title hai hướng, chrF corpus ({r['title']['corpus']['moc']:.2f} → {r['title']['corpus']['muc']:.2f}) | như trên | {_k(r['title']['corpus'])} |",
              f"| Title zh→vi, theo mục | {r['title_zh_vi']['n']} | {_k(r['title_zh_vi']['theo_muc'])} |",
              f"| Title zh→vi, chrF corpus ({r['title_zh_vi']['corpus']['moc']:.2f} → {r['title_zh_vi']['corpus']['muc']:.2f}) | như trên | {_k(r['title_zh_vi']['corpus'])} |", "",
              f"Độ tập trung, dòng mới zh→vi: tăng {t['tang']}, giảm {t['giam']}, bằng {t['bang']} sản phẩm; trung bình {t['trung_binh']:+.2f}, trung vị {t['trung_vi']:+.2f}; "
              f"bỏ {SO_CAT} sản phẩm tăng nhiều nhất còn {t['tb_bo_tang_nhat']:+.2f}; cắt {SO_CAT} đầu và {SO_CAT} cuối còn {t['tb_cat_hai_dau']:+.2f}.", "",
              "Độ nhạy, dòng mới zh→vi (thêm sau khi đọc mẫu, không thay luật khóa):", "",
              "| Cách tính | Chênh [khoảng 95%] |", "|---|---|",
              f"| Không phân biệt hoa thường, theo sản phẩm | {_k(z['do_nhay']['khong_phan_biet_hoa'])} |",
              f"| Không phân biệt hoa thường, chrF corpus | {_k(z['do_nhay']['khong_phan_biet_hoa_corpus'])} |",
              f"| Bootstrap theo {z['do_nhay']['theo_cum']['so_cum']} cụm chung dòng | {_k(z['do_nhay']['theo_cum'])} |",
              f"| Không phân biệt hoa thường và theo cụm | {_k(z['do_nhay']['khong_phan_biet_hoa_theo_cum'])} |", "",
              f"Dòng mới mà hai mức dịch khác nhau: {z['do_nhay']['chi_khac_hoa']['dong_khac']}, trong đó chỉ khác chữ hoa {z['do_nhay']['chi_khac_hoa']['chi_khac_hoa']}.", "",
              f"Sản phẩm bị loại: không khớp dòng ở 1 : 1 {z['khong_khop_dong_moc']}, ở ~11% {z['khong_khop_dong_muc']}; chỉ một bên có dòng mới khớp {z['sp_loai_vi_chi_mot_ben_khop']}. "
              f"Sinh lặp trên 1.000 mẫu: 1 : 1 {r['sinh_lap']['moc']}, ~11% {r['sinh_lap']['muc']}.", ""]
    return "\n".join(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thu-muc-tl", default=str(THU_MUC_TL))
    p.add_argument("--ra", default=str(THU_MUC_TL.parent / "tong_hop"))
    a = p.parse_args()
    cong_cu = _nap_cong_cu()
    pool = {n: _doc_jsonl(DUONG_DAN_THAT[f"pool_{n}"]) for n in NHANH}
    kq = {n: xu_ly_nhanh(cong_cu, n, pool, a.thu_muc_tl) for n in NHANH}
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "kiem_dong_moi.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    md = viet_md(kq)
    (ra / "kiem_dong_moi.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
