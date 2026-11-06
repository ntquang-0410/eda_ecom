"""Áp luật khóa so hai lần chạy cùng fold, cùng 1.000 mẫu chấm (Prompt 794, mở rộng ở Prompt 800). Chạy trên máy, không GPU.

Chạy:  python -I -X utf8 scripts/ft_train/kiem_fold_ti_le.py --fold 1 --moc <.../sinh_ep2.jsonl> --muc <.../sinh_ep2.jsonl> --ra <thu_muc> [--nhanh cum|pair] [--luat ti_le|lr|pair] [--ten kiem_x] [--nhan-moc "1 : 1"] [--nhan-muc "~11%"]
Ghi <thu_muc>/<ten>.md và <ten>.json (ten mặc định kiem_fold<F>).

Ba luật khóa (đều khóa trước khi có số mới; không đổi sau khi thấy số):
- ti_le (Prompt 794-796, bản cụm): "dat" khi cận dưới khoảng 95% của chênh dòng mới mô tả zh->vi theo sản phẩm > 0, và title (trung bình chrF corpus hai hướng) không tụt quá 1 điểm.
  Không đạt thì chốt 1 : 1. Đạt chỉ có nghĩa là chạy tiếp fold còn lại: đổi tỉ lệ chỉ khi đạt tiêu chí 3 fold (tong_hop_ti_le.ket_luan_kfold).
- lr (Prompt 800, bước kiểm lr 1e-4 ở mức ~11% bản cụm, fold 0, so với lr 2e-4 cùng mức): "dat" (đổi lr) khi cận dưới khoảng 95% của chênh chrF chung theo câu > 0,
  VÀ không học lệch (title, mô tả, zh->vi không tụt quá 1 điểm chrF corpus) VÀ sinh lặp của lr mới không quá 10 trên 1.000 mẫu [MẶC ĐỊNH]. Khoảng chứa 0 là hòa và giữ lr 2e-4.
- pair (Prompt 800, cổng fold 1 của bản pair, ~11% so với 1 : 1): "dat" khi (a) cận dưới khoảng 95% của chênh dòng mới mô tả zh->vi theo sản phẩm > 0, (b) title hai hướng chrF corpus
  không tụt quá 1 điểm, và (c) title zh->vi theo mục (chênh) không tụt quá 1 điểm. "dat" chỉ có nghĩa là hỏi Nhật Anh có chạy fold 2 không, không phải đổi tỉ lệ pair.
Kiểm công cụ: --fold 0 với bản dịch fold 0 phải ra đúng số đã công bố (cụm +1.43 [+0.54; +2.32]; pair +1.36 [+0.57; +2.19]).
KHÔNG đọc test_khoa, split.parquet, ds_test.txt: chỉ đọc pool_*.jsonl và bản dịch của fold đó."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kiem_dong_moi import _k, cham_file, so_ca_muc, so_dong_moi  # noqa: E402
from tong_hop_ti_le import DUONG_DAN_THAT, KHOA_LECH, NGUONG_LECH, _doc_jsonl, _nap_cong_cu, chuan_bi_mau, gom_so, khong_hoc_lech  # noqa: E402

DAT, KHONG_DAT, CHUA_DU = "dat", "khong_dat", "chua_du_du_lieu"
SINH_LAP_TOI_DA = 10      # trên 1.000 mẫu, luật lr [MẶC ĐỊNH]


def huu_han(x):
    return x is not None and math.isfinite(x)


def quyet(thap, title_moc, title_muc):
    """Luật khóa ti_le. Có bằng chứng không đạt thì khong_dat dù thiếu số khác; thiếu số mà chưa có bằng chứng thì chưa kết luận. Trả (kết luận, lý do)."""
    ly_do, thieu = [], []
    if huu_han(thap):
        if not thap > 0:
            ly_do.append(f"cận dưới khoảng 95% của chênh là {thap:+.2f}, không dương")
    else:
        thieu.append("khoảng tin cậy")
    if huu_han(title_moc) and huu_han(title_muc):
        tut = title_moc - title_muc
        if tut > NGUONG_LECH:
            ly_do.append(f"title tụt {tut:.2f} điểm, quá ngưỡng {NGUONG_LECH}")
    else:
        thieu.append("điểm title")
    if ly_do:
        return KHONG_DAT, ly_do
    if thieu:
        return CHUA_DU, ["thiếu " + ", ".join(thieu)]
    return DAT, [f"cận dưới {thap:+.2f} > 0 và title không tụt quá {NGUONG_LECH} điểm"]


def quyet_pair(r):
    """Luật khóa pair: luật ti_le cộng điều kiện title zh->vi theo mục không tụt quá NGUONG_LECH."""
    kl, ly = quyet(r["zh_vi"]["chinh"]["thap"], r["title"]["corpus"]["moc"], r["title"]["corpus"]["muc"])
    c = r["title_zh_vi"]["theo_muc"]["chenh"]
    if huu_han(c) and c < -NGUONG_LECH:
        return KHONG_DAT, (ly if kl == KHONG_DAT else []) + [f"title zh->vi theo mục tụt {-c:.2f} điểm, quá ngưỡng {NGUONG_LECH}"]
    if not huu_han(c):
        return (KHONG_DAT if kl == KHONG_DAT else CHUA_DU), ly + ["thiếu chênh title zh->vi theo mục"]
    if kl == DAT:
        return DAT, ly + [f"title zh->vi theo mục {c:+.2f}, không tụt quá {NGUONG_LECH} điểm"]
    return kl, ly


def quyet_lr(r):
    """Luật khóa lr: đổi lr khi cận dưới chênh chrF chung theo câu > 0, không học lệch, sinh lặp không quá SINH_LAP_TOI_DA."""
    thap = r["theo_cau"]["thap"]
    ly_do, thieu = [], []
    if huu_han(thap):
        if not thap > 0:
            ly_do.append(f"cận dưới khoảng 95% của chênh chrF chung theo câu là {thap:+.2f}, không dương (hòa hoặc kém, giữ lr cũ)")
    else:
        thieu.append("khoảng tin cậy")
    lech = r["khong_hoc_lech"]
    if lech["dat"] is False:
        ly_do.append("học lệch: " + "; ".join(lech["ly_do"]))
    elif lech["dat"] is None:
        thieu.append("điểm không học lệch")
    lap = r["sinh_lap"]["muc"]
    if lap > SINH_LAP_TOI_DA:
        ly_do.append(f"sinh lặp {lap} trên 1.000 mẫu, quá {SINH_LAP_TOI_DA}")
    if ly_do:
        return KHONG_DAT, ly_do
    if thieu:
        return CHUA_DU, ["thiếu " + ", ".join(thieu)]
    return DAT, [f"cận dưới chrF chung theo câu {thap:+.2f} > 0, không học lệch, sinh lặp {lap} trên 1.000"]


def so_cua(th):
    return gom_so({"so_mau": 1000, "sinh_lap": 0, "chrf": th["diem_chinh"], "theo_huong": th["theo_huong"], "theo_nhom": th["theo_nhom"], "dong_da_thay_moi": th.get("dong_da_thay_moi")})


def chay(fold, duong_moc, duong_muc, nhanh="cum", luat="ti_le"):
    cong_cu = _nap_cong_cu()
    K = cong_cu[0]
    pool = {"cum": _doc_jsonl(DUONG_DAN_THAT["pool_cum"])}
    pool[nhanh] = pool["cum"] if nhanh == "cum" else _doc_jsonl(DUONG_DAN_THAT[f"pool_{nhanh}"])
    bo = chuan_bi_mau(cong_cu, nhanh, pool["cum"], pool[nhanh], fold=fold)
    df_moc, th_moc = cham_file(cong_cu, nhanh, bo, duong_moc)
    df_muc, th_muc = cham_file(cong_cu, nhanh, bo, duong_muc)
    hang_moc, hang_muc = df_moc.to_dict("records"), df_muc.to_dict("records")
    sa, sb = so_cua(th_moc), so_cua(th_muc)
    ids = bo["ids"]
    a = df_muc.set_index("id")["chrf"].reindex(ids).tolist()
    b = df_moc.set_index("id")["chrf"].reindex(ids).tolist()
    r = {"fold": fold, "nhanh": nhanh, "luat": luat, "chrf_chung": {"moc": th_moc["diem_chinh"]["chrf_macro_huong"], "muc": th_muc["diem_chinh"]["chrf_macro_huong"]},
         "theo_cau": K.bootstrap_chenh(a, b, [f"{m['huong']}|{m['loai']}" for m in bo["mau"]]),
         "khong_hoc_lech": khong_hoc_lech(sa, sb), "so_goc": {"moc": sa, "muc": sb},
         "zh_vi": so_dong_moi(cong_cu, hang_moc, hang_muc, ids, "zh->vi"),
         "vi_zh": so_dong_moi(cong_cu, hang_moc, hang_muc, ids, "vi->zh"),
         "title": so_ca_muc(cong_cu, df_moc, df_muc, "title", ("zh->vi", "vi->zh")),
         "title_zh_vi": so_ca_muc(cong_cu, df_moc, df_muc, "title", ("zh->vi",)),
         "mo_ta_zh_vi": so_ca_muc(cong_cu, df_moc, df_muc, "mo_ta", ("zh->vi",)),
         "sinh_lap": {"moc": int(df_moc["sinh_lap"].sum()), "muc": int(df_muc["sinh_lap"].sum())}}
    if luat == "lr":
        r["ket_luan"], r["ly_do"] = quyet_lr(r)
    elif luat == "pair":
        r["ket_luan"], r["ly_do"] = quyet_pair(r)
    else:
        r["ket_luan"], r["ly_do"] = quyet(r["zh_vi"]["chinh"]["thap"], r["title"]["corpus"]["moc"], r["title"]["corpus"]["muc"])
    return r


def viet_md(r, nhan_moc="1 : 1", nhan_muc="~11%"):
    z, t = r["zh_vi"], r["zh_vi"]["tap_trung"]
    c, lech = r["theo_cau"], r["khong_hoc_lech"]
    d = [f"# {nhan_muc} so với {nhan_moc}, bản {r['nhanh']}, fold {r['fold']} (luật {r['luat']})", "", "Đây là phần đọc số chrF, không phải đánh giá của người.", "",
         f"**Kết luận theo luật khóa: {r['ket_luan']}.** " + "; ".join(r["ly_do"]), "",
         f"chrF chung: {nhan_moc} {r['chrf_chung']['moc']:.2f}, {nhan_muc} {r['chrf_chung']['muc']:.2f} (chênh corpus {r['chrf_chung']['muc'] - r['chrf_chung']['moc']:+.2f}; theo câu {c['chenh']:+.2f} [{c['thap']:+.2f}; {c['cao']:+.2f}]). "
         f"Sinh lặp trên 1.000 mẫu: {r['sinh_lap']['moc']} và {r['sinh_lap']['muc']}.", "",
         "Không học lệch (chrF corpus, mức trừ mốc): " + ", ".join(f"{k} {lech['chenh'].get(k):+.2f}" for k in KHOA_LECH if lech["chenh"].get(k) is not None) + f"; đạt = {lech['dat']}.", "",
         f"| Chỉ số | n | Chênh {nhan_muc} trừ {nhan_moc} [khoảng 95%] |", "|---|---|---|",
         f"| **Dòng mới zh→vi, theo sản phẩm** | {z['n_sp']} sp, {z['n_dong']} dòng | **{_k(z['chinh'])}** |",
         f"| Dòng mới zh→vi, chrF corpus ({z['corpus']['moc']:.2f} → {z['corpus']['muc']:.2f}) | như trên | {_k(z['corpus'])} |",
         f"| Không phân biệt hoa thường, theo sản phẩm | như trên | {_k(z['do_nhay']['khong_phan_biet_hoa'])} |",
         f"| Bootstrap theo {z['do_nhay']['theo_cum']['so_cum']} cụm chung dòng | như trên | {_k(z['do_nhay']['theo_cum'])} |",
         f"| Dòng mới vi→zh, theo sản phẩm | {r['vi_zh']['n_sp']} sp, {r['vi_zh']['n_dong']} dòng | {_k(r['vi_zh']['chinh'])} |",
         f"| Mô tả zh→vi toàn bộ, theo mục | {r['mo_ta_zh_vi']['n']} | {_k(r['mo_ta_zh_vi']['theo_muc'])} |",
         f"| **Title hai hướng, chrF corpus ({r['title']['corpus']['moc']:.2f} → {r['title']['corpus']['muc']:.2f})** | {r['title']['n']} | **{_k(r['title']['corpus'])}** |",
         f"| Title zh→vi, theo mục | {r['title_zh_vi']['n']} | {_k(r['title_zh_vi']['theo_muc'])} |", "",
         f"Độ tập trung, dòng mới zh→vi: tăng {t['tang']}, giảm {t['giam']}, bằng {t['bang']} sản phẩm; trung vị {t['trung_vi']:+.2f}; bỏ 5 sản phẩm tăng nhiều nhất còn {t['tb_bo_tang_nhat']:+.2f}.",
         f"Sản phẩm không khớp dòng: {nhan_moc} {z['khong_khop_dong_moc']}, {nhan_muc} {z['khong_khop_dong_muc']}; dòng khác nhau {z['do_nhay']['chi_khac_hoa']['dong_khac']}, chỉ khác chữ hoa {z['do_nhay']['chi_khac_hoa']['chi_khac_hoa']}.", ""]
    return "\n".join(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--moc", required=True)
    p.add_argument("--muc", required=True)
    p.add_argument("--ra", required=True)
    p.add_argument("--nhanh", choices=("cum", "pair"), default="cum")
    p.add_argument("--luat", choices=("ti_le", "lr", "pair"), default="ti_le")
    p.add_argument("--ten")
    p.add_argument("--nhan-moc", default="1 : 1")
    p.add_argument("--nhan-muc", default="~11%")
    a = p.parse_args()
    r = chay(a.fold, a.moc, a.muc, a.nhanh, a.luat)
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    ten = a.ten or f"kiem_fold{a.fold}"
    (ra / f"{ten}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = viet_md(r, a.nhan_moc, a.nhan_muc)
    (ra / f"{ten}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
