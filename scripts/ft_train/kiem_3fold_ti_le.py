"""Áp tiêu chí 3 fold đã khóa cho bản cụm, ~11% so với 1 : 1, từ bản dịch đã có (0 GPU, không đọc test). Plan mục 0, Prompt 796.
Đổi tỉ lệ chỉ khi: chrF chung cao hơn ở cả 3 fold, chênh trung bình lớn hơn độ lệch chuẩn, không học lệch theo trung bình 3 fold (ket_luan_kfold);
và trên fold 2, khoảng 95% dòng mới mô tả zh→vi nằm hẳn trên 0, title hai hướng không tụt quá 1 điểm (đọc từ kiem_fold2.json).
Chạy:  python -I -X utf8 scripts/ft_train/kiem_3fold_ti_le.py"""
import json
import statistics
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tong_hop_ti_le import DUONG_DAN_THAT, KHOA_LECH, _doc_jsonl, _nap_cong_cu, chuan_bi_mau, gom_so, ket_luan_kfold, khong_hoc_lech  # noqa: E402
from kiem_dong_moi import cham_file  # noqa: E402

FT = GOC / "data" / "processed" / "ft_run"
F1 = FT / "kaggle_b8f1" / "phien_1_20261106_102741"
F2 = FT / "kaggle_b8f2" / "phien_1_20261106_165058"
DS = {0: (DUONG_DAN_THAT["moc_cum"], FT / "kaggle_tl" / "phien_1_20261105_231941" / "tl" / "cum_f0_p11" / "sinh_ep2.jsonl"),
      1: (F1 / "b78" / "cum_f1" / "sinh_ep2.jsonl", F1 / "b78" / "cum_f1_p11" / "sinh_ep2.jsonl"),
      2: (F1 / "b78" / "cum_f2" / "sinh_ep2.jsonl", F2 / "b8f2" / "cum_f2_p11" / "sinh_ep2.jsonl")}


def so_cua(th):
    return gom_so({"so_mau": 1000, "sinh_lap": 0, "chrf": th["diem_chinh"], "theo_huong": th["theo_huong"], "theo_nhom": th["theo_nhom"], "dong_da_thay_moi": th.get("dong_da_thay_moi")})


def main():
    cong_cu = _nap_cong_cu()
    K, _, _ = cong_cu
    pool = _doc_jsonl(DUONG_DAN_THAT["pool_cum"])
    hang = {}
    for fold, (moc, muc) in DS.items():
        bo = chuan_bi_mau(cong_cu, "cum", pool, pool, fold=fold)
        df_a, th_a = cham_file(cong_cu, "cum", bo, moc)
        df_b, th_b = cham_file(cong_cu, "cum", bo, muc)
        ids = bo["ids"]
        a = df_b.set_index("id")["chrf"].reindex(ids).tolist()
        b = df_a.set_index("id")["chrf"].reindex(ids).tolist()
        cap = K.bootstrap_chenh(a, b, [f"{m['huong']}|{m['loai']}" for m in bo["mau"]])
        sa, sb = so_cua(th_a), so_cua(th_b)
        hang[fold] = {"moc": sa, "muc": sb, "chenh_corpus": sb["chrf"] - sa["chrf"], "theo_cau": cap, "khong_hoc_lech": khong_hoc_lech(sa, sb)}
    chenh = [hang[f]["chenh_corpus"] for f in DS]
    tb = lambda k, nhom: statistics.fmean(hang[f][nhom][k] for f in DS)  # noqa: E731
    moc_tb, muc_tb = ({k: tb(k, n) for k in KHOA_LECH} for n in ("moc", "muc"))
    lech = khong_hoc_lech(moc_tb, muc_tb)
    kl = ket_luan_kfold(chenh, lech)
    f2 = json.loads((F2 / "tong_hop" / "kiem_fold2.json").read_text(encoding="utf-8"))
    dieu_kien_them = f2.get("ket_luan") or f2.get("quyet")
    ket = {"chenh_chrf_chung": chenh, "trung_binh": statistics.fmean(chenh), "do_lech_chuan": statistics.stdev(chenh), "khong_hoc_lech_tb": lech,
           "ket_luan_kfold": kl, "dieu_kien_them_fold2": dieu_kien_them, "theo_fold": hang}
    ra = F2 / "tong_hop" / "kiem_3fold.json"
    ra.write_text(json.dumps(ket, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for f in DS:
        h = hang[f]
        c = h["theo_cau"]
        print(f"fold {f}: chrF chung {h['moc']['chrf']:.2f} -> {h['muc']['chrf']:.2f} ({h['chenh_corpus']:+.2f}); theo cau {c['chenh']:+.2f} [{c['thap']:+.2f}; {c['cao']:+.2f}]")
        print("   title %.2f -> %.2f; mo ta %.2f -> %.2f; zh->vi %.2f -> %.2f; vi->zh %.2f -> %.2f; lech %s" % (h["moc"]["title"], h["muc"]["title"], h["moc"]["mo_ta"], h["muc"]["mo_ta"], h["moc"]["zh_vi"], h["muc"]["zh_vi"], h["moc"]["vi_zh"], h["muc"]["vi_zh"], h["khong_hoc_lech"]["dat"]))
    print("chenh", [round(x, 3) for x in chenh], "tb", round(ket["trung_binh"], 3), "sd", round(ket["do_lech_chuan"], 3))
    print("lech tb:", lech["dat"], {k: round(v, 3) for k, v in lech["chenh"].items()})
    print("ket luan kfold:", kl)
    print("dieu kien them fold 2:", dieu_kien_them)
    print("ghi:", ra)


if __name__ == "__main__":
    main()
