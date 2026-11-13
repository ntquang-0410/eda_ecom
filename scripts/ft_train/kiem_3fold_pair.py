"""Luật 3 fold cho bản PAIR, ~11% so với 1 : 1, từ bản dịch đã có (0 GPU, không đọc test). Nhật Anh duyệt 07/11/2026 (Prompt 804).
KHÓA TRƯỚC khi có số fold 2 (đã ghi nguyên văn trong WALKTHROUGH, Prompt 803 và 804). Không đổi sau khi thấy số.

Pair đổi sang ~11% (p_title 0,1086) khi ĐÚNG CẢ BỐN điều:
 (a) dòng mới mô tả zh->vi, theo sản phẩm, có chênh DƯƠNG ở cả 3 fold, và cận dưới khoảng 95% ở fold 2 > 0;
 (b) chrF chung (corpus, trung bình hai hướng) trung bình 3 fold của (~11% trừ 1 : 1) >= 0, và không fold nào tụt quá 1 điểm;
 (c) title hai hướng, chrF corpus, chênh trung bình 3 fold không tụt quá 1 điểm;
 (d) title zh->vi, theo mục, chênh trung bình 3 fold không tụt quá 1 điểm.
Không đạt thì giữ 1 : 1 (kết quả hợp lệ). Thiếu bản dịch fold 2 thì "chua_du_du_lieu", không kết luận.
Khác luật cụm (kiem_3fold_ti_le.py): KHÔNG đòi chrF chung cao hơn ở cả 3 fold, vì fold 0 pair đã -0,10; thay bằng "không tụt".
Điểm cần nói thẳng: (c), (d) tính trung bình nên một fold tụt quá 1 điểm (fold 0: title hai hướng -0,96, title zh->vi -1,19) có thể được fold khác bù.
Dòng mới zh->vi đã được xem ở fold 0 (nơi phát hiện tín hiệu), nên fold 1 và 2 mới là kiểm độc lập; kết luận ghi rõ điều này.

Chạy:  python -I -X utf8 scripts/ft_train/kiem_3fold_pair.py [--ra <thu_muc>]
KHÔNG đọc test_khoa, split.parquet, ds_test.txt: chỉ đọc pool_*.jsonl và bản dịch của từng fold."""
import argparse
import json
import statistics
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kiem_fold_ti_le import chay  # noqa: E402
from tong_hop_ti_le import NGUONG_LECH  # noqa: E402

FT = GOC / "data" / "processed" / "ft_run"
T2 = FT / "kaggle_tl2" / "phien_1_20261106_210034" / "tl2"
DS = {0: (FT / "kaggle_b78" / "phien_1_20261105_181337" / "b78" / "pair_f0" / "sinh_ep2.jsonl",
          FT / "kaggle_tl" / "phien_1_20261105_231941" / "tl" / "pair_f0_p11" / "sinh_ep2.jsonl"),
      1: (T2 / "pair_f1" / "sinh_ep2.jsonl", T2 / "pair_f1_p11" / "sinh_ep2.jsonl")}


def tim_fold2():
    a = sorted(FT.glob("kaggle_tl3/phien_*/tl3/pair_f2/sinh_ep2.jsonl"))
    b = sorted(FT.glob("kaggle_tl3/phien_*/tl3/pair_f2_p11/sinh_ep2.jsonl"))
    return (a[-1], b[-1]) if a and b else None


def hop(x):
    return x is not None and x == x


def quyet_3fold(r):
    """r: {fold: kết quả của kiem_fold_ti_le.chay}. Trả (kết luận, danh sách điều kiện kèm lý do). Thuần, test được không cần dữ liệu thật."""
    if set(r) != {0, 1, 2}:
        return "chua_du_du_lieu", [f"thiếu fold {sorted({0, 1, 2} - set(r))}"]
    dm = {f: r[f]["zh_vi"]["chinh"]["chenh"] for f in r}
    thap2 = r[2]["zh_vi"]["chinh"]["thap"]
    chung = {f: r[f]["chrf_chung"]["muc"] - r[f]["chrf_chung"]["moc"] for f in r}
    tit = {f: r[f]["title"]["corpus"]["chenh"] for f in r}
    tzv = {f: r[f]["title_zh_vi"]["theo_muc"]["chenh"] for f in r}
    cac = [dm, chung, tit, tzv]
    if not all(hop(v) for d in cac for v in d.values()) or not hop(thap2):
        return "chua_du_du_lieu", ["có chỉ số không hữu hạn"]
    tb = lambda d: statistics.fmean(d.values())  # noqa: E731
    dk = {
        "a": (all(v > 0 for v in dm.values()) and thap2 > 0,
              f"dòng mới zh->vi theo fold {[round(dm[f], 2) for f in sorted(dm)]}, cận dưới fold 2 {thap2:+.2f}"),
        "b": (tb(chung) >= 0 and all(v >= -NGUONG_LECH for v in chung.values()),
              f"chrF chung theo fold {[round(chung[f], 2) for f in sorted(chung)]}, trung bình {tb(chung):+.2f}"),
        "c": (tb(tit) >= -NGUONG_LECH, f"title hai hướng theo fold {[round(tit[f], 2) for f in sorted(tit)]}, trung bình {tb(tit):+.2f}"),
        "d": (tb(tzv) >= -NGUONG_LECH, f"title zh->vi theo fold {[round(tzv[f], 2) for f in sorted(tzv)]}, trung bình {tb(tzv):+.2f}"),
    }
    kl = "dat" if all(v[0] for v in dk.values()) else "khong_dat"
    return kl, [f"({k}) {'ĐẠT' if ok else 'KHÔNG ĐẠT'}: {ly}" for k, (ok, ly) in dk.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", default=str(FT / "kaggle_tl3" / "cham"))
    a = ap.parse_args()
    ds = dict(DS)
    f2 = tim_fold2()
    if f2:
        ds[2] = f2
    r = {}
    for fold, (moc, muc) in ds.items():
        r[fold] = chay(fold, moc, muc, nhanh="pair", luat="pair")
        c = r[fold]["zh_vi"]["chinh"]
        print(f"fold {fold}: dòng mới zh->vi {c['chenh']:+.2f} [{c['thap']:+.2f}; {c['cao']:+.2f}]; chrF chung {r[fold]['chrf_chung']['moc']:.2f} -> {r[fold]['chrf_chung']['muc']:.2f}; "
              f"title {r[fold]['title']['corpus']['chenh']:+.2f}; title zh->vi {r[fold]['title_zh_vi']['theo_muc']['chenh']:+.2f}; luật 1 fold: {r[fold]['ket_luan']}", flush=True)
    kl, ly = quyet_3fold(r)
    print("KẾT LUẬN 3 FOLD (pair):", kl)
    for d in ly:
        print("  ", d)
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "kiem_3fold_pair.json").write_text(json.dumps({"ket_luan": kl, "ly_do": ly, "theo_fold": r}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = ["# Luật 3 fold cho bản pair, ~11% so với 1 : 1", "", "Đây là phần đọc số chrF, không phải đánh giá của người. Dòng mới zh→vi đã được xem ở fold 0 nên chỉ fold 1, 2 là kiểm độc lập.", "",
          f"**Kết luận: {kl}**", ""] + [f"- {d}" for d in ly]
    (ra / "kiem_3fold_pair.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("ghi:", ra / "kiem_3fold_pair.json")


if __name__ == "__main__":
    main()
