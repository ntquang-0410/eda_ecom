"""SO CHẾ ĐỘ GOM (nhiều mục/lời gọi, một lượt) với kết quả HAI THỨ TỰ KHÔNG GOM (rubric v2) trên CÙNG các mục.

  python -I -X utf8 so_sanh_gom.py --hai-luot m2_v2_title_150,m2_v2_cum_ctx_150,m2_v2_pair_150 --gom b_title_150,b_cum_ctx_150,b_pair_150

Chỉ đọc file kết quả. Với mỗi mục có ở cả hai: lấy từ kết quả hai thứ tự đúng LƯỢT ứng với thứ tự ngẫu nhiên theo băm (chính là
kết quả nếu hỏi MỘT mục MỘT lời gọi), rồi so với kết quả gom. Câu hỏi: gom có làm judge kém đi không?
Tiêu chí đề xuất để chấp nhận gom: (1) kết quả gom trùng kết quả hai thứ tự ở các mục BỀN VỮNG >= 95%; (2) tỉ lệ khai tương đương
gần như không đổi; (3) tỉ lệ mục lỗi/thiếu thấp (< 2%); (4) token mỗi mục giảm rõ rệt."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so_cap_du_lieu as DL  # noqa: E402
import so_cap_rubric as SR  # noqa: E402
import so_sanh_quy_tac as SQ  # noqa: E402


def doc_het(ten_ds, ra):
    """Bản ghi ok cuối cùng của mỗi mục qua các đợt, và tập mục còn LỖI (chưa có bản ghi ok)."""
    cuoi, loi = {}, set()
    for ten in ten_ds:
        for dong in open(Path(ra) / f"ket_qua_{ten}.jsonl", encoding="utf-8"):
            d = json.loads(dong)
            if d.get("trang_thai") == "ok":
                cuoi[d["item_id"]] = d
                loi.discard(d["item_id"])
            elif d["item_id"] not in cuoi:
                loi.add(d["item_id"])
    return cuoi, loi


def so_sanh(hai, gom, so_loi=0):
    chung = [i for i in gom if i in hai]
    n = len(chung)
    r = {"so_muc_chung": n, "so_muc_loi_hoac_thieu": so_loi}
    if not n:
        return r
    don = {i: SQ.luot(hai[i], SR.thu_tu_bang_bam(i)) for i in chung}          # kết quả nếu hỏi MỘT mục MỘT lời gọi
    gm = {i: gom[i]["luot"][0] for i in chung}
    ben = [i for i in chung if hai[i]["ket_qua"] in ("gemma", "qwen")]
    r["muc_ben_vung_hai_thu_tu"] = len(ben)
    r["gom_trung_ket_qua_ben_vung"] = sum(gm[i]["chon"] == hai[i]["ket_qua"] for i in ben)
    r["ti_le_gom_trung_ben_vung"] = round(r["gom_trung_ket_qua_ben_vung"] / max(len(ben), 1), 3)
    r["gom_trung_mot_loi_goi_moi_muc"] = round(sum(gm[i]["chon"] == don[i]["chon"] for i in chung) / n, 3)
    r["ti_le_khai_tuong_duong"] = {"mot_muc_mot_loi_goi": round(sum(don[i]["tuong_duong"] for i in chung) / n, 3),
                                   "gom": round(sum(gm[i]["tuong_duong"] for i in chung) / n, 3)}
    kt = [i for i in chung if not gm[i]["tuong_duong"]]
    ben_set = set(ben)
    r["trong_cac_lan_gom_khong_khai_tuong_duong"] = {
        "so_muc": len(kt), "ben_vung": sum(i in ben_set and gm[i]["chon"] == hai[i]["ket_qua"] for i in kt),
        "lech_khong_nhan_ra": sum(hai[i]["ket_qua"] == "khong_nhat_quan" for i in kt)}
    r["chon_vi_tri_1"] = {"mot_muc_mot_loi_goi": round(sum(don[i]["vi_tri_chon"] == 1 for i in chung) / n, 3),
                          "gom": round(sum(gm[i]["vi_tri_chon"] == 1 for i in chung) / n, 3)}
    c = Counter(gm[i]["chon"] for i in chung)
    r["gemma_qwen"] = {"gom": [c["gemma"], c["qwen"]], "mot_muc_mot_loi_goi": [sum(don[i]["chon"] == "gemma" for i in chung),
                                                                               sum(don[i]["chon"] == "qwen" for i in chung)]}
    tv_g = sum(gom[i]["tokens_vao"] + gom[i]["tokens_ra"] for i in chung) / n
    tv_d = sum(hai[i]["tokens_vao"] + hai[i]["tokens_ra"] for i in chung) / n / 2        # hai lời gọi => nửa là một lời gọi một mục
    r["token_moi_muc"] = {"mot_muc_mot_loi_goi": round(tv_d), "gom": round(tv_g), "ti_so": round(tv_g / tv_d, 3) if tv_d else None}
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hai-luot", required=True)
    p.add_argument("--gom", required=True)
    p.add_argument("--ra", default=None)
    a = p.parse_args(argv)
    ra = Path(a.ra) if a.ra else DL.RA
    hai, _ = doc_het([t for t in a.hai_luot.split(",") if t], ra)
    gom, loi = doc_het([t for t in a.gom.split(",") if t], ra)
    kq = {"tat_ca": so_sanh(hai, gom, len(loi))}
    for ten, loc in (("dev", lambda i: SQ.nua(i) == 0), ("test", lambda i: SQ.nua(i) == 1)):
        kq[ten] = so_sanh({i: d for i, d in hai.items() if loc(i)}, {i: d for i, d in gom.items() if loc(i)})
    for loai, pre in (("title", ("title:",)), ("cum", ("pair:", "cumctx:")), ("pair", ("unit:",))):
        kq[f"loai_{loai}"] = so_sanh({i: d for i, d in hai.items() if i.startswith(pre)}, {i: d for i, d in gom.items() if i.startswith(pre)})
    print(json.dumps(kq, ensure_ascii=False, indent=1))
    (ra / "so_sanh_gom.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n(đã ghi {ra / 'so_sanh_gom.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
