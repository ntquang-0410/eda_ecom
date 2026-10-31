"""ĐO BỘ QUY TẮC v2 so với lời nhắc v1 trên CÙNG các mục (đều chạy ĐỦ HAI THỨ TỰ), và MÔ PHỎNG chế độ MỘT LƯỢT từ kết quả hai thứ tự.

  python -I -X utf8 so_sanh_quy_tac.py --v1 title_100,cum_100,pair_100 --v2 qt_title_100,qt_cum_100,qt_pair_100

Chỉ đọc file kết quả (không gọi API); ghi so_sanh_quy_tac.json. Chia mục thành tập PHÁT TRIỂN (dev: quy tắc v2 soạn từ các mục
không nhất quán của nửa này) và tập KIỂM (test: nửa còn lại) theo băm item_id; số trên tập kiểm là số đáng tin hơn.
Câu hỏi chính: với v2, hai thứ tự còn lệch nhau bao nhiêu (tỉ lệ 'không nhất quán'), trong đó bao nhiêu là lệch CÓ khai báo
equivalent (hai bản tương đương, tung đồng xu công bằng) và bao nhiêu là lệch KHÔNG khai báo (judge thật sự không ổn định);
nếu chỉ hỏi một lượt (thứ tự ngẫu nhiên theo băm) thì kết quả trùng kết quả hai thứ tự ở bao nhiêu mục đã quyết."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so_cap_du_lieu as DL  # noqa: E402
import so_cap_rubric as SR  # noqa: E402


def nua(item_id):
    """0 = tập phát triển (dev), 1 = tập kiểm (test). Cùng công thức với lúc đọc mục để soạn quy tắc."""
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 2


def doc(ten_ds, ra=None):
    out = {}
    for ten in ten_ds:
        for dong in open(Path(ra or DL.RA) / f"ket_qua_{ten}.jsonl", encoding="utf-8"):
            d = json.loads(dong)
            if d.get("trang_thai") == "ok":
                out[d["item_id"]] = d
    return out


def luot(d, thu_tu):
    return next(l for l in d["luot"] if l["thu_tu"] == thu_tu)


def thong_ke(rows):
    """rows: bản ghi hai lượt (v1 hoặc v2)."""
    n = len(rows)
    r = {"so_muc": n}
    if not n:
        return r
    c = Counter(d["ket_qua"] for d in rows)
    r["ket_qua"] = {k: c.get(k, 0) for k in ("gemma", "qwen", "hoa_that", "khong_nhat_quan")}
    r["ti_le_khong_nhat_quan"] = round(c.get("khong_nhat_quan", 0) / n, 3)
    r["ti_le_co_nguoi_thang"] = round((c.get("gemma", 0) + c.get("qwen", 0)) / n, 3)
    luot_ko_hoa = [l for d in rows for l in d["luot"] if l["vi_tri_chon"] in (1, 2)]
    r["chon_vi_tri_1"] = round(sum(l["vi_tri_chon"] == 1 for l in luot_ko_hoa) / max(len(luot_ko_hoa), 1), 3)
    gq = [luot(d, "gq") for d in rows]
    qg = [luot(d, "qg") for d in rows]
    r["gemma_khi_dung_truoc"] = round(sum(l["chon"] == "gemma" for l in gq) / n, 3)
    r["gemma_khi_dung_sau"] = round(sum(l["chon"] == "gemma" for l in qg) / n, 3)
    if "tuong_duong" in rows[0]["luot"][0]:                                      # rubric v2
        kn = [d for d in rows if d["ket_qua"] == "khong_nhat_quan"]
        hai_co = sum(luot(d, "gq")["tuong_duong"] and luot(d, "qg")["tuong_duong"] for d in kn)
        mot_co = sum(luot(d, "gq")["tuong_duong"] != luot(d, "qg")["tuong_duong"] for d in kn)
        r["khong_nhat_quan_theo_khai_bao"] = {"hai_luot_deu_khai_bao_tuong_duong": hai_co,
                                              "mot_luot_khai_bao_tuong_duong": mot_co,
                                              "khong_khai_bao_tuong_duong_nao": len(kn) - hai_co - mot_co}
        r["ti_le_luot_khai_bao_tuong_duong"] = round(sum(l["tuong_duong"] for d in rows for l in d["luot"]) / (2 * n), 3)
        r["ti_le_khong_nhat_quan_that_su"] = round((len(kn) - hai_co) / n, 3)    # lệch mà không phải cả hai lượt đều khai tương đương
    return r


def mo_phong_mot_luot(rows):
    """Từ kết quả hai thứ tự của v2, lấy đúng lượt ứng với thứ tự ngẫu nhiên theo băm: đó là kết quả nếu chỉ hỏi MỘT lần."""
    n = len(rows)
    r = {"so_muc": n}
    if not n:
        return r
    chon = [(d, luot(d, SR.thu_tu_bang_bam(d["item_id"]))) for d in rows]
    r["gemma"] = sum(l["chon"] == "gemma" for _, l in chon)
    r["qwen"] = sum(l["chon"] == "qwen" for _, l in chon)
    r["ti_le_khai_bao_tuong_duong"] = round(sum(l["tuong_duong"] for _, l in chon) / n, 3)
    r["chon_vi_tri_1"] = round(sum(l["vi_tri_chon"] == 1 for _, l in chon) / n, 3)
    dq = [(d, l) for d, l in chon if d["ket_qua"] in ("gemma", "qwen")]
    r["muc_hai_thu_tu_da_dong_y"] = len(dq)
    r["trong_do_mot_luot_trung"] = sum(l["chon"] == d["ket_qua"] for d, l in dq)
    kn = [(d, l) for d, l in chon if d["ket_qua"] == "khong_nhat_quan"]
    r["muc_hai_thu_tu_lech"] = len(kn)
    r["trong_do_luot_duoc_chon_khai_bao_tuong_duong"] = sum(l["tuong_duong"] for _, l in kn)
    r["muc_lech_ma_luot_duoc_chon_khong_khai_bao"] = len(kn) - r["trong_do_luot_duoc_chon_khai_bao_tuong_duong"]
    return r


def so_v1_v2(v1, v2):
    chung = [i for i in v2 if i in v1]
    r = {"so_muc_chung": len(chung)}
    quyet = ("gemma", "qwen")
    ca_hai = [i for i in chung if v1[i]["ket_qua"] in quyet and v2[i]["ket_qua"] in quyet]
    r["ca_hai_deu_co_nguoi_thang"] = len(ca_hai)
    r["trong_do_cung_ket_qua"] = sum(v1[i]["ket_qua"] == v2[i]["ket_qua"] for i in ca_hai)
    r["v1_khong_nhat_quan_nay_co_nguoi_thang_o_v2"] = sum(v1[i]["ket_qua"] == "khong_nhat_quan" and v2[i]["ket_qua"] in quyet for i in chung)
    r["v1_co_nguoi_thang_nay_khong_nhat_quan_o_v2"] = sum(v1[i]["ket_qua"] in quyet and v2[i]["ket_qua"] == "khong_nhat_quan" for i in chung)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", required=True, help="các đợt v1 (hai thứ tự), cách nhau bằng dấu phẩy")
    p.add_argument("--v2", required=True, help="các đợt v2 (hai thứ tự) trên CÙNG mục")
    p.add_argument("--ra", default=None)
    a = p.parse_args(argv)
    ra = Path(a.ra) if a.ra else DL.RA
    v1 = doc([t for t in a.v1.split(",") if t], ra)
    v2 = doc([t for t in a.v2.split(",") if t], ra)
    chung = [i for i in v2 if i in v1]
    kq = {"so_muc_v1": len(v1), "so_muc_v2": len(v2), "so_muc_chung": len(chung)}
    for ten, loc in (("tat_ca", lambda i: True), ("dev", lambda i: nua(i) == 0), ("test", lambda i: nua(i) == 1)):
        ids = [i for i in chung if loc(i)]
        kq[ten] = {"v1": thong_ke([v1[i] for i in ids]), "v2": thong_ke([v2[i] for i in ids]),
                   "mot_luot_mo_phong_tu_v2": mo_phong_mot_luot([v2[i] for i in ids]),
                   "v1_so_voi_v2": so_v1_v2({i: v1[i] for i in ids}, {i: v2[i] for i in ids})}
    for loai in ("title", "cum", "pair"):
        pre = {"title": ("title:",), "cum": ("pair:", "cumctx:"), "pair": ("unit:",)}[loai]
        ids = [i for i in chung if i.startswith(pre)]
        kq[f"loai_{loai}"] = {"v1": thong_ke([v1[i] for i in ids]), "v2": thong_ke([v2[i] for i in ids])}
    print(json.dumps(kq, ensure_ascii=False, indent=1))
    # các mục v2 còn lệch mà KHÔNG khai báo tương đương, để đọc
    muc = {}
    import pandas as pd
    for f in ("muc_title", "muc_cum", "muc_cum_ctx", "muc_pair"):
        x = pd.read_parquet(DL.RA / f"{f}.parquet", columns=["item_id", "nguon", "cand_gemma", "cand_qwen"])
        muc.update({r.item_id: r for r in x.itertuples()})
    doc_ra = []
    for i in chung:
        d = v2[i]
        if d["ket_qua"] == "khong_nhat_quan" and not (luot(d, "gq")["tuong_duong"] and luot(d, "qg")["tuong_duong"]) and i in muc:
            m = muc[i]
            doc_ra.append({"item_id": i, "nua": "dev" if nua(i) == 0 else "test", "nguon": m.nguon, "gemma": m.cand_gemma, "qwen": m.cand_qwen,
                           "luot": [{k: l.get(k) for k in ("thu_tu", "chon", "tuong_duong", "so_loi_1", "so_loi_2", "tieu_chi", "ly_do")} for l in d["luot"]]})
    (ra / "so_sanh_quy_tac.json").write_text(json.dumps({"tong_hop": kq, "muc_lech_khong_khai_bao": doc_ra}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"\n(đã ghi {ra / 'so_sanh_quy_tac.json'}; {len(doc_ra)} mục còn lệch mà không khai báo tương đương)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
