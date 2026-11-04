"""So sánh hai đợt chạy judge trên cùng sản phẩm (vd. bật suy luận và tắt suy luận).

Dùng:  python -I -X utf8 so_sanh_hai_lan.py thu_300 thu_300_nothink
Lấy bản ghi CUỐI CÙNG của mỗi product_id trong từng file (chạy lại thêm dòng vào cuối cùng một file)."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cau_hinh as CH  # noqa: E402


def doc(ten, thu_muc=None):
    cuoi = {}
    with open(Path(thu_muc or CH.THU_MUC_RA) / f"ket_qua_{ten}.jsonl", encoding="utf-8") as f:
        for dong in f:
            d = json.loads(dong)
            cuoi[d["product_id"]] = d
    return cuoi


def so_sanh(a, b):
    chung = [p for p in a if p in b and a[p]["trang_thai"] == "ok" and b[p]["trang_thai"] == "ok"]
    kq = {"chung_ok": len(chung), "a_ok": sum(x["trang_thai"] == "ok" for x in a.values()),
          "b_ok": sum(x["trang_thai"] == "ok" for x in b.values())}
    if not chung:
        return kq
    kq["giong_ket_luan"] = round(sum(a[p]["ket_luan"] == b[p]["ket_luan"] for p in chung) / len(chung), 3)
    ca, cb = {p for p in chung if a[p]["co_cam_co"]}, {p for p in chung if b[p]["co_cam_co"]}
    kq["co_ca_hai"], kq["chi_a_co"], kq["chi_b_co"] = len(ca & cb), len(ca - cb), len(cb - ca)
    kq["diem_lech_tb"] = {k: round(sum(abs(a[p][k] - b[p][k]) for p in chung) / len(chung), 3) for k in CH.CHIEU}
    kq["diem_tb_a"] = {k: round(sum(a[p][k] for p in chung) / len(chung), 3) for k in CH.CHIEU}
    kq["diem_tb_b"] = {k: round(sum(b[p][k] for p in chung) / len(chung), 3) for k in CH.CHIEU}
    for ten, x in (("a", a), ("b", b)):
        kq[f"token_ra_tb_{ten}"] = round(sum(x[p]["tokens_ra"] for p in chung) / len(chung))
        kq[f"chi_phi_uoc_{ten}"] = round(sum(x[p]["chi_phi_niem_yet_uoc_usd"] for p in chung), 4)
    kq["loi_dem_a"] = collections.Counter(e["loai"] for p in chung for e in a[p]["loi"]).most_common()
    kq["loi_dem_b"] = collections.Counter(e["loai"] for p in chung for e in b[p]["loi"]).most_common()
    return kq


if __name__ == "__main__":
    ten_a, ten_b = sys.argv[1], sys.argv[2]
    print(json.dumps(so_sanh(doc(ten_a), doc(ten_b)), ensure_ascii=False, indent=1))
