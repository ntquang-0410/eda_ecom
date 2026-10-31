"""So sánh chất lượng hai bộ mô tả (bản CỤM và bản PAIR) do CÙNG một judge chấm, trên các sản phẩm có kết quả ok ở cả hai.

  python -I -X utf8 so_sanh_hai_bo.py <ten_cum> <ten_pair> [--ten-title <ten_title>]
vd: so_sanh_hai_bo.py toan_bo mo_ta_pair_toan_bo --ten-title title_toan_bo

Khoảng tin cậy 95% lấy bằng bootstrap có cặp theo sản phẩm (2.000 lần, hạt giống cố định). Đây là so sánh CHẤT LƯỢNG DỮ LIỆU theo
judge; chưa phải kết luận về kết quả fine-tune, và chưa được đối chiếu với tập người chấm."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cau_hinh as CH  # noqa: E402
from so_sanh_hai_lan import doc  # noqa: E402

CHIEU = CH.CHIEU + ["diem_thap_nhat"]


def _diem(x, k):
    return min(x[c] for c in CH.CHIEU) if k == "diem_thap_nhat" else x[k]


def _ci(d, rng, so_lan=2000):
    if len(d) == 0:
        return [None, None]
    m = rng.choice(d, size=(so_lan, len(d)), replace=True).mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def so_sanh(a, b, hat=20261030):
    rng = np.random.default_rng(hat)
    chung = sorted(p for p in a if p in b and a[p]["trang_thai"] == "ok" and b[p]["trang_thai"] == "ok")
    r = {"so_san_pham_chung": len(chung), "a_ok": sum(x["trang_thai"] == "ok" for x in a.values()),
         "b_ok": sum(x["trang_thai"] == "ok" for x in b.values())}
    if not chung:
        return r

    def khoi(ids):
        out = {}
        for k in CHIEU:
            xa = np.array([_diem(a[p], k) for p in ids], dtype=float)
            xb = np.array([_diem(b[p], k) for p in ids], dtype=float)
            d = xa - xb
            out[k] = {"tb_a": round(float(xa.mean()), 3), "tb_b": round(float(xb.mean()), 3),
                      "a_tru_b": round(float(d.mean()), 3), "ci95": _ci(d, rng),
                      "a_cao_hon": int((d > 0).sum()), "b_cao_hon": int((d < 0).sum()), "bang": int((d == 0).sum())}
        dat_a = np.array([a[p]["ket_luan"] == "dat" for p in ids])
        dat_b = np.array([b[p]["ket_luan"] == "dat" for p in ids])
        cam_a = np.array([bool(a[p]["co_cam_co"]) for p in ids])
        cam_b = np.array([bool(b[p]["co_cam_co"]) for p in ids])
        out["ti_le_dat"] = {"a": round(float(dat_a.mean()), 3), "b": round(float(dat_b.mean()), 3),
                            "chi_a_dat": int((dat_a & ~dat_b).sum()), "chi_b_dat": int((~dat_a & dat_b).sum())}
        out["ti_le_cam_co"] = {"a": round(float(cam_a.mean()), 3), "b": round(float(cam_b.mean()), 3),
                               "chi_a_cam_co": int((cam_a & ~cam_b).sum()), "chi_b_cam_co": int((~cam_a & cam_b).sum())}
        out["loi_theo_loai"] = {"a": dict(Counter(e["loai"] for p in ids for e in a[p]["loi"])),
                                "b": dict(Counter(e["loai"] for p in ids for e in b[p]["loi"]))}
        return out
    r["tat_ca"] = khoi(chung)
    r["theo_nguon"] = {s: khoi([p for p in chung if a[p]["source"] == s]) for s in sorted({a[p]["source"] for p in chung})}
    r["ghi_chu"] = "A = bộ thứ nhất (vd. cụm), B = bộ thứ hai (vd. pair); a_tru_b > 0 nghĩa là A có điểm cao hơn."
    return r


def thong_ke_title(b):
    ok = [x for x in b.values() if x["trang_thai"] == "ok"]
    if not ok:
        return {}
    return {"so_title_ok": len(ok), "tb": {k: round(float(np.mean([_diem(x, k) for x in ok])), 3) for k in CHIEU},
            "ti_le_dat": round(sum(x["ket_luan"] == "dat" for x in ok) / len(ok), 3),
            "ti_le_cam_co": round(sum(bool(x["co_cam_co"]) for x in ok) / len(ok), 3),
            "theo_nguon": {s: {"so": sum(1 for x in ok if x["source"] == s),
                               "ti_le_cam_co": round(sum(bool(x["co_cam_co"]) for x in ok if x["source"] == s) /
                                                     max(sum(1 for x in ok if x["source"] == s), 1), 3)}
                           for s in sorted({x["source"] for x in ok})}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cum")
    p.add_argument("pair")
    p.add_argument("--ten-title")
    a = p.parse_args()
    kq = {"mo_ta_cum_so_voi_pair": so_sanh(doc(a.cum), doc(a.pair))}
    if a.ten_title:
        kq["title"] = thong_ke_title(doc(a.ten_title))
    print(json.dumps(kq, ensure_ascii=False, indent=1))
