# -*- coding: utf-8 -*-
"""3.1 Tai lap bang 5 mo hinh tren eval fold 0 (pair, cum). Khong ghi de ket qua cu.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p1_tai_lap.py
"""
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import tong_hop_5mh as T  # noqa: E402

RA = GOC / "data" / "processed" / "ft_run" / "phan_bien_v2"
SAI_SO = 0.01


def mot_bo(nhanh):
    T.NHANH = nhanh
    ket = {}
    for m in T.cac_mo_hinh():
        cac = []
        for ten_lan, ds in m["lan_chay"].items():
            df, th = T.cham_mot(ds)
            gc = {k: th["theo_nhom"][k]["chrf_corpus"] for k in T.NHOM}
            chinh, macro = T.chinh_macro(gc)
            cac.append({"lan": ten_lan, "chrf_corpus": gc, "chinh": chinh, "macro": macro})
        ket[m["ten"]] = {"loai": m["loai"], "cac_lan": cac,
                         "tb_chinh": float(np.mean([c["chinh"] for c in cac])),
                         "tb_macro": float(np.mean([c["macro"] for c in cac]))}
    return ket


def so_cu(nhanh):
    p = GOC / "data" / "processed" / "ft_run" / ("tong_hop_5mh" if nhanh == "pair" else "tong_hop_5mh_cum") / "ket_qua.json"
    return json.loads(p.read_text(encoding="utf-8"))


def so_sanh(ten, moi, cu):
    ra = []
    if ten not in cu:
        return [{"ten": ten, "loi": "khong co trong ket qua cu"}]
    for k in ("tb_chinh", "tb_macro"):
        a, b = float(moi[k]), float(cu[ten][k])
        ra.append({"ten": ten, "o": k, "moi": a, "cu": b, "lech": abs(a - b), "dat": abs(a - b) <= SAI_SO})
    for lan in moi["cac_lan"]:
        ten_lan = lan["lan"]
        if ten_lan not in cu[ten]["cac_lan"]:
            continue
        for g in T.NHOM:
            a = lan["chrf_corpus"][g]
            b = cu[ten]["cac_lan"][ten_lan]["chrf_corpus"][g]
            ra.append({"ten": ten, "o": f"{ten_lan}:{g}", "moi": a, "cu": b, "lech": abs(a - b), "dat": abs(a - b) <= SAI_SO})
    return ra


def main():
    RA.mkdir(parents=True, exist_ok=True)
    tat = {}
    for nhanh in ("pair", "cum"):
        moi = mot_bo(nhanh)
        cu = so_cu(nhanh)
        dong = []
        for ten in moi:
            dong += so_sanh(ten, moi[ten], cu)
        tat[nhanh] = {"cac_o": dong, "dat": all(x.get("dat", False) for x in dong if "loi" not in x),
                      "lech_lon_nhat": max((x.get("lech", 0) for x in dong), default=0)}
        (RA / f"tai_lap_{nhanh}.json").write_text(json.dumps({"moi": moi, "so_sanh": tat[nhanh]}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(nhanh, "dat:", tat[nhanh]["dat"], "lech lon nhat:", round(tat[nhanh]["lech_lon_nhat"], 5))
        for x in dong:
            if not x.get("dat", False):
                print("  LECH", x)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "cac_o"} for k, v in tat.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
