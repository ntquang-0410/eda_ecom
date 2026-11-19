# -*- coding: utf-8 -*-
"""P2-D: chay lai `p1_phan_tich.vong_tham_chieu` voi thu tu ma san pham XAC DINH (sap xep), de xem hai dau khoang bootstrap cua P1 dich chuyen bao nhieu va co khoang nao DOI DAU
(chua 0 <-> khong chua 0). P1 rut bootstrap theo thu tu bam cua `set` Python (khong xac dinh giua cac lan chay), nen hai dau khoang co the lech ~0,1; diem uoc khong doi.

Khong ghi de tep P1: ket qua ghi vao `phan_bien_v2/kiem_xac_dinh_vong_tham_chieu.json`. Chi eval fold 0, khong doc test.
Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p7_kiem_p1_xac_dinh.py   (OPENBLAS_NUM_THREADS=1)
"""
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
sys.path.insert(0, str(GOC / "scripts" / "phan_bien_v2"))
import p1_phan_tich as P1  # noqa: E402

RA = GOC / "data" / "processed" / "ft_run" / "phan_bien_v2"
_kep_goc = P1._kep


def _kep_xac_dinh(a_vals, b_vals, nhom_loai, M, B, *args, **kw):
    return _kep_goc(dict(sorted(a_vals.items())), dict(sorted(b_vals.items())), nhom_loai, M, B, *args, **kw)


P1._kep = _kep_xac_dinh


def duyet(cu, moi, duong, ra):
    """Duyet de quy hai cay JSON: moi nut co khoa 'kep' (chenh, thap, cao) thi so sanh."""
    if isinstance(cu, dict) and "kep" in cu and isinstance(cu["kep"], dict) and "thap" in cu["kep"]:
        a, b = cu["kep"], moi["kep"]
        ra.append({"nut": duong, "diem_cu": a["chenh"], "diem_moi": b["chenh"], "lech_diem": abs(a["chenh"] - b["chenh"]),
                   "ci_cu": [a["thap"], a["cao"]], "ci_moi": [b["thap"], b["cao"]],
                   "lech_hai_dau": max(abs(a["thap"] - b["thap"]), abs(a["cao"] - b["cao"])),
                   "chua_0_cu": a["thap"] <= 0 <= a["cao"], "chua_0_moi": b["thap"] <= 0 <= b["cao"]})
        return
    if isinstance(cu, dict):
        for k in cu:
            if k in moi:
                duyet(cu[k], moi[k], f"{duong}/{k}", ra)


def main():
    ra = []
    for nhanh in ("pair", "cum"):
        refs = P1.refs_bo(nhanh)
        cm = P1.map_category(nhanh)
        refs["category"] = refs["id"].str.split("_").str[0].map(cm).fillna("")
        tap = P1.tap_da_thay(nhanh)
        models = P1.mo_hinh(nhanh)
        nhom = P1.nhom_tham_chieu(nhanh)
        moi = P1.vong_tham_chieu(nhanh, models, refs, tap, nhom)
        cu = json.loads((RA / f"vong_tham_chieu_{nhanh}.json").read_text(encoding="utf-8"))
        duyet({k: cu[k] for k in ("title", "desc", "posthoc_desc_any_qwen")}, {k: moi[k] for k in ("title", "desc", "posthoc_desc_any_qwen")}, nhanh, ra)
        print(nhanh, "xong")
    doi_dau = [r for r in ra if r["chua_0_cu"] != r["chua_0_moi"]]
    kq = {"so_khoang": len(ra), "lech_diem_lon_nhat": max(r["lech_diem"] for r in ra), "lech_hai_dau_lon_nhat": max(r["lech_hai_dau"] for r in ra),
          "so_khoang_doi_dau": len(doi_dau), "doi_dau": doi_dau, "chi_tiet": ra}
    (RA / "kiem_xac_dinh_vong_tham_chieu.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"so khoang {kq['so_khoang']} | lech diem lon nhat {kq['lech_diem_lon_nhat']:.2e} | lech hai dau lon nhat {kq['lech_hai_dau_lon_nhat']:.3f} | DOI DAU: {kq['so_khoang_doi_dau']}")
    for r in doi_dau:
        print("  doi dau:", r["nut"], r["ci_cu"], "->", r["ci_moi"])
    gan0 = sorted(ra, key=lambda r: min(abs(r["ci_moi"][0]), abs(r["ci_moi"][1])))[:6]
    print("6 khoang co dau gan 0 nhat (moi):")
    for r in gan0:
        print(f"  {r['nut']}: cu [{r['ci_cu'][0]:+.2f}; {r['ci_cu'][1]:+.2f}] moi [{r['ci_moi'][0]:+.2f}; {r['ci_moi'][1]:+.2f}]")


if __name__ == "__main__":
    main()
