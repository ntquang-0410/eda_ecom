# -*- coding: utf-8 -*-
"""P2-B: khoang tin cay cho ti trong title 25% va 35% so voi 11% (cum, fold 0, seed 42, 1.000 mau). Ke hoach khoa: docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P2.md.

Dung lai mau va tham chieu y het chay_b78: kaggle_train.chon_mau_sinh(pool_cum, pool_cum, fold 0, 250, seed 42) tren pool_cum.jsonl (pool, khong phai test),
cham bang cham_diem.cham(..., tap_da_thay=...). Kiem tai lap truoc: chrf_macro_huong va 4 chrf_corpus theo nhom phai khop bao cao Kaggle (sai so 1e-6).
Chi so: chrF CAU trung binh (xap xi chrF corpus, nhu tong_hop_5mh): chinh 0,10/0,90 zh->vi, macro, tung nhom. Bootstrap ghep cap theo ma san pham, 2.000 lan, default_rng(42), cung ma tran rut.
Khoang tin cay chi phu dao dong do chon mau eval, KHONG phu dao dong khoi tao (1 seed).

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p5_w_title_ci.py   (OPENBLAS_NUM_THREADS=1)
Ra:   data/processed/ft_run/phan_bien_v2/w_title_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import cham_diem as D  # noqa: E402
import kaggle_train as K  # noqa: E402
import tong_hop_5mh as T  # noqa: E402

R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"
PH2 = R / "kaggle_tlc_v2_tiep"
VIEC = {"p25": "cum_f0_p25", "p35": "cum_f0_p35", "p11": "cum_f0_p11"}
CAP = [("p25", "p11"), ("p35", "p11"), ("p35", "p25")]
N_BOOT, SEED = 2000, 42


def doc_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def khoang(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def tom_tat(d):
    lo, hi = khoang(d)
    return {"chenh": float(d.mean()), "ci95": [lo, hi], "chua_0": bool(lo <= 0 <= hi)}


def cham_viec(ten, mau, tap):
    du = doc_jsonl(PH2 / f"tlc_{ten}" / ten / "sinh_ep2.jsonl")
    ids = {m["id"] for m in mau}
    du = [r for r in du if r["id"] in ids]
    assert len(du) == len(mau), (ten, len(du), len(mau))
    return D.cham(pd.DataFrame(mau), pd.DataFrame(du), nhanh="cum", tap_da_thay=tap)


def main():
    pool = doc_jsonl(R / "mau" / "pool_cum.jsonl")
    mau, thieu = K.chon_mau_sinh(pool, pool, 0, 250, 42)
    assert thieu == 0 and len(mau) == 1000, (thieu, len(mau))
    tap = D.xay_tap_da_thay(pd.DataFrame([{c: r.get(c) for c in ("loai", "category", "nguon", "nhanh", "fold5")} for r in pool]), 0, "cum")
    dfs, tai_lap = {}, {}
    for k, ten in VIEC.items():
        df, th = cham_viec(ten, mau, tap)
        df = df.sort_values("id").reset_index(drop=True)
        bc = json.loads((PH2 / f"bao_cao_tlc_{ten}.json").read_text(encoding="utf-8"))
        ch = bc["b78"]["viec"][ten]["cham"] if "b78" in bc else None
        if ch is None:      # cau truc bao cao: tim khoa cham trong trang thai
            ch = json.loads((PH2 / f"tlc_{ten}" / f"tlc_{ten}_trang_thai.json").read_text(encoding="utf-8"))["viec"][ten]["cham"]
        lech = abs(th["diem_chinh"]["chrf_macro_huong"] - ch["chrf"]["chrf_macro_huong"])
        for g in T.NHOM:
            lech = max(lech, abs(th["theo_nhom"][g]["chrf_corpus"] - ch["theo_nhom"][g]["chrf_corpus"]))
        tai_lap[k] = {"macro_moi": th["diem_chinh"]["chrf_macro_huong"], "macro_kaggle": ch["chrf"]["chrf_macro_huong"], "lech_lon_nhat": float(lech)}
        assert lech <= 1e-6, f"TAI LAP LECH {k}: {lech}"
        dfs[k] = df
    ids0 = dfs["p11"]["id"].tolist()
    assert all(d["id"].tolist() == ids0 for d in dfs.values())
    S0, N0 = T.chuan_bi_boot(dfs["p11"])
    P = S0.shape[1]
    w = np.random.default_rng(SEED).multinomial(P, np.ones(P) / P, size=N_BOOT).astype(float)
    thong = {}
    for k, d in dfs.items():
        S, N = T.chuan_bi_boot(d)
        assert S.shape == S0.shape
        gc = (w @ S.T) / np.maximum(w @ N.T, 1)            # (B, 4 nhom)
        c, m = T.thong_ke(S, N, w)
        thong[k] = {"chinh": c, "macro": m, "nhom": gc,
                    "diem_goc": {"chinh": float(np.mean(T.chung_stat(d, "chinh"))), "macro": float(np.mean(T.chung_stat(d, "macro")))}}
    ra = {"ke_hoach": "docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P2.md", "n_boot": N_BOOT, "seed_boot": SEED, "so_san_pham": int(P),
          "tai_lap_diem_kaggle": tai_lap, "diem_cau_tb": {k: v["diem_goc"] for k, v in thong.items()}, "cap": {}}
    for a, b in CAP:
        r = {"chinh": tom_tat(thong[a]["chinh"] - thong[b]["chinh"]), "macro": tom_tat(thong[a]["macro"] - thong[b]["macro"]), "nhom": {}}
        for i, g in enumerate(T.NHOM):
            r["nhom"][g] = tom_tat(thong[a]["nhom"][:, i] - thong[b]["nhom"][:, i])
        ra["cap"][f"{a} - {b}"] = r
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "w_title_ci.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tai lap diem Kaggle:", {k: f"lech {v['lech_lon_nhat']:.2e}" for k, v in tai_lap.items()})
    print("diem cau TB:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in ra["diem_cau_tb"].items()})
    for ten, r in ra["cap"].items():
        print(f"== {ten}: chinh {r['chinh']['chenh']:+.2f} [{r['chinh']['ci95'][0]:+.2f}; {r['chinh']['ci95'][1]:+.2f}] chua0={r['chinh']['chua_0']} | "
              f"macro {r['macro']['chenh']:+.2f} [{r['macro']['ci95'][0]:+.2f}; {r['macro']['ci95'][1]:+.2f}] chua0={r['macro']['chua_0']}")
        for g, v in r["nhom"].items():
            print(f"     {g:14s} {v['chenh']:+.2f} [{v['ci95'][0]:+.2f}; {v['ci95'][1]:+.2f}] chua0={v['chua_0']}")


if __name__ == "__main__":
    main()
