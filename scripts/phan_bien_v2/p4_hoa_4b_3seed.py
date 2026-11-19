# -*- coding: utf-8 -*-
"""P2-A: "OpenNMT hoa Qwen3-4B" tinh lai cho 3 seed (42 = D3, 43/44 = Colab 18/11/2026). Ke hoach khoa: docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P2.md.

Cung chi so va cung bootstrap voi tong_hop_5mh.py (chrF CAU trung binh, chinh 0,10/0,90 zh->vi, macro; bootstrap theo san pham, 2.000 lan, default_rng(7), cung ma tran rut).
Kiem tai lap truoc: OpenNMT seed 42 mot minh so voi 4B phai ra dung so trong tong_hop_5mh{,_cum}/ket_qua.json (lech > 0,01 thi dung).
Chi eval fold 0, khong doc test.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p4_hoa_4b_3seed.py   (OPENBLAS_NUM_THREADS=1)
Ra:   data/processed/ft_run/phan_bien_v2/hoa_4b_3seed.json
"""
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import tong_hop_5mh as T  # noqa: E402

R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"
N_BOOT, SEED = 2000, 7
SEEDS = (42, 43, 44)


def duong_dan_onmt(seed, nhanh):
    if seed == 42:
        goc = R / "kaggle_onmt_that" / "onmt" if nhanh == "pair" else R / "kaggle_onmt_cum" / "onmt_cum"
    else:
        goc = R / "colab_onmt" / f"s{seed}" / "out" / ("onmt" if nhanh == "pair" else "onmt_cum")
    if nhanh == "pair":
        return [goc / f"{hg}_train" / "sinh.jsonl" for hg in ("zh-vi", "vi-zh")]
    return [goc / hg / "sinh_cum.jsonl" for hg in ("zh-vi", "vi-zh")]


def khoang(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def tom_tat(d):
    return {"chenh": float(d.mean()), "ci95": khoang(d), "chua_0": bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))}


def chay(nhanh):
    T.NHANH = nhanh
    # Qwen3-4B: trung binh chrF cau cua 2 seed (42, 123) tung mau, nhu tong_hop_5mh
    m4b = T.mo_hinh_llm("Qwen3-4B", R / "kaggle_quota_step1")
    dfs4 = [T.cham_mot(ds)[0] for ds in m4b["lan_chay"].values()]
    ids = dfs4[0]["id"].tolist()
    assert all(d["id"].tolist() == ids for d in dfs4)
    d4 = dfs4[0].copy()
    d4["chrf"] = np.mean([d["chrf"].to_numpy() for d in dfs4], axis=0)
    # OpenNMT tung seed
    dfs = {}
    for s in SEEDS:
        ps = duong_dan_onmt(s, nhanh)
        assert all(p.exists() for p in ps), ps
        df, _ = T.cham_mot(ps)
        assert df["id"].tolist() == ids, f"lech id seed {s}"
        dfs[s] = df
    d3 = dfs[42].copy()
    d3["chrf"] = np.mean([dfs[s]["chrf"].to_numpy() for s in SEEDS], axis=0)
    # bootstrap: cung ma tran rut cho moi bien the
    S4, N4 = T.chuan_bi_boot(d4)
    P = S4.shape[1]
    w = np.random.default_rng(SEED).multinomial(P, np.ones(P) / P, size=N_BOOT).astype(float)
    c4, m4 = T.thong_ke(S4, N4, w)
    bien_the = {f"seed{s}": dfs[s] for s in SEEDS}
    bien_the["tb3seed"] = d3
    ra = {"so_san_pham": int(P)}
    for ten, d in bien_the.items():
        S, N = T.chuan_bi_boot(d)
        assert S.shape == S4.shape
        c, m = T.thong_ke(S, N, w)
        ra[ten] = {"chinh_cau_tb": float(np.mean(T.chung_stat(d, "chinh"))), "macro_cau_tb": float(np.mean(T.chung_stat(d, "macro"))),
                   "chenh_chinh": tom_tat(c - c4), "chenh_macro": tom_tat(m - m4)}
    ra["qwen3_4b"] = {"chinh_cau_tb": float(np.mean(T.chung_stat(d4, "chinh"))), "macro_cau_tb": float(np.mean(T.chung_stat(d4, "macro")))}
    # miền dao động của chênh điểm gốc giữa 3 seed (điểm ước, không phải CI)
    ra["mien_3seed_chenh_chinh"] = [min(ra[f"seed{s}"]["chenh_chinh"]["chenh"] for s in SEEDS), max(ra[f"seed{s}"]["chenh_chinh"]["chenh"] for s in SEEDS)]
    ra["mien_3seed_chenh_macro"] = [min(ra[f"seed{s}"]["chenh_macro"]["chenh"] for s in SEEDS), max(ra[f"seed{s}"]["chenh_macro"]["chenh"] for s in SEEDS)]
    return ra


def main():
    cu = {}
    for nhanh, tep in (("pair", "tong_hop_5mh"), ("cum", "tong_hop_5mh_cum")):
        k = json.loads((R / tep / "ket_qua.json").read_text(encoding="utf-8"))["OpenNMT (tu dau, train day du)"]
        cu[nhanh] = k["chenh_so_voi_4B"]
    ra = {"ke_hoach": "docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P2.md", "n_boot": N_BOOT, "seed_boot": SEED}
    for nhanh in ("pair", "cum"):
        r = chay(nhanh)
        # kiem tai lap: seed 42 mot minh phai trung so da cong bo
        for loai, ck in (("chenh_chinh", "chinh"), ("chenh_macro", "macro")):
            moi, cho = r["seed42"][loai], cu[nhanh][ck]
            sai = max(abs(moi["chenh"] - cho[0]), abs(moi["ci95"][0] - cho[1]), abs(moi["ci95"][1] - cho[2]))
            r.setdefault("kiem_tai_lap", {})[ck] = {"moi": [moi["chenh"]] + moi["ci95"], "da_cong_bo": cho, "lech_lon_nhat": float(sai)}
            assert sai <= 0.01, f"TAI LAP LECH {nhanh} {ck}: {sai}"
        ra[nhanh] = r
        print(f"== {nhanh}: tai lap khop (lech lon nhat chinh {r['kiem_tai_lap']['chinh']['lech_lon_nhat']:.5f}, macro {r['kiem_tai_lap']['macro']['lech_lon_nhat']:.5f})")
        for ten in ("seed42", "seed43", "seed44", "tb3seed"):
            c, m = r[ten]["chenh_chinh"], r[ten]["chenh_macro"]
            print(f"   {ten:8s} chinh {c['chenh']:+.2f} [{c['ci95'][0]:+.2f}; {c['ci95'][1]:+.2f}] chua0={c['chua_0']} | macro {m['chenh']:+.2f} [{m['ci95'][0]:+.2f}; {m['ci95'][1]:+.2f}] chua0={m['chua_0']}")
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "hoa_4b_3seed.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ghi", RA / "hoa_4b_3seed.json")


if __name__ == "__main__":
    main()
