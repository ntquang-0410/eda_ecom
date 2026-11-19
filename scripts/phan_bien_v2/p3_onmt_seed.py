# -*- coding: utf-8 -*-
"""Cham OpenNMT 3 seed (42 = D3 Kaggle, 43/44 = Colab T4 18/11/2026) bang DUNG cham_co_dien.cham (cung chrF, cung tham chieu, cung id eval fold 0). Khong dung test.
Seed 42 duoc cham lai de doi chieu voi so da ghi o D3 (pair chinh 74,38 macro 54,89; cum chinh 80,61 macro 58,72): lech thi dung lai.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p3_onmt_seed.py
Ra:   data/processed/ft_run/phan_bien_v2/onmt_3seed.json
"""
import json
import statistics as st
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import cham_co_dien as C  # noqa: E402

R = GOC / "data" / "processed" / "ft_run"
NGUON = {
    42: (R / "kaggle_onmt_that" / "onmt", R / "kaggle_onmt_cum" / "onmt_cum"),
    43: (R / "colab_onmt" / "s43" / "out" / "onmt", R / "colab_onmt" / "s43" / "out" / "onmt_cum"),
    44: (R / "colab_onmt" / "s44" / "out" / "onmt", R / "colab_onmt" / "s44" / "out" / "onmt_cum"),
}
KHOA = ["zh->vi/title", "zh->vi/mo_ta", "chinh_w0.10", "vi->zh/title", "vi->zh/mo_ta", "macro"]


def main():
    ra = {"pair": {}, "cum": {}}
    for seed, (pair_d, cum_d) in NGUON.items():
        sp = {hg: str(pair_d / f"{hg}_train" / "sinh.jsonl") for hg in ("zh-vi", "vi-zh")}
        sc = {hg: str(cum_d / hg / "sinh_cum.jsonl") for hg in ("zh-vi", "vi-zh")}
        for p in list(sp.values()) + list(sc.values()):
            assert Path(p).exists(), p
        _, tp = C.cham(sp, "pair")
        _, tc = C.cham(sc, "cum")
        ra["pair"][seed], ra["cum"][seed] = tp, tc
        print(f"seed {seed}")
        for ten, t in (("pair", tp), ("cum ", tc)):
            print("  ", ten, " | ".join(f"{k} {t[k]:.2f}" for k in KHOA), "| lap", {k: round(v, 4) for k, v in t["ti_le_sinh_lap"].items() if v is not None})
    ra["thong_ke"] = {}
    for nhanh in ("pair", "cum"):
        ra["thong_ke"][nhanh] = {}
        for k in KHOA:
            v = [ra[nhanh][s][k] for s in NGUON]
            ra["thong_ke"][nhanh][k] = {"tb": st.mean(v), "do_lech_chuan": st.stdev(v), "min": min(v), "max": max(v), "khoang": max(v) - min(v)}
    print("\nthong ke 3 seed (tb / do lech chuan / min-max)")
    for nhanh in ("pair", "cum"):
        for k in KHOA:
            s = ra["thong_ke"][nhanh][k]
            print(f"  {nhanh} {k:14s} {s['tb']:.2f} / {s['do_lech_chuan']:.2f} / {s['min']:.2f}-{s['max']:.2f}")
    out = R / "phan_bien_v2" / "onmt_3seed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\nghi", out)


if __name__ == "__main__":
    main()
