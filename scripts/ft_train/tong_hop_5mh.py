# -*- coding: utf-8 -*-
"""Bang so sanh 5 mo hinh (Qwen3-4B, Qwen3-8B, EcomGPT-7B, OpenNMT, Moses) tren DUNG 1.198 mau eval PAIR (cung id, cung tham chieu, cung bo cham cham_diem).

Moi mo hinh LLM: 2 seed (42, 123) cua "natural", rank 16, lr 2e-4, 4M token (bao cao v55 va nlp-8b/nlp-ecom). OpenNMT va Moses: 1 mo hinh moi huong.
Chi so (zh->vi): chrF corpus title, mo ta, "chinh" = 0,10 title + 0,90 mo ta; macro = trung binh deu hai huong (moi huong = trung binh deu title va mo ta).
Khoang tin cay 95%: bootstrap THEO SAN PHAM (id.split('_')[0]) tren chrF CAU trung binh (thong ke xap xi cua chrF corpus, de tinh nhanh); chenh so voi Qwen3-4B cung phep bootstrap.
Chay: python -X utf8 tong_hop_5mh.py   (OPENBLAS_NUM_THREADS=1)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cham_co_dien as C  # noqa: E402
import cham_diem as D  # noqa: E402

R = C.GOC / "data" / "processed" / "ft_run"
RA = R / "tong_hop_5mh"
W = C.W_TITLE_ZHVI
NHOM = ["zh->vi/title", "zh->vi/mo_ta", "vi->zh/title", "vi->zh/mo_ta"]


def doc_sinh(duong_dan):
    return [json.loads(l) for l in Path(duong_dan).read_text(encoding="utf-8").split("\n") if l.strip()]


NHANH = "pair"          # "pair" (eval dong le) hoac "cum" (eval mau cum: ca mo ta, cac manh noi bang "; ")


def mo_hinh_llm(ten, goc):
    return {"ten": ten, "loai": "LLM", "lan_chay": {f"s{s}": [Path(goc) / "quota" / f"{NHANH}_natural_s{s}" / "sinh.jsonl"] for s in (42, 123)}}


def cac_mo_hinh():
    M = [mo_hinh_llm("Qwen3-4B", R / "kaggle_quota_step1"), mo_hinh_llm("Qwen3-8B", R / "kaggle_quota_nlp-8b"),
         mo_hinh_llm("EcomGPT-7B", R / "kaggle_quota_nlp-ecom")]
    tep = "sinh.jsonl" if NHANH == "pair" else "sinh_cum.jsonl"
    onmt = [R / "kaggle_onmt_that" / "onmt" / f"{hg}_train" / "sinh.jsonl" for hg in ("zh-vi", "vi-zh")] if NHANH == "pair" else \
           [R / "kaggle_onmt_cum" / "onmt_cum" / hg / "sinh_cum.jsonl" for hg in ("zh-vi", "vi-zh")]
    M.append({"ten": "OpenNMT (tu dau, train day du)", "loai": "co_dien", "lan_chay": {"1": onmt}})
    # Moses v2 (train day du + khop 4M token, char; dich ca eval pair va eval cum); neu chua co thi dung ban cu (train day du, chi eval pair)
    v2 = [R / "kaggle_moses_v2_zhvi", R / "kaggle_moses_v2_vizh"]
    if all((p / "moses_report.json").exists() for p in v2):
        for bt, ten in (("train", "Moses (train day du)"), ("khop4m_s42", "Moses (khop 4M token)")):
            ps = [p / "moses" / f"{hg}_char_{bt}" / tep for p, hg in zip(v2, ("zh-vi", "vi-zh"))]
            if all(x.exists() for x in ps):
                M.append({"ten": ten, "loai": "co_dien", "lan_chay": {"1": ps}})
    elif NHANH == "pair":
        mo = {}
        for hg, thu in (("zh-vi", "kaggle_moses_zhvi"), ("vi-zh", "kaggle_moses_vizh")):
            bc = R / thu / "moses_report.json"
            if bc.exists():
                v = [x for x in json.loads(bc.read_text(encoding="utf-8"))["viec"] if "dev_chrf" in x]
                if v:
                    tot = max(v, key=lambda x: x["dev_chrf"])
                    mo[hg] = (R / thu / "moses" / f"{hg}_{tot['seg']}" / "sinh.jsonl", tot["seg"], {x["seg"]: x["dev_chrf"] for x in v})
        if len(mo) == 2:
            M.append({"ten": "Moses (SMT)", "loai": "co_dien", "lan_chay": {"1": [mo["zh-vi"][0], mo["vi-zh"][0]]},
                      "ghi_chu": {"seg_chon_theo_dev": {h: mo[h][1] for h in mo}, "dev_chrf": {h: mo[h][2] for h in mo}}})
    return M

def cham_mot(duong_dan_list):
    refs = [C.tham_chieu(hg, NHANH) for hg in ("zh-vi", "vi-zh")]
    hyps = []
    for p in duong_dan_list:
        hyps += [{"id": r["id"], "hyp": r["hyp"]} for r in doc_sinh(p)]
    df, th = D.cham(pd.concat(refs, ignore_index=True), pd.DataFrame(hyps), nhanh=NHANH)
    df = df.sort_values("id").reset_index(drop=True)
    return df, th


def chinh_macro(gc):
    """gc: dict nhom -> gia tri (chrF) ; tra (chinh, macro)."""
    chinh = W * gc["zh->vi/title"] + (1 - W) * gc["zh->vi/mo_ta"]
    macro = np.mean([(gc["zh->vi/title"] + gc["zh->vi/mo_ta"]) / 2, (gc["vi->zh/title"] + gc["vi->zh/mo_ta"]) / 2])
    return float(chinh), float(macro)


def chuan_bi_boot(df):
    """Ma tran theo san pham: tong chrF cau va so muc cua tung (nhom, san pham)."""
    df = df.copy()
    df["pid"] = df["id"].str.split("_").str[0]
    df["nhom"] = df["huong"] + "/" + df["loai"]
    pids = sorted(df["pid"].unique())
    idx = {p: i for i, p in enumerate(pids)}
    S = np.zeros((len(NHOM), len(pids)))
    N = np.zeros((len(NHOM), len(pids)))
    for g, gname in enumerate(NHOM):
        d = df[df["nhom"] == gname]
        for p, v in zip(d["pid"], d["chrf"]):
            S[g, idx[p]] += v
            N[g, idx[p]] += 1
    return S, N


def thong_ke(S, N, w):
    """w: (B, P) so lan boc; tra (B,) chinh va macro theo chrF cau."""
    sw = w @ S.T            # (B, nhom)
    nw = w @ N.T
    gc = sw / np.maximum(nw, 1)
    chinh = W * gc[:, 0] + (1 - W) * gc[:, 1]
    macro = ((gc[:, 0] + gc[:, 1]) / 2 + (gc[:, 2] + gc[:, 3]) / 2) / 2
    return chinh, macro


def main(n_boot=2000, seed=7):
    global RA
    RA = R / ("tong_hop_5mh" if NHANH == "pair" else "tong_hop_5mh_cum")
    RA.mkdir(parents=True, exist_ok=True)
    ket, chung = {}, {}
    for m in cac_mo_hinh():
        cac, tb_df = {}, []
        for ten_lan, ds in m["lan_chay"].items():
            df, th = cham_mot(ds)
            nh = th["theo_nhom"]
            gc = {k: nh[k]["chrf_corpus"] for k in NHOM}
            chinh, macro = chinh_macro(gc)
            cac[ten_lan] = {"chrf_corpus": gc, "chinh": chinh, "macro": macro, "thieu_hyp": th["thieu_hyp"],
                            "ti_le_sinh_lap": {k: nh[k]["ti_le_sinh_lap"] for k in NHOM}, "ti_le_giu_so_dat": {k: nh[k]["ti_le_giu_so_dat"] for k in NHOM},
                            "ti_le_dung_so_dong": {k: nh[k].get("ti_le_dung_so_dong") for k in NHOM}}
            tb_df.append(df)
        # chrF cau trung binh qua cac lan chay (seed)
        ids = tb_df[0]["id"].tolist()
        assert all(d["id"].tolist() == ids for d in tb_df), "lech id giua cac lan chay"
        dm = tb_df[0].copy()
        dm["chrf"] = np.mean([d["chrf"].to_numpy() for d in tb_df], axis=0)
        chung[m["ten"]] = dm
        ket[m["ten"]] = {"loai": m["loai"], "cac_lan": cac, "ghi_chu": m.get("ghi_chu"),
                         "tb_chinh": float(np.mean([v["chinh"] for v in cac.values()])), "tb_macro": float(np.mean([v["macro"] for v in cac.values()])),
                         "lech_chinh_giua_lan": (max(v["chinh"] for v in cac.values()) - min(v["chinh"] for v in cac.values())) if len(cac) > 1 else None}
    # bootstrap theo san pham (cung mau boc cho moi mo hinh => so sanh cap)
    ten0 = list(chung)[0]
    S0, N0 = chuan_bi_boot(chung[ten0])
    P = S0.shape[1]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(P, np.ones(P) / P, size=n_boot).astype(float)
    boot = {}
    for t, d in chung.items():
        S, N = chuan_bi_boot(d)
        assert S.shape == S0.shape
        boot[t] = thong_ke(S, N, w)
    co_so = "Qwen3-4B"
    for t in ket:
        c, mc = boot[t]
        ket[t]["chrf_cau_tb"] = {"chinh": float(np.mean(chung_stat(chung[t], "chinh"))), "macro": float(np.mean(chung_stat(chung[t], "macro")))}
        ket[t]["ci95_chinh"] = [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
        ket[t]["ci95_macro"] = [float(np.percentile(mc, 2.5)), float(np.percentile(mc, 97.5))]
        if t != co_so:
            dc, dm_ = c - boot[co_so][0], mc - boot[co_so][1]
            ket[t]["chenh_so_voi_4B"] = {"chinh": [float(dc.mean()), float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
                                         "macro": [float(dm_.mean()), float(np.percentile(dm_, 2.5)), float(np.percentile(dm_, 97.5))]}
    (RA / "ket_qua.json").write_text(json.dumps(ket, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    dong = ["| Mo hinh | zh->vi title (corpus) | mo ta (corpus) | chinh (corpus) | macro (corpus) | chinh (chrF cau TB) [CI95] | chenh chinh cau so voi 4B [CI95] | sinh lap | lech seed chinh (corpus) |", "|---|---|---|---|---|---|---|---|---|"]
    for t, v in ket.items():
        cl = v["cac_lan"]
        tt = np.mean([c["chrf_corpus"]["zh->vi/title"] for c in cl.values()])
        mt = np.mean([c["chrf_corpus"]["zh->vi/mo_ta"] for c in cl.values()])
        lap = np.mean([np.mean(list(c["ti_le_sinh_lap"].values())) for c in cl.values()])
        ch = v.get("chenh_so_voi_4B", {}).get("chinh")
        dong.append(f"| {t} | {tt:.2f} | {mt:.2f} | {v['tb_chinh']:.2f} | {v['tb_macro']:.2f} | {v['chrf_cau_tb']['chinh']:.2f} [{v['ci95_chinh'][0]:.1f}; {v['ci95_chinh'][1]:.1f}] | "
                    f"{'' if ch is None else f'{ch[0]:+.2f} [{ch[1]:+.2f}; {ch[2]:+.2f}]'} | {lap:.4f} | {'' if v['lech_chinh_giua_lan'] is None else f'{v['lech_chinh_giua_lan']:.2f}'} |")
    (RA / "bang.md").write_text("\n".join(dong) + "\n", encoding="utf-8")
    print("\n".join(dong))


def chung_stat(df, ten):
    """Gia tri chrF cau trung binh (chinh hoac macro) tren toan mau, de doi chieu voi chrF corpus."""
    df = df.copy()
    df["nhom"] = df["huong"] + "/" + df["loai"]
    g = df.groupby("nhom")["chrf"].mean()
    c, m = chinh_macro({k: float(g[k]) for k in NHOM})
    return [c if ten == "chinh" else m]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        NHANH = sys.argv[1]
    main()
