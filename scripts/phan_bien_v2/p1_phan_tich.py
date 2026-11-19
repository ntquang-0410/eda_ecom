# -*- coding: utf-8 -*-
"""P1: tach dong khuon/moi, vong tham chieu, do dai, cho 5 mo hinh tren eval fold 0.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p1_phan_tich.py
Khong doc test, khong sua ket qua cu, khong train.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import cham_co_dien as C  # noqa: E402
import cham_diem as D  # noqa: E402
import tong_hop_5mh as T  # noqa: E402

R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"
SONG_NGU = R / "song_ngu"
BO_CHON = GOC / "data" / "processed" / "bo_chon_hf"
SEED, N_BOOT = 42, 2000


# ---------------------------------------------------------------- du lieu
def doc_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def refs_bo(nhanh):
    refs = pd.concat([C.tham_chieu(hg, nhanh) for hg in ("zh-vi", "vi-zh")], ignore_index=True)
    return refs


def map_category(nhanh):
    return {str(r["product_id"]): str(r.get("category") or "") for r in doc_jsonl(R / "mau" / f"pool_{nhanh}.jsonl")}


def tap_da_thay(nhanh):
    cot = ["loai", "category", "nguon", "nhanh", "fold5"]
    df = pd.DataFrame([{c: r.get(c) for c in cot} for r in doc_jsonl(R / "mau" / f"pool_{nhanh}.jsonl")])
    return D.xay_tap_da_thay(df, 0, nhanh)


def mo_hinh(nhanh):
    T.NHANH = nhanh
    ds = T.cac_mo_hinh()
    # dong thu sau: cong thuc chinh, chi khi id khop toan bo eval
    tep = R / "kaggle_dong6" / "dong6" / ("sinh_pair_f0_p11_pair.jsonl" if nhanh == "pair" else "sinh_cum_f0_p11_cum.jsonl")
    if tep.exists():
        ds.append({"ten": "Qwen3-4B công thức cuối (học fold 1-4, T4 fp16)", "loai": "LLM_cong_thuc_cuoi",
                   "lan_chay": {"1": [tep]}, "ghi_chu": "khuon loi dan dung_mau"})
    return ds


def cham_bo(nhanh, files, refs, tap):
    hyps = []
    for p in files:
        for r in doc_jsonl(p):
            hyps.append({"id": str(r["id"]), "hyp": r["hyp"]})
    df, th = D.cham(refs, pd.DataFrame(hyps), nhanh=nhanh, tap_da_thay=tap)
    thieu = int(df["thieu_hyp"].sum()) if "thieu_hyp" in df else -1
    if thieu:
        raise RuntimeError(f"thieu {thieu} hyp cho bo {nhanh}")
    return df, th


# ---------------------------------------------------------------- 3.2
def _corpus(h, hyps, refs):
    return D.chrf_corpus(h, list(hyps), list(refs))


def tach_dong(nhanh, models, refs, tap):
    ra = {}
    for m in models:
        ten = m["ten"]
        cac = {}
        for lan, files in m["lan_chay"].items():
            df, th = cham_bo(nhanh, files, refs, tap)
            dm = th.get("dong_da_thay_moi") or {}
            bo = {}
            for h in ("zh->vi", "vi->zh"):
                d_all = df[(df.loai == "mo_ta") & (df.huong == h)]
                pids_all = {str(x).split("_")[0] for x in d_all["id"]}
                d = d_all[d_all.can_dong == True]  # noqa: E712
                # gom theo san pham: moi san pham co the co nhieu dong/mau
                g = {}
                for r in d.to_dict("records"):
                    pid = str(r["id"]).split("_")[0]
                    rec = g.setdefault(pid, {"n": 0, "thay": 0, "moi": [], "khuon": []})
                    for (s, hy, rf) in list(r["_dong"]):
                        rec["n"] += 1
                        if s:
                            rec["thay"] += 1
                            rec["khuon"].append((hy, rf))
                        else:
                            rec["moi"].append((hy, rf))
                so_dong = int(sum(x["n"] for x in g.values()))
                so_thay = int(sum(x["thay"] for x in g.values()))
                theosp = [x["thay"] / x["n"] for x in g.values() if x["n"]]
                moi_sp, thay_sp = {}, {}
                for pid, x in g.items():
                    if x["moi"]:
                        moi_sp[pid] = _corpus(h, [y[0] for y in x["moi"]], [y[1] for y in x["moi"]])
                    if x["khuon"]:
                        thay_sp[pid] = _corpus(h, [y[0] for y in x["khuon"]], [y[1] for y in x["khuon"]])
                bo[h] = {
                    "n_sp": int(len(g)), "n_sp_tong": len(pids_all), "n_sp_bi_loai": len(pids_all - set(g)),
                    "n_muc_tong": int(len(d_all)), "n_muc_bi_loai": int(len(d_all) - len(d)),
                    "n_dong": so_dong, "n_dong_da_thay": so_thay,
                    "ti_le_dong": so_thay / so_dong if so_dong else None,
                    "ti_le_sp": float(np.mean(theosp)) if theosp else None,
                    "n_sp_co_dong_moi": len(moi_sp), "n_sp_co_dong_khuon": len(thay_sp),
                    "corpus_khuon": ((dm.get(h) or {}).get("da_thay") or {}).get("chrf_corpus"),
                    "corpus_moi": ((dm.get(h) or {}).get("moi") or {}).get("chrf_corpus"),
                    "n_dong_khuon": ((dm.get(h) or {}).get("da_thay") or {}).get("n_dong"),
                    "n_dong_moi": ((dm.get(h) or {}).get("moi") or {}).get("n_dong"),
                    "moi_sp": moi_sp, "khuon_sp": thay_sp,
                }
            cac[lan] = bo
        ra[ten] = {"loai": m["loai"], "lan": cac}
    return ra


def boot_chenh(a, b, n=N_BOOT, seed=SEED):
    """a, b: dict pid -> gia tri. Chi giu pid co ca hai. Tra mean, thap, cao, so pid."""
    pid = sorted(set(a) & set(b))
    if not pid:
        return None
    x = np.array([a[p] - b[p] for p in pid], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return {"chenh": float(x.mean()), "thap": float(np.percentile(m, 2.5)), "cao": float(np.percentile(m, 97.5)), "n_sp": len(pid)}


def gop_lan(bo, ten_lan):
    """Gop nhieu lan chay (seed) bang trung binh cac dict pid -> diem, va trung binh cac so tong."""
    ds = [bo["lan"][k] for k in ten_lan]
    ra = {"so_lan": len(ds)}
    for h in ("zh->vi", "vi->zh"):
        ra[h] = {}
        for k in ("ti_le_dong", "ti_le_sp", "corpus_khuon", "corpus_moi", "n_sp", "n_sp_tong", "n_sp_bi_loai",
                  "n_muc_tong", "n_muc_bi_loai", "n_dong",
                  "n_dong_khuon", "n_dong_moi", "n_sp_co_dong_moi", "n_sp_co_dong_khuon"):
            v = [x[h].get(k) for x in ds if x[h].get(k) is not None]
            ra[h][k] = float(np.mean(v)) if v else None
        for k in ("moi_sp", "khuon_sp"):
            pid = set(ds[0][h][k])
            for x in ds[1:]:
                pid &= set(x[h][k])
            ra[h][k] = {p: float(np.mean([x[h][k][p] for x in ds])) for p in pid}
    return ra


# ---------------------------------------------------------------- 3.3
def nhom_tham_chieu(nhanh):
    d = pd.read_parquet(BO_CHON / f"bo_{nhanh}_chon_deepseek.parquet", columns=["product_id", "judge_chon_title", "judge_chon_description"])
    d["pid"] = d["product_id"].astype(str)
    title = dict(zip(d["pid"], d["judge_chon_title"].fillna("").astype(str).str.lower()))
    desc, any_q = {}, {}
    for pid, s in zip(d["pid"], d["judge_chon_description"].fillna("").astype(str)):
        a = re.search(r"gemma\s*:\s*(\d+)", s, re.I)
        b = re.search(r"qwen\s*:\s*(\d+)", s, re.I)
        g, q = int(a.group(1)) if a else 0, int(b.group(1)) if b else 0
        any_q[pid] = "co_qwen" if q > 0 else "khong_qwen"
        if g + q == 0:
            desc[pid] = "khong_xac_dinh"
        else:
            r = q / (g + q)
            desc[pid] = "qwen" if r >= 2 / 3 else ("gemma" if r <= 1 / 3 else "tron")
    return {"title": title, "desc": desc, "desc_any": any_q}


def _theo_nhom(vals, nhom, ten_nhom):
    return {p: v for p, v in vals.items() if nhom.get(p) == ten_nhom}


def _tb(vals, pids):
    x = [vals[p] for p in pids if p in vals]
    return float(np.mean(x)) if x else None


def _kep(a_vals, b_vals, nhom_loai, M, B, g1="qwen", g2="gemma", n_boot=N_BOOT):
    vals = {}
    for g in (g1, g2):
        pids = [p for p in a_vals if nhom_loai.get(p) == g and p in b_vals]
        if not pids:
            vals[g] = None
            continue
        a = np.array([a_vals[p] for p in pids])
        b = np.array([b_vals[p] for p in pids])
        vals[g] = {"ten_M": M, "ten_B": B, "n_sp": len(pids), "M": float(a.mean()), "B": float(b.mean()),
                   "chenh": float((a - b).mean()), "_d": a - b}
    if vals.get(g1) is None or vals.get(g2) is None:
        return None
    rng = np.random.default_rng(SEED)
    dq, dg = vals[g1]["_d"], vals[g2]["_d"]
    bq = np.array([dq[rng.integers(0, len(dq), len(dq))].mean() for _ in range(n_boot)])
    bg = np.array([dg[rng.integers(0, len(dg), len(dg))].mean() for _ in range(n_boot)])
    kep = bq - bg
    return {g1: {k: v for k, v in vals[g1].items() if k != "_d"},
            g2: {k: v for k, v in vals[g2].items() if k != "_d"},
            "kep": {"chenh": vals[g1]["chenh"] - vals[g2]["chenh"],
                    "thap": float(np.percentile(kep, 2.5)), "cao": float(np.percentile(kep, 97.5))}}


def vong_tham_chieu(nhanh, models, refs, tap, nhom):
    # tinh per-product cho title va mo ta zh->vi
    diem = {}
    for m in models:
        ten = m["ten"]
        per = {}
        for lan, files in m["lan_chay"].items():
            df, _ = cham_bo(nhanh, files, refs, tap)
            t_acc, d_acc = {}, {}
            for r in df[df.huong == "zh->vi"].to_dict("records"):
                pid = str(r["id"]).split("_")[0]
                if r["loai"] == "title":
                    t_acc.setdefault(pid, []).append((r["hyp_cham"], r["_ref"]))
                else:
                    d_acc.setdefault(pid, []).append((r["hyp_cham"], r["_ref"]))
            t = {p: _corpus("zh->vi", [x[0] for x in v], [x[1] for x in v]) for p, v in t_acc.items()}
            d = {p: _corpus("zh->vi", [x[0] for x in v], [x[1] for x in v]) for p, v in d_acc.items()}
            per[lan] = {"title": t, "desc": d}
        pid_t = set.intersection(*[set(x["title"]) for x in per.values()])
        pid_d = set.intersection(*[set(x["desc"]) for x in per.values()])
        diem[ten] = {"title": {p: float(np.mean([x["title"][p] for x in per.values()])) for p in pid_t},
                     "desc": {p: float(np.mean([x["desc"][p] for x in per.values()])) for p in pid_d}}
    ra = {"nhom_toan_bo": {k: {g: sum(1 for v in vv.values() if v == g) for g in set(vv.values())}
                           for k, vv in nhom.items()},
          "nhom_trong_eval": {}, "title": {}, "desc": {}, "posthoc_desc_any_qwen": {}}
    Ms = [x for x in diem if x.startswith(("Qwen3-4B", "Qwen3-8B"))]
    Bs = [x for x in diem if ("Moses (train" in x or "OpenNMT" in x or "EcomGPT" in x)]
    for loai, nhom_loai in (("title", nhom["title"]), ("desc", nhom["desc"])):
        pids = set(diem["Qwen3-4B"][loai])
        ra["nhom_trong_eval"][loai] = {g: sum(1 for p in pids if nhom_loai.get(p) == g)
                                       for g in sorted(set(nhom_loai.get(p) for p in pids))}
        for M in Ms:
            for B in Bs:
                r = _kep(diem[M][loai], diem[B][loai], nhom_loai, M, B)
                if r:
                    ra[loai][f"{M} − {B}"] = r
    for M in Ms:
        for B in Bs:
            r = _kep(diem[M]["desc"], diem[B]["desc"], nhom["desc_any"], M, B, "co_qwen", "khong_qwen")
            if r:
                ra["posthoc_desc_any_qwen"][f"{M} − {B}"] = r
    return ra


# ---------------------------------------------------------------- 3.4
def do_dai(nhanh, models, refs, tap):
    ra = {}
    for m in models:
        ten = m["ten"]
        cac = {}
        for lan, files in m["lan_chay"].items():
            df, _ = cham_bo(nhanh, files, refs, tap)
            gia = {}
            for h in ("zh->vi", "vi->zh"):
                for loai in ("title", "mo_ta"):
                    d = df[(df.huong == h) & (df.loai == loai)].copy()
                    if d.empty:
                        continue
                    L = d["_nguon"].str.len().to_numpy()
                    q = np.percentile(L, [25, 50, 75])
                    cat = np.where(L <= q[0], "Q1", np.where(L <= q[1], "Q2", np.where(L <= q[2], "Q3", "Q4")))
                    d["nhom_dai"] = cat
                    g = {}
                    for k in ("Q1", "Q2", "Q3", "Q4"):
                        x = d[d.nhom_dai == k]
                        if len(x):
                            g[k] = {"n": int(len(x)), "chrf": _corpus(h, x["hyp_cham"], x["_ref"]),
                                    "q": [float(v) for v in q]}
                    gia[f"{h}/{loai}"] = g
            cac[lan] = gia
        # trung binh cac seed theo tung o
        ra[ten] = {}
        for k in cac[list(cac)[0]]:
            ra[ten][k] = {}
            for b in ("Q1", "Q2", "Q3", "Q4"):
                v = [x[k][b]["chrf"] for x in cac.values() if b in x.get(k, {})]
                n = [x[k][b]["n"] for x in cac.values() if b in x.get(k, {})]
                if v:
                    ra[ten][k][b] = {"chrf": float(np.mean(v)), "n": int(np.mean(n))}
    return ra


# ---------------------------------------------------------------- main
def main():
    RA.mkdir(parents=True, exist_ok=True)
    tong = {}
    for nhanh in ("pair", "cum"):
        refs = refs_bo(nhanh)
        cm = map_category(nhanh)
        refs["category"] = refs["id"].str.split("_").str[0].map(cm).fillna("")
        tap = tap_da_thay(nhanh)
        models = mo_hinh(nhanh)
        # 3.2
        t = tach_dong(nhanh, models, refs, tap)
        # gop seed + bootstrap vs Qwen3-4B
        base = "Qwen3-4B"
        base_lan = list(t[base]["lan"])
        base_g = gop_lan(t[base], base_lan)
        ra = {"theo_mo_hinh": {}, "so_sanh_4B": {}}
        for ten, v in t.items():
            g = gop_lan(v, list(v["lan"]))
            ra["theo_mo_hinh"][ten] = {"loai": v["loai"], **{h: {k: g[h][k] for k in ("ti_le_dong", "ti_le_sp", "corpus_khuon", "corpus_moi", "n_sp", "n_sp_tong", "n_sp_bi_loai", "n_muc_tong", "n_muc_bi_loai", "n_dong", "n_dong_khuon", "n_dong_moi", "n_sp_co_dong_moi", "n_sp_co_dong_khuon")} for h in ("zh->vi", "vi->zh")}}
            if ten != base:
                for h in ("zh->vi", "vi->zh"):
                    ra["so_sanh_4B"].setdefault(ten, {})[h] = {
                        "moi": boot_chenh(g[h]["moi_sp"], base_g[h]["moi_sp"]),
                        "khuon": boot_chenh(g[h]["khuon_sp"], base_g[h]["khuon_sp"]),
                    }
        (RA / f"tach_dong_{nhanh}.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(nhanh, "tach dong xong", len(ra["theo_mo_hinh"]), "mo hinh")
        # 3.3
        nhom = nhom_tham_chieu(nhanh)
        v = vong_tham_chieu(nhanh, models, refs, tap, nhom)
        (RA / f"vong_tham_chieu_{nhanh}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(nhanh, "vong tham chieu xong", len(v["desc"]), "cap")
        # 3.4
        d = do_dai(nhanh, models, refs, tap)
        (RA / f"do_dai_{nhanh}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(nhanh, "do dai xong")
        tong[nhanh] = {"so_mau": len(refs), "so_dong_tap": len(tap)}


if __name__ == "__main__":
    main()
