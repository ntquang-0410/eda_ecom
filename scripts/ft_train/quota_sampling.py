# -*- coding: utf-8 -*-
"""Dung mau quota cho Survey A (title + attribute/description) - pha 0, 0 GPU.

- Cum: block-level (Quang B): 1688_title, 1688_description, tiki_title, tiki_description.
- Pair: line-level (Quang A): 1688_title, 1688_attribute, tiki_title, tiki_attribute.
- 5 cau hinh: natural, balanced, title_heavy, attr/desc_heavy, tiki_heavy.
- Quota: T_g = s_g * BUDGET; lay deu trong nhom den du T_g; cap e_i o max_len.
- Co the loc title bang loc_title/pool.parquet (cot co_cờ).
- Khong train, khong mo test.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

ROOT = r"D:\nam4_ky1\e-commerce - EDA"
TRAIN_INDEX = os.path.join(ROOT, r"data\interim\khao_sat_trong_so\index_{tap}_train.csv")
DEV_INDEX = os.path.join(ROOT, r"data\interim\khao_sat_trong_so\index_{tap}_dev.csv")
LOC_TITLE = os.path.join(ROOT, r"data\processed\ft_run\loc_title\pool.parquet")
OUT_EDA = os.path.join(ROOT, r"data\eda\khao_sat_trong_so\quota_phase0")
OUT_INTERIM = os.path.join(ROOT, r"data\interim\quota_phase0")
os.makedirs(OUT_EDA, exist_ok=True)
os.makedirs(OUT_INTERIM, exist_ok=True)

BUDGET = 4_000_000
SEED = 42
PROMPT = {
    "cum": {"1688_title": 28.0, "1688_description": 97.0, "tiki_title": 28.0, "tiki_description": 104.0},
    "pair": {"1688_title": 28.0, "1688_attribute": 88.0, "tiki_title": 28.0, "tiki_attribute": 95.0},
}
CONFIGS = {
    "cum": {
        "natural": None,
        "balanced": {"1688_title": 25, "1688_description": 25, "tiki_title": 25, "tiki_description": 25},
        "title_heavy": {"1688_title": 40, "1688_description": 10, "tiki_title": 40, "tiki_description": 10},
        "desc_heavy": {"1688_title": 2, "1688_description": 55, "tiki_title": 3, "tiki_description": 40},
        "tiki_heavy": {"1688_title": 10, "1688_description": 15, "tiki_title": 35, "tiki_description": 40},
    },
    "pair": {
        "natural": None,
        "balanced": {"1688_title": 25, "1688_attribute": 25, "tiki_title": 25, "tiki_attribute": 25},
        "title_heavy": {"1688_title": 40, "1688_attribute": 20, "tiki_title": 25, "tiki_attribute": 15},
        "attr_heavy": {"1688_title": 10, "1688_attribute": 60, "tiki_title": 5, "tiki_attribute": 25},
        "tiki_heavy": {"1688_title": 10, "1688_attribute": 20, "tiki_title": 30, "tiki_attribute": 40},
    },
}
THU_TU = {
    "cum": ["1688_title", "1688_description", "tiki_title", "tiki_description"],
    "pair": ["1688_title", "1688_attribute", "tiki_title", "tiki_attribute"],
}


def doc_index(tap, split):
    p = (TRAIN_INDEX if split == "train" else DEV_INDEX).format(tap=tap)
    d = pd.read_csv(p, encoding="utf-8-sig")
    return d


def loc_title_map():
    f = pd.read_parquet(LOC_TITLE)
    cot = [c for c in f.columns if c.startswith("co_")][0]
    return dict(zip(f["product_id"].astype(str), f[cot].astype(bool)))


def loc_title(df, flag_map):
    giu = ~((df["loai"] == "title") & df["product_id"].astype(str).map(flag_map).fillna(False).astype(bool))
    return df[giu].reset_index(drop=True)


def natural_share(df, e):
    groups = df["group"].to_numpy()
    M = {g: float(e[groups == g].sum()) for g in THU_TU[df["tap"].iloc[0]] if (groups == g).any()}
    tong = sum(M.values())
    return {g: M[g] / tong for g in M}, M


def lay_nhom(rng, idx_g, e, T):
    if T <= 0 or len(idx_g) == 0:
        return np.array([], dtype=np.int64)
    mean = float(e[idx_g].mean())
    k = max(10, int(T / mean * 1.1) + 10)
    out = []
    total = 0.0
    while total < T:
        g = idx_g[rng.choice(len(idx_g), size=k, replace=True)]
        out.append(g)
        total += float(e[g].sum())
        if total >= T:
            break
        k = max(10, int((T - total) / mean * 1.1) + 10)
    idx = np.concatenate(out)
    cs = np.cumsum(e[idx])
    cut = int(np.searchsorted(cs, T, side="left")) + 1
    return idx[:cut]


def build(tap, cap, flt, flag_map):
    df = doc_index(tap, "train")
    df["tap"] = tap
    if flt:
        df = loc_title(df, flag_map)
    df = df.reset_index(drop=True)
    train_ref = os.path.join(OUT_INTERIM, f"train_ref_{tap}_flt{flt}.csv")
    df.to_csv(train_ref, index=False, encoding="utf-8-sig")

    e = np.minimum(pd.to_numeric(df["e"]).to_numpy(dtype=np.float64), cap)
    t = np.minimum(pd.to_numeric(df["tgt"]).to_numpy(dtype=np.float64), cap)
    groups = df["group"].to_numpy()
    idx_g = {g: np.where(groups == g)[0] for g in THU_TU[tap] if (groups == g).any()}
    M = {g: float(e[idx_g[g]].sum()) for g in idx_g}
    Mt = {g: float(t[idx_g[g]].sum()) for g in idx_g}
    tong = sum(M.values())
    natural = {g: M[g] / tong for g in idx_g}

    rows, summary = [], []
    for cname, cfg in CONFIGS[tap].items():
        if cfg is None:
            s = natural
        else:
            s = {g: cfg[g] / 100.0 for g in THU_TU[tap]}
        assert abs(sum(s.values()) - 1) < 1e-9, (tap, cname, sum(s.values()))
        rng = np.random.default_rng(SEED)
        parts = []
        for g in THU_TU[tap]:
            T = s[g] * BUDGET
            parts.append(lay_nhom(rng, idx_g[g], e, T) if g in idx_g else np.array([], dtype=np.int64))
        idx = np.concatenate([x for x in parts if len(x)])
        rng.shuffle(idx)
        np.save(os.path.join(OUT_INTERIM, f"idx_q_{tap}_{cname}_cap{cap}_flt{flt}.npy"), idx)

        ge, ee, tt = groups[idx], e[idx], t[idx]
        tong_e = float(ee.sum())
        tong_t = float(tt.sum())
        n_draw = len(idx)
        uniq = len(np.unique(idx))
        max_rep = int(np.bincount(idx).max()) if len(idx) else 0
        compute = tong_e + sum(float((ge == g).sum()) * PROMPT[tap].get(g, 0) for g in THU_TU[tap])
        title_e = sum(float(ee[ge == g].sum()) for g in THU_TU[tap] if g.endswith("_title"))
        title_t = sum(float(tt[ge == g].sum()) for g in THU_TU[tap] if g.endswith("_title"))
        summary.append({
            "tap": tap, "cap": cap, "flt": flt, "config": cname,
            "title_share_e": title_e / tong_e, "title_share_t": title_t / tong_t,
            "n_draw": n_draw, "n_unique": uniq, "max_rep": max_rep,
            "total_e": tong_e, "total_t": tong_t, "compute_est": compute,
            "overshoot_total": tong_e / BUDGET - 1,
        })
        for g in THU_TU[tap]:
            m = ge == g
            rows.append({
                "tap": tap, "cap": cap, "flt": flt, "config": cname, "group": g,
                "s_dat": s[g],
                "e_share": float(ee[m].sum() / tong_e) if tong_e else 0,
                "t_share": float(tt[m].sum() / tong_t) if tong_t else 0,
                "n_draw": int(m.sum()),
                "n_unique": int(len(np.unique(idx[m]))),
                "max_rep": int(np.bincount(idx[m]).max()) if m.any() else 0,
                "total_e": tong_e, "total_t": tong_t,
                "overshoot_g": float(ee[m].sum() / (s[g] * BUDGET) - 1) if s[g] > 0 else 0.0,
                "compute_est": compute,
            })
    return rows, summary, natural, {g: len(idx_g[g]) for g in idx_g}


def eval_sample(tap):
    df = doc_index(tap, "dev")
    rng = np.random.default_rng(20261109)
    chon = []
    for g in THU_TU[tap]:
        idx = np.where(df["group"].to_numpy() == g)[0]
        k = min(300, len(idx))
        chon.append(rng.choice(idx, size=k, replace=False))
    idx = np.concatenate(chon)
    np.save(os.path.join(OUT_INTERIM, f"eval_{tap}_sample.npy"), idx)
    df.iloc[idx].to_csv(os.path.join(OUT_INTERIM, f"eval_{tap}_sample.csv"), index=False, encoding="utf-8-sig")
    return len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="960,2048")
    ap.add_argument("--flt", default="0,1")
    a = ap.parse_args()
    caps = [int(x) for x in a.caps.split(",")]
    flts = [int(x) for x in a.flt.split(",")]
    flag_map = loc_title_map()
    all_rows, all_sum, meta = [], [], {}
    for tap in ["cum", "pair"]:
        for cap in caps:
            for flt in flts:
                rows, summary, natural, counts = build(tap, cap, flt, flag_map)
                all_rows += rows
                all_sum += summary
                meta[f"{tap}_cap{cap}_flt{flt}"] = {"natural": natural, "group_counts": counts}
        meta[f"{tap}_eval_n"] = eval_sample(tap)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT_EDA, "dry_run.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_sum).to_csv(os.path.join(OUT_EDA, "summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_EDA, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("GHI", OUT_EDA)
    print("GHI", OUT_INTERIM)


if __name__ == "__main__":
    main()
