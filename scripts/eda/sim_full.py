# -*- coding: utf-8 -*-
"""bge-m3 (via local Ollama) zh-vi title similarity on the FULL bilingual set,
with mismatched-pair baselines to calibrate the threshold.

    python sim_full.py check          # compare against 4 scores the notebook reported
    python sim_full.py full <out_dir>   # e.g. data/eda
"""
import json, re, sys, time, urllib.request
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OLLAMA = "http://127.0.0.1:11434/api/embed"
CJK = re.compile(r"[\u4e00-\u9fff]")


def embed(texts, bs=64):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        body = json.dumps({"model": "bge-m3", "input": texts[i:i + bs]}).encode("utf-8")
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out.extend(json.loads(r.read())["embeddings"])
        if (i // bs) % 50 == 0:
            print(f"  embedded {len(out)}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    e = np.asarray(out, dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def load_bi():
    df = pd.read_parquet(ROOT / "data" / "snapshot" / "bilingual_zh_vi.parquet").reset_index(drop=True)
    df["title_zh"] = df["title_zh"].fillna("").astype(str)
    df["title_vi"] = df["title_vi"].fillna("").astype(str)
    return df


def check():
    df = load_bi()
    expected = {
        "Gaba Dahongpao tea Габа чай 嘎巴大红袍乌龙茶": 0.9236,
        "凤凰单枞鸭屎香茶叶高端罐装特级老丛地标认证潮州单丛茶散批代发": 0.4748,
        "Minitutu儿童PPSU水杯翻盖奶瓶330mL宝宝学饮杯9个月以上好吸防呛": 0.8540,
        "家城沙嗲素牛肉干怀旧辣条牛肉粒儿时经典校园素肉沙爹素小吃零食": 0.4940,
    }
    for zh, exp in expected.items():
        row = df[df["title_zh"] == zh]
        if row.empty:
            print("not found:", zh[:30]); continue
        vi = row.iloc[0]["title_vi"]
        e = embed([zh, vi])
        print(f"notebook={exp:.4f}  ollama={float(e[0] @ e[1]):.4f}  | {zh[:28]}")


def full(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = load_bi()
    n = len(df)
    print(f"rows: {n}", flush=True)
    t0 = time.time()
    zh = embed(df["title_zh"].tolist())
    vi = embed(df["title_vi"].tolist())
    print(f"embedding done in {time.time() - t0:.0f}s", flush=True)

    df["sim"] = (zh * vi).sum(axis=1)

    rng = np.random.default_rng(42)
    # Baseline 1: random product's Vietnamese title (certainly wrong pair)
    perm = rng.permutation(n)
    clash = perm == np.arange(n)
    perm[clash] = (perm[clash] + 1) % n
    df["sim_rand"] = (zh * vi[perm]).sum(axis=1)
    # Baseline 2 (harder): another product of the SAME category -- shares vocabulary
    same = np.empty(n, dtype=np.int64)
    for cat, idx in df.groupby("category").indices.items():
        p = rng.permutation(idx)
        c = p == idx
        p[c] = np.roll(p, 1)[c]
        same[idx] = p
    df["sim_samecat"] = (zh * vi[same]).sum(axis=1)

    df["cjk_title_vi"] = df["title_vi"].map(lambda x: bool(CJK.search(x)))
    df["cjk_desc_vi"] = df["description_vi"].fillna("").map(lambda x: bool(CJK.search(x)))
    df["cjk_ratio_title_vi"] = df["title_vi"].map(lambda x: len(CJK.findall(x)) / max(len(x), 1))

    q = [.01, .05, .25, .5, .75, .95, .99]
    res = {"n": n, "seconds": round(time.time() - t0)}
    for col in ("sim", "sim_samecat", "sim_rand"):
        d = df[col].describe(percentiles=q)
        res[col] = {k: round(float(v), 4) for k, v in d.items()}
    thr99 = float(df["sim_samecat"].quantile(.99))
    thr95 = float(df["sim_samecat"].quantile(.95))
    res["thr_p99_samecat"] = round(thr99, 4)
    res["thr_p95_samecat"] = round(thr95, 4)
    res["true_below_p99_samecat"] = int((df["sim"] < thr99).sum())
    res["true_below_p95_samecat"] = int((df["sim"] < thr95).sum())
    res["true_below_0.5"] = int((df["sim"] < 0.5).sum())
    res["true_below_0.6"] = int((df["sim"] < 0.6).sum())
    # How separable are right vs wrong pairs? AUC = P(true pair > same-category wrong pair)
    a, b = df["sim"].to_numpy(), df["sim_samecat"].to_numpy()
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().to_numpy()
    res["auc_true_vs_samecat"] = round(float((ranks[:n].sum() - n * (n + 1) / 2) / (n * n)), 4)
    ranks2 = pd.Series(np.concatenate([a, df["sim_rand"].to_numpy()])).rank().to_numpy()
    res["auc_true_vs_rand"] = round(float((ranks2[:n].sum() - n * (n + 1) / 2) / (n * n)), 4)

    per_cat = df.groupby("category").agg(
        n=("sim", "size"), sim_mean=("sim", "mean"), sim_p05=("sim", lambda s: s.quantile(.05)),
        samecat_mean=("sim_samecat", "mean"), samecat_p99=("sim_samecat", lambda s: s.quantile(.99)),
    )
    per_cat["below_own_samecat_p99"] = [
        int((g["sim"] < g["sim_samecat"].quantile(.99)).sum()) for _, g in df.groupby("category")
    ]
    res["per_category"] = per_cat.round(4).reset_index().to_dict(orient="records")

    res["cjk_title_vi"] = int(df["cjk_title_vi"].sum())
    res["cjk_desc_vi"] = int(df["cjk_desc_vi"].sum())
    res["cjk_title_vi_rows"] = df[df["cjk_title_vi"]][["product_id", "category", "title_vi", "sim", "cjk_ratio_title_vi"]] \
        .sort_values("sim", ascending=False).round(4).to_dict(orient="records")
    res["cjk_desc_vi_by_category"] = df[df["cjk_desc_vi"]]["category"].value_counts().to_dict()
    res["sim_mean_cjk_title"] = round(float(df.loc[df["cjk_title_vi"], "sim"].mean()), 4)
    res["sim_mean_clean_title"] = round(float(df.loc[~df["cjk_title_vi"], "sim"].mean()), 4)

    res["lowest_30"] = df.nsmallest(30, "sim")[["product_id", "category", "title_zh", "title_vi", "sim"]].round(4).to_dict(orient="records")
    res["highest_10"] = df.nlargest(10, "sim")[["product_id", "category", "title_zh", "title_vi", "sim"]].round(4).to_dict(orient="records")
    res["lowest_per_category"] = {
        c: g.nsmallest(3, "sim")[["product_id", "title_zh", "title_vi", "sim"]].round(4).to_dict(orient="records")
        for c, g in df.groupby("category")
    }

    cols = ["product_id", "category", "title_zh", "title_vi", "sim", "sim_samecat", "sim_rand", "cjk_title_vi", "cjk_desc_vi"]
    df["below_p99_samecat"] = df["sim"] < thr99
    df[cols + ["below_p99_samecat"]].to_csv(f"{out_dir}/title_similarity_full.csv", index=False, encoding="utf-8-sig")
    with open(f"{out_dir}/title_similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print("DONE", json.dumps({k: res[k] for k in ("n", "seconds", "thr_p99_samecat", "true_below_p99_samecat", "auc_true_vs_samecat")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        full(sys.argv[2])
