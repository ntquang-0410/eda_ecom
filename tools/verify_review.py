#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit tái lập cho 3 phản biện (data leakage, OpenNMT fairness, quota/length).

Sinh:
  - review_audit.json
  - data/cleaned/eval_F_v2.parquet  (disjoint theo product/source/pair/group/near-dup)
  - data/cleaned/eval_F_v2_meta.json
  - data/cleaned/units_real_tokens_{A,B}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = [("1688", "title"), ("1688", "attribute"), ("1688", "description"),
          ("tiki", "title"), ("tiki", "attribute"), ("tiki", "description")]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).casefold()
    return re.sub(r"\s+", " ", s).strip()


def strong(s: str) -> str:
    return re.sub(r"[\W_]+", "", norm(s), flags=re.UNICODE)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["src_norm"] = df.source_text.map(norm)
    df["src_strong"] = df.source_text.map(strong)
    df["src_hash"] = df.src_norm.map(sha)
    df["src_strong_hash"] = df.src_strong.map(sha)
    df["pair_hash"] = [sha(norm(s) + "\x1f" + norm(t))
                       for s, t in zip(df.source_text, df.target_text)]
    df["product_id_s"] = df.product_id.astype(str)
    return df


def tiki_title_map(root: Path) -> dict[str, str]:
    d = pd.read_parquet(root / "data/cleaned/tiki_vi_cado.parquet",
                        columns=["product_id", "title_vi_cleaned"])
    d = d.drop_duplicates("product_id")
    return {str(r.product_id): norm(r.title_vi_cleaned) for r in d.itertuples()}


def nhom_trung_map(root: Path) -> dict[str, int]:
    p = root / "data/upstream/1688_description_pairs_zh_vi.parquet"
    d = pd.read_parquet(p, columns=["product_id", "nhom_trung"]).drop_duplicates("product_id")
    return {str(r.product_id): int(r.nhom_trung) for r in d.itertuples()}


def group_ids(df: pd.DataFrame, g1688: dict[str, int], tiki_titles: dict[str, str]) -> pd.Series:
    out = []
    domains = df["domain"] if "domain" in df.columns else df["dataset"]
    for pid, domain in zip(df.product_id_s, domains):
        pid = str(pid)
        if domain == "tiki":
            out.append("TIKI:" + sha(tiki_titles.get(pid, pid)))
        else:
            g = g1688.get(pid)
            out.append(f"NHOM:{g}" if g is not None else "PID:" + pid)
    return pd.Series(out, index=df.index)


def near_dup_flags(train: pd.DataFrame, ev: pd.DataFrame, threshold=0.8,
                   num_perm=32) -> pd.Series:
    """MinHash LSH 5-char shingles; trả về Series bool theo index eval."""
    from datasketch import MinHash, MinHashLSH

    def shingles(text: str) -> set[str]:
        s = re.sub(r"\s+", "", norm(text))
        if len(s) < 5:
            return {s}
        return {s[i:i + 5] for i in range(len(s) - 4)}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, text in enumerate(train.source_text):
        m = MinHash(num_perm=num_perm)
        for sh in shingles(text):
            m.update(sh.encode("utf-8"))
        lsh.insert(str(i), m)
    hits = []
    for text in ev.source_text:
        m = MinHash(num_perm=num_perm)
        for sh in shingles(text):
            m.update(sh.encode("utf-8"))
        hits.append(bool(lsh.query(m)))
    return pd.Series(hits, index=ev.index)


def real_tokens(df: pd.DataFrame, tokenizer) -> np.ndarray:
    src = [len(tokenizer(str(x), add_special_tokens=False).input_ids) for x in df.source_text]
    tgt = [len(tokenizer(str(x), add_special_tokens=False).input_ids) for x in df.target_text]
    return np.asarray(src) + np.asarray(tgt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--base", default=".artifacts/matrix_out/m4b_a/matrix_data")
    ap.add_argument("--scores", default=".artifacts/matrix_out/comet5/scores")
    ap.add_argument("--out", default="review_audit.json")
    ap.add_argument("--no-near-dup", action="store_true")
    ap.add_argument("--no-real-tokens", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base = root / args.base
    scores_dir = root / args.scores
    t0 = time.time()

    ev = pd.read_parquet(base / "eval_A.parquet")
    ev_b = pd.read_parquet(base / "eval_B.parquet")
    units = {ds: pd.read_parquet(base / f"processed_{ds}/units.parquet") for ds in ("A", "B")}
    ev = add_hashes(ev)
    ev_b = add_hashes(ev_b)
    assert ev.eval_id.tolist() == ev_b.eval_id.tolist()
    for ds in ("A", "B"):
        units[ds] = add_hashes(units[ds])

    g1688 = nhom_trung_map(root)
    tiki_titles = tiki_title_map(root)
    for df in [ev] + list(units.values()):
        df["group_id"] = group_ids(df, g1688, tiki_titles)

    # ---------- leakage criteria ----------
    leak = pd.DataFrame(index=ev.index)
    audit = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "dataset_hashes": {
                 "eval_A": file_sha(base / "eval_A.parquet"),
                 "eval_B": file_sha(base / "eval_B.parquet"),
                 "units_A": file_sha(base / "processed_A/units.parquet"),
                 "units_B": file_sha(base / "processed_B/units.parquet"),
             },
             "leakage": {}, "near_dup": {}, "prefix": {}}

    per_ds_flags = {}
    for ds in ("A", "B"):
        train = units[ds][units[ds].split.isin(["train", "dev"])]
        sets = {
            "product_id": set(train.product_id_s),
            "source_hash": set(train.src_hash),
            "source_hash_strong": set(train.src_strong_hash),
            "pair_hash": set(train.pair_hash),
            "near_dup_cluster_id": set(train.group_id),
        }
        ev_ds = ev.copy()
        flags = pd.DataFrame(index=ev_ds.index)
        flags["product_id"] = ev_ds.product_id_s.isin(sets["product_id"])
        flags["source_hash"] = ev_ds.src_hash.isin(sets["source_hash"])
        flags["source_hash_strong"] = ev_ds.src_strong_hash.isin(sets["source_hash_strong"])
        flags["pair_hash"] = ev_ds.pair_hash.isin(sets["pair_hash"])
        flags["near_dup_cluster_id"] = ev_ds.group_id.isin(sets["near_dup_cluster_id"])
        flags["leak"] = flags.any(axis=1)
        per_ds_flags[ds] = flags

        by = {}
        for pt in ("title", "attribute", "description"):
            m = ev_ds.pair_type == pt
            by[pt] = {"n": int(m.sum()),
                      **{c: int(flags.loc[m, c].sum()) for c in flags.columns}}
        audit["leakage"][ds] = {
            "n": int(len(ev_ds)),
            "criteria": {c: int(flags[c].sum()) for c in flags.columns},
            "leaked": int(flags["leak"].sum()),
            "clean": int((~flags["leak"]).sum()),
            "by_stratum": by,
        }
        target = [(d, pt) for d, pt in STRATA if d == ("1688" if ds == "A" else "tiki")]
        # per dataset x stratum
        by2 = {}
        for d in ("1688", "tiki"):
            for pt in ("title", "attribute", "description"):
                m = (ev_ds.dataset == d) & (ev_ds.pair_type == pt)
                by2[f"{d}/{pt}"] = {"n": int(m.sum()), "leak": int(flags.loc[m, "leak"].sum())}
        audit["leakage"][ds]["by_dataset_stratum"] = by2

    # ---------- near-dup on residual rows ----------
    near = pd.Series(False, index=ev.index)
    near_counts = {}
    near_cache = root / "data/cleaned/eval_near_dup_flags.parquet"
    if near_cache.exists() and not args.no_near_dup:
        nc = pd.read_parquet(near_cache).set_index("eval_id").near
        near = ev.eval_id.map(nc).fillna(False).astype(bool)
        near.index = ev.index
        near_counts = {"loaded_from": str(near_cache), "hits": int(near.sum())}
        print(f"near-dup loaded {int(near.sum())} hits", flush=True)
    elif not args.no_near_dup:
        for ds in ("A", "B"):
            residual = per_ds_flags[ds]["leak"] == False
            train = units[ds][units[ds].split.isin(["train", "dev"])]
            for pt in ("title", "attribute", "description"):
                ei = ev.index[residual & (ev.pair_type == pt)]
                ti = train.index[train.pair_type == pt]
                if len(ei) == 0 or len(ti) == 0:
                    continue
                hit = near_dup_flags(train.loc[ti], ev.loc[ei])
                near.loc[ei] = near.loc[ei] | hit.to_numpy()
                near_counts[f"{ds}/{pt}"] = {"queries": int(len(ei)),
                                             "index": int(len(ti)),
                                             "hits": int(hit.sum())}
                print(f"near-dup {ds}/{pt}: {len(ei)} q / {len(ti)} idx -> {int(hit.sum())} hits", flush=True)
        pd.DataFrame({"eval_id": ev.eval_id, "near": near.to_numpy()}).to_parquet(near_cache, index=False)
        audit["near_dup"] = near_counts

    # ---------- union v2 ----------
    leaked_union = per_ds_flags["A"]["leak"] | per_ds_flags["B"]["leak"] | near
    v2 = ev.loc[~leaked_union].copy()
    v2["leak_A"] = per_ds_flags["A"]["leak"].loc[v2.index]
    v2["leak_B"] = per_ds_flags["B"]["leak"].loc[v2.index]
    v2["near_dup"] = near.loc[v2.index]
    v2_out = v2[["eval_id", "dataset", "pair_type", "product_id", "source", "reference",
                 "source_text", "target_text", "max_new"]].copy()
    v2_out.to_parquet(root / "data/cleaned/eval_F_v2.parquet", index=False)
    # fail cứng: không được còn bất kỳ khoá nào trùng train+dev
    for ds in ("A", "B"):
        train = units[ds][units[ds].split.isin(["train", "dev"])]
        v2h = add_hashes(v2)
        v2h["group_id"] = group_ids(v2h, g1688, tiki_titles)
        bad = (v2h.src_hash.isin(set(train.src_hash))
               | v2h.src_strong_hash.isin(set(train.src_strong_hash))
               | v2h.pair_hash.isin(set(train.pair_hash))
               | v2h.product_id_s.isin(set(train.product_id_s))
               | v2h.group_id.isin(set(train.group_id))
               | near.loc[v2h.index])
        assert not bad.any(), f"eval_F_v2 còn leak ở dataset {ds}: {int(bad.sum())}"

    by_stratum = {}
    for d, pt in STRATA:
        m = (v2.dataset == d) & (v2.pair_type == pt)
        by_stratum[f"{d}/{pt}"] = int(m.sum())
    meta = {"n": int(len(v2)), "by_stratum": by_stratum,
            "leaked_union": int(leaked_union.sum()),
            "leaked_A": int(per_ds_flags["A"]["leak"].sum()),
            "leaked_B": int(per_ds_flags["B"]["leak"].sum()),
            "near_dup_extra": int(near.sum()),
            "criteria": ["product_id", "source_hash", "source_hash_strong",
                         "pair_hash", "near_dup_cluster_id"],
            "normalization": "NFKC + casefold + whitespace collapse; strong = bỏ punctuation/space",
            "source": str((root / "data/cleaned/eval_F_v2.parquet").relative_to(root))}
    (root / "data/cleaned/eval_F_v2_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit["eval_F_v2"] = meta

    # ---------- prefix ----------
    prefix = ev.eval_id.str.split(":").str[0].value_counts().to_dict()
    audit["prefix"] = {"eval_id_prefix": {str(k): int(v) for k, v in prefix.items()},
                       "n_prefix": len(prefix)}

    # ---------- dedup file A ----------
    ded = pd.read_parquet(root / "data/cleaned/weight_splits_v2/A_title_attribute.parquet",
                          columns=["source_text", "target_text", "split"])
    ded["pair"] = [sha(norm(s) + "\x1f" + norm(t)) for s, t in zip(ded.source_text, ded.target_text)]
    g = ded.groupby("pair")["split"].nunique()
    cross = g[g > 1]
    dup_rows = int(ded.duplicated("pair", keep="first").sum())
    audit["dedup_file_A"] = {"rows": int(len(ded)),
                             "cross_split_groups": int(len(cross)),
                             "duplicate_rows_beyond_first": dup_rows,
                             "expected_review": {"groups": 583, "rows": 3191}}

    # ---------- quota / exposure ----------
    def quota_one(ds: str, sample_path: Path):
        u = units[ds]
        s = pd.read_parquet(sample_path)
        # sample dùng row_index (0-based) cho cả A/B
        s = s.merge(u.reset_index()[["index", "product_id_s", "group", "tokens_est",
                                     "src_chars", "tgt_chars", "source_text", "target_text"]],
                    left_on="row_index", right_on="index", how="left", suffixes=("", "_u"))
        s["group"] = s["group_u"] if "group_u" in s else s["group"]
        s["tokens_uncapped"] = s.src_chars / 3.6 + s.tgt_chars / 1.6
        out = {}
        for g, d in s.groupby("group"):
            pop = u[u.group == g]
            out[str(g)] = {
                "population_rows": int(len(pop)),
                "population_tokens_est_M": round(float(pop.tokens_est.sum()) / 1e6, 4),
                "population_tokens_uncapped_M": round(float((pop.src_chars / 3.6 + pop.tgt_chars / 1.6).sum()) / 1e6, 4),
                "draws": int(len(d)),
                "unique_rows": int(d.row_index.nunique()),
                "coverage_pct": round(100 * d.row_index.nunique() / max(1, len(pop)), 3),
                "epochs_token_ratio": round(float(d.eff_tokens.sum()) / max(1e-9, float(pop.eff_tokens.sum())), 4),
                "draws_per_row": round(len(d) / max(1, d.row_index.nunique()), 4),
                "rows_per_step": round(len(d) / 500, 2),
                "truncated_pct": round(100 * float((d.tokens_uncapped > 960).mean()), 2),
                "median_tokens_est": round(float(d.tokens_est.median()), 2),
            }
        return {"sample_rows": int(len(s)), "unique_rows": int(s.row_index.nunique()),
                "groups": out,
                "total_draws": int(len(s)),
                "total_unique": int(s.row_index.nunique())}

    audit["quota"] = {
        "A": quota_one("A", root / ".artifacts/matrix_out/m4b_a/out_A_m4b_natural/sample_natural_seed42.parquet"),
        "B": quota_one("B", root / ".artifacts/matrix_out/m4b_b/out_B_m4b_balanced/sample_balanced_seed42.parquet"),
        "accounting": {
            "tokens_est_formula": "src_chars/3.6 + tgt_chars/1.6",
            "eff_tokens_cap": 960,
            "note": "token ước lượng theo ký tự, không phải tokenizer thật; cap 960 làm mất khối lượng đuôi dài",
        },
    }

    # real tokenizer accounting
    if not args.no_real_tokens:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
            for ds in ("A", "B"):
                u = units[ds].copy()
                cache = root / f"data/cleaned/units_real_tokens_{ds}.parquet"
                if cache.exists():
                    cached = pd.read_parquet(cache)
                    u["real_tokens"] = u.pair_id.map(cached.set_index("pair_id").real_tokens)
                    if u.real_tokens.isna().any():
                        u["real_tokens"] = real_tokens(u, tok)
                else:
                    u["real_tokens"] = real_tokens(u, tok)
                u[["pair_id", "group", "product_id_s", "real_tokens"]].to_parquet(
                    root / f"data/cleaned/units_real_tokens_{ds}.parquet", index=False)
                sample_path = (root / ".artifacts/matrix_out/m4b_a/out_A_m4b_natural/sample_natural_seed42.parquet"
                               if ds == "A" else
                               root / ".artifacts/matrix_out/m4b_b/out_B_m4b_balanced/sample_balanced_seed42.parquet")
                s = pd.read_parquet(sample_path)
                s = s.merge(u.reset_index()[["index", "real_tokens"]], left_on="row_index",
                            right_on="index", how="left")
                audit["quota"][ds]["real_qwen_tokens"] = {
                    "sample_total_M": round(float(s.real_tokens.sum()) / 1e6, 4),
                    "population_total_M": round(float(u.real_tokens.sum()) / 1e6, 4),
                    "epochs_real": round(float(s.real_tokens.sum()) / max(1e-9, float(u.real_tokens.sum())), 4),
                }
                print(f"real tokens {ds}: sample {audit['quota'][ds]['real_qwen_tokens']}", flush=True)
        except Exception as e:  # pragma: no cover
            audit["quota"]["real_tokenizer_error"] = f"{type(e).__name__}: {e}"

    # ---------- score coverage + common subset ----------
    pat = re.compile(r"scores_(?P<label>.+)_seed42_(?P<ds>1688|tiki)\.parquet")
    score_files = {}
    for p in sorted(scores_dir.glob("scores_*.parquet")):
        m = pat.match(p.name)
        if m:
            d = pd.read_parquet(p)
            d["eval_id"] = d.eval_id.astype(str)
            score_files[(m.group("label"), m.group("ds"))] = d.set_index("eval_id").score
    labels = sorted({k[0] for k in score_files})
    coverage = {}
    for (lab, ds), s in score_files.items():
        part = ev[ev.dataset == ds].set_index("eval_id")
        coverage[f"{lab}::{ds}"] = {
            "n": int(s.notna().sum()),
            "n_v2": int(len(set(s.index) & set(v2.eval_id))),
            "by_stratum": {pt: int(((part.pair_type == pt) & part.index.isin(s.index)).sum())
                           for pt in ("title", "attribute", "description")},
        }
    audit["score_coverage"] = coverage

    # common per dataset across all labels
    common = {}
    for ds in ("1688", "tiki"):
        ids = None
        for lab in labels:
            s = score_files.get((lab, ds))
            if s is None:
                continue
            ids = set(s.index) if ids is None else ids & set(s.index)
        ids = (ids or set()) & set(v2.eval_id)
        common[ds] = ids
    audit["common_eval_ids"] = {ds: int(len(ids)) for ds, ids in common.items()}
    common_ids = common["1688"] | common["tiki"]
    ev[ev.eval_id.isin(common_ids)][["eval_id", "dataset", "pair_type", "product_id",
                                     "source", "reference", "source_text", "target_text",
                                     "max_new"]].to_parquet(
        root / "data/cleaned/eval_F_v2_common.parquet", index=False)
    common_scores = {}
    for lab in labels:
        for ds in ("1688", "tiki"):
            s = score_files.get((lab, ds))
            if s is None:
                continue
            common_scores[f"{lab}::{ds}"] = round(float(s[s.index.isin(common[ds])].mean()), 4)
    audit["common_scores"] = common_scores

    # Tiki description common subset (review claim)
    tiki_desc_ids = set(ev[(ev.dataset == "tiki") & (ev.pair_type == "description")].eval_id) & common["tiki"]
    tiki_desc = {}
    for lab in labels:
        s = score_files.get((lab, "tiki"))
        if s is None:
            continue
        all_ids = set(ev[(ev.dataset == "tiki") & (ev.pair_type == "description")].eval_id) & set(s.index)
        tiki_desc[lab] = {
            "n_all": int(len(all_ids)),
            "full": round(float(s[s.index.isin(all_ids)].mean()), 4),
            "n_common": int(len(tiki_desc_ids)),
            "common": round(float(s[s.index.isin(tiki_desc_ids)].mean()), 4),
        }
    audit["tiki_description_common_subset"] = tiki_desc

    # leakage score sensitivity per cell
    leak_sens = {}
    for (lab, ds), s in score_files.items():
        part = ev[ev.dataset == ds].set_index("eval_id").loc[s.index]
        leak_by_id = pd.Series(leaked_union.to_numpy(), index=ev.eval_id)
        lk = leak_by_id.reindex(part.index).fillna(False).to_numpy()
        if lk.sum() and (~lk).sum():
            leak_sens[f"{lab}::{ds}"] = {
                "n_leak": int(lk.sum()), "n_clean": int((~lk).sum()),
                "leak_score": round(float(s[lk].mean()), 4),
                "clean_score": round(float(s[~lk].mean()), 4),
                "delta": round(float(s[~lk].mean() - s[lk].mean()), 4),
            }
    audit["leak_score_sensitivity"] = leak_sens

    # ---------- OpenNMT fairness ----------
    st_path = root / ".artifacts/matrix_out/onmt3/onmt_data_stats.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {}
    params = 2 * 32000 * 256 + 4 * (4 * 256 * 256 + 2 * 256 * 1024 + 4 * 256 * 256) + 2 * 32000 * 256
    audit["opennmt"] = {
        "config": {"d_model": 256, "layers": 4, "ff": 1024, "vocab": 32000, "params_estimate_M": round(params / 1e6, 1)},
        "data_stats": st,
        "token_pass": {"A": 30000 * 8192, "B": 22990 * 8192},
        "llm_token_pass": {"A": 4000038, "B": 4002207},
        "ratios": {"A": round(30000 * 8192 / 4000038, 1), "B": round(22990 * 8192 / 4002207, 1)},
        "B_label_issue": {
            "nmt_uses": "units_B toàn bộ split train (B-natural): 49.774 dòng / 18,83M token",
            "llm_uses": "quota balanced 4M: 52.131 draw / 25.453 unique",
            "conclusion": "cell OpenNMT-B hiện là B-natural, không phải B-balanced",
        },
        "eval_src_over_512": st.get("A", {}).get("eval_src_over_512"),
    }

    (root / args.out).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("eval_F_v2", "leakage", "common_eval_ids")},
                     ensure_ascii=False, indent=2))
    print("wrote", root / args.out, "in", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
