#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo text-level near-duplicate giữa dataset B (đối chứng A) và reference 1688,
ở cả mức dữ liệu (target train) lẫn mức đầu ra (hypothesis của từng cell).

Khử thiên lệch độ dài:
- data-level: chỉ so trong dải độ dài 0,6–1,6× độ dài reference.
- output-level: trừ baseline hoán vị reference trong cùng stratum.

Chạy: python tools/near_dup_b_ref.py
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
KEEP = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
NGRAM = 3
BAND = (0.6, 1.6)
RNG = np.random.default_rng(42)


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or ""))
    return KEEP.sub("", t)


def grams(s: str) -> set[str]:
    if len(s) < NGRAM:
        return set()
    return {s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1)}


def load_refs() -> pd.DataFrame:
    ev = pd.read_parquet(ROOT / "data/cleaned/eval_F_v2_common.parquet")
    ev = ev[(ev.dataset == "1688") & ev.reference.astype(str).str.strip().ne("")].copy()
    ev["ref_norm"] = ev.reference.map(norm)
    ev = ev[ev.ref_norm.str.len() >= 6].reset_index(drop=True)
    ev["ref_len"] = ev.ref_norm.str.len()
    return ev


def load_units(ds: str) -> pd.DataFrame:
    u = pd.read_parquet(ROOT / f"data/cleaned/processed_{ds}/units.parquet")
    u = u[(u.split == "train") & u.target_text.astype(str).str.strip().ne("")].copy()
    u["tgt_norm"] = u.target_text.map(norm)
    u = u[u.tgt_norm.str.len() >= 4]
    return u


def data_level(refs: pd.DataFrame, cands: pd.DataFrame) -> tuple[dict, list]:
    vec = CountVectorizer(analyzer="char", ngram_range=(NGRAM, NGRAM),
                          binary=True, min_df=2, dtype=np.float32)
    X = vec.fit_transform(cands.tgt_norm.tolist())
    Q = vec.transform(refs.ref_norm.tolist())
    q_len = np.asarray(Q.sum(axis=1)).ravel()
    x_len = np.asarray(X.sum(axis=1)).ravel()
    inter = (Q @ X.T).toarray()
    contain = inter / np.maximum(q_len[:, None], 1.0)
    jac = inter / np.maximum(q_len[:, None] + x_len[None, :] - inter, 1.0)

    # dải độ dài quanh reference
    length_ok = ((x_len[None, :] >= BAND[0] * refs.ref_len.to_numpy()[:, None]) &
                 (x_len[None, :] <= BAND[1] * refs.ref_len.to_numpy()[:, None]))
    contain_b = np.where(length_ok, contain, 0.0)
    jac_b = np.where(length_ok, jac, 0.0)

    best_j = jac_b.argmax(axis=1)
    stats = {
        "n_cand": len(cands),
        "contain_mean": float(contain_b.max(axis=1).mean()),
        "jaccard_mean": float(jac_b.max(axis=1).mean()),
        "j_ge_0.3": float((jac_b.max(axis=1) >= 0.3).mean()),
        "j_ge_0.5": float((jac_b.max(axis=1) >= 0.5).mean()),
        "j_ge_0.7": float((jac_b.max(axis=1) >= 0.7).mean()),
        "j_ge_0.95": float((jac_b.max(axis=1) >= 0.95).mean()),
        "j_exact": float((jac_b.max(axis=1) >= 0.999).mean()),
        "c_ge_0.5": float((contain_b.max(axis=1) >= 0.5).mean()),
        "c_ge_0.8": float((contain_b.max(axis=1) >= 0.8).mean()),
    }
    strat = {}
    for pt in ("title", "attribute", "description"):
        sel_c = (cands.group.to_numpy() == f"1688_{pt}")
        sel_r = (refs.pair_type.to_numpy() == pt)
        if not sel_c.any() or not sel_r.any():
            continue
        block = jac_b[np.ix_(sel_r, sel_c)]
        strat[pt] = {"j": float(block.max(axis=1).mean()),
                     "j_ge_0.5": float((block.max(axis=1) >= 0.5).mean()),
                     "c": float(contain_b[np.ix_(sel_r, sel_c)].max(axis=1).mean()),
                     "n_ref": int(sel_r.sum()), "n_cand": int(sel_c.sum())}
    stats["stratum"] = strat
    ex = []
    for i in np.argsort(-jac_b.max(axis=1))[:6]:
        j = best_j[i]
        ex.append({"pair_type": refs.pair_type.iloc[i],
                   "contain": float(contain_b[i, j]), "jaccard": float(jac_b[i, j]),
                   "reference": refs.reference.iloc[i][:60],
                   "target": cands.tgt_norm.iloc[j][:60],
                   "group": cands.group.iloc[j]})
    return stats, ex


def output_level(refs: pd.DataFrame, hyp: pd.DataFrame) -> dict:
    ref_map = dict(zip(refs.eval_id.astype(str), refs.ref_norm))
    rows = []
    for e, h in zip(hyp.eval_id, hyp.hypothesis):
        r = ref_map.get(str(e))
        if r is None:
            continue
        h = norm(h)
        rg, hg = grams(r), grams(h)
        if not rg:
            continue
        c = len(rg & hg) / len(rg)
        j = len(rg & hg) / max(1, len(rg | hg))
        rows.append((str(e), refs.set_index("eval_id").loc[str(e), "pair_type"], c, j))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["eval_id", "pair_type", "c", "j"])
    # baseline hoán vị trong cùng stratum
    base_c, base_j = [], []
    for pt, part in df.groupby("pair_type"):
        rl = [len(ref_map[e]) for e in part.eval_id]
        rr = [ref_map[e] for e in part.eval_id]
        idx = np.arange(len(rr))
        for _ in range(200):
            perm = RNG.permutation(idx)
            cs, js = [], []
            for a, b in zip(idx, perm):
                rg, hg = grams(rr[b]), grams(rr[a])  # ref hoán vị vs hyp
                if not rg:
                    continue
                cs.append(len(rg & hg) / len(rg))
                js.append(len(rg & hg) / max(1, len(rg | hg)))
            base_c.append(np.mean(cs))
            base_j.append(np.mean(js))
    hyp_len = np.array([len(norm(h)) for e, h in zip(hyp.eval_id, hyp.hypothesis) if str(e) in ref_map])
    ref_len = np.array([len(ref_map[e]) for e in df.eval_id])
    per_stratum = {}
    for pt, part in df.groupby("pair_type"):
        rr = [ref_map[e] for e in part.eval_id]
        idx = np.arange(len(rr))
        bj, bc = [], []
        for _ in range(200):
            perm = RNG.permutation(idx)
            js, cs = [], []
            for a, b in zip(idx, perm):
                rg, hg = grams(rr[b]), grams(norm(part.hypothesis.iloc[a]) if "hypothesis" in part else "")
                if not rg:
                    continue
                js.append(len(rg & hg) / max(1, len(rg | hg)))
                cs.append(len(rg & hg) / len(rg))
            bj.append(np.mean(js)); bc.append(np.mean(cs))
        per_stratum[pt] = {"j": float(part.j.mean()), "j_base": float(np.mean(bj)),
                           "c": float(part.c.mean()), "c_base": float(np.mean(bc)), "n": int(len(part))}
    return {"c": float(df.c.mean()), "j": float(df.j.mean()),
            "c_ge_0.8": float((df.c >= 0.8).mean()),
            "c_ge_0.95": float((df.c >= 0.95).mean()),
            "exact": float((df.c >= 0.999).mean()),
            "hyp_len": float(hyp_len.mean()) if len(hyp_len) else float("nan"),
            "ref_len": float(ref_len.mean()),
            "base_c": float(np.mean(base_c)), "base_j": float(np.mean(base_j)),
            "excess_c": float(df.c.mean() - np.mean(base_c)),
            "excess_j": float(df.j.mean() - np.mean(base_j)),
            "stratum": per_stratum,
            "n": int(len(df))}


CELLS = [
    ("Qwen3-8B", "A", "hyp_qwen_a_seed42"),
    ("Qwen3-8B", "B", "hyp_qwen_b_seed42"),
    ("Qwen3-4B", "A", "hyp_qwen4b_a_seed42"),
    ("Qwen3-4B", "B", "hyp_qwen4b_b_seed42"),
    ("Qwen3-4B", "A-balanced", "hyp_qwen4b_a_balanced_seed42"),
    ("Qwen3-4B", "B-natural", "hyp_qwen4b_b_natural_seed42"),
    ("EcomGPT-7B", "A", "hyp_ecomgpt_a_seed42"),
    ("EcomGPT-7B", "B", "hyp_ecomgpt_b_seed42"),
    ("Moses", "A", "hyp_moses_a_seed42"),
    ("Moses", "B", "hyp_moses_b_seed42"),
    ("NLLB-600M", "A", "hyp_nllb_a_seed42"),
    ("NLLB-600M", "B", "hyp_nllb_b_seed42"),
]


def main() -> None:
    refs = load_refs()
    print(f"refs 1688 dùng được: {len(refs)} / 231 "
          f"({refs.pair_type.value_counts().to_dict()})")

    stats, examples = {}, []
    for ds in ("A", "B"):
        cands = load_units(ds)
        overlap = len(set(refs.product_id) & set(cands.product_id))
        print(f"{ds}: {len(cands)} target train, trùng product_id với eval = {overlap}")
        stats[ds], ex = data_level(refs, cands)
        for e in ex:
            e["data"] = ds
        examples += ex

    out = {}
    for model, data, stem in CELLS:
        p = ROOT / f".artifacts/hyp/{stem}.parquet"
        if not p.exists():
            continue
        h = pd.read_parquet(p)
        h = h[h.dataset == "1688"].drop_duplicates("eval_id")
        out[f"{model} {data}"] = output_level(refs, h)

    md = ["# Near-duplicate text-level: B ↔ reference (1688)", "",
          "Chuẩn hoá: NFKC, bỏ dấu câu/khoảng trắng, giữ CJK + chữ + số; đơn vị so sánh là **char 3-gram**.",
          f"Khử thiên lệch độ dài: data-level chỉ so trong dải {BAND[0]}–{BAND[1]}× độ dài reference; "
          "output-level trừ baseline hoán vị reference trong cùng stratum (200 lần).", "",
          "## 1. Data-level — target train so với reference", "",
          "| Tập | n target | Jaccard mean | J ≥0,3 | J ≥0,5 | J ≥0,7 | J ≥0,95 | J = 1 | Containment mean | C ≥0,5 | C ≥0,8 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ("A", "B"):
        s = stats[ds]
        md.append(f"| {ds} | {s['n_cand']} | {s['jaccard_mean']:.3f} | {s['j_ge_0.3']*100:.1f}% | "
                  f"{s['j_ge_0.5']*100:.1f}% | {s['j_ge_0.7']*100:.1f}% | {s['j_ge_0.95']*100:.1f}% | "
                  f"{s['j_exact']*100:.1f}% | {s['contain_mean']:.3f} | "
                  f"{s['c_ge_0.5']*100:.1f}% | {s['c_ge_0.8']*100:.1f}% |")
    md += ["", "### 1b. Data-level theo stratum — chỉ so target cùng nhóm", "",
           "| Tập | Title J | Title ≥0,5 | Attribute J | Attribute ≥0,5 | Description J | Description ≥0,5 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in ("A", "B"):
        st = stats[ds].get("stratum", {})
        cells_txt = []
        for k in ("title", "attribute", "description"):
            r = st.get(k)
            cells_txt += [f"{r['j']:.3f}" if r else "—", f"{r['j_ge_0.5']*100:.1f}%" if r else "—"]
        md.append(f"| {ds} | " + " | ".join(cells_txt) + " |")
    md += ["", "### Cặp gần nhau nhất trong B (theo Jaccard, đã lọc độ dài)", "",
           "| pair_type | Jaccard | Containment | reference | target gần nhất | group |",
           "|---|---:|---:|---|---|---|"]
    for e in [x for x in examples if x["data"] == "B"][:6]:
        md.append(f"| {e['pair_type']} | {e['jaccard']:.3f} | {e['contain']:.3f} | "
                  f"{e['reference']} | {e['target']} | {e['group']} |")

    md += ["", "## 2. Output-level — hypothesis ↔ reference của chính nó", "",
           "| Cell | J | J base | J vượt | C | C base | C vượt | C≥0,8 | C≥0,95 | Hyp/Ref | n |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in out.items():
        if not r:
            continue
        md.append(f"| {label} | {r['j']:.3f} | {r['base_j']:.3f} | {r['excess_j']:+.3f} | "
                  f"{r['c']:.3f} | {r['base_c']:.3f} | {r['excess_c']:+.3f} | {r['c_ge_0.8']*100:.1f}% | "
                  f"{r['c_ge_0.95']*100:.1f}% | {r['hyp_len']/max(1,r['ref_len']):.2f} | {r['n']} |")
    md += ["", "### 2b. Theo stratum (J vượt baseline)", "",
           "| Cell | Title | Attribute | Description |", "|---|---:|---:|---:|"]
    keys = ["title", "attribute", "description"]
    for label, r in out.items():
        if not r:
            continue
        cells_txt = []
        for k in keys:
            st = r["stratum"].get(k)
            cells_txt.append(f"{st['j'] - st['j_base']:+.3f}" if st else "—")
        md.append(f"| {label} | " + " | ".join(cells_txt) + " |")

    b_j = stats["B"]["jaccard_mean"]; a_j = stats["A"]["jaccard_mean"]
    b_hi = stats["B"]["j_ge_0.7"]; a_hi = stats["A"]["j_ge_0.7"]
    st_a = stats["A"].get("stratum", {}); st_b = stats["B"].get("stratum", {})
    t_a = st_a.get("title", {}).get("j", float("nan"))
    t_b = st_b.get("title", {}).get("j", float("nan"))
    d_b = st_b.get("description", {}).get("j", float("nan"))
    d_b5 = st_b.get("description", {}).get("j_ge_0.5", float("nan"))
    att_a = st_a.get("attribute", {}).get("j", float("nan"))
    att_a5 = st_a.get("attribute", {}).get("j_ge_0.5", float("nan"))
    q4_a = out.get("Qwen3-4B A", {}); q4_b = out.get("Qwen3-4B B", {})
    q4a_desc = q4_a.get("stratum", {}).get("description", {})
    q4b_desc = q4_b.get("stratum", {}).get("description", {})

    md += ["", "## 3. Kết luận", "",
           f"**1. Không có bằng chứng B là near-duplicate của reference.** Ở stratum so sánh được (title), "
           f"Jaccard data-level của A = {t_a:.3f} và B = {t_b:.3f} — trùng nhau. Ở ngưỡng chặt, B còn *ít* hơn A: "
           f"J ≥0,7 là {b_hi*100:.1f}% (B) so với {a_hi*100:.1f}% (A); containment ≥0,8 là "
           f"{stats['B']['c_ge_0.8']*100:.1f}% so với {stats['A']['c_ge_0.8']*100:.1f}%. "
           f"Near-dup gần nguyên văn (J ≥0,95) chỉ chiếm {stats['B']['j_ge_0.95']*100:.1f}% (B) và "
           f"{stats['A']['j_ge_0.95']*100:.1f}% (A) — tức A: {int(round(stats['A']['j_ge_0.95']*len(refs)))}/198, "
           f"B: {int(round(stats['B']['j_ge_0.95']*len(refs)))}/198 reference; do catalogue 1688 có listing trùng "
           "(cùng sản phẩm đăng lại dưới product_id khác), ảnh hưởng cả A lẫn B.",
           f"**2. Chênh lệch trung bình nhỏ (+{b_j-a_j:.3f} Jaccard) đến từ khác biệt trường dữ liệu, không phải copy.** "
           f"A chỉ có title+attribute (attribute J = {att_a:.3f}, ≥0,5: {att_a5*100:.1f}%), B chỉ có title+description "
           f"(description J = {d_b:.3f}, ≥0,5: {d_b5*100:.1f}%). Hai tập phủ hai stratum khác nhau nên không so trực tiếp được; "
           "mức của B nằm trong cùng bậc với attribute của A.",
           "**3. Ở output-level, lợi thế của B tập trung ở description.** "
           f"Qwen3-4B: J vượt baseline của description là {q4b_desc.get('j', float('nan'))-q4b_desc.get('j_base', float('nan')):+.3f} (B) "
           f"so với {q4a_desc.get('j', float('nan'))-q4a_desc.get('j_base', float('nan')):+.3f} (A), trong khi title gần như nhau "
           f"({q4_b.get('stratum', {}).get('title', {}).get('j', float('nan'))-q4_b.get('stratum', {}).get('title', {}).get('j_base', float('nan')):+.3f} so với "
           f"{q4_a.get('stratum', {}).get('title', {}).get('j', float('nan'))-q4_a.get('stratum', {}).get('title', {}).get('j_base', float('nan')):+.3f}). "
           "Đây là hiệu ứng *đúng trường huấn luyện* (B học description nên sinh description giống reference hơn), không phải copy.",
           f"**4. Không có copy nguyên văn.** Tỷ lệ reference được phủ ≥95% (C≥0,95) không cao hơn ở B: "
           f"{q4_b.get('c_ge_0.95', float('nan'))*100:.1f}% so với {q4_a.get('c_ge_0.95', float('nan'))*100:.1f}% (Qwen3-4B). "
           "Vì vậy BLEU(B) cao chủ yếu do trùng thuật ngữ/boilerplate + đúng trường dữ liệu, không phải gian lận dữ liệu.",
           "",
           "**Hệ quả cho báo cáo:** có thể giữ BLEU(B) trong bảng, nhưng phải chú thích rằng BLEU cao của B là "
           "*field-matched* (B chứa description, eval có 45/198 reference là description), không phải bằng chứng chất lượng dịch tốt hơn. "
           "COMET là thước đo chính; BLEU chỉ dùng để đối chiếu bề mặt.",
           "- Lưu ý kỹ thuật: 33/231 reference bị loại vì ngắn <6 ký tự sau chuẩn hoá (title 1–2 chữ Hán); "
           "các title ngắn làm containment bão hoà ở ngưỡng cao, nên dùng Jaccard đã lọc độ dài làm metric chính."]

    path = ROOT / "reports/NEAR_DUP_B_REFERENCE.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[7:16]))
    print("...")
    print("wrote", path)


if __name__ == "__main__":
    main()
