#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tổng hợp canonical: mean ± std qua seed, coverage, paired cluster bootstrap.

Chỉ COMET-da (1688) là metric xếp hạng; COMET-QE/BLEU/chrF là chẩn đoán.
Chạy: python tools/aggregate_canonical.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import canonical_eval as ce

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"
INDEX = ce.load_index()
CLUSTERS1688 = INDEX[INDEX.dataset == "1688"].set_index("eval_id").product_id.to_dict()

PRETTY = {
    "qwen8b_a": "Qwen3-8B A-natural", "qwen8b_b": "Qwen3-8B B-balanced",
    "qwen8b_a_balanced": "Qwen3-8B A-balanced", "qwen8b_b_natural": "Qwen3-8B B-natural",
    "qwen4b_a": "Qwen3-4B A-natural", "qwen4b_b": "Qwen3-4B B-balanced",
    "qwen4b_a_balanced": "Qwen3-4B A-balanced", "qwen4b_b_natural": "Qwen3-4B B-natural",
    "ecomgpt_a": "EcomGPT-7B A-natural", "ecomgpt_b": "EcomGPT-7B B-balanced",
    "ecomgpt_a_balanced": "EcomGPT-7B A-balanced",
    "moses_a": "Moses A", "moses_b": "Moses B",
    "onmt_a": "OpenNMT A", "onmt_b": "OpenNMT B-natural",
    "nllb_a": "NLLB-600M A", "nllb_b": "NLLB-600M B",
}
CONTRASTS = [
    ("qwen8b_a", "qwen8b_b"), ("qwen8b_a", "qwen8b_a_balanced"), ("qwen8b_b", "qwen8b_b_natural"),
    ("qwen4b_a", "qwen4b_b"), ("qwen4b_a", "qwen4b_a_balanced"), ("qwen4b_b", "qwen4b_b_natural"),
    ("qwen4b_a", "qwen8b_a"), ("qwen4b_b", "qwen8b_b"),
    ("qwen4b_a", "ecomgpt_a"), ("qwen4b_b", "ecomgpt_b"),
    ("nllb_a", "nllb_b"), ("onmt_a", "onmt_b"),
    ("qwen4b_a", "nllb_a"), ("qwen4b_b", "nllb_b"),
]


def boot_diff(diff: np.ndarray, clusters: np.ndarray, n: int = 2000, seed: int = 42) -> dict:
    order = np.argsort(clusters, kind="stable")
    diff, clusters = diff[order], clusters[order]
    _, starts = np.unique(clusters, return_index=True)
    ends = np.r_[starts[1:], len(diff)]
    sums = np.array([diff[s:e].sum() for s, e in zip(starts, ends)])
    lens = np.array([e - s for s, e in zip(starts, ends)], float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(sums), len(sums))
        means[i] = sums[idx].sum() / max(1.0, lens[idx].sum())
    return {"mean": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(diff))}


def main() -> None:
    files = sorted(CANON.glob("*/seed*.parquet"))
    per_seed: dict[str, dict[int, pd.DataFrame]] = {}
    for f in files:
        cell = f.parent.name
        seed = int(f.stem.replace("seed", ""))
        per_seed.setdefault(cell, {})[seed] = pd.read_parquet(f)

    rows = []
    for cell, seeds in sorted(per_seed.items()):
        da, qe, cov1, covt, bleu, chrf = [], [], [], [], [], []
        mac = {"1688_unw": [], "1688_w": [], "tiki_unw": [], "tiki_w": []}
        for s, df in seeds.items():
            a = df[df.dataset == "1688"]
            t = df[df.dataset == "tiki"]
            da.append(a.comet_da.mean()); qe.append(t.comet_qe.mean())
            cov1.append(a.nonempty.mean()); covt.append(t.nonempty.mean())
            bleu.append(a.bleu.mean()); chrf.append(a.chrf.mean())
            for ds, part, col in (("1688", a, "comet_da"), ("tiki", t, "comet_qe")):
                means = {st: part[part.stratum == st][col].mean()
                         for st in ("title", "attribute", "description")}
                cnt = part.stratum.value_counts()
                mac[f"{ds}_unw"].append(float(np.nanmean(list(means.values()))))
                mac[f"{ds}_w"].append(float(sum(means[k] * cnt.get(k, 0) for k in means) / max(1, cnt.sum())))
        rows.append({
            "cell": cell, "label": PRETTY.get(cell, cell), "seeds": ",".join(map(str, sorted(seeds))),
            "n_seeds": len(seeds),
            "da_mean": float(np.nanmean(da)), "da_std": float(np.nanstd(da, ddof=1)) if len(da) > 1 else float("nan"),
            "qe_mean": float(np.nanmean(qe)), "qe_std": float(np.nanstd(qe, ddof=1)) if len(qe) > 1 else float("nan"),
            "cov1688": float(np.mean(cov1)), "covtiki": float(np.mean(covt)),
            "bleu": float(np.nanmean(bleu)), "chrf": float(np.nanmean(chrf)),
            "macro_unw_1688": float(np.nanmean(mac["1688_unw"])),
            "macro_w_1688": float(np.nanmean(mac["1688_w"])),
            "macro_unw_tiki": float(np.nanmean(mac["tiki_unw"])),
            "macro_w_tiki": float(np.nanmean(mac["tiki_w"])),
        })
    agg = pd.DataFrame(rows)
    missing_cells = agg[agg.da_mean.isna()].label.tolist()
    agg = agg[agg.da_mean.notna()].sort_values("da_mean", ascending=False)

    # per-row trung bình qua seed (paired theo eval_id)
    def seed_avg(cell: str) -> pd.Series:
        dfs = [df[df.dataset == "1688"].set_index("eval_id").comet_da
               for df in per_seed[cell].values()]
        return pd.concat(dfs, axis=1).mean(axis=1)

    contrasts = []
    for a, b in CONTRASTS:
        if a not in per_seed or b not in per_seed:
            continue
        sa, sb = seed_avg(a), seed_avg(b)
        if sa.isna().all() or sb.isna().all():
            continue
        common = sa.index.intersection(sb.index)
        diff = (sa.loc[common] - sb.loc[common]).to_numpy()
        clusters = np.array([CLUSTERS1688[i] for i in common])
        r = boot_diff(diff, clusters)
        contrasts.append({"a": PRETTY.get(a, a), "b": PRETTY.get(b, b), **r})

    def f(v, nd=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

    md = ["# Canonical eval — 858 dòng cố định, zero-fill + coverage", "",
          "Index: `data/cleaned/eval_paired_858.parquet` (1688 = 231, Tiki = 627, hash trong `.json`).",
          "Quy ước: join theo `eval_id`; hyp rỗng ⇒ 0 điểm, **không lọc dòng**; coverage = % hyp không rỗng.",
          "Metric xếp hạng: **COMET-da (1688)**. COMET-QE (Tiki), BLEU, chrF chỉ là chẩn đoán.",
          "Human eval Tiki: **MISSING** (chưa có file annotation).", "",
          "## 1. Tổng hợp theo cell (mean ± std qua seed)", "",
          "| Cell | Seeds | #seed | COMET-da 1688 | Coverage 1688 | COMET-QE Tiki | Coverage Tiki | BLEU (sent) | chrF (sent) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in agg.itertuples():
        std = "" if np.isnan(r.da_std) else f" ± {r.da_std:.4f}"
        qstd = "" if np.isnan(r.qe_std) else f" ± {r.qe_std:.4f}"
        md.append(f"| {r.label} | {r.seeds} | {r.n_seeds} | {r.da_mean:.4f}{std} | {r.cov1688*100:.1f}% | "
                  f"{r.qe_mean:.4f}{qstd} | {r.covtiki*100:.1f}% | {r.bleu:.2f} | {r.chrf:.2f} |")

    md += ["", "### 1b. Macro theo stratum", "",
           "`macro_unw` = trung bình 3 stratum (mỗi stratum trọng số bằng nhau); "
           "`macro_w` = trọng số theo số dòng (bằng mean tổng).", "",
           "| Cell | macro_unw 1688 | macro_w 1688 | macro_unw Tiki | macro_w Tiki |",
           "|---|---:|---:|---:|---:|"]
    for r in agg.itertuples():
        md.append(f"| {r.label} | {r.macro_unw_1688:.4f} | {r.macro_w_1688:.4f} | "
                  f"{r.macro_unw_tiki:.4f} | {r.macro_w_tiki:.4f} |")

    md += ["", "## 2. Paired cluster bootstrap (COMET-da 1688, 2.000 lần, cluster = product_id)", "",
           "| A | B | Δ (A−B) | 95% CI | n |", "|---|---|---:|---|---:|"]
    for c in contrasts:
        sig = "*" if (c["lo"] > 0 or c["hi"] < 0) else ""
        md.append(f"| {c['a']} | {c['b']} | {c['mean']:+.4f}{sig} | "
                  f"[{c['lo']:+.4f}; {c['hi']:+.4f}] | {c['n']} |")

    md += ["", "## 3. Vi phạm / khoảng trống so với policy", ""]
    few = agg[agg.n_seeds < 5]
    if missing_cells:
        md.append("- **Không có score từng dòng (loại khỏi bảng xếp hạng):** " + ", ".join(missing_cells) + ".")
    md.append(f"- **Seed:** {len(few)}/{len(agg)} cell có <5 seed. Các cell đã có ≥2 seed: "
              + ", ".join(f"{r.label} ({r.n_seeds})" for r in agg.itertuples() if r.n_seeds >= 2) + ".")
    md.append("- **Thiếu score từng dòng:** " +
              (", ".join(missing_cells) if missing_cells else "không còn cell nào") + ".")
    if (agg.cov1688 < 1).any() or (agg.covtiki < 1).any():
        low = [r for r in agg.itertuples() if r.cov1688 < 1 or r.covtiki < 1]
        md.append(f"- **Coverage <100%:** {len(low)} cell (Qwen 98,6% do 9 description dài; "
                  "EcomGPT B 97,1%; NLLB B 99,8%) — đã zero-fill, chi tiết trong artefact.")
    md.append("- **Human eval Tiki:** chưa có; bảng xếp hạng hiện chỉ dựa COMET-da 1688.")
    md.append("- **9 dòng description dài (3,0–3,7k ký tự) bị Qwen bỏ vì prompt >952 token** "
              "(guard `max_seq=960`), không phải model fail — xem `reports/AUDIT_EMPTY_QWEN4B.md`.")
    md.append("- **NLLB B:** seed 42 = 0,5494 vs seed 7 = 0,6382 (Δ −0,0888 [−0,1076; −0,0698]); "
              "run hợp lệ, đây là run-variance thật — xem `reports/AUDIT_NLLB_B_SEED.md`.")
    md.append("- **Nguồn gốc:** các artefact này là retrofit từ score cũ; run mới phải dùng "
              "`tools/score_canonical.py` để log manifest đầy đủ.")

    path = ROOT / "reports/CANONICAL_EVAL.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:24]))
    print("...")
    print("wrote", path)


if __name__ == "__main__":
    main()
