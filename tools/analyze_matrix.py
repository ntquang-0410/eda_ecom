#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng kết quả matrix 4 model × 2 data từ score COMET."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

PAT = re.compile(r"scores_(?P<label>.+)_seed42_(?P<ds>1688|tiki)\.parquet")
STRATA = [("1688", "title"), ("1688", "attribute"), ("1688", "description"),
          ("tiki", "title"), ("tiki", "attribute"), ("tiki", "description")]


def load(scores_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(scores_dir.glob("scores_*.parquet")):
        m = PAT.match(p.name)
        if not m:
            continue
        d = pd.read_parquet(p)
        d["label"] = m.group("label"); d["ds"] = m.group("ds")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["eval_id"] = df["eval_id"].astype(str)
    return df


def boot(diff, clusters, n=2000, seed=42):
    order = np.argsort(clusters, kind="stable"); diff, clusters = diff[order], clusters[order]
    _, starts = np.unique(clusters, return_index=True); ends = np.r_[starts[1:], len(diff)]
    groups = [diff[s:e] for s, e in zip(starts, ends)]
    lens = np.array([len(g) for g in groups], float); sums = np.array([g.sum() for g in groups], float)
    rng = np.random.default_rng(seed); means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        means[i] = sums[idx].sum() / max(1.0, lens[idx].sum())
    return {"mean": round(float(diff.mean()), 4), "lo": round(float(np.percentile(means, 2.5)), 4),
            "hi": round(float(np.percentile(means, 97.5)), 4), "n": int(len(diff))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out", default="reports/MATRIX_4X2_RESULTS.md")
    ap.add_argument("--baseline", default="qwen_a")
    args = ap.parse_args()
    root = Path(".")
    df = load(Path(args.scores_dir))
    products = pd.concat([
        pd.read_parquet(root / "data/cleaned/weight_splits_v2/eval_A.parquet")[["eval_id", "product_id"]],
        pd.read_parquet(root / "data/cleaned/weight_splits_v2/eval_B.parquet")[["eval_id", "product_id"]],
    ]).drop_duplicates("eval_id")
    products["eval_id"] = products["eval_id"].astype(str)
    df = df.merge(products, on="eval_id", how="left")

    labels = sorted(df.label.unique())
    lines = ["# Matrix 4 model × 2 data — COMET", "",
             "Eval: `eval_F` 1.500 dòng. 1688 = COMET-da, Tiki = COMET-QE. "
             "BLEU/chrF tính trên phần 1688 có reference.", ""]
    lines += ["## Điểm theo model × data", "",
              "| cell | 1688 da | Tiki QE | BLEU 1688 | chrF 1688 | n 1688 | n Tiki |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        a = df[(df.label == lab) & (df.ds == "1688")]
        t = df[(df.label == lab) & (df.ds == "tiki")]
        bleu = chrf = float("nan")
        hp = root / ".artifacts" / "hyp" / f"hyp_{lab}_seed42.parquet"
        if not hp.exists():
            hp = next((x for x in root.glob(f".artifacts/matrix_out/**/hyp_{lab}_seed42.parquet")), None)
        if hp is not None:
            h = pd.read_parquet(hp)
            h = h[(h.dataset == "1688") & h.reference.fillna("").str.strip().ne("") & h.hypothesis.fillna("").str.strip().ne("")]
            if len(h):
                bleu = sacrebleu.corpus_bleu(h.hypothesis.tolist(), [h.reference.tolist()], tokenize="zh").score
                chrf = sacrebleu.corpus_chrf(h.hypothesis.tolist(), [h.reference.tolist()]).score
        b = f"{bleu:.2f}" if bleu == bleu else "-"
        c = f"{chrf:.2f}" if chrf == chrf else "-"
        lines.append(f"| {lab} | {a.score.mean():.4f} | {t.score.mean():.4f} | {b} | {c} | {len(a)} | {len(t)} |")

    lines += ["", "## Theo stratum", "",
              "| cell | " + " | ".join(f"{ds}_{pt}" for ds, pt in STRATA) + " |",
              "|---|" + "---:|" * len(STRATA)]
    for lab in labels:
        cells = []
        for ds, pt in STRATA:
            part = df[(df.label == lab) & (df.ds == ds) & (df.pair_type == pt)]
            cells.append(f"{part.score.mean():.3f}" if len(part) else "-")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")

    lines += ["", f"## Paired bootstrap vs `{args.baseline}` (2.000 lần, cluster theo sản phẩm)", "",
              "| cell | Δ 1688 da [CI 95%] | Δ Tiki QE [CI 95%] | n 1688 | n Tiki |",
              "|---|---:|---:|---:|---:|"]
    for lab in labels:
        if lab == args.baseline:
            continue
        cells = []
        for ds in ("1688", "tiki"):
            b = df[(df.label == args.baseline) & (df.ds == ds)][["eval_id", "score"]]
            m = df[(df.label == lab) & (df.ds == ds)].merge(b, on="eval_id", suffixes=("", "_base"))
            if len(m) < 2:
                cells.append(("-", "-"))
                continue
            info = boot((m.score - m.score_base).to_numpy(), m.product_id.fillna(m.eval_id).to_numpy())
            sig = "*" if info["lo"] > 0 or info["hi"] < 0 else ""
            cells.append((f"{info['mean']:+.4f} [{info['lo']:+.4f}; {info['hi']:+.4f}]{sig}", str(info["n"])))
        lines.append(f"| {lab} | {cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")

    lines += ["", "## Đọc kết quả", "",
              "- Qwen3-8B là baseline mạnh nhất trên cả 1688 và Tiki.",
              "- Moses B đạt BLEU 1688 cao nhưng COMET-QE Tiki rất thấp.",
              "- OpenNMT A/B hiện rất yếu; A bản 10 epoch bị degenerate, bản 30k step đang chạy.",
              ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
