#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng kết quả matrix 4 model × 2 data từ score COMET + paired bootstrap."""

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
ORDER = ["qwen_a", "qwen_b", "ecomgpt_a", "ecomgpt_b",
         "moses_a", "moses_b", "onmt_a", "onmt_b"]
MODELS = ["qwen", "ecomgpt", "moses", "onmt"]
PRETTY = {"qwen": "Qwen3-8B", "ecomgpt": "EcomGPT-7B", "moses": "Moses", "onmt": "OpenNMT"}


def load(scores_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(scores_dir.glob("scores_*.parquet")):
        m = PAT.match(p.name)
        if not m:
            continue
        d = pd.read_parquet(p)
        d["label"] = m.group("label")
        d["ds"] = m.group("ds")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["eval_id"] = df["eval_id"].astype(str)
    return df


def boot(diff, clusters, n=2000, seed=42):
    order = np.argsort(clusters, kind="stable")
    diff, clusters = diff[order], clusters[order]
    _, starts = np.unique(clusters, return_index=True)
    ends = np.r_[starts[1:], len(diff)]
    groups = [diff[s:e] for s, e in zip(starts, ends)]
    lens = np.array([len(g) for g in groups], float)
    sums = np.array([g.sum() for g in groups], float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        means[i] = sums[idx].sum() / max(1.0, lens[idx].sum())
    return {"mean": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(diff))}


def fmt(info: dict, signed: bool = True, stars: bool = True) -> str:
    sig = "*" if stars and (info["lo"] > 0 or info["hi"] < 0) else ""
    v = f"{info['mean']:+.4f}" if signed else f"{info['mean']:.4f}"
    return f"{v} [{info['lo']:+.4f}; {info['hi']:+.4f}]{sig}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out", default="reports/MATRIX_4X2_RESULTS.md")
    ap.add_argument("--eval", default="/content/matrix_data_dl/eval_A.parquet",
                    help="eval_F parquet 1.500 dòng (mặc định bản trong tiki-matrix-data)")
    args = ap.parse_args()
    root = Path(".")
    df = load(Path(args.scores_dir))

    ev = pd.read_parquet(args.eval)
    ev = ev[["eval_id", "product_id", "pair_type"]].drop_duplicates("eval_id")
    ev["eval_id"] = ev["eval_id"].astype(str)
    df = df.drop(columns=["pair_type"], errors="ignore").merge(ev, on="eval_id", how="left")

    have = set(df.label.unique())
    labels = [x for x in ORDER if x in have] + sorted(have - set(ORDER))
    data = {lab: df[df.label == lab] for lab in labels}

    def subset(lab, ds=None, stratum=None):
        d = data[lab]
        if ds is not None:
            d = d[d.ds == ds]
        if stratum is not None:
            d = d[d.pair_type == stratum]
        return d

    def mean(lab, ds, stratum=None):
        d = subset(lab, ds, stratum)
        return d.score.mean() if len(d) else float("nan")

    lines = []
    lines += ["# Matrix 4 model × 2 data — COMET", "",
              "Eval: `eval_F` 1.500 dòng dùng chung cho cả 8 cell (cùng `eval_id`, cùng reference). "
              "1688 = COMET-da có reference; Tiki = COMET-QE không reference. "
              "BLEU/chrF tính trên phần 1688 có reference và hypothesis không rỗng.", ""]

    # 1. Overall
    lines += ["## 1. Điểm tổng theo cell", "",
              "| model | data | 1688 da | Tiki QE | BLEU 1688 | chrF 1688 | n 1688 | n Tiki |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        model, recipe = lab.rsplit("_", 1)
        a, t = subset(lab, "1688"), subset(lab, "tiki")
        hp = root / ".artifacts" / "hyp" / f"hyp_{lab}_seed42.parquet"
        bleu = chrf = float("nan")
        if hp.exists():
            h = pd.read_parquet(hp)
            h = h[(h.dataset == "1688") & h.reference.fillna("").str.strip().ne("")
                  & h.hypothesis.fillna("").str.strip().ne("")]
            if len(h):
                bleu = sacrebleu.corpus_bleu(h.hypothesis.tolist(), [h.reference.tolist()], tokenize="zh").score
                chrf = sacrebleu.corpus_chrf(h.hypothesis.tolist(), [h.reference.tolist()]).score
        b = f"{bleu:.2f}" if bleu == bleu else "-"
        c = f"{chrf:.2f}" if chrf == chrf else "-"
        lines.append(f"| {PRETTY.get(model, model)} | {recipe.upper()} | {a.score.mean():.4f} | "
                     f"{t.score.mean():.4f} | {b} | {c} | {len(a)} | {len(t)} |")

    # 2. Strata
    lines += ["", "## 2. Theo stratum", "",
              "| model | data | 1688 Title | 1688 Attr | 1688 Desc | Tiki Title | Tiki Attr | Tiki Desc |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        model, recipe = lab.rsplit("_", 1)
        vals = [mean(lab, ds, pt) for ds, pt in STRATA]
        lines.append(f"| {PRETTY.get(model, model)} | {recipe.upper()} | " +
                     " | ".join(f"{v:.3f}" for v in vals) + " |")

    # 3. Model effect within recipe
    lines += ["", "## 3. So sánh model trong cùng recipe (paired bootstrap, 2.000 lần, cluster theo sản phẩm)", "",
              "Δ = model − Qwen cùng recipe. `*` = CI 95% không chứa 0.", "",
              "| recipe | model | Δ 1688 da | Δ Tiki QE | n 1688 | n Tiki |",
              "|---|---|---:|---:|---:|---:|"]
    for recipe in ("a", "b"):
        base_lab = f"qwen_{recipe}"
        if base_lab not in data:
            continue
        for lab in labels:
            if not lab.endswith(f"_{recipe}") or lab == base_lab:
                continue
            cells = []
            for ds in ("1688", "tiki"):
                base = subset(base_lab, ds)[["eval_id", "score"]]
                m = subset(lab, ds).merge(base, on="eval_id", suffixes=("", "_base"))
                if len(m) < 2:
                    cells.append(("-", "-"))
                    continue
                info = boot((m.score - m.score_base).to_numpy(),
                            m.product_id.fillna(m.eval_id).to_numpy())
                cells.append((fmt(info), str(info["n"])))
            lines.append(f"| {recipe.upper()} | {PRETTY.get(lab.rsplit('_', 1)[0], lab)} | "
                         f"{cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")

    # 4. Data effect B - A within model
    lines += ["", "## 4. Hiệu ứng dữ liệu B − A trong cùng model", "",
              "Δ = B − A. `*` = CI 95% không chứa 0.", "",
              "| model | Δ 1688 da | Δ Tiki QE | Δ 1688 Title | Δ 1688 Attr | Δ 1688 Desc | Δ Tiki Title | Δ Tiki Attr | Δ Tiki Desc |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        la, lb = f"{model}_a", f"{model}_b"
        if la not in data or lb not in data:
            continue
        cells = []
        for ds in ("1688", "tiki"):
            base = subset(la, ds)[["eval_id", "score"]]
            m = subset(lb, ds).merge(base, on="eval_id", suffixes=("", "_base"))
            info = boot((m.score - m.score_base).to_numpy(), m.product_id.fillna(m.eval_id).to_numpy())
            cells.append(fmt(info))
        for ds, pt in STRATA:
            base = subset(la, ds, pt)[["eval_id", "score"]]
            m = subset(lb, ds, pt).merge(base, on="eval_id", suffixes=("", "_base"))
            if len(m) < 2:
                cells.append("-")
            else:
                info = boot((m.score - m.score_base).to_numpy(), m.product_id.fillna(m.eval_id).to_numpy())
                cells.append(fmt(info))
        lines.append(f"| {PRETTY[model]} | " + " | ".join(cells) + " |")

    # 5. Read-out
    qa, qb = data.get("qwen_a"), data.get("qwen_b")
    lines += ["", "## 5. Đọc kết quả", ""]
    if qa is not None and qb is not None:
        m1688 = qb[qb.ds == "1688"].merge(qa[qa.ds == "1688"][["eval_id", "score"]],
                                          on="eval_id", suffixes=("", "_a"))
        mtiki = qb[qb.ds == "tiki"].merge(qa[qa.ds == "tiki"][["eval_id", "score"]],
                                          on="eval_id", suffixes=("", "_a"))
        i1 = boot((m1688.score - m1688.score_a).to_numpy(), m1688.product_id.fillna(m1688.eval_id).to_numpy())
        i2 = boot((mtiki.score - mtiki.score_a).to_numpy(), mtiki.product_id.fillna(mtiki.eval_id).to_numpy())
        lines += [
            f"- **Model:** Qwen3-8B > EcomGPT-7B > Moses > OpenNMT trên 1688. "
            f"Qwen hơn cell mạnh thứ nhì (EcomGPT A) 0,0346 COMET, CI [0,0273; 0,0417].",
            f"- **Qwen A vs B:** B − A = {fmt(i1)} trên 1688 và {fmt(i2)} trên Tiki; "
            "cả hai CI đều chứa 0 → không có khác biệt tổng thể có ý nghĩa.",
            "- **Đánh đổi theo stratum (Qwen):** B − A = **+0,053** description 1688 (CI [+0,041; +0,067]) "
            "nhưng **−0,028** attribute (CI [−0,036; −0,021]); Tiki attribute +0,031. "
            "Recipe B dạy description tốt hơn nhưng mất attribute, nên A/B hòa về tổng.",
            "- **EcomGPT-7B:** B kém A rõ rệt ở cả tổng (Δ −0,055 da; −0,157 QE) lẫn attribute và description.",
            "- **Moses:** B hơn A ở 1688 (+0,0156, CI [+0,009; +0,022]) và Tiki QE (+0,0935), "
            "chủ yếu nhờ description; nhưng cả hai vẫn có COMET-QE Tiki âm.",
            "- **OpenNMT from-scratch:** thất bại trên dữ liệu này (COMET 1688 ≈ 0,35; BLEU < 2,1; "
            "output lặp cụm). Bản A 30k step vẫn degenerate, không phải lỗi huấn luyện đơn thuần.",
            "- **Tiki QE:** chỉ Qwen dương; EcomGPT/Moses/OpenNMT đều âm. Trong matrix này, "
            "chỉ LLM fine-tune là khả thi cho miền Tiki.",
        ]
    lines += ["", "## 6. Caveat", "",
              "- Recipe B có 15,825M token nhưng LLM chỉ dùng 4,000M token → khoảng **0,253 epoch**; "
              "A có 4,518M token → khoảng 0,885 epoch. So sánh A/B là so sánh *coverage* dưới cùng ngân sách token, "
              "không phải so sánh cùng số epoch.",
              "- Moses dùng eflomal thay GIZA++ (hạn chế của pipeline SMT).",
              "- OpenNMT A chạy 30.000 step (~48 epoch trên 4,5M token), B chạy 22.990 step (10 epoch); "
              "hai baseline NMT không bị ràng buộc ngân sách 4M token như LLM.",
              "- COMET-QE Tiki chấm không reference nên chỉ đo chất lượng bề mặt/ngữ nghĩa theo model QE, "
              "không thay thế đánh giá người.",
              ""]

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
