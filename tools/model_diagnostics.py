#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chẩn đoán lỗi theo model × recipe × stratum.

Sinh bảng định lượng (rỗng, độ dài, lặp, Hán/Việt, giữ số/thương hiệu)
và ví dụ lỗi nặng nhất cho báo cáo matrix.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

HAN = re.compile(r"[\u4e00-\u9fff]")
DIGITS = re.compile(r"\d+(?:[.,]\d+)?")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9&+._-]{1,}")
STRATA = [("1688", "title"), ("1688", "attribute"), ("1688", "description"),
          ("tiki", "title"), ("tiki", "attribute"), ("tiki", "description")]
PRETTY = {"qwen": "Qwen3-8B", "qwen4b": "Qwen3-4B", "ecomgpt": "EcomGPT-7B",
          "moses": "Moses", "onmt": "OpenNMT"}


def vi_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return float("nan")
    n = 0
    for c in chars:
        if c.isascii():
            continue
        try:
            name = unicodedata.name(c)
        except ValueError:
            continue
        if name.startswith("LATIN") and ("WITH" in name or c in "ăâđêôơưĂÂĐÊÔƠƯ"):
            n += 1
    return n / len(chars)


def han_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return len(HAN.findall(text)) / len(chars) if chars else float("nan")


def rep4(text: str) -> float:
    t = re.sub(r"\s+", "", text)
    if len(t) < 5:
        return 0.0
    grams = [t[i:i + 4] for i in range(len(t) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def digit_recall(source: str, hyp: str) -> float:
    nums = DIGITS.findall(str(source or ""))
    if not nums:
        return float("nan")
    h = str(hyp or "").replace(" ", "")
    return sum(n in h for n in nums) / len(nums)


def latin_recall(source: str, hyp: str) -> float:
    toks = [t.lower() for t in LATIN.findall(str(source or "")) if len(t) >= 2]
    if not toks:
        return float("nan")
    h = str(hyp or "").lower()
    return sum(t in h for t in toks) / len(toks)


def load_hyps(hyp_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(hyp_dir.glob("hyp_*_seed42.parquet")):
        label = re.sub(r"^hyp_|_seed42\.parquet$", "", p.name)
        d = pd.read_parquet(p)
        d["label"] = label
        frames.append(d[["eval_id", "source", "reference", "hypothesis", "label"]])
    df = pd.concat(frames, ignore_index=True)
    df["eval_id"] = df["eval_id"].astype(str)
    df["hypothesis"] = df["hypothesis"].fillna("").astype(str)
    return df


def load_scores(score_dir: Path) -> pd.DataFrame:
    frames = []
    pat = re.compile(r"scores_(?P<label>.+)_seed42_(?P<ds>1688|tiki)\.parquet")
    for p in sorted(score_dir.glob("scores_*.parquet")):
        m = pat.match(p.name)
        if not m:
            continue
        d = pd.read_parquet(p)
        d["label"] = m.group("label")
        d["ds"] = m.group("ds")
        frames.append(d[["eval_id", "score", "label", "ds"]])
    df = pd.concat(frames, ignore_index=True)
    df["eval_id"] = df["eval_id"].astype(str)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default=".artifacts/hyp")
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--eval", default="/content/matrix_data_dl/eval_A.parquet")
    ap.add_argument("--out", default="reports/MODEL_DIAGNOSTICS.md")
    args = ap.parse_args()

    ev = pd.read_parquet(args.eval)
    ev["eval_id"] = ev["eval_id"].astype(str)
    hyp = load_hyps(Path(args.hyp_dir))
    sc = load_scores(Path(args.scores_dir))
    df = hyp.merge(ev[["eval_id", "dataset", "pair_type", "source_text", "target_text"]],
                   on="eval_id", how="left", suffixes=("", "_ev"))
    df["ref"] = np.where(df["dataset"] == "tiki", df["target_text"], df["reference"])
    df["ref"] = pd.Series(df["ref"], index=df.index).fillna("")
    df["pair_type"] = df["pair_type"].fillna("")
    df["dataset"] = df["dataset"].fillna("")

    df["is_empty"] = df.hypothesis.str.strip().eq("")
    df["is_short"] = df.hypothesis.str.len() < 5
    df["len"] = df.hypothesis.str.len()
    df["ref_len"] = np.where(df["dataset"] == "tiki", np.nan, df.ref.str.len().clip(lower=1))
    df["len_ratio"] = df.len / df.ref_len
    df["rep4"] = df.hypothesis.map(rep4)
    df["han"] = df.hypothesis.map(han_ratio)
    df["vi"] = df.hypothesis.map(vi_ratio)
    df["digit"] = [digit_recall(s, h) for s, h in zip(df.source_text, df.hypothesis)]
    df["latin"] = [latin_recall(s, h) for s, h in zip(df.source_text, df.hypothesis)]

    merged = df.merge(sc, on=["eval_id", "label"], how="left", suffixes=("", "_sc"))
    # score file đã tách ds; chỉ giữ score đúng dataset
    merged["score"] = np.where(merged.ds == merged.dataset, merged.score, np.nan)

    labels = sorted(merged.label.unique())
    rows = []
    for lab in labels:
        d = merged[merged.label == lab]
        for ds, pt in STRATA:
            part = d[(d.dataset == ds) & (d.pair_type == pt)]
            if not len(part):
                continue
            rows.append({
                "label": lab, "dataset": ds, "stratum": pt, "n": len(part),
                "comet": part.score.mean(),
                "empty_pct": 100 * part.is_empty.mean(),
                "short_pct": 100 * part.is_short.mean(),
                "med_len": part.len.median(),
                "len_ratio": part.len_ratio.median(),
                "rep4": part.rep4.mean(),
                "han_pct": 100 * part.han.mean(),
                "vi_pct": 100 * part.vi.mean(),
                "digit_recall": part.digit.mean(),
                "latin_recall": part.latin.mean(),
            })
    tab = pd.DataFrame(rows)

    # BLEU/chrF theo stratum (chỉ 1688, bỏ hyp rỗng)
    bleu_rows = []
    for lab in labels:
        d = merged[(merged.label == lab) & (merged.dataset == "1688")
                   & (~merged.is_empty) & merged.ref.str.strip().ne("")]
        for pt in ("title", "attribute", "description"):
            part = d[d.pair_type == pt]
            if not len(part):
                continue
            bleu_rows.append({
                "label": lab, "stratum": pt, "n": len(part),
                "bleu": sacrebleu.corpus_bleu(part.hypothesis.tolist(), [part.ref.tolist()], tokenize="zh").score,
                "chrf": sacrebleu.corpus_chrf(part.hypothesis.tolist(), [part.ref.tolist()]).score,
            })
    bl = pd.DataFrame(bleu_rows)

    # overall
    overall = []
    for lab in labels:
        d = merged[merged.label == lab]
        d1688 = d[(d.dataset == "1688") & (~d.is_empty) & d.ref.str.strip().ne("")]
        overall.append({
            "label": lab,
            "comet1688": d[d.dataset == "1688"].score.mean(),
            "comettiki": d[d.dataset == "tiki"].score.mean(),
            "bleu": sacrebleu.corpus_bleu(d1688.hypothesis.tolist(), [d1688.ref.tolist()], tokenize="zh").score if len(d1688) else np.nan,
            "chrf": sacrebleu.corpus_chrf(d1688.hypothesis.tolist(), [d1688.ref.tolist()]).score if len(d1688) else np.nan,
            "empty_pct": 100 * d.is_empty.mean(),
            "short_pct": 100 * d.is_short.mean(),
            "rep4": d.rep4.mean(),
            "vi_pct": 100 * d.vi.mean(),
            "digit_recall": d.digit.mean(),
            "latin_recall": d.latin.mean(),
            "unique_hyp_pct": 100 * d.hypothesis.nunique() / max(1, len(d)),
        })
    ov = pd.DataFrame(overall).set_index("label")

    # examples: worst COMET per label (description 1688), fallback any
    examples = {}
    for lab in labels:
        d = merged[(merged.label == lab) & (merged.dataset == "1688")
                   & merged.score.notna()].sort_values("score")
        desc = d[d.pair_type == "description"]
        pick = (desc if len(desc) else d).head(1)
        if len(pick):
            r = pick.iloc[0]
            examples[lab] = {
                "score": r.score, "stratum": r.pair_type,
                "source": str(r.source_text)[:160],
                "ref": str(r.ref)[:160],
                "hyp": str(r.hypothesis)[:160],
            }

    lines = ["# Chẩn đoán lỗi theo model — matrix 5×2", "",
             "Sinh từ `eval_F` (1.500 dòng) + hypothesis + score COMET. "
             "Các chỉ số: `rep4` = tỉ lệ 4-gram lặp trong câu (càng cao càng máy móc); "
             "`vi%` = % ký tự Latin có dấu (dấu hiệu sót tiếng Việt); "
             "`digit`/`latin` = recall số/thương hiệu từ nguồn.", ""]

    lines += ["## 1. Tổng quan", "",
              "| model | COMET 1688 | COMET-QE Tiki | BLEU | chrF | rỗng % | <5 ký tự % | rep4 | vi % | digit | latin | hyp unique % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lab in labels:
        r = ov.loc[lab]
        lines.append(f"| {lab} | {r.comet1688:.4f} | {r.comettiki:.4f} | {r.bleu:.2f} | {r.chrf:.2f} | "
                     f"{r.empty_pct:.1f} | {r.short_pct:.1f} | {r.rep4:.3f} | {r.vi_pct:.1f} | "
                     f"{r.digit_recall:.2f} | {r.latin_recall:.2f} | {r.unique_hyp_pct:.1f} |")

    lines += ["", "## 2. Theo stratum", "",
              "| model | dataset | stratum | n | COMET | rỗng % | med len | len/ref | rep4 | Hán % | vi % | digit | latin |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.label} | {r.dataset} | {r.stratum} | {r.n} | {r.comet:.3f} | {r.empty_pct:.1f} | "
                     f"{r.med_len:.0f} | {r.len_ratio:.2f} | {r.rep4:.3f} | {r.han_pct:.1f} | "
                     f"{r.vi_pct:.1f} | {r.digit_recall:.2f} | {r.latin_recall:.2f} |")

    # coverage gap: stratum có trong training vs stratum bị bỏ
    gap_rows = []
    for lab in labels:
        recipe = lab.rsplit("_", 1)[-1]
        unseen = "description" if recipe == "a" else "attribute"
        for ds in ("1688", "tiki"):
            d = tab[(tab.label == lab) & (tab.dataset == ds)]
            if not len(d):
                continue
            seen = d[d.stratum != unseen]
            miss = d[d.stratum == unseen]
            gap_rows.append({"label": lab, "dataset": ds,
                             "seen": seen.comet.mean(), "unseen": miss.comet.mean(),
                             "gap": seen.comet.mean() - miss.comet.mean(),
                             "unseen_stratum": unseen})
    gap = pd.DataFrame(gap_rows)
    lines += ["", "## 3. Khoảng trống stratum bị bỏ khỏi training", "",
              "A chỉ có title+attribute; B chỉ có title+description. Cột `unseen` là stratum không hề xuất hiện trong training.", "",
              "| model | dataset | stratum unseen | COMET seen | COMET unseen | gap |",
              "|---|---|---|---:|---:|---:|"]
    for _, r in gap.iterrows():
        lines.append(f"| {r.label} | {r.dataset} | {r.unseen_stratum} | {r.seen:.3f} | {r.unseen:.3f} | {r.gap:+.3f} |")

    lines += ["", "## 4. BLEU/chrF theo stratum (1688)", "",
              "| model | stratum | n | BLEU | chrF |", "|---|---|---:|---:|---:|"]
    for _, r in bl.iterrows():
        lines.append(f"| {r.label} | {r.stratum} | {r.n} | {r.bleu:.2f} | {r.chrf:.2f} |")

    lines += ["", "## 5. Ví dụ lỗi nặng nhất (1688, COMET thấp nhất)", ""]
    for lab in labels:
        e = examples.get(lab)
        if not e:
            continue
        lines += [f"### {lab} — COMET {e['score']:.3f} ({e['stratum']})", "",
                  f"- **Nguồn:** {e['source']}",
                  f"- **Tham chiếu:** {e['ref']}",
                  f"- **Hypothesis:** {e['hyp']}", ""]

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    tab.to_csv(Path(args.out).with_suffix(".strata.csv"), index=False)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
