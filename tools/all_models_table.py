#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng tổng hợp MỌI metric cho mọi cell (model × data × recipe × seed).

Nguồn: score từng dòng (COMET-da 1688 / COMET-QE Tiki) + hyp parquet để tính
BLEU/chrF trên đúng tập common 831 eval_id.

Chạy: python tools/all_models_table.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / ".artifacts/hyp"
COMMON = pd.read_parquet(ROOT / "data/cleaned/eval_F_v2_common.parquet")
IDS = set(COMMON.eval_id.astype(str))

OFF = ROOT / ".artifacts/matrix_out/comet5_v2_common/scores"
GPU = ROOT / ".artifacts/comet_gpu/scores"
S7 = ROOT / ".artifacts/comet_s7_common"
A7 = ROOT / ".artifacts/comet_a7"
B8 = ROOT / ".artifacts/comet_8b"

# (model, data, recipe, seed, hyp stem, score dir, score stem, note)
CELLS = [
    ("Qwen3-8B", "A", "natural", 42, "hyp_qwen_a_seed42", OFF, "qwen_a_seed42", ""),
    ("Qwen3-8B", "B", "balanced", 42, "hyp_qwen_b_seed42", OFF, "qwen_b_seed42", ""),
    ("Qwen3-8B", "A", "balanced", 42, "hyp_qwen8b_a_balanced_seed42", None, "qwen8b_a_balanced_seed42",
     "chỉ có điểm tổng, chưa có score từng dòng"),
    ("Qwen3-8B", "B", "natural", 42, "hyp_m8b_b_natural_seed42", B8, "m8b_b_natural_seed42", "tái lập 24/11"),
    ("Qwen3-4B", "A", "natural", 42, "hyp_qwen4b_a_seed42", OFF, "qwen4b_a_seed42", ""),
    ("Qwen3-4B", "B", "balanced", 42, "hyp_qwen4b_b_seed42", OFF, "qwen4b_b_seed42", ""),
    ("Qwen3-4B", "A", "balanced", 42, "hyp_qwen4b_a_balanced_seed42", GPU, "qwen4b_a_balanced_seed42", ""),
    ("Qwen3-4B", "B", "natural", 42, "hyp_qwen4b_b_natural_seed42", GPU, "qwen4b_b_natural_seed42", ""),
    ("Qwen3-4B", "A", "natural", 123, "hyp_qwen4b_a_seed123", GPU, "qwen4b_a_seed123", ""),
    ("Qwen3-4B", "B", "balanced", 123, "hyp_qwen4b_b_balanced_seed123", GPU, "qwen4b_b_balanced_seed123", ""),
    ("Qwen3-4B", "A", "natural", 7, "hyp_qwen4b_a_natural_seed7", A7, "natural_seed7", ""),
    ("EcomGPT-7B", "A", "natural", 42, "hyp_ecomgpt_a_seed42", OFF, "ecomgpt_a_seed42", ""),
    ("EcomGPT-7B", "B", "balanced", 42, "hyp_ecomgpt_b_seed42", OFF, "ecomgpt_b_seed42", ""),
    ("EcomGPT-7B", "A", "balanced", 42, "hyp_ecomgpt_a_balanced_seed42", S7, "ecomgpt_a_balanced_seed42", "tái lập 24/11"),
    ("Moses", "A", "natural", 42, "hyp_moses_a_seed42", OFF, "moses_a_seed42", ""),
    ("Moses", "B", "balanced", 42, "hyp_moses_b_seed42", OFF, "moses_b_seed42", ""),
    ("OpenNMT", "A", "natural", 42, "hyp_onmt_a_seed42", OFF, "onmt_a_seed42", ""),
    ("OpenNMT", "B", "natural", 42, "hyp_onmt_b_seed42", OFF, "onmt_b_seed42", ""),
    ("NLLB-600M", "A", "natural", 42, "hyp_nllb_a_seed42", GPU, "nllb_a_seed42", ""),
    ("NLLB-600M", "B", "balanced", 42, "hyp_nllb_b_seed42", GPU, "nllb_b_seed42", ""),
    ("NLLB-600M", "A", "natural", 7, "hyp_nllb_a_seed7", S7, "nllb_a_seed7", ""),
    ("NLLB-600M", "B", "balanced", 7, "hyp_nllb_b_seed7", S7, "nllb_b_seed7", ""),
]

AGG_ONLY = {  # điểm tổng đã công bố khi không có score từng dòng
    ("Qwen3-8B", "A", "balanced", 42): (0.8553, 0.0562),
}

STRATA = ["title", "attribute", "description"]
COLS = ["1688_title", "1688_attribute", "1688_description", "macro_1688",
        "tiki_title", "tiki_attribute", "tiki_description", "macro_tiki"]


def load_pair(stem: str, sdir: Path | None):
    if sdir is None:
        return None
    out = {}
    for ds in ("1688", "tiki"):
        p = sdir / f"scores_{stem}_{ds}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        d["eval_id"] = d["eval_id"].astype(str)
        out[ds] = d[d.eval_id.isin(IDS)].drop_duplicates("eval_id")
    return out["1688"], out["tiki"]


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.4f}".replace(".", ",")


def main() -> None:
    rows, strat_rows = [], []
    for model, data, recipe, seed, hyp_stem, sdir, stem, note in CELLS:
        hyp = pd.read_parquet(HYP / f"{hyp_stem}.parquet")
        hyp["eval_id"] = hyp["eval_id"].astype(str)
        hyp = hyp[hyp.eval_id.isin(IDS)].drop_duplicates("eval_id")

        pair = load_pair(stem, sdir)
        key = (model, data, recipe, seed)
        if pair is None:
            da, qe = AGG_ONLY[key]
            s1688 = s_tiki = None
            n1688 = int(((hyp.dataset == "1688") & hyp.hypothesis.astype(str).str.strip().ne("")).sum())
            n_tiki = int(((hyp.dataset == "tiki") & hyp.hypothesis.astype(str).str.strip().ne("")).sum())
        else:
            s1688, s_tiki = pair
            if "hypothesis" in s1688:
                s1688 = s1688[s1688.hypothesis.astype(str).str.strip().ne("")]
                s_tiki = s_tiki[s_tiki.hypothesis.astype(str).str.strip().ne("")]
            da, qe = float(s1688.score.mean()), float(s_tiki.score.mean())
            n1688, n_tiki = len(s1688), len(s_tiki)

        h1688 = hyp[(hyp.dataset == "1688") & hyp.hypothesis.astype(str).str.strip().ne("")]
        h1688 = h1688[h1688.reference.astype(str).str.strip().ne("")]
        if s1688 is not None:
            h1688 = h1688[h1688.eval_id.isin(set(s1688.eval_id))]
        bleu = sacrebleu.corpus_bleu(h1688.hypothesis.tolist(), [h1688.reference.tolist()], tokenize="zh").score
        chrf = sacrebleu.corpus_chrf(h1688.hypothesis.tolist(), [h1688.reference.tolist()]).score

        rows.append({"model": model, "data": data, "recipe": recipe, "seed": seed,
                     "comet_da": da, "comet_qe": qe, "bleu": float(bleu), "chrf": float(chrf),
                     "n1688": n1688, "n_tiki": n_tiki, "note": note})

        if pair is not None:
            rec = {"model": model, "data": data, "recipe": recipe, "seed": seed}
            for ds, df in (("1688", s1688), ("tiki", s_tiki)):
                for st in STRATA:
                    sub = df[df.pair_type == st]
                    rec[f"{ds}_{st}"] = float(sub.score.mean()) if len(sub) else np.nan
            rec["macro_1688"] = float(np.nanmean([rec[f"1688_{s}"] for s in STRATA]))
            rec["macro_tiki"] = float(np.nanmean([rec[f"tiki_{s}"] for s in STRATA]))
            strat_rows.append(rec)

    df = pd.DataFrame(rows)
    st = pd.DataFrame(strat_rows)
    if len(st):
        df = df.merge(st[["model", "data", "recipe", "seed", "macro_1688", "macro_tiki"]],
                      on=["model", "data", "recipe", "seed"], how="left")

    t1 = ["| Model | Data | Recipe | Seed | COMET-da 1688 | COMET-QE Tiki | Macro 1688 | Macro Tiki | BLEU | chrF | n 1688 | n Tiki | Ghi chú |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for rec in df.to_dict("records"):
        t1.append(f"| {rec['model']} | {rec['data']} | {rec['recipe']} | {rec['seed']} | {fmt(rec['comet_da'])} | "
                  f"{fmt(rec['comet_qe'])} | {fmt(rec.get('macro_1688', float('nan')))} | {fmt(rec.get('macro_tiki', float('nan')))} | "
                  f"{rec['bleu']:.2f} | {rec['chrf']:.2f} | {rec['n1688']} | {rec['n_tiki']} | {rec['note']} |")

    t2 = ["| Model | Data | Recipe | Seed | " + " | ".join(COLS) + " |",
          "|---|---|---|---:|" + "---:|" * len(COLS)]
    for rec in st.to_dict("records"):
        t2.append(f"| {rec['model']} | {rec['data']} | {rec['recipe']} | {rec['seed']} | " +
                  " | ".join(fmt(rec[c]) for c in COLS) + " |")

    footer = """
## Ghi chú

- Đơn vị: COMET-da/QE là điểm hệ thống (trung bình per-row); BLEU/chrF thang 0–100.
- `n` = số dòng thực sự được chấm sau khi lọc common + hyp không rỗng; NLLB B seed 42 thiếu 1 dòng 1688 và 3 dòng Tiki (hyp rỗng).
- Seed noise ước lượng 0,005–0,007 COMET (1688) ⇒ chênh lệch <0,01 giữa hai cell cùng cấu hình không nên đọc là hiệu ứng.
- **Loại khỏi bảng:** Qwen3-4B B-balanced seed 7 (hyp cũ sinh bằng adapter mẫu competition, đang chờ inference lại) và EcomGPT-7B B-natural (generation thoái hoá: 1.471/1.500 hyp rỗng).
- Qwen3-8B A-balanced chỉ có điểm tổng từ `CHECKPOINT_2026-11-24.md`; BLEU/chrF trong bảng tính từ hyp cùng adapter, COMET chưa có score từng dòng nên không bootstrap được.
- EcomGPT dùng ngân sách ~1,42 M token hiệu dụng (recipe chia theo token đã encode), không phải 4 M như Qwen/NLLB — xem `MATRIX_5X2_EVAL_F_V2.md` §11.3.
"""
    body = ("# Bảng tổng hợp mọi metric — toàn bộ cell\n\n"
            "Sinh bởi `tools/all_models_table.py`. Eval `eval_F_v2_common` (831 eval_id: 1688 = 231, Tiki = 600).\n"
            "COMET-da = `Unbabel/wmt22-comet-da` (1688, có reference); COMET-QE = `Unbabel/wmt20-comet-qe-da` (Tiki, không reference).\n"
            "BLEU/chrF: 1688, hyp không rỗng, tokenize `zh` (sacrebleu).\n\n"
            "## 1. Tổng hợp\n\n" + "\n".join(t1) +
            "\n\n## 2. COMET theo stratum + macro-average\n\n" + "\n".join(t2) + "\n" + footer)
    out = ROOT / "reports/ALL_MODELS_METRICS.md"
    out.write_text(body, encoding="utf-8")
    print("\n".join(t1))
    print()
    print("\n".join(t2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
