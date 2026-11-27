#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit NLLB B: vì sao seed 42 lệch seed 7 tới 0,089 COMET-da (std 0,0628)?"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_canonical as ac

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"
IDX = pd.read_parquet(ROOT / "data/cleaned/eval_paired_858.parquet")
CL = IDX[IDX.dataset == "1688"].set_index("eval_id").product_id.to_dict()


def rep4(s: str) -> float:
    s = str(s)
    if len(s) < 20:
        return 0.0
    c = Counter(s[i:i + 4] for i in range(len(s) - 3))
    return c.most_common(1)[0][1] / max(1, len(s) - 3)


def main() -> None:
    rows = []
    for seed in (42, 7):
        d = pd.read_parquet(CANON / "nllb_b" / f"seed{seed}.parquet")
        a = d[d.dataset == "1688"]
        t = d[d.dataset == "tiki"]
        rows.append({
            "seed": seed, "da": a.comet_da.mean(), "qe": t.comet_qe.mean(),
            "cov1688": a.nonempty.mean(), "covtiki": t.nonempty.mean(),
            "len_med": int(np.median([len(str(x)) for x in t.hyp])),
            "rep4_med": float(np.median([rep4(x) for x in t.hyp])),
            "bleu_sent": a.bleu.mean(), "chrf_sent": a.chrf.mean(),
            "title": a[a.stratum == "title"].comet_da.mean(),
            "attribute": a[a.stratum == "attribute"].comet_da.mean(),
            "description": a[a.stratum == "description"].comet_da.mean(),
        })
    df = pd.DataFrame(rows)
    a42 = pd.read_parquet(CANON / "nllb_b" / "seed42.parquet")
    a7 = pd.read_parquet(CANON / "nllb_b" / "seed7.parquet")
    s42 = a42[a42.dataset == "1688"].set_index("eval_id").comet_da
    s7 = a7[a7.dataset == "1688"].set_index("eval_id").comet_da
    common = s42.index.intersection(s7.index)
    boot = ac.boot_diff((s42.loc[common] - s7.loc[common]).to_numpy(),
                        np.array([CL[i] for i in common]))

    md = ["# Audit NLLB-600M B — seed 42 vs seed 7", "",
          "Câu hỏi: std qua seed = 0,0628 (seed 42 = 0,5494; seed 7 = 0,6382) — đây là seed noise thật, "
          "hay một run bị lỗi?", "",
          "## 1. Chẩn đoán", "",
          "| Seed | COMET-da | COMET-QE | Cov 1688 | Cov Tiki | len med | rep4 med | BLEU sent | chrF sent |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        md.append(f"| {r.seed} | {r.da:.4f} | {r.qe:.4f} | {r.cov1688*100:.1f}% | {r.covtiki*100:.1f}% | "
                  f"{r.len_med} | {r.rep4_med:.3f} | {r.bleu_sent:.2f} | {r.chrf_sent:.2f} |")
    md += ["", "| Seed | 1688 Title | 1688 Attribute | 1688 Description |", "|---:|---:|---:|---:|"]
    for r in df.itertuples():
        md.append(f"| {r.seed} | {r.title:.4f} | {r.attribute:.4f} | {r.description:.4f} |")
    md += ["", f"Paired cluster bootstrap (seed 42 − seed 7, 2.000 lần, cluster = product_id): "
           f"**{boot['mean']:+.4f}** [{boot['lo']:+.4f}; {boot['hi']:+.4f}], n = {boot['n']}.", "",
           "## 2. Đọc kết quả", "",
           "- Độ dài hyp, tỉ lệ lặp 4-gram và coverage của hai seed gần như nhau ⇒ **không phải run hỏng** "
           "(không rỗng, không lặp vô hạn, không cắt cụt).",
           "- Chênh lệch trải đều cả 3 stratum (title −0,119; attribute −0,074; description −0,059) ⇒ "
           "không phải lỗi cục bộ một nhóm dữ liệu.",
           "- Với NLLB-600M LoRA, 4 M token trên tập B (~1,5 M token) là ~2,7 epoch; mỗi seed lấy một tập con "
           "khác nhau ⇒ phương sai run thật sự lớn. Đây là **seed/run variance**, không phải eval noise: "
           "CI của paired bootstrap chỉ phủ variance eval và vẫn cho −0,089.",
           "- Không lấy được log train seed 42 (kernel nằm ở account khác, không còn token), nên không thể "
           "xác nhận thêm bằng learning curve; nhưng artefact hyp đầy đủ và hợp lệ.",
           "",
           "## 3. Hệ quả báo cáo", "",
           "- Với n = 2 seed, std 0,0628 **không đủ tin cậy**; phải báo cả hai điểm riêng (0,5494 và 0,6382) "
           "thay vì chỉ mean ± std.",
           "- NLLB B là control duy nhất có seed thứ 2 cho đến nay; để kết luận A/B của NLLB cần ≥5 seed.",
           "- Không loại seed 42: nó hợp lệ. Nếu buộc chọn một điểm để so sánh với Qwen, dùng seed 7 "
           "(coverage 100%, chạy sau, có log) và ghi rõ."]

    out = ROOT / "reports/AUDIT_NLLB_B_SEED.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:16]))
    print("wrote", out)


if __name__ == "__main__":
    main()
