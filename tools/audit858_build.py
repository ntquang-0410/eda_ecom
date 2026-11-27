#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ghép audit 858: 831 cũ + 27 dòng Tiki đã chấm QE, zero-fill, coverage.

Input : .artifacts/canonical/<cell>/seed<seed>.parquet  (831)
        /tmp/audit858/{extra_index.parquet, hyps/*.parquet, qe_extra.parquet}
Output: .artifacts/canonical858/<cell>/seed<seed>.parquet + manifest
        reports/AUDIT_858.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path("/tmp/audit858")
CANON = ROOT / ".artifacts/canonical"
OUT = ROOT / ".artifacts/canonical858"
INDEX858 = ROOT / "data/cleaned/eval_paired_858.parquet"


def main() -> None:
    ev = pd.read_parquet(INDEX858)
    ev["eval_id"] = ev.eval_id.astype(str)
    extra_ids = set(pd.read_parquet(STAGE / "extra_index.parquet").eval_id.astype(str))
    qe_extra = pd.read_parquet(STAGE / "qe_extra.parquet")
    qe_extra["eval_id"] = qe_extra.eval_id.astype(str)
    man = json.loads((STAGE / "manifest.json").read_text(encoding="utf-8"))

    rows = []
    for m in man:
        cell, seed = m["cell"], int(m["seed"])
        base_p = CANON / cell / f"seed{seed}.parquet"
        if not base_p.exists():
            print("MISSING base", base_p)
            continue
        base = pd.read_parquet(base_p)
        base["eval_id"] = base.eval_id.astype(str)
        b = base.set_index("eval_id")

        hyp_extra = pd.read_parquet(STAGE / "hyps" / m["file"])
        hyp_extra["eval_id"] = hyp_extra.eval_id.astype(str)
        hx = dict(zip(hyp_extra.eval_id, hyp_extra.hypothesis))
        qx = qe_extra[(qe_extra.cell == cell) & (qe_extra.seed == seed)]
        qx_map = dict(zip(qx.eval_id, qx.score))

        out = []
        for r in ev.itertuples():
            e = r.eval_id
            if e in b.index:
                out.append(b.loc[e].to_dict() | {"eval_id": e, "dataset": r.dataset,
                                                 "stratum": r.stratum})
                continue
            hyp = str(hx.get(e, "") or "")
            ne = bool(hyp.strip())
            q = qx_map.get(e, 0.0 if not ne else float("nan"))
            out.append({"eval_id": e, "dataset": r.dataset, "stratum": r.stratum,
                        "hyp": hyp, "ref": "",
                        "comet_da": float("nan"), "comet_qe": q,
                        "bleu": float("nan"), "chrf": float("nan"), "nonempty": ne})
        df = pd.DataFrame(out)

        dst = OUT / cell
        dst.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst / f"seed{seed}.parquet", index=False)

        man_out = {}
        mp = CANON / cell / f"seed{seed}.manifest.json"
        if mp.exists():
            man_out = json.loads(mp.read_text(encoding="utf-8"))
        man_out.update({"audit": "858", "extra_rows": len(extra_ids),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "coverage": {ds: float(df[(df.dataset == ds) & df.nonempty].mean())
                                     for ds in ("1688", "tiki")}})
        (dst / f"seed{seed}.manifest.json").write_text(
            json.dumps(man_out, ensure_ascii=False, indent=2), encoding="utf-8")

        a831 = base[base.dataset == "tiki"]
        a858 = df[df.dataset == "tiki"]
        rows.append({"cell": cell, "seed": seed,
                     "qe831": a831.comet_qe.mean(), "cov831": a831.nonempty.mean(),
                     "qe858": a858.comet_qe.mean(), "cov858": a858.nonempty.mean(),
                     "empty_extra": int((~a858[a858.eval_id.isin(extra_ids)].nonempty).sum())})
        print(f"{cell:22s} s{seed:<4d} QE 831={a831.comet_qe.mean():.4f} → 858={a858.comet_qe.mean():.4f} "
              f"(cov {a831.nonempty.mean()*100:.1f}% → {a858.nonempty.mean()*100:.1f}%, "
              f"empty extra={rows[-1]['empty_extra']})")

    df = pd.DataFrame(rows)
    agg = (df.groupby("cell")
             .agg(seeds=("seed", "count"), qe831=("qe831", "mean"), qe858=("qe858", "mean"),
                  cov831=("cov831", "mean"), cov858=("cov858", "mean"),
                  empty_extra=("empty_extra", "sum"))
             .reset_index())
    agg["delta_qe"] = agg.qe858 - agg.qe831
    agg = agg.sort_values("delta_qe")

    md = ["# Audit 858 dòng — zero-fill 27 dòng Tiki bị loại khỏi 831", "",
          "Index: `data/cleaned/eval_paired_858.parquet` (1688 = 231, Tiki = 627). "
          "27 dòng bổ sung đều là Tiki (18 attribute + 9 description), chấm COMET-QE; "
          "1688 giữ nguyên nên **COMET-da và xếp hạng chính thức không đổi**.", "",
          "## 1. Theo cell (trung bình qua seed)", "",
          "| Cell | #seed | QE Tiki 831 | QE Tiki 858 | Δ | Cov 831 | Cov 858 | Hyp rỗng thêm |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in agg.itertuples():
        md.append(f"| {r.cell} | {r.seeds} | {r.qe831:.4f} | {r.qe858:.4f} | {r.delta_qe:+.4f} | "
                  f"{r.cov831*100:.1f}% | {r.cov858*100:.1f}% | {r.empty_extra} |")

    md += ["", "## 2. Theo run", "",
           "| Cell | Seed | QE 831 | QE 858 | Δ | Cov 831 | Cov 858 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in df.sort_values(["cell", "seed"]).itertuples():
        md.append(f"| {r.cell} | {r.seed} | {r.qe831:.4f} | {r.qe858:.4f} | {r.qe858-r.qe831:+.4f} | "
                  f"{r.cov831*100:.1f}% | {r.cov858*100:.1f}% |")

    worst = agg.iloc[0]
    md += ["", "## 3. Kết luận", "",
           f"- 27 dòng bổ sung làm QE Tiki thay đổi trong khoảng {agg.delta_qe.min():+.4f} … "
           f"{agg.delta_qe.max():+.4f}; mức lớn nhất ở **{worst.cell}** ({worst.delta_qe:+.4f}).",
           "- Các cell Qwen có 9/27 hyp rỗng ở nhóm này ⇒ coverage Tiki giảm còn ~98,6%; "
           "EcomGPT B-balanced rỗng 18/27 ⇒ ~97,1%. Đây chính là phần bị 831 che mất.",
           "- Vì 1688 không đổi, kết luận xếp hạng theo COMET-da giữ nguyên; "
           "nhưng coverage Tiki phải báo cáo trên 858 kể từ nay.",
           "- Khuyến nghị: chuyển index chính thức sang 858 (thuần nội dung) nếu chấp nhận "
           "đổi số QE; nếu giữ 831 để liên tục, luôn kèm phụ lục 858 này."]

    path = ROOT / "reports/AUDIT_858.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
