#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng tổng hợp mọi metric — sinh từ artefact canonical (zero-fill + coverage).

Chạy: python tools/all_models_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"
PROV = CANON / "_provenance.csv"
ORDER = ["Qwen3-8B", "Qwen3-4B", "EcomGPT-7B", "Moses", "OpenNMT", "NLLB-600M"]
COLS = ["1688_title", "1688_attribute", "1688_description", "macro_unw_1688", "macro_w_1688",
        "tiki_title", "tiki_attribute", "tiki_description", "macro_unw_tiki", "macro_w_tiki"]


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if np.isfinite(x).any() else float("nan")


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def f(v, nd=4) -> str:
    if v is None:
        return "—"
    try:
        if np.isnan(v):
            return "—"
    except TypeError:
        pass
    return f"{v:.{nd}f}"


def provenance_ok() -> set[str]:
    if not PROV.exists():
        return set()
    d = pd.read_csv(PROV)
    return set(d.groupby("cell")["pass"].all().loc[lambda s: s].index)


def main() -> None:
    ok_cells = provenance_ok()
    rows, strata = [], []
    for cell_dir in sorted(CANON.glob("*")):
        files = sorted(cell_dir.glob("seed*.parquet"))
        if not files:
            continue
        mans = []
        for p in files:
            mp = cell_dir / f"{p.stem}.manifest.json"
            mans.append(json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {})
        model = next((m.get("model") for m in mans if m.get("model")), cell_dir.name)
        data = next((m.get("data") for m in mans if m.get("data")), "")
        recipe = next((m.get("recipe") for m in mans if m.get("recipe")), "")
        seeds = sorted(int(p.stem.replace("seed", "")) for p in files)

        da, qe, cov1, covt, bleu, chrf = [], [], [], [], [], []
        st = {c: [] for c in COLS}
        for p in files:
            d = pd.read_parquet(p)
            a = d[d.dataset == "1688"]
            t = d[d.dataset == "tiki"]
            da.append(a.comet_da.mean()); qe.append(t.comet_qe.mean())
            cov1.append(a.nonempty.mean()); covt.append(t.nonempty.mean())
            bleu.append(a.bleu.mean()); chrf.append(a.chrf.mean())
            for ds, part, col in (("1688", a, "comet_da"), ("tiki", t, "comet_qe")):
                vals, cnt = {}, part.stratum.value_counts()
                for s in ("title", "attribute", "description"):
                    m = part[part.stratum == s][col].mean()
                    st[f"{ds}_{s}"].append(m)
                    vals[s] = m
                st[f"macro_unw_{ds}"].append(float(np.mean(list(vals.values()))))
                st[f"macro_w_{ds}"].append(float(sum(vals[k] * cnt.get(k, 0) for k in vals) / max(1, cnt.sum())))
        if cell_dir.name not in ok_cells:
            qe = [float("nan")] * len(qe)
        rec = {
            "model": model, "data": data, "recipe": recipe,
            "seeds": ",".join(map(str, seeds)), "n_seeds": len(seeds),
            "da": safe_mean(da), "da_std": safe_std(da),
            "qe": safe_mean(qe), "qe_std": safe_std(qe),
            "cov1": float(np.nanmean(cov1)), "covt": float(np.nanmean(covt)),
            "bleu": float(np.nanmean(bleu)), "chrf": float(np.nanmean(chrf)),
        }
        rows.append(rec)
        if not np.isnan(rec["da"]) and not np.isnan(rec["qe"]):
            sr = {"model": model, "data": data, "recipe": recipe, "seeds": rec["seeds"]}
            for c in COLS:
                sr[c] = safe_mean(st[c])
            strata.append(sr)

    df = pd.DataFrame(rows)
    missing = df[df.da.isna()]
    df = df[df.da.notna()].copy()
    df["morder"] = df.model.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    df = df.sort_values(["morder", "data", "recipe"]).drop(columns="morder")
    st = pd.DataFrame(strata)

    t1 = ["| Model | Data | Recipe | Seeds | COMET-da 1688 | Cov 1688 | COMET-QE Tiki (chẩn đoán) | Cov Tiki | BLEU sent | chrF sent |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        ds = "" if np.isnan(r.da_std) else f" ± {r.da_std:.4f}"
        qs = "" if np.isnan(r.qe_std) else f" ± {r.qe_std:.4f}"
        t1.append(f"| {r.model} | {r.data} | {r.recipe} | {r.seeds} | {f(r.da)}{ds} | {r.cov1*100:.1f}% | "
                  f"{f(r.qe)}{qs} | {r.covt*100:.1f}% | {r.bleu:.2f} | {r.chrf:.2f} |")

    t2 = ["| Model | Data | Recipe | " + " | ".join(COLS) + " |",
          "|---|---|---|" + "---:|" * len(COLS)]
    for rec in st.to_dict("records"):
        t2.append(f"| {rec['model']} | {rec['data']} | {rec['recipe']} | " +
                  " | ".join(f(rec[c]) for c in COLS) + " |")

    notes = ["", "## Ghi chú", "",
             "- **Baseline lịch sử** (đơn seed, `max_seq=960`): không dùng để xếp hạng chính thức; "
             "bảng chính thức sẽ là campaign ≥5 seed với `max_seq=1792`.",
             "- Bảng sinh từ `.artifacts/canonical/*/seed*.parquet` — index 858 dòng cố định, "
             "hyp rỗng = 0, **không lọc dòng**. Quy ước đầy đủ: `reports/EVAL_PROTOCOL.md`.",
             "- **Xếp hạng chỉ dùng COMET-da 1688.** COMET-QE, BLEU, chrF là chẩn đoán.",
             "- `COMET-da`/`COMET-QE` là mean ± std qua seed (chỉ có std khi ≥2 seed).",
             "- `BLEU sent`/`chrF sent` = trung bình điểm câu (không phải corpus BLEU).",
             f"- Cell không có score từng dòng (loại khỏi bảng): "
             + (", ".join(missing.model + " " + missing.data) if len(missing) else "không") + ".",
             "- Human eval Tiki: **MISSING** (sheet đã sinh, chưa có người chấm).",
             "- **QE ẩn cho cell chưa pass adapter provenance (4.4)** — xem "
             "`reports/ADAPTER_PROVENANCE_AUDIT.md`; COMET-da vẫn đầy đủ."]
    if (df.cov1 < 1).any() or (df.covt < 1).any():
        notes.append("- Coverage <100%: " + ", ".join(
            f"{r.model} {r.data} ({r.cov1*100:.1f}%/{r.covt*100:.1f}%)"
            for r in df.itertuples() if r.cov1 < 1 or r.covt < 1) + " — đã zero-fill.")

    md = ["# Bảng tổng hợp mọi metric (canonical)", "",
          "> Sinh bởi `tools/all_models_table.py` từ artefact canonical. "
          "Số liệu thay thế bảng cũ khi có khác biệt (zero-fill + coverage).",
          "", "## 1. Tổng hợp", "", "\n".join(t1),
          "", "## 2. COMET theo stratum + macro", "",
          "`macro_unw` = trung bình 3 stratum; `macro_w` = trọng số theo số dòng (bằng mean tổng).",
          "", "\n".join(t2)] + notes
    out = ROOT / "reports/ALL_MODELS_METRICS.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(t1))
    print("wrote", out)


if __name__ == "__main__":
    main()
