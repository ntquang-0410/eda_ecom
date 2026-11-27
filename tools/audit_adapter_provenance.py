#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit adapter provenance toàn bộ cell canonical (4.4).

Điều kiện pass:
1. `adapter_provenance.recorded == true`
2. đường dẫn (dataset_ref + adapter_name) khớp token dataset của cell (A/B, model family)
3. có md5; nếu file local còn tồn tại thì md5 phải khớp lại

Kết quả: reports/ADAPTER_PROVENANCE_AUDIT.md + .artifacts/canonical/_provenance.csv
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"

# cell -> regex bắt buộc phải có trong path
EXPECT = {
    "qwen4b_a": r"(m4b[-_]?a|qwen4b[-_]?a)",
    "qwen4b_b": r"(m4b[-_]?b|qwen4b[-_]?b)",
    "qwen8b_a": r"(m8b[-_]?a|qwen8b[-_]?a)",
    "qwen8b_a_balanced": r"(m8b[-_]?a|qwen8b[-_]?a)",
    "qwen8b_b": r"(m8b[-_]?b|qwen8b[-_]?b)",
    "qwen8b_b_natural": r"(m8b[-_]?b|qwen8b[-_]?b)",
    "ecomgpt_a": r"ecomgpt(?!-b)",
    "ecomgpt_a_balanced": r"ecomgpt(?!-b)",
    "ecomgpt_b": r"ecomgpt[-_]?b",
    "moses_a": r"moses[-_]?a",
    "moses_b": r"moses[-_]?b",
    "onmt_a": r"(onmt|nmt)[-_]?a",
    "onmt_b": r"(onmt|nmt)[-_]?b",
    "nllb_a": r"nllb[-_]?a",
    "nllb_b": r"nllb[-_]?b",
}


def file_md5(p: str | None) -> str | None:
    if not p:
        return None
    f = Path(p)
    if not f.exists():
        return None
    h = hashlib.md5()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    for mp in sorted(CANON.glob("*/seed*.manifest.json")):
        cell = mp.parent.name
        seed = int(mp.name.replace("seed", "").split(".")[0])
        man = json.loads(mp.read_text(encoding="utf-8"))
        p = man.get("adapter_provenance", {}) or {}
        path = " ".join(str(x) for x in (p.get("dataset_ref"), p.get("adapter_name")) if x)
        pat = EXPECT.get(cell)
        path_ok = bool(pat and re.search(pat, path, re.IGNORECASE))
        md5_ok = bool(p.get("md5"))
        if p.get("local_file"):
            md5_ok = md5_ok and (file_md5(p["local_file"]) == p.get("md5"))
        passed = bool(p.get("recorded")) and path_ok and md5_ok
        rows.append({"cell": cell, "seed": seed, "source": p.get("source", "unknown"),
                     "dataset_ref": p.get("dataset_ref") or "",
                     "adapter_name": p.get("adapter_name") or "",
                     "path_ok": path_ok, "md5_ok": md5_ok,
                     "recorded": bool(p.get("recorded")), "pass": passed,
                     "reason": "" if passed else (
                         "chưa ghi provenance" if not p.get("recorded") else
                         "path không khớp dataset" if not path_ok else "thiếu/sai md5")})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(CANON / "_provenance.csv", index=False)

    cells = df.groupby("cell").pass_.all() if "pass_" in df else df.groupby("cell")["pass"].all()
    md = ["# Audit adapter provenance — 22 run canonical (4.4)", "",
          "Điều kiện pass: `recorded=true` + path khớp token dataset của cell + có md5 "
          "(và md5 khớp lại nếu file còn).", "",
          "## 1. Theo run", "",
          "| Cell | Seed | Nguồn | dataset_ref | adapter | Path khớp | md5 | Pass | Lý do |",
          "|---|---:|---|---|---|---:|---:|---|---|"]
    for r in df.to_dict("records"):
        md.append(f"| {r['cell']} | {r['seed']} | {r['source']} | {r['dataset_ref']} | {r['adapter_name']} | "
                  f"{'✓' if r['path_ok'] else '✗'} | {'✓' if r['md5_ok'] else '✗'} | "
                  f"{'✅' if r['pass'] else '❌'} | {r['reason']} |")

    md += ["", "## 2. Theo cell (pass = mọi seed đều pass)", "",
           "| Cell | #seed | Pass | QE 858 hiển thị |", "|---|---:|---|---|"]
    for cell, ok in cells.items():
        n = int((df.cell == cell).sum())
        md.append(f"| {cell} | {n} | {'✅' if ok else '❌'} | {'có' if ok else '**ẩn**'} |")
    md += ["", f"**Tổng:** {int(df['pass'].sum())}/{len(df)} run pass; "
           f"{int(cells.sum())}/{len(cells)} cell pass.", "",
           "Cell fail bị **ẩn cột COMET-QE** trong `CANONICAL_EVAL.md` (QE là chẩn đoán); "
           "COMET-da và xếp hạng chính thức giữ nguyên."]

    (ROOT / "reports/ADAPTER_PROVENANCE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(df[["cell", "seed", "source", "path_ok", "md5_ok", "pass", "reason"]].to_string(index=False))
    print("\ncell pass:", list(cells[cells].index))


if __name__ == "__main__":
    main()
