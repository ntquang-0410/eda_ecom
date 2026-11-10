#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3/T4 — Phát hiện auto loanword rồi phân loại bằng CC-CEDICT.

    python cado_layer/cado_tier3_loanword.py
    python cado_layer/cado_tier3_loanword.py --limit 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.cado import CadoConfig, run_cado
from cado_core.config import CADO_CSV_DIR, RAW_PARQUET, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Parquet đầu vào (mặc định data/raw/bilingual_zh_vi.parquet)")
    parser.add_argument("--output", help="Parquet đầu ra (tuỳ chọn)")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    df = pd.read_parquet(resolve_path(args.input, RAW_PARQUET))
    if args.limit:
        df = df.head(args.limit)

    result = run_cado(
        df,
        CadoConfig(columns=("title_zh",), max_tier=4, use_loanwords=True, progress=print),
    )

    report = result.loanword_report
    if not report.empty:
        report = report.sort_values("total_count", ascending=False)
        filename = (
            "cado_tier3_loanword_report_preview.csv"
            if args.limit
            else "cado_tier3_loanword_report.csv"
        )
        path = CADO_CSV_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n💾 {path} ({len(report):,} dòng)")

        print("\nTop 20 ứng viên:")
        print(report.head(20).to_string(index=False))

        loanwords = report[report["decision"] == "loanword_keep"]
        print(f"\n✅ {len(loanwords):,} từ mượn được giữ tự động")
        print(loanwords.head(30)[["word", "total_count", "overlap_rate"]].to_string(index=False))

    if args.output:
        output_path = resolve_path(args.output, None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.df.to_parquet(output_path, index=False)
        print(f"\n💾 {output_path}")


if __name__ == "__main__":
    main()
