#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T5 — BGE-M3 semantic fallback cho các từ không có trong CC-CEDICT.

Quyết định được đưa ra THEO TỪNG LẦN XUẤT HIỆN (không áp dụng toàn cục theo
word như bản cũ), nên một từ hiếm khi bị xoá oan ở mọi ngữ cảnh.

    python cado_layer/cado_tier5_bge.py --limit 200
    python cado_layer/cado_tier5_bge.py --device cuda
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
    parser.add_argument("--device", default="cpu", help="cpu/cuda/mps")
    parser.add_argument("--threshold", type=float, default=0.98,
                        help="Ngưỡng cosine để coi từ là dư thừa (mặc định 0.98)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Giới hạn số cặp cần encode (mặc định không giới hạn)")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    df = pd.read_parquet(resolve_path(args.input, RAW_PARQUET))
    if args.limit:
        df = df.head(args.limit)

    result = run_cado(
        df,
        CadoConfig(
            columns=("title_zh",),
            max_tier=5,
            use_loanwords=True,
            tier5_device=args.device,
            tier5_threshold=args.threshold,
            tier5_max_pairs=args.max_pairs,
            tier5_progress_bar=not args.no_progress_bar,
            progress=print,
        ),
    )

    report = result.t5_report
    if not report.empty:
        filename = (
            "cado_tier5_similarity_preview.csv"
            if args.limit
            else "cado_tier5_similarity.csv"
        )
        path = CADO_CSV_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n💾 {path} ({len(report):,} cặp)")
        print("\nTop 20 cặp giống nhau nhất:")
        print(report.head(20)[["word", "similarity", "decision"]].to_string(index=False))

    if args.output:
        output_path = resolve_path(args.output, None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.df.to_parquet(output_path, index=False)
        print(f"\n💾 {output_path}")


if __name__ == "__main__":
    main()
