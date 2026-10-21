#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T0 — Trích xuất toàn bộ từ Latin trong title/description tiếng Trung.

Chỉ là bước thống kê: KHÔNG sửa dữ liệu. Kết quả dùng để xây whitelist và
đánh giá mức độ nhiễm tiếng Anh của dataset.

    python cado_layer/cado_tier0_extract.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.config import ENGLISH_REMOVER_CSV_DIR, RAW_PARQUET, resolve_path
from cado_core.text import extract_latin_words


def extract_inventory(df: pd.DataFrame, columns: tuple[str, ...]) -> list[dict]:
    records: dict[str, dict] = {}

    for column in columns:
        if column not in df.columns:
            continue
        for product_id, text in zip(df["product_id"], df[column]):
            for word in extract_latin_words(text):
                record = records.get(word)
                if record is None:
                    record = records[word] = {
                        "word": word,
                        "count": 0,
                        "columns": set(),
                        "product_ids": [],
                    }
                record["count"] += 1
                record["columns"].add(column)
                record["product_ids"].append(str(product_id))

    # Giữ schema legacy: sort theo word (ASCII) và product_id unique tăng dần.
    for record in records.values():
        record["product_ids"] = sorted(set(record["product_ids"]))
    return sorted(records.values(), key=lambda r: r["word"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Parquet đầu vào (mặc định data/raw/bilingual_zh_vi.parquet)")
    parser.add_argument("--columns", default="title_zh,description_zh")
    parser.add_argument("--output", default=str(ENGLISH_REMOVER_CSV_DIR / "english_words.csv"))
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu")
    args = parser.parse_args()

    input_path = resolve_path(args.input, RAW_PARQUET)
    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit)

    columns = tuple(part.strip() for part in args.columns.split(",") if part.strip())
    inventory = extract_inventory(df, columns)

    title_words = {r["word"] for r in inventory if "title_zh" in r["columns"]}
    desc_words = {r["word"] for r in inventory if "description_zh" in r["columns"]}

    print(f"📂 {input_path} — {len(df):,} dòng")
    print(f"🔤 Từ unique: {len(inventory):,}")
    print(f"   title_zh: {len(title_words):,} | description_zh: {len(desc_words):,}")
    print(f"   title-only: {len(title_words - desc_words):,}")
    print("\nTop 20:")
    for record in sorted(inventory, key=lambda r: -r["count"])[:20]:
        print(
            f"  {record['count']:>6,}  {record['word']:<20} "
            f"{'|'.join(sorted(record['columns']))}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["word", "count", "columns", "product_id"]
        )
        writer.writeheader()
        for record in inventory:
            writer.writerow({
                "word": record["word"],
                "count": record["count"],
                "columns": "|".join(sorted(record["columns"])),
                "product_id": ",".join(record["product_ids"]),
            })
    print(f"\n💾 {output_path} ({len(inventory):,} từ)")


if __name__ == "__main__":
    main()
