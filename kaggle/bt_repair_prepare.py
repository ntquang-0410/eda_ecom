#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn bị input cho vòng repair BT: lọc các dòng "chưa dịch" của từng model.

Một dòng bị coi là chưa dịch khi:
  - json_ok = False, hoặc target rỗng, hoặc
  - target không có chữ Hán **và** còn dấu tiếng Việt.

Ngoại lệ hợp lệ: target Latin/số thuần (brand, model code, size) — không có dấu
tiếng Việt nên không bị lọc.

Output: 1 parquet gộp (model_key, cache_key, source_text_bt, key_vi,
key_zh_effective, value_class) + stats json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(
    r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")


def needs_repair(target: str, json_ok: bool) -> bool:
    text = (target or "").strip()
    if not json_ok or not text:
        return True
    if CJK_RE.search(text):
        return False
    return bool(VI_RE.search(text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True,
                        help="thư mục chứa <model>/predictions.parquet")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, stats = [], {}
    for model in MODELS:
        path = pred_root / model / "predictions.parquet"
        if not path.exists():
            print(f"SKIP {model}: không thấy {path}")
            continue
        df = pd.read_parquet(path)
        flag = [
            needs_repair(row.target_text_bt, bool(row.json_ok))
            for row in df.itertuples()
        ]
        bad = df.loc[flag, [
            "cache_key", "source_text_bt", "key_vi",
            "key_zh_effective", "value_class",
        ]].copy()
        bad.insert(0, "model_key", model)
        frames.append(bad)
        stats[model] = {
            "rows": int(len(df)),
            "repair": int(len(bad)),
            "repair_pct": round(len(bad) / max(1, len(df)) * 100, 2),
        }
        print(f"{model}: {len(bad):,}/{len(df):,} cần repair")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(out_dir / "repair_input.parquet", index=False)
    payload = {
        "models": stats,
        "rows_total": int(len(merged)),
        "gate": "json_ok=False OR empty OR (no CJK AND has Vietnamese diacritics)",
    }
    (out_dir / "repair_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "Tiki BT Repair Input",
        "id": "ntquang0410/tiki-bt-repair-input",
        "licenses": [{"name": "CC0-1.0"}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
