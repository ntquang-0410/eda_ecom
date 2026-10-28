#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge kết quả repair trở lại predictions của từng model, xuất dataset BT sạch.

Với mỗi model:
  - dòng repair thành công  -> thay target, qc_status='repaired'
  - dòng repair vẫn hỏng    -> giữ target cũ, qc_status='still_untranslated'
  - dòng không cần repair   -> qc_status='ok'

Xuất 2 file / model (không ghi đè bất kỳ artifact cũ nào):
  <out>/tiki_bt_<model>_full.parquet   : đủ 29.446 dòng + audit
  <out>/tiki_bt_<model>_train.parquet  : chỉ dòng dùng được để fine-tune
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(
    r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")

AUDIT_COLUMNS = [
    "cache_key", "source_text_bt", "key_vi", "key_zh_effective", "value_class",
    "target_text_bt", "model_id", "prompt_version", "json_ok",
    "n_new_tokens", "input_truncated", "attempt", "qc_status",
    "repair_pass", "latency_ms",
]


def is_good_translation(target: str) -> bool:
    text = (target or "").strip()
    if not text:
        return False
    if CJK_RE.search(text):
        return True
    return not VI_RE.search(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True, help="thư mục chứa <model>/predictions.parquet")
    ap.add_argument("--repair-root", required=True, help="thư mục chứa <model>/repair.parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    repair_root = Path(args.repair_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for model in MODELS:
        pred_path = pred_root / model / "predictions.parquet"
        repair_path = repair_root / model / "repair.parquet"
        if not pred_path.exists():
            print(f"SKIP {model}: thiếu predictions")
            continue

        df = pd.read_parquet(pred_path)
        before = df["target_text_bt"].fillna("").apply(is_good_translation)
        n_before = int(before.sum())

        df["qc_status"] = before.map({True: "ok", False: "needs_repair"})
        df["repair_pass"] = 0

        n_repaired = n_still = 0
        if repair_path.exists():
            rep = pd.read_parquet(repair_path)
            rep = rep.drop_duplicates("cache_key").set_index("cache_key")
            for idx, row in df.iterrows():
                key = str(row["cache_key"])
                if key not in rep.index:
                    continue
                fix = rep.loc[key]
                if bool(fix.get("good")) and str(fix.get("target_text_bt") or "").strip():
                    df.at[idx, "target_text_bt"] = fix["target_text_bt"]
                    df.at[idx, "raw_output"] = fix.get("raw_output")
                    df.at[idx, "json_ok"] = bool(fix.get("json_ok", True))
                    df.at[idx, "n_new_tokens"] = int(fix.get("n_new_tokens", 0))
                    df.at[idx, "attempt"] = 2
                    df.at[idx, "prompt_version"] = "plan_b_repair_v4"
                    df.at[idx, "repair_pass"] = 1
                    df.at[idx, "qc_status"] = "repaired"
                    n_repaired += 1
                else:
                    df.at[idx, "qc_status"] = "still_untranslated"
                    n_still += 1
        else:
            print(f"{model}: không có repair.parquet, chỉ đánh dấu")

        after = df["target_text_bt"].fillna("").apply(is_good_translation)
        train = df[after].copy()

        for col in AUDIT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        full_cols = [c for c in AUDIT_COLUMNS if c in df.columns]
        full_cols += [c for c in df.columns if c not in full_cols]

        full_path = out_dir / f"tiki_bt_{model}_full.parquet"
        train_path = out_dir / f"tiki_bt_{model}_train.parquet"
        df[full_cols].to_parquet(full_path, index=False)
        train[[c for c in full_cols if c in train.columns]].to_parquet(train_path, index=False)

        report[model] = {
            "rows": int(len(df)),
            "good_before": n_before,
            "good_before_pct": round(n_before / max(1, len(df)) * 100, 2),
            "repaired": n_repaired,
            "still_untranslated": n_still,
            "good_after": int(after.sum()),
            "good_after_pct": round(after.mean() * 100, 2),
            "train_rows": int(len(train)),
            "full_file": str(full_path),
            "train_file": str(train_path),
        }
        print(json.dumps({model: report[model]}, ensure_ascii=False, indent=2))

    (out_dir / "bt_repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_dir / "bt_repair_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
