#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QC + EDA + processing cho Tiki description BT.

Input : bt_desc_v1/<model>/predictions.parquet
Output: data/cleaned/bt_tiki_desc/<model>/{full,train,units}.parquet
        reports/TIKI_DESC_EDA.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_chunk_pairs import chunk_pair  # noqa: E402

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
SPACE = re.compile(r"\s+")
BOILER_RE = re.compile(r"Giá sản phẩm trên Tiki đã bao gồm thuế.*?(?:\.{2,}|…+)", re.I | re.S)


def clean_source(text) -> str:
    return BOILER_RE.sub("", str(text or "")).strip()


def clean_target(text) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[。！？])", t)
    kept = []
    for part in parts:
        if re.search(r"(Tiki|TikTok)", part, re.I) and re.search(r"(价格|税费|费用|运费|税)", part):
            continue
        if part.startswith("此外") and re.search(r"(费用|税费|运费|进口税)", part):
            continue
        kept.append(part)
    return "".join(kept).strip()


def compact_len(text) -> int:
    return len(SPACE.sub("", str(text or "")))


def norm_num(value: str) -> str:
    return value.replace(",", ".").rstrip(".")


def nums(text) -> Counter:
    return Counter(norm_num(x) for x in NUM_RE.findall(str(text or "")))


def qc_row(row) -> dict:
    src = str(row.get("source_text") or "")
    tgt = str(row.get("target_text_bt") or "").strip()
    cjk = len(CJK_RE.findall(tgt))
    vi = len(VI_RE.findall(tgt))
    ratio = len(tgt) / max(1, len(src))
    src_n, tgt_n = nums(src), nums(tgt)
    missing = sum((src_n - tgt_n).values())
    extra = sum((tgt_n - src_n).values())
    flags = {
        "f_empty": not tgt,
        "f_short": len(tgt) < 10,
        "f_untranslated": vi > max(3, int(0.03 * len(tgt))),
        "f_no_cjk": cjk < max(2, int(0.08 * len(tgt))),
        "f_ratio": not (0.15 <= ratio <= 2.5),
        "f_identical": SPACE.sub("", tgt).lower() == SPACE.sub("", src).lower(),
        "f_input_truncated": bool(row.get("input_truncated")),
        "f_number_mismatch": (missing + extra) > max(1, int(0.1 * len(src_n))) if src_n else False,
        "f_boilerplate": "giá sản phẩm trên tiki" in str(row.get("source_raw") or src).lower()
                          or "tiki đã bao gồm thuế" in str(row.get("source_raw") or src).lower(),
    }
    blocking = ("f_empty", "f_short", "f_no_cjk", "f_untranslated", "f_identical", "f_input_truncated")
    ok = bool(row.get("good")) and not any(flags[k] for k in blocking)
    return {
        **flags,
        "cjk_chars": cjk,
        "cjk_ratio": round(cjk / max(1, len(tgt)), 4),
        "char_ratio": round(ratio, 4),
        "vi_chars": vi,
        "missing_numbers": int(missing),
        "extra_numbers": int(extra),
        "qc_status_new": "ok" if ok else "needs_review",
    }


def process_model(model: str, src_root: Path, out_root: Path) -> dict:
    path = src_root / model / "predictions.parquet"
    if not path.exists():
        return {"model": model, "error": "missing predictions.parquet"}
    df = pd.read_parquet(path).drop_duplicates("pair_id").reset_index(drop=True)
    df["source_raw"] = df["source_text"]
    df["target_raw"] = df["target_text_bt"]
    df["source_text"] = df["source_raw"].map(clean_source)
    df["target_text_bt"] = df["target_raw"].map(clean_target)
    qc = pd.DataFrame([qc_row(r) for r in df.to_dict("records")])
    df = pd.concat([df.drop(columns=["qc_status"], errors="ignore"), qc], axis=1)
    df["qc_status"] = df["qc_status_new"]
    df = df.drop(columns=["qc_status_new"])

    # EDA before dedup
    eda = {
        "rows": int(len(df)),
        "good_provider": int(df["good"].sum()),
        "qc_ok": int((df["qc_status"] == "ok").sum()),
        "qc_ok_rate": round(float((df["qc_status"] == "ok").mean()), 4),
        "empty": int(df["f_empty"].sum()),
        "short": int(df["f_short"].sum()),
        "untranslated": int(df["f_untranslated"].sum()),
        "no_cjk": int(df["f_no_cjk"].sum()),
        "ratio_bad": int(df["f_ratio"].sum()),
        "identical": int(df["f_identical"].sum()),
        "input_truncated": int(df["f_input_truncated"].sum()),
        "number_mismatch": int(df["f_number_mismatch"].sum()),
        "boilerplate": int(df["f_boilerplate"].sum()),
        "source_chars_mean": round(float(df["source_text"].map(compact_len).mean()), 1),
        "target_chars_mean": round(float(df["target_text_bt"].map(compact_len).mean()), 1),
        "cjk_ratio_mean": round(float(df["cjk_ratio"].mean()), 4),
    }

    # dedup exact pair
    df["dup_count"] = df.groupby(["source_text", "target_text_bt"])["pair_id"].transform("size")
    before = len(df)
    df = df.drop_duplicates(["source_text", "target_text_bt"], keep="first").reset_index(drop=True)
    eda["dedup_removed"] = int(before - len(df))

    full = df.copy()
    full.to_parquet(out_root / model / "full.parquet", index=False)

    # chunk long description; only qc ok
    usable = full[full["qc_status"] == "ok"].copy()
    rows, chunked = [], 0
    for row in usable.itertuples():
        src, tgt = str(row.source_text), str(row.target_text_bt)
        parts = chunk_pair(src, tgt, max_tokens=800) if compact_len(src) / 3.6 + compact_len(tgt) / 1.6 > 1280 else [(src, tgt)]
        if len(parts) > 1:
            chunked += 1
        for index, (s, t) in enumerate(parts):
            record = row._asdict()
            record.pop("Index", None)
            record["pair_id"] = f"{row.pair_id}:chunk:{index}" if len(parts) > 1 else row.pair_id
            record["parent_pair_id"] = row.pair_id
            record["chunk_index"] = index
            record["n_chunks"] = len(parts)
            record["source_text"] = s
            record["target_text_bt"] = t
            record["src_chars"] = compact_len(s)
            record["tgt_chars"] = compact_len(t)
            record["tokens_est"] = round(compact_len(s) / 3.6 + compact_len(t) / 1.6, 4)
            rows.append(record)
    units = pd.DataFrame(rows)
    units["unit_type"] = "description"
    units.to_parquet(out_root / model / "units.parquet", index=False)
    train = units[units["split"] == "train"].copy()
    train.to_parquet(out_root / model / "train.parquet", index=False)

    eda["chunked_products"] = int(chunked)
    eda["units"] = int(len(units))
    eda["train_units"] = int(len(train))
    eda["split"] = {k: int(v) for k, v in units["split"].value_counts().items()}
    eda["token_million"] = round(float(units["tokens_est"].sum()) / 1e6, 3)
    eda["boilerplate_in_train"] = int(units["f_boilerplate"].sum()) if "f_boilerplate" in units else 0
    (out_root / model / "eda.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model": model, **eda}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/content/bt_desc_out/bt_desc_v1")
    ap.add_argument("--out", default="data/cleaned/bt_tiki_desc")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models:
        (out / model).mkdir(parents=True, exist_ok=True)
        results.append(process_model(model, src, out))
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
