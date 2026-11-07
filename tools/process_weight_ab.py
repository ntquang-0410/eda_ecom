#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA + processing cuối cho A/B (dùng dữ liệu đã chia trước EDA).

A: dedup exact pair, cap 960, giữ split.
B: strip boilerplate Tiki, bỏ target rỗng, cap 960, chunk >1280,
   QC Tiki description, dedup exact pair.

Output:
  data/cleaned/processed_A/{full,train,dev,test,units}.parquet
  data/cleaned/processed_B/{full,train,dev,test,units}.parquet
  reports/PROCESSING_A_B.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dedup_chunk_pairs import chunk_pair  # noqa: E402
from process_bt_desc import clean_source, clean_target  # noqa: E402

SPACE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
PRIORITY = {"train": 0, "dev": 1, "test": 2}


def compact(x) -> int:
    return len(SPACE.sub("", str(x or "")))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_text"] = df["target_text"].fillna("").astype(str)
    df["src_len"] = df["source_text"].map(compact)
    df["tgt_len"] = df["target_text"].map(compact)
    df["empty"] = df["tgt_len"] == 0
    df["cjk"] = [len(CJK_RE.findall(x)) for x in df["target_text"]]
    df["cjk_ratio"] = df["cjk"] / df["tgt_len"].clip(lower=1)
    df["vi"] = [len(VI_RE.findall(x)) for x in df["target_text"]]
    df["vi_ratio"] = df["vi"] / df["tgt_len"].clip(lower=1)
    df["ratio"] = df["tgt_len"] / df["src_len"].clip(lower=1)
    df["eff_tokens"] = df["tokens_est"].clip(lower=1.0, upper=960.0)
    return df


def qc_status(df: pd.DataFrame) -> pd.Series:
    ok = ~df["empty"]
    trunc = (df["input_truncated"] if "input_truncated" in df.columns
             else pd.Series(False, index=df.index))
    ok &= ~trunc.fillna(False).astype(bool)
    ok &= df["cjk_ratio"] >= 0.08
    ok &= df["vi_ratio"] <= 0.03
    ok &= df["ratio"].between(0.15, 2.5)
    return ok.map({True: "ok", False: "needs_review"})


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    x = df.copy()
    x["_rank"] = x["split"].map(PRIORITY).fillna(9)
    before = len(x)
    x = (x.sort_values(["_rank", "pair_id"])
         .drop_duplicates(["source_text", "target_text"], keep="first")
         .drop(columns=["_rank"]).reset_index(drop=True))
    return x, before - len(x)


def chunk_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows, chunked = [], 0
    for row in df.itertuples():
        src, tgt = str(row.source_text), str(row.target_text)
        total = compact(src) / 3.6 + compact(tgt) / 1.6
        parts = chunk_pair(src, tgt, max_tokens=800) if total > 1280 else [(src, tgt)]
        if len(parts) > 1:
            chunked += 1
        for i, (s, t) in enumerate(parts):
            rec = row._asdict(); rec.pop("Index", None)
            rec["pair_id"] = f"{row.pair_id}:chunk:{i}" if len(parts) > 1 else row.pair_id
            rec["parent_pair_id"] = row.pair_id
            rec["chunk_index"] = i
            rec["n_chunks"] = len(parts)
            rec["source_text"], rec["target_text"] = s, t
            rec["src_len"], rec["tgt_len"] = compact(s), compact(t)
            rec["tokens_est"] = round(compact(s) / 3.6 + compact(t) / 1.6, 4)
            rec["eff_tokens"] = min(960.0, max(1.0, rec["tokens_est"]))
            rec["cjk"] = len(CJK_RE.findall(t))
            rec["cjk_ratio"] = rec["cjk"] / max(1, rec["tgt_len"])
            rec["vi"] = len(VI_RE.findall(t))
            rec["vi_ratio"] = rec["vi"] / max(1, rec["tgt_len"])
            rec["ratio"] = rec["tgt_len"] / max(1, rec["src_len"])
            rows.append(rec)
    return pd.DataFrame(rows), chunked


def process(name: str, df: pd.DataFrame, eval_path: Path) -> dict:
    out_dir = ROOT / "data/cleaned" / f"processed_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = len(df)
    df = enrich(df)
    if name == "B":
        df["source_raw"] = df["source_text"]
        df["target_raw"] = df["target_text"]
        df["source_text"] = df["source_raw"].map(clean_source)
        df["target_text"] = df["target_raw"].map(clean_target)
        df = enrich(df)

    # bỏ target rỗng sau clean; không bỏ vì quality khác
    empty_dropped = int(df["empty"].sum())
    df = df[~df["empty"]].copy()
    # 1688/Tiki title/attribute là dữ liệu người/BT đã qua gate trước đó:
    # giữ tất cả; chỉ QC Tiki description (raw BT 4B).
    df["qc_status"] = "ok"
    if name == "B":
        desc_mask = df["group"] == "tiki_description"
        if desc_mask.any():
            df.loc[desc_mask, "qc_status"] = qc_status(df.loc[desc_mask])
    df, dedup_removed = dedup(df)
    df, chunked = chunk_rows(df)

    # chỉ units qc ok để train, nhưng full giữ tất cả
    full = df.copy()
    units = full[full["qc_status"] == "ok"].copy()
    full.to_parquet(out_dir / "full.parquet", index=False)
    units.to_parquet(out_dir / "units.parquet", index=False)
    for split in ("train", "dev", "test"):
        units[units["split"] == split].to_parquet(out_dir / f"{split}.parquet", index=False)

    mass = units.groupby("group")["eff_tokens"].sum()
    total = float(mass.sum())
    meta = {
        "dataset": name,
        "raw_rows": int(before),
        "empty_dropped": empty_dropped,
        "rows_full": int(len(full)),
        "rows_units": int(len(units)),
        "dedup_removed": int(dedup_removed),
        "chunked_parents": int(chunked),
        "tokens_million": round(total / 1e6, 3),
        "group_share": {k: round(float(v / total), 4) for k, v in mass.items()},
        "group_rows": {k: int(v) for k, v in units.groupby("group").size().items()},
        "split": {k: int(v) for k, v in units["split"].value_counts().items()},
        "empty": int(full["empty"].sum()),
        "qc_ok": int((full["qc_status"] == "ok").sum()),
        "eval_rows": int(len(pd.read_parquet(eval_path))),
    }
    (out_dir / "processing.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    A = pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/A_title_attribute.parquet")
    B = pd.read_parquet(ROOT / "data/cleaned/weight_splits_v2/B_title_description.parquet")
    ea = ROOT / "data/cleaned/weight_splits_v2/eval_A.parquet"
    eb = ROOT / "data/cleaned/weight_splits_v2/eval_B.parquet"
    a = process("A", A, ea)
    b = process("B", B, eb)
    (ROOT / "data/cleaned/processed_A_B.json").write_text(
        json.dumps({"A": a, "B": b}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"A": a, "B": b}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
