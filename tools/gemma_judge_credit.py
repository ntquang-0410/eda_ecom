#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm Gemma theo thang tín dụng AAA..C (9 mức), blind, xoay vòng API key.

Output tương thích `tools/aggregate_weight_eval.py`:
  gemma_scores.json -> summary.by_stratum[config::dataset::pair_type].mean
  mean là điểm số 1..9 (C=1 ... AAA=9).

Dùng:
  python tools/gemma_judge_credit.py \
    --input data/cleaned/judge/judge_input.jsonl \
    --map data/cleaned/judge/judge_map.jsonl \
    --keys keys.txt --out-dir data/cleaned/judge
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_judge import (  # noqa: E402
    KeyPool, classify_error, load_keys, log, strip_code_fence, ts,
)

INSTRUCTION = """You are a strict bilingual evaluator for Vietnamese-to-Simplified-Chinese e-commerce translation.

Assign ONE credit-style grade to each hypothesis:
AAA (9): fully correct, natural, terminology/numbers exact.
AA  (8): minor style/word-choice issue, no meaning loss.
A   (7): small errors, meaning preserved, usable.
BBB (6): noticeable errors, still understandable.
BB  (5): some meaning loss, main idea preserved.
B   (4): significant meaning loss or awkward Chinese.
CCC (3): partial mistranslation, key information missing.
CC  (2): mostly wrong or incoherent.
C   (1): unusable, empty, or hallucinated.

Rules:
- Use the reference when provided; otherwise judge against the source only.
- Penalize untranslated Vietnamese, missing numbers/units, and hallucinated content.
- Do not reward literal word-for-word copying.
- Return ONLY a JSON array, one object per input:
  {"id": "...", "rating": "AAA", "note": "<=20 words"}
- rating must be exactly one of: AAA, AA, A, BBB, BB, B, CCC, CC, C.

Data:
"""

GRADES = ["C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
SCORE = {g: i + 1 for i, g in enumerate(GRADES)}


def parse_credit(text: str) -> list[dict]:
    value = strip_code_fence(text or "")
    data = None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("["), value.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    rows = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        raw = str(item.get("rating") or "").strip().upper()
        match = re.search(r"\b(AAA|AA|A|BBB|BB|B|CCC|CC|C)\b", raw)
        if not match:
            continue
        rating = match.group(1)
        rows.append({
            "id": str(item["id"]),
            "rating": rating,
            "score": SCORE[rating],
            "note": str(item.get("note") or "")[:200],
        })
    return rows


def build_prompt(rows: list[dict]) -> str:
    keep = ("id", "source", "hypothesis", "reference")
    clean = []
    for row in rows:
        item = {k: row[k] for k in keep if k in row and str(row.get(k, "")).strip()}
        clean.append(item)
    return INSTRUCTION + json.dumps(clean, ensure_ascii=False)


def score_to_grade(value: float) -> str:
    return GRADES[max(0, min(len(GRADES) - 1, int(round(value)) - 1))]


def summarize(records: list[dict], mapping: dict[str, dict] | None = None) -> dict:
    import pandas as pd
    df = pd.DataFrame(records).drop_duplicates("id")
    if mapping:
        meta = df["id"].map(lambda x: mapping.get(str(x), {}))
        for key in ("config", "seed", "dataset", "pair_type", "eval_id"):
            df[key] = [m.get(key, "") for m in meta]
    else:
        df["config"] = ""
        df["seed"] = ""
        df["dataset"] = ""
        df["pair_type"] = ""

    def agg(part):
        mean = float(part["score"].mean())
        counts = {g: int((part["rating"] == g).sum()) for g in GRADES}
        return {"n": int(len(part)), "mean": round(mean, 3),
                "mean_rating": score_to_grade(mean), "rating_counts": counts}

    result = {
        "overall": agg(df),
        "by_dataset": {str(k): agg(v) for k, v in df.groupby("dataset")},
        "by_pair_type": {str(k): agg(v) for k, v in df.groupby("pair_type")},
        "by_config": {str(k): agg(v) for k, v in df.groupby("config")},
        "by_seed": {str(k): agg(v) for k, v in df.groupby("seed")},
        "by_config_seed": {f"{c}::{s}": agg(v)
                           for (c, s), v in df.groupby(["config", "seed"])},
        "by_stratum": {f"{c}::{d}::{t}": agg(v)
                       for (c, d, t), v in df.groupby(["config", "dataset", "pair_type"])},
        "rows": df.to_dict("records"),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--keys", default="keys.txt")
    ap.add_argument("--out-dir", default="data/cleaned/judge")
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--map", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    batch_dir = out_dir / "batches_credit"
    batch_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "judge_credit_state.json"

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    log(f"input {len(rows)} dòng | model {args.model} | batch {args.batch_size}")

    done: set[str] = set()
    if state_path.exists():
        try:
            done = set(json.loads(state_path.read_text(encoding="utf-8")).get("done", []))
        except json.JSONDecodeError:
            done = set()
    rows = [r for r in rows if str(r.get("id")) not in done]
    log(f"còn {len(rows)} dòng chưa chấm (resume {len(done)})")

    keys = load_keys(Path(args.keys))
    if not keys:
        log("không tìm thấy key")
        return 2
    log(f"nạp {len(keys)} key")

    if args.dry_run:
        pool = KeyPool(keys)
        prompt = build_prompt(rows[:args.batch_size])
        log(f"dry-run OK | prompt {len(prompt)} ký tự | key mẫu {pool.mask(keys[0])}")
        return 0

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log("thiếu google-genai: pip install google-genai")
        return 2

    pool = KeyPool(keys)
    clients: dict[str, object] = {}

    def client_for(key: str):
        if key not in clients:
            clients[key] = genai.Client(api_key=key)
        return clients[key]

    pending = rows
    # đếm số lần THẤT BẠI LIÊN TIẾP, không phải tổng số batch,
    # để chấm được toàn bộ input dài mà vẫn dừng khi API hỏng.
    failures = 0
    batch_index = len(list(batch_dir.glob("batch_*.json")))
    while pending and failures < args.max_retries * 4:
        batch = pending[:args.batch_size]
        prompt = build_prompt(batch)
        key = pool.acquire()
        try:
            response = client_for(key).models.generate_content(
                model=args.model, contents=prompt,
                config=types.GenerateContentConfig(temperature=args.temperature))
            got = parse_credit(getattr(response, "text", "") or "")
            got_ids = {r["id"] for r in got}
            want_ids = {str(r["id"]) for r in batch}
            if not got:
                raise ValueError("response không có JSON hợp lệ")
            pool.reward(key)
            batch_path = batch_dir / f"batch_{batch_index:04d}.json"
            batch_path.write_text(json.dumps(got, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            batch_index += 1
            done |= (got_ids & want_ids)
            state_path.write_text(json.dumps({"done": sorted(done), "updated_at": ts()},
                                             ensure_ascii=False), encoding="utf-8")
            log(f"batch {batch_index}: {len(got_ids & want_ids)}/{len(want_ids)}")
            missing = [r for r in batch if str(r["id"]) not in got_ids]
            pending = missing + pending[len(batch):]
            failures = 0 if (got_ids & want_ids) else failures + 1
        except Exception as error:  # noqa: BLE001
            failures += 1
            kind = classify_error(error)
            pool.penalize(key, str(error), kind)
            if kind == "other":
                log(f"lỗi khác: {type(error).__name__}: {str(error)[:160]}")
                # 500/intermittent: cho key tạm nghỉ để xoay sang key khác
                pool.cooldown_until[key] = time.time() + 15
                time.sleep(2 + random.random() * 2)

    records = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        try:
            records.extend(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue

    mapping = None
    if args.map and Path(args.map).exists():
        mapping = {}
        for line in Path(args.map).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                mapping[str(item["judge_id"])] = item
    summary = summarize(records, mapping) if records else {"overall": {"n": 0}}
    result = {"model": args.model, "scale": "credit_AAA_C", "summary": summary,
              "key_usage": pool.summary(), "updated_at": ts()}
    (out_dir / "gemma_scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"xong: {summary['overall'].get('n', 0)} dòng | còn {len(pending)} chưa chấm")
    log(json.dumps(pool.summary(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
