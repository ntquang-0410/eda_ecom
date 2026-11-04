#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chấm bản dịch bằng Gemma qua Google AI Studio API, tự xoay vòng nhiều API key.

Cách dùng tối thiểu:

    cp keys.example.txt keys.txt      # dán mỗi key 1 dòng
    python tools/gemma_judge.py \
        --input data/cleaned/judge/judge_input.jsonl \
        --out-dir data/cleaned/judge

Script sẽ:
  1. chia input thành batch (mặc định 25 dòng)
  2. gọi Gemma, parse JSON trả về
  3. gặp 429/quota -> đổi key khác, có cooldown tăng dần
  4. lưu từng batch, resume nếu chạy lại
  5. gộp điểm + in summary theo dataset/pair_type

Yêu cầu: pip install google-genai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

INSTRUCTION = """You are a strict bilingual evaluator for Vietnamese-to-Simplified-Chinese e-commerce translation.

For each item, score the hypothesis on three axes (1-5):
- adequacy: does it preserve the meaning of the source?
- fluency: is the Chinese natural and grammatical?
- terminology: are brand names, numbers, units and product terms correct?

Rules:
- Use the reference when provided; otherwise judge against the source only.
- Penalize untranslated Vietnamese, missing numbers/units, and hallucinated content.
- Do not reward literal word-for-word copying.

Return ONLY a JSON array, one object per input, with this schema:
{"id": "...", "adequacy": 1-5, "fluency": 1-5, "terminology": 1-5, "note": "<=20 words"}

Data:
"""

QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "exceeded",
                 "too many requests", "rate_limit")
AUTH_MARKERS = ("401", "403", "api key", "permission denied", "unauthenticated",
                "invalid key")


def now() -> float:
    return time.time()


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


class KeyPool:
    """Xoay vòng key, cooldown tăng dần, loại key hỏng vĩnh viễn."""

    def __init__(self, keys: list[str], base_cooldown: float = 90.0,
                 max_cooldown: float = 3600.0, all_exhausted_sleep: float = 60.0):
        if not keys:
            raise ValueError("không có API key nào")
        self.keys = keys
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self.all_exhausted_sleep = all_exhausted_sleep
        self.cooldown_until = {k: 0.0 for k in keys}
        self.failures = {k: 0 for k in keys}
        self.disabled = set()
        self.usage = {k: {"ok": 0, "quota": 0, "auth": 0, "other": 0} for k in keys}

    def mask(self, key: str) -> str:
        return key[:6] + "..." + key[-4:] if len(key) > 12 else key[:6] + "..."

    def acquire(self) -> str:
        while True:
            available = [k for k in self.keys
                         if k not in self.disabled and self.cooldown_until[k] <= now()]
            if available:
                return min(available, key=lambda k: self.cooldown_until[k])
            alive = [k for k in self.keys if k not in self.disabled]
            if not alive:
                raise RuntimeError("tất cả API key đều không dùng được")
            wait = min(self.cooldown_until[k] for k in alive) - now()
            wait = max(1.0, min(wait, self.all_exhausted_sleep))
            log(f"tất cả key đang cooldown, chờ {wait:.0f}s")
            time.sleep(wait)

    def penalize(self, key: str, error: str, kind: str = "quota") -> None:
        if kind == "auth":
            self.disabled.add(key)
            self.usage[key]["auth"] += 1
            log(f"key {self.mask(key)} bị vô hiệu hoá (auth): {error[:120]}")
            return
        if kind == "quota":
            self.failures[key] += 1
            cooldown = min(self.max_cooldown,
                           self.base_cooldown * (2 ** (self.failures[key] - 1)))
            self.cooldown_until[key] = now() + cooldown
            self.usage[key]["quota"] += 1
            log(f"key {self.mask(key)} hết quota, cooldown {cooldown:.0f}s")
            return
        self.usage[key]["other"] += 1

    def reward(self, key: str) -> None:
        self.failures[key] = 0
        self.cooldown_until[key] = 0.0
        self.usage[key]["ok"] += 1

    def summary(self) -> dict:
        return {
            "keys": len(self.keys),
            "disabled": len(self.disabled),
            "usage": {self.mask(k): v for k, v in self.usage.items()},
        }


def classify_error(error: Exception) -> str:
    text = str(error).lower()
    if any(marker in text for marker in AUTH_MARKERS):
        return "auth"
    if any(marker in text for marker in QUOTA_MARKERS):
        return "quota"
    return "other"


def strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_scores(text: str) -> list[dict]:
    value = strip_code_fence(text)
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
        try:
            rows.append({
                "id": str(item["id"]),
                "adequacy": max(1, min(5, int(round(float(item["adequacy"]))))),
                "fluency": max(1, min(5, int(round(float(item["fluency"]))))),
                "terminology": max(1, min(5, int(round(float(item["terminology"]))))),
                "note": str(item.get("note") or "")[:200],
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def load_keys(path: Path, env_name: str = "GEMINI_API_KEYS") -> list[str]:
    keys = []
    env = os.getenv(env_name, "")
    if env.strip():
        keys.extend([k.strip() for k in env.split(",") if k.strip()])
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    seen, unique = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def chunk(rows: list[dict], size: int):
    for index in range(0, len(rows), size):
        yield rows[index:index + size]


def build_prompt(rows: list[dict]) -> str:
    # chỉ gửi các trường cần chấm; không gửi config/seed/dataset để tránh bias
    keep = ("id", "source", "hypothesis", "reference")
    clean = []
    for row in rows:
        item = {k: row[k] for k in keep if k in row and str(row.get(k, "")).strip()}
        clean.append(item)
    return INSTRUCTION + json.dumps(clean, ensure_ascii=False)


def summarize(records: list[dict], mapping: dict[str, dict] | None = None) -> dict:
    import pandas as pd
    df = pd.DataFrame(records).drop_duplicates("id")
    if mapping:
        meta = df["id"].map(lambda x: mapping.get(str(x), {}))
        for key in ("config", "seed", "dataset", "pair_type", "eval_id"):
            df[key] = [m.get(key, "") for m in meta]
        df["mean"] = df[["adequacy", "fluency", "terminology"]].mean(axis=1)
        return _summarize_frame(df)

    def split_id(value: str):
        text = str(value)
        config = ""
        if "::" in text:
            config, text = text.split("::", 1)
        dataset = "1688" if text.startswith("1688") else ("tiki" if text.startswith("tiki") else "unknown")
        if ":title" in text or text.endswith(":title"):
            pair_type = "title"
        elif ":attr" in text:
            pair_type = "attribute"
        elif ":desc" in text:
            pair_type = "description"
        else:
            pair_type = "unknown"
        return config, dataset, pair_type

    parsed = df["id"].map(split_id)
    df["config"] = [x[0] for x in parsed]
    df["dataset"] = [x[1] for x in parsed]
    df["pair_type"] = [x[2] for x in parsed]
    df["mean"] = df[["adequacy", "fluency", "terminology"]].mean(axis=1)
    return _summarize_frame(df)


def _summarize_frame(df) -> dict:
    def agg(part):
        return {"n": int(len(part)), **{k: round(float(part[k].mean()), 3)
                                        for k in ("adequacy", "fluency", "terminology", "mean")}}

    return {
        "overall": agg(df),
        "by_dataset": {name: agg(part) for name, part in df.groupby("dataset")},
        "by_pair_type": {name: agg(part) for name, part in df.groupby("pair_type")},
        "by_config": {name: agg(part) for name, part in df.groupby("config")} if df["config"].astype(bool).any() else {},
        "by_seed": {str(name): agg(part) for name, part in df.groupby("seed")} if "seed" in df else {},
        "by_config_seed": {f"{c}::{s}": agg(part) for (c, s), part in df.groupby(["config", "seed"])} if "seed" in df else {},
        "by_stratum": {f"{c}::{d}::{t}": agg(part) for (c, d, t), part in df.groupby(["config", "dataset", "pair_type"])} if "config" in df else {},
        "rows": df.to_dict("records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL judge input")
    ap.add_argument("--keys", default="keys.txt")
    ap.add_argument("--out-dir", default="data/cleaned/judge")
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N dòng đầu (0 = tất cả)")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--map", default="", help="JSONL mapping judge_id -> metadata")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    batch_dir = out_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "judge_state.json"

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
        log("không tìm thấy key: tạo keys.txt hoặc set GEMINI_API_KEYS")
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
    attempt = 0
    batch_index = len(list(batch_dir.glob("batch_*.json")))
    while pending and attempt < args.max_retries * 4:
        attempt += 1
        batch = pending[:args.batch_size]
        prompt = build_prompt(batch)
        key = pool.acquire()
        try:
            response = client_for(key).models.generate_content(
                model=args.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=args.temperature),
            )
            got = parse_scores(getattr(response, "text", "") or "")
            got_ids = {r["id"] for r in got}
            want_ids = {str(r["id"]) for r in batch}
            if not got:
                raise ValueError("response không có JSON hợp lệ")
            pool.reward(key)

            batch_path = batch_dir / f"batch_{batch_index:04d}.json"
            batch_path.write_text(json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8")
            batch_index += 1

            done |= (got_ids & want_ids)
            state_path.write_text(json.dumps({"done": sorted(done), "updated_at": ts()},
                                             ensure_ascii=False), encoding="utf-8")
            log(f"batch {batch_index}: {len(got_ids & want_ids)}/{len(want_ids)} | còn {len(want_ids - got_ids)} thiếu")

            # dòng thiếu trong batch -> đẩy lại đầu hàng đợi
            missing = [r for r in batch if str(r["id"]) not in got_ids]
            pending = missing + pending[len(batch):]
        except Exception as error:  # noqa: BLE001
            kind = classify_error(error)
            pool.penalize(key, str(error), kind)
            if kind == "other":
                log(f"lỗi khác: {type(error).__name__}: {str(error)[:160]}")
                time.sleep(2 + random.random() * 2)

    # gộp kết quả
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
    result = {"model": args.model, "summary": summary, "key_usage": pool.summary(),
              "updated_at": ts()}
    (out_dir / "gemma_scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    log(f"xong: {summary['overall'].get('n', 0)} dòng | còn {len(pending)} chưa chấm")
    log(json.dumps(pool.summary(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
