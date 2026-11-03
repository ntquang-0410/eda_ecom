#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedup pair table + chunk description dài (≤800 token, không overlap).

- Description > 1280 token ước lượng được cắt theo câu/đoạn, pack thành chunk
  ≤800 token, không overlap.
- Sau đó dedup toàn bộ theo (source_text, target_text), giữ `dup_count`.
- Không đụng tới weight quantity (để khảo sát sau).

Output: data/cleaned/pairs_dedup/
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

SPACE = re.compile(r"\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?;。！？；])\s*|\n+")


def compact_len(text) -> int:
    return len(SPACE.sub("", str(text or "")))


def tokens(src: str, tgt: str) -> float:
    return compact_len(src) / 3.6 + compact_len(tgt) / 1.6


def sentence_weight(sentence: str, lang: str) -> float:
    return compact_len(sentence) / (3.6 if lang == "vi" else 1.6)


def split_long_sentence(sentence: str, lang: str, max_tokens: int) -> list[str]:
    """Cắt câu quá dài (không có dấu ngắt) theo từ (VI) hoặc ký tự (ZH)."""
    limit = int(max_tokens * (3.6 if lang == "vi" else 1.6))
    if compact_len(sentence) <= limit * 1.05:
        return [sentence]
    if lang == "vi":
        words = sentence.split()
        chunks, current, size = [], [], 0
        for word in words:
            if current and size + len(word) + 1 > limit:
                chunks.append(" ".join(current))
                current, size = [word], len(word)
            else:
                current.append(word)
                size += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks
    return [sentence[i:i + limit] for i in range(0, len(sentence), limit)]


def pack(sentences: list[str], k: int, lang: str) -> list[str]:
    if k <= 1 or len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []
    weights = [sentence_weight(s, lang) for s in sentences]
    total = sum(weights)
    target = total / k
    chunks, current, current_w = [], [], 0.0
    for sentence, weight in zip(sentences, weights):
        if current and current_w + weight > target and len(chunks) < k - 1:
            chunks.append(" ".join(current))
            current, current_w = [sentence], weight
        else:
            current.append(sentence)
            current_w += weight
    if current:
        chunks.append(" ".join(current))
    return chunks


def slice_proportional(text: str, lang: str, k: int) -> list[str]:
    """Cắt theo token mass (không theo ký tự) để cân bằng VI/ZH."""
    text = str(text or "")
    if k <= 1 or not text:
        return [text]
    per_char = 1 / (3.6 if lang == "vi" else 1.6)
    total = len(text) * per_char
    targets = [i * total / k for i in range(1, k)]
    cuts, cumulative, target_index = [0], 0.0, 0
    for index, _ in enumerate(text):
        cumulative += per_char
        while target_index < len(targets) and cumulative >= targets[target_index]:
            cuts.append(index + 1)
            target_index += 1
    cuts.append(len(text))
    pieces = []
    for i in range(len(cuts) - 1):
        piece = text[cuts[i]:cuts[i + 1]].strip()
        if piece:
            pieces.append(piece)
    return pieces


def chunk_pair(src: str, tgt: str, max_tokens: int = 800):
    total = tokens(src, tgt)
    if total <= 1280:
        return [(src, tgt)]
    k = max(2, math.ceil(total / max_tokens))
    src_sent = [s for s in SENT_SPLIT.split(str(src)) if s.strip()]
    tgt_sent = [s for s in SENT_SPLIT.split(str(tgt)) if s.strip()]
    src_sent = [piece for s in src_sent for piece in split_long_sentence(s, "vi", max_tokens)]
    tgt_sent = [piece for s in tgt_sent for piece in split_long_sentence(s, "zh", max_tokens)]
    src_parts = pack(src_sent, k, "vi")
    tgt_parts = pack(tgt_sent, k, "zh")
    if not src_parts or not tgt_parts:
        return [(src, tgt)]
    if len(src_parts) != len(tgt_parts):
        # cắt tỉ lệ để đảm bảo cùng số chunk và không phình kích thước
        src_parts = slice_proportional(src, "vi", k)
        tgt_parts = slice_proportional(tgt, "zh", k)

    # đảm bảo không chunk nào vượt 1280 token
    result = []
    for src_part, tgt_part in zip(src_parts, tgt_parts):
        if tokens(src_part, tgt_part) > 1280:
            kk = max(2, math.ceil(tokens(src_part, tgt_part) / max_tokens))
            result.extend(zip(slice_proportional(src_part, "vi", kk),
                              slice_proportional(tgt_part, "zh", kk)))
        else:
            result.append((src_part, tgt_part))
    return result


def process(path: Path, out: Path) -> dict:
    df = pd.read_parquet(path)
    rows = []
    chunked = 0
    for row in df.itertuples():
        if row.pair_type == "description" and row.tokens_est > 1280:
            parts = chunk_pair(row.source_text, row.target_text)
            if len(parts) > 1:
                chunked += 1
                for index, (src, tgt) in enumerate(parts):
                    record = row._asdict()
                    record.pop("Index", None)
                    record["pair_id"] = f"{row.pair_id}:chunk:{index}"
                    record["parent_pair_id"] = row.pair_id
                    record["chunk_index"] = index
                    record["n_chunks"] = len(parts)
                    record["source_text"] = src
                    record["target_text"] = tgt
                    rows.append(record)
                continue
        record = row._asdict()
        record.pop("Index", None)
        record["parent_pair_id"] = row.pair_id
        record["chunk_index"] = 0
        record["n_chunks"] = 1
        rows.append(record)

    data = pd.DataFrame(rows)
    data["src_chars"] = data["source_text"].map(compact_len)
    data["tgt_chars"] = data["target_text"].map(compact_len)
    data["tokens_est"] = [tokens(s, t) for s, t in zip(data["source_text"], data["target_text"])]
    data["dup_count"] = data.groupby(["source_text", "target_text"])["pair_id"].transform("size")
    before = len(data)
    data = data.drop_duplicates(["source_text", "target_text"], keep="first").reset_index(drop=True)
    data.to_parquet(out / path.name, index=False)
    return {"rows_before": before, "rows_after": int(len(data)),
            "dedup_removed": int(before - len(data)), "descriptions_chunked": chunked,
            "tokens_million": round(float(data["tokens_est"].sum()) / 1e6, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/cleaned/pairs")
    ap.add_argument("--out", default="data/cleaned/pairs_dedup")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in sorted(src.glob("*.parquet")):
        summary[path.stem] = process(path, out)
        print(path.name, summary[path.stem])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
