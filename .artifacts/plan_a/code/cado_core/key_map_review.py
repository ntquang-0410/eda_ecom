# -*- coding: utf-8 -*-
"""Plan B0 — review key map bằng Qwen qua Alibaba Cloud DashScope.

Không có chuyên gia: dùng LLM làm reviewer có kiểm soát.

* Input: ``tiki_key_map_review.csv`` + sample values từ Plan A.
* Gọi Qwen qua DashScope OpenAI-compatible API.
* Chạy silver evaluation trên các key 1688 đã biết trước khi auto-accept.
* Consensus nhiều vote, validation CJK/key format.
* Chỉ apply high/medium-confidence; low giữ pending.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from .config import (
    KEY_MAP_FINAL_CSV,
    KEY_MAP_QWEN_REVIEW_CSV,
    KEY_MAP_QWEN_STATS_JSON,
    KEY_MAP_QWEN_VOTES_CSV,
    KEY_MAP_REVIEW_INPUT_CSV,
    KEY_MAP_SILVER_EVAL_CSV,
    LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV,
    TIKI_BT_UNITS_PLAN_A_PARQUET,
    TIKI_KEY_MAP_CSV,
)
from .tiki_units import normalize_key

DEFAULT_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
DEFAULT_MODEL = os.getenv("QWEN_REVIEW_MODEL", "qwen3.8-max")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_ALLOWED_KEY_RE = re.compile(
    r"^[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaffA-Za-z0-9+#&/()（）\-]+$"
)

class QuotaExhaustedError(RuntimeError):
    """DashScope báo hết quota; không retry vô ích."""


SYSTEM_PROMPT = """You are a Vietnamese–Chinese e-commerce terminology expert.
Given a Vietnamese product attribute key, its class, category context, sample values,
and similar 1688 key pairs, propose the most likely short Simplified Chinese key.

Rules:
- Return Simplified Chinese used in Chinese e-commerce listings.
- Keep it short: 1-10 Chinese characters (Latin brand/model tokens allowed).
- No punctuation except parentheses/hyphen when unavoidable.
- Do not translate sample values.
- If uncertain, return key_zh = null.
- Return JSON only, with no reasoning or markdown.

JSON schema:
{"key_zh": "材质", "alternatives": ["材料"], "confidence": 0.0, "reason": "short reason"}
"""


def _extract_json(text: object) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_key_zh(value: object) -> tuple[str, str]:
    """Trả về ``(normalized_key, status)``; status rỗng nếu hợp lệ."""
    text = str(value or "").strip()
    if not text:
        return "", "empty"
    text = re.sub(r"\s+", "", text)
    if len(text) > 16:
        return text, "too_long"
    if not _ALLOWED_KEY_RE.match(text):
        return text, "invalid_chars"
    if not _CJK_RE.search(text):
        return text, "non_cjk"
    return text, ""


def normalize_key_zh(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def _similarity(left: str, right: str) -> float:
    left = normalize_key(left)
    right = normalize_key(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    else:
        jaccard = 0.0
    containment = 1.0 if left in right or right in left else 0.0
    return round(0.65 * jaccard + 0.35 * containment, 4)


def _sample_values(group: pd.DataFrame, limit: int = 5) -> list[str]:
    if "value_vi" not in group.columns:
        return []
    values = (
        group["value_vi"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values != ""]
    if values.empty:
        return []
    return values.value_counts().head(limit).index.tolist()


def _similar_pairs(
    key_norm: str,
    alignment: pd.DataFrame,
    *,
    limit: int = 5,
    exclude: str | None = None,
) -> list[dict[str, Any]]:
    if alignment.empty:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    exclude_norm = normalize_key(exclude) if exclude else ""
    for row in alignment.to_dict("records"):
        vi = normalize_key(row.get("key_vi_norm", ""))
        if not vi or vi == exclude_norm:
            continue
        score = _similarity(key_norm, vi)
        if score <= 0:
            continue
        scored.append((
            score,
            {
                "key_vi": vi,
                "key_zh": str(row.get("key_zh_norm", "")).strip(),
                "freq": int(float(row.get("freq", 0) or 0)),
            },
        ))
    scored.sort(key=lambda item: (-item[0], -item[1]["freq"]))
    return [item[1] for item in scored[:limit]]


def build_review_input(
    units: pd.DataFrame,
    review: pd.DataFrame,
    *,
    alignment_map_path: str | Path = LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV,
    sample_values: int = 5,
    similar_pairs: int = 5,
) -> pd.DataFrame:
    """Tạo prompt input cho từng key pending/known."""
    if "key_norm" not in review.columns:
        raise KeyError("review cần cột key_norm")
    alignment = pd.DataFrame()
    alignment_path = Path(alignment_map_path)
    if alignment_path.exists():
        alignment = pd.read_csv(
            alignment_path, dtype=str, encoding="utf-8-sig"
        ).fillna("")
        if "key_zh_norm" in alignment.columns:
            alignment["key_zh_norm"] = alignment["key_zh_norm"].map(normalize_key)
        if "key_vi_norm" in alignment.columns:
            alignment["key_vi_norm"] = alignment["key_vi_norm"].map(normalize_key)

    groups = {
        normalize_key(key): group
        for key, group in units.groupby("key_norm", dropna=False)
        if normalize_key(key)
    }
    records: list[dict[str, Any]] = []
    for row in review.to_dict("records"):
        key_norm = normalize_key(row.get("key_norm"))
        if not key_norm:
            continue
        group = groups.get(key_norm)
        sample_values_list = _sample_values(group, sample_values) if group is not None else []
        categories = (
            group["category"].dropna().astype(str).value_counts().head(3).index.tolist()
            if group is not None and "category" in group.columns
            else []
        )
        records.append({
            "key_norm": key_norm,
            "key_vi": row.get("key_vi", ""),
            "key_class": row.get("key_class", ""),
            "n_units": int(row.get("n_units", 0) or 0),
            "n_products": int(row.get("n_products", 0) or 0),
            "n_train_eligible": int(row.get("n_train_eligible", 0) or 0),
            "sample_values_json": json.dumps(sample_values_list, ensure_ascii=False),
            "categories_json": json.dumps(categories, ensure_ascii=False),
            "similar_1688_pairs_json": json.dumps(
                _similar_pairs(
                    key_norm,
                    alignment,
                    limit=similar_pairs,
                    exclude=key_norm,
                ),
                ensure_ascii=False,
            ),
        })
    return pd.DataFrame(records)


def build_silver_input(
    key_map: pd.DataFrame,
    units: pd.DataFrame,
    *,
    n: int = 100,
    seed: int = 42,
    alignment_map_path: str | Path = LENGTH_PRIOR_V3_ALIGNMENT_MAP_CSV,
) -> pd.DataFrame:
    """Lấy key đã biết làm silver set, ẩn nhãn khi prompt."""
    known = key_map[
        key_map["key_zh"].notna()
        & (key_map["key_zh"].astype(str).str.strip() != "")
        & (key_map["key_map_status"].isin(["manual", "manual_seed", "auto_1688"]))
    ].copy()
    if "auto_share" in known.columns:
        known = known[
            pd.to_numeric(known["auto_share"], errors="coerce").fillna(1.0) >= 0.80
        ]
    if len(known) > n:
        known = known.sample(n=n, random_state=seed)
    review = known.rename(columns={"key_zh": "gold_key_zh"}).copy()
    prompt = build_review_input(
        units,
        review,
        alignment_map_path=alignment_map_path,
    )
    return prompt.merge(
        review[["key_norm", "gold_key_zh"]],
        on="key_norm",
        how="left",
    )


def _review_messages(row: dict[str, Any], variant: int = 0) -> list[dict[str, str]]:
    payload = {
        "key_vi": row.get("key_vi", ""),
        "key_class": row.get("key_class", ""),
        "n_units": row.get("n_units", 0),
        "n_products": row.get("n_products", 0),
        "sample_values": json.loads(row.get("sample_values_json", "[]") or "[]"),
        "categories": json.loads(row.get("categories_json", "[]") or "[]"),
        "similar_1688_pairs": json.loads(
            row.get("similar_1688_pairs_json", "[]") or "[]"
        ),
    }
    extra = {
        0: "",
        1: "Focus on concise 1688/Tmall terminology. Prefer the common listing key.",
        2: "Be conservative. If the key is ambiguous, return key_zh=null.",
    }.get(variant, "")
    user = (
        f"Review variant {variant}. {extra}\n"
        "Return the JSON object now.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass
class QwenReviewer:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 90.0
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 256,
                }
                if not self.enable_thinking:
                    # DashScope/Qwen3 hỗ trợ enable_thinking; nếu model/endpoint
                    # không hỗ trợ, fallback sang request thường.
                    try:
                        response = self.client.chat.completions.create(
                            **kwargs,
                            extra_body={"enable_thinking": False},
                        )
                    except Exception:
                        response = self.client.chat.completions.create(**kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return str(content or "")
            except Exception as error:  # noqa: BLE001 - API/network errors
                message = str(error)
                if (
                    "insufficient_quota" in message
                    or "quota has been exhausted" in message.lower()
                    or "token-plan" in message.lower() and "quota" in message.lower()
                ):
                    raise QuotaExhaustedError(message) from error
                last_error = error
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Qwen API failed: {last_error}") from last_error

    def review(
        self,
        row: dict[str, Any],
        *,
        votes: int = 3,
        progress: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for vote in range(votes):
            variant = vote % 3
            try:
                raw = self.chat(
                    _review_messages(row, variant),
                    temperature=0.0 if vote == 0 else 0.2,
                )
                payload = _extract_json(raw) or {}
                key_zh, validation = validate_key_zh(payload.get("key_zh"))
                alternatives = payload.get("alternatives") or []
                if not isinstance(alternatives, list):
                    alternatives = []
                confidence = float(payload.get("confidence", 0) or 0)
                results.append({
                    "key_norm": row["key_norm"],
                    "vote": vote,
                    "model": self.model,
                    "key_zh": key_zh,
                    "alternatives": json.dumps(
                        [str(item) for item in alternatives[:5]],
                        ensure_ascii=False,
                    ),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "validation_status": validation or "valid",
                    "raw_output": raw,
                    "error": "",
                })
            except Exception as error:  # noqa: BLE001
                results.append({
                    "key_norm": row["key_norm"],
                    "vote": vote,
                    "model": self.model,
                    "key_zh": "",
                    "alternatives": "[]",
                    "confidence": 0.0,
                    "validation_status": "api_error",
                    "raw_output": "",
                    "error": str(error),
                })
            if progress:
                progress(f"  review {row['key_norm']} vote {vote + 1}/{votes}")
        return results


def run_reviews(
    reviewer: QwenReviewer,
    review_input: pd.DataFrame,
    *,
    votes: int = 3,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Review song song nhiều key; mỗi key trả về nhiều vote."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = review_input.to_dict("records")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(reviewer.review, row, votes=votes, progress=None): row
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.extend(future.result())
            except QuotaExhaustedError as error:
                results.append({
                    "key_norm": row.get("key_norm", ""),
                    "vote": -1,
                    "model": reviewer.model,
                    "key_zh": "",
                    "alternatives": "[]",
                    "confidence": 0.0,
                    "validation_status": "quota_exhausted",
                    "raw_output": "",
                    "error": str(error),
                })
                for pending in futures:
                    pending.cancel()
            except Exception as error:  # noqa: BLE001
                results.append({
                    "key_norm": row.get("key_norm", ""),
                    "vote": -1,
                    "model": reviewer.model,
                    "key_zh": "",
                    "alternatives": "[]",
                    "confidence": 0.0,
                    "validation_status": "api_error",
                    "raw_output": "",
                    "error": str(error),
                })
            completed += 1
            if progress and (completed % 10 == 0 or completed == len(rows)):
                progress(f"  reviewed {completed:,}/{len(rows):,} key")
    return pd.DataFrame(results)


def aggregate_reviews(
    votes: pd.DataFrame,
    review_input: pd.DataFrame,
    *,
    votes_expected: int = 3,
) -> pd.DataFrame:
    if votes.empty:
        return review_input.assign(
            qwen_key_zh="", qwen_status="qwen_low", qwen_votes=0,
            qwen_agreement=0.0, qwen_confidence=0.0, qwen_alternatives="[]",
        )
    valid = votes[votes["validation_status"] == "valid"].copy()
    rows: list[dict[str, Any]] = []
    for key_norm, group in votes.groupby("key_norm"):
        valid_group = valid[valid["key_norm"] == key_norm]
        counts = (
            valid_group["key_zh"].value_counts()
            if not valid_group.empty
            else pd.Series(dtype="int64")
        )
        top_key = str(counts.index[0]) if len(counts) else ""
        top_count = int(counts.iloc[0]) if len(counts) else 0
        total_votes = int(len(group))
        n_valid = int(len(valid_group))
        agreement = top_count / total_votes if total_votes else 0.0
        confidence = (
            float(valid_group["confidence"].mean()) if n_valid else 0.0
        )
        if top_count >= max(2, math.ceil(total_votes * 2 / 3)):
            status = "qwen_high" if agreement >= 0.66 else "qwen_medium"
        elif top_count >= 2:
            status = "qwen_medium"
        else:
            status = "qwen_low"
        rows.append({
            "key_norm": key_norm,
            "qwen_key_zh": top_key,
            "qwen_status": status,
            "qwen_votes": n_valid,
            "qwen_total_votes": total_votes,
            "qwen_agreement": round(agreement, 4),
            "qwen_confidence": round(confidence, 4),
            "qwen_alternatives": json.dumps(
                [str(item) for item in counts.index[:3].tolist()]
                if len(counts)
                else [],
                ensure_ascii=False,
            ),
        })
    aggregated = pd.DataFrame(rows)
    return review_input.merge(aggregated, on="key_norm", how="left").fillna({
        "qwen_key_zh": "",
        "qwen_status": "qwen_low",
        "qwen_votes": 0,
        "qwen_total_votes": 0,
        "qwen_agreement": 0.0,
        "qwen_confidence": 0.0,
        "qwen_alternatives": "[]",
    })


def evaluate_silver(silver: pd.DataFrame, votes: pd.DataFrame) -> dict[str, Any]:
    if silver.empty:
        return {"n": 0, "exact": 0.0, "normalized": 0.0, "high_precision": 0.0}
    aggregated = aggregate_reviews(
        votes[votes["key_norm"].isin(silver["key_norm"])],
        silver,
        votes_expected=3,
    )
    merged = aggregated.merge(
        silver[["key_norm", "gold_key_zh"]],
        on="key_norm",
        how="left",
        suffixes=("", "_silver"),
    )
    gold = merged["gold_key_zh"].map(normalize_key_zh)
    pred = merged["qwen_key_zh"].map(normalize_key_zh)
    exact = (gold == pred) & (gold != "")
    high = merged["qwen_status"] == "qwen_high"
    return {
        "n": int(len(merged)),
        "exact": round(float(exact.mean()), 4) if len(merged) else 0.0,
        "normalized": round(float(exact.mean()), 4) if len(merged) else 0.0,
        "high_n": int(high.sum()),
        "high_precision": round(
            float(exact[high].mean()), 4
        ) if high.any() else 0.0,
    }


def apply_reviews(
    key_map: pd.DataFrame,
    review: pd.DataFrame,
    *,
    statuses: Sequence[str] = ("qwen_high", "qwen_medium"),
) -> pd.DataFrame:
    result = key_map.copy()
    proposal = {
        str(row["key_norm"]): row
        for row in review.to_dict("records")
        if row.get("qwen_key_zh")
    }
    for index, row in result.iterrows():
        key_norm = normalize_key(row.get("key_norm"))
        item = proposal.get(key_norm)
        if item is None:
            continue
        if str(row.get("key_zh", "")).strip():
            continue
        if item.get("qwen_status") not in statuses:
            continue
        result.at[index, "key_zh"] = item["qwen_key_zh"]
        result.at[index, "key_map_status"] = item["qwen_status"]
        result.at[index, "source"] = "qwen_review"
        result.at[index, "auto_key_zh"] = item.get("qwen_key_zh", "")
        result.at[index, "auto_freq"] = int(item.get("qwen_votes", 0) or 0)
        result.at[index, "auto_share"] = float(item.get("qwen_agreement", 0) or 0)
    return result


def write_review_artifacts(
    *,
    review_input: pd.DataFrame,
    silver: pd.DataFrame,
    votes: pd.DataFrame,
    review: pd.DataFrame,
    key_map: pd.DataFrame,
    final_map: pd.DataFrame,
    stats: dict[str, Any],
    input_path: str | Path = KEY_MAP_REVIEW_INPUT_CSV,
    silver_path: str | Path = KEY_MAP_SILVER_EVAL_CSV,
    votes_path: str | Path = KEY_MAP_QWEN_VOTES_CSV,
    review_path: str | Path = KEY_MAP_QWEN_REVIEW_CSV,
    final_path: str | Path = KEY_MAP_FINAL_CSV,
    stats_path: str | Path = KEY_MAP_QWEN_STATS_JSON,
    write_key_map: bool = True,
) -> dict[str, str]:
    paths = {
        "input": Path(input_path),
        "silver": Path(silver_path),
        "votes": Path(votes_path),
        "review": Path(review_path),
        "final": Path(final_path),
        "key_map": Path(TIKI_KEY_MAP_CSV),
        "stats": Path(stats_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    review_input.to_csv(paths["input"], index=False, encoding="utf-8-sig")
    silver.to_csv(paths["silver"], index=False, encoding="utf-8-sig")
    votes.to_csv(paths["votes"], index=False, encoding="utf-8-sig")
    review.to_csv(paths["review"], index=False, encoding="utf-8-sig")
    final_map.to_csv(paths["final"], index=False, encoding="utf-8-sig")
    # Ghi final map vào key map chuẩn để Plan A dùng lại.
    if write_key_map:
        final_map.to_csv(paths["key_map"], index=False, encoding="utf-8-sig")
    with paths["stats"].open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return {key: str(value) for key, value in paths.items()}
