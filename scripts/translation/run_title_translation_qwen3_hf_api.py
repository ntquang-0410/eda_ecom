"""Run resumable Qwen3-14B title translation through HF Inference Providers.

This is an opt-in, no-local-GPU alternative to the Colab AWQ notebook. It is
deliberately inert unless the caller supplies both --allow-paid-inference and
an explicit --max-cost-usd. No token or product text is written to logs.
"""

from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import requests

import title_translation_pipeline as quality


ROOT = Path(os.environ.get("ECOM_WORKSPACE") or _repo_root)


def default_source_path(root: Path = ROOT, dataset_file: str = quality.DATASET_FILE) -> Path:
    """Resolve the pinned dataset whether stored at its repo path or project root.

    ``quality.load_source`` already supports the flat project-root layout. Keep
    the CLI default consistent with that behavior so a no-GPU dry run/full run
    does not fail before reaching the source checksum validation.
    """
    configured = root / dataset_file
    flat_copy = root / Path(dataset_file).name
    return next((path for path in (configured, flat_copy) if path.is_file()), configured)


DEFAULT_SOURCE = default_source_path()
DEFAULT_WORKDIR = ROOT / "title_translation_qwen3_hf_api_run"
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "Qwen/Qwen3-14B:deepinfra"
MODEL_DISPLAY_ID = "Qwen/Qwen3-14B"
PROMPT_VERSION = "qwen3-14b-hf-provider-zh-vi-nonthinking-sampled-v1"
BASE_MAX_TOKENS = 128
TRUNCATION_RETRY_TOKENS = 256
MAX_ATTEMPTS = 3
HTTP_TIMEOUT_SECONDS = 90
CHECKPOINT_EVERY = 50
INPUT_USD_PER_MILLION = 0.12
OUTPUT_USD_PER_MILLION = 0.24
OUTPUT_COLUMN = "title_vi_candidate_qwen3_14b_api"
TEMPERATURE = 0.7
TOP_P = 0.8
FIRST_ATTEMPT_PRESENCE_PENALTY = 0.3
RETRY_PRESENCE_PENALTY = 0.8

SYSTEM_PROMPT = (
    "Translate one Chinese e-commerce product title into concise, natural Vietnamese. "
    "Return only the Vietnamese title, with no explanation or quotation marks. "
    "Preserve every number, currency amount, size, quantity, meaningful unit, brand, "
    "English brand/model term, and alphanumeric product code. Do not invent details."
)


class ApiRunError(RuntimeError):
    pass


class ProviderAuthError(ApiRunError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _current_code_sha256() -> str:
    digest = hashlib.sha256()
    for source_path in (Path(__file__), Path(quality.__file__)):
        resolved = source_path.resolve()
        digest.update(str(resolved.name).encode("utf-8"))
        digest.update(resolved.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _redact(value: Any) -> str:
    text = str(value)
    text = re.sub(r"hf_[A-Za-z0-9]{16,}", "[REDACTED_HF_TOKEN]", text)
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    return text[:2000]


def config_payload(provider_policy: str, workers: int) -> dict[str, Any]:
    return {
        "dataset_id": quality.DATASET_ID,
        "dataset_revision": quality.DATASET_REVISION,
        "dataset_sha256": quality.DATASET_SHA256,
        "model_id": MODEL_DISPLAY_ID,
        "provider_policy": provider_policy,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "presence_penalty_first_attempt": FIRST_ATTEMPT_PRESENCE_PENALTY,
            "presence_penalty_retry": RETRY_PRESENCE_PENALTY,
            "max_tokens": BASE_MAX_TOKENS,
            "truncation_retry_tokens": TRUNCATION_RETRY_TOKENS,
            "max_attempts": MAX_ATTEMPTS,
            "workers": int(workers),
        },
        "qa_version": quality.QA_VERSION,
    }


def config_id(payload: dict[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def check_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source parquet not found: {path}")
    actual_sha = quality.sha256_file(path)
    if actual_sha != quality.DATASET_SHA256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {quality.DATASET_SHA256}, got {actual_sha}"
        )
    frame = pd.read_parquet(path)
    quality.validate_input_frame(frame)
    return frame


def source_titles(frame: pd.DataFrame, stage: str) -> list[str]:
    if stage == "smoke":
        return list(quality.build_sample_manifest(frame)["smoke"])
    return list(dict.fromkeys(frame["title_zh"].astype(str).tolist()))


class TranslationCache:
    """SQLite cache scoped by the exact run configuration; safe to resume."""

    def __init__(self, path: Path, run_config_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_config_id = run_config_id
        with self.connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS translations (
                    config_id TEXT NOT NULL,
                    source_title TEXT NOT NULL,
                    output_text TEXT,
                    status TEXT NOT NULL,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    provider TEXT,
                    model TEXT,
                    error_text TEXT,
                    updated_at_utc TEXT NOT NULL,
                    PRIMARY KEY (config_id, source_title)
                )"""
            )
            db.commit()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def get_many(self, titles: list[str]) -> dict[str, dict[str, Any]]:
        if not titles:
            return {}
        result: dict[str, dict[str, Any]] = {}
        with self.connect() as db:
            for start in range(0, len(titles), 400):
                group = titles[start : start + 400]
                marks = ",".join("?" for _ in group)
                rows = db.execute(
                    "SELECT source_title,output_text,status,truncated,input_tokens,output_tokens,"
                    "estimated_cost_usd,attempts,provider,model,error_text,updated_at_utc "
                    f"FROM translations WHERE config_id=? AND source_title IN ({marks})",
                    [self.run_config_id, *group],
                ).fetchall()
                for row in rows:
                    result[row[0]] = {
                        "output_text": row[1], "status": row[2], "truncated": bool(row[3]),
                        "input_tokens": row[4], "output_tokens": row[5],
                        "estimated_cost_usd": row[6], "attempts": row[7],
                        "provider": row[8], "model": row[9], "error_text": row[10],
                        "updated_at_utc": row[11],
                    }
        return result

    def put(self, title: str, record: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO translations (
                    config_id,source_title,output_text,status,truncated,input_tokens,
                    output_tokens,estimated_cost_usd,attempts,provider,model,error_text,updated_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(config_id,source_title) DO UPDATE SET
                    output_text=excluded.output_text,status=excluded.status,
                    truncated=excluded.truncated,
                    input_tokens=translations.input_tokens+excluded.input_tokens,
                    output_tokens=translations.output_tokens+excluded.output_tokens,
                    estimated_cost_usd=translations.estimated_cost_usd+excluded.estimated_cost_usd,
                    attempts=translations.attempts+excluded.attempts,
                    provider=excluded.provider,model=excluded.model,
                    error_text=excluded.error_text,updated_at_utc=excluded.updated_at_utc""",
                (
                    self.run_config_id, title, record.get("output_text"),
                    record.get("status", "error"), int(bool(record.get("truncated"))),
                    int(record.get("input_tokens", 0)), int(record.get("output_tokens", 0)),
                    float(record.get("estimated_cost_usd", 0.0)),
                    int(record.get("attempts", 0)), record.get("provider"),
                    record.get("model"), record.get("error_text"), _utc_now(),
                ),
            )
            db.commit()

    def complete(self, record: dict[str, Any] | None) -> bool:
        if not record or record.get("status") != "ok" or record.get("truncated"):
            return False
        text = str(record.get("output_text") or "").strip()
        return bool(text and not quality._repeated_phrase(text))

    def totals(self) -> dict[str, float | int]:
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*),SUM(CASE WHEN status='ok' AND truncated=0 THEN 1 ELSE 0 END),"
                "SUM(CASE WHEN status!='ok' OR truncated=1 THEN 1 ELSE 0 END),"
                "COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),"
                "COALESCE(SUM(estimated_cost_usd),0) FROM translations WHERE config_id=?",
                (self.run_config_id,),
            ).fetchone()
        return {
            "cached_records": int(row[0] or 0), "completed_records": int(row[1] or 0),
            "error_records": int(row[2] or 0), "input_tokens": int(row[3] or 0),
            "output_tokens": int(row[4] or 0), "estimated_cost_usd": float(row[5] or 0),
        }

    def total_estimated_cost_usd(self) -> float:
        """Include all run configurations and retries stored in this cache."""
        with self.connect() as db:
            row = db.execute(
                "SELECT COALESCE(SUM(estimated_cost_usd),0) FROM translations"
            ).fetchone()
        return float(row[0] or 0.0)


def request_payload(title: str, *, max_tokens: int, presence_penalty: float) -> dict[str, Any]:
    user_text = f"{title}\n/no_think"
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "stream": False,
    }


def estimate_max_request_cost(title: str, workers: int) -> float:
    # Conservative in-flight reservation only; actual spend is computed from
    # provider usage. The small concurrency window bounds cap overshoot.
    estimated_input_tokens = max(160, min(2048, 160 + 2 * len(title)))
    per_attempt = (
        estimated_input_tokens * INPUT_USD_PER_MILLION
        + TRUNCATION_RETRY_TOKENS * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return per_attempt * MAX_ATTEMPTS * max(1, workers)


def _cost_from_usage(usage: dict[str, Any], payload: dict[str, Any]) -> tuple[int, int, float]:
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    provider_cost = usage.get("estimated_cost")
    if provider_cost is None:
        provider_cost = (
            input_tokens * INPUT_USD_PER_MILLION
            + output_tokens * OUTPUT_USD_PER_MILLION
        ) / 1_000_000
    return input_tokens, output_tokens, max(0.0, float(provider_cost or 0.0))


def call_provider(
    title: str,
    token: str,
    *,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    """Translate one title, retrying only truncation/repetition/temporary errors."""
    total_input = total_output = attempts = 0
    total_cost = 0.0
    last_error = None
    last_text = ""
    last_truncated = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        repeated_retry = attempt > 1 and not last_truncated
        max_tokens = TRUNCATION_RETRY_TOKENS if last_truncated else BASE_MAX_TOKENS
        presence_penalty = RETRY_PRESENCE_PENALTY if repeated_retry else FIRST_ATTEMPT_PRESENCE_PENALTY
        payload = request_payload(title, max_tokens=max_tokens, presence_penalty=presence_penalty)
        try:
            response = post(
                API_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {_redact(exc)}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"HF Inference Providers auth/permission error (HTTP {response.status_code}); "
                "check token permission and account billing."
            )
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}: {_redact(response.text)}"
            if attempt < MAX_ATTEMPTS:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 1.0), 30.0)
                except (TypeError, ValueError):
                    delay = min(2 ** (attempt - 1), 8)
                time.sleep(delay)
                continue
            break
        if not response.ok:
            last_error = f"HTTP {response.status_code}: {_redact(response.text)}"
            break

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            last_error = "Provider response has no choices."
            continue
        choice = choices[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "").strip()
        finish_reason = str(choice.get("finish_reason") or "")
        usage = body.get("usage") or {}
        in_tokens, out_tokens, cost = _cost_from_usage(usage, payload)
        total_input += in_tokens
        total_output += out_tokens
        total_cost += cost
        last_text = text
        last_truncated = finish_reason in {"length", "max_tokens"}
        qa = quality.check_translation(title, text, generation_truncated=last_truncated)
        repeated = quality._repeated_phrase(text)
        if text and not last_truncated and not repeated:
            return {
                "output_text": text, "status": "ok", "truncated": False,
                "input_tokens": total_input, "output_tokens": total_output,
                "estimated_cost_usd": total_cost, "attempts": attempts,
                "provider": response.headers.get("x-inference-provider", "deepinfra"),
                "model": body.get("model", MODEL_DISPLAY_ID), "error_text": None,
                "qa": qa,
            }
        last_error = "truncated_output" if last_truncated else (
            "repeated_phrase_guard" if repeated else "empty_output"
        )
        if attempt < MAX_ATTEMPTS and last_truncated and attempt == 1:
            continue
        if attempt < MAX_ATTEMPTS and repeated:
            continue
    return {
        "output_text": last_text, "status": "error", "truncated": last_truncated,
        "input_tokens": total_input, "output_tokens": total_output,
        "estimated_cost_usd": total_cost, "attempts": attempts,
        "provider": "deepinfra", "model": MODEL_DISPLAY_ID,
        "error_text": _redact(last_error or "generation_failed"),
    }


def _write_status(
    workdir: Path,
    status: dict[str, Any],
    mirror_dir: Path | None = None,
) -> None:
    payload = {"updated_at_utc": _utc_now(), **status}
    _atomic_json(workdir / "qwen_api_live_status.json", payload)
    if mirror_dir and mirror_dir.resolve() != workdir.resolve():
        _atomic_json(mirror_dir / "qwen_api_live_status.json", payload)


def _mirror_file(source_path: Path, mirror_dir: Path) -> Path:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    target = mirror_dir / source_path.name
    temp = target.with_suffix(target.suffix + ".uploading")
    shutil.copy2(source_path, temp)
    temp.replace(target)
    return target


def _mirror_cache(cache_path: Path, mirror_dir: Path | None) -> None:
    if not mirror_dir:
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    target = mirror_dir / cache_path.name
    temp = target.with_suffix(target.suffix + ".uploading")
    temp.unlink(missing_ok=True)
    source_db = sqlite3.connect(cache_path, timeout=60)
    target_db = sqlite3.connect(temp, timeout=60)
    try:
        source_db.backup(target_db)
    finally:
        target_db.close()
        source_db.close()
    temp.replace(target)


def _make_output_frame(
    frame: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    run_config_id: str,
) -> pd.DataFrame:
    output = frame.copy()
    unique_titles = list(dict.fromkeys(frame["title_zh"].astype(str).tolist()))
    missing = [title for title in unique_titles if title not in records or records[title].get("status") != "ok"]
    if missing:
        raise ApiRunError(f"Cannot export: {len(missing)} titles are not completed.")
    output[OUTPUT_COLUMN] = frame["title_zh"].astype(str).map(
        lambda title: records[title]["output_text"]
    )
    checks = {
        title: quality.check_translation(
            title, records[title]["output_text"],
            generation_truncated=bool(records[title].get("truncated")),
        )
        for title in unique_titles
    }
    output["title_vi_qwen3_14b_api_qa_status"] = frame["title_zh"].astype(str).map(
        lambda title: checks[title]["status"]
    )
    output["title_vi_qwen3_14b_api_qa_flags"] = frame["title_zh"].astype(str).map(
        lambda title: json.dumps(checks[title]["hard_flags"], ensure_ascii=False)
    )
    output["title_vi_qwen3_14b_api_advisory_flags"] = frame["title_zh"].astype(str).map(
        lambda title: json.dumps(checks[title].get("advisory_flags", []), ensure_ascii=False)
    )
    output["title_vi_qwen3_14b_api_model_id"] = MODEL_DISPLAY_ID
    output["title_vi_qwen3_14b_api_model_revision"] = "provider-managed; see run config and timestamp"
    output["title_vi_qwen3_14b_api_provider"] = frame["title_zh"].astype(str).map(
        lambda title: records[title].get("provider", "deepinfra")
    )
    output["title_vi_qwen3_14b_api_semantic_verified"] = False
    output["title_vi_qwen3_14b_api_pipeline_config_id"] = run_config_id
    return output


def run(
    *,
    source_path: Path,
    workdir: Path,
    stage: str,
    workers: int,
    max_cost_usd: float,
    provider_policy: str = "deepinfra",
    allow_paid_inference: bool = False,
    dry_run: bool = False,
    mirror_dir: Path | None = None,
    request_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be smoke or full")
    if not 1 <= int(workers) <= 32:
        raise ValueError("workers must be between 1 and 32")
    if not math.isfinite(float(max_cost_usd)) or float(max_cost_usd) <= 0:
        raise ValueError("max_cost_usd must be an explicit positive amount")
    if provider_policy not in {"deepinfra", "fastest", "cheapest"}:
        raise ValueError("provider_policy must be deepinfra, fastest, or cheapest")
    frame = check_source(source_path)
    titles = source_titles(frame, stage)
    payload = config_payload(provider_policy, workers)
    payload["model_id"] = MODEL_DISPLAY_ID + ":" + provider_policy
    run_id = config_id(payload)
    workdir.mkdir(parents=True, exist_ok=True)
    code_sha256_at_start = _current_code_sha256()
    cache_path = workdir / "qwen3_14b_hf_api.sqlite3"
    cache = TranslationCache(cache_path, run_id)
    records = cache.get_many(titles)
    complete = [title for title in titles if cache.complete(records.get(title))]
    pending = [title for title in titles if title not in set(complete)]
    totals = cache.totals()
    if stage == "full":
        smoke_titles = source_titles(frame, "smoke")
        smoke_records = cache.get_many(smoke_titles)
        if any(not cache.complete(smoke_records.get(title)) for title in smoke_titles):
            raise ApiRunError("Run the 20-title smoke stage with this exact config before full.")

    estimate = estimate_max_request_cost(pending[0], workers) if pending else 0.0
    preview = {
        "stage": stage, "model": payload["model_id"], "source_titles": len(titles),
        "cached_complete": len(complete), "pending": len(pending),
        "workers": int(workers), "max_cost_usd": float(max_cost_usd),
        "estimated_inflight_reserve_usd": estimate,
        "config_id": run_id, "provider_calls_started": 0,
        "cumulative_estimated_cost_usd": cache.total_estimated_cost_usd(),
    }
    _write_status(workdir, {"state": "preflight", **preview, **totals}, mirror_dir)
    print("Preflight:", json.dumps(preview, ensure_ascii=False), flush=True)
    if dry_run:
        return {"status": "dry_run", **preview}
    if not allow_paid_inference:
        raise ApiRunError("Refusing provider calls without --allow-paid-inference.")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            from huggingface_hub import get_token
            token = str(get_token() or "").strip()
        except Exception:
            token = ""
    if not token:
        _write_status(workdir, {
            "state": "error",
            "phase": "missing_token",
            "last_error": "HF_TOKEN is not configured; no provider request was sent.",
            "config_id": run_id,
            "provider_calls_started": 0,
        }, mirror_dir)
        raise ApiRunError("HF_TOKEN is not configured; no provider request was sent.")
    if request_fn is None:
        request_fn = lambda title, secret: call_provider(title, secret)

    started = time.monotonic()
    spent = cache.total_estimated_cost_usd()
    completed_this_run = errors_this_run = calls_started = since_mirror = 0
    completed_before = len(complete)
    _write_status(workdir, {"state": "running", "phase": "generating", **preview}, mirror_dir)

    # Bounded batches make the maximum spend overrun at most one small in-flight
    # batch. The provider usage is authoritative after each response.
    cursor = 0
    while cursor < len(pending):
        batch: list[str] = []
        reserved = 0.0
        while cursor + len(batch) < len(pending) and len(batch) < workers:
            title = pending[cursor + len(batch)]
            reserve = estimate_max_request_cost(title, 1)
            if spent + reserved + reserve > max_cost_usd:
                break
            batch.append(title)
            reserved += reserve
        if not batch:
            _write_status(workdir, {
                "state": "budget_cap_reached", "phase": "paused_at_cost_limit",
                "completed_before_session": completed_before,
                "completed_this_session": completed_this_run,
                "errors_this_session": errors_this_run,
                "estimated_cost_usd": spent, "max_cost_usd": max_cost_usd,
                "remaining_titles": len(pending) - cursor,
                "config_id": run_id,
            }, mirror_dir)
            return {"status": "budget_cap_reached", "cost_usd": spent, "remaining": len(pending)-cursor}

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
            futures = {executor.submit(request_fn, title, token): title for title in batch}
            calls_started += len(futures)
            for future in as_completed(futures):
                title = futures[future]
                try:
                    record = future.result()
                except ProviderAuthError as exc:
                    _write_status(workdir, {
                        "state": "error", "phase": "provider_auth", "last_error": _redact(exc),
                        "completed_before_session": completed_before,
                        "completed_this_session": completed_this_run,
                        "estimated_cost_usd": spent, "config_id": run_id,
                    }, mirror_dir)
                    raise
                except Exception as exc:
                    record = {
                        "status": "error", "output_text": "", "truncated": False,
                        "attempts": 0, "estimated_cost_usd": 0.0,
                        "error_text": f"{type(exc).__name__}: {_redact(exc)}",
                        "provider": provider_policy, "model": MODEL_DISPLAY_ID,
                    }
                results[title] = record

        for title in batch:
            record = results[title]
            cache.put(title, record)
            records[title] = record
            spent += float(record.get("estimated_cost_usd", 0.0) or 0.0)
            since_mirror += 1
            if cache.complete(record):
                completed_this_run += 1
            else:
                errors_this_run += 1
        cursor += len(batch)
        if mirror_dir and since_mirror >= CHECKPOINT_EVERY:
            _mirror_cache(cache_path, mirror_dir)
            since_mirror = 0
        elapsed = max(0.001, time.monotonic() - started)
        processed = cursor
        status = {
            "state": "running", "phase": "generating",
            "model_id": payload["model_id"], "provider_policy": provider_policy,
            "processed_this_session": processed, "total_this_session": len(pending),
            "completed_this_session": completed_this_run,
            "errors_this_session": errors_this_run,
            "completed_before_session": completed_before,
            "completed_estimate": completed_before + completed_this_run,
            "throughput_titles_per_hour": processed / elapsed * 3600,
            "elapsed_seconds": elapsed, "estimated_cost_usd": spent,
            "max_cost_usd": max_cost_usd, "calls_started": calls_started,
            "input_tokens": sum(int(x.get("input_tokens", 0) or 0) for x in results.values()),
            "output_tokens": sum(int(x.get("output_tokens", 0) or 0) for x in results.values()),
            "recent_errors": [
                {
                    "title_sha256_prefix": hashlib.sha256(title.encode("utf-8")).hexdigest()[:12],
                    "error": _redact(results[title].get("error_text", "")),
                }
                for title in batch if not cache.complete(results[title])
            ][-3:],
            "config_id": run_id,
        }
        _write_status(workdir, status, mirror_dir)
        print(
            f"[qwen3-api] {processed}/{len(pending)}; complete={completed_this_run}; "
            f"errors={errors_this_run}; rate={status['throughput_titles_per_hour']:.1f}/h; "
            f"cost=${spent:.4f}/${max_cost_usd:.2f}",
            flush=True,
        )
        if _current_code_sha256() != code_sha256_at_start:
            final = {
                "state": "reload_requested",
                "phase": "code_changed_at_batch_boundary",
                "completed_before_session": completed_before,
                "completed_this_session": completed_this_run,
                "errors_this_session": errors_this_run,
                "estimated_cost_usd": spent,
                "max_cost_usd": max_cost_usd,
                "config_id": run_id,
            }
            _write_status(workdir, final, mirror_dir)
            return final
    if mirror_dir and since_mirror:
        _mirror_cache(cache_path, mirror_dir)

    records = cache.get_many(titles)
    incomplete = [title for title in titles if not cache.complete(records.get(title))]
    if incomplete:
        final = {
            "state": "checkpoint_saved_continue", "phase": "incomplete_records",
            "completed": len(titles) - len(incomplete), "total": len(titles),
            "remaining": len(incomplete), "estimated_cost_usd": spent,
            "max_cost_usd": max_cost_usd, "config_id": run_id,
        }
        _write_status(workdir, final, mirror_dir)
        return final
    if stage == "smoke":
        final = {
            "state": "smoke_complete", "phase": "ready_for_full",
            "completed": len(titles), "total": len(titles),
            "estimated_cost_usd": spent, "max_cost_usd": max_cost_usd,
            "config_id": run_id,
        }
        _write_status(workdir, final, mirror_dir)
        return final

    output = _make_output_frame(frame, records, run_id)
    for col in frame.columns:
        if not quality._series_values_equal(frame[col], output[col]):
            raise AssertionError(f"Source column changed during export: {col}")
    output_path = workdir / "bilingual_zh_vi_cado_title_qwen3_14b_api.parquet"
    fd, temp_name = tempfile.mkstemp(prefix=".qwen3-api-", suffix=".parquet.tmp", dir=workdir)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        output.to_parquet(temp_path, index=False)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    verified = pd.read_parquet(output_path)
    if len(verified) != len(frame):
        raise AssertionError("Parquet row count changed after write/read verification.")
    if mirror_dir:
        _mirror_file(output_path, mirror_dir)
    report = {
        "state": "all_titles_complete", "phase": "export_verified",
        "completed": len(titles), "total": len(titles),
        "qa_pass": int((output["title_vi_qwen3_14b_api_qa_status"] == "pass").sum()),
        "qa_fail": int((output["title_vi_qwen3_14b_api_qa_status"] == "fail").sum()),
        "estimated_cost_usd": spent, "max_cost_usd": max_cost_usd,
        "output_parquet": str(output_path), "config_id": run_id,
        "semantic_verified": False,
    }
    _atomic_json(workdir / "qwen3_api_run_report.json", report)
    if mirror_dir:
        _mirror_file(workdir / "qwen3_api_run_report.json", mirror_dir)
    _write_status(workdir, report, mirror_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--mirror-dir", type=Path, default=None,
                        help="Optional folder synchronized by Google Drive for Desktop.")
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--provider-policy", choices=("deepinfra", "fastest", "cheapest"), default="deepinfra")
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--allow-paid-inference", action="store_true",
                        help="Explicitly permit paid HF Inference Provider calls.")
    parser.add_argument("--dry-run", action="store_true", help="Validate source/config; send no requests.")
    args = parser.parse_args(argv)
    try:
        result = run(
            source_path=args.source, workdir=args.workdir, stage=args.stage,
            workers=args.workers, max_cost_usd=args.max_cost_usd,
            provider_policy=args.provider_policy,
            allow_paid_inference=args.allow_paid_inference,
            dry_run=args.dry_run, mirror_dir=args.mirror_dir,
        )
    except Exception as exc:
        print(f"Run stopped: {type(exc).__name__}: {_redact(exc)}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
