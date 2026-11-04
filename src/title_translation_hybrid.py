"""Resumable Chinese-to-Vietnamese title runs: Gemma API plus local Qwen GPU."""

from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[1]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()

import hashlib
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import pandas as pd

import title_translation_pipeline as quality


DATASET_ID = quality.DATASET_ID
DATASET_REVISION = quality.DATASET_REVISION
DATASET_FILE = quality.DATASET_FILE
DATASET_SHA256 = quality.DATASET_SHA256
EXPECTED_ROWS = quality.EXPECTED_ROWS
EXPECTED_UNIQUE_TITLES = quality.EXPECTED_UNIQUE_TITLES
QA_VERSION = quality.QA_VERSION
SAMPLE_VERSION = quality.SAMPLE_VERSION
SEED = quality.SEED

MODELS: dict[str, dict[str, Any]] = {
    "gemma4": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "api_model_id": "gemma-4-26b-a4b-it",
        "backend": "google_gemini_api",
        "provider": "google_ai_studio",
        "revision": None,
        "template_version": "gemma4-zh-vi-google-api-paired-v1",
        "candidate_column": "title_vi_candidate_gemma4",
        "min_request_interval_seconds": 6.0,
    },
    "qwen35": {
        "model_id": "Qwen/Qwen3.5-9B",
        "backend": "local_transformers_cuda",
        "provider": "local_cuda_8bit",
        "revision": None,
        "template_version": "qwen35-zh-vi-local-text-no-thinking-v1",
        "candidate_column": "title_vi_candidate_qwen35",
    },
}

PROMPT_VERSION = "zh-vi-ecommerce-title-translation-v6-robust-code-restoration-and-numeric-qa"
GEMMA_BATCH_PROMPT_VERSION = "gemma4-paired-title-translation-v1-exact-id-lines"
GEMMA_BATCH_SIZE = 2
GEMMA_BATCH_CANARY_TITLES = 40
MAX_GEMMA_BATCH_RETRY_OVERHEAD = 0.15
GEMMA_NORMALIZATION_VERSION = "gemma-residual-recording-screen-and-duration-terms-v6-title-units-and-variants"
GEMMA_QA_REPAIR_PROMPT_VERSION = "gemma4-technical-qa-repair-v2-title-numeric-context"
GEMMA_DETERMINISTIC_REPAIR_VERSION = "gemma4-deterministic-source-anchored-spec-repair-v4"
SMOKE_MIN_TECHNICAL_PASS_RATE = 0.90
MAX_NEW_TOKENS = 256
TRUNCATION_RETRY_TOKENS = 512
MAX_API_RETRIES = 3
FULL_WINDOW_SIZE = 200
FULL_MAX_INCOMPLETE_RATE = 0.01
FULL_MAX_QA_FAILURE_RATE = 0.05
MAX_FULL_INCOMPLETE_RECOVERY_ROUNDS = 2
MAX_GEMMA_QA_REPAIR_ATTEMPTS = 2
MIN_COMPLETION_RATE = 0.99
MIN_REPRESENTATIVE_QA_RATE = 0.95
DEFAULT_FULL_TITLES_PER_SESSION = 1000
REMOTE_OUTPUT_PREFIX = "data/title_translations"


class DailyQuotaExceeded(RuntimeError):
    """Raised when the hosted inference provider exhausts the current quota."""


class GemmaConfigurationError(RuntimeError):
    """Raised for credentials, permissions, or model/configuration errors that need user action."""


class GemmaBatchFormatError(ValueError):
    """A paired response could not be mapped unambiguously back to source titles."""


class GemmaRequestFailed(RuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class Cache:
    """Durable per-model SQLite cache; one completed API response commits at once."""

    def __init__(self, path: Path, model_key: str, revision: str, config_id: str):
        self.path = Path(path)
        self.model_key = model_key
        self.model = MODELS[model_key]
        self.revision = revision
        self.config_id = config_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    source_title TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    output_text TEXT,
                    status TEXT NOT NULL,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    api_requests REAL NOT NULL DEFAULT 0,
                    elapsed_seconds REAL NOT NULL DEFAULT 0,
                    error_text TEXT,
                    model_version TEXT,
                    response_id TEXT,
                    generation_prompt_version TEXT NOT NULL DEFAULT '',
                    updated_at_utc TEXT NOT NULL
                )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(translations)")}
            if "input_tokens" not in columns:
                db.execute("ALTER TABLE translations ADD COLUMN input_tokens INTEGER")
            if "generation_prompt_version" not in columns:
                db.execute("ALTER TABLE translations ADD COLUMN generation_prompt_version TEXT NOT NULL DEFAULT ''")
            if "api_requests" not in columns:
                db.execute("ALTER TABLE translations ADD COLUMN api_requests REAL NOT NULL DEFAULT 0")
                db.execute("UPDATE translations SET api_requests=attempts")
            db.execute(
                "UPDATE translations SET generation_prompt_version=? "
                "WHERE generation_prompt_version IS NULL OR generation_prompt_version=''",
                (PROMPT_VERSION,),
            )
            db.commit()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def key(self, title: str) -> str:
        payload = "\n".join(
            (
                title,
                self.model_key,
                self.model["model_id"],
                self.revision,
                self.config_id,
                PROMPT_VERSION,
                str(MAX_NEW_TOKENS),
                str(TRUNCATION_RETRY_TOKENS),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_many(self, titles: Iterable[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(titles))
        key_to_title = {self.key(title): title for title in unique}
        keys = list(key_to_title)
        rows = []
        with self.connect() as db:
            for start in range(0, len(keys), 400):
                group = keys[start : start + 400]
                marks = ",".join("?" for _ in group)
                rows.extend(
                    db.execute(
                        "SELECT cache_key,output_text,status,truncated,input_tokens,output_tokens,attempts,api_requests,"
                        "elapsed_seconds,error_text,model_version,response_id,generation_prompt_version FROM translations "
                        f"WHERE cache_key IN ({marks})",
                        group,
                    ).fetchall()
                )
        return {
            key_to_title[row[0]]: {
                "output_text": row[1],
                "status": row[2],
                "truncated": bool(row[3]),
                "input_tokens": row[4],
                "output_tokens": row[5],
                "attempts": row[6],
                "api_requests": row[7],
                "elapsed_seconds": row[8],
                "error_text": row[9],
                "model_version": row[10],
                "response_id": row[11],
                "generation_prompt_version": row[12],
            }
            for row in rows
        }

    def put(self, title: str, result: dict[str, Any]) -> None:
        row = (
            self.key(title), title, self.model_key, self.model["model_id"], self.revision,
            self.config_id, result.get("output_text"), result.get("status", "error"),
            int(bool(result.get("truncated"))), result.get("input_tokens"), result.get("output_tokens"),
            int(result.get("attempts", 0)),
            float(result.get("api_requests", result.get("attempts", 0))),
            float(result.get("elapsed_seconds", 0.0)),
            result.get("error_text"), result.get("model_version"), result.get("response_id"),
            result.get("generation_prompt_version") or PROMPT_VERSION,
            utc_now(),
        )
        with self.connect() as db:
            db.execute(
                """INSERT INTO translations (
                    cache_key,source_title,model_key,model_id,model_revision,config_id,
                    output_text,status,truncated,input_tokens,output_tokens,attempts,api_requests,elapsed_seconds,
                    error_text,model_version,response_id,generation_prompt_version,updated_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET output_text=excluded.output_text,
                    status=excluded.status,truncated=excluded.truncated,
                    input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens,attempts=excluded.attempts,
                    api_requests=excluded.api_requests,
                    elapsed_seconds=excluded.elapsed_seconds,error_text=excluded.error_text,
                    model_version=excluded.model_version,response_id=excluded.response_id,
                    generation_prompt_version=excluded.generation_prompt_version,
                    updated_at_utc=excluded.updated_at_utc""",
                row,
            )
            db.commit()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _resolve_input(data_path: str | Path | None, hf_token: str | None) -> Path:
    if data_path:
        candidate = Path(data_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Input parquet does not exist: {candidate}")
        return candidate
    local = Path.cwd() / Path(DATASET_FILE).name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
            token=hf_token,
        )
    )


def _copy_checkpoint_inputs(source: str | Path | None, workdir: Path) -> None:
    if not source:
        return
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint input directory does not exist: {source_path}")
    names = {
        "hybrid_environment_lock.json",
        "hybrid_sample_manifest.json",
        "hybrid_sample_manifest.csv",
        "hybrid_run_state.json",
        "runtime_gemma4.json",
        "runtime_qwen35.json",
        "cache_gemma4.sqlite3",
        "cache_qwen35.sqlite3",
        "gemma4_report.json",
        "gemma4_report.md",
        "qwen35_report.json",
        "qwen35_report.md",
        "bilingual_zh_vi_cado_title_gemma4.parquet",
        "bilingual_zh_vi_cado_title_qwen35.parquet",
        "title_translation_comparison.csv",
        "title_translation_upload_manifest.json",
    }
    for name in names:
        origin = source_path / name
        destination = workdir / name
        if origin.is_file() and not destination.exists():
            shutil.copy2(origin, destination)


def _model_revision_lock(hf_token: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    locked = {}
    for key, model in MODELS.items():
        if model["backend"] == "google_gemini_api":
            revision = model["api_model_id"]
        else:
            info = api.model_info(model["model_id"], revision="main")
            revision = info.sha
        locked[key] = {
            "model_id": model["model_id"],
            "backend": model["backend"],
            "provider": model["provider"],
            "revision": revision,
            "template_version": model["template_version"],
        }
    return locked


def _config_payload(models: dict[str, Any], *, qwen_batch_size: int = 1) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "models": models,
        "prompt_version": PROMPT_VERSION,
        "qa_version": QA_VERSION,
        "gemma_normalization_version": GEMMA_NORMALIZATION_VERSION,
        "sample_version": SAMPLE_VERSION,
        "generation": {
            "max_output_tokens": MAX_NEW_TOKENS,
            "truncation_retry_tokens": TRUNCATION_RETRY_TOKENS,
            "greedy_decoding": True,
            "gemma_api_model_id": MODELS["gemma4"]["api_model_id"],
            "gemma_provider": MODELS["gemma4"]["provider"],
            "gemma_temperature": 0.0,
            "gemma_thinking_level": "minimal",
            "gemma_default_min_request_interval_seconds": MODELS["gemma4"]["min_request_interval_seconds"],
            "gemma_batch_size": GEMMA_BATCH_SIZE,
            "qwen_enable_thinking": False,
            "qwen_provider": MODELS["qwen35"]["provider"],
            "qwen_quantization": "bitsandbytes-8bit",
            "qwen_model_class": "Qwen3_5ForCausalLM",
            "qwen_batch_size": int(qwen_batch_size),
        },
        "quality_gates": {
            "sample_completion_rate": MIN_COMPLETION_RATE,
            "representative_qa_rate": MIN_REPRESENTATIVE_QA_RATE,
            "full_window_size": FULL_WINDOW_SIZE,
            "full_max_incomplete_rate": FULL_MAX_INCOMPLETE_RATE,
            "full_max_qa_failure_rate": FULL_MAX_QA_FAILURE_RATE,
        },
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _runtime_signature(model_key: str) -> dict[str, Any]:
    common = {
        "python": platform.python_version(),
    }
    package_names = ["huggingface_hub", "sacrebleu", "pyarrow"]
    if model_key == "gemma4":
        package_names.append("google-genai")
        common["device"] = f"hosted API: {MODELS[model_key]['provider']}"
    else:
        package_names.extend(["torch", "transformers", "accelerate", "bitsandbytes"])
        try:
            import torch

            common["device"] = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpus": [
                    {
                        "name": torch.cuda.get_device_properties(index).name,
                        "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
        except ImportError:
            common["device"] = "torch-not-installed"
    common["packages"] = {name: _package_version(name) for name in package_names}
    return common


def prepare_pipeline(
    *,
    mode: str,
    model_key: str,
    data_path: str | Path | None,
    workdir: str | Path,
    checkpoint_input: str | Path | None = None,
    hf_token: str | None = None,
    qwen_batch_size: int | None = None,
) -> dict[str, Any]:
    mode = mode.lower().strip()
    model_key = model_key.lower().strip()
    if mode not in {"smoke", "pilot", "validate", "full"}:
        raise ValueError("mode must be smoke, pilot, validate, or full")
    if model_key not in MODELS:
        raise ValueError("model_key must be gemma4 or qwen35")
    if qwen_batch_size is not None and int(qwen_batch_size) < 1:
        raise ValueError("qwen_batch_size must be at least 1")
    output_dir = Path(workdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_checkpoint_inputs(checkpoint_input, output_dir)

    input_path = _resolve_input(data_path, hf_token)
    actual_sha = quality.sha256_file(input_path)
    if actual_sha != DATASET_SHA256:
        raise ValueError(f"Input parquet SHA-256 mismatch: expected {DATASET_SHA256}, got {actual_sha}.")
    frame = pd.read_parquet(input_path)
    quality.validate_input_frame(frame)

    lock_path = output_dir / "hybrid_environment_lock.json"
    if not lock_path.exists():
        if mode != "smoke":
            raise RuntimeError("Run smoke first to pin the model revisions and write the environment lock.")
        models = _model_revision_lock(hf_token)
        locked_qwen_batch_size = int(qwen_batch_size or 1)
        config = _config_payload(models, qwen_batch_size=locked_qwen_batch_size)
        lock = {
            "created_at_utc": utc_now(),
            "dataset_sha256": DATASET_SHA256,
            "models": models,
            "config": config,
            "config_id": fingerprint(config),
            "backend_revision_note": (
                "Gemma uses the Google AI Studio API model identifier; Qwen uses the immutable Hugging Face "
                "Hub commit SHA and is loaded locally on the selected CUDA device."
            ),
        }
        atomic_json(lock_path, lock)
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locked_qwen_batch_size = int(
            lock.get("config", {}).get("generation", {}).get("qwen_batch_size", 1)
        )
        if qwen_batch_size is not None and int(qwen_batch_size) != locked_qwen_batch_size:
            raise RuntimeError(
                f"Run is locked to Qwen batch size {locked_qwen_batch_size}; "
                f"requested {qwen_batch_size}. Resume with the locked value or start a new run."
            )
        if lock.get("dataset_sha256") != DATASET_SHA256:
            raise RuntimeError("The saved environment lock belongs to another source dataset.")
        if lock.get("config_id") != fingerprint(
            _config_payload(lock.get("models", {}), qwen_batch_size=locked_qwen_batch_size)
        ):
            raise RuntimeError("Environment lock fingerprint is invalid; do not mix runs.")

    manifest_path = output_dir / "hybrid_sample_manifest.json"
    if manifest_path.exists():
        samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        if samples.get("dataset_sha256") != DATASET_SHA256 or samples.get("sample_version") != SAMPLE_VERSION:
            raise RuntimeError("Saved sample manifest does not match the pinned dataset.")
    else:
        if mode != "smoke":
            raise RuntimeError("Run smoke first to persist the deterministic sample manifest.")
        samples = quality.build_sample_manifest(frame)
        atomic_json(manifest_path, samples)
        records = []
        for title in samples["smoke"]:
            records.append({"stage": "smoke", "sample_type": "pilot_subset", "title_zh": title})
        for stage in ("pilot", "validate"):
            for sample_type in ("representative", "challenge"):
                records.extend(
                    {"stage": stage, "sample_type": sample_type, "title_zh": title}
                    for title in samples[stage][sample_type]
                )
        pd.DataFrame(records).drop_duplicates(["stage", "title_zh"]).to_csv(
            output_dir / "hybrid_sample_manifest.csv", index=False, encoding="utf-8-sig"
        )

    state_path = output_dir / "hybrid_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_id") != lock["config_id"]:
            raise RuntimeError("Saved progress has another config ID; refusing to mix translations.")
    else:
        if mode != "smoke":
            raise RuntimeError("Run smoke first to initialize resumable state.")
        state = {
            "config_id": lock["config_id"],
            "dataset_sha256": DATASET_SHA256,
            "models": {key: {"stages": {}} for key in MODELS},
            "created_at_utc": utc_now(),
        }
        atomic_json(state_path, state)

    return {
        "mode": mode,
        "model_key": model_key,
        "data_path": str(input_path),
        "workdir": output_dir,
        "frame": frame,
        "lock": lock,
        "samples": samples,
        "state": state,
        "qwen_batch_size": locked_qwen_batch_size,
    }


def _stage_titles(mode: str, samples: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    if mode == "smoke":
        return list(samples["smoke"])
    if mode in {"pilot", "validate"}:
        return list(samples[mode]["titles"])
    return list(dict.fromkeys(frame["title_zh"].astype(str).tolist()))


def _runtime_lock_path(context: dict[str, Any], model_key: str) -> Path:
    return context["workdir"] / f"runtime_{model_key}.json"


def _check_or_write_runtime_lock(
    context: dict[str, Any], model_key: str, *, allow_qwen_hardware_change: bool = False
) -> dict[str, Any]:
    current = _runtime_signature(model_key)
    path = _runtime_lock_path(context, model_key)
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        if locked.get("signature") != current:
            previous = locked.get("signature", {})
            previous_device = previous.get("device", {})
            current_device = current.get("device", {})
            previous_software = {key: value for key, value in previous.items() if key != "device"}
            current_software = {key: value for key, value in current.items() if key != "device"}
            previous_cuda = {
                key: value for key, value in previous_device.items() if key != "gpus"
            }
            current_cuda = {
                key: value for key, value in current_device.items() if key != "gpus"
            }
            if (
                allow_qwen_hardware_change
                and model_key == "qwen35"
                and previous_software == current_software
                and previous_cuda == current_cuda
            ):
                history = locked.setdefault("gpu_hardware_history", [])
                for device in (previous_device.get("gpus", []), current_device.get("gpus", [])):
                    if device and device not in history:
                        history.append(device)
                locked["signature"] = current
                locked["last_hardware_change_at_utc"] = utc_now()
                atomic_json(path, locked)
                return locked
            raise RuntimeError(
                f"Runtime changed since {model_key} smoke; restore the matching Colab/Kaggle environment "
                "or start a new run and repeat validation."
            )
        return locked
    if context["mode"] != "smoke":
        raise RuntimeError(f"Run {model_key} smoke first to lock its runtime.")
    lock = {"model_key": model_key, "signature": current, "created_at_utc": utc_now()}
    atomic_json(path, lock)
    return lock


def _check_prerequisites(
    context: dict[str, Any],
    model_key: str,
    mode: str,
    *,
    allow_unvalidated_full: bool = False,
    defer_full_quality_review: bool = False,
) -> None:
    state = context["state"]
    if state.get("config_id") != context["lock"]["config_id"]:
        raise RuntimeError("Run state and model configuration do not match.")
    model_state = state["models"].get(model_key, {"stages": {}})
    stages = model_state.get("stages", {})
    if mode == "pilot" and not stages.get("smoke", {}).get("passed"):
        raise RuntimeError(f"{model_key} pilot is blocked until its smoke passes.")
    if mode == "validate" and not stages.get("pilot", {}).get("passed"):
        raise RuntimeError(f"{model_key} validation is blocked until its pilot passes.")
    if mode == "full":
        validation = stages.get("validate", {})
        if (
            not allow_unvalidated_full
            and (not validation.get("passed") or validation.get("config_id") != context["lock"]["config_id"])
        ):
            raise RuntimeError(f"Full is blocked until {model_key} validation passes with this config.")
        prior_full = stages.get("full", {})
        if (
            not defer_full_quality_review
            and prior_full.get("status") == "stopped_quality_gate"
            and not _full_gate_stop_is_incomplete_only(prior_full)
            and not _full_gate_stop_is_resolved(prior_full, context, model_key)
        ):
            raise RuntimeError(f"{model_key} full previously stopped on a quality gate; review and revalidate.")


def _full_gate_stop_is_incomplete_only(result: dict[str, Any]) -> bool:
    """Allow a full resume only when the stop was caused by retryable missing outputs."""
    failed = result.get("failed_window") or {}
    try:
        incomplete_rate = float(failed.get("incomplete_rate", 0.0))
        qa_failure_rate = float(failed.get("technical_qa_failure_rate", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        result.get("status") == "stopped_quality_gate"
        and incomplete_rate > FULL_MAX_INCOMPLETE_RATE + 1e-12
        and qa_failure_rate <= FULL_MAX_QA_FAILURE_RATE + 1e-12
    )


def _full_gate_stop_is_resolved(
    result: dict[str, Any], context: dict[str, Any], model_key: str
) -> bool:
    """Allow a quality-gated full run to resume only after its exact window passes."""
    if result.get("status") != "stopped_quality_gate":
        return False
    failed = result.get("failed_window") or {}
    try:
        start_index = int(failed["start"])
        end_index = int(failed["end"])
    except (KeyError, TypeError, ValueError):
        return False
    all_titles = list(dict.fromkeys(context["frame"]["title_zh"].astype(str).tolist()))
    if start_index < 0 or end_index <= start_index or end_index > len(all_titles):
        return False
    reports = _full_window_gates(
        all_titles[start_index:end_index], all_titles, _cache(context, model_key)
    )
    return bool(
        len(reports) == 1
        and reports[0].get("start") == start_index
        and reports[0].get("passed")
    )


def _cache(context: dict[str, Any], model_key: str) -> Cache:
    info = context["lock"]["models"][model_key]
    cache_paths = context.get("cache_paths", {})
    cache_path = cache_paths.get(model_key, context["workdir"] / f"cache_{model_key}.sqlite3")
    return Cache(
        cache_path,
        model_key,
        info["revision"],
        context["lock"]["config_id"],
    )


def _complete(record: dict[str, Any] | None) -> bool:
    if not record or record.get("status") != "ok" or record.get("truncated"):
        return False
    output_text = str(record.get("output_text") or "").strip()
    return bool(output_text and not quality._repeated_phrase(output_text))


def _gemma_system_instruction() -> str:
    return (
        "Translate one Chinese e-commerce product title into natural, concise Vietnamese. "
        "Return only the Vietnamese title, with no explanation or quotation marks, and never leave Chinese Han characters in the output. "
        "Any ZXQCODETOKEN marker is a protected product-code placeholder: copy it exactly once, unchanged, in the matching position. "
        "Keep existing Latin brand names and model identifiers exactly character-for-character; do not insert, remove, or alter letters, digits, or separators in a code. "
        "For a Chinese brand or proper name without a supplied Latin spelling, use its standard pinyin/Latin rendering rather than copying Han characters. "
        "Preserve every source size, quantity, number, currency amount, and meaningful unit; translate all other details, and do not omit repetitive specifications or invent extra numbers. "
        "Do not turn 纯棉 (pure cotton) into 100% cotton; translate it as cotton nguyên chất. "
        "一件代发 is the commerce term dropshipping, not a one-piece quantity. 一折 means the price is 10% of the original, preferably render as giá còn 10% giá gốc. "
        "One 斤 equals 500 g: render its true equivalent in g/kg (for example, 一斤 = 500 g, 半斤 = 250 g), never translate one 斤 as one Vietnamese cân. "
        "For Arabic digits followed by 万, preserve the combined value (2万 = 20,000), not the two parts separately. "
        "Preserve garment and count details: 单肩 means one shoulder; 双肩包 means a backpack with two shoulder straps; "
        "一片式 clothing means a one-piece/continuous garment; 吊带 is the Vietnamese garment term váy hai dây. "
        "For common compounds, preserve their meaning explicitly: "
        "(ba lô hai quai), 双层 means two layers, 双面 means two sides, and 双录 may be rendered as recording "
        "both front and rear. 万向轮 means a swivel/universal wheel; do not add 360 degrees unless the source says so."
    )


def _protect_model_codes(title: str) -> tuple[str, list[tuple[str, str]]]:
    matches = [
        match
        for match in quality.CODE_RE.finditer(title)
        if quality._is_contextual_model_code(title, match)
    ]
    protected = title
    replacements: list[tuple[str, str]] = []
    for index, match in reversed(list(enumerate(matches))):
        marker = f"ZXQCODETOKEN{chr(65 + index % 26)}ZXQ"
        replacements.append((marker, match.group(0)))
        protected = protected[: match.start()] + marker + protected[match.end() :]
    return protected, list(reversed(replacements))


def _normalize_gemma_measurements(text: str) -> str:
    # Normalize only a small set of unambiguous recording/screen feature terms
    # that Gemma occasionally leaves partially untranslated.
    residual_terms = (
        (r"单录", "ghi hình một kênh"),
        (r"雙錄|双录", "ghi hình hai kênh"),
        (r"三錄|三录", "ghi hình ba kênh"),
        (r"四錄|四录", "ghi hình bốn kênh"),
        (r"(?i:đơn\s*录)", "ghi hình một kênh"),
        (r"(?i:song\s*录)", "ghi hình hai kênh"),
        (r"投屏", "chiếu màn hình"),
    )
    for pattern, replacement in residual_terms:
        text = re.sub(pattern, f" {replacement} ", text)
    # Translate a currency unit the model occasionally copies verbatim.
    text = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*(?:人民币|元)",
        r"\1 yuan",
        text,
    )
    # Convert any residual Chinese 斤 to its true mass equivalent (500 g per 斤).
    for match in reversed(list(quality.JIN_RE.finditer(text))):
        raw = match.group("amount") or match.group("half_only")
        if not raw:
            continue
        try:
            grams = (quality._number_value(raw) + (0.5 if match.group("extra_half") else 0.0)) * 500.0
        except ValueError:
            continue
        if grams >= 1000:
            amount = grams / 1000
            formatted = f"{amount:.3f}".rstrip("0").rstrip(".")
            replacement = f"{formatted} kg"
        else:
            formatted = f"{grams:.3f}".rstrip("0").rstrip(".")
            replacement = f"{formatted} g"
        text = text[: match.start()] + replacement + text[match.end() :]
    text = re.sub(r"\s*/\s*", "/", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _trim_excess_trailing_numbers(title: str, text: str) -> tuple[str, list[str]]:
    """Remove only trailing repeated Arabic numbers beyond the source multiplicity."""
    source_counts: Counter[float] = Counter()
    for raw, _, _ in quality._number_tokens(title):
        try:
            source_counts[round(quality._number_value(raw), 6)] += 1
        except ValueError:
            continue

    result = text
    repairs: list[str] = []
    while True:
        tokens = quality._number_tokens(result)
        if not tokens:
            break
        raw, start, end = tokens[-1]
        if not re.fullmatch(r"\d+(?:[.,]\d+)*", raw):
            break
        if result[end:].strip(" \t.,;:!?，。；：！？"):
            break
        try:
            value = round(quality._number_value(raw), 6)
        except ValueError:
            break
        candidate_count = sum(
            1
            for candidate_raw, _, _ in tokens
            if _is_same_numeric_value(candidate_raw, value)
        )
        if source_counts[value] == 0 or candidate_count <= source_counts[value]:
            break
        prefix = result[:start].rstrip(" \t,;，；")
        suffix = result[end:]
        result = re.sub(r"\s{2,}", " ", prefix + suffix).strip()
        repairs.append(f"trim-trailing-duplicate-number:{raw}")
    return result, repairs


def _is_same_numeric_value(raw: str, target: float) -> bool:
    try:
        return math.isclose(quality._number_value(raw), target, rel_tol=1e-9, abs_tol=1e-6)
    except ValueError:
        return False


def _repair_gemma_specs_from_source(title: str, text: str) -> tuple[str, list[str]]:
    """Apply narrowly source-anchored repairs for clear product-title specifications."""
    result = text
    repairs: list[str] = []

    # Preserve these explicit source brand names in Latin script rather than
    # accepting them as untranslated Han text. 咖 is rendered here by its
    # standard pinyin syllable; 华熙生物 is the commonly used Huaxi Bio form.
    brand_repairs = (
        ("华熙生物", "Huaxi Bio", "brand:huaxi-bio"),
        ("c咖", "C-Ka", "brand:c-ka-pinyin"),
    )
    for source_term, latin_form, rule in brand_repairs:
        if source_term.casefold() in title.casefold():
            result, count = re.subn(re.escape(source_term), latin_form, result, flags=re.IGNORECASE)
            if count:
                repairs.append(rule)

    # Translate common resolution wording only when the model left the exact
    # Han phrase in its otherwise Vietnamese output.
    for source_term in ("高清", "高淸", "高清晰"):
        if source_term in title and source_term in result:
            result = result.replace(source_term, "độ nét cao")
            repairs.append(f"residual:{source_term}")

    # The Chinese left/right marker in this exact source construction is
    # directional language, not a model code.
    if re.search(r"前\s*4K\s*后\s*2K", title, re.IGNORECASE):
        rewritten, count = re.subn(r"4K\s*后\s*2K", "4K trước, 2K sau", result, count=1, flags=re.IGNORECASE)
        if count:
            result = rewritten
            repairs.append("front-rear-resolution-order")

    # Preserve exact screen sizes; an embedded value such as ADAS12 or 4K10
    # is not sufficient evidence that the inch unit was translated.
    for match in re.finditer(r"(?<!\d)(?P<size>\d+(?:[.,]\d+)?)\s*寸", title):
        size = match.group("size")
        escaped = re.escape(size)
        if re.search(rf"(?<!\d){escaped}\s*(?:inch(?:es)?|英寸|in\b|[\"″])", result, re.IGNORECASE):
            continue
        result = f"{result.rstrip(' ,;')} màn hình {size} inch"
        repairs.append(f"screen-size:{size}-inch")

    # A Chinese single-lens marker maps to one lens, even if Vietnamese
    # grammar omitted the numeral in the model's draft.
    if "单镜头" in title and re.search(r"ống kính", result, re.IGNORECASE):
        if not re.search(r"\b(?:một|1|single)\s+ống kính\b", result, re.IGNORECASE):
            result = re.sub(r"\bống kính\b", "một ống kính", result, count=1, flags=re.IGNORECASE)
            repairs.append("single-lens-count")

    # Recording-channel counts are explicit product specs. Add a short
    # clarifier only if the draft did not already express the count.
    channel_terms = (
        ("单录", "một", r"\b(?:một|1)\s*kênh\b", r"\b(?:một|1)\s*(?:video|camera)\b"),
        ("双录", "hai", r"\b(?:hai|2)\s*kênh\b", r"\b(?:hai|2)\s*(?:video|camera|chiều)\b"),
        ("雙錄", "hai", r"\b(?:hai|2)\s*kênh\b", r"\b(?:hai|2)\s*(?:video|camera|chiều)\b"),
        ("三录", "ba", r"\b(?:ba|3)\s*kênh\b", r"\b(?:ba|3)\s*(?:video|camera)\b"),
        ("三錄", "ba", r"\b(?:ba|3)\s*kênh\b", r"\b(?:ba|3)\s*(?:video|camera)\b"),
        ("四录", "bốn", r"\b(?:bốn|4)\s*kênh\b", r"\b(?:bốn|4)\s*(?:video|camera)\b"),
        ("四錄", "bốn", r"\b(?:bốn|4)\s*kênh\b", r"\b(?:bốn|4)\s*(?:video|camera)\b"),
    )
    for source_term, count_word, candidate_pattern, alternate_pattern in channel_terms:
        if source_term not in title:
            continue
        alternate_match = re.search(alternate_pattern, result, re.IGNORECASE)
        if alternate_match:
            # Earlier normalization versions appended this exact suffix even
            # when the draft already contained an equivalent numbered video.
            # Remove only that redundant generated clause, not source text.
            result, removed = re.subn(
                rf"\s*;\s*ghi hình {re.escape(count_word)} kênh\s*$",
                "",
                result,
                count=1,
                flags=re.IGNORECASE,
            )
            if removed:
                repairs.append(f"duplicate-recording-channel:{source_term}")
            continue
        if not re.search(candidate_pattern, result, re.IGNORECASE):
            result = f"{result.rstrip(' ,;')}; ghi hình {count_word} kênh"
            repairs.append(f"recording-channel:{source_term}")

    # Preserve explicit operating durations such as 24小时监控 when Gemma
    # weakens them to an unnumbered phrase like “giám sát trong nhiều giờ”.
    if "监控" in title:
        for match in re.finditer(r"(?<!\d)(?P<hours>\d+)\s*小时", title):
            hours = match.group("hours")
            if re.search(
                rf"(?<!\d){re.escape(hours)}\s*(?:giờ|h\b|hours?\b)",
                result,
                re.IGNORECASE,
            ):
                continue
            result, replaced = re.subn(
                r"\b(?:nhiều|vài)\s+giờ\b",
                f"{hours} giờ",
                result,
                count=1,
                flags=re.IGNORECASE,
            )
            if not replaced:
                result = f"{result.rstrip(' ,;')} giám sát trong {hours} giờ"
            repairs.append(f"monitoring-duration:{hours}-hours")

    # Titles that explicitly say 2分割 advertise a two-way split view.
    for match in re.finditer(r"(?<!\d)(?P<count>\d+)\s*分割", title):
        count = match.group("count")
        if not re.search(rf"\b{re.escape(count)}\s*(?:phần|khung|màn hình)\b", result, re.IGNORECASE):
            result = f"{result.rstrip(' ,;')}; chia màn hình thành {count} phần"
            repairs.append(f"split-screen:{count}")

    # Preserve explicitly stated use duration (e.g. 5分钟面膜). If the draft
    # already contains one duration phrase but attached the wrong source number,
    # correct that number only; other quantities such as sheet counts are untouched.
    source_minutes = list(re.finditer(r"(?<!\d)(?P<minutes>\d+(?:[.,]\d+)?)\s*分钟", title))
    draft_minutes = list(
        re.finditer(r"(?<!\d)(?P<minutes>\d+(?:[.,]\d+)?)\s*(?:phút|minutes?|mins?)\b", result, re.IGNORECASE)
    )
    if len(source_minutes) == 1 and len(draft_minutes) == 1:
        expected = source_minutes[0].group("minutes")
        actual = draft_minutes[0].group("minutes")
        if not math.isclose(quality._number_value(expected), quality._number_value(actual)):
            match = draft_minutes[0]
            result = result[: match.start("minutes")] + expected + result[match.end("minutes"):]
            repairs.append(f"source-duration:{expected}-minutes")

    # Translate explicit letter variants without guessing any product details.
    for match in re.finditer(r"(?<![A-Za-z0-9])(?P<variant>[A-Z])款", title):
        variant = match.group("variant")
        result, count = re.subn(
            rf"(?<![A-Za-z0-9]){re.escape(variant)}款",
            f"loại {variant} ",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
        if count:
            repairs.append(f"letter-variant:{variant}")

    result, duplicate_repairs = _trim_excess_trailing_numbers(title, result)
    repairs.extend(duplicate_repairs)

    return re.sub(r"\s{2,}", " ", result).strip(), repairs


def _restore_gemma_model_codes(text: str, protected_codes: list[tuple[str, str]]) -> str:
    """Restore exact codes, repairing shortened placeholders without leaking them."""
    restored = text
    unresolved: list[str] = []
    damaged_marker = re.compile(r"(?:ZXQ)?CODETOKEN(?:[A-Z])?(?:ZXQ)?", re.IGNORECASE)
    for marker, source_code in protected_codes:
        full_marker = re.compile(re.escape(marker), re.IGNORECASE)
        if full_marker.search(restored):
            restored = full_marker.sub(lambda _: source_code, restored)
        elif damaged_marker.search(restored):
            restored = damaged_marker.sub(lambda _: source_code, restored, count=1)
        elif re.search(re.escape(source_code), restored, re.IGNORECASE):
            continue
        else:
            unresolved.append(source_code)
    # Remove any unpaired or shortened sentinel debris. When there was no
    # protected source code this also handles a hallucinated placeholder word.
    restored = re.sub(
        r"(?:ZXQ)?CODETOKEN(?:[A-Z])?(?:ZXQ)?",
        "",
        restored,
        flags=re.IGNORECASE,
    )
    # ZXQ is the sentinel's closing fragment. Do not leave it attached to a
    # code when the model copied only part of the marker.
    if not any("ZXQ" in code.upper() for _, code in protected_codes):
        restored = re.sub(r"ZXQ", "", restored, flags=re.IGNORECASE)
    if unresolved:
        restored = f"{restored} {' '.join(unresolved)}".strip()
    return re.sub(r"\s{2,}", " ", restored).strip(" \t,，;；:-")


def _gemma_is_daily_quota_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    daily_markers = (
        "perday", "per_day", "per day", "requestsperday", "requests_per_day",
        "daily quota", "quota per day", "rpd", "free_tier_requests_per_day",
    )
    return any(marker in message for marker in daily_markers)


def _gemma_is_configuration_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(exc, "code", None) or getattr(response, "status_code", None)
    try:
        if int(status) in {400, 401, 402, 403, 404, 422}:
            return True
    except (TypeError, ValueError):
        pass
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "api key not valid", "api_key_invalid", "invalid api key", "unauthenticated",
            "permission denied", "model is not found", "model not found", "not supported for this model",
        )
    )


def _safe_error(exc: BaseException, secret: str | None = None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)(key|token|authorization)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", message)
    return message[:1200]


class RequestPacer:
    def __init__(self, interval_seconds: float):
        self.interval = max(0.0, float(interval_seconds))
        self.last_start = 0.0
        self.lock = Lock()

    def wait_turn(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.interval - (now - self.last_start)
            if delay > 0:
                time.sleep(delay)
            self.last_start = time.monotonic()


def _gemma_response(
    client: Any,
    title: str,
    max_tokens: int,
    *,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    from google.genai import types

    protected_title, protected_codes = _protect_model_codes(title)
    response = client.models.generate_content(
        model=MODELS["gemma4"]["api_model_id"],
        contents=protected_title,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction or _gemma_system_instruction(),
            temperature=0.0,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    text = str(getattr(response, "text", None) or "").strip()
    text = _restore_gemma_model_codes(text, protected_codes)
    text = _normalize_gemma_measurements(text)
    text, _ = _repair_gemma_specs_from_source(title, text)
    candidates = getattr(response, "candidates", None) or []
    finish_reason = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    return {
        "output_text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "truncated": finish_reason.lower() in {"length", "max_tokens"},
        "model_version": getattr(response, "model_version", None) or MODELS["gemma4"]["api_model_id"],
        "response_id": getattr(response, "response_id", None),
        "finish_reason": finish_reason,
    }


def _allocate_token_usage(total: int | None, weights: list[int]) -> list[int | None]:
    if total is None:
        return [None] * len(weights)
    if not weights:
        return []
    denominator = max(1, sum(weights))
    allocated: list[int] = []
    remaining = int(total)
    for weight in weights[:-1]:
        share = min(remaining, int(round(int(total) * max(1, weight) / denominator)))
        allocated.append(share)
        remaining -= share
    allocated.append(remaining)
    return allocated


def _gemma_batch_response(client: Any, titles: list[str], max_tokens: int) -> dict[str, Any]:
    """Translate a small, explicitly indexed group and map each output back safely."""
    from google.genai import types

    if len(titles) < 2:
        raise ValueError("Paired Gemma generation requires at least two titles.")
    protected: list[tuple[str, list[tuple[str, str]]]] = [
        _protect_model_codes(title) for title in titles
    ]
    system_instruction = (
        _gemma_system_instruction()
        + " Translate every indexed source independently. Return exactly one line for each item, "
        "in the exact format [[ID]] followed by a tab and the Vietnamese title. Preserve the IDs "
        "and order. Do not add a preamble, explanation, bullets, or combine titles."
    )
    contents = "Translate these separate product titles:\n" + "\n".join(
        f"[[{index}]]\t{source.replace(chr(10), ' ').replace(chr(13), ' ')}"
        for index, (source, _codes) in enumerate(protected, start=1)
    )
    response = client.models.generate_content(
        model=MODELS["gemma4"]["api_model_id"],
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    raw_text = str(getattr(response, "text", None) or "").strip()
    parsed: dict[int, str] = {}
    parse_error = None
    for line in raw_text.splitlines():
        match = re.fullmatch(r"\s*\[\[(\d+)\]\]\s+(.+?)\s*", line)
        if not match:
            parse_error = "response contains a line outside the required [[id]] format"
            break
        item_id = int(match.group(1))
        if item_id in parsed or not 1 <= item_id <= len(titles):
            parse_error = "response contains a duplicate or unexpected item id"
            break
        parsed[item_id] = match.group(2).strip()
    if parse_error is None and set(parsed) != set(range(1, len(titles) + 1)):
        parse_error = "response omitted one or more indexed titles"

    candidates = getattr(response, "candidates", None) or []
    finish_reason = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
    truncated = finish_reason.lower() in {"length", "max_tokens"}
    usage = getattr(response, "usage_metadata", None)
    input_total = getattr(usage, "prompt_token_count", None) if usage else None
    output_total = getattr(usage, "candidates_token_count", None) if usage else None
    items: list[dict[str, Any]] = []
    if parse_error is None:
        outputs = []
        for index, (title, (_protected_title, codes)) in enumerate(zip(titles, protected), start=1):
            text = _restore_gemma_model_codes(parsed[index], codes)
            text = _normalize_gemma_measurements(text)
            text, _ = _repair_gemma_specs_from_source(title, text)
            outputs.append(text)
        input_shares = _allocate_token_usage(input_total, [len(x[0]) for x in protected])
        output_shares = _allocate_token_usage(output_total, [len(x) for x in outputs])
        for index, output_text in enumerate(outputs):
            items.append(
                {
                    "output_text": output_text,
                    "input_tokens": input_shares[index],
                    "output_tokens": output_shares[index],
                    "truncated": truncated,
                    "model_version": getattr(response, "model_version", None)
                    or MODELS["gemma4"]["api_model_id"],
                    "response_id": getattr(response, "response_id", None),
                    "finish_reason": finish_reason,
                }
            )
    return {
        "items": items if parse_error is None else None,
        "parse_error": parse_error,
        "raw_output": raw_text,
        "input_tokens": input_total,
        "output_tokens": output_total,
        "truncated": truncated,
        "model_version": getattr(response, "model_version", None)
        or MODELS["gemma4"]["api_model_id"],
        "response_id": getattr(response, "response_id", None),
        "finish_reason": finish_reason,
    }


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) if response else {}
    if headers:
        try:
            return max(0.0, float(headers.get("Retry-After") or headers.get("retry-after")))
        except (TypeError, ValueError):
            pass
    match = re.search(r"retry\s+in\s+([0-9.]+)\s*s", str(exc), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _close_inference_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _new_gemma_client(api_key: str) -> Any:
    """Create a client with SDK retries disabled; the paced outer loop owns retries."""
    from google import genai

    types = genai.types
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def _call_gemma_with_retries(
    client: Any,
    title: str,
    max_tokens: int,
    pacer: RequestPacer,
    hf_token: str,
    system_instruction: str | None = None,
    batch_titles: list[str] | None = None,
) -> dict[str, Any]:
    for attempt in range(1, MAX_API_RETRIES + 2):
        pacer.wait_turn()
        try:
            if batch_titles is not None:
                result = _gemma_batch_response(client, batch_titles, max_tokens)
                result["api_attempts"] = attempt
                return result
            result = _gemma_response(client, title, max_tokens, system_instruction=system_instruction)
            result["api_attempts"] = attempt
            return result
        except Exception as exc:
            if isinstance(exc, GemmaBatchFormatError):
                raise
            status = _http_status(exc)
            if _gemma_is_daily_quota_error(exc) or (status == 429 and attempt == MAX_API_RETRIES + 1):
                raise DailyQuotaExceeded(_safe_error(exc, hf_token)) from exc
            if _gemma_is_configuration_error(exc):
                raise GemmaConfigurationError(_safe_error(exc, hf_token)) from exc
            safe_error = _safe_error(exc, hf_token)
            if attempt == MAX_API_RETRIES + 1:
                print(
                    f"[gemma4] request failed after {attempt} attempts ({'HTTP ' + str(status) if status else 'network/SDK error'}): "
                    f"{safe_error[:300]}",
                    flush=True,
                )
                raise GemmaRequestFailed(safe_error, attempts=attempt) from exc
            wait_seconds = _retry_after_seconds(exc)
            if wait_seconds is None:
                wait_seconds = min(60.0, 3.0 * (2 ** (attempt - 1)))
            print(
                f"[gemma4] transient request error on attempt {attempt} "
                f"({'HTTP ' + str(status) if status else 'network/SDK'}); "
                f"retrying after {max(wait_seconds, 1.0):.1f}s: {safe_error[:300]}",
                flush=True,
            )
            time.sleep(max(wait_seconds, 1.0))
    raise AssertionError("unreachable retry loop")


def _run_gemma(
    titles: list[str],
    context: dict[str, Any],
    google_api_key: str,
    min_interval_seconds: float | None,
    *,
    batch_size: int = 1,
) -> dict[str, Any]:
    model_key = "gemma4"
    cache = _cache(context, model_key)
    cached = cache.get_many(titles)
    pending = [title for title in titles if not _complete(cached.get(title))]
    if not pending:
        return {"model_key": model_key, "requested": len(titles), "completed": len(titles), "cached": len(titles)}

    batch_size = max(1, min(GEMMA_BATCH_SIZE, int(batch_size)))
    client = _new_gemma_client(google_api_key)
    interval = MODELS[model_key]["min_request_interval_seconds"] if min_interval_seconds is None else min_interval_seconds
    pacer = RequestPacer(interval)
    start = time.perf_counter()
    completed = errors = input_tokens = output_tokens = api_attempts = batch_requests = batch_fallbacks = 0
    processed = 0

    def add_usage(record: dict[str, Any]) -> None:
        nonlocal input_tokens, output_tokens
        input_tokens += int(record.get("input_tokens") or 0)
        output_tokens += int(record.get("output_tokens") or 0)

    def run_single(
        title: str, *, prior_input: int = 0, prior_output: int = 0, prior_calls: float = 0
    ) -> None:
        nonlocal completed, errors, api_attempts
        item_start = time.perf_counter()
        attempts = 0
        try:
            output = _call_gemma_with_retries(client, title, MAX_NEW_TOKENS, pacer, google_api_key)
            attempts = int(output.pop("api_attempts", 1))
            api_attempts += attempts
            if output["truncated"]:
                initial_input = int(output.get("input_tokens") or 0)
                initial_output = int(output.get("output_tokens") or 0)
                retry = _call_gemma_with_retries(
                    client, title, TRUNCATION_RETRY_TOKENS, pacer, google_api_key
                )
                retry_attempts = int(retry.pop("api_attempts", 1))
                attempts += retry_attempts
                api_attempts += retry_attempts
                retry["input_tokens"] = initial_input + int(retry.get("input_tokens") or 0)
                retry["output_tokens"] = initial_output + int(retry.get("output_tokens") or 0)
                output = retry
            record = {
                **output,
                "status": "ok" if output["output_text"] else "error",
                "attempts": attempts + (1 if prior_calls else 0),
                "api_requests": attempts + prior_calls,
                "input_tokens": int(output.get("input_tokens") or 0) + prior_input,
                "output_tokens": int(output.get("output_tokens") or 0) + prior_output,
                "elapsed_seconds": time.perf_counter() - item_start,
                "generation_prompt_version": PROMPT_VERSION,
            }
            cache.put(title, record)
            add_usage(record)
            if _complete(record):
                completed += 1
            else:
                errors += 1
        except DailyQuotaExceeded:
            raise
        except GemmaConfigurationError:
            raise
        except Exception as exc:
            failed_attempts = int(getattr(exc, "attempts", 0))
            api_attempts += failed_attempts
            errors += 1
            cache.put(
                title,
                {
                    "status": "error",
                    "truncated": False,
                    "attempts": attempts + (1 if prior_calls else 0) + failed_attempts,
                    "api_requests": attempts + prior_calls + failed_attempts,
                    "input_tokens": prior_input,
                    "output_tokens": prior_output,
                    "elapsed_seconds": time.perf_counter() - item_start,
                    "error_text": _safe_error(exc, google_api_key),
                    "generation_prompt_version": PROMPT_VERSION,
                },
            )

    try:
        cursor = 0
        while cursor < len(pending):
            group = pending[cursor : cursor + batch_size]
            group_started = time.perf_counter()
            fallback_to_single = len(group) == 1
            prior_inputs = [0] * len(group)
            prior_outputs = [0] * len(group)
            prior_calls = 0
            batch_items: list[dict[str, Any]] | None = None
            first_input = first_output = 0

            if len(group) > 1:
                batch_requests += 1
                try:
                    first = _call_gemma_with_retries(
                        client,
                        "\n".join(group),
                        MAX_NEW_TOKENS * len(group),
                        pacer,
                        google_api_key,
                        batch_titles=group,
                    )
                    first_calls = int(first.pop("api_attempts", 1))
                    api_attempts += first_calls
                    prior_calls += first_calls
                    first_input = int(first.get("input_tokens") or 0)
                    first_output = int(first.get("output_tokens") or 0)
                    input_tokens += first_input
                    output_tokens += first_output
                    prior_inputs = [int(x or 0) for x in _allocate_token_usage(first.get("input_tokens"), [len(t) for t in group])]
                    prior_outputs = [int(x or 0) for x in _allocate_token_usage(first.get("output_tokens"), [len(t) for t in group])]
                    batch_items = first.get("items")

                    if first.get("truncated"):
                        retry = _call_gemma_with_retries(
                            client,
                            "\n".join(group),
                            TRUNCATION_RETRY_TOKENS * len(group),
                            pacer,
                            google_api_key,
                            batch_titles=group,
                        )
                        retry_calls = int(retry.pop("api_attempts", 1))
                        api_attempts += retry_calls
                        prior_calls += retry_calls
                        input_tokens += int(retry.get("input_tokens") or 0)
                        output_tokens += int(retry.get("output_tokens") or 0)
                        retry_inputs = [int(x or 0) for x in _allocate_token_usage(retry.get("input_tokens"), [len(t) for t in group])]
                        retry_outputs = [int(x or 0) for x in _allocate_token_usage(retry.get("output_tokens"), [len(t) for t in group])]
                        prior_inputs = [a + b for a, b in zip(prior_inputs, retry_inputs)]
                        prior_outputs = [a + b for a, b in zip(prior_outputs, retry_outputs)]
                        batch_items = retry.get("items")
                        if retry.get("truncated"):
                            batch_items = None

                    if batch_items is not None:
                        per_title_elapsed = (time.perf_counter() - group_started) / len(group)
                        for title, item, item_input, item_output in zip(group, batch_items, prior_inputs, prior_outputs):
                            record = {
                                **item,
                                "status": "ok" if item.get("output_text") else "error",
                                "attempts": prior_calls,
                                "api_requests": prior_calls / len(group),
                                "input_tokens": item_input,
                                "output_tokens": item_output,
                                "elapsed_seconds": per_title_elapsed,
                                "generation_prompt_version": GEMMA_BATCH_PROMPT_VERSION,
                            }
                            cache.put(title, record)
                            if _complete(record):
                                completed += 1
                            else:
                                errors += 1
                    else:
                        fallback_to_single = True
                        batch_fallbacks += 1
                except DailyQuotaExceeded:
                    raise
                except GemmaConfigurationError:
                    raise
                except Exception as exc:
                    fallback_to_single = True
                    batch_fallbacks += 1
                    prior_calls += int(getattr(exc, "attempts", 0))
                    api_attempts += int(getattr(exc, "attempts", 0))
                    # If a batch call fails before returning usage, preserve its
                    # attempt count and continue safely with one-title requests.
                if fallback_to_single:
                    # Usage from the unusable paired response is attached to
                    # the per-title fallback rows below, so do not count it twice.
                    input_tokens -= sum(prior_inputs)
                    output_tokens -= sum(prior_outputs)
                    for index, title in enumerate(group):
                        run_single(
                            title,
                            prior_input=int(prior_inputs[index] or 0),
                            prior_output=int(prior_outputs[index] or 0),
                            prior_calls=prior_calls / len(group),
                        )
            else:
                run_single(group[0])

            cursor += len(group)
            processed += len(group)
            if processed % 20 == 0 or processed >= len(pending):
                print(
                    f"[gemma4] uncached {processed}/{len(pending)}; new complete={completed}; errors={errors}; "
                    f"api_requests={api_attempts}; paired_requests={batch_requests}; fallback={batch_fallbacks}; "
                    f"input_tokens={input_tokens}; output_tokens={output_tokens}",
                    flush=True,
                )
    except DailyQuotaExceeded:
        return {
            "model_key": model_key,
            "requested": len(titles),
            "completed_this_session": completed,
            "failed_this_session": errors,
            "pending_after_quota": sum(not _complete(value) for value in cache.get_many(titles).values()),
            "elapsed_seconds": time.perf_counter() - start,
            "input_tokens_this_session": input_tokens,
            "output_tokens_this_session": output_tokens,
            "api_attempts_this_session": api_attempts,
            "batch_requests_this_session": batch_requests,
            "batch_fallbacks_this_session": batch_fallbacks,
            "status": "quota_exhausted_checkpoint_saved",
            "checkpoint_preserved": True,
        }
    finally:
        _close_inference_client(client)
    current = cache.get_many(titles)
    return {
        "model_key": model_key,
        "requested": len(titles),
        "completed": sum(_complete(value) for value in current.values()),
        "failed": sum(not _complete(value) for value in current.values()),
        "new_completed": completed,
        "input_tokens_this_session": input_tokens,
        "output_tokens_this_session": output_tokens,
        "api_attempts_this_session": api_attempts,
        "batch_requests_this_session": batch_requests,
        "batch_fallbacks_this_session": batch_fallbacks,
        "batch_size": batch_size,
        "provider": MODELS[model_key]["provider"],
        "elapsed_seconds": time.perf_counter() - start,
        "requests_per_minute_max": 60.0 / interval if interval else None,
        "status": "finished_current_batch",
    }


def _gemma_canary_metrics(
    titles: list[str], cache: Cache, *, batch_size: int = GEMMA_BATCH_SIZE
) -> dict[str, Any]:
    records = cache.get_many(titles)
    complete = [title for title in titles if _complete(records.get(title))]
    qa_pass = [
        title
        for title in complete
        if quality.check_translation(
            title,
            records[title].get("output_text"),
            generation_truncated=bool(records[title].get("truncated")),
        )["status"] == "pass"
    ]
    completion_rate = len(complete) / len(titles) if titles else 1.0
    technical_pass_rate = len(qa_pass) / len(complete) if complete else 0.0
    actual_api_requests = sum(
        float((records.get(title) or {}).get("api_requests", (records.get(title) or {}).get("attempts") or 0))
        for title in titles
    )
    baseline_api_requests = math.ceil(len(titles) / max(1, int(batch_size))) if titles else 0
    retry_overhead_rate = (
        max(0.0, actual_api_requests - baseline_api_requests) / baseline_api_requests
        if baseline_api_requests
        else 0.0
    )
    quality_passed = completion_rate >= MIN_COMPLETION_RATE and technical_pass_rate >= MIN_REPRESENTATIVE_QA_RATE
    return {
        "titles": len(titles),
        "completed": len(complete),
        "technical_pass": len(qa_pass),
        "completion_rate": completion_rate,
        "technical_pass_rate": technical_pass_rate,
        "actual_api_requests": actual_api_requests,
        "baseline_api_requests": baseline_api_requests,
        "retry_overhead_rate": retry_overhead_rate,
        "retry_overhead_passed": retry_overhead_rate <= MAX_GEMMA_BATCH_RETRY_OVERHEAD,
        "failed_titles": [title for title in titles if title not in qa_pass],
        "quality_passed": quality_passed,
        "passed": quality_passed,
    }


def run_gemma_batch_canary(
    context: dict[str, Any],
    google_api_key: str,
    *,
    min_interval_seconds: float | None,
    title_limit: int = GEMMA_BATCH_CANARY_TITLES,
) -> dict[str, Any]:
    """Validate indexed title batches on uncached titles, then select the efficient batch size."""
    _check_prerequisites(context, "gemma4", "full")
    _check_or_write_runtime_lock(context, "gemma4")
    state = context["state"]
    existing = state.get("gemma_batch_canary") or {}
    effective_interval = (
        MODELS["gemma4"]["min_request_interval_seconds"]
        if min_interval_seconds is None
        else float(min_interval_seconds)
    )
    try:
        same_interval = abs(float(existing.get("min_interval_seconds")) - effective_interval) < 1e-9
    except (TypeError, ValueError):
        same_interval = False
    if (
        existing.get("config_id") == context["lock"]["config_id"]
        and existing.get("passed")
        and same_interval
    ):
        return existing

    model_key = "gemma4"
    all_titles = list(dict.fromkeys(context["frame"]["title_zh"].astype(str).tolist()))
    cache = _cache(context, model_key)
    current = cache.get_many(all_titles)
    titles = [title for title in all_titles if not _complete(current.get(title))][: max(1, title_limit)]
    if not titles:
        result = {
            "status": "no_uncached_titles",
            "passed": True,
            "config_id": context["lock"]["config_id"],
            "min_interval_seconds": effective_interval,
            "selected_batch_size": GEMMA_BATCH_SIZE,
            "canary_metrics": {"titles": 0, "completed": 0, "technical_pass": 0, "passed": True},
        }
        state["gemma_batch_canary"] = result
        atomic_json(context["workdir"] / "hybrid_run_state.json", state)
        return result

    generated = _run_gemma(
        titles,
        context,
        google_api_key,
        min_interval_seconds,
        batch_size=GEMMA_BATCH_SIZE,
    )
    if generated.get("status") == "quota_exhausted_checkpoint_saved":
        return {
            "status": generated["status"],
            "passed": False,
            "config_id": context["lock"]["config_id"],
            "min_interval_seconds": effective_interval,
            "selected_batch_size": GEMMA_BATCH_SIZE,
            "generation": generated,
        }

    initial_metrics = _gemma_canary_metrics(titles, cache, batch_size=GEMMA_BATCH_SIZE)
    metrics = initial_metrics
    fallback_generation = None
    if not metrics["quality_passed"]:
        # Keep the attempted batch draft in an audit record, then retry only
        # incomplete or QA-failing items individually. If the canary still
        # misses its gate, do not begin the full stage.
        audit_path = context["workdir"] / "gemma_batch_canary_audit.jsonl"
        before = cache.get_many(metrics["failed_titles"])
        with audit_path.open("a", encoding="utf-8") as audit_file:
            for title in metrics["failed_titles"]:
                record = before.get(title) or {}
                qa = quality.check_translation(title, record.get("output_text") or "")
                audit_file.write(
                    json.dumps(
                        {
                            "source_title": title,
                            "paired_candidate": record.get("output_text"),
                            "paired_qa_flags": qa.get("hard_flags", []),
                            "repair": "retry individually because paired canary missed gate",
                            "recorded_at_utc": utc_now(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                cache.put(
                    title,
                    {
                        **record,
                        "status": "error",
                        "error_text": "Paired canary QA failed; queued for individual retry.",
                    },
                )
        if metrics["failed_titles"]:
            fallback_generation = _run_gemma(
                metrics["failed_titles"],
                context,
                google_api_key,
                min_interval_seconds,
                batch_size=1,
            )
        metrics = _gemma_canary_metrics(titles, cache, batch_size=GEMMA_BATCH_SIZE)

    used_batch_size = (
        GEMMA_BATCH_SIZE
        if initial_metrics["quality_passed"] and initial_metrics["retry_overhead_passed"]
        else min(2, GEMMA_BATCH_SIZE)
    )

    result = {
        "status": "passed" if metrics["quality_passed"] else "quality_gate_failed",
        "passed": bool(metrics["quality_passed"]),
        "config_id": context["lock"]["config_id"],
        "min_interval_seconds": effective_interval,
        "selected_batch_size": used_batch_size,
        "canary_metrics": metrics,
        "initial_canary_metrics": initial_metrics,
        "paired_generation": generated,
        "individual_fallback_generation": fallback_generation,
        "recorded_at_utc": utc_now(),
    }
    state["gemma_batch_canary"] = result
    atomic_json(context["workdir"] / "gemma_batch_canary.json", result)
    atomic_json(context["workdir"] / "hybrid_run_state.json", state)
    return result


def _gemma_qa_repair_instruction(title: str, previous_output: str, hard_flags: list[str]) -> str:
    source_numbers = [raw for raw, _, _ in quality._number_tokens(title)]
    instructions = [
        _gemma_system_instruction(),
        "This is a quality-repair pass because the previous Vietnamese draft failed deterministic checks. ",
        "Treat the Chinese source title as authoritative and rewrite the whole title; do not summarize, ",
        "compress specifications, or merely copy the previous draft. Translate every Han character unless ",
        "it is a proper name rendered in Latin letters. Preserve every distinct number, size, quantity, ",
        "unit, camera count, and channel count from the source. The source-number tokens detected are: ",
        json.dumps(source_numbers, ensure_ascii=False), ". Do not invent numeric specifications. ",
        "The deterministic checks reported: ", json.dumps(hard_flags, ensure_ascii=False), ". ",
        "Previous draft (may contain omissions or mistranslations): ", previous_output,
    ]
    if any(marker in title for marker in ("行车记录仪", "记录仪", "DVR", "行車記錄儀")):
        instructions.extend(
            [
                " Dash-camera terminology: 单录 means single-channel recording; 双录 means dual-channel, ",
                "usually front and rear; 三录 and 四录 mean three-channel and four-channel recording. ",
                "Translate 2路/4路 as two/four channels when they describe video inputs. Do not omit ",
                "screen inches, lens counts, resolution, voltage, or recording-channel counts.",
            ]
        )
    return "".join(instructions)


def _run_gemma_qa_repair(
    context: dict[str, Any],
    google_api_key: str,
    failed_window: dict[str, Any],
    min_interval_seconds: float | None,
) -> dict[str, Any]:
    """Retry technical-QA failures with a specific repair prompt, preserving lineage."""
    model_key = "gemma4"
    cache = _cache(context, model_key)
    all_titles = list(dict.fromkeys(context["frame"]["title_zh"].astype(str).tolist()))
    start_index = max(0, int(failed_window.get("start", 0)))
    end_index = min(len(all_titles), int(failed_window.get("end", len(all_titles))))
    window_titles = all_titles[start_index:end_index]
    before = cache.get_many(window_titles)
    repair_titles = []
    for title in window_titles:
        record = before.get(title)
        if not _complete(record):
            continue
        qa = quality.check_translation(
            title,
            record.get("output_text"),
            generation_truncated=bool(record.get("truncated")),
        )
        if qa["status"] != "pass":
            repair_titles.append(title)
    if not repair_titles:
        return {"status": "no_qa_failures_to_repair", "window_start": start_index, "window_end": end_index}

    interval = MODELS[model_key]["min_request_interval_seconds"] if min_interval_seconds is None else min_interval_seconds
    pacer = RequestPacer(interval)
    client = _new_gemma_client(google_api_key)
    audit_path = context["workdir"] / "gemma_qa_recovery.jsonl"
    recovered = 0
    api_errors = 0
    api_attempts = 0
    input_tokens = 0
    output_tokens = 0
    started = time.perf_counter()
    try:
        for index, title in enumerate(repair_titles, start=1):
            original = before[title]
            original_text = str(original.get("output_text") or "")
            draft, deterministic_repairs = _repair_gemma_specs_from_source(title, original_text)
            original_qa = quality.check_translation(title, original_text)
            deterministic_qa = quality.check_translation(title, draft)
            if deterministic_repairs and deterministic_qa["status"] == "pass":
                cache.put(
                    title,
                    {
                        **original,
                        "output_text": draft,
                        "status": "ok",
                        "generation_prompt_version": GEMMA_DETERMINISTIC_REPAIR_VERSION,
                    },
                )
                with audit_path.open("a", encoding="utf-8") as audit_file:
                    audit_file.write(
                        json.dumps(
                            {
                                "source_title": title,
                                "window_start": start_index,
                                "window_end": end_index,
                                "repair_prompt_version": GEMMA_DETERMINISTIC_REPAIR_VERSION,
                                "deterministic_rules": deterministic_repairs,
                                "candidate": draft,
                                "candidate_hard_flags": [],
                                "accepted": True,
                                "recorded_at_utc": utc_now(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                recovered += 1
                continue
            accepted = False
            title_api_attempts = 0
            title_elapsed = 0.0
            title_input_tokens = 0
            title_output_tokens = 0
            for repair_attempt in range(1, MAX_GEMMA_QA_REPAIR_ATTEMPTS + 1):
                system_instruction = _gemma_qa_repair_instruction(
                    title, draft, original_qa.get("hard_flags", [])
                )
                request_started = time.perf_counter()
                attempts_for_request = 0
                try:
                    output = _call_gemma_with_retries(
                        client,
                        title,
                        MAX_NEW_TOKENS,
                        pacer,
                        google_api_key,
                        system_instruction=system_instruction,
                    )
                    attempts_for_request += int(output.pop("api_attempts", 1))
                    if output["truncated"]:
                        initial_input = output.get("input_tokens") or 0
                        initial_output = output.get("output_tokens") or 0
                        retry = _call_gemma_with_retries(
                            client,
                            title,
                            TRUNCATION_RETRY_TOKENS,
                            pacer,
                            google_api_key,
                            system_instruction=system_instruction,
                        )
                        attempts_for_request += int(retry.pop("api_attempts", 1))
                        retry["input_tokens"] = initial_input + (retry.get("input_tokens") or 0)
                        retry["output_tokens"] = initial_output + (retry.get("output_tokens") or 0)
                        output = retry
                    api_attempts += attempts_for_request
                    title_api_attempts += attempts_for_request
                    title_elapsed += time.perf_counter() - request_started
                    input_tokens += int(output.get("input_tokens") or 0)
                    output_tokens += int(output.get("output_tokens") or 0)
                    title_input_tokens += int(output.get("input_tokens") or 0)
                    title_output_tokens += int(output.get("output_tokens") or 0)
                    candidate = str(output.get("output_text") or "")
                    qa = quality.check_translation(
                        title, candidate, generation_truncated=bool(output.get("truncated"))
                    )
                    audit = {
                        "source_title": title,
                        "window_start": start_index,
                        "window_end": end_index,
                        "repair_prompt_version": GEMMA_QA_REPAIR_PROMPT_VERSION,
                        "attempt": repair_attempt,
                        "previous_hard_flags": original_qa.get("hard_flags", []),
                        "deterministic_rules": deterministic_repairs,
                        "candidate": candidate,
                        "candidate_hard_flags": qa.get("hard_flags", []),
                        "accepted": qa["status"] == "pass",
                        "api_attempts": attempts_for_request,
                        "input_tokens": output.get("input_tokens"),
                        "output_tokens": output.get("output_tokens"),
                        "recorded_at_utc": utc_now(),
                    }
                    with audit_path.open("a", encoding="utf-8") as audit_file:
                        audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                    if qa["status"] == "pass":
                        cache.put(
                            title,
                            {
                                **output,
                                "status": "ok",
                                "attempts": int(original.get("attempts") or 0) + title_api_attempts,
                                "api_requests": float(original.get("api_requests", original.get("attempts") or 0))
                                + title_api_attempts,
                                "elapsed_seconds": float(original.get("elapsed_seconds") or 0.0)
                                + title_elapsed,
                                "input_tokens": int(original.get("input_tokens") or 0)
                                + title_input_tokens,
                                "output_tokens": int(original.get("output_tokens") or 0)
                                + title_output_tokens,
                                "generation_prompt_version": GEMMA_QA_REPAIR_PROMPT_VERSION,
                            },
                        )
                        recovered += 1
                        accepted = True
                        break
                    draft = candidate or draft
                except DailyQuotaExceeded as exc:
                    return {
                        "status": "quota_exhausted_checkpoint_saved",
                        "error": str(exc),
                        "window_start": start_index,
                        "window_end": end_index,
                        "repaired": recovered,
                        "api_errors": api_errors,
                        "api_attempts": api_attempts,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                except GemmaConfigurationError:
                    raise
                except Exception as exc:
                    api_errors += 1
                    failed_attempts = int(getattr(exc, "attempts", 0))
                    api_attempts += failed_attempts
                    title_api_attempts += failed_attempts
                    title_elapsed += time.perf_counter() - request_started
                    with audit_path.open("a", encoding="utf-8") as audit_file:
                        audit_file.write(
                            json.dumps(
                                {
                                    "source_title": title,
                                    "window_start": start_index,
                                    "window_end": end_index,
                                    "repair_prompt_version": GEMMA_QA_REPAIR_PROMPT_VERSION,
                                    "attempt": repair_attempt,
                                    "accepted": False,
                                    "error": _safe_error(exc, google_api_key),
                                    "elapsed_seconds": time.perf_counter() - request_started,
                                    "recorded_at_utc": utc_now(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            if not accepted and index % 10 == 0:
                print(
                    f"[gemma4-repair] reviewed {index}/{len(repair_titles)}; repaired={recovered}; api_errors={api_errors}",
                    flush=True,
                )
        after = cache.get_many(window_titles)
        after_failures = sum(
            _complete(after.get(title))
            and quality.check_translation(title, after[title].get("output_text"))["status"] != "pass"
            for title in window_titles
        )
        report = {
            "status": "repair_finished",
            "repair_prompt_version": GEMMA_QA_REPAIR_PROMPT_VERSION,
            "window_start": start_index,
            "window_end": end_index,
            "qa_failures_before": len(repair_titles),
            "qa_failures_after": int(after_failures),
            "repaired": recovered,
            "api_errors": api_errors,
            "api_attempts": api_attempts,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": time.perf_counter() - started,
            "audit_path": str(audit_path),
            "recorded_at_utc": utc_now(),
        }
        atomic_json(context["workdir"] / "gemma_qa_recovery_report.json", report)
        print(
            f"[gemma4-repair] window {start_index}:{end_index}; "
            f"repaired={recovered}/{len(repair_titles)}; QA failures after={after_failures}",
            flush=True,
        )
        return report
    finally:
        _close_inference_client(client)


def _load_qwen(revision: str, hf_token: str | None, device_index: int = 0) -> tuple[Any, Any]:
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 needs a Kaggle GPU. Select GPU accelerator before running this task.")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested cuda:{device_index}, but only {torch.cuda.device_count()} GPU is visible.")
    if torch.cuda.get_device_properties(device_index).total_memory < 14 * 1024**3:
        raise RuntimeError("Qwen3.5-9B 8-bit run expects a GPU with at least 14 GiB VRAM; stop rather than spill to CPU.")
    quantization = BitsAndBytesConfig(load_in_8bit=True)
    model_id = MODELS["qwen35"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=hf_token)
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=hf_token,
        quantization_config=quantization,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device_index},
    )
    model.eval()
    return model, tokenizer


def _qwen_messages(title: str) -> list[dict[str, Any]]:
    prompt = (
        "Translate this Chinese e-commerce product title into Vietnamese. "
        "Return only the Vietnamese title. Preserve brand names, English brand/model terms, "
        "alphanumeric model codes, numbers, currency amounts and units. Do not add explanations.\n\n"
        + title
    )
    return [{"role": "user", "content": prompt}]


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(response, "status_code", None)
    )
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _qwen_eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """Return every configured Qwen end token, including multi-EOS model configs."""
    token_ids: set[int] = set()

    def add_ids(value: Any) -> None:
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        for value_item in values:
            try:
                token_ids.add(int(value_item))
            except (TypeError, ValueError):
                continue

    add_ids(getattr(tokenizer, "eos_token_id", None))
    add_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    add_ids(getattr(getattr(model, "config", None), "eos_token_id", None))

    convert_token = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_token):
        im_end_id = convert_token("<|im_end|>")
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        try:
            im_end_id = int(im_end_id)
            if unknown_id is None or im_end_id != int(unknown_id):
                token_ids.add(im_end_id)
        except (TypeError, ValueError):
            pass
    return sorted(token_ids)


def _qwen_generate(
    model: Any, tokenizer: Any, titles: list[str], device: str, max_new_tokens: int
) -> list[dict[str, Any]]:
    import torch

    messages = [_qwen_messages(title) for title in titles]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = inputs.to(device)
    input_width = int(inputs["input_ids"].shape[-1])
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1}
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    eos_ids = _qwen_eos_token_ids(model, tokenizer)
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    with torch.inference_mode():
        sequences = model.generate(**inputs, **kwargs)
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    eos_set = set(eos_ids)
    results = []
    for sequence in sequences:
        generated_ids = sequence[input_width:].tolist()
        token_count = len(generated_ids)
        found_eos = False
        for pos, token_id in enumerate(generated_ids):
            if int(token_id) in eos_set:
                token_count = pos + 1
                found_eos = True
                break
        text = tokenizer.decode(
            generated_ids[:token_count], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        repetition_detected = quality._repeated_phrase(text)
        results.append(
            {
                "output_text": text,
                "input_tokens": int(inputs["attention_mask"][len(results)].sum().item()),
                "output_tokens": token_count,
                "truncated": not found_eos and token_count >= max_new_tokens,
                "repetition_detected": repetition_detected,
            }
        )
    return results


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in text or "cuda error: out of memory" in text


def _run_qwen(
    titles: list[str], context: dict[str, Any], hf_token: str | None, batch_size: int = 1
) -> dict[str, Any]:
    model_key = "qwen35"
    cache = _cache(context, model_key)
    cached = cache.get_many(titles)
    pending = [title for title in titles if not _complete(cached.get(title))]
    if not pending:
        return {"model_key": model_key, "requested": len(titles), "completed": len(titles), "cached": len(titles)}
    model, tokenizer = _load_qwen(context["lock"]["models"][model_key]["revision"], hf_token)
    import torch

    device = "cuda:0"
    microbatch = max(1, int(batch_size))
    completed = 0
    failures = 0
    input_tokens = 0
    output_tokens = 0
    started = time.perf_counter()
    cursor = 0
    last_checkpoint = 0
    checkpoint_interval = max(1, int(context.get("cache_checkpoint_interval", 100)))
    checkpoint_callback = context.get("cache_checkpoint_callback")
    try:
        while cursor < len(pending):
            group = pending[cursor : cursor + microbatch]
            group_started = time.perf_counter()
            try:
                outputs = _qwen_generate(model, tokenizer, group, device, MAX_NEW_TOKENS)
                truncated_indices = [
                    index
                    for index, item in enumerate(outputs)
                    if item["truncated"] and not item.get("repetition_detected")
                ]
                for output_index in truncated_indices:
                    retry = _qwen_generate(
                        model, tokenizer, [group[output_index]], device, TRUNCATION_RETRY_TOKENS
                    )[0]
                    retry["input_tokens"] = int(outputs[output_index].get("input_tokens") or 0) + int(
                        retry.get("input_tokens") or 0
                    )
                    retry["output_tokens"] = int(outputs[output_index].get("output_tokens") or 0) + int(
                        retry.get("output_tokens") or 0
                    )
                    outputs[output_index] = retry
                elapsed_each = (time.perf_counter() - group_started) / max(1, len(group))
                for title, output in zip(group, outputs):
                    repetition_detected = bool(output.get("repetition_detected")) or quality._repeated_phrase(
                        str(output.get("output_text") or "")
                    )
                    incomplete = bool(output["truncated"] or repetition_detected or not output["output_text"])
                    record = {
                        **output,
                        "status": "error" if incomplete else "ok",
                        "error_text": "repeated_phrase_guard" if repetition_detected else (
                            "truncated_output" if output["truncated"] else None
                        ),
                        "attempts": 1,
                        "elapsed_seconds": elapsed_each,
                        "model_version": f"transformers-local/{_package_version('transformers')}",
                        "response_id": None,
                    }
                    cache.put(title, record)
                    input_tokens += int(record.get("input_tokens") or 0)
                    output_tokens += int(record.get("output_tokens") or 0)
                    if _complete(record):
                        completed += 1
                    else:
                        failures += 1
                cursor += len(group)
                if checkpoint_callback and (
                    cursor - last_checkpoint >= checkpoint_interval or cursor >= len(pending)
                ):
                    checkpoint_callback(cache.path)
                    last_checkpoint = cursor
            except Exception as exc:
                if _is_cuda_oom(exc) and microbatch > 1:
                    microbatch = max(1, microbatch // 2)
                    torch.cuda.empty_cache()
                    print(f"[qwen35] CUDA OOM; reducing local batch size to {microbatch}", flush=True)
                    continue
                if _is_cuda_oom(exc):
                    raise RuntimeError(
                        "Qwen local generation ran out of GPU memory even at batch size 1; "
                        "use a GPU with more VRAM or lower quantization in a separately validated run."
                    ) from exc
                raise
            if cursor % 20 == 0 or cursor >= len(pending):
                print(
                    f"[qwen35] uncached {cursor}/{len(pending)}; complete={completed}; errors={failures}; "
                    f"input_tokens={input_tokens}; output_tokens={output_tokens}",
                    flush=True,
                )
    finally:
        try:
            if checkpoint_callback:
                checkpoint_callback(cache.path)
        finally:
            del model, tokenizer
            torch.cuda.empty_cache()
    total = cache.get_many(titles)
    return {
        "model_key": model_key,
        "requested": len(titles),
        "completed": sum(_complete(value) for value in total.values()),
        "failed": sum(not _complete(value) for value in total.values()),
        "new_completed": completed,
        "input_tokens_this_session": input_tokens,
        "output_tokens_this_session": output_tokens,
        "provider": MODELS[model_key]["provider"],
        "batch_size_used": microbatch,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "finished_current_batch",
    }


def _evaluate_model_titles(
    model_key: str, titles: list[str], cache: Cache
) -> dict[str, Any]:
    results = cache.get_many(titles)
    qa_results = {
        title: quality.check_translation(
            title,
            (results.get(title) or {}).get("output_text"),
            generation_truncated=bool((results.get(title) or {}).get("truncated")),
        )
        for title in titles
    }
    completed = sum(_complete(results.get(title)) for title in titles)
    passing = sum(item["status"] == "pass" for item in qa_results.values())
    return {
        "model_key": model_key,
        "requested": len(titles),
        "completed": completed,
        "completion_rate": completed / len(titles) if titles else 0.0,
        "technical_pass": passing,
        "technical_pass_rate": passing / len(titles) if titles else 0.0,
        "input_tokens": sum(int((results.get(title) or {}).get("input_tokens") or 0) for title in titles),
        "output_tokens": sum(int((results.get(title) or {}).get("output_tokens") or 0) for title in titles),
        "api_attempts": sum(int((results.get(title) or {}).get("attempts") or 0) for title in titles),
        "qa": qa_results,
        "cache": results,
    }


def _stage_gate(mode: str, model_key: str, titles: list[str], samples: dict[str, Any], cache: Cache) -> dict[str, Any]:
    evaluation = _evaluate_model_titles(model_key, titles, cache)
    if mode == "smoke":
        passed = (
            evaluation["completion_rate"] >= MIN_COMPLETION_RATE
            and evaluation["technical_pass_rate"] >= SMOKE_MIN_TECHNICAL_PASS_RATE
        )
        return {key: value for key, value in evaluation.items() if key not in {"qa", "cache"}} | {
            "passed": passed,
            "gate": (
                f"completion >= {MIN_COMPLETION_RATE:.0%}; technical QA >= "
                f"{SMOKE_MIN_TECHNICAL_PASS_RATE:.0%}"
            ),
        }
    representative = set(samples[mode]["representative"])
    challenge = set(samples[mode]["challenge"])
    rep_qa = [evaluation["qa"][title]["status"] == "pass" for title in titles if title in representative]
    challenge_qa = [evaluation["qa"][title]["status"] == "pass" for title in titles if title in challenge]
    rep_rate = sum(rep_qa) / len(rep_qa) if rep_qa else 0.0
    challenge_rate = sum(challenge_qa) / len(challenge_qa) if challenge_qa else 0.0
    passed = evaluation["completion_rate"] >= MIN_COMPLETION_RATE and rep_rate >= MIN_REPRESENTATIVE_QA_RATE
    return {
        key: value for key, value in evaluation.items() if key not in {"qa", "cache"}
    } | {
        "representative_technical_pass_rate": rep_rate,
        "challenge_technical_pass_rate": challenge_rate,
        "passed": passed,
        "gate": (
            f"completion >= {MIN_COMPLETION_RATE:.0%}; representative technical pass >= "
            f"{MIN_REPRESENTATIVE_QA_RATE:.0%}; challenge rate reported separately"
        ),
    }


def _full_window_gates(titles: list[str], all_titles: list[str], cache: Cache) -> list[dict[str, Any]]:
    selected = set(titles)
    cached = cache.get_many(all_titles)
    reports = []
    for start in range(0, len(all_titles), FULL_WINDOW_SIZE):
        window = all_titles[start : start + FULL_WINDOW_SIZE]
        if not selected.intersection(window):
            continue
        records = [cached.get(title) for title in window]
        if any(record is None for record in records):
            continue
        complete = [_complete(record) for record in records]
        qa_status = [
            quality.check_translation(
                title,
                (record or {}).get("output_text"),
                generation_truncated=bool((record or {}).get("truncated")),
            )["status"] == "pass"
            for title, record in zip(window, records)
            if _complete(record)
        ]
        incomplete_rate = 1.0 - (sum(complete) / len(window)) if window else 0.0
        qa_failure_rate = 1.0 - (sum(qa_status) / len(qa_status)) if qa_status else (1.0 if window else 0.0)
        passed = (
            incomplete_rate <= FULL_MAX_INCOMPLETE_RATE + 1e-12
            and qa_failure_rate <= FULL_MAX_QA_FAILURE_RATE + 1e-12
        )
        reports.append(
            {
                "start": start,
                "end": start + len(window),
                "titles": len(window),
                "incomplete_rate": incomplete_rate,
                "technical_qa_failure_rate": qa_failure_rate,
                "passed": passed,
            }
        )
        if not passed:
            break
    return reports


def _write_stage_report(context: dict[str, Any], result: dict[str, Any]) -> None:
    model_key = context["model_key"]
    atomic_json(context["workdir"] / f"{model_key}_report.json", result)
    lines = [
        f"# Báo cáo dịch title: {model_key}",
        "",
        f"- Stage: {context['mode']}",
        f"- Status: {result.get('status')}",
        f"- Passed: {result.get('passed')}",
        f"- Config ID: {context['lock']['config_id']}",
        f"- Model: {context['lock']['models'][model_key]['model_id']}",
        f"- Source titles: {result.get('requested', result.get('unique_titles'))}",
        f"- Completed: {result.get('completed', result.get('completed_this_session'))}",
        f"- Remaining: {result.get('remaining_unique_titles', result.get('pending_after_quota'))}",
        f"- Technical pass rate: {result.get('technical_pass_rate')}",
        f"- Representative QA pass rate: {result.get('representative_technical_pass_rate')}",
        f"- Challenge QA pass rate: {result.get('challenge_technical_pass_rate')}",
        f"- Input tokens: {result.get('input_tokens', (result.get('generation') or {}).get('input_tokens_this_session'))}",
        f"- Output tokens: {result.get('output_tokens', (result.get('generation') or {}).get('output_tokens_this_session'))}",
        f"- API attempts: {result.get('api_attempts')}",
        f"- Output: {result.get('output_path')}",
        f"- Output SHA-256: {result.get('output_sha256')}",
        "",
        "API or model failures remain pending in SQLite and are eligible for resume. No key is written here.",
        "Technical QA and model agreement do not establish semantic correctness.",
        "",
    ]
    atomic_text(context["workdir"] / f"{model_key}_report.md", "\n".join(lines))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def run_stage(
    context: dict[str, Any],
    *,
    google_api_key: str | None = None,
    hf_token: str | None = None,
    batch_size: int | None = None,
    gemma_batch_size: int = 1,
    full_titles_per_session: int = DEFAULT_FULL_TITLES_PER_SESSION,
    gemma_min_interval_seconds: float | None = None,
    allow_unvalidated_full: bool = False,
    defer_full_quality_review: bool = False,
    allow_qwen_hardware_change: bool = False,
    allow_qwen_batch_change: bool = False,
) -> dict[str, Any]:
    mode = context["mode"]
    model_key = context["model_key"]
    if allow_unvalidated_full and (mode != "full" or model_key != "qwen35"):
        raise ValueError("allow_unvalidated_full is an explicit full-run option for qwen35 only.")
    if defer_full_quality_review and (mode != "full" or model_key not in {"qwen35", "gemma4"}):
        raise ValueError("defer_full_quality_review is supported only for full runs of qwen35 or gemma4.")
    if allow_qwen_hardware_change and model_key != "qwen35":
        raise ValueError("allow_qwen_hardware_change is an explicit option for qwen35 only.")
    if allow_qwen_batch_change and (model_key != "qwen35" or mode not in {"smoke", "full"}):
        raise ValueError("allow_qwen_batch_change is an explicit option for qwen35 smoke/full only.")
    _check_prerequisites(
        context,
        model_key,
        mode,
        allow_unvalidated_full=allow_unvalidated_full,
        defer_full_quality_review=defer_full_quality_review,
    )
    _check_or_write_runtime_lock(
        context, model_key, allow_qwen_hardware_change=allow_qwen_hardware_change
    )
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    google_api_key = google_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if model_key == "gemma4" and not google_api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY (Google AI Studio key) in the runtime secret store. "
            "This key is used only for Gemma requests and is never written to checkpoints."
        )

    all_titles = list(dict.fromkeys(context["frame"]["title_zh"].astype(str).tolist()))
    stage_titles = _stage_titles(mode, context["samples"], context["frame"])
    cache = _cache(context, model_key)
    previous = context["state"]["models"].setdefault(model_key, {"stages": {}}).setdefault("stages", {}).get(mode, {})
    started = time.perf_counter()

    if mode == "full":
        complete_before = cache.get_many(all_titles)
        pending = [title for title in all_titles if not _complete(complete_before.get(title))]
        session_limit = max(1, int(full_titles_per_session))
        session_titles = pending[:session_limit]
    else:
        session_titles = stage_titles

    if model_key == "gemma4":
        generation = _run_gemma(
            session_titles,
            context,
            google_api_key or "",
            gemma_min_interval_seconds,
            batch_size=gemma_batch_size if mode == "full" else 1,
        )
    else:
        locked_batch_size = int(context.get("qwen_batch_size", 1))
        selected_batch_size = locked_batch_size if batch_size is None else max(1, int(batch_size))
        if selected_batch_size != locked_batch_size and not allow_qwen_batch_change:
            raise RuntimeError(
                f"Qwen run is locked to batch size {locked_batch_size}; "
                f"requested {selected_batch_size}. Resume with the locked value or start a new run."
            )
        generation = _run_qwen(session_titles, context, hf_token, batch_size=selected_batch_size)
        generation["batch_size_locked_in_config"] = locked_batch_size
        generation["batch_size_requested"] = selected_batch_size

    result: dict[str, Any] = {
        "mode": mode,
        "model_key": model_key,
        "config_id": context["lock"]["config_id"],
        "requested": len(stage_titles),
        "session_titles": len(session_titles),
        "generation": generation,
        "started_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "semantic_verified": False,
        "validation_bypassed": bool(mode == "full" and allow_unvalidated_full),
    }
    if generation.get("status") == "quota_exhausted_checkpoint_saved":
        result.update(
            {
                "passed": False,
                "status": "quota_exhausted_checkpoint_saved",
                "checkpoint_preserved": True,
            }
        )
        _write_stage_report(context, result)
        context["state"]["models"][model_key]["stages"][mode] = result
        atomic_json(context["workdir"] / "hybrid_run_state.json", context["state"])
        return result

    if mode in {"smoke", "pilot", "validate"}:
        gate = _stage_gate(mode, model_key, stage_titles, context["samples"], cache)
        result.update(gate)
        result["status"] = "passed" if result["passed"] else "quality_gate_failed"
    else:
        reports = [] if defer_full_quality_review else _full_window_gates(session_titles, all_titles, cache)
        failed_gate = next((window for window in reports if not window["passed"]), None)
        recovery_log: list[dict[str, Any]] = []
        recovery_quota_exhausted = False

        # Retry only already-recorded incomplete titles in the failed window. This
        # preserves the strict window gate while giving transient provider 500s a
        # second bounded chance before stopping the long-running full job.
        if (
            not defer_full_quality_review
            and model_key == "gemma4"
            and _full_gate_stop_is_incomplete_only({"status": "stopped_quality_gate", "failed_window": failed_gate or {}})
        ):
            for recovery_round in range(1, MAX_FULL_INCOMPLETE_RECOVERY_ROUNDS + 1):
                assert failed_gate is not None
                start_index = int(failed_gate["start"])
                end_index = int(failed_gate["end"])
                window_titles = all_titles[start_index:end_index]
                window_cache = cache.get_many(window_titles)
                retry_titles = [
                    title
                    for title in window_titles
                    if window_cache.get(title) is not None and not _complete(window_cache.get(title))
                ]
                if not retry_titles:
                    break
                print(
                    f"[gemma4] full gate window {start_index}:{end_index} has "
                    f"{len(retry_titles)} incomplete title(s); targeted recovery "
                    f"{recovery_round}/{MAX_FULL_INCOMPLETE_RECOVERY_ROUNDS}",
                    flush=True,
                )
                recovery = _run_gemma(
                    retry_titles,
                    context,
                    google_api_key or "",
                    gemma_min_interval_seconds,
                )
                recovery_log.append(
                    {
                        "round": recovery_round,
                        "window_start": start_index,
                        "window_end": end_index,
                        "titles_retried": len(retry_titles),
                        "completed": recovery.get("new_completed", recovery.get("completed_this_session")),
                        "failed": recovery.get("failed_this_session", recovery.get("failed")),
                        "api_status": recovery.get("status"),
                    }
                )
                if recovery.get("status") == "quota_exhausted_checkpoint_saved":
                    recovery_quota_exhausted = True
                    break
                reports = _full_window_gates(session_titles, all_titles, cache)
                failed_gate = next((window for window in reports if not window["passed"]), None)
                if not failed_gate or not _full_gate_stop_is_incomplete_only(
                    {"status": "stopped_quality_gate", "failed_window": failed_gate}
                ):
                    break

        full_cache = cache.get_many(all_titles)
        pending_after = [title for title in all_titles if not _complete(full_cache.get(title))]
        result["input_tokens"] = sum(
            int((full_cache.get(title) or {}).get("input_tokens") or 0) for title in all_titles
        )
        result["output_tokens"] = sum(
            int((full_cache.get(title) or {}).get("output_tokens") or 0) for title in all_titles
        )
        result["api_attempts"] = sum(
            float((full_cache.get(title) or {}).get("api_requests", (full_cache.get(title) or {}).get("attempts")) or 0)
            for title in all_titles
        )
        if recovery_log:
            result["incomplete_recovery"] = recovery_log
        failed_gate = next((window for window in reports if not window["passed"]), None)
        result["window_gates"] = reports
        result["completed_unique_titles"] = len(all_titles) - len(pending_after)
        result["total_unique_titles"] = len(all_titles)
        result["remaining_unique_titles"] = len(pending_after)
        result["quality_review_deferred"] = bool(defer_full_quality_review)
        if defer_full_quality_review:
            result["quality_review_note"] = (
                "Full-run window QA gates were intentionally deferred; the completed Parquet will contain "
                "per-title technical QA flags for post-run review. Semantic accuracy remains unverified."
            )
        if recovery_quota_exhausted:
            result.update(
                {
                    "passed": False,
                    "status": "quota_exhausted_checkpoint_saved",
                    "checkpoint_preserved": True,
                }
            )
        elif failed_gate:
            result.update({"passed": False, "status": "stopped_quality_gate", "failed_window": failed_gate})
        elif pending_after:
            result.update({"passed": False, "status": "checkpoint_saved_continue", "checkpoint_preserved": True})
        else:
            result.update({"passed": True, "status": "all_titles_complete"})
            output_path = export_model_parquet(context, model_key)
            result["output_path"] = str(output_path)
            result["output_sha256"] = quality.sha256_file(output_path)

    result["elapsed_seconds"] = time.perf_counter() - started
    _write_stage_report(context, result)
    context["state"]["models"][model_key]["stages"][mode] = result
    atomic_json(context["workdir"] / "hybrid_run_state.json", context["state"])
    if mode == "full" and result.get("status") == "all_titles_complete":
        _export_comparison_if_ready(context)
        for key in MODELS:
            stage = context["state"]["models"].get(key, {}).get("stages", {}).get("full", {})
            output_path = context["workdir"] / f"bilingual_zh_vi_cado_title_{key}.parquet"
            if stage.get("status") == "all_titles_complete" and output_path.exists():
                stage["output_path"] = str(output_path)
                stage["output_sha256"] = quality.sha256_file(output_path)
                model_context = {**context, "model_key": key}
                _write_stage_report(model_context, stage)
        atomic_json(context["workdir"] / "hybrid_run_state.json", context["state"])
    return result


def _export_model_frame(context: dict[str, Any], model_key: str) -> pd.DataFrame:
    frame = context["frame"].copy()
    titles = list(dict.fromkeys(frame["title_zh"].astype(str).tolist()))
    cache = _cache(context, model_key)
    records = cache.get_many(titles)
    if any(not _complete(records.get(title)) for title in titles):
        raise RuntimeError(f"Cannot export {model_key}: at least one unique title has no complete cached translation.")

    model = MODELS[model_key]
    prefix = model_key
    frame[model["candidate_column"]] = frame["title_zh"].astype(str).map(
        lambda title: records[title]["output_text"]
    )
    checks = {
        title: quality.check_translation(
            title,
            records[title]["output_text"],
            generation_truncated=records[title]["truncated"],
        )
        for title in titles
    }
    frame[f"title_vi_{prefix}_qa_status"] = frame["title_zh"].astype(str).map(
        lambda title: checks[title]["status"]
    )
    frame[f"title_vi_{prefix}_qa_flags"] = frame["title_zh"].astype(str).map(
        lambda title: json.dumps(checks[title]["hard_flags"], ensure_ascii=False)
    )
    frame[f"title_vi_{prefix}_advisory_flags"] = frame["title_zh"].astype(str).map(
        lambda title: json.dumps(checks[title].get("advisory_flags", []), ensure_ascii=False)
    )
    frame[f"title_vi_{prefix}_model_id"] = model["model_id"]
    frame[f"title_vi_{prefix}_model_revision"] = context["lock"]["models"][model_key]["revision"]
    frame[f"title_vi_{prefix}_api_version"] = frame["title_zh"].astype(str).map(
        lambda title: records[title].get("model_version")
    )
    frame[f"title_vi_{prefix}_generation_prompt_version"] = frame["title_zh"].astype(str).map(
        lambda title: records[title].get("generation_prompt_version") or PROMPT_VERSION
    )
    frame[f"title_vi_{prefix}_api_requests_attributed"] = frame["title_zh"].astype(str).map(
        lambda title: records[title].get("api_requests", records[title].get("attempts") or 0)
    )
    frame[f"title_vi_{prefix}_provider"] = MODELS[model_key]["provider"]
    frame[f"title_vi_{prefix}_semantic_verified"] = False
    frame[f"title_vi_{prefix}_pipeline_config_id"] = context["lock"]["config_id"]
    return frame


def _verify_output_frame(source: pd.DataFrame, output: pd.DataFrame) -> None:
    if len(output) != len(source):
        raise AssertionError(f"Output rows changed: expected {len(source)}, found {len(output)}.")
    if not quality._series_values_equal(source["product_id"], output["product_id"]):
        raise AssertionError("Output product_id values or order changed.")
    for column in quality.ORIGINAL_COLUMNS:
        if column not in output or not quality._series_values_equal(source[column], output[column]):
            raise AssertionError(f"Original source column changed during export: {column}.")


def _verify_output_candidates(
    source: pd.DataFrame,
    output: pd.DataFrame,
    model_key: str,
    lock: dict[str, Any],
    cache: Cache,
) -> None:
    model = MODELS[model_key]
    column = model["candidate_column"]
    if column not in output:
        raise RuntimeError(f"Refusing upload: {output.name} has no {column} field.")
    titles = list(dict.fromkeys(source["title_zh"].astype(str).tolist()))
    records = cache.get_many(titles)
    expected = {title: records[title]["output_text"] for title in titles if _complete(records.get(title))}
    actual = output[column].tolist()
    expected_rows = [expected.get(str(title)) for title in source["title_zh"].tolist()]
    if actual != expected_rows:
        raise RuntimeError(f"Refusing upload: {column} values do not match the completed SQLite cache.")
    config_column = f"title_vi_{model_key}_pipeline_config_id"
    if config_column not in output or not output[config_column].eq(lock["config_id"]).all():
        raise RuntimeError(f"Refusing upload: config ID mismatch in {output.name}.")
    semantic_column = f"title_vi_{model_key}_semantic_verified"
    if semantic_column not in output or output[semantic_column].fillna(True).astype(bool).any():
        raise RuntimeError(f"Refusing upload: {semantic_column} must remain false for this automatic run.")


def _atomic_write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_parquet(temp_path, index=False)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def export_model_parquet(context: dict[str, Any], model_key: str) -> Path:
    frame = _export_model_frame(context, model_key)
    _verify_output_frame(context["frame"], frame)
    name = f"bilingual_zh_vi_cado_title_{model_key}.parquet"
    destination = context["workdir"] / name
    _atomic_write_parquet(frame, destination)
    verified = pd.read_parquet(destination)
    _verify_output_frame(context["frame"], verified)
    return destination


def _export_comparison_if_ready(context: dict[str, Any]) -> Path | None:
    state = context["state"]
    if not all(
        state["models"].get(key, {}).get("stages", {}).get("full", {}).get("status") == "all_titles_complete"
        for key in MODELS
    ):
        return None
    gemma = _cache(context, "gemma4").get_many(context["frame"]["title_zh"].astype(str).tolist())
    qwen = _cache(context, "qwen35").get_many(context["frame"]["title_zh"].astype(str).tolist())
    unique = list(dict.fromkeys(context["frame"]["title_zh"].astype(str).tolist()))
    rows = []
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF()
    except ImportError:
        chrf = None
    for title in unique:
        g = gemma[title]["output_text"]
        q = qwen[title]["output_text"]
        rows.append(
            {
                "title_zh": title,
                "title_vi_candidate_gemma4": g,
                "title_vi_candidate_qwen35": q,
                "candidate_agreement": quality.normalize_title_key(g) == quality.normalize_title_key(q),
                "chrf_gemma4_qwen35": chrf.sentence_score(q, [g]).score if chrf else None,
                "title_vi_semantic_verified": False,
            }
        )
    path = context["workdir"] / "title_translation_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    for key, cache_records in (("gemma4", gemma), ("qwen35", qwen)):
        output_path = context["workdir"] / f"bilingual_zh_vi_cado_title_{key}.parquet"
        if output_path.exists():
            model_frame = pd.read_parquet(output_path)
            agree_map = {
                title: quality.normalize_title_key(gemma[title]["output_text"])
                == quality.normalize_title_key(qwen[title]["output_text"])
                for title in unique
            }
            model_frame["title_vi_candidate_agreement"] = model_frame["title_zh"].astype(str).map(agree_map)
            _atomic_write_parquet(model_frame, output_path)
            _verify_output_frame(context["frame"], pd.read_parquet(output_path))
    return path


def upload_outputs_to_hf(
    workdir: str | Path,
    *,
    hf_token: str,
    source_path: str | Path | None = None,
    repo_id: str = DATASET_ID,
    revision: str = "main",
) -> dict[str, Any]:
    """Upload two immutable model outputs beneath a new run folder; never replace the source parquet."""
    if not hf_token:
        raise ValueError("HF_TOKEN is required for the upload step.")
    root = Path(workdir)
    state_path = root / "hybrid_run_state.json"
    lock_path = root / "hybrid_environment_lock.json"
    if not state_path.exists() or not lock_path.exists():
        raise RuntimeError("No completed hybrid run state or model lock is present.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if any(
        state.get("models", {}).get(key, {}).get("stages", {}).get("full", {}).get("status") != "all_titles_complete"
        for key in MODELS
    ):
        raise RuntimeError("Upload is blocked until both model full runs are complete.")

    pinned_source_path = _resolve_input(source_path, hf_token)
    if quality.sha256_file(pinned_source_path) != DATASET_SHA256:
        raise RuntimeError("Refusing upload: the pinned source dataset could not be revalidated.")
    source_frame = pd.read_parquet(pinned_source_path)
    quality.validate_input_frame(source_frame)

    files = []
    for key in MODELS:
        path = root / f"bilingual_zh_vi_cado_title_{key}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing complete {key} parquet: {path}")
        actual_file_hash = quality.sha256_file(path)
        expected_file_hash = state["models"][key]["stages"]["full"].get("output_sha256")
        if not expected_file_hash or expected_file_hash != actual_file_hash:
            raise RuntimeError(f"Refusing upload: {path.name} differs from its completed run-state checksum.")
        verified = pd.read_parquet(path)
        if len(verified) != EXPECTED_ROWS:
            raise RuntimeError(f"Refusing upload: {path.name} has {len(verified)} rows, expected {EXPECTED_ROWS}.")
        _verify_output_frame(source_frame, verified)
        _verify_output_candidates(
            source_frame,
            verified,
            key,
            lock,
            Cache(root / f"cache_{key}.sqlite3", key, lock["models"][key]["revision"], lock["config_id"]),
        )
        files.append(path)

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_folder = f"{created}_{lock['config_id'][:10]}"
    from huggingface_hub import CommitOperationAdd

    remote_paths = []
    operations = []
    for path in files:
        remote_path = f"{REMOTE_OUTPUT_PREFIX}/{run_folder}/{path.name}"
        operations.append(CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(path)))
        remote_paths.append(remote_path)
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        operations=operations,
        commit_message=f"Add bilingual title translations ({run_folder})",
    )
    commit_oid = getattr(commit, "oid", None)
    if not commit_oid:
        raise RuntimeError("HF commit was submitted but returned no commit SHA; verify the repository before retrying.")
    from huggingface_hub import hf_hub_download

    verified_remote = []
    for path, remote in zip(files, remote_paths):
        if not api.file_exists(
            repo_id=repo_id,
            filename=remote,
            repo_type="dataset",
            revision=commit_oid,
        ):
            raise RuntimeError(f"HF commit {commit_oid} does not expose expected file {remote}.")
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                repo_type="dataset",
                revision=commit_oid,
                token=hf_token,
            )
        )
        expected_hash = quality.sha256_file(path)
        actual_hash = quality.sha256_file(downloaded)
        if actual_hash != expected_hash:
            raise RuntimeError(f"HF read-back checksum mismatch for {remote}.")
        verified_remote.append({"remote_path": remote, "sha256_verified": actual_hash})
    result = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "run_folder": run_folder,
        "config_id": lock["config_id"],
        "uploaded_at_utc": utc_now(),
        "commit_url": getattr(commit, "commit_url", None),
        "commit_oid": commit_oid,
        "readback_verified": True,
        "verified_remote": verified_remote,
        "files": [
            {"local_path": str(path), "remote_path": remote, "sha256": quality.sha256_file(path), "rows": EXPECTED_ROWS}
            for path, remote in zip(files, remote_paths)
        ],
    }
    atomic_json(root / "title_translation_upload_manifest.json", result)
    return result


def environment_summary() -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("huggingface_hub", "sacrebleu", "pyarrow")
        },
        "execution_mode": "hosted API providers; GPU and model-weight downloads are not required",
    }
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu_count"] = int(torch.cuda.device_count())
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except ImportError:
        result["cuda_available"] = False
        result["gpu_count"] = 0
        result["gpus"] = []
    return result
