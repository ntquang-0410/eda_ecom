"""Safely migrate a paused Gemma cache to a newer batch/config revision.

The source checkpoint is never modified. Cached generation outputs and passed
full-window QA evidence are retained where compatible with the active QA rules.
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
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import title_translation_hybrid as pipeline


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "bilingual_zh_vi_cado.parquet"
OLD_DIR = ROOT / "title_translation_cado_free_hybrid_run_v14"
NEW_DIR = ROOT / "title_translation_cado_free_hybrid_run_v15"


def _load_and_validate_old() -> tuple[dict[str, Any], dict[str, Any], list[tuple[Any, ...]]]:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source Parquet not found: {SOURCE}")
    if not OLD_DIR.is_dir():
        raise FileNotFoundError(f"Old checkpoint not found: {OLD_DIR}")

    old_lock = json.loads((OLD_DIR / "hybrid_environment_lock.json").read_text(encoding="utf-8"))
    old_state = json.loads((OLD_DIR / "hybrid_run_state.json").read_text(encoding="utf-8"))
    if old_lock.get("dataset_sha256") != pipeline.DATASET_SHA256:
        raise RuntimeError("The source checkpoint is not locked to the current source dataset.")
    if old_state.get("config_id") != old_lock.get("config_id"):
        raise RuntimeError("The source run state and environment lock do not match.")
    if old_lock.get("config_id") != pipeline.fingerprint(old_lock.get("config", {})):
        raise RuntimeError("The source environment lock fingerprint is invalid.")
    if pipeline.quality.sha256_file(SOURCE) != pipeline.DATASET_SHA256:
        raise RuntimeError("The local source Parquet no longer matches its pinned SHA-256.")

    old_config = dict(old_lock["config"])
    new_config = pipeline._config_payload(old_lock["models"])
    old_compare = dict(old_config)
    new_compare = dict(new_config)
    old_compare.pop("qa_version", None)
    old_compare.pop("gemma_normalization_version", None)
    new_compare.pop("qa_version", None)
    new_compare.pop("gemma_normalization_version", None)
    old_compare.get("models", {}).get("gemma4", {}).pop("template_version", None)
    new_compare.get("models", {}).get("gemma4", {}).pop("template_version", None)
    old_compare.get("generation", {}).pop("gemma_batch_size", None)
    new_compare.get("generation", {}).pop("gemma_batch_size", None)
    if old_compare != new_compare:
        raise RuntimeError(
            "Unexpected checkpoint config changes beyond deterministic QA/normalization; migration refused."
        )
    if new_config.get("qa_version") != pipeline.QA_VERSION:
        raise RuntimeError("The active pipeline QA version does not match the expected current version.")

    db_path = OLD_DIR / "cache_gemma4.sqlite3"
    with sqlite3.connect(db_path) as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(translations)")]
        required = {
            "source_title", "model_key", "model_id", "model_revision", "config_id",
            "output_text", "status", "truncated", "input_tokens", "output_tokens",
            "attempts", "elapsed_seconds", "error_text", "model_version", "response_id",
        }
        if not required.issubset(columns):
            raise RuntimeError("The source Gemma cache schema is incomplete; migration refused.")
        prompt_version_column = (
            "generation_prompt_version" if "generation_prompt_version" in columns else "''"
        )
        api_requests_column = "api_requests" if "api_requests" in columns else "attempts"
        rows = db.execute(
            "SELECT source_title,model_key,model_id,model_revision,config_id,output_text,status,"
            "truncated,input_tokens,output_tokens,attempts,elapsed_seconds,error_text,model_version,response_id,"
            f"{prompt_version_column},{api_requests_column} "
            "FROM translations ORDER BY rowid"
        ).fetchall()

    model = old_lock["models"]["gemma4"]
    for row in rows:
        title, model_key, model_id, revision, config_id = row[:5]
        if model_key != "gemma4" or model_id != model["model_id"] or revision != model["revision"]:
            raise RuntimeError("A cache row belongs to another model/revision; migration refused.")
        if config_id != old_lock["config_id"]:
            raise RuntimeError("A cache row belongs to another config; migration refused.")
        if not isinstance(title, str) or not title:
            raise RuntimeError("A cache row has an invalid source title.")

    if len({row[0] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate source titles exist in the source cache; migration refused.")
    return old_lock, old_state, rows


def _copy_checkpoint_inputs(staging_dir: Path) -> None:
    old_lock, old_state, rows = _load_and_validate_old()
    models = {key: dict(value) for key, value in old_lock["models"].items()}
    for key, model in models.items():
        if key in pipeline.MODELS:
            model["template_version"] = pipeline.MODELS[key]["template_version"]
    config = pipeline._config_payload(models)
    config_id = pipeline.fingerprint(config)
    lock = {
        "created_at_utc": pipeline.utc_now(),
        "dataset_sha256": pipeline.DATASET_SHA256,
        "models": models,
        "config": config,
        "config_id": config_id,
        "backend_revision_note": old_lock.get("backend_revision_note", ""),
    }
    pipeline.atomic_json(staging_dir / "hybrid_environment_lock.json", lock)

    for name in (
        "hybrid_sample_manifest.json",
        "hybrid_sample_manifest.csv",
        "runtime_gemma4.json",
        "gemma_qa_recovery.jsonl",
        "gemma_qa_recovery_report.json",
    ):
        source = OLD_DIR / name
        if not source.is_file() and name.startswith("hybrid_sample_manifest"):
            raise FileNotFoundError(f"Required checkpoint input is missing: {name}")
        if source.is_file():
            shutil.copy2(source, staging_dir / name)

    new_state = {
        "config_id": config_id,
        "dataset_sha256": pipeline.DATASET_SHA256,
        "models": {key: {"stages": {}} for key in pipeline.MODELS},
        "created_at_utc": pipeline.utc_now(),
        "migrated_from": {
            "workdir": OLD_DIR.name,
            "config_id": old_lock["config_id"],
            "reason": (
                "Re-evaluate deterministic QA and source-anchored normalization; reuse cached outputs, "
                "preserve per-record lineage, and canary paired Gemma requests for future uncached titles."
            ),
        },
    }
    pipeline.atomic_json(staging_dir / "hybrid_run_state.json", new_state)

    new_cache = pipeline.Cache(
        staging_dir / "cache_gemma4.sqlite3",
        "gemma4",
        models["gemma4"]["revision"],
        config_id,
    )
    changed_outputs = 0
    for row in rows:
        (
            title, _model_key, _model_id, _revision, _config_id, output_text, status,
            truncated, input_tokens, output_tokens, attempts, elapsed_seconds,
            error_text, model_version, response_id, generation_prompt_version, api_requests,
        ) = row
        prompt_version = generation_prompt_version or pipeline.PROMPT_VERSION
        if status == "ok" and output_text:
            normalized = pipeline._normalize_gemma_measurements(output_text)
            normalized, source_repairs = pipeline._repair_gemma_specs_from_source(title, normalized)
            changed_outputs += normalized != output_text
            if source_repairs:
                prompt_version = pipeline.GEMMA_DETERMINISTIC_REPAIR_VERSION
            output_text = normalized
        new_cache.put(
            title,
            {
                "output_text": output_text,
                "status": status,
                "truncated": bool(truncated),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "attempts": attempts,
                "elapsed_seconds": elapsed_seconds,
                "error_text": error_text,
                "model_version": model_version,
                "response_id": response_id,
                "generation_prompt_version": prompt_version,
                "api_requests": api_requests,
            },
        )

    context = pipeline.prepare_pipeline(
        mode="smoke",
        model_key="gemma4",
        data_path=SOURCE,
        workdir=staging_dir,
    )
    pipeline._check_or_write_runtime_lock(context, "gemma4")
    cache = pipeline._cache(context, "gemma4")
    old_gemma_stages = old_state.get("models", {}).get("gemma4", {}).get("stages", {})
    recalculated: dict[str, Any] = {}
    for mode in ("smoke", "pilot", "validate"):
        titles = pipeline._stage_titles(mode, context["samples"], context["frame"])
        gate = pipeline._stage_gate(mode, "gemma4", titles, context["samples"], cache)
        prior = old_gemma_stages.get(mode, {})
        result = {
            "mode": mode,
            "model_key": "gemma4",
            "config_id": config_id,
            "requested": len(titles),
            "session_titles": len(titles),
            "generation": prior.get("generation", {"status": "reused_cached_generation"}),
            "started_at_utc": prior.get("started_at_utc", pipeline.utc_now()),
            "elapsed_seconds": prior.get("elapsed_seconds", 0),
            "semantic_verified": False,
            "qa_recalculated_from_cache": True,
            "migrated_from_config_id": old_lock["config_id"],
            **gate,
        }
        result["status"] = "passed" if result["passed"] else "quality_gate_failed"
        recalculated[mode] = result

    prior_full = old_gemma_stages.get("full", {})
    if prior_full:
        migrated_full = {
            "mode": "full",
            "model_key": "gemma4",
            "config_id": config_id,
            "status": prior_full.get("status", "checkpoint_saved_continue"),
            "passed": False,
            "semantic_verified": False,
            "migrated_from_config_id": old_lock["config_id"],
            "qa_recalculated_from_cache": True,
            "window_gates": prior_full.get("window_gates", []),
            "completed_unique_titles": sum(
                pipeline._complete(record)
                for record in cache.get_many(context["frame"]["title_zh"].astype(str).tolist()).values()
            ),
            "total_unique_titles": pipeline.EXPECTED_UNIQUE_TITLES,
        }
        migrated_full["remaining_unique_titles"] = (
            pipeline.EXPECTED_UNIQUE_TITLES - migrated_full["completed_unique_titles"]
        )
        prior_failed = prior_full.get("failed_window") or {}
        try:
            failed_start = int(prior_failed["start"])
            failed_end = int(prior_failed["end"])
        except (KeyError, TypeError, ValueError):
            failed_start = failed_end = -1
        all_titles = list(dict.fromkeys(context["frame"]["title_zh"].astype(str).tolist()))
        if prior_full.get("status") == "stopped_quality_gate" and 0 <= failed_start < failed_end <= len(all_titles):
            window = all_titles[failed_start:failed_end]
            refreshed = pipeline._full_window_gates(window, all_titles, cache)
            if refreshed and refreshed[0].get("passed"):
                migrated_full["status"] = "checkpoint_saved_continue"
                migrated_full["failed_window_rechecked"] = {
                    **refreshed[0],
                    "resolved_by_new_qa_and_normalization": True,
                }
            elif refreshed:
                migrated_full["status"] = "stopped_quality_gate"
                migrated_full["failed_window"] = refreshed[0]
            else:
                migrated_full["status"] = "stopped_quality_gate"
                migrated_full["failed_window"] = prior_failed
        elif prior_full.get("status") == "stopped_quality_gate":
            migrated_full["status"] = "stopped_quality_gate"
            migrated_full["failed_window"] = prior_failed
        recalculated["full"] = migrated_full

    context["state"]["models"]["gemma4"]["stages"] = recalculated
    context["state"]["migration_audit"] = {
        "from": OLD_DIR.name,
        "old_config_id": old_lock["config_id"],
        "new_config_id": config_id,
        "cache_rows_migrated": len(rows),
        "normalized_outputs_changed": changed_outputs,
        "migrated_at_utc": pipeline.utc_now(),
        "old_full_status": old_gemma_stages.get("full", {}).get("status"),
        "old_full_completed_unique_titles": old_gemma_stages.get("full", {}).get("completed_unique_titles"),
        "migrated_full_status": recalculated.get("full", {}).get("status"),
        "failed_window_recheck": recalculated.get("full", {}).get("failed_window_rechecked")
        or recalculated.get("full", {}).get("failed_window"),
    }
    pipeline.atomic_json(staging_dir / "hybrid_run_state.json", context["state"])

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row[6]] = status_counts.get(row[6], 0) + 1
    report = {
        "migration": f"{OLD_DIR.name}_to_{NEW_DIR.name}",
        "old_config_id": old_lock["config_id"],
        "new_config_id": config_id,
        "dataset_sha256": pipeline.DATASET_SHA256,
        "cache_rows_migrated": len(rows),
        "cache_status_counts": status_counts,
        "normalized_outputs_changed": changed_outputs,
        "qa_version": pipeline.QA_VERSION,
        "gemma_normalization_version": pipeline.GEMMA_NORMALIZATION_VERSION,
        "stage_gates": {
            key: {
                "passed": value["passed"],
                "completion_rate": value.get("completion_rate"),
                "technical_pass_rate": value.get("technical_pass_rate"),
                "representative_technical_pass_rate": value.get("representative_technical_pass_rate"),
                "challenge_technical_pass_rate": value.get("challenge_technical_pass_rate"),
            }
            for key, value in recalculated.items()
        },
        "full_resume_allowed": all(recalculated[key]["passed"] for key in ("smoke", "pilot", "validate")),
        "full_window_requires_qa_repair": recalculated.get("full", {}).get("status") == "stopped_quality_gate",
        "full_resume_from_cached_outputs": True,
        "old_checkpoint_modified": False,
    }
    pipeline.atomic_json(staging_dir / "checkpoint_migration_report.json", report)
    if not report["full_resume_allowed"]:
        raise RuntimeError(
            "At least one smoke/pilot/validate gate fails under the active QA; migration staged for diagnosis only."
        )
    return report


def main() -> int:
    global OLD_DIR, NEW_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create the target checkpoint after all checks pass; default is dry-run.")
    parser.add_argument("--from-dir", default=OLD_DIR.name, help="Existing checkpoint folder directly under the project root.")
    parser.add_argument("--to-dir", default=NEW_DIR.name, help="New checkpoint folder directly under the project root.")
    args = parser.parse_args()

    for label, value in (("--from-dir", args.from_dir), ("--to-dir", args.to_dir)):
        candidate = Path(value)
        if candidate.name != value or value in {".", ".."}:
            raise ValueError(f"{label} must be a folder name directly under {ROOT}.")
    OLD_DIR = ROOT / args.from_dir
    NEW_DIR = ROOT / args.to_dir

    if args.execute and NEW_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {NEW_DIR}")
    if not args.execute and NEW_DIR.exists():
        raise FileExistsError(f"Dry-run expects the destination not to exist: {NEW_DIR}")

    with tempfile.TemporaryDirectory(prefix=".title-checkpoint-migration-", dir=ROOT) as temp:
        staging_dir = Path(temp) / NEW_DIR.name
        staging_dir.mkdir()
        report = _copy_checkpoint_inputs(staging_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.execute:
            staging_dir.rename(NEW_DIR)
            print(f"Created migrated checkpoint: {NEW_DIR}")
        else:
            print("Dry run only: no checkpoint folder was created and the source checkpoint remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
