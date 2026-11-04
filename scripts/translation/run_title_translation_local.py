"""Run Gemma locally through Google AI Studio, Qwen locally on a GPU, and upload later.

Gemma-only commands need no local GPU. Qwen3.5-9B is downloaded and quantized on
the local CUDA GPU by the legacy combined command. API keys are requested with
hidden terminal input only when launched directly; the PowerShell wrapper loads
the Windows-user DPAPI secret for unattended local runs without storing it in code.
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

import json
import argparse
import os
import re
import shutil
import sys
from getpass import getpass
from pathlib import Path
from typing import Any

import title_translation_hybrid as pipeline


ROOT = Path(os.environ.get("ECOM_WORKSPACE") or _repo_root)
SOURCE = Path(os.environ.get("ECOM_SOURCE") or ROOT / "data/inputs/bilingual_zh_vi_cado.parquet")
WORKDIR = Path(os.environ.get("ECOM_WORKDIR") or ROOT / "outputs/title_translation_cado_free_hybrid_run_v15")
MODELS = ("gemma4", "qwen35")
# Keep the previously tested request cadence. Faster throughput is tested by
# fitting two independently indexed titles into each paced request.
GEMMA_MIN_INTERVAL_SECONDS = max(
    5.0, float(os.environ.get("GEMMA_MIN_REQUEST_INTERVAL_SECONDS", "6.0"))
)
FULL_TITLES_PER_SESSION = 200


def _canary_matches_runtime(canary: dict[str, Any], config_id: str) -> bool:
    try:
        same_interval = abs(float(canary.get("min_interval_seconds")) - GEMMA_MIN_INTERVAL_SECONDS) < 1e-9
    except (TypeError, ValueError):
        same_interval = False
    return bool(canary.get("config_id") == config_id and canary.get("passed") and same_interval)


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    generation = result.get("generation") or {}
    keys = (
        "mode",
        "model_key",
        "status",
        "passed",
        "requested",
        "session_titles",
        "completed_unique_titles",
        "total_unique_titles",
        "remaining_unique_titles",
        "completion_rate",
        "technical_pass_rate",
        "representative_technical_pass_rate",
        "challenge_technical_pass_rate",
        "input_tokens",
        "output_tokens",
        "output_path",
        "output_sha256",
    )
    summary = {key: result[key] for key in keys if key in result}
    summary["api_status"] = generation.get("status")
    summary["api_completed_this_session"] = generation.get("completed_this_session")
    summary["api_attempts_this_session"] = generation.get("attempts_this_session")
    summary["api_input_tokens_this_session"] = generation.get("input_tokens_this_session")
    summary["api_output_tokens_this_session"] = generation.get("output_tokens_this_session")
    return summary


def _safe_exception_message(exc: Exception, *secrets: str) -> str:
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED_HF_TOKEN]", message)


def _get_google_api_key() -> str:
    """Prefer a locally supplied secret; prompt hidden only when none is configured."""
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or getpass(
        "Google AI Studio GEMINI_API_KEY (input hidden): "
    )).strip()


def _local_qwen_gpu_ready() -> bool:
    try:
        import torch

        return bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory >= 14 * 1024**3
        )
    except (ImportError, RuntimeError, IndexError):
        return False


def _context(mode: str, model_key: str) -> dict[str, Any]:
    return pipeline.prepare_pipeline(
        mode=mode,
        model_key=model_key,
        data_path=SOURCE,
        workdir=WORKDIR,
        hf_token=None,
    )


def _run_gated_stage(mode: str, model_key: str, google_api_key: str) -> bool:
    context = _context(mode, model_key)
    result = pipeline.run_stage(
        context,
        google_api_key=google_api_key,
        batch_size=1,
        full_titles_per_session=FULL_TITLES_PER_SESSION,
        gemma_min_interval_seconds=GEMMA_MIN_INTERVAL_SECONDS,
    )
    print(json.dumps(_summary(result), ensure_ascii=False), flush=True)
    return result.get("status") == "passed" and bool(result.get("passed"))


def _run_full(
    model_key: str,
    google_api_key: str,
    *,
    defer_full_quality_review: bool = False,
) -> str:
    context = _context("full", model_key)
    while True:
        prior_full = (
            context.get("state", {})
            .get("models", {})
            .get(model_key, {})
            .get("stages", {})
            .get("full", {})
        )
        if (
            not defer_full_quality_review
            and prior_full.get("status") == "stopped_quality_gate"
            and not pipeline._full_gate_stop_is_incomplete_only(prior_full)
        ):
            recovery = pipeline._run_gemma_qa_repair(
                context,
                google_api_key,
                prior_full.get("failed_window") or {},
                GEMMA_MIN_INTERVAL_SECONDS,
            )
            print(json.dumps({"qa_recovery": recovery}, ensure_ascii=False), flush=True)
            if recovery.get("status") == "quota_exhausted_checkpoint_saved":
                return "quota_exhausted_checkpoint_saved"
            context = _context("full", model_key)
            if not pipeline._full_gate_stop_is_resolved(prior_full, context, model_key):
                print(
                    "Gemma QA recovery did not bring the stopped window inside its gate; "
                    "checkpoint preserved and no later titles will be processed.",
                    flush=True,
                )
                return "stopped_quality_gate"
        canary = context.get("state", {}).get("gemma_batch_canary", {})
        if not _canary_matches_runtime(canary, context["lock"]["config_id"]):
            canary = pipeline.run_gemma_batch_canary(
                context,
                google_api_key,
                min_interval_seconds=GEMMA_MIN_INTERVAL_SECONDS,
                title_limit=pipeline.GEMMA_BATCH_CANARY_TITLES,
            )
            print(
                json.dumps(
                    {
                        "batch_canary": {
                            "status": canary.get("status"),
                            "passed": canary.get("passed"),
                            "selected_batch_size": canary.get("selected_batch_size"),
                            "canary_metrics": canary.get("canary_metrics"),
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if canary.get("status") == "quota_exhausted_checkpoint_saved":
                return "quota_exhausted_checkpoint_saved"
            if not canary.get("passed"):
                return "stopped_quality_gate"
            context = _context("full", model_key)
        selected_batch_size = int(canary.get("selected_batch_size") or 1)
        result = pipeline.run_stage(
            context,
            google_api_key=google_api_key,
            batch_size=1,
            gemma_batch_size=selected_batch_size,
            full_titles_per_session=FULL_TITLES_PER_SESSION,
            gemma_min_interval_seconds=GEMMA_MIN_INTERVAL_SECONDS,
            defer_full_quality_review=defer_full_quality_review,
        )
        print(json.dumps(_summary(result), ensure_ascii=False), flush=True)
        status = result.get("status")
        if status == "all_titles_complete":
            return status
        if status == "checkpoint_saved_continue":
            continue
        if status == "quota_exhausted_checkpoint_saved":
            return status
        if status == "stopped_quality_gate":
            continue
        raise RuntimeError(f"Unexpected full-stage status for {model_key}: {status!r}")


def _run_full_session(model_key: str, google_api_key: str) -> str:
    """Process one bounded full-stage session so API usage stays user-controlled."""
    context = _context("full", model_key)
    result = pipeline.run_stage(
        context,
        google_api_key=google_api_key,
        batch_size=1,
        full_titles_per_session=FULL_TITLES_PER_SESSION,
        gemma_min_interval_seconds=GEMMA_MIN_INTERVAL_SECONDS,
    )
    print(json.dumps(_summary(result), ensure_ascii=False), flush=True)
    return str(result.get("status"))


def _run_gemma_command(command: str, *, confirmed: bool = False) -> int:
    """Run exactly one Gemma stage locally; all API usage is explicit and resumable."""
    stage_by_command = {
        "gemma-smoke": "smoke",
        "gemma-pilot": "pilot",
        "gemma-validate": "validate",
        "gemma-full": "full",
    }
    mode = stage_by_command[command]
    if not SOURCE.exists():
        print(f"Source Parquet is missing: {SOURCE}")
        return 2

    google_api_key = _get_google_api_key()
    if not google_api_key:
        print("No API key entered. No API calls were started.")
        return 2

    print(f"Gemma-only local run: stage={mode}; no local GPU or Qwen model is used.", flush=True)
    print(f"Checkpoint directory: {WORKDIR}", flush=True)
    print("Results remain local; no Hugging Face upload will occur.", flush=True)
    if mode == "full":
        print(
            "Full-run rolling QA is deferred as requested; per-title technical QA flags remain in the cache/output for review after translation.",
            flush=True,
        )
    if confirmed:
        print("Gemma full start explicitly confirmed for unattended execution.", flush=True)
    else:
        confirmation = "START FULL" if mode == "full" else "START"
        if input(f"Type {confirmation} to run Gemma {mode}: ").strip() != confirmation:
            print("Cancelled before any API request.", flush=True)
            return 0

    try:
        if mode == "full":
            status = _run_full(
                "gemma4",
                google_api_key,
                defer_full_quality_review=True,
            )
            if status == "quota_exhausted_checkpoint_saved":
                print("API quota exhausted; completed work is checkpointed. Resume with gemma-full when quota is available.", flush=True)
                return 4
            if status != "all_titles_complete":
                print(f"Gemma full stopped with status={status}; checkpoint preserved.", flush=True)
                return 5
            output_path = WORKDIR / "bilingual_zh_vi_cado_title_gemma4.parquet"
            print(f"Gemma Parquet complete (local only): {output_path}", flush=True)
            return 0

        passed = _run_gated_stage(mode, "gemma4", google_api_key)
        if not passed:
            print(
                f"Gemma {mode} did not pass its gate. Checkpoint is preserved; do not start a later stage.",
                flush=True,
            )
            return 3
        next_command = {
            "smoke": "gemma-pilot",
            "pilot": "gemma-validate",
            "validate": "gemma-full",
        }[mode]
        print(f"Gemma {mode} passed. Next command when ready: .\\run_title_translation_local.ps1 {next_command}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Completed API responses are checkpointed; rerun the same stage to resume.", flush=True)
        return 130
    except Exception as exc:
        message = _safe_exception_message(exc, google_api_key)
        print(f"\nGemma {mode} stopped: {type(exc).__name__}: {message}", flush=True)
        print(f"Checkpoint directory (if initialized): {WORKDIR}", flush=True)
        return 1


def _pack_checkpoint() -> int:
    if not WORKDIR.is_dir():
        print(f"No Gemma checkpoint folder exists yet: {WORKDIR}")
        return 2
    archive_base = ROOT / "title_translation_cado_free_hybrid_checkpoint"
    archive_path = archive_base.with_suffix(".zip")
    suffix = 2
    while archive_path.exists():
        archive_base = ROOT / f"title_translation_cado_free_hybrid_checkpoint_{suffix}"
        archive_path = archive_base.with_suffix(".zip")
        suffix += 1
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=WORKDIR.parent, base_dir=WORKDIR.name))
    print(f"Checkpoint archive created: {archive}")
    print("Upload this ZIP to Google Drive/Colab to continue Qwen in the same run directory.")
    return 0


def _print_local_outputs() -> None:
    state_path = WORKDIR / "hybrid_run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    outputs = []
    for model_key in MODELS:
        stage = state.get("models", {}).get(model_key, {}).get("stages", {}).get("full", {})
        output_path = WORKDIR / f"bilingual_zh_vi_cado_title_{model_key}.parquet"
        if stage.get("status") != "all_titles_complete" or not output_path.is_file():
            raise RuntimeError(f"Local output for {model_key} is not complete: {output_path}")
        actual_hash = pipeline.quality.sha256_file(output_path)
        if stage.get("output_sha256") != actual_hash:
            raise RuntimeError(f"Local output checksum does not match run state: {output_path}")
        outputs.append(
            {
                "model_key": model_key,
                "path": str(output_path),
                "sha256": actual_hash,
                "rows": pipeline.EXPECTED_ROWS,
            }
        )
    print("\nLocal Parquet files (not uploaded):", flush=True)
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


def _upload_local_outputs() -> int:
    if not (WORKDIR / "hybrid_run_state.json").is_file():
        print(f"No local run state found: {WORKDIR}", flush=True)
        return 2
    try:
        _print_local_outputs()
    except Exception as exc:
        print(f"Upload not ready: {type(exc).__name__}: {exc}", flush=True)
        return 2

    write_token = getpass("HF dataset-write token (write permission; input hidden): ").strip()
    if not write_token:
        print("No token entered. Nothing was uploaded.", flush=True)
        return 2
    if input("Type UPLOAD to publish both local Parquet files to HF: ").strip() != "UPLOAD":
        print("Cancelled. Nothing was uploaded.", flush=True)
        return 0

    try:
        manifest = pipeline.upload_outputs_to_hf(
            WORKDIR,
            hf_token=write_token,
            source_path=SOURCE,
        )
        print("\nHF upload and checksum read-back verification:", flush=True)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        message = _safe_exception_message(exc, write_token)
        print(f"\nUpload stopped: {type(exc).__name__}: {message}", flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemma through Google AI Studio and Qwen locally on a sufficiently large CUDA GPU; "
            "upload completed local Parquets separately. For Colab/Kaggle, use the provided notebook."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "gemma-smoke", "gemma-pilot", "gemma-validate", "gemma-full", "gemma-pack", "upload"),
        default="run",
        help="run Gemma one stage at a time, run the legacy local-GPU pair, or upload completed Parquets",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm only gemma-full for unattended execution; requires the API key from the secure launcher",
    )
    args = parser.parse_args(argv)

    if args.yes and args.command != "gemma-full":
        parser.error("--yes is supported only for gemma-full")
    if args.yes and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Unattended gemma-full requires the DPAPI-backed PowerShell launcher to supply the API key.")
        return 2
    if not sys.stdin.isatty() and not args.yes:
        print("Run this script from an interactive PowerShell terminal so tokens can be entered hidden.")
        return 2

    if args.command == "upload":
        return _upload_local_outputs()

    if args.command == "gemma-pack":
        return _pack_checkpoint()

    if args.command.startswith("gemma-"):
        return _run_gemma_command(args.command, confirmed=args.yes)

    if not SOURCE.exists():
        print(f"Source Parquet is missing: {SOURCE}")
        return 2

    if not _local_qwen_gpu_ready():
        print(
            "Qwen3.5-9B local mode needs a CUDA GPU with at least 14 GiB VRAM. "
            "This computer's runtime does not meet the preflight (or PyTorch/CUDA is not installed). "
            "Use title_translation_free_hybrid.ipynb on Colab/Kaggle GPU instead; no API call was started."
        )
        return 2

    google_api_key = _get_google_api_key()
    if not google_api_key:
        print("No API key entered. No API calls were started.")
        return 2

    if (WORKDIR / "hybrid_run_state.json").exists():
        print(f"Resuming existing checkpoint: {WORKDIR}", flush=True)
    else:
        print(f"Starting a new run. Checkpoints: {WORKDIR}", flush=True)
    print("Gemma uses Google AI Studio API; Qwen3.5-9B weights will download to the CUDA runtime.", flush=True)
    print("Google free-tier quotas apply; local Qwen inference does not use HF Inference credits.", flush=True)
    print("This command only saves results locally. It will not upload anything to HF.", flush=True)
    if input("Type START to run smoke → pilot → validate → full and save local Parquets: ").strip() != "START":
        print("Cancelled before any API request.", flush=True)
        return 0

    try:
        for mode in ("smoke", "pilot", "validate"):
            for model_key in MODELS:
                print(f"\n=== {mode.upper()} / {model_key} ===", flush=True)
                if not _run_gated_stage(mode, model_key, google_api_key):
                    print(
                        f"Stopped at {mode}/{model_key}: gate did not pass or provider quota was exhausted. "
                        "Completed responses remain checkpointed; rerun after fixing the cause. No upload was made.",
                        flush=True,
                    )
                    return 3

        for model_key in MODELS:
            print(f"\n=== FULL / {model_key} ===", flush=True)
            status = _run_full(model_key, google_api_key)
            if status == "quota_exhausted_checkpoint_saved":
                print("Provider quota exhausted. Checkpoint is preserved; rerun this script after quota resets.", flush=True)
                return 4
            if status != "all_titles_complete":
                print(f"Full stage stopped with status={status}; checkpoint preserved; no upload was made.", flush=True)
                return 5

        _print_local_outputs()
        print("When you are ready, run: .\\run_title_translation_local.ps1 upload", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed API responses are checkpointed; rerun to resume.", flush=True)
        return 130
    except Exception as exc:
        message = _safe_exception_message(exc, google_api_key)
        print(f"\nRun stopped: {type(exc).__name__}: {message}", flush=True)
        print(f"Checkpoint directory (if initialized): {WORKDIR}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
