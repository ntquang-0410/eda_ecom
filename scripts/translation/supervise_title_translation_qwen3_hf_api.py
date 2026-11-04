"""Run Qwen API translation under a live, checkpoint-aware code supervisor.

The supervisor is intended to be launched by Codex in a persistent terminal
session. It runs tests before each worker launch, waits when a run checkpoints
with unresolved titles, and automatically restarts at a batch boundary after
the worker or QA code changes. HF_TOKEN is inherited from the environment and
is never printed or copied into status files.
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
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUPERVISOR = Path(__file__).resolve()
RUNNER = ROOT / "run_title_translation_qwen3_hf_api.py"
QUALITY_SOURCE = Path(__import__('title_translation_pipeline').__file__).resolve()
WATCHED_SOURCES = (SUPERVISOR, RUNNER, QUALITY_SOURCE)


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in WATCHED_SOURCES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_status(workdir: Path) -> dict:
    path = workdir / "qwen_api_live_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def validate_code() -> bool:
    print("[supervisor] compile + unit tests before starting/reloading", flush=True)
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SUPERVISOR), str(RUNNER), str(QUALITY_SOURCE)],
        cwd=ROOT,
        check=False,
    )
    if compile_result.returncode:
        print("[supervisor] syntax check failed; waiting for a code fix", flush=True)
        return False
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/titles"],
        cwd=_repo_root,
        check=False,
    )
    if test_result.returncode:
        print("[supervisor] tests failed; waiting for a code fix", flush=True)
        return False
    return True


def build_worker_command(args: argparse.Namespace, stage: str | None = None) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--source", str(args.source),
        "--workdir", str(args.workdir),
        "--stage", stage or args.stage,
        "--workers", str(args.workers),
        "--provider-policy", args.provider_policy,
        "--max-cost-usd", str(args.max_cost_usd),
    ]
    if args.mirror_dir:
        command.extend(["--mirror-dir", str(args.mirror_dir)])
    if args.allow_paid_inference:
        command.append("--allow-paid-inference")
    if args.dry_run:
        command.append("--dry-run")
    return command


def wait_for_source_change(previous_digest: str, poll_seconds: float) -> str:
    print(
        "[supervisor] worker checkpointed; inspect status/cache, patch code, "
        "and this process will test then resume automatically",
        flush=True,
    )
    while True:
        time.sleep(poll_seconds)
        current = source_digest()
        if current != previous_digest:
            return current


def supervise(args: argparse.Namespace) -> int:
    requested_stage = args.stage
    active_stage = "smoke" if requested_stage == "full" else requested_stage
    last_validated_digest = source_digest()
    if not validate_code():
        last_validated_digest = wait_for_source_change(last_validated_digest, args.poll_seconds)
        while not validate_code():
            last_validated_digest = wait_for_source_change(last_validated_digest, args.poll_seconds)

    if args.dry_run:
        command = build_worker_command(args, active_stage)
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    if not args.allow_paid_inference:
        print("Refusing to start: add --allow-paid-inference only after explicitly approving provider charges.", file=sys.stderr)
        return 2

    print("[supervisor] direct local worker; no local GPU; token remains in HF_TOKEN environment", flush=True)
    while True:
        run_digest = source_digest()
        command = build_worker_command(args, active_stage)
        process = subprocess.Popen(command, cwd=ROOT)
        return_code = process.wait()
        state = read_status(args.workdir)
        run_state = state.get("state")
        print(
            f"[supervisor] worker exit={return_code}; state={run_state or 'unknown'}; "
            f"remaining={state.get('remaining', 'unknown')}",
            flush=True,
        )

        if run_state == "all_titles_complete":
            return 0
        if run_state == "smoke_complete":
            if requested_stage == "full" and active_stage == "smoke":
                active_stage = "full"
                print("[supervisor] smoke gate complete; starting full from its checkpoint", flush=True)
                continue
            print("[supervisor] smoke completed; restart supervisor with --stage full when ready", flush=True)
            return 0
        if run_state == "budget_cap_reached":
            print("[supervisor] configured spend limit reached; stopped without launching more requests", flush=True)
            return 3
        if run_state == "error" and state.get("phase") in {"provider_auth", "missing_token"}:
            print("[supervisor] provider authentication/billing error; fix credentials or account before resume", flush=True)
            return 4

        observed_digest = source_digest()
        if run_state != "reload_requested" and observed_digest == run_digest:
            observed_digest = wait_for_source_change(observed_digest, args.poll_seconds)
        else:
            print("[supervisor] code changed at a safe checkpoint; validating before resume", flush=True)

        while not validate_code():
            observed_digest = wait_for_source_change(observed_digest, args.poll_seconds)
        if requested_stage == "full":
            active_stage = "smoke"
        last_validated_digest = observed_digest
        print(
            f"[supervisor] validation passed; resuming from checkpoint; "
            f"cumulative cost is controlled by the same SQLite ledger ({last_validated_digest[:12]})",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "bilingual_zh_vi_cado.parquet")
    parser.add_argument("--workdir", type=Path, default=ROOT / "title_translation_qwen3_hf_api_run")
    parser.add_argument("--mirror-dir", type=Path, default=None,
                        help="Optional Google Drive for Desktop sync directory.")
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--provider-policy", choices=("deepinfra", "fastest", "cheapest"), default="deepinfra")
    parser.add_argument("--max-cost-usd", type=float, required=True,
                        help="Cumulative local stop threshold; set an account-side provider limit too.")
    parser.add_argument("--allow-paid-inference", action="store_true",
                        help="Explicitly authorize paid HF Inference Provider requests.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; no provider calls.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.poll_seconds < 0.25:
        parser.error("--poll-seconds must be at least 0.25")
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
