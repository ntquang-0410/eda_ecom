"""Print a privacy-safe, read-only summary of a title-translation checkpoint."""

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
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKDIR = ROOT / "title_translation_cado_free_hybrid_run_v15"
PROGRESS_LINE = re.compile(
    r"^\[(?P<model>[^]]+)\] uncached (?P<done>\d+)/(?P<requested>\d+); "
    r"new complete=(?P<completed>\d+); errors=(?P<errors>\d+); "
    r"api_requests=(?P<attempts>\d+); paired_requests=(?P<paired>\d+); "
    r"fallback=(?P<fallback>\d+)"
)
HTTP_RETRY = re.compile(r"transient request error on attempt \d+ \(HTTP (\d{3})\)")


def _read_run_state(workdir: Path, model_key: str) -> dict[str, Any]:
    state_path = workdir / "hybrid_run_state.json"
    if not state_path.is_file():
        return {"run_state_file": "missing"}

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"run_state_file": "unreadable"}

    model = (state.get("models") or {}).get(model_key) or {}
    stages = model.get("stages") or {}
    full = stages.get("full") or {}
    # Only return aggregate fields; the run-state file may gain richer metadata later.
    field_names = {
        "status": "last_saved_stage_status",
        "config_id": "config_id",
        "completed_unique_titles": "run_state_completed_unique_titles",
        "remaining_unique_titles": "run_state_remaining_unique_titles",
        "total_unique_titles": "total_unique_titles",
        "elapsed_seconds": "run_state_elapsed_seconds",
        "started_at_utc": "run_state_started_at_utc",
        "checkpoint_preserved": "checkpoint_preserved",
    }
    summary = {
        output_name: full[input_name]
        for input_name, output_name in field_names.items()
        if input_name in full
    }
    summary["run_state_file"] = "ok"
    return summary


def _summarize_log(workdir: Path, model_key: str, now_utc: datetime) -> dict[str, Any]:
    log_paths = list(workdir.glob(f"{model_key}_background_*.stdout.log"))
    if model_key == "gemma4":
        log_paths.extend(workdir.glob("gemma_background_*.stdout.log"))
    logs = sorted(log_paths, key=lambda path: path.stat().st_mtime)
    if not logs:
        return {"log_file": "missing"}

    log_path = logs[-1]
    try:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 512_000))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return {"log_file": "unreadable"}

    recent_lines = lines[-1000:]
    latest_progress = None
    transient_http_errors: dict[str, int] = {}
    for line in recent_lines:
        match = PROGRESS_LINE.search(line)
        if match and match.group("model") == model_key:
            latest_progress = {
                key: int(match.group(key))
                for key in ("done", "requested", "completed", "errors", "attempts", "paired", "fallback")
            }
        error_match = HTTP_RETRY.search(line)
        if error_match:
            key = f"http_{error_match.group(1)}"
            transient_http_errors[key] = transient_http_errors.get(key, 0) + 1

    log_age = max(0.0, now_utc.timestamp() - log_path.stat().st_mtime)
    return {
        "log_file": log_path.name,
        "log_age_seconds": round(log_age, 1),
        "log_lines_scanned": len(recent_lines),
        "latest_logged_batch_progress": latest_progress,
        "http_retry_counts_in_log_tail": transient_http_errors,
    }


def build_status(
    workdir: Path,
    model_key: str = "gemma4",
    window_minutes: int = 15,
    now: datetime | None = None,
) -> dict[str, Any]:
    if window_minutes < 1:
        raise ValueError("window_minutes must be positive")

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)

    database = workdir / f"cache_{model_key}.sqlite3"
    if not database.is_file():
        return {
            "model_key": model_key,
            "workdir_exists": workdir.is_dir(),
            "database": "missing",
            **_read_run_state(workdir, model_key),
        }

    # SQLite URI read-only mode prevents this diagnostic tool from changing a live cache.
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM translations GROUP BY status"
            ).fetchall()
        )
        total_records = int(
            connection.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
        )
        completed = int(
            connection.execute(
                "SELECT COUNT(*) FROM translations "
                "WHERE status='ok' AND truncated=0"
            ).fetchone()[0]
        )
        truncated = int(
            connection.execute(
                "SELECT COUNT(*) FROM translations WHERE truncated=1"
            ).fetchone()[0]
        )
        latest = connection.execute(
            "SELECT MAX(updated_at_utc) FROM translations"
        ).fetchone()[0]
        cutoff = (now_utc - timedelta(minutes=window_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        recent_completed = int(
            connection.execute(
                "SELECT COUNT(*) FROM translations "
                "WHERE status='ok' AND truncated=0 AND updated_at_utc >= ?",
                (cutoff,),
            ).fetchone()[0]
        )

    latest_dt = _parse_utc(latest)
    age_seconds = (
        max(0.0, (now_utc - latest_dt).total_seconds()) if latest_dt else None
    )
    state_summary = _read_run_state(workdir, model_key)
    log_summary = _summarize_log(workdir, model_key, now_utc)
    expected = state_summary.get("total_unique_titles")
    remaining = max(0, int(expected) - completed) if expected is not None else None

    return {
        **state_summary,
        **log_summary,
        "model_key": model_key,
        "database": "ok",
        "workdir_exists": workdir.is_dir(),
        "cache_records": total_records,
        "completed_unique_titles": completed,
        "completed_ok": completed,
        "total_unique_titles": int(expected) if expected is not None else None,
        "remaining_unique_titles": remaining,
        "status_counts": counts,
        "truncated_records": truncated,
        "last_record_updated_at_utc": latest,
        "last_record_age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "recent_window_minutes": window_minutes,
        "recent_completed_titles": recent_completed,
        "recent_rate_titles_per_hour": round(
            recent_completed * 60.0 / window_minutes, 1
        ),
        "activity_hint": (
            "recent_writes" if age_seconds is not None and age_seconds <= 300
            else "no_recent_writes"
        ),
    }


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only, privacy-safe checkpoint status for title translation."
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--model-key", default="gemma4")
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--json", action="store_true", help="emit compact machine-readable JSON")
    args = parser.parse_args()

    result = build_status(args.workdir, args.model_key, args.window_minutes)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("database") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
