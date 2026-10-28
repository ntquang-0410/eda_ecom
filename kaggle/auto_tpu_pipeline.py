#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrator TPU tự động cho chuỗi Plan B.

Luồng:
  1. Chờ kernel BT 3 model (0.6B/1.7B/4B) kết thúc -> tải output.
  2. Push kernel spike sharding -> chờ -> tải report.
  3. Nếu spike pass (n_dev >= 2, validation 3/3, 8B generate được)
     -> push kernel BT 8B sharded -> chờ -> tải slices.
  4. Ghi summary JSON.

Có state file để resume nếu tiến trình bị dừng giữa chừng:
  /tmp/opencode/auto_pipeline_state.json

Chạy:  python /tmp/opencode/auto_tpu_pipeline.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/content/eda_ecom")
LOG = Path("/tmp/opencode/auto_pipeline.log")
STATE = Path("/tmp/opencode/auto_pipeline_state.json")
SUMMARY = Path("/tmp/opencode/auto_pipeline_summary.json")

KERNEL_3MODEL = "ntquang0410/tiki-plan-b-bt-tpu-v5e-8"
KERNEL_SPIKE = "ntquang0410/tiki-plan-b-tpu-shard-spike"
KERNEL_8B = "ntquang0410/tiki-plan-b-bt-tpu-8b"

DIR_3MODEL = "kaggle/tpu_bt"
DIR_SPIKE = "kaggle/tpu_shard"
DIR_8B = "kaggle/tpu_bt_8b"

OUT_3MODEL = Path("/tmp/opencode/tpu_bt_output")
OUT_SPIKE = Path("/tmp/opencode/shard_spike")
OUT_8B = Path("/tmp/opencode/tpu_8b")

TERMINAL = ("COMPLETE", "ERROR", "CANCEL", "TIMEOUT")
SLOT_BUSY = "Maximum batch"

# Giới hạn chờ
WAIT_3MODEL_HOURS = 36
WAIT_SPIKE_HOURS = 12
WAIT_8B_HOURS = 14
PUSH_RETRY_HOURS = 10
POLL_SECONDS = 60


# ---------------------------------------------------------------- utils
def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    line = f"[{ts()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def kaggle(*args: str, timeout: int = 600) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["kaggle", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as error:  # noqa: BLE001
        return 1, f"{type(error).__name__}: {error}"


def kernel_status(kernel: str) -> str:
    _, output = kaggle("kernels", "status", kernel)
    match = re.search(r'has status "([^"]+)"', output)
    if match:
        return match.group(1)
    if "not been shared" in output or "Permission" in output or "not found" in output:
        return "MISSING"
    return "UNKNOWN"


def is_terminal(status: str) -> bool:
    return any(marker in status for marker in TERMINAL)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"stage": "wait_3model", "history": []}


def save_state(state: dict) -> None:
    state["updated_at"] = ts()
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def set_stage(state: dict, stage: str, **extra) -> None:
    state["stage"] = stage
    state["history"].append({"stage": stage, "at": ts(), **extra})
    save_state(state)
    log(f"STAGE -> {stage} {extra if extra else ''}")


def wait_kernel(kernel: str, max_hours: float, label: str) -> str:
    deadline = time.time() + max_hours * 3600
    last = None
    while time.time() < deadline:
        status = kernel_status(kernel)
        if status != last:
            log(f"{label}: {status}")
            last = status
        if is_terminal(status):
            return status
        time.sleep(POLL_SECONDS)
    log(f"{label}: hết thời gian chờ {max_hours}h")
    return "WATCH_TIMEOUT"


def download_outputs(kernel: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    code, output = kaggle("kernels", "output", kernel, "-p", str(dest), timeout=3600)
    ok = code == 0
    log(f"download {kernel} -> {dest} ok={ok}")
    if not ok:
        log(output.strip()[:1500])
    return ok


def push_with_retry(kernel_dir: str, label: str, max_hours: float = PUSH_RETRY_HOURS) -> tuple[bool, str]:
    """Push kernel, retry khi hết slot TPU."""
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        code, output = kaggle("kernels", "push", "-p", str(ROOT / kernel_dir), timeout=900)
        if "successfully pushed" in output:
            version = re.search(r"Kernel version (\d+) successfully pushed", output)
            log(f"push {label} OK version={version.group(1) if version else '?'}")
            return True, output
        if SLOT_BUSY in output:
            log(f"push {label}: hết slot TPU, chờ 3 phút")
            time.sleep(180)
            continue
        log(f"push {label} FAILED: {output.strip()[:800]}")
        time.sleep(300)
    return False, "push timeout"


def find_report(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def run_summary(text: str) -> dict:
    """Đọc summary JSON của kernel 8B; fallback: parse log."""
    info = {}
    for line in text.splitlines():
        if line.startswith("coverage") or "coverage" in line:
            info.setdefault("raw", []).append(line.strip())
    return info


# ---------------------------------------------------------------- stages
def stage_3model(state: dict) -> dict:
    status = wait_kernel(KERNEL_3MODEL, WAIT_3MODEL_HOURS, "3model")
    state["status_3model"] = status
    download_outputs(KERNEL_3MODEL, OUT_3MODEL)

    coverage = {}
    for state_file in OUT_3MODEL.rglob("run_state.json"):
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        coverage[payload.get("model_key", state_file.parent.name)] = {
            "n_done": payload.get("n_done"),
            "n_total": payload.get("n_total"),
        }
    state["coverage_3model"] = coverage
    log(f"3model coverage: {json.dumps(coverage, ensure_ascii=False)}")

    incomplete = []
    for model_key, payload in coverage.items():
        done = payload.get("n_done") or 0
        total = payload.get("n_total") or 0
        if total and done < total:
            incomplete.append(f"{model_key}:{done}/{total}")
    state["incomplete_3model"] = incomplete
    if incomplete:
        log(f"3model INCOMPLETE -> cần re-run: {', '.join(incomplete)}")
    if not coverage:
        log("3model: không đọc được run_state.json, coverage chưa xác định")

    set_stage(state, "spike_push", status_3model=status, coverage=coverage, incomplete=incomplete)
    return state


def stage_spike(state: dict) -> dict:
    ok, _ = push_with_retry(DIR_SPIKE, "spike")
    if not ok:
        set_stage(state, "spike_push_failed")
        state["result"] = "spike_push_failed"
        return state
    set_stage(state, "spike_run")

    status = wait_kernel(KERNEL_SPIKE, WAIT_SPIKE_HOURS, "spike")
    state["status_spike"] = status
    download_outputs(KERNEL_SPIKE, OUT_SPIKE)

    report_path = find_report(OUT_SPIKE, "shard_spike_report.json")
    if report_path is None:
        log("spike: không thấy shard_spike_report.json")
        set_stage(state, "spike_no_report", status=status)
        state["result"] = "spike_no_report"
        return state

    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = report.get("validation") or {}
    eightb = report.get("eightb") or {}
    verdict = {
        "n_dev": report.get("n_dev"),
        "spmd_api": report.get("spmd_api"),
        "validation_match": validation.get("match"),
        "validation_error": validation.get("error"),
        "eightb_ok": bool(eightb.get("outputs")),
        "eightb_error": eightb.get("error"),
        "eightb_tokens_per_sec": eightb.get("tokens_per_sec"),
        "eightb_device_memory": (eightb.get("stats") or {}).get("device_memory"),
    }
    state["spike"] = verdict
    log(f"spike verdict: {json.dumps(verdict, ensure_ascii=False)}")

    passed = (
        isinstance(verdict["n_dev"], int)
        and verdict["n_dev"] >= 2
        and verdict["validation_match"] == "3/3"
        and verdict["eightb_ok"]
    )
    state["spike_passed"] = passed
    if passed:
        set_stage(state, "8b_push", spike=verdict)
    else:
        set_stage(state, "spike_failed", spike=verdict)
        state["result"] = "spike_failed"
    return state


def stage_8b(state: dict) -> dict:
    ok, _ = push_with_retry(DIR_8B, "8b")
    if not ok:
        set_stage(state, "8b_push_failed")
        state["result"] = "8b_push_failed"
        return state
    set_stage(state, "8b_run")

    status = wait_kernel(KERNEL_8B, WAIT_8B_HOURS, "8b")
    state["status_8b"] = status
    download_outputs(KERNEL_8B, OUT_8B)

    summary_path = find_report(OUT_8B, "plan_b_qwen3_8b_summary.json")
    if summary_path is not None:
        state["summary_8b"] = json.loads(summary_path.read_text(encoding="utf-8"))
        log(f"8b summary: {json.dumps(state['summary_8b'], ensure_ascii=False)[:1200]}")
    else:
        log("8b: không thấy plan_b_qwen3_8b_summary.json")

    set_stage(state, "done_8b", status=status)
    state["result"] = "8b_finished"
    return state


# ---------------------------------------------------------------- main
def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log("=" * 70)
    log(f"auto pipeline start | stage={state.get('stage')}")

    try:
        stage = state.get("stage", "wait_3model")
        if stage == "wait_3model":
            state = stage_3model(state)
            stage = state["stage"]

        if stage == "spike_push":
            state = stage_spike(state)
            stage = state["stage"]

        if stage == "8b_push":
            state = stage_8b(state)
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        state["error"] = f"{type(error).__name__}: {error}"
        save_state(state)
        log(f"PIPELINE ERROR: {state['error']}")

    state["finished_at"] = ts()
    save_state(state)

    compact = {
        "stage": state.get("stage"),
        "result": state.get("result"),
        "status_3model": state.get("status_3model"),
        "coverage_3model": state.get("coverage_3model"),
        "incomplete_3model": state.get("incomplete_3model"),
        "spike": state.get("spike"),
        "spike_passed": state.get("spike_passed"),
        "status_8b": state.get("status_8b"),
        "summary_8b": state.get("summary_8b"),
        "error": state.get("error"),
    }
    next_actions = []
    if state.get("incomplete_3model"):
        next_actions.append(
            "re-run 3 model còn thiếu: " + ", ".join(state["incomplete_3model"])
        )
    if state.get("status_3model") and is_terminal(state["status_3model"]) and "COMPLETE" not in state["status_3model"]:
        next_actions.append("kernel 3 model kết thúc bất thường: " + state["status_3model"])
    if state.get("spike_passed") is False:
        next_actions.append("spike fail -> xem /tmp/opencode/shard_spike/shard_spike_report.json")
    summary_8b = state.get("summary_8b") or {}
    if summary_8b.get("missing_slices"):
        next_actions.append(
            f"8B còn slice {summary_8b['missing_slices']} -> push lại với SLICE_START={summary_8b.get('next_slice_start')}"
        )
    compact["next_actions"] = next_actions
    if next_actions:
        log("NEXT ACTIONS: " + json.dumps(next_actions, ensure_ascii=False))
    SUMMARY.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    log("SUMMARY " + json.dumps(compact, ensure_ascii=False))
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
