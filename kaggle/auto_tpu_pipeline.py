#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrator TPU/GPU tự động cho chuỗi Plan B (4 Qwen3 model).

Luật:
  - Có 2 kernel cùng xếp hàng:
      TPU : 3model (0.6B/1.7B/4B)   -> rồi shard spike -> 8B sharded
      GPU : 4model (0.6B..8B)       -> chạy hết trên RTX Pro 6000
  - **Ai được cấp máy trước thì thắng**, kernel còn lại bị huỷ để không đốt quota.
  - GPU thắng  -> lấy cả 4 model từ GPU, xong pipeline.
  - TPU thắng  -> chạy 3 model, validate sharding, rồi chạy 8B sharded.
  - Nhánh thua vẫn được "harvest" nếu lỡ chạy xong (tải output, ghi coverage).

State: /tmp/opencode/auto_pipeline_state.json (resume được)
Log:   /tmp/opencode/auto_pipeline.log
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

K_TPU3 = "ntquang0410/tiki-plan-b-bt-tpu-v5e-8"
K_SPIKE = "ntquang0410/tiki-plan-b-tpu-shard-spike"
K_TPU8B = "ntquang0410/tiki-plan-b-bt-tpu-8b"
K_GPU = "ntquang0410/tiki-plan-b-bt-rtx-4-model"

D_TPU3 = "kaggle/tpu_bt"
D_SPIKE = "kaggle/tpu_shard"
D_TPU8B = "kaggle/tpu_bt_8b"
D_GPU = "kaggle/rtx_bt"

O_TPU3 = Path("/tmp/opencode/tpu_bt_output")
O_SPIKE = Path("/tmp/opencode/shard_spike")
O_TPU8B = Path("/tmp/opencode/tpu_8b")
O_GPU = Path("/tmp/opencode/rtx_bt_output")

TERMINAL = ("COMPLETE", "ERROR", "CANCEL", "TIMEOUT")
SLOT_BUSY = "Maximum batch"

RACE_MAX_HOURS = 40
WAIT_TPU3_HOURS = 36
WAIT_SPIKE_HOURS = 12
WAIT_TPU8B_HOURS = 14
WAIT_GPU_HOURS = 16
PUSH_RETRY_HOURS = 10
POLL_SECONDS = 60
GPU_GUARD_MIN = 15  # GPU chạy ổn định chừng này thì mới huỷ kernel TPU


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
            ["kaggle", *args], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT)
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
    if "denied" in output or "not been shared" in output or "not found" in output:
        return "MISSING"
    return "UNKNOWN"


def is_terminal(status: str) -> bool:
    return any(marker in status for marker in TERMINAL)


def is_running(status: str) -> bool:
    return "RUNNING" in status


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"stage": "race", "history": []}


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
        log(output.strip()[:1200])
    return ok


def push_with_retry(kernel_dir: str, label: str, max_hours: float = PUSH_RETRY_HOURS) -> bool:
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        code, output = kaggle("kernels", "push", "-p", str(ROOT / kernel_dir), timeout=900)
        if "successfully pushed" in output:
            version = re.search(r"Kernel version (\d+) successfully pushed", output)
            log(f"push {label} OK version={version.group(1) if version else '?'}")
            return True
        if SLOT_BUSY in output:
            log(f"push {label}: hết slot, chờ 3 phút")
            time.sleep(180)
            continue
        log(f"push {label} FAILED: {output.strip()[:600]}")
        time.sleep(300)
    return False


def ensure_kernel(kernel: str, kernel_dir: str, label: str) -> bool:
    if kernel_status(kernel) != "MISSING":
        return True
    log(f"{label}: kernel chưa tồn tại -> push")
    return push_with_retry(kernel_dir, label, max_hours=2)


def delete_kernel(kernel: str, label: str) -> bool:
    code, output = kaggle("kernels", "delete", kernel, "-y", timeout=300)
    ok = "deleted successfully" in output
    log(f"delete {label} ({kernel}) ok={ok} {'' if ok else output.strip()[:300]}")
    return ok


def collect_coverage(root: Path) -> dict:
    coverage = {}
    for state_file in root.rglob("run_state.json"):
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        key = payload.get("model_key") or state_file.parent.name
        coverage[key] = {
            "n_done": payload.get("n_done"),
            "n_total": payload.get("n_total"),
            "entries_per_min": payload.get("entries_per_min"),
            "gpu": payload.get("gpu"),
        }
    return coverage


def find_report(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def incomplete_models(coverage: dict) -> list[str]:
    out = []
    for key, payload in coverage.items():
        done = payload.get("n_done") or 0
        total = payload.get("n_total") or 0
        if total and done < total:
            out.append(f"{key}:{done}/{total}")
    return out


# ---------------------------------------------------------------- stages
def stage_race(state: dict) -> dict:
    """Ai được cấp máy trước thì thắng."""
    ensure_kernel(K_GPU, D_GPU, "gpu")
    ensure_kernel(K_TPU3, D_TPU3, "tpu3")

    deadline = time.time() + RACE_MAX_HOURS * 3600
    last = (None, None)
    while time.time() < deadline:
        s_tpu = kernel_status(K_TPU3)
        s_gpu = kernel_status(K_GPU)
        if (s_tpu, s_gpu) != last:
            log(f"race | tpu3={s_tpu} | gpu={s_gpu}")
            last = (s_tpu, s_gpu)

        tpu_run, gpu_run = is_running(s_tpu), is_running(s_gpu)

        if tpu_run and gpu_run:
            state["winner"] = "gpu"
            state["race_note"] = "cả hai cùng chạy; ưu tiên GPU"
            set_stage(state, "gpu_run", winner="gpu", note="both running")
            return state

        if tpu_run:
            state["winner"] = "tpu"
            set_stage(state, "tpu_run", winner="tpu", tpu_status=s_tpu, gpu_status=s_gpu)
            return state

        if gpu_run:
            state["winner"] = "gpu"
            set_stage(state, "gpu_run", winner="gpu", tpu_status=s_tpu, gpu_status=s_gpu)
            return state

        if is_terminal(s_tpu):
            state["winner"] = "tpu"
            set_stage(state, "tpu_run", winner="tpu", tpu_status=s_tpu, gpu_status=s_gpu)
            return state

        if is_terminal(s_gpu):
            state["winner"] = "gpu"
            set_stage(state, "gpu_run", winner="gpu", tpu_status=s_tpu, gpu_status=s_gpu)
            return state

        time.sleep(POLL_SECONDS)

    state["winner"] = None
    state["result"] = "race_timeout"
    set_stage(state, "race_timeout")
    return state


def stage_gpu(state: dict) -> dict:
    """RTX Pro 6000 chạy cả 4 model."""
    started_at = time.time()
    guard_done = False
    status = "UNKNOWN"

    while True:
        status = kernel_status(K_GPU)
        elapsed_min = (time.time() - started_at) / 60

        if not guard_done and is_running(status) and elapsed_min >= GPU_GUARD_MIN:
            guard_done = True
            tpu_status = kernel_status(K_TPU3)
            if tpu_status == "KernelWorkerStatus.QUEUED":
                log(f"GPU đã chạy {elapsed_min:.0f} phút -> huỷ kernel TPU đang xếp hàng để tiết kiệm quota")
                delete_kernel(K_TPU3, "tpu3")
                state["tpu3_deleted"] = True
            else:
                log(f"GPU guard: TPU status = {tpu_status}, không huỷ")

        if is_terminal(status):
            break
        if elapsed_min > WAIT_GPU_HOURS * 60:
            status = "WATCH_TIMEOUT"
            break
        if elapsed_min > 60 and state.get("status_gpu") != status:
            pass
        state["status_gpu"] = status
        time.sleep(POLL_SECONDS)

    state["status_gpu"] = status
    log(f"gpu kết thúc: {status}")
    download_outputs(K_GPU, O_GPU)
    coverage = collect_coverage(O_GPU)
    state["coverage_gpu"] = coverage
    state["incomplete_gpu"] = incomplete_models(coverage)
    log(f"gpu coverage: {json.dumps(coverage, ensure_ascii=False)}")

    summary_path = find_report(O_GPU, "plan_b_rtx_summary.json")
    if summary_path is not None:
        state["summary_gpu"] = json.loads(summary_path.read_text(encoding="utf-8"))
        gpu_name = state["summary_gpu"].get("gpu")
        state["gpu_name"] = gpu_name
        log(f"gpu name: {gpu_name} | vram {state['summary_gpu'].get('vram_gb')}GB")

    if status == "KernelWorkerStatus.ERROR" and not coverage and not state.get("summary_gpu"):
        log("GPU fail sớm, không có output -> chuyển sang nhánh TPU")
        set_stage(state, "tpu_run", reason="gpu_failed_early", status_gpu=status)
        return state

    set_stage(state, "done_gpu", status=status, coverage=coverage)
    state["result"] = "gpu_finished"
    return state


def stage_tpu(state: dict) -> dict:
    """TPU: 3 model -> spike sharding -> 8B sharded."""
    ensure_kernel(K_TPU3, D_TPU3, "tpu3")

    gpu_status = kernel_status(K_GPU)
    if gpu_status == "KernelWorkerStatus.QUEUED":
        log("TPU thắng race -> huỷ kernel GPU đang xếp hàng")
        delete_kernel(K_GPU, "gpu")
        state["gpu_deleted"] = True
    elif gpu_status != "MISSING":
        log(f"TPU thắng race nhưng GPU status = {gpu_status}")

    status = wait_kernel(K_TPU3, WAIT_TPU3_HOURS, "tpu3")
    state["status_3model"] = status
    download_outputs(K_TPU3, O_TPU3)
    coverage = collect_coverage(O_TPU3)
    state["coverage_3model"] = coverage
    state["incomplete_3model"] = incomplete_models(coverage)
    log(f"3model coverage: {json.dumps(coverage, ensure_ascii=False)}")
    if state["incomplete_3model"]:
        log(f"3model INCOMPLETE -> cần re-run: {', '.join(state['incomplete_3model'])}")
    if not coverage:
        log("3model: không đọc được run_state.json")

    set_stage(state, "spike_push", status_3model=status, coverage=coverage)

    if not push_with_retry(D_SPIKE, "spike"):
        state["result"] = "spike_push_failed"
        set_stage(state, "spike_push_failed")
        return state
    set_stage(state, "spike_run")

    spike_status = wait_kernel(K_SPIKE, WAIT_SPIKE_HOURS, "spike")
    state["status_spike"] = spike_status
    download_outputs(K_SPIKE, O_SPIKE)

    report_path = find_report(O_SPIKE, "shard_spike_report.json")
    if report_path is None:
        log("spike: không thấy shard_spike_report.json")
        state["result"] = "spike_no_report"
        set_stage(state, "spike_no_report", status=spike_status)
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
    if not passed:
        state["result"] = "spike_failed"
        set_stage(state, "spike_failed", spike=verdict)
        return state

    set_stage(state, "8b_push", spike=verdict)
    if not push_with_retry(D_TPU8B, "8b"):
        state["result"] = "8b_push_failed"
        set_stage(state, "8b_push_failed")
        return state
    set_stage(state, "8b_run")

    status_8b = wait_kernel(K_TPU8B, WAIT_TPU8B_HOURS, "8b")
    state["status_8b"] = status_8b
    download_outputs(K_TPU8B, O_TPU8B)
    summary_8b = find_report(O_TPU8B, "plan_b_qwen3_8b_summary.json")
    if summary_8b is not None:
        state["summary_8b"] = json.loads(summary_8b.read_text(encoding="utf-8"))
        log(f"8b summary: {json.dumps(state['summary_8b'], ensure_ascii=False)[:1000]}")
    else:
        log("8b: không thấy plan_b_qwen3_8b_summary.json")

    state["result"] = "tpu_finished"
    set_stage(state, "done_tpu", status=status_8b)
    return state


def harvest(state: dict) -> None:
    """Tải output của nhánh thua nếu lỡ chạy xong (không bỏ phí)."""
    harvested = {}
    for kernel, dest, label in ((K_GPU, O_GPU, "gpu"), (K_TPU3, O_TPU3, "tpu3")):
        status = kernel_status(kernel)
        if not is_terminal(status):
            continue
        if dest.exists() and any(dest.rglob("run_state.json")):
            coverage = collect_coverage(dest)
        else:
            download_outputs(kernel, dest)
            coverage = collect_coverage(dest)
        if coverage:
            harvested[label] = {"status": status, "coverage": coverage}
    if harvested:
        state["harvested"] = harvested
        log(f"harvest: {json.dumps(harvested, ensure_ascii=False)}")


# ---------------------------------------------------------------- main
def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log("=" * 70)
    log(f"pipeline start | stage={state.get('stage')}")

    try:
        if state.get("stage") == "race":
            state = stage_race(state)
        if state.get("stage") == "gpu_run":
            state = stage_gpu(state)
        if state.get("stage") == "tpu_run":
            state = stage_tpu(state)
        harvest(state)
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        state["error"] = f"{type(error).__name__}: {error}"
        log(f"PIPELINE ERROR: {state['error']}")

    state["finished_at"] = ts()
    save_state(state)

    compact = {
        "result": state.get("result"),
        "stage": state.get("stage"),
        "winner": state.get("winner"),
        "gpu_name": state.get("gpu_name"),
        "status_gpu": state.get("status_gpu"),
        "coverage_gpu": state.get("coverage_gpu"),
        "status_3model": state.get("status_3model"),
        "coverage_3model": state.get("coverage_3model"),
        "spike": state.get("spike"),
        "spike_passed": state.get("spike_passed"),
        "summary_8b": state.get("summary_8b"),
        "harvested": state.get("harvested"),
        "error": state.get("error"),
    }

    next_actions = []
    coverage = state.get("coverage_gpu") or state.get("coverage_3model") or {}
    missing = incomplete_models(coverage)
    if missing:
        next_actions.append("model còn thiếu entry: " + ", ".join(missing))
    if state.get("status_gpu") and is_terminal(state["status_gpu"]) and "COMPLETE" not in state["status_gpu"]:
        next_actions.append(f"GPU kết thúc bất thường: {state['status_gpu']}")
    if state.get("spike_passed") is False:
        next_actions.append("spike fail -> xem /tmp/opencode/shard_spike/shard_spike_report.json")
    summary_8b = state.get("summary_8b") or {}
    if summary_8b.get("missing_slices"):
        next_actions.append(
            f"8B còn slice {summary_8b['missing_slices']} -> SLICE_START={summary_8b.get('next_slice_start')}"
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
