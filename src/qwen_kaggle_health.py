"""Small, privacy-safe notebook heartbeat; safe to embed before notebook cells."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[1]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import contextlib as _health_contextlib
import json as _health_json
import os as _health_os
import re as _health_re
import subprocess as _health_subprocess
import threading as _health_threading
import time as _health_time
from datetime import datetime as _HealthDateTime, timezone as _HealthTimezone
from pathlib import Path as _HealthPath


class QwenHealthMonitor:
    def __init__(self, path, interval=30, clock=None, checkpoint_name='cache_qwen3_14b_awq.sqlite3'):
        self.path = _HealthPath(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or _health_time.monotonic
        self.interval = interval
        self.checkpoint_name = checkpoint_name
        self.lock = _health_threading.RLock()
        self.stop_event = _health_threading.Event()
        self.started = self.clock()
        self.last_progress = self.started
        self.last_emit = -float('inf')
        self.last_signature = None
        self.payload = {'phase': 'initializing', 'cell': None, 'state': 'starting'}
        self.worker = _health_threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def set_phase(self, phase, cell=None):
        if not isinstance(phase, str) or not _health_re.fullmatch(r'[a-z_]{1,48}', phase):
            phase = 'unknown'
        with self.lock:
            self.payload.update(phase=phase, state='running')
            if cell is not None:
                self.payload['cell'] = int(cell)
            self.last_progress = self.clock()
            self.last_signature = None
            for key in ('progress_done', 'progress_total', 'progress_unit'):
                self.payload.pop(key, None)
        self.emit(force=True)

    def progress(self, progress=None, **kwargs):
        values = dict(progress or {})
        values.update(kwargs)
        allowed = ('processed_this_session', 'completed_this_session', 'errors_this_session',
                   'total_this_session', 'completed_before_session', 'progress_done', 'progress_total')
        with self.lock:
            phase = values.get('phase')
            if isinstance(phase, str) and _health_re.fullmatch(r'[a-z_]{1,48}', phase):
                self.payload['phase'] = phase
            for key in allowed:
                value = values.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.payload[key] = max(0, int(value))
            unit = values.get('progress_unit')
            if unit in ('bytes', 'installer_lines', 'titles', 'files'):
                self.payload['progress_unit'] = unit
            signature = (self.payload['phase'],) + tuple(self.payload.get(k) for k in allowed)
            if signature != self.last_signature:
                self.last_progress = self.clock()
                self.last_signature = signature
        self.emit()

    def installer_activity(self):
        with self.lock:
            done = self.payload.get('progress_done', 0) + 1
        self.progress(progress_done=done, progress_unit='installer_lines')

    def cell_finished(self, final=False):
        with self.lock:
            self.payload['state'] = 'completed' if final else 'waiting_next_cell'
            self.last_progress = self.clock()
        self.emit(force=True)
        if final:
            self.stop_event.set()

    def fail(self, error):
        with self.lock:
            self.payload.update(state='error', last_error_class=type(error).__name__)
        self.emit(force=True)
        self.stop_event.set()

    def snapshot(self):
        with self.lock:
            now = self.clock()
            age = max(0, now - self.last_progress)
            payload = dict(self.payload)
            terminal = payload['state'] in ('completed', 'error')
            threshold = {'install_dependencies': 900, 'verify_model': 600,
                         'loading_model': 600, 'generation': 300}.get(payload['phase'], 600)
            if not terminal and payload['state'] != 'waiting_next_cell':
                payload['state'] = 'suspected_stall' if age > threshold else 'running'
            payload.update(heartbeat_at_utc=_HealthDateTime.now(_HealthTimezone.utc).isoformat(),
                           last_progress_at_utc=_HealthDateTime.fromtimestamp(
                               _health_time.time() - age, _HealthTimezone.utc).isoformat(),
                           progress_age_seconds=round(age, 1), elapsed_seconds=round(now-self.started, 1),
                           stall_threshold_seconds=threshold, process_id=_health_os.getpid(),
                           process_cpu_seconds=round(sum(_health_os.times()[:2]), 2))
            checkpoint = self.path.parent / self.checkpoint_name
            try:
                stat = checkpoint.stat()
                payload.update(checkpoint_bytes=stat.st_size,
                               checkpoint_age_seconds=round(max(0, _health_time.time()-stat.st_mtime), 1))
            except OSError:
                payload['checkpoint_available'] = False
            return payload

    def emit(self, force=False, gpu=False):
        with self.lock:
            if not force and self.clock()-self.last_emit < 10:
                return
            payload = self.snapshot()
            if gpu:
                try:
                    result = _health_subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=3)
                    row = result.stdout.splitlines()[0].split(',')
                    payload.update(gpu_utilization_percent=int(row[0].strip()), gpu_memory_used_mib=int(row[1].strip()))
                except Exception:
                    payload['gpu_metrics_available'] = False
            temporary = self.path.with_suffix('.json.tmp')
            try:
                temporary.write_text(_health_json.dumps(payload, ensure_ascii=True), encoding='utf-8')
                temporary.replace(self.path)
            except Exception as exc:
                payload['health_write_error_class'] = type(exc).__name__
            print('[qwen-health] ' + _health_json.dumps(payload, ensure_ascii=True), flush=True)
            self.last_emit = self.clock()

    @_health_contextlib.contextmanager
    def stage(self, phase, cell=None, final=False):
        self.set_phase(phase, cell)
        try:
            yield
        except BaseException as exc:
            with self.lock:
                self.payload.update(state='error', last_error_class=type(exc).__name__)
            self.emit(force=True)
            self.stop_event.set()
            raise
        else:
            with self.lock:
                self.payload['state'] = 'completed' if final else 'waiting_next_cell'
                self.last_progress = self.clock()
            self.emit(force=True)
            if final:
                self.stop_event.set()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.emit(force=True, gpu=True)

    def close(self):
        self.stop_event.set()
