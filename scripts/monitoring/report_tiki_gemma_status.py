"""Read safe Tiki worker telemetry without opening translations or API logs."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def process_alive(pid):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, default=Path(__file__).parent/'title_translation_tiki_gemma4_vi_zh_v1')
    args = parser.parse_args()
    path = args.workdir/'tiki_live_status.json'
    if not path.is_file():
        print(json.dumps({'status':'no_telemetry', 'status_file_exists':False}))
        return
    state = json.loads(path.read_text(encoding='utf-8'))
    now = datetime.now(timezone.utc)
    age = max(0, (now-datetime.fromisoformat(state['heartbeat_at_utc'])).total_seconds())
    state['heartbeat_age_seconds'] = round(age, 1)
    state['last_progress_age_seconds'] = round(age + state.get('last_progress_age_seconds', 0), 1)
    state['process_alive'] = process_alive(state['pid'])
    if state['status']=='running':
        if not state['process_alive']:
            state['health'] = 'stopped_without_final_status'
        elif age > 90:
            state['health'] = 'heartbeat_stale'
        elif state['last_progress_age_seconds'] > 600:
            state['health'] = 'suspected_stall'
    print(json.dumps(state, ensure_ascii=True))


if __name__=='__main__':
    main()
