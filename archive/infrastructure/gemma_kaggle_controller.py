"""Private CPU Gemma handoff; never embeds credentials in saved code or status."""

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
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from kaggle_live_session import KaggleLiveSession
from report_tiki_gemma_status import process_alive

ROOT = Path(__file__).parent
WORK = ROOT/'title_translation_tiki_gemma4_vi_zh_v1'
STATE = Path(r'C:\Users\Lenovo\.codex\gemma-kaggle-controller.json')
OP = Path(r'C:\Users\Lenovo\.codex\gemma-interactive-operation.json')
REMOTE = 'gemma_tiki_live'
SOURCE = Path(r'D:\HuggingFace_Models\hub\datasets--ntquang0410--zh-vie_ecom\snapshots\51540c9e189a6747e768cec08c25c0d4a68776da\data\cleaned\tiki\tiki_vi_cado.parquet')

def save(**values):
    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    old.update(values, updated_at_utc=datetime.now(timezone.utc).isoformat())
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(old));temp.replace(STATE)

def checked(session, code):
    result=session.execute(code, timeout=120)
    if result['status']!='ok':raise RuntimeError('remote_step_failed')
    return result['values']

def snapshot(source, destination):
    with sqlite3.connect('file:'+source.as_posix()+'?mode=ro',uri=True) as src:
        with sqlite3.connect(destination) as dst:src.backup(dst)

def count(path):
    with sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True) as db:
        return db.execute("select count(*) from translations where status='ok'").fetchone()[0]

def stop_local(pid):
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.restype=ctypes.c_void_p
    api.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    api.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=api.OpenProcess(1|0x100000,False,pid)
    if not handle:raise RuntimeError('local_process_unavailable')
    try:
        if not api.TerminateProcess(handle,0):raise RuntimeError('local_stop_failed')
        api.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if api.WaitForSingleObject(handle,10000)!=0:raise RuntimeError('local_stop_not_confirmed')
    finally:api.CloseHandle(handle)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--status',action='store_true');args=parser.parse_args()
    if args.status:
        print(STATE.read_text() if STATE.exists() else '{"phase":"not_started"}');return
    # Hold a local lock for this controller's entire lifetime.
    import msvcrt
    lock=STATE.with_suffix('.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    key=os.environ.pop('GEMINI_API_KEY',None)
    if not key:raise RuntimeError('credential_missing')
    save(phase='preparing',controller_pid=os.getpid(),notebook='nadsa091/gemma-tiki-vi-zh-live')
    session=KaggleLiveSession(OP)
    for attempt in range(40):
        if session.connect():break
        save(phase='waiting_for_session',attempt=attempt+1)
        time.sleep(10)
    else:raise RuntimeError('session_pending')
    checked(session,"import socket,json,importlib.util\nsocket.create_connection(('generativelanguage.googleapis.com',443),timeout=10).close()\nassert importlib.util.find_spec('google.genai')\nimport pandas,pyarrow\nprint('[codex-json] '+json.dumps({'preflight':True}))")
    for name in ('run_tiki_gemma_supervised.py','tiki_translation_pipeline.py'):
        session.put_bytes(REMOTE+'/'+name,(ROOT/name).read_bytes())
    session.put_bytes(REMOTE+'/source.parquet',SOURCE.read_bytes())
    cfg=json.loads((WORK/'run_config.json').read_text())
    checked(session,f"import hashlib,json\nfrom pathlib import Path\np=Path('/kaggle/working/{REMOTE}')\nassert hashlib.sha256((p/'source.parquet').read_bytes()).hexdigest()=={cfg['source_sha256']!r}\nassert hashlib.sha256((p/'tiki_translation_pipeline.py').read_bytes()).hexdigest()=={cfg['pipeline_sha256']!r}\nprint('[codex-json] '+json.dumps({{'source_verified':True}}))")
    backup=WORK/'kaggle_handoff.sqlite3'
    local_status=json.loads((WORK/'tiki_live_status.json').read_text())
    if process_alive(local_status['pid']):
        # Stop only the original local worker, after its API request is committed.
        save(phase='waiting_local_request_boundary')
        deadline=time.monotonic()+180
        while True:
            status=json.loads((WORK/'tiki_live_status.json').read_text())
            if not status.get('request_in_flight'):break
            if time.monotonic()>deadline:raise RuntimeError('local_request_boundary_timeout')
            time.sleep(.1)
        stop_local(int(status['pid']))
        save(phase='local_stopped',local_pid=status['pid'])
    candidates=[p for p in (WORK/'cache_tiki_gemma4.sqlite3',WORK/'kaggle_mirror.sqlite3',backup) if p.exists()]
    latest=max(candidates,key=count)
    if latest!=backup:snapshot(latest,backup)
    baseline=count(backup)
    for name,data in [('cache_tiki_gemma4.sqlite3',backup.read_bytes()),('run_config.json',(WORK/'run_config.json').read_bytes())]:
        session.put_bytes(REMOTE+'/run/'+name,data)
    checked(session,f"import sqlite3,json\nc=sqlite3.connect('/kaggle/working/{REMOTE}/run/cache_tiki_gemma4.sqlite3')\nassert c.execute(\"select count(*) from translations where status='ok'\").fetchone()[0]=={baseline}\nc.close()\nprint('[codex-json] '+json.dumps({{'checkpoint_verified':True}}))")
    # A transient private file supplies only the child environment; unlink it before work.
    session.put_bytes(REMOTE+'/.api-key',key.encode())
    launch=f"""import subprocess,os,json
from pathlib import Path
p=Path('/kaggle/working/{REMOTE}')
secret=p/'.api-key'
env=os.environ.copy();env['GEMINI_API_KEY']=secret.read_text();secret.unlink()
out=(p/'worker.stdout.log').open('ab');err=(p/'worker.stderr.log').open('ab')
worker=subprocess.Popen(['python','-u',str(p/'run_tiki_gemma_supervised.py'),'--source',str(p/'source.parquet'),'--workdir',str(p/'run'),'--revision',{cfg['dataset_revision']!r},'--source-sha256',{cfg['source_sha256']!r},'--defer-full-quality-review'],env=env,stdout=out,stderr=err,start_new_session=True)
env.pop('GEMINI_API_KEY',None)
print('[codex-json] '+json.dumps({{'worker_pid':worker.pid}}))
"""
    values=checked(session,launch)
    save(phase='monitoring',worker_pid=values[0]['worker_pid'],handoff_completed=baseline,completed=baseline)
    key=None
    while True:
        time.sleep(30)
        status=session.read_json(REMOTE+'/run/tiki_live_status.json')
        # SQLite backup is a transactionally consistent snapshot, even while writing.
        checked(session,f"import sqlite3,json\ns=sqlite3.connect('/kaggle/working/{REMOTE}/run/cache_tiki_gemma4.sqlite3');d=sqlite3.connect('/kaggle/working/{REMOTE}/snapshot.sqlite3');s.backup(d);d.close();s.close()\nprint('[codex-json] '+json.dumps({{'snapshot':True}}))")
        raw=session.get_bytes(REMOTE+'/snapshot.sqlite3')
        mirror=WORK/'kaggle_mirror.sqlite3';mirror.write_bytes(raw)
        completed=count(mirror)
        if completed<baseline:raise RuntimeError('checkpoint_regression')
        save(phase='monitoring',completed=completed,total=status.get('total'),health=status.get('health'),worker_status=status.get('status'),heartbeat_at_utc=status.get('heartbeat_at_utc'),last_progress_age_seconds=status.get('last_progress_age_seconds'),qa_passed=status.get('qa_passed'),quota_status=status.get('quota_status'))
        if status['status']=='completed':
            report=session.read_json(REMOTE+'/run/tiki_gemma4_report.json')
            name=Path(report['output_path']).name
            raw=session.get_bytes(REMOTE+'/run/'+name)
            if hashlib.sha256(raw).hexdigest()!=report['output_sha256']:raise RuntimeError('output_checksum_mismatch')
            if report['completed']!=13102 or report['output_rows']!=13200:raise RuntimeError('output_counts_mismatch')
            (WORK/name).write_bytes(raw)
            (WORK/'tiki_gemma4_report.json').write_text(json.dumps(report))
            save(phase='completed',output_sha256=report['output_sha256'],output_rows=13200,quality_review_passed=report['quality_review_passed']);return
        if status['status']!='running':
            save(phase='paused',worker_status=status['status']);return

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(phase='controller_error',error_class=type(exc).__name__)
        print(json.dumps({'error_class':type(exc).__name__}))
        raise SystemExit(1)
