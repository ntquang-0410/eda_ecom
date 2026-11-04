"""Bounded Kaggle batch execution with consistent output checkpoints, no local host."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[1]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import hashlib
from contextlib import closing
import json
import re
from pathlib import Path
import sqlite3
import subprocess
import threading
import time

def cache_progress(source):
    """Read counts only; reject schemas unrelated to the translation caches."""
    source=Path(source)
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=5)) as db:
        columns={row[1] for row in db.execute('pragma table_info(translations)')}
        if 'status' not in columns:return None
        condition="status='ok'" + (' AND truncated=0' if 'truncated' in columns else '')
        completed,total=db.execute('SELECT sum(CASE WHEN '+condition+' THEN 1 ELSE 0 END),count(*) FROM translations').fetchone()
    return {'cache':source.name,'completed':int(completed or 0),'cached_records':total}

def snapshot_all(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    counts={}
    for base in (Path('/content'),Path('/kaggle/working')):
        if not base.exists():continue
        for source in base.rglob('*.sqlite3'):
            if destination in source.parents:continue
            name=hashlib.sha256(str(source).encode()).hexdigest()[:12]+'_'+source.name
            target=destination/name
            try:
                with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=5)) as src:
                    with closing(sqlite3.connect(str(target)+'.tmp',timeout=5)) as dst:src.backup(dst)
                Path(str(target)+'.tmp').replace(target)
                progress=cache_progress(target)
                if progress:
                    progress['cache']=source.name
                    previous=counts.get(source.name)
                    if previous is None or progress['completed']>previous['completed']:
                        counts[source.name]=progress
            except sqlite3.Error:continue
    for value in counts.values():
        print('[batch-cache] '+json.dumps(value),flush=True)

def run(command,workdir,timeout=9*3600,env=None):
    workdir=Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
    stop=threading.Event()
    def mirror():
        offset=0
        while not stop.wait(15):
            log=workdir/'worker.stdout.log'
            if log.exists():
                with log.open(encoding='utf-8',errors='replace') as stream:
                    stream.seek(offset)
                    chunk=stream.read()
                    offset=stream.tell()
                    for line in chunk.splitlines():
                        value=None
                        text=line.strip()
                        if text.startswith('[qwen-health] '):text=text[len('[qwen-health] '):]
                        try:
                            parsed=json.loads(text)
                            if isinstance(parsed,dict):
                                allowed=('phase','state','status','completed','total','qa_passed','api_attempts','api_errors','quota_status','heartbeat_at_utc','last_progress_age_seconds','processed_this_session','completed_this_session','errors_this_session','total_this_session','gpu_utilization_percent','gpu_memory_used_mib')
                                value={k:v for k,v in parsed.items() if k in allowed}
                        except ValueError:
                            match=re.search(r'uncached (\d+)/(\d+); complete=(\d+); errors=(\d+)',text)
                            if match:value=dict(zip(('processed','session_total','session_completed','session_errors'),map(int,match.groups())))
                        if value:print('[batch-health] '+json.dumps(value),flush=True)
            snapshot_all('/kaggle/working/durable_checkpoints')
    thread=threading.Thread(target=mirror,daemon=True);thread.start()
    outcome={'status':'starting'}
    try:
        with (workdir/'worker.stdout.log').open('ab') as stdout,(workdir/'worker.stderr.log').open('ab') as stderr:
            worker=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
            try:
                code=worker.wait(timeout=timeout)
                outcome={'status':'worker_finished' if code==0 else 'worker_failed','returncode':code}
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
                outcome={'status':'resumable_time_limit'}
    finally:
        stop.set();thread.join(timeout=15)
        snapshot_all('/kaggle/working/durable_checkpoints')
        (workdir/'batch_outcome.json').write_text(json.dumps(outcome))
    return outcome
