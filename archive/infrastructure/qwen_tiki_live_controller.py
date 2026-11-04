"""Start Qwen Tiki only after the 1688 export is complete and verified."""
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
import ast
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from kaggle_live_session import KaggleLiveSession
from report_tiki_gemma_status import process_alive

ROOT=Path(__file__).parent
RUN='title_translation_tiki_qwen3_14b_awq_vi_zh_v1'
STATE=Path(r'C:\Users\Lenovo\.codex\qwen-tiki-live-controller.json')
LOCAL=ROOT/RUN
SOURCE=Path(r'D:\HuggingFace_Models\hub\datasets--ntquang0410--zh-vie_ecom\snapshots\51540c9e189a6747e768cec08c25c0d4a68776da\data\cleaned\tiki\tiki_vi_cado.parquet')
CACHE='cache_tiki_qwen3_14b_awq.sqlite3'
SOURCE_SHA='26e027f714764d16cdd76ae711892b869931fa7302fff5bc75b64a43415e4297'
state={'controller_pid':os.getpid(),'dataset':'tiki','direction':'vi_to_zh',
       'model':'Qwen/Qwen3-14B-AWQ','dataset_total':13102,'phase':'starting',
       'publishes_new_kaggle_versions':False}


def validate_predecessor(first):
    if not first.get('qwen_1688_completed') or first.get('output_rows')!=16348:
        return False
    output=Path(first.get('output_path',''))
    return output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest()==first.get('output_sha256')


def save(**values):
    state.update(values,updated_at_utc=datetime.now(timezone.utc).isoformat())
    temporary=STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8');temporary.replace(STATE)
    print(json.dumps(state),flush=True)


def status(session):
    code="""
from pathlib import Path
import json,os,sqlite3
root=Path('/kaggle/working/"""+RUN+"""')
result={}
for key,name in [('health','qwen_health.json'),('report','qwen_tiki_report.json')]:
    path=root/name
    if path.exists():
        value=json.loads(path.read_text())
        allowed=('phase','state','process_id','heartbeat_at_utc','progress_age_seconds','elapsed_seconds',
                 'completed_this_session','errors_this_session','last_error_class','status',
                 'completed_unique_titles','total_unique_titles','qa_passed','quality_review_passed','output_rows','output_sha256')
        result[key]={k:v for k,v in value.items() if k in allowed}
result['worker_alive']=False
pid=result.get('health',{}).get('process_id')
if pid:
    try:
        os.kill(pid,0)
        result['worker_alive']=b'qwen_tiki_live_worker.py' in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except OSError: pass
db=root/'"""+CACHE+"""'
if db.exists():
    connection=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=2)
    result['completed_total']=connection.execute("SELECT count(*) FROM translations WHERE status='ok'").fetchone()[0]
    connection.close()
print('[codex-json] '+json.dumps(result),flush=True)
"""
    result=session.execute(code,timeout=60)
    if result['status']!='ok' or not result['values']:raise RuntimeError('tiki_status_probe_failed')
    return result['values'][-1]


def backup(session):
    LOCAL.mkdir(exist_ok=True)
    previous=0
    if (LOCAL/CACHE).exists():
        with closing(sqlite3.connect((LOCAL/CACHE).as_uri()+'?mode=ro',uri=True)) as db:
            previous=db.execute("SELECT count(*) FROM translations WHERE status='ok'").fetchone()[0]
    for name in (CACHE,'run_config.json','stage_reports.json','qwen_tiki_report.json'):
        try:data=session.get_bytes(RUN+'/'+name)
        except Exception:continue
        temporary=LOCAL/(name+'.download');temporary.write_bytes(data)
        if name==CACHE:
            if not data.startswith(b'SQLite format 3\0'):raise RuntimeError('invalid_tiki_checkpoint')
            with closing(sqlite3.connect(temporary.as_uri()+'?mode=ro',uri=True)) as db:
                completed=db.execute("SELECT count(*) FROM translations WHERE status='ok'").fetchone()[0]
            if completed<previous:raise RuntimeError('tiki_checkpoint_regression')
        temporary.replace(LOCAL/name)
    save(checkpoint_saved_at_utc=datetime.now(timezone.utc).isoformat(),checkpoint_directory=str(LOCAL))


def launch(session):
    from build_title_translation_qwen3_14b_awq_kaggle_v6 import LOCAL_QWEN_MODEL_SETUP
    tree=ast.parse(LOCAL_QWEN_MODEL_SETUP)
    manifest=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='QWEN_MODEL_FILES' for t in n.targets))
    for name in ('qwen_tiki_live_worker.py','qwen_kaggle_health.py','tiki_translation_pipeline.py'):
        session.put_bytes(RUN+'/'+name,(ROOT/name).read_bytes())
    session.put_bytes(RUN+'/model_manifest.json',json.dumps(manifest).encode())
    session.put_bytes(RUN+'/source_tiki.parquet',SOURCE.read_bytes())
    if (LOCAL/CACHE).exists():session.put_bytes(RUN+'/'+CACHE,(LOCAL/CACHE).read_bytes())
    if (LOCAL/'run_config.json').exists():session.put_bytes(RUN+'/run_config.json',(LOCAL/'run_config.json').read_bytes())
    code="import sys,subprocess,json;from pathlib import Path;r=Path('/kaggle/working/"+RUN+"');f=(r/'qwen_tiki_worker.stdout.log').open('a');p=subprocess.Popen([sys.executable,'-u',str(r/'qwen_tiki_live_worker.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print('[codex-json] '+json.dumps({'worker_pid':p.pid}))"
    result=session.execute(code,timeout=60)
    if result['status']!='ok' or not result['values']:raise RuntimeError('tiki_worker_launch_failed')
    save(phase='worker_starting',worker_pid=result['values'][-1]['worker_pid'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--status',action='store_true');parser.add_argument('--ensure-running',action='store_true');args=parser.parse_args()
    if STATE.exists():
        previous=json.loads(STATE.read_text());alive=process_alive(previous.get('controller_pid',0))
        if args.status or (args.ensure_running and alive):
            previous['controller_alive']=alive;print(json.dumps(previous));return
        if alive:raise SystemExit('Tiki controller already running')
    elif args.status:
        print(json.dumps({'phase':'not_started','controller_alive':False}));return
    first=json.loads(Path(r'C:\Users\Lenovo\.codex\qwen-live-controller.json').read_text())
    if not validate_predecessor(first):
        raise SystemExit('Qwen 1688 has not been verified complete; Tiki remains queued.')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:raise SystemExit('Tiki source checksum mismatch')
    lock=STATE.with_suffix('.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        save();session=KaggleLiveSession()
        if not session.connect():raise RuntimeError('session_not_ready')
        current=status(session)
        if not current.get('worker_alive'):launch(session)
        last_backup=time.monotonic();retries=0;best=current.get('completed_total',0)
        while True:
            current=status(session);completed=current.get('completed_total',0)
            save(phase='monitoring',completed_total=completed,worker_alive=current.get('worker_alive'),health=current.get('health',{}),report=current.get('report',{}))
            if time.monotonic()-last_backup>120 or not current.get('worker_alive'):
                backup(session);last_backup=time.monotonic()
            if not current.get('worker_alive'):
                report=current.get('report',{})
                if completed==13102 and report.get('status')=='all_titles_complete' and report.get('output_rows')==13200:
                    name='tiki_vi_cado_title_qwen3_14b_awq.parquet';data=session.get_bytes(RUN+'/'+name)
                    if hashlib.sha256(data).hexdigest()!=report.get('output_sha256'):raise RuntimeError('tiki_export_checksum_mismatch')
                    (LOCAL/name).write_bytes(data)
                    save(phase='completed',output_path=str(LOCAL/name),output_sha256=report['output_sha256'],output_rows=13200)
                    return
                if completed>best:retries=0;best=completed
                retries+=1
                if retries>3:save(phase='paused_worker_error');return
                launch(session)
            time.sleep(60)
    except Exception as exc:save(phase='paused_controller_error',controller_error_class=type(exc).__name__)
    finally:lock.close()


if __name__=='__main__':main()
