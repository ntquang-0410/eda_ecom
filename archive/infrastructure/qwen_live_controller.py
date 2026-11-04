"""Resume Qwen in the same editable Kaggle session, without publishing versions."""
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
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

from kaggle_live_session import KaggleLiveSession
from prepare_live_qwen_worker import build as build_worker

ROOT=Path(__file__).parent
RUN='title_translation_cado_qwen3_14b_awq_full_v3_sampled'
STATE=Path(r'C:\Users\Lenovo\.codex\qwen-live-controller.json')
CHECKPOINT=Path(r'C:\Users\Lenovo\.codex\qwen-live-checkpoint')/RUN
INITIAL=Path(r'C:\Users\Lenovo\.codex\qwen-v18-checkpoint')/RUN
FILES=['cache_qwen3_14b_awq.sqlite3','hybrid_environment_lock.json','hybrid_run_state.json',
       'hybrid_sample_manifest.csv','hybrid_sample_manifest.json','legacy_cache_import.json',
       'qwen3_14b_awq_kaggle_package_lock.json','runtime_qwen3_14b_awq.json','qwen3_14b_awq_report.json']
state={'controller_pid':os.getpid(),'notebook':'nadsa091/qwen3-14b-awq-kaggle-v6',
       'dataset':'1688','direction':'zh_to_vi','dataset_total':15863,'phase':'connecting',
       'publishes_new_kaggle_versions':False,'next_dataset':'tiki_vi_to_zh','started_at_utc':datetime.now(timezone.utc).isoformat()}


def save(**values):
    state.update(values,updated_at_utc=datetime.now(timezone.utc).isoformat())
    temporary=STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state,ensure_ascii=True,indent=2),encoding='utf-8')
    temporary.replace(STATE)
    print(json.dumps(state,ensure_ascii=True),flush=True)


def count_ok(path):
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        return db.execute("SELECT count(*) FROM translations WHERE status='ok' AND truncated=0").fetchone()[0]


def on_event(tag,value):
    if tag=='[qwen-health]':
        allowed=('phase','state','cell','progress_age_seconds','elapsed_seconds','progress_done',
                 'progress_total','progress_unit','processed_this_session','completed_this_session',
                 'errors_this_session','total_this_session','gpu_utilization_percent','gpu_memory_used_mib')
        save(phase='bootstrap',health={k:value[k] for k in allowed if k in value})
    elif tag=='[qwen-boot]':
        save(gpu_preflight={k:v for k,v in value.items() if k in
             ('torch','cuda_available','nvidia_device_present','nvidia_smi_available')})


def remote_status(session):
    code='''
from pathlib import Path
import json, os, sqlite3
root=Path('/kaggle/working/'''+RUN+'''')
result={}
for key,name in [('health','qwen_health.json'),('live','qwen_live_status.json'),('report','qwen3_14b_awq_report.json')]:
    path=root/name
    if path.exists():
        try:
            value=json.loads(path.read_text())
            allowed=('phase','state','pid','process_id','heartbeat_at_utc','last_progress_at_utc',
                     'progress_age_seconds','elapsed_seconds','processed_this_session','completed_this_session',
                     'errors_this_session','total_this_session','completed_before_session','completed_estimate',
                     'last_error_class','status','completed_unique_titles','total_unique_titles','remaining_unique_titles')
            result[key]={k:v for k,v in value.items() if k in allowed}
        except Exception:
            pass
pid=result.get('health',{}).get('process_id')
result['worker_alive']=False
if pid:
    try:
        os.kill(int(pid),0)
        cmd=Path('/proc')/str(pid)/'cmdline'
        result['worker_alive']=b'qwen_live_worker.py' in cmd.read_bytes()
    except (OSError,ValueError):
        pass
db=Path('/kaggle/temp/'''+RUN+'''/cache_qwen3_14b_awq.sqlite3')
if not db.exists(): db=root/'cache_qwen3_14b_awq.sqlite3'
if db.exists():
    try:
        con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=2)
        result['completed_total']=con.execute("SELECT count(*) FROM translations WHERE status='ok' AND truncated=0").fetchone()[0]
        result['incomplete_records']=con.execute("SELECT count(*) FROM translations WHERE status!='ok' OR truncated!=0").fetchone()[0]
        con.close()
    except sqlite3.Error:
        result['cache_busy']=True
print('[codex-json] '+json.dumps(result),flush=True)
'''
    reply=session.execute(code,timeout=60)
    if reply['status']!='ok' or not reply['values']:
        raise RuntimeError('status_probe_failed')
    return reply['values'][-1]


def backup(session):
    CHECKPOINT.mkdir(parents=True,exist_ok=True)
    previous=count_ok(CHECKPOINT/FILES[0]) if (CHECKPOINT/FILES[0]).exists() else 0
    updated=previous
    for name in FILES:
        try:
            data=session.get_bytes(RUN+'/'+name)
        except Exception:
            continue
        temporary=CHECKPOINT/(name+'.download')
        temporary.write_bytes(data)
        if name.endswith('.sqlite3'):
            if not data.startswith(b'SQLite format 3\x00'):
                temporary.unlink(missing_ok=True)
                raise RuntimeError('invalid_checkpoint_download')
            updated=count_ok(temporary)
            if updated<previous:
                temporary.unlink(missing_ok=True)
                raise RuntimeError('checkpoint_regression')
        temporary.replace(CHECKPOINT/name)
    save(checkpoint_ok=updated,checkpoint_directory=str(CHECKPOINT),checkpoint_saved_at_utc=datetime.now(timezone.utc).isoformat())
    return updated


def launch(session):
    worker=build_worker()
    session.put_bytes(RUN+'/qwen_live_worker.py',worker.read_bytes())
    code='''
import subprocess,sys,json
from pathlib import Path
root=Path('/kaggle/working/'''+RUN+'''')
if '_qwen_health' in globals():
    _qwen_health.close()
    _qwen_health.worker.join(timeout=5)
ip=__import__('IPython').get_ipython()
if ip is not None:
    for name in ('_qwen_health_post_run','_qwen_kaggle_error_hook'):
        hook=getattr(ip,name,None)
        if hook is not None:
            try: ip.events.unregister('post_run_cell',hook)
            except Exception: pass
log=(root/'qwen_live_worker.stdout.log').open('a',encoding='utf-8')
worker=subprocess.Popen([sys.executable,'-u',str(root/'qwen_live_worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('[codex-json] '+json.dumps({'worker_pid':worker.pid,'started':True}),flush=True)
'''
    reply=session.execute(code,timeout=60)
    if reply['status']!='ok' or not reply['values']:
        raise RuntimeError('worker_launch_failed')
    save(phase='worker_starting',worker_pid=reply['values'][-1]['worker_pid'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--ensure-running',action='store_true')
    args=parser.parse_args()
    from report_tiki_gemma_status import process_alive
    if STATE.exists():
        previous=json.loads(STATE.read_text(encoding='utf-8'))
        alive=process_alive(previous.get('controller_pid',0))
        if args.status or (args.ensure_running and alive):
            previous['controller_alive']=alive
            print(json.dumps(previous,ensure_ascii=True));return
        if alive:
            raise SystemExit('An existing controller is still alive.')
    elif args.status:
        print(json.dumps({'phase':'not_started','controller_alive':False}));return
    lock=STATE.with_suffix('.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        save()
        if not CHECKPOINT.exists(): shutil.copytree(INITIAL,CHECKPOINT)
        session=KaggleLiveSession()
        for attempt in range(40):
            if session.connect():break
            save(phase='waiting_for_session',attempt=attempt+1)
            time.sleep(10)
        else:raise RuntimeError('session_not_ready')
        status=remote_status(session)
        if not status.get('worker_alive'):
            existing=status.get('completed_total',0)
            local=count_ok(CHECKPOINT/FILES[0])
            if existing<local:
                save(phase='restoring_checkpoint',checkpoint_ok=local)
                for name in FILES:
                    path=CHECKPOINT/name
                    if path.exists(): session.put_bytes(RUN+'/'+name,path.read_bytes())
            notebook=json.loads((ROOT/'title_translation_qwen3_14b_awq_kaggle_v6.ipynb').read_text(encoding='utf-8'))
            bootstrap='\n\n'.join(''.join(notebook['cells'][i]['source']) for i in (2,3))
            save(phase='bootstrap')
            result=session.execute(bootstrap,timeout=1800,on_event=on_event)
            if result['status']!='ok':raise RuntimeError('bootstrap_failed')
            launch(session)
        last_backup=time.monotonic()
        retries=0; best=status.get('completed_total',0)
        while True:
            status=remote_status(session)
            completed=status.get('completed_total',0)
            health=status.get('health',{})
            save(phase='monitoring',completed_total=completed,worker_alive=status.get('worker_alive'),
                 health=health,live=status.get('live',{}),report=status.get('report',{}))
            if time.monotonic()-last_backup>=120 or not status.get('worker_alive'):
                backup(session);last_backup=time.monotonic()
            if not status.get('worker_alive'):
                if completed>=15863 and status.get('report',{}).get('status')=='all_titles_complete':
                    output='bilingual_zh_vi_cado_title_qwen3_14b_awq.parquet'
                    verify=session.execute("from pathlib import Path;import json,hashlib,pyarrow.parquet as pq; p=Path('/kaggle/working/"+RUN+"/"+output+"');print('[codex-json] '+json.dumps({'rows':pq.ParquetFile(p).metadata.num_rows,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))")
                    if verify['status']!='ok' or not verify['values'] or verify['values'][-1]['rows']!=16348:
                        raise RuntimeError('output_row_count_mismatch')
                    data=session.get_bytes(RUN+'/'+output)
                    import hashlib
                    if hashlib.sha256(data).hexdigest()!=verify['values'][-1]['sha256']:
                        raise RuntimeError('output_download_checksum_mismatch')
                    destination=CHECKPOINT/output;destination.write_bytes(data)
                    save(phase='ready_for_tiki',qwen_1688_completed=True,output_path=str(destination),
                         output_sha256=hashlib.sha256(data).hexdigest(),output_rows=16348,next_dataset='tiki_vi_to_zh')
                    return
                if completed>best:retries=0;best=completed
                retries+=1
                if retries>3:
                    save(phase='paused_worker_error',retry_count=retries);return
                save(phase='retrying_same_worker',retry_count=retries)
                launch(session)
            time.sleep(60)
    except Exception as exc:
        save(phase='paused_controller_error',controller_error_class=type(exc).__name__)
    finally:
        lock.close()


if __name__=='__main__':
    main()
