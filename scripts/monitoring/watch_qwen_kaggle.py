"""Background read-only observer; never launches workers or calls translation APIs."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

root=Path(__file__).parent
active=json.loads((root/'qwen_active_run.json').read_text(encoding='utf-8'))
output=Path(active['status_file'])
state_file=output.with_name(output.stem+'-watcher.json')
deadline=time.monotonic()+12*3600
checks=0
while time.monotonic()<deadline:
    checks+=1
    try:
        result=subprocess.run([sys.executable,str(root/'inspect_qwen_kaggle.py'),'--kernel',active['kernel'],
            '--seconds','20','--output',str(output)],capture_output=True,text=True,timeout=50)
        observation=json.loads(output.read_text(encoding='utf-8')) if result.returncode==0 else {}
        state={'kernel':active['kernel'],'checked_at_utc':datetime.now(timezone.utc).isoformat(),
               'checks':checks,'observer_exit_code':result.returncode,'stage':observation.get('stage'),
               'health_state':observation.get('health',{}).get('state'),
               'kernel_status':observation.get('kernel_status'),
               'completed_this_session':observation.get('completed_this_session')}
        terminal=state['health_state'] in ('error','completed') or any(
            value in str(state['kernel_status']) for value in ('ERROR','COMPLETE','CANCEL'))
    except Exception as exc:
        state={'kernel':active['kernel'],'checked_at_utc':datetime.now(timezone.utc).isoformat(),
               'checks':checks,'observer_error_class':type(exc).__name__}
        terminal=False
    temporary=state_file.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state),encoding='utf-8')
    temporary.replace(state_file)
    print(json.dumps(state),flush=True)
    if terminal:
        break
    time.sleep(90)
