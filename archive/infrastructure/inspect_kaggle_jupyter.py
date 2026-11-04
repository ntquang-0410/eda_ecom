"""Inspect the returned interactive-session connection without printing tokens."""

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
from urllib.parse import urlsplit, parse_qs
import requests
import importlib.util
from kaggle.api.kaggle_api_extended import KaggleApi

api=KaggleApi();api.authenticate()
operation_name=json.loads(Path(r'C:\Users\Lenovo\.codex\qwen-interactive-operation.json').read_text())['operation_name']
with api.build_kaggle_client() as client:
    operation=client.common.operations_client.get_operation(name=operation_name)
result={'done':operation.done}
for key in ('jupyterUrl','tokenizedJupyterUrl'):
    raw=(operation.response or {}).get(key,'')
    url=urlsplit(raw)
    result[key]={'scheme':url.scheme,'host_domain_suffix':'.'.join((url.hostname or '').split('.')[-3:]),
                 'path_segments':[s if s.isdigit() or s in ('lab','tree','proxy','jupyter','notebooks','kernels') else '<segment>' for s in url.path.split('/') if s],
                 'query_keys':list(parse_qs(url.query))}
tokenized=urlsplit((operation.response or {}).get('tokenizedJupyterUrl',''))
token=parse_qs(tokenized.query).get('token',[''])[0]
session=requests.Session()
session.headers['Authorization']='token '+token
base=(operation.response or {}).get('jupyterUrl','').rstrip('/')
result['websocket_library_available']=importlib.util.find_spec('websocket') is not None
try:
    login=session.get((operation.response or {})['tokenizedJupyterUrl'],timeout=20)
    result['session_entry_status']=login.status_code
except Exception as exc:
    result['session_entry_error_class']=type(exc).__name__
for endpoint in ('api/status','api/kernels'):
    try:
        response=session.get(base+'/'+endpoint,timeout=20)
        entry={'http_status':response.status_code}
        if response.status_code==200:
            payload=response.json()
            if endpoint=='api/status':
                entry.update({k:payload.get(k) for k in ('connections','kernels')})
            elif isinstance(payload,list):
                entry['kernel_count']=len(payload)
                entry['execution_states']=[x.get('execution_state') for x in payload]
        result[endpoint]=entry
    except Exception as exc:
        result[endpoint]={'error_class':type(exc).__name__}
print(json.dumps(result))
