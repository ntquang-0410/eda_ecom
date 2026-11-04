"""Inspect worker log readiness with a short, explicit server-side wait."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleEnv

api=KaggleApi();api.authenticate()
result={}
try:
    with api.build_kaggle_client() as client:
        http=client._http_client
        http._init_session()
        base=http._endpoint if http._env==KaggleEnv.PROD else http._endpoint+'/api'
        headers=dict(http._session.headers)
        headers['Accept']='text/event-stream, */*'
        headers.pop('Content-Type',None)
        response=http._session.get(base+'/v1/kernels/logs/stream/nadsa091/qwen3-14b-awq-kaggle-v6',
            params={'versionLabel':'v17','waitForLogsUrlSeconds':5},stream=True,headers=headers,
            auth=http._session.auth,timeout=(10,15))
        result.update(http_status=response.status_code,content_type=response.headers.get('Content-Type'))
        if response.status_code!=200:
            payload=response.json()
            result['response_keys']=list(payload) if isinstance(payload,dict) else []
            text=json.dumps(payload).lower()
            result['categories']=[x for x in ('queued','not ready','not found','not available','timeout','permission','unavailable') if x in text]
        else:
            result['log_connection_opened']=True
        response.close()
except Exception as exc:
    result['exception_class']=type(exc).__name__
print(json.dumps(result))
