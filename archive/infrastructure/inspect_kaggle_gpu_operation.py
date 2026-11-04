"""Poll the already-created T4 session without creating another worker."""

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
from kaggle.api.kaggle_api_extended import KaggleApi

path=Path(r'C:\Users\Lenovo\.codex\qwen-interactive-operation.json')
previous=json.loads(path.read_text(encoding='utf-8'))
api=KaggleApi(); api.authenticate()
with api.build_kaggle_client() as client:
    operation=client.common.operations_client.get_operation(name=previous['operation_name'])
result={'operation_name':operation.name,'done':operation.done,
        'metadata_keys':list(operation.metadata or {}),
        'response_keys':list(operation.response or {})}
if previous.get('kernel_session_id'):
    result['kernel_session_id']=previous['kernel_session_id']
def ids(value):
    if isinstance(value,dict):
        for key, child in value.items():
            if key.lower() in ('kernelsessionid','kernel_session_id','sessionid') and isinstance(child,int):
                result['kernel_session_id']=child
            ids(child)
    elif isinstance(value,list):
        for child in value: ids(child)
ids(operation.metadata); ids(operation.response)
if operation.error:
    result['error_code']=operation.error.code
    message=operation.error.message.lower()
    result['error_category']=next((kind for kind in ('verification','verified','phone','quota','permission','capacity','gpu') if kind in message),'operation_error')
path.write_text(json.dumps(result),encoding='utf-8')
print(json.dumps(result))
