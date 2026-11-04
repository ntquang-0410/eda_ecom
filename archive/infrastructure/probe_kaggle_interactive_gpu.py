"""Request a T4 session and retain only safe operation/session identifiers."""

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
from kagglesdk.kernels.types.kernels_api_service import ApiCreateKernelSessionRequest

api = KaggleApi()
api.authenticate()
def collect_ids(value, result):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in ('kernelsessionid','kernel_session_id','sessionid') and isinstance(child,int):
                result['kernel_session_id'] = child
            collect_ids(child, result)
    elif isinstance(value, list):
        for child in value:
            collect_ids(child, result)
request = ApiCreateKernelSessionRequest()
request.slug = 'nadsa091/qwen3-14b-awq-kaggle-v6'
request.language = 'python'
request.kernel_type = 'notebook'
request.machine_shape = 'NvidiaTeslaT4'
request.enable_internet = False
try:
    with api.build_kaggle_client() as client:
        operation = client.kernels.kernels_api_client.create_kernel_session(request)
    result = {'operation_name':operation.name, 'done':operation.done,
              'metadata_type':type(operation.metadata).__name__,
              'response_type':type(operation.response).__name__}
    if isinstance(operation.metadata, dict):
        result['metadata_keys'] = list(operation.metadata)
        collect_ids(operation.metadata, result)
    if isinstance(operation.response, dict):
        result['response_keys'] = list(operation.response)
        collect_ids(operation.response, result)
    if operation.error:
        result['error_code'] = operation.error.code
        message = operation.error.message.lower()
        result['error_category'] = next((kind for kind in ('verification','phone','quota','permission','capacity') if kind in message),'operation_error')
except Exception as exc:
    response = getattr(exc,'response',None)
    message = str(exc).lower()
    result = {'exception_class':type(exc).__name__, 'http_status':getattr(response,'status_code',None),
              'error_category':next((kind for kind in ('verification','verified','phone','quota','permission','capacity','unsupported') if kind in message),'request_failed')}
Path(r'C:\Users\Lenovo\.codex\qwen-interactive-operation.json').write_text(json.dumps(result),encoding='utf-8')
print(json.dumps(result))
