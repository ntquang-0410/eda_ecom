"""Close only the CPU diagnostic session created in this task."""

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
from urllib.parse import urlsplit
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest

path=Path(r'C:\Users\Lenovo\.codex\qwen-interactive-operation.json')
saved=json.loads(path.read_text())
api=KaggleApi();api.authenticate()
with api.build_kaggle_client() as client:
    operation=client.common.operations_client.get_operation(name=saved['operation_name'])
    url=(operation.response or {}).get('jupyterUrl','')
    observed=[int(segment) for segment in urlsplit(url).path.split('/') if segment.isdigit()]
    operation_suffix=saved['operation_name'].rsplit('/',1)[-1]
    if len(observed)!=1 or str(observed[0])!=operation_suffix:
        raise RuntimeError('Cannot verify the diagnostic session identity; no cancellation sent.')
    request=ApiCancelKernelSessionRequest()
    request.kernel_session_id=observed[0]
    client.kernels.kernels_api_client.cancel_kernel_session(request)
result={'diagnostic_session_id':observed[0],'cancellation_requested':True,
        'scope':'new_cpu_diagnostic_session_only'}
saved.update(result)
path.write_text(json.dumps(saved),encoding='utf-8')
print(json.dumps(result))
