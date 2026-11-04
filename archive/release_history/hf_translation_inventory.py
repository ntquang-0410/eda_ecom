
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json
from huggingface_hub import HfApi
api=HfApi(token=os.environ['HF_TOKEN'])
print(json.dumps([f for f in api.list_repo_files('ntquang0410/zh-vie_ecom',repo_type='dataset') if f.startswith('data/translated/completed/')]))
