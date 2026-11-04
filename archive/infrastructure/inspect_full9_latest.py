
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json
from huggingface_hub import HfApi,hf_hub_download
import pandas as pd
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';i=api.repo_info(repo,repo_type='dataset');fs=api.list_repo_files(repo,repo_type='dataset',revision=i.sha);files=[f for f in fs if 'description_products_' in f and f.endswith('.parquet')];print(json.dumps({'revision':i.sha,'files':files}))
for f in files:
 p=hf_hub_download(repo,f,repo_type='dataset',revision=i.sha,token=os.environ['HF_TOKEN']);d=pd.read_parquet(p);print(json.dumps({'path':f,'rows':len(d),'columns':list(d.columns),'desc_nguon_dich_present':'desc_nguon_dich' in d.columns}))
