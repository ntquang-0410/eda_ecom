
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
out=Path('outputs/tiki_full_sources');out.mkdir(parents=True,exist_ok=True)
for method,name in [('vi_to_zh','description_pairs_vi_zh.parquet'),('vi_to_zh_context','description_pairs_vi_zh_ctx.parquet'),('vi_to_zh_context','description_products_vi_zh.parquet')]:
 p=hf_hub_download('ntquang0410/zh-vie_ecom',f'data/tiki/descriptions/{method}/{name}',repo_type='dataset',token=os.environ['HF_TOKEN']);d=pd.read_parquet(p)
 (out/(name+'.path')).write_text(p)
 print(json.dumps({'file':name,'rows':len(d),'products':d.product_id.nunique(),'columns':list(d.columns),'types':{c:str(t) for c,t in d.dtypes.items()}}))
