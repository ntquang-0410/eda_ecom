
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars();p=Path('outputs/translation_only/qwen_1688_qa_repaired/qwen_1688_qa_repaired.parquet');remote='data/translated/completed/qwen_1688_qa_repaired/qwen_1688_qa_repaired.parquet';api=HfApi(token=os.environ['HF_TOKEN']);c=api.upload_file(path_or_fileobj=str(p),path_in_repo=remote,repo_id='ntquang0410/zh-vie_ecom',repo_type='dataset',commit_message='Add title-only Qwen 1688 after bounded QA repair');f=hf_hub_download('ntquang0410/zh-vie_ecom',remote,repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']);assert hashlib.sha256(Path(f).read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest();Path('outputs/translation_only/qwen_1688_qa_repaired/upload_result.json').write_text(json.dumps({'revision':c.oid,'path':remote,'verified':True}));print('qwen1688_uploaded_verified')
