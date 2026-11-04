
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
import pandas as pd
from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationDelete,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
root=Path(__file__).resolve().parent;os.chdir(root)
out=root/'outputs/hf_title_exports';out.mkdir(parents=True,exist_ok=True)
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom'
items=[
 ('1688_zh_to_vi_gemma4',Path('outputs/translation_only/gemma_1688/bilingual_zh_vi_cado_title_gemma4.parquet'),'complete_with_review_flags'),
 ('1688_zh_to_vi_qwen3_14b_awq',Path('outputs/translation_only/qwen_1688_qa_repaired/qwen_1688_qa_repaired.parquet'),'repair_completed_with_review_flags'),
 ('tiki_vi_to_zh_qwen3_14b_awq',Path('outputs/translation_only/qwen_tiki_qa_repaired/qwen_tiki_qa_repaired.parquet'),'repair_completed_with_review_flags'),
]
gemma=next((Path.home()/'.codex/gemma-tiki-final-verified').rglob('tiki_vi_cado_title_gemma4_vi_zh.parquet'))
d=pd.read_parquet(gemma)
s=d[['product_id','title_vi','title_zh_gemma4','title_zh_gemma4_status','title_zh_gemma4_qa_passed','title_zh_gemma4_qa_flags']].rename(columns={'title_zh_gemma4':'title_zh'}).copy()
s['translation_model']='gemma-4-26b-a4b-it';s['semantic_verified']=False
assert len(s)==13200 and s.title_vi.nunique()==13102
unique=s.drop_duplicates('title_vi');assert unique.title_zh_gemma4_status.eq('ok').sum()==13090
p=out/'tiki_gemma4_source.parquet';s.to_parquet(p,index=False)
items.append(('tiki_vi_to_zh_gemma4',p,'incomplete_12_generation_errors'))
operations=[];manifest=[]
for name,source,status in items:
 frame=pd.read_parquet(source);assert not any('description' in c for c in frame)
 path=out/(name+'.parquet');frame.to_parquet(path,index=False)
 remote=f'data/translated/title_only/{name}/{name}.parquet'
 manifest.append(dict(name=name,path=remote,rows=len(frame),status=status,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 operations.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
readme='''# Title translation exports

Folders and file names identify dataset, translation direction, and model.

| Folder | Direction | Status |
|---|---|---|
| 1688_zh_to_vi_gemma4 | Chinese → Vietnamese | Translation run completed |
| 1688_zh_to_vi_qwen3_14b_awq | Chinese → Vietnamese | Repair run completed |
| tiki_vi_to_zh_gemma4 | Vietnamese → Chinese | 13090/13102 unique titles successful; 12 generation errors retained |
| tiki_vi_to_zh_qwen3_14b_awq | Vietnamese → Chinese | Repair run completed |

Exports contain product IDs, source/translated titles, and translation metadata only. Processing status does not imply human semantic verification. COMETKiwi best-per-title exports are pending.
'''
for filename,text in [('README.md',readme),('manifest.json',json.dumps(manifest,indent=2))]:
 path=out/filename;path.write_text(text,encoding='utf-8');operations.append(CommitOperationAdd(path_in_repo='data/translated/title_only/'+filename,path_or_fileobj=str(path)))
old=[f for f in api.list_repo_files(repo,repo_type='dataset') if f.startswith('data/translated/completed/')]
operations.extend(CommitOperationDelete(path_in_repo=f) for f in old)
c=api.create_commit(repo,repo_type='dataset',operations=operations,commit_message='Organize four title-only translation exports by dataset direction and model')
for item in manifest:
 path=hf_hub_download(repo,item['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN'])
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==item['sha256']
result=dict(revision=c.oid,exports=manifest,renamed_old_paths=old)
(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
