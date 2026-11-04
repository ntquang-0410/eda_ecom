
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,re,json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha;files=api.list_repo_files(repo,repo_type='dataset',revision=head);ops=[];fixed=[]
for ds in ['1688','tiki']:
 for model in ['gemma4','qwen3_14b_awq','best_cometkiwi']:
  folder=f'data/{ds}/title_translations/{model}/';parquets=[f for f in files if f.startswith(folder) and f.endswith('.parquet')];readme=folder+'README.md'
  if len(parquets)!=1 or readme not in files:continue
  info=api.get_paths_info(repo,parquets,repo_type='dataset',revision=head)[0];sha=info.lfs.sha256 if info.lfs else None
  p=Path(hf_hub_download(repo,readme,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));s=p.read_text(encoding='utf8');new=re.sub(r'(SHA256:\s*`)[0-9a-f]{64}(`)',lambda m:m.group(1)+sha+m.group(2),s) if sha else s
  if new!=s:
   dest=Path('outputs/readme_checksum_fixes')/readme;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(new,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=readme,path_or_fileobj=str(dest)));fixed.append(readme)
for item in ['independent','context']:
 folder='data/tiki/full_dataset/';parquet=folder+f'tiki_full_attributes_{item}_vi_zh.parquet';readme=folder+f'tiki_full_attributes_{item}_vi_zh.README.md'
 info=api.get_paths_info(repo,[parquet],repo_type='dataset',revision=head)[0];sha=info.lfs.sha256 if info.lfs else None;p=Path(hf_hub_download(repo,readme,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));s=p.read_text(encoding='utf8');new=re.sub(r'(SHA256:\s*`)[0-9a-f]{64}(`)',lambda m:m.group(1)+sha+m.group(2),s) if sha else s
 if new!=s:
  dest=Path('outputs/readme_checksum_fixes')/readme;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(new,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=readme,path_or_fileobj=str(dest)));fixed.append(readme)
if ops:
 c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Refresh README checksums after title cleanup');print(json.dumps({'revision':c.oid,'readmes_fixed':fixed}))
else:print(json.dumps({'status':'already_current'}))
