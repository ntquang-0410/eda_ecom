
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
from pathlib import Path
import os,json,re
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
repo='ntquang0410/zh-vie_ecom';api=HfApi(token=os.environ['HF_TOKEN']);head=api.repo_info(repo,repo_type='dataset').sha;r=json.loads(Path('outputs/title_review_labels/upload_result.json').read_text());out=Path('outputs/title_review_labels/doc_fixes');ops=[]
def get(p):return Path(hf_hub_download(repo,p,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN'])).read_text(encoding='utf8')
def put(p,s):
 f=out/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(s,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=p,path_or_fileobj=str(f)))
m=json.loads(get('manifest.json'))
for summary in r['summaries']:
 ds=summary['dataset'];best=f'data/{ds}/title_translations/best_cometkiwi/';name=('1688_zh_to_vi_best_cometkiwi.parquet' if ds=='1688' else 'tiki_vi_to_zh_best_cometkiwi.parquet')
 for x in m.get('title_exports',[]):
  if x.get('path')==best+name:x['sha256']=summary['best_sha256']
 p=best+'selection_report.json';z=json.loads(get(p));z['sha256']=summary['best_sha256'];z['title_review_indicator']=dict(flagged_unique=summary['flagged_unique'],note='labels are screening indicators, not semantic verdicts');put(p,json.dumps(z,indent=2))
 p=best+'README.md';s=get(p);s=re.sub(r'(SHA256:\s*`)[0-9a-f]{64}(`)',lambda v:v.group(1)+summary['best_sha256']+v.group(2),s);put(p,s)
 for x in summary['full_exports']:
  p=x['path'].replace('.parquet','.README.md');s=get(p).replace('đúng 9 cột','10 cột');s=re.sub(r'(SHA256:\s*`)[0-9a-f]{64}(`)',lambda v:v.group(1)+x['sha256']+v.group(2),s);s+='\n\nCột thứ 10 `title_van_de` là nhãn gợi ý rà soát; xem README bộ full để biết quy tắc và giới hạn.\n';put(p,s)
put('manifest.json',json.dumps(m,indent=2));c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Update schema documentation and checksums for title review labels');print(json.dumps({'revision':c.oid,'documentation_files':len(ops)}))
