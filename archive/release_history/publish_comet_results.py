
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,hashlib,sqlite3
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
import compare_1688_cometkiwi as common
disable_progress_bars()
root=Path(__file__).resolve().parent;os.chdir(root)
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom'
out=root/'outputs/comet_final_verified';out.mkdir(exist_ok=True,parents=True)
ops=[];manifest=[]
for kind,rows,unique,sourcecol,destcol in [('1688',16348,15863,'title_zh','title_vi'),('tiki',13200,13102,'title_vi','title_zh')]:
 base=Path.home()/'.codex'/f'comet-{kind}-final'
 report=json.loads(next(base.rglob('comparison_report.json')).read_text())
 file=next(base.rglob('*best_cometkiwi.parquet'));data=pd.read_parquet(file)
 assert common.sha(file)==report.get('output_sha256',report.get('sha256'))
 assert len(data)==rows and data[sourcecol].nunique()==unique
 source=pd.read_parquet(root/('cometkiwi_1688_after_qa_repair' if kind=='1688' else 'cometkiwi_tiki_comparison')/'source_rows.parquet')
 assert data['product_id'].equals(source['product_id']) and data[sourcecol].equals(source[sourcecol])
 provenance=json.loads(next(base.rglob('scorer_provenance.json')).read_text())
 remote_manifest=next(base.rglob('comparison_manifest.json'))
 assert provenance['model']==common.MODEL and provenance['revision']==common.REVISION
 assert provenance['comparison_manifest_sha256']==common.sha(remote_manifest)
 if kind=='1688':
  for r in data.drop_duplicates(sourcecol).to_dict('records'):
   choice=common.choose(r['gemma4'],r['qwen3_14b_awq'],r['cometkiwi_gemma4'],r['cometkiwi_qwen3_14b_awq'],json.loads(r['gemma4_qa']),json.loads(r['qwen3_14b_awq_qa']))
   assert choice['model']==r['title_vi_selection_source']
  keep=['product_id','title_zh','title_vi','gemma4','qwen3_14b_awq','gemma4_qa','qwen3_14b_awq_qa','gemma4_candidate_stage','qwen3_14b_awq_candidate_stage','cometkiwi_gemma4','cometkiwi_qwen3_14b_awq','title_vi_selection_source','title_vi_selection_decision','title_vi_selected_cometkiwi_score','review_required','semantic_verified','cometkiwi_model','cometkiwi_revision']
  slim=data[keep].copy()
 else:
  for r in data.drop_duplicates(sourcecol).to_dict('records'):
   eligible=[(r[n+'_score'],n=='gemma4',n) for n in ['gemma4','qwen3_14b_awq'] if r[n+'_eligible'] and pd.notna(r[n+'_score']) and r[n+'_score']>=.72]
   assert r['selected_model']==(max(eligible)[2] if eligible else 'hold')
  slim=data.copy();slim['cometkiwi_model']=common.MODEL;slim['cometkiwi_revision']=common.REVISION
 assert slim[destcol].isna().equals(slim['review_required'])
 assert not any('description' in c for c in slim)
 name=f'{kind}_{"zh_to_vi" if kind=="1688" else "vi_to_zh"}_best_cometkiwi'
 folder=out/name;folder.mkdir(exist_ok=True);path=folder/(name+'.parquet');slim.to_parquet(path,index=False)
 pd.testing.assert_frame_equal(slim,pd.read_parquet(path))
 remote=f'data/translated/title_only/{name}/{name}.parquet'
 entry=dict(dataset=kind,path=remote,rows=rows,unique_titles=unique,sha256=common.sha(path),selection_counts=report['selection_counts'],score_floor=.72,semantic_verified=False)
 manifest.append(entry);ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
 metadata=folder/'selection_report.json';metadata.write_text(json.dumps({**report,'published_export':entry,'provenance':provenance},indent=2))
 ops.append(CommitOperationAdd(path_in_repo=f'data/translated/title_only/{name}/selection_report.json',path_or_fileobj=str(metadata)))
c=api.create_commit(repo,repo_type='dataset',operations=ops,commit_message='Add verified COMETKiwi title selection exports for 1688 and Tiki')
for entry in manifest:
 p=hf_hub_download(repo,entry['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN'])
 assert common.sha(p)==entry['sha256']
result=dict(status='uploaded_verified',revision=c.oid,exports=manifest)
(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
