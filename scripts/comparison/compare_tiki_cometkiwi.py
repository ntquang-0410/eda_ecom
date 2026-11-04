"""VI->ZH comparison. The shared scoring adapter's title_zh field contains VI source."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import argparse,json,hashlib,sqlite3
from pathlib import Path
from collections import Counter
import pandas as pd
import compare_1688_cometkiwi as common

def prepare(a):
    g=pd.read_parquet(a.gemma);q=pd.read_parquet(a.qwen)
    assert len(g)==len(q)==13200
    for c in ('product_id','title_vi'):assert g[c].equals(q[c]),'alignment_mismatch'
    u=g[['title_vi']].drop_duplicates().copy();assert len(u)==13102
    gg=g.drop_duplicates('title_vi').set_index('title_vi');qq=q.drop_duplicates('title_vi').set_index('title_vi')
    u['gemma4']=u.title_vi.map(gg.title_zh_gemma4).fillna('')
    u['qwen3_14b_awq']=u.title_vi.map(qq.title_zh).fillna('')
    u['gemma4_eligible']=[bool(gg.loc[s,'title_zh_gemma4_qa_passed']) and gg.loc[s,'title_zh_gemma4_status']=='ok' and bool(mt.strip()) for s,mt in zip(u.title_vi,u.gemma4)]
    u['qwen3_14b_awq_eligible']=[qq.loc[s,'qa_status']=='pass' and qq.loc[s,'generation_status']=='ok' and not bool(qq.loc[s,'generation_truncated']) and bool(mt.strip()) for s,mt in zip(u.title_vi,u.qwen3_14b_awq)]
    u['gemma4_qa_flags']=u.title_vi.map(gg.title_zh_gemma4_qa_flags)
    u['qwen3_14b_awq_qa_flags']=u.title_vi.map(qq.qa_flags)
    # Shared scorer treats this column only as src text, never as language detection.
    u['title_zh']=u.title_vi
    a.workdir.mkdir(exist_ok=True)
    u.to_parquet(a.workdir/'comparison_inputs.parquet',index=False)
    g[['product_id','title_vi']].to_parquet(a.workdir/'source_rows.parquet',index=False)
    m=dict(direction='vi_to_zh',score_adapter_source_column='title_zh contains original Vietnamese',model=common.MODEL,revision=common.REVISION,score_floor=.72,expected_rows=13200,expected_unique=13102,gemma_sha256=common.sha(a.gemma),qwen_sha256=common.sha(a.qwen),semantic_verified=False)
    (a.workdir/'comparison_manifest.json').write_text(json.dumps(m,indent=2))

def select(a):
    w=a.workdir;m=json.loads((w/'comparison_manifest.json').read_text());p=json.loads((w/'scorer_provenance.json').read_text())
    assert p['comparison_manifest_sha256']==common.sha(w/'comparison_manifest.json') and p['revision']==common.REVISION
    u=pd.read_parquet(w/'comparison_inputs.parquet')
    with sqlite3.connect(w/'cometkiwi_scores.sqlite3') as db:scores=dict(db.execute('select key,score from scores'))
    selected=[];counts=Counter()
    for row in u.to_dict('records'):
        options=[]
        for name in ('gemma4','qwen3_14b_awq'):
            mt=row[name];s=scores.get(common.score_key(row['title_vi'],mt)) if mt.strip() else None
            row[name+'_score']=s
            if row[name+'_eligible'] and s is not None and s>=.72:options.append((s,name=='gemma4',name))
        name=max(options)[2] if options else 'hold';counts[name]+=1
        row['title_zh_selected']=row[name] if options else None;row['selected_model']=name;row['review_required']=not bool(options)
        selected.append(row)
    out=pd.read_parquet(w/'source_rows.parquet').merge(pd.DataFrame(selected).drop(columns='title_zh'),on='title_vi',how='left',sort=False,validate='many_to_one')
    out['title_zh']=out.pop('title_zh_selected');out['semantic_verified']=False
    assert len(out)==13200 and out.title_vi.nunique()==13102
    f=w/'tiki_title_best_cometkiwi.parquet';out.to_parquet(f,index=False)
    (w/'comparison_report.json').write_text(json.dumps(dict(rows=len(out),unique_titles=13102,selection_counts=dict(counts),sha256=common.sha(f),semantic_verified=False,provenance=p),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score','select']);p.add_argument('--workdir',type=Path,required=True);p.add_argument('--gemma',type=Path);p.add_argument('--qwen',type=Path);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--gpus',type=int,default=1);a=p.parse_args()
    {'prepare':prepare,'score':common.score,'select':select}[a.stage](a)
