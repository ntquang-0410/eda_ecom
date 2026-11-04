"""Retain the higher-scoring usable translation without a score floor or QA exclusion."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,math,hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
out=ROOT/'outputs/comet_best_all_titles';out.mkdir(parents=True,exist_ok=True)
ops=[];reports=[]
for kind,expected,total,src,dst in [('1688',16348,15863,'title_zh','title_vi'),('tiki',13200,13102,'title_vi','title_zh')]:
    original=next((Path.home()/'.codex'/f'comet-{kind}-final').rglob('*best_cometkiwi.parquet'))
    d=pd.read_parquet(original)
    if kind=='1688':
        repair=pd.read_parquet(json.loads(Path('qwen1688_repair_verified.json').read_text())['full_path']).drop_duplicates(src).set_index(src)
        d['g_generation_ok']=True
        d['q_generation_ok']=[bool(repair.loc[s,'qa_repair_accepted']) or (repair.loc[s,'generation_status']=='ok' and not bool(repair.loc[s,'generation_truncated'])) for s in d[src]]
        gscore='cometkiwi_gemma4';qscore='cometkiwi_qwen3_14b_awq'
    else:
        gemma=pd.read_parquet(next((Path.home()/'.codex/gemma-tiki-final-verified').rglob('tiki_vi_cado_title_gemma4_vi_zh.parquet'))).drop_duplicates(src).set_index(src)
        qwen=pd.read_parquet('title_translation_tiki_qwen3_14b_awq_qa_repaired/qwen_tiki_qa_repaired.parquet').drop_duplicates(src).set_index(src)
        d['g_generation_ok']=[gemma.loc[s,'title_zh_gemma4_status']=='ok' for s in d[src]]
        d['q_generation_ok']=[qwen.loc[s,'generation_status']=='ok' and not bool(qwen.loc[s,'generation_truncated']) for s in d[src]]
        gscore='gemma4_score';qscore='qwen3_14b_awq_score'
    selected=[];counts=Counter();fallback=0
    for r in d.to_dict('records'):
        candidates=[];all_nonempty=[]
        for name,scorecol,gen in [('gemma4',gscore,'g_generation_ok'),('qwen3_14b_awq',qscore,'q_generation_ok')]:
            text=r[name];score=r[scorecol]
            if isinstance(text,str) and text.strip() and pd.notna(score) and math.isfinite(float(score)):
                value=(float(score),name=='gemma4',name,text)
                all_nonempty.append(value)
                if r[gen]:candidates.append(value)
        if not candidates:
            candidates=all_nonempty;fallback+=1
        assert candidates,'no_nonempty_candidate'
        score,_,winner,text=max(candidates)
        counts[winner]+=1
        item={'product_id':r['product_id'],src:r[src],dst:text,'gemma4_translation':r['gemma4'],'qwen3_14b_awq_translation':r['qwen3_14b_awq'],'cometkiwi_gemma4':r[gscore],'cometkiwi_qwen3_14b_awq':r[qscore],'selected_model':winner,'selected_cometkiwi_score':score,'selection_policy':'higher_cometkiwi_no_floor_qa_advisory_only','selection_reason':'higher_score' if len(candidates)==2 else 'only_usable_candidate','gemma4_generation_ok':r['g_generation_ok'],'qwen3_14b_awq_generation_ok':r['q_generation_ok'],'semantic_verified':False}
        if kind=='1688':
            item['gemma4_qa']=r['gemma4_qa'];item['qwen3_14b_awq_qa']=r['qwen3_14b_awq_qa']
        else:
            item['gemma4_qa_flags']=r['gemma4_qa_flags'];item['qwen3_14b_awq_qa_flags']=r['qwen3_14b_awq_qa_flags']
        selected.append(item)
    result=pd.DataFrame(selected)
    assert len(result)==expected and result[src].nunique()==total
    assert result.product_id.equals(d.product_id) and result[src].equals(d[src])
    assert result[dst].notna().all() and result[dst].str.strip().ne('').all()
    name=f'{kind}_{"zh_to_vi" if kind=="1688" else "vi_to_zh"}_best_cometkiwi';folder=out/name;folder.mkdir(exist_ok=True)
    p=folder/(name+'.parquet');result.to_parquet(p,index=False);pd.testing.assert_frame_equal(result,pd.read_parquet(p))
    remote=f'data/translated/title_only/{name}/{name}.parquet'
    unique=result.drop_duplicates(src)
    report={'dataset':kind,'rows':expected,'unique_titles':total,'blank_selected':0,'selection_counts_unique':dict(Counter(unique.selected_model)),'selection_counts_product_rows':dict(counts),'fallback_generation_error_rows':fallback,'score_floor':None,'qa_used_as_exclusion':False,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_scored_output_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'semantic_verified':False,'path':remote}
    reportfile=folder/'selection_report.json';reportfile.write_text(json.dumps(report,indent=2));reports.append(report)
    ops.extend([CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)),CommitOperationAdd(path_in_repo=f'data/translated/title_only/{name}/selection_report.json',path_or_fileobj=str(reportfile))])
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom'
c=api.create_commit(repo,repo_type='dataset',operations=ops,commit_message='Select higher COMETKiwi translation for every title without score-floor exclusion')
for r in reports:
    p=hf_hub_download(repo,r['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN'])
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==r['sha256']
final={'status':'uploaded_verified','revision':c.oid,'reports':reports};(out/'upload_result.json').write_text(json.dumps(final,indent=2));print(json.dumps(final))
