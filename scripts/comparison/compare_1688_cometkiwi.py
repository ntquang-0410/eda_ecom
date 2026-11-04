"""Reference-free Gemma/Qwen comparison, with the existing technical QA gate."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd
import title_translation_pipeline as quality

MODEL='Unbabel/wmt22-cometkiwi-da'
REVISION='1ad785194e391eebc6c53e2d0776cada8f83179a'
FLOOR=0.72
POLICY='technical-gate-cometkiwi-title-floor072-v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def score_key(source,candidate):
    return hashlib.sha256(json.dumps([MODEL,REVISION,source,candidate],ensure_ascii=False).encode()).hexdigest()

def choose(gemma,qwen,gemma_score,qwen_score,gemma_qa,qwen_qa,floor=FLOOR):
    scores={'gemma4':float(gemma_score),'qwen3_14b_awq':float(qwen_score)}
    if not all(math.isfinite(s) for s in scores.values()):raise ValueError('nonfinite_score')
    values={'gemma4':gemma,'qwen3_14b_awq':qwen}
    checks={'gemma4':gemma_qa,'qwen3_14b_awq':qwen_qa}
    valid=[name for name in scores if checks[name]['status']=='pass' and scores[name]>=floor]
    if not valid:return {'selected':None,'model':'hold','decision':'neither_candidate_passes_technical_and_score_floor'}
    # Exact ties are reproducible and explicitly recorded; no global winner assumed.
    best=max(valid,key=lambda name:(scores[name],name=='gemma4'))
    decision='single_eligible_candidate' if len(valid)==1 else ('equal_score_tie_gemma4' if scores['gemma4']==scores['qwen3_14b_awq'] else 'higher_cometkiwi_score')
    return {'selected':values[best],'model':best,'decision':decision}

def prepare(args):
    if sha(args.source)!=quality.DATASET_SHA256:raise ValueError('source_hash_mismatch')
    source=pd.read_parquet(args.source)
    quality.validate_input_frame(source)
    gemma=pd.read_parquet(args.gemma);qwen=pd.read_parquet(args.qwen)
    for frame in (gemma,qwen):
        if len(frame)!=len(source) or not frame['product_id'].equals(source['product_id']) or not frame['title_zh'].equals(source['title_zh']):
            raise ValueError('candidate_source_alignment_mismatch')
    columns={'gemma4':'title_vi_candidate_gemma4','qwen3_14b_awq':'title_vi_candidate_qwen3_14b_awq'}
    unique=source[['title_zh']].drop_duplicates().copy()
    generation_errors={}
    for name,frame in [('gemma4',gemma),('qwen3_14b_awq',qwen)]:
        col=columns[name]
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq('').any():raise ValueError('candidate_incomplete')
        if frame.groupby('title_zh')[col].nunique().gt(1).any():raise ValueError('duplicate_candidate_disagreement')
        by_source=frame.drop_duplicates('title_zh').set_index('title_zh')
        unique[name]=unique['title_zh'].map(by_source[col])
        checks=[];error_count=0
        for zh,mt in unique[['title_zh',name]].itertuples(index=False,name=None):
            record=by_source.loc[zh]
            failed=str(record.get('generation_status','ok'))!='ok'
            truncated=bool(record.get('generation_truncated',False))
            qa=quality.check_translation(zh,mt,generation_truncated=truncated)
            if failed:
                error_count+=1;qa['status']='fail'
                qa['hard_flags']=sorted(set(qa['hard_flags']+['generation_error']))
            checks.append(json.dumps(qa,ensure_ascii=False,sort_keys=True))
        unique[name+'_qa']=checks
        unique[name+'_candidate_stage']=[
            'technical_qa_repair' if bool(by_source.loc[zh].get('qa_repair_accepted',False))
            else str(by_source.loc[zh].get('title_vi_gemma4_generation_prompt_version','original_translation'))
            for zh in unique['title_zh']]
        generation_errors[name]=error_count
    args.workdir.mkdir(parents=True,exist_ok=True)
    manifest={'model':MODEL,'revision':REVISION,'score_floor':FLOOR,'selection_policy':POLICY,
              'qa_version':quality.QA_VERSION,'qa_source_sha256':sha(Path(quality.__file__)),
              'retained_generation_errors_unique':generation_errors,
              'source_sha256':sha(args.source),'gemma_sha256':sha(args.gemma),'qwen_sha256':sha(args.qwen),
              'expected_unique':quality.EXPECTED_UNIQUE_TITLES,'expected_rows':quality.EXPECTED_ROWS}
    path=args.workdir/'comparison_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('comparison_inputs_changed')
    path.write_text(json.dumps(manifest,indent=2))
    unique.to_parquet(args.workdir/'comparison_inputs.parquet',index=False)
    source.to_parquet(args.workdir/'source_rows.parquet',index=False)
    return manifest

def open_scores(workdir):
    db=sqlite3.connect(workdir/'cometkiwi_scores.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS scores(key TEXT PRIMARY KEY,score REAL NOT NULL)');db.commit()
    return db

def score(args):
    from huggingface_hub import snapshot_download
    from comet import load_from_checkpoint
    work=args.workdir
    m=json.loads((work/'comparison_manifest.json').read_text())
    if m['model']!=MODEL or m['revision']!=REVISION:raise ValueError('scorer_manifest_mismatch')
    model_dir=Path(snapshot_download(MODEL,revision=REVISION))
    checkpoint=model_dir/'checkpoints/model.ckpt'
    if not checkpoint.is_file():raise FileNotFoundError('cometkiwi_checkpoint_missing')
    model=load_from_checkpoint(str(checkpoint));model.eval()
    frame=pd.read_parquet(work/'comparison_inputs.parquet')
    telemetry=work/'comparison_status.json'
    provenance={'model':MODEL,'revision':REVISION,'checkpoint_sha256':sha(checkpoint),
                'unbabel_comet_version':version('unbabel-comet'),'torch_version':version('torch'),
                'comparison_manifest_sha256':sha(work/'comparison_manifest.json')}
    provenance_path=work/'scorer_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text())!=provenance:
        raise ValueError('scorer_environment_changed')
    provenance_path.write_text(json.dumps(provenance,indent=2))
    with closing(open_scores(work)) as db:
        pairs={score_key(src,mt):(src,mt) for src,g,q in frame[['title_zh','gemma4','qwen3_14b_awq']].itertuples(index=False,name=None) for mt in (g,q) if isinstance(mt,str) and mt.strip()}
        cached=dict(db.execute('select key,score from scores'))
        pending=[(k,v) for k,v in pairs.items() if k not in cached]
        for offset in range(0,len(pending),256):
            chunk=pending[offset:offset+256]
            result=model.predict([{'src':src,'mt':mt} for _,(src,mt) in chunk],batch_size=args.batch_size,gpus=args.gpus,num_workers=0,progress_bar=False)
            if len(result.scores)!=len(chunk):raise ValueError('score_count_mismatch')
            values=[(k,float(s)) for (k,_),s in zip(chunk,result.scores)]
            if not all(math.isfinite(v) for _,v in values):raise ValueError('nonfinite_score')
            db.executemany('insert or replace into scores values(?,?)',values);db.commit()
            status={'phase':'scoring','completed_pairs':len(cached)+min(offset+256,len(pending)),'total_pairs':len(pairs),'heartbeat_at_utc':datetime.now(timezone.utc).isoformat()}
            telemetry.write_text(json.dumps(status));print(json.dumps(status),flush=True)

def select(args):
    work=args.workdir;m=json.loads((work/'comparison_manifest.json').read_text())
    provenance=json.loads((work/'scorer_provenance.json').read_text())
    if provenance['revision']!=REVISION or provenance['comparison_manifest_sha256']!=sha(work/'comparison_manifest.json'):
        raise ValueError('scorer_provenance_mismatch')
    frame=pd.read_parquet(work/'comparison_inputs.parquet')
    with closing(open_scores(work)) as db:scores=dict(db.execute('select key,score from scores'))
    gs=[];qs=[];choices=[]
    for src,g,q,gqa,qqa in frame[['title_zh','gemma4','qwen3_14b_awq','gemma4_qa','qwen3_14b_awq_qa']].itertuples(index=False,name=None):
        a=scores[score_key(src,g)];b=scores[score_key(src,q)]
        gs.append(a);qs.append(b);choices.append(choose(g,q,a,b,json.loads(gqa),json.loads(qqa)))
    frame['cometkiwi_gemma4']=gs;frame['cometkiwi_qwen3_14b_awq']=qs
    frame['title_vi_selected']=[r['selected'] for r in choices]
    frame['title_vi_selection_source']=[r['model'] for r in choices]
    frame['title_vi_selection_decision']=[r['decision'] for r in choices]
    source=pd.read_parquet(work/'source_rows.parquet')
    output=source.merge(frame,on='title_zh',how='left',sort=False,validate='many_to_one')
    output['title_vi_original']=output['title_vi'];output['title_vi']=output['title_vi_selected']
    if 'title_vi_cleaned' in output:
        output['title_vi_cleaned_original']=output['title_vi_cleaned']
        output['title_vi_cleaned']=output['title_vi_selected']
    output['review_required']=output['title_vi_selected'].isna()
    output['title_vi_selected_cometkiwi_score']=[
        a if name=='gemma4' else b if name=='qwen3_14b_awq' else None
        for name,a,b in zip(output['title_vi_selection_source'],output['cometkiwi_gemma4'],output['cometkiwi_qwen3_14b_awq'])]
    output['semantic_verified']=False;output['cometkiwi_model']=MODEL;output['cometkiwi_revision']=REVISION
    if len(output)!=m['expected_rows'] or output['title_zh'].nunique()!=m['expected_unique'] or not output['product_id'].equals(source['product_id']):
        raise ValueError('output_alignment_mismatch')
    name=work/'bilingual_zh_vi_cado_title_best_cometkiwi.parquet'
    output.to_parquet(name,index=False)
    report={'unique_titles':len(frame),'rows':len(output),'selected_unique':int(frame['title_vi_selected'].notna().sum()),
            'held_unique':int(frame['title_vi_selected'].isna().sum()),'selection_counts':dict(Counter(r['model'] for r in choices)),
            'mean_cometkiwi_gemma4':sum(gs)/len(gs),'mean_cometkiwi_qwen3_14b_awq':sum(qs)/len(qs),
            'gemma_higher_score_unique':sum(a>b for a,b in zip(gs,qs)),
            'qwen_higher_score_unique':sum(b>a for a,b in zip(gs,qs)),
            'equal_score_unique':sum(a==b for a,b in zip(gs,qs)),
            'selection_policy':POLICY,'score_floor':FLOOR,'semantic_verified':False,
            'output_sha256':sha(name),'provenance':provenance}
    import numpy as np
    delta=np.asarray(qs)-np.asarray(gs);rng=np.random.default_rng(42)
    means=np.concatenate([rng.choice(delta,size=(100,len(delta)),replace=True).mean(axis=1) for _ in range(10)])
    report['paired_mean_qwen_minus_gemma']=float(delta.mean())
    report['paired_bootstrap_95pct_ci']=[float(v) for v in np.quantile(means,[.025,.975])]
    report['bootstrap_replicates']=1000
    (work/'comparison_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score','select'])
    p.add_argument('--workdir',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--gemma',type=Path);p.add_argument('--qwen',type=Path)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--gpus',type=int,default=1)
    args=p.parse_args()
    if args.stage=='prepare':
        if not all((args.source,args.gemma,args.qwen)):p.error('prepare requires source, gemma and qwen paths')
        prepare(args)
    elif args.stage=='score':score(args)
    else:select(args)

if __name__=='__main__':main()
