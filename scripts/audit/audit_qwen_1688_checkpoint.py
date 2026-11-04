"""Read-only technical audit of a verified, possibly incomplete Qwen checkpoint."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
from collections import Counter
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
import title_translation_pipeline as quality

ROOT=Path(__file__).parent
SOURCE=ROOT/'bilingual_zh_vi_cado.parquet'
CACHE=Path(r'C:\Users\Lenovo\.codex\qwen22-verified\title_translation_cado_qwen3_14b_awq_full_v3_sampled\cache_qwen3_14b_awq.sqlite3')
OUT=ROOT/'qwen1688_checkpoint_audit'

def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=quality.DATASET_SHA256:raise ValueError('source_hash_mismatch')
    source=pd.read_parquet(SOURCE);quality.validate_input_frame(source)
    with closing(sqlite3.connect(CACHE.as_uri()+'?mode=ro',uri=True)) as db:
        cache={row[0]:row[1:] for row in db.execute('SELECT source_title,output_text,status,truncated FROM translations')}
    if len(cache)!=quality.EXPECTED_UNIQUE_TITLES:raise ValueError('checkpoint_coverage_mismatch')
    rows=[];hard=Counter();advisory=Counter();generation=Counter()
    for title in source['title_zh'].drop_duplicates():
        text,status,truncated=cache[title]
        qa=quality.check_translation(title,text,generation_truncated=bool(truncated))
        if status!='ok':
            qa['status']='fail';qa['hard_flags']=sorted(set(qa['hard_flags']+['generation_error']))
        hard.update(flag.split(':',1)[0] for flag in qa['hard_flags'])
        advisory.update(flag.split(':',1)[0] for flag in qa['advisory_flags'])
        generation[status]+=1
        rows.append({'title_zh':title,'title_vi_candidate_qwen3_14b_awq':text,
                     'generation_status':status,'generation_truncated':bool(truncated),
                     'qa_status':qa['status'],'qa_hard_flags':json.dumps(qa['hard_flags'],ensure_ascii=False),
                     'qa_advisory_flags':json.dumps(qa['advisory_flags'],ensure_ascii=False),
                     'semantic_verified':False})
    unique=pd.DataFrame(rows)
    product=source.merge(unique,on='title_zh',how='left',sort=False,validate='many_to_one')
    if len(product)!=quality.EXPECTED_ROWS or not product['product_id'].equals(source['product_id']):raise ValueError('source_alignment_changed')
    OUT.mkdir(exist_ok=True)
    unique[unique['qa_status'].ne('pass')].to_csv(OUT/'qwen1688_review_required_unique.csv',index=False,encoding='utf-8-sig')
    path=OUT/'qwen1688_checkpoint_audited.parquet';product.to_parquet(path,index=False)
    summary={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'snapshot_version':22,
             'unique_titles':len(unique),'product_rows':len(product),'generation_status_counts':dict(generation),
             'technical_qa_passed_unique':int(unique['qa_status'].eq('pass').sum()),
             'technical_qa_flagged_unique':int(unique['qa_status'].ne('pass').sum()),
             'hard_flag_categories_unique':dict(hard),'advisory_flag_categories_unique':dict(advisory),
             'checkpoint_modified':False,'translations_modified':False,'semantic_verified':False,
             'full_generation_complete':generation.get('ok',0)==quality.EXPECTED_UNIQUE_TITLES,
             'source_sha256':quality.DATASET_SHA256,'qa_source_sha256':hashlib.sha256(Path(quality.__file__).read_bytes()).hexdigest(),
             'audited_preview_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))

if __name__=='__main__':main()
