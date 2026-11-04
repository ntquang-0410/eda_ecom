"""Audit the pinned Tiki Qwen export without changing translations or QA policy."""

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
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tiki_translation_pipeline import qa_check_vi_to_zh

ROOT=Path(__file__).parent
RUN=ROOT/'title_translation_tiki_qwen3_14b_awq_vi_zh_v1'
SOURCE=Path(r'D:\HuggingFace_Models\hub\datasets--ntquang0410--zh-vie_ecom\snapshots\51540c9e189a6747e768cec08c25c0d4a68776da\data\cleaned\tiki\tiki_vi_cado.parquet')
SOURCE_SHA='26e027f714764d16cdd76ae711892b869931fa7302fff5bc75b64a43415e4297'
EXPORT_SHA='f101d803efe4c65b7a1131dd0652281679e8bd5202db75b304008025537340a5'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_frames(source, output):
    column='title_vi_cleaned' if 'title_vi_cleaned' in source else 'title_vi'
    if len(source)!=len(output):raise ValueError('row_count_changed')
    for name in source.columns:
        # The export deliberately promotes the translated title into title_zh.
        exported='title_zh_source' if name=='title_zh' and 'title_zh_source' in output else name
        if name=='title_zh' and exported==name:continue
        # Arrow compares nested list/struct values reliably; pandas object equality
        # can report unequal for identical arrays of attribute dictionaries.
        if exported not in output or not pa.array(source[name]).equals(pa.array(output[exported])):
            raise ValueError('source_column_or_order_changed:'+name)
    translated=output['title_zh_qwen3_14b_awq']
    if translated.isna().any() or translated.astype(str).str.strip().eq('').any():
        raise ValueError('blank_translation')
    if not output['title_zh'].equals(translated):raise ValueError('promoted_translation_mismatch')
    if output.groupby(column,dropna=False)['title_zh_qwen3_14b_awq'].nunique().gt(1).any():
        raise ValueError('inconsistent_duplicate_title')
    return column

def main():
    output=RUN/'tiki_vi_cado_title_qwen3_14b_awq.parquet'
    if sha(SOURCE)!=SOURCE_SHA or sha(output)!=EXPORT_SHA:raise ValueError('checksum_mismatch')
    report=json.loads((RUN/'qwen_tiki_report.json').read_text())
    source=pd.read_parquet(SOURCE);frame=pd.read_parquet(output)
    column=validate_frames(source,frame)
    unique=frame.drop_duplicates(column).copy()
    if len(frame)!=13200 or len(unique)!=13102:raise ValueError('source_shape_mismatch')
    with closing(sqlite3.connect((RUN/'cache_tiki_qwen3_14b_awq.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        cache=pd.read_sql_query('SELECT title_vi,title_zh,status,qa_passed,qa_flags FROM translations',db).set_index('title_vi')
    categories=Counter();review=[];mismatches=0;detail_variations=0
    passed_col='title_zh_qwen3_14b_awq_qa_passed'
    flags_col='title_zh_qwen3_14b_awq_qa_flags'
    for title,text,passed,flags in unique[[column,'title_zh_qwen3_14b_awq',passed_col,flags_col]].itertuples(index=False,name=None):
        record=cache.loc[title]
        if record['status']!='ok' or text!=record['title_zh']:raise ValueError('cache_export_mismatch')
        stored=json.loads(record['qa_flags'] or '[]')
        if flags!=','.join(stored):raise ValueError('cache_export_flags_mismatch')
        qa=qa_check_vi_to_zh(title,text)
        # Keep detailed flags from the original cache, including number details.
        # The original number checker samples up to three values from a set;
        # hash randomization can change that sample/order across processes.
        if sorted(qa['flags'])!=sorted(stored):detail_variations+=1
        if bool(passed)!=bool(record['qa_passed']) or sorted(f.split(':',1)[0] for f in qa['flags'])!=sorted(f.split(':',1)[0] for f in stored) or qa['passed']!=bool(passed):
            mismatches+=1
        if not passed:
            categories.update(f.split(':',1)[0] for f in stored)
            review.append({'title_vi':title,'title_zh_qwen3_14b_awq':text,'qa_flags':json.dumps(stored,ensure_ascii=False),
                           'review_status':'pending','semantic_verified':False})
    if mismatches:raise ValueError('qa_recomputation_mismatch')
    if len(cache)!=len(unique):raise ValueError('cache_coverage_mismatch')
    artifacts=RUN/'postrun_audit';artifacts.mkdir(exist_ok=True)
    review_frame=pd.DataFrame(review)
    review_frame.to_csv(artifacts/'qwen_tiki_review_required_unique.csv',index=False,encoding='utf-8-sig')
    review_frame.to_parquet(artifacts/'qwen_tiki_review_required_unique.parquet',index=False)
    strict=frame[frame[passed_col].astype(bool)].copy()
    strict['semantic_verified']=False
    strict.to_parquet(artifacts/'tiki_qwen_technical_qa_passed.parquet',index=False)
    summary={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'model':'Qwen/Qwen3-14B-AWQ',
             'source_revision':'51540c9e189a6747e768cec08c25c0d4a68776da',
             'source_sha256':SOURCE_SHA,'full_export_sha256':EXPORT_SHA,'rows':len(frame),'unique_titles':len(unique),
             'technical_qa_passed_unique':int(unique[passed_col].sum()),'review_required_unique':len(review),
             'technical_qa_passed_product_rows':len(strict),'flag_categories_unique':dict(categories),
             'qa_recomputation_mismatches':mismatches,'source_columns_and_order_verified':True,
             'original_flag_detail_sample_variations':detail_variations,
             'cache_export_verified':True,'semantic_verified':False,'quality_review_passed':False,
             'original_cache_modified':False,'translations_modified':False,'publication_status':'not_published'}
    original_schema=pq.read_schema(SOURCE);exported_schema=pq.read_schema(output)
    summary['storage_type_changes']={field.name:{'source':str(field.type),'export':str(exported_schema.field(field.name).type)}
                                    for field in original_schema if field.name in exported_schema.names
                                    and field.type!=exported_schema.field(field.name).type}
    if summary['technical_qa_passed_unique']!=report['qa_passed']:raise ValueError('report_count_mismatch')
    (artifacts/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=f"""# Qwen Tiki: post-run technical audit

Source: ntquang0410/zh-vie_ecom, revision {summary['source_revision']}.
Model: Qwen/Qwen3-14B-AWQ. Direction: Vietnamese to Simplified Chinese.

- Full export: {len(frame)} product rows / {len(unique)} unique titles.
- Technical QA passed: {summary['technical_qa_passed_unique']} unique titles / {len(strict)} product rows.
- Review required: {len(review)} unique titles. Categories overlap.
- Original source fields, row order, cache alignment, checksums and QA recomputation verified.

## Limitations

Technical checks do not establish translation accuracy. `semantic_verified=false` for ALL rows.
The full export still has unresolved QA flags and does not pass the original 90% technical QA gate.
The technical-pass subset is a derived selection, not a repaired full dataset or proof of semantic quality.
No translations, cache entries or original QA flags have been altered. Nothing was published to Hugging Face.

## Files

`qwen_tiki_review_required_unique.csv` / `.parquet`: review queue with original flags; review_status=pending.
`tiki_qwen_technical_qa_passed.parquet`: product rows whose titles pass the unchanged technical checks.
`audit_summary.json`: aggregate findings and provenance.
`manifest.json`: file hashes for integrity verification.
"""
    (artifacts/'README.md').write_text(readme,encoding='utf-8')
    files=[output,RUN/'qwen_tiki_report.json']+sorted(artifacts.glob('*'))
    files=[p for p in files if p.name!='manifest.json']
    manifest={p.relative_to(RUN).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    (artifacts/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    bundle=RUN/'qwen_tiki_audited_export.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files+[artifacts/'manifest.json']:archive.write(p,p.relative_to(RUN).as_posix())
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':main()
