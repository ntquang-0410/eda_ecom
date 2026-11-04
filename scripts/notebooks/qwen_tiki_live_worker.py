"""Translate Tiki with the same Qwen3-14B-AWQ generator and existing Tiki QA."""
from __future__ import annotations

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
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,'/content/title_translation_qwen3_14b_awq_v6_sampled_source')
from qwen_kaggle_health import QwenHealthMonitor

SOURCE_SHA='26e027f714764d16cdd76ae711892b869931fa7302fff5bc75b64a43415e4297'
REVISION='31c69efc29464b6bb0aee1398b5a7b50a99340c3'
CACHE='cache_tiki_qwen3_14b_awq.sqlite3'


def atomic(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=True,indent=2),encoding='utf-8')
    temporary.replace(path)


def counts(cache):
    with closing(sqlite3.connect(cache.db_path)) as db:
        completed,passed=db.execute("SELECT count(*),coalesce(sum(qa_passed),0) FROM translations WHERE status='ok'").fetchone()
        errors=db.execute("SELECT count(*) FROM translations WHERE status!='ok'").fetchone()[0]
    return {'completed':completed,'qa_passed':passed,'errors':errors}


def verify_model(monitor):
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    roots=[p.parent for p in Path('/kaggle/input').rglob('model.safetensors.index.json')
           if all((p.parent/name).is_file() for name in manifest)]
    if len(roots)!=1:raise RuntimeError('pinned_model_input_not_unique')
    root=roots[0]
    monitor.set_phase('verify_model')
    for name,expected in manifest.items():
        path=root/name
        if path.stat().st_size!=expected['size']:raise RuntimeError('model_size_mismatch')
        digest=hashlib.sha256() if 'sha256' in expected else hashlib.sha1()
        if 'sha256' not in expected:digest.update(f"blob {expected['size']}\0".encode('ascii'))
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(block)
                monitor.progress(progress_done=stream.tell(),progress_total=expected['size'],progress_unit='bytes')
        if digest.hexdigest()!=expected.get('sha256',expected.get('git_sha1')):
            raise RuntimeError('model_checksum_mismatch')
    return root


def main():
    monitor=QwenHealthMonitor(ROOT/'qwen_health.json',checkpoint_name=CACHE)
    try:
        import pandas as pd
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from tiki_translation_pipeline import SQLiteTranslationCache,qa_check_vi_to_zh,QWEN_PROMPT_TEMPLATE_VI_ZH
        import importlib.util,shutil
        if importlib.util.find_spec('title_translation_hybrid') is None:
            generated=ROOT/'generator_source';generated.mkdir(exist_ok=True)
            for name in ('title_translation_hybrid.py','title_translation_pipeline.py'):
                matches=[p for p in Path('/kaggle/input').rglob(name) if 'bundle' in str(p)]
                if len(matches)!=1:raise RuntimeError('generator_bundle_not_unique')
                shutil.copy2(matches[0],generated/name)
            sys.path.insert(0,str(generated))
        import title_translation_hybrid as generator
        source=ROOT/'source_tiki.parquet'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('source_checksum_mismatch')
        frame=pd.read_parquet(source)
        column='title_vi_cleaned' if 'title_vi_cleaned' in frame else 'title_vi'
        titles=frame[column].dropna().drop_duplicates().tolist()
        if len(frame)!=13200 or len(titles)!=13102 or any(not isinstance(t,str) or not t.strip() for t in titles):
            raise RuntimeError('source_shape_mismatch')
        model_path=verify_model(monitor)
        config={'dataset_sha256':SOURCE_SHA,'dataset_revision':'51540c9e189a6747e768cec08c25c0d4a68776da',
            'direction':'vi_to_zh','model_id':'Qwen/Qwen3-14B-AWQ','model_revision':REVISION,
            'prompt_sha256':hashlib.sha256(QWEN_PROMPT_TEMPLATE_VI_ZH.encode()).hexdigest(),
            'temperature':0.7,'top_p':0.8,'top_k':20,'repetition_penalty':1.05,'max_new_tokens':256,
            'enable_thinking':False,'quality_review_deferred':True,'rows':13200,'unique_titles':13102}
        path=ROOT/'run_config.json'
        if path.exists() and json.loads(path.read_text())!=config:raise RuntimeError('run_configuration_changed')
        atomic(path,config)
        cache=SQLiteTranslationCache(ROOT/CACHE,'qwen3_14b_awq')
        monitor.set_phase('loading_model')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        tokenizer.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
            dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':0})
        model.eval()
        generator._qwen_messages=lambda title:[
            {'role':'system','content':'You are a professional e-commerce translator. Translate Vietnamese titles to concise Simplified Chinese.'},
            {'role':'user','content':QWEN_PROMPT_TEMPLATE_VI_ZH.format(title=title)}]
        gates={}; batch_size=4
        for phase,target in [('smoke',titles[:10]),('pilot',titles[:50]),('full',titles)]:
            monitor.set_phase(phase)
            for pass_index in range(5):
                pending=[t for t in target if not (cache.get(t) or {}).get('status')=='ok']
                if not pending:break
                position=0
                while position<len(pending):
                    group=pending[position:position+(1 if phase!='full' else batch_size)]
                    started=time.monotonic()
                    try:
                        records=generator._qwen_generate(model,tokenizer,group,'cuda:0',256)
                    except Exception as exc:
                        if generator._is_cuda_oom(exc) and len(group)>1:
                            batch_size=max(1,len(group)//2);torch.cuda.empty_cache();continue
                        raise
                    for title,record in zip(group,records):
                        if record['truncated'] or record.get('repetition_detected'):
                            record=generator._qwen_generate(model,tokenizer,[title],'cuda:0',512 if record['truncated'] else 256)[0]
                        text=record['output_text'].strip().strip('"\'「」『』')
                        ok=bool(text) and not record['truncated'] and not record.get('repetition_detected')
                        qa=qa_check_vi_to_zh(title,text)
                        if not ok:
                            qa['passed']=False;qa['flags'].append('generation_incomplete')
                        cache.put(title,text,'ok' if ok else 'error',qa,elapsed=time.monotonic()-started)
                    position+=len(group)
                    total=counts(cache)
                    monitor.progress(phase='generating',processed_this_session=total['completed'],
                        completed_this_session=total['completed'],total_this_session=len(titles),
                        errors_this_session=total['errors'],progress_unit='titles',progress_done=total['completed'],progress_total=len(titles))
            records=[cache.get(t) for t in target]
            completed=sum(bool(r and r['status']=='ok') for r in records)
            passed=sum(bool(r and r['status']=='ok' and r['qa_passed']) for r in records)
            gates[phase]={'requested':len(target),'completed':completed,'qa_passed':passed,
                'completion_rate':completed/len(target),'technical_pass_rate':passed/len(target)}
            atomic(ROOT/'stage_reports.json',gates)
            if phase!='full' and completed/len(target)<0.99:raise RuntimeError('preflight_completion_failed')
        total=counts(cache)
        if total['completed']!=len(titles):raise RuntimeError('full_incomplete_resume_required')
        records=[cache.get(str(title)) for title in frame[column]]
        if 'title_zh' in frame and frame['title_zh'].fillna('').astype(str).str.strip().ne('').any():
            frame['title_zh_source']=frame['title_zh']
        frame['title_zh_qwen3_14b_awq']=[r['title_zh'] for r in records]
        frame['title_zh_qwen3_14b_awq_qa_passed']=[r['qa_passed'] for r in records]
        frame['title_zh_qwen3_14b_awq_qa_flags']=[','.join(r['qa_flags']) for r in records]
        frame['title_zh']=frame['title_zh_qwen3_14b_awq']
        destination=ROOT/'tiki_vi_cado_title_qwen3_14b_awq.parquet'
        frame.to_parquet(destination,index=False)
        report={'status':'all_titles_complete','completed_unique_titles':total['completed'],'total_unique_titles':len(titles),
            'qa_passed':total['qa_passed'],'quality_review_passed':total['qa_passed']/len(titles)>=0.9,
            'semantic_verified':False,'quality_review_deferred':True,'output_rows':len(frame),
            'output_path':str(destination),'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'stages':gates,'config':config}
        atomic(ROOT/'qwen_tiki_report.json',report)
        monitor.cell_finished(final=True)
    except BaseException as exc:
        monitor.fail(exc)
        print('[qwen-worker-error] '+json.dumps({'error_class':type(exc).__name__}),flush=True)
        raise SystemExit(1) from None
    finally:monitor.close()


if __name__=='__main__':main()
