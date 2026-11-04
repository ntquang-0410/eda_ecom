
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,hashlib,re
from pathlib import Path
from collections import Counter
import pandas as pd
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
root=Path(__file__).resolve().parent;os.chdir(root)
work=root/'outputs/1688_cjk_repair';corrections=json.loads((work/'alternatives.json').read_text(encoding='utf-8'))
if (work/'corrections.json').exists():
    corrections.update(json.loads((work/'corrections.json').read_text(encoding='utf-8')))
if (work/'manual_token_corrections.json').exists():
    corrections.update(json.loads((work/'manual_token_corrections.json').read_text(encoding='utf-8')))
original=pd.read_parquet(root/'outputs/comet_best_all_titles/1688_zh_to_vi_best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet')
selected=original.copy();applied=Counter()
for idx,r in selected.iterrows():
    c=corrections.get(r.title_zh)
    if not c:continue
    selected.at[idx,'title_vi']=c['text'];applied[c['method']]+=1
    if c['method']=='existing_alternative':
        selected.at[idx,'selected_model']=c['model']
        selected.at[idx,'selected_cometkiwi_score']=r.cometkiwi_gemma4 if c['model']=='gemma4' else r.cometkiwi_qwen3_14b_awq
    elif c['method']=='manual_cjk_token':
        selected.at[idx,'selected_model']='manual_cjk_cleanup'
        selected.at[idx,'selected_cometkiwi_score']=float('nan')
    else:
        selected.at[idx,'selected_model']='gemma4_cjk_repair'
        selected.at[idx,'selected_cometkiwi_score']=float('nan')
    selected.at[idx,'selection_policy']='cjk_cleanup_after_comet'
    selected.at[idx,'selection_reason']=c['method']
assert len(selected)==16348 and selected.product_id.equals(original.product_id) and selected.title_zh.equals(original.title_zh)
assert selected.title_vi.notna().all() and selected.title_vi.str.strip().ne('').all()
before=original.drop_duplicates('title_zh').title_vi.str.contains(r'[\u3400-\u9fff]').sum()
after=selected.drop_duplicates('title_zh').title_vi.str.contains(r'[\u3400-\u9fff]').sum()
assert before==487
lookup=selected.set_index('product_id')['title_vi'];ops=[];results=[];out=work/'publish';out.mkdir(exist_ok=True)
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha
targets=[('data/1688/title_translations/best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet',selected,'selected'),
 ('data/1688/title_translations/gemma4/1688_zh_to_vi_gemma4.parquet',None,'model'),
 ('data/1688/title_translations/qwen3_14b_awq/1688_zh_to_vi_qwen3_14b_awq.parquet',None,'model'),
 ('data/1688/full_dataset/1688_full_attributes_independent_zh_vi.parquet',None,'full'),
 ('data/1688/full_dataset/1688_full_attributes_context_zh_vi.parquet',None,'full')]
for remote,frame,kind in targets:
    remote_file=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']))
    old=pd.read_parquet(remote_file)
    assert len(old)==len(original) and old.product_id.equals(original.product_id) and old.title_zh.equals(original.title_zh)
    if kind=='selected':
        assert ((old.title_vi==original.title_vi)|(old.title_vi==selected.title_vi)).all(),'selected_source_changed_on_hf'
        frame=selected
    else:
        frame=old.copy();values=frame.product_id.map(lookup)
        if kind=='full':
            assert ((old.title_vi==original.title_vi)|(old.title_vi==selected.title_vi)).all(),'full_source_changed_on_hf'
            frame['title_vi']=values
        else:frame['title_vi_corrected_final']=values
    assert frame.product_id.equals(old.product_id) and frame.title_zh.equals(old.title_zh)
    if kind=='full':
        for col in ['category','description_zh','description_vi']:pd.testing.assert_series_equal(frame[col],old[col])
    path=out/remote;path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)
    reread=pd.read_parquet(path);assert reread.product_id.equals(frame.product_id)
    sha=hashlib.sha256(path.read_bytes()).hexdigest();results.append({'path':remote,'rows':len(frame),'sha256':sha,'kind':kind})
    ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
unique_methods=Counter(c['method'] for c in corrections.values())
report={'source_revision':head,'original_unique_with_cjk':int(before),'remaining_unique_with_cjk':int(after),'corrected_unique':len(corrections),'methods_unique':dict(unique_methods),'models_separate_originals_preserved':True,'new_api_repairs_rescored_by_comet':False,'semantic_verified':False,'exports':results}
reportfile=out/'correction_report.json';reportfile.write_text(json.dumps(report,indent=2));ops.append(CommitOperationAdd(path_in_repo='data/1688/title_translations/best_cometkiwi/cjk_correction_report.json',path_or_fileobj=str(reportfile)))
for remote in ['data/1688/full_dataset/manifest.json','manifest.json']:
    p=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']))
    metadata=json.loads(p.read_text(encoding='utf-8'))
    for entries_key in ['exports','title_exports']:
        for entry in metadata.get(entries_key,[]):
            for updated in results:
                if entry.get('path')==updated['path']:entry['sha256']=updated['sha256']
    p=out/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)))
remote='data/1688/title_translations/best_cometkiwi/selection_report.json'
p=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']))
metadata=json.loads(p.read_text(encoding='utf-8'));metadata['post_comet_cjk_cleanup']=report
metadata['sha256']=next(r['sha256'] for r in results if r['kind']=='selected')
metadata['selection_counts_unique_after_cleanup']=dict(Counter(selected.drop_duplicates('title_zh').selected_model))
metadata['selection_counts_product_rows_after_cleanup']=dict(Counter(selected.selected_model))
p=out/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)))
docs=['README.md','data/1688/title_translations/best_cometkiwi/README.md','data/1688/title_translations/gemma4/README.md','data/1688/title_translations/qwen3_14b_awq/README.md','data/1688/full_dataset/README.md']
note=f'''\n\n## Cập nhật sửa title còn chữ Trung\n\nBản 1688 kết hợp đã sửa {len(corrections)}/487 title có chữ Hán; còn {int(after)} title cần kiểm tra. Cách xử lý: ưu tiên ứng viên đã có của model còn lại nếu không còn chữ Hán và qua kiểm tra kỹ thuật; các bản sửa mới được ghi nguồn riêng. Đây là bước sửa sau khi chọn COMET: bản sửa mới chưa được COMET chấm lại, điểm cũ không đại diện cho bản sửa mới. Các bản riêng Gemma/Qwen giữ cột `title_vi` gốc và bổ sung `title_vi_corrected_final` để đối chiếu với bản kết hợp. Hai file full dùng title đã sửa và giữ nguyên description/category. Dữ liệu dịch máy chưa được xác nhận ngữ nghĩa thủ công toàn bộ.\n'''
for remote in docs:
    f=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));content=f.read_text(encoding='utf-8').split('\n\n## Cập nhật sửa title còn chữ Trung')[0]
    unique_counts=Counter(selected.drop_duplicates('title_zh').selected_model);row_counts=Counter(selected.selected_model)
    fmt=lambda n:f'{n:,}'.replace(',','.')
    # Existing selection statistics refer to the COMET run; cleanup counts are reported separately below.
    content=content.replace('Gemma **9,563**, Qwen **6,300**',f"Gemma **{unique_counts['gemma4']+unique_counts['gemma4_cjk_repair']:,}**, Qwen **{unique_counts['qwen3_14b_awq']:,}**")
    content=content.replace('Gemma **9,889**, Qwen **6,459**',f"Gemma **{row_counts['gemma4']+row_counts['gemma4_cjk_repair']:,}**, Qwen **{row_counts['qwen3_14b_awq']:,}**")
    content+=note
    path=out/remote;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Repair residual Chinese in 1688 titles and propagate selected corrections')
for item in results:
    f=hf_hub_download(repo,item['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN'])
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==item['sha256']
report['revision']=c.oid;(work/'upload_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
