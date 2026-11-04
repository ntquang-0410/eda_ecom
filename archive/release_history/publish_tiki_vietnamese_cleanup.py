
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
from tiki_translation_pipeline import qa_check_vi_to_zh
disable_progress_bars()
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
work=ROOT/'outputs/tiki_vi_cleanup';work.mkdir(parents=True,exist_ok=True)
viet=r'[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]'
basepath=ROOT/'outputs/comet_best_all_titles/tiki_vi_to_zh_best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet'
original=pd.read_parquet(basepath)
src=pd.read_parquet(r'D:\HuggingFace_Models\hub\datasets--ntquang0410--zh-vie_ecom\snapshots\51540c9e189a6747e768cec08c25c0d4a68776da\data\cleaned\tiki\tiki_vi_cado.parquet',columns=['product_id','title_vi','brand'])
assert len(original)==len(src)==13200 and original.product_id.equals(src.product_id) and original.title_vi.equals(src.title_vi)
brands={s:{x.strip() for x in g.brand if isinstance(x,str) and len(x.strip())>1} for s,g in src.groupby('title_vi',sort=False)}
corrections={};counts=Counter()
for r in original.drop_duplicates('title_vi').itertuples():
    if not re.search(viet,r.title_zh or ''):continue
    alt=r.qwen3_14b_awq_translation if r.selected_model=='gemma4' else r.gemma4_translation
    ok=r.qwen3_14b_awq_generation_ok if r.selected_model=='gemma4' else r.gemma4_generation_ok
    if not isinstance(alt,str) or not alt.strip() or re.search(viet,alt) or not ok:counts['no_clean_alternative']+=1;continue
    if not qa_check_vi_to_zh(r.title_vi,alt)['passed']:counts['alternative_failed_technical_checks']+=1;continue
    if any(b.casefold() in r.title_zh.casefold() and b.casefold() not in alt.casefold() for b in brands.get(r.title_vi,set())):
        counts['retained_brand_for_review']+=1;continue
    corrections[r.title_vi]={'text':alt,'model':'qwen3_14b_awq' if r.selected_model=='gemma4' else 'gemma4'}
selected=original.copy()
for i,r in selected.iterrows():
    c=corrections.get(r.title_vi)
    if c:
        selected.at[i,'title_zh']=c['text'];selected.at[i,'selected_model']=c['model']
        selected.at[i,'selected_cometkiwi_score']=r.cometkiwi_gemma4 if c['model']=='gemma4' else r.cometkiwi_qwen3_14b_awq
        selected.at[i,'selection_policy']='vietnamese_cleanup_after_comet';selected.at[i,'selection_reason']='existing_alternative_technical_pass_brand_preserved'
before=original.drop_duplicates('title_vi').title_zh.str.contains(viet,regex=True,na=False).sum();after=selected.drop_duplicates('title_vi').title_zh.str.contains(viet,regex=True,na=False).sum()
assert before==608 and after==before-len(corrections)
lookup=selected.set_index('product_id').title_zh;api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha
targets=[('data/tiki/title_translations/best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet','selected'),('data/tiki/title_translations/gemma4/tiki_vi_to_zh_gemma4.parquet','model'),('data/tiki/title_translations/qwen3_14b_awq/tiki_vi_to_zh_qwen3_14b_awq.parquet','model'),('data/tiki/full_dataset/tiki_full_attributes_independent_vi_zh.parquet','full'),('data/tiki/full_dataset/tiki_full_attributes_context_vi_zh.parquet','full')]
ops=[];exports=[]
for remote,kind in targets:
    f=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));old=pd.read_parquet(f)
    assert len(old)==len(original) and old.product_id.equals(original.product_id) and old.title_vi.equals(original.title_vi)
    if kind in ('selected','full'):assert old.title_zh.equals(original.title_zh),'source_changed'
    frame=selected.copy() if kind=='selected' else old.copy()
    if kind=='full':frame['title_zh']=frame.product_id.map(lookup)
    if kind=='model':frame['title_zh_corrected_final']=frame.product_id.map(lookup)
    if kind=='full':
        for col in ('description_vi','description_zh','category'):pd.testing.assert_series_equal(frame[col],old[col])
    path=work/remote;path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)
    check=pd.read_parquet(path);assert check.product_id.equals(frame.product_id) and check.title_vi.equals(frame.title_vi)
    item={'path':remote,'kind':kind,'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};exports.append(item);ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
report={'source_revision':head,'original_unique_with_vietnamese_diacritics':int(before),'corrected_unique':len(corrections),'remaining_unique_with_vietnamese_diacritics':int(after),'remaining_reasons':dict(counts),'heuristic':'existing alternative without Vietnamese diacritics, generation valid, technical checks passed, and exact source brand present in previous selected title preserved','semantic_verified':False,'description_text_modified':False,'exports':exports}
p=work/'cleanup_report.json';p.write_text(json.dumps(report,indent=2));ops.append(CommitOperationAdd(path_in_repo='data/tiki/title_translations/best_cometkiwi/vietnamese_cleanup_report.json',path_or_fileobj=str(p)))
for remote in ['data/tiki/full_dataset/manifest.json','manifest.json']:
    p=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));m=json.loads(p.read_text(encoding='utf8'))
    for key in ['exports','title_exports']:
        for e in m.get(key,[]):
            for item in exports:
                if e.get('path')==item['path']:e['sha256']=item['sha256']
    p=work/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m,indent=2),encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)))
remote='data/tiki/title_translations/best_cometkiwi/selection_report.json';p=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));m=json.loads(p.read_text(encoding='utf8'));m['post_comet_vietnamese_cleanup']=report;m['sha256']=next(e['sha256'] for e in exports if e['kind']=='selected');m['selection_counts_unique_after_cleanup']=dict(Counter(selected.drop_duplicates('title_vi').selected_model));m['selection_counts_product_rows_after_cleanup']=dict(Counter(selected.selected_model));p=work/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m,indent=2));ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)))
note=f"\n\n## Rà soát title còn ký tự Việt\n\nSau COMET, thay {len(corrections)} trong {int(before)} title có ký tự Việt có dấu bằng ứng viên sẵn có của model còn lại khi ứng viên đó không có dấu Việt, sinh hợp lệ, qua kiểm tra kỹ thuật và không làm mất tên thương hiệu được giữ nguyên theo dữ liệu nguồn. Còn {int(after)} title có dấu Việt; một phần có thể là thương hiệu/tên riêng, không kết luận tất cả là lỗi. Bản riêng Gemma/Qwen giữ `title_zh` gốc và có thêm `title_zh_corrected_final` để đối chiếu. Hai file full dùng title đã rà; description và category không thay đổi. Cách dò không phát hiện hết tiếng Việt không dấu, và không thay thế kiểm tra ngữ nghĩa thủ công.\n"
for remote in ['README.md','data/tiki/title_translations/best_cometkiwi/README.md','data/tiki/title_translations/gemma4/README.md','data/tiki/title_translations/qwen3_14b_awq/README.md','data/tiki/full_dataset/README.md']:
    f=Path(hf_hub_download(repo,remote,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']));content=f.read_text(encoding='utf8')
    uc=Counter(selected.drop_duplicates('title_vi').selected_model);rc=Counter(selected.selected_model);fmt=lambda n:f'{n:,}'.replace(',','.')
    content=content.replace('| Tiki | 6.240 | 6.862 | 13.102 | 0 |',f"| Tiki | {fmt(uc['gemma4'])} | {fmt(uc['qwen3_14b_awq'])} | 13.102 | 0 |")
    content=content.replace('| Tiki | 6.288 | 6.912 | 13.200 |',f"| Tiki | {fmt(rc['gemma4'])} | {fmt(rc['qwen3_14b_awq'])} | 13.200 |")
    content=content.replace('Gemma **6,240**, Qwen **6,862**',f"Gemma **{uc['gemma4']:,}**, Qwen **{uc['qwen3_14b_awq']:,}**").replace('Gemma **6,288**, Qwen **6,912**',f"Gemma **{rc['gemma4']:,}**, Qwen **{rc['qwen3_14b_awq']:,}**")
    p=work/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content+note,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)))
c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Review residual Vietnamese in Tiki titles and propagate verified alternatives')
for item in exports:
    f=hf_hub_download(repo,item['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']);assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==item['sha256']
report['revision']=c.oid;(work/'upload_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
