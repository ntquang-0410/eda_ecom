from pathlib import Path
import pandas as pd,json,hashlib,unicodedata,re

W=Path(r'D:\download\NCKH\judge_kaggle_preflight');A=W/'audit_detailed_20261101';O=W/'review_draft_20261101';O.mkdir(exist_ok=True)
B=W/'hf_judged_backup/126a564f88c72ac7feb2d4cc79e7183348f5aad9';revision='82aa9c98a0b4ec9255b63b1bfc379c66f52a5b25'
for m,expected in {'cum':'e0a625b2c1159133af24a8ae43c3ec4af4911ada68e10cf392920b8e49a20def','pair':'d4535c62ed1e3692f1563143d10c5c612aaec30d5e574eb9969ef003c627230d'}.items():
 assert hashlib.sha256((B/f'bo_{m}_chon_deepseek.parquet').read_bytes()).hexdigest()==expected
base={m:pd.read_parquet(B/f'bo_{m}_chon_deepseek.parquet') for m in ('cum','pair')};assert base['cum'][['source','product_id']].equals(base['pair'][['source','product_id']])
changes={
 ('1688','1072701738254'):{'field':'title_vi','old_part':'cho trẻ sơ sinh','new_part':'cho trẻ sơ sinh vào mùa xuân và mùa thu','choice':'manual_edit','reason':'Bổ sung 春秋 (xuân–thu) bị lược trong bản đang giữ; giữ các ý còn lại.'},
 ('tiki','275038735'):{'field':'title_zh','old_part':'磁性遮阳帘 汽车专用 适用于马自达车型 - 原厂正品','new_part':'马自达专用汽车磁吸遮阳帘-正品','choice':'gemma_candidate','reason':'Dùng ứng viên Gemma: giữ hàng chính hãng, bỏ khẳng định 原厂 không có trong nguồn.'},
 ('tiki','190194536'):{'field':'title_zh','old_part':'儿童生长迟缓专用','new_part':'适用于佝偻病及生长迟缓儿童','choice':'manual_edit_requires_medical_wording_review','reason':'Bổ sung nghĩa còi xương còn thiếu; cần người có chuyên môn xác minh thuật ngữ và cách diễn đạt trước khi dùng.'}}
edits=[];draft={}
for method,d in base.items():
 d=d.copy();old=d.copy()
 for (src,pid),v in changes.items():
  mask=d.source.eq(src)&d.product_id.astype(str).eq(pid);assert int(mask.sum())==1
  idx=d.index[mask][0];before=d.at[idx,v['field']];assert before.count(v['old_part'])==1
  after=before.replace(v['old_part'],v['new_part']); old_model=d.at[idx,'judge_chon_title'];new_model='gemma' if v['choice']=='gemma_candidate' else old_model
  d.at[idx,v['field']]=after;d.at[idx,'judge_chon_title']=new_model
  edits.append({'method':method,'source':src,'product_id':pid,'field':v['field'],'before':before,'after':after,'model_before':old_model,'model_after':new_model,'approach':v['choice'],'reason':v['reason'],'status':'draft_needs_review' if 'requires' in v['choice'] else 'draft_checked_not_published'})
 assert len(d)==len(old) and d[['source','product_id']].equals(old[['source','product_id']])
 for col in d:
  if col not in ('title_vi','title_zh','judge_chon_title'):assert d[col].equals(old[col])
  else:assert int(d[col].ne(old[col]).sum())<=3
 d.to_parquet(O/f'bo_{method}_title_review_draft.parquet',index=False);draft[method]=d
assert draft['cum'][['title_zh','title_vi','judge_chon_title']].equals(draft['pair'][['title_zh','title_vi','judge_chon_title']])
ed=pd.DataFrame(edits);ed.to_parquet(O/'title_draft_changes.parquet',index=False);ed.to_csv(O/'title_draft_changes.csv',index=False,encoding='utf-8-sig')
f=pd.read_parquet(A/'attribute_signals_triaged.parquet');f['component']=f.component.replace({'ten':'key','gia_tri':'value'});cats=base['cum'][['source','product_id','category']].copy();cats.product_id=cats.product_id.astype(str);f.product_id=f.product_id.astype(str);f=f.merge(cats,on=['source','product_id'],how='left',validate='many_to_one');assert f.category.notna().all()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s))).strip()
f['source_normalized']=f.source_text.map(norm);f['target_normalized']=f.target_text.map(norm)
keys=['source','category','component','signal','source_normalized','target_normalized'];grp=f.groupby(keys,dropna=False,sort=False).agg(luot=('product_id','size'),so_san_pham=('product_id','nunique'),so_phuong_phap=('method','nunique'),triage=('triage','first')).reset_index();grp.insert(0,'review_group_id',range(1,len(grp)+1));f=f.merge(grp[keys+['review_group_id']],on=keys,how='left',validate='many_to_one');assert f.review_group_id.notna().all()
grp.sort_values(['luot','so_san_pham'],ascending=False).to_parquet(O/'attribute_review_groups_private.parquet',index=False);f[['review_group_id','method','source','category','product_id','component','signal','item_id']].drop_duplicates().to_parquet(O/'attribute_group_membership.parquet',index=False)
triage=pd.read_parquet(A/'repeated_character_triage.parquet');triage['result']='can_xac_minh';triage['next_step']=triage.classification.map({'possible_placeholder_code_review':'Xác minh mã với nguồn hoặc nhà cung cấp; không tự xóa.','repeated_separator_review':'Ký tự phân cách; kiểm tra yêu cầu làm sạch trước khi sửa.','float_precision_artifact_review':'Kiểm tra giá trị nguồn và quy tắc làm tròn trước khi sửa.','very_small_source_value_review':'Giá trị nguồn rất nhỏ; xác minh đơn vị và ngữ cảnh.','brand_identifier_review':'Chuỗi nằm trong nhãn thương hiệu; không tự sửa.'});assert triage.next_step.notna().all();triage.to_parquet(O/'repeated_items_followup.parquet',index=False)
summary={'input_revision':revision,'draft_only':True,'hf_modified':False,'products_each':len(base['cum']),'title_changed_per_method':3,'title_manual_wording_requires_medical_review':1,'attribute_signal_occurrences':len(f),'distinct_context_sensitive_groups':len(grp),'repeat_products_needing_context_review':int(triage.product_id.nunique()),'sample_titles_without_technical_flags_reviewed':15,'semantic_coverage':'sample only; not exhaustive','input_sha256':{m:hashlib.sha256((B/f'bo_{m}_chon_deepseek.parquet').read_bytes()).hexdigest() for m in base}}
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False),flush=True)
