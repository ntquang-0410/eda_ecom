from pathlib import Path
from collections import defaultdict, Counter
import sys, json, hashlib
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd

W=Path(r'D:\download\NCKH\judge_kaggle_preflight'); R=Path(r'D:\download\NCKH\judge_kaggle_handoff'); O=W/'pair_provenance_update'; O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'scripts/judge'))
import ghep_bo as G
repo='ntquang0410/zh-vie_ecom'; prefix='data/LLM/bilingual_judged/'; api=HfApi(); head=api.dataset_info(repo).sha
def get(n,rev=None): return Path(hf_hub_download(repo,prefix+n,repo_type='dataset',revision=rev or head))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
old=pd.read_parquet(get('bo_pair_chon_deepseek.parquet')); assert len(old)==25575 and not old.duplicated(['source','product_id']).any()
base=pd.read_parquet(G.BASE_PAIR); base.product_id=base.product_id.astype(str)
ck=W/'title_v1_output/judge_full/data/processed/so_cap'; kq,eq=G.doc_ket_qua_va_tuong_duong('lon_pair',thu_muc=ck); qd,_=G.ap_chinh_sach_tuong_duong(kq,eq,'giu_hien_tai')
raw={
 '1688':pd.read_parquet(G.D1/'don_vi_da_chon_20261022_2126.parquet'),
 'tiki':pd.read_parquet(G.D2/'don_vi_da_chon_20261023_1656.parquet')}
data=G.nap_pair_unit(); trace=[]; all_ids=set()
for src in ('1688','tiki'):
 b=base[base.source.eq(src)]; pairs,units=data[src]; pairs.product_id=pairs.product_id.astype(str)
 p=pairs[pairs.product_id.isin(set(b.product_id))].sort_values(['product_id','pair_id']).copy()
 for col in ('nguon_ten','nguon_gt','cuoi_ten','cuoi_gt'):p[col]=p[col].fillna('').astype(str)
 lookup={}; original=raw[src].set_index('ma_don_vi',drop=False)
 assert original.index.is_unique
 for u in units.to_dict('records'):
  key=(u['loai'],u['van_ban'],u['category']); lookup.setdefault(key,u)
 cats=dict(zip(b.product_id,b.category)); generated=defaultdict(list)
 def render(pid,cat,kind,source,current,pairid):
  u=lookup.get((kind,source,cat)); final=current; origin='chua_ro'; reason='unit_missing'; iid=''; decision=''
  if u is not None:
   iid='unit:'+src+':'+str(u['ma_don_vi']); record=original.loc[u['ma_don_vi']]; decision=G.chon(iid,qd,'giu_hien_tai') or ''
   a='' if pd.isna(u['A']) else str(u['A']); b='' if pd.isna(u['B']) else str(u['B']); tag='' if pd.isna(record['tu']) else str(record['tu']); selected='' if pd.isna(record['ban']) else str(record['ban'])
   if decision and G.chuan(current) in (G.chuan(a),G.chuan(b)):
    candidate=a if decision=='gemma' else b
    if G.chuan(candidate)!=G.chuan(current) and G.an_toan(src,candidate,current):
     final=candidate; origin=decision; reason='judge_changed_to_model'
   if final==current:
    if G.chuan(selected)==G.chuan(current) and tag in ('A','B','co_dinh'):
     origin={'A':'gemma','B':'qwen','co_dinh':'quy_tac'}[tag]; reason='retained_explicit_upstream_source'
    else: reason='retained_source_unverified'
  if final.strip():
   trace.append({'source':src,'product_id':pid,'pair_id':str(pairid),'component':kind,'item_id':iid,'final_origin':origin,'evidence':reason,'judge_decision':decision,'judge_equivalent':iid in eq})
  return final
 for r in p.to_dict('records'):
  pid=r['product_id']; cat=cats[pid]; pairid=r['pair_id']
  n=render(pid,cat,'ten',r['nguon_ten'],r['cuoi_ten'],pairid); v=render(pid,cat,'gia_tri',r['nguon_gt'],r['cuoi_gt'],pairid)
  generated[pid].append(n+': '+v)
 expected=old[old.source.eq(src)].copy(); expected.product_id=expected.product_id.astype(str); expected=expected.set_index('product_id')[G.DICH[src][1]]
 assert set(generated)==set(expected.index)
 assert all('\n'.join(parts)==expected.loc[pid] for pid,parts in generated.items()), 'Final description differs from replay'
 all_ids.update((src,pid) for pid in generated)
 print(json.dumps({'source':src,'products':len(generated),'pairs':len(p)}),flush=True)
t=pd.DataFrame(trace); assert not t.duplicated(['source','product_id','pair_id','component']).any()
keys=['product_id','source','category','title_zh','title_vi','description_zh','description_vi']; new=old[keys].copy(); new.product_id=new.product_id.astype(str)
new['judge_chon_title']=old.title_nguon_dich.map({'gemma4':'gemma','qwen3_14b_awq':'qwen'}); assert new.judge_chon_title.notna().all()
counts=t.groupby(['source','product_id','final_origin']).size(); labels=['gemma','qwen','quy_tac','chua_ro']
def show(s,pid):
 parts=[f'{k}: {int(counts.get((s,pid,k),0))}' for k in labels if counts.get((s,pid,k),0)]
 return ' | '.join(parts) if parts else 'khong_co_don_vi'
new['judge_chon_description']=[show(s,pid) for s,pid in zip(new.source,new.product_id)]
assert new[keys].equals(old[keys].assign(product_id=old.product_id.astype(str)))
for r in new.itertuples():
 total=sum(int(x.split(': ')[1]) for x in r.judge_chon_description.split(' | ')) if r.judge_chon_description!='khong_co_don_vi' else 0
 assert total==sum(counts.get((r.source,r.product_id,k),0) for k in labels)
new.to_parquet(O/'bo_pair_chon_deepseek.parquet',index=False);t.to_parquet(O/'bo_pair_description_provenance.parquet',index=False)
meta=old[['source','product_id']+[c for c in old if c not in keys]].copy();meta.to_parquet(O/'bo_pair_metadata.parquet',index=False)
restored=new[keys].merge(meta,on=['source','product_id'],how='left',validate='one_to_one')[list(old.columns)];assert restored.equals(old)
report={'source_revision':head,'rows':len(new),'columns':list(new),'nonempty_units':len(t),'unit_origins':t.final_origin.value_counts().to_dict(),'unmapped_products':int(new.judge_chon_description.eq('khong_co_don_vi').sum()),'content_unchanged':True,'old_schema_reconstructable':True,'description_replay_exact':True}
(O/'pair_provenance_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True)

readme=get('README.md').read_text(encoding='utf8')
readme=readme.replace('Bản PAIR vẫn giữ cấu trúc 17 cột; cập nhật này chỉ đổi schema bản CỤM theo yêu cầu.', 'Bản PAIR cũng có 9 cột; hai bản dùng đơn vị đếm description khác nhau như giải thích dưới đây.')
readme=readme.replace('Bản CỤM hiện có **9 cột**; bản PAIR có **17 cột**.', 'Bản CỤM và bản PAIR đều có **9 cột**.')
readme=readme.replace('## Schema metadata 17 cột trước đây và cấu trúc bản PAIR','## Schema metadata 17 cột trước đây')
readme=readme.replace('Các giải thích bên dưới về nhãn nguồn cũ, nhóm và cờ áp dụng cho file metadata của bản CỤM và các cột còn giữ trong bản PAIR.', 'Các giải thích bên dưới về nhãn nguồn cũ, nhóm và cờ áp dụng cho file metadata của cả hai bản.')
readme=readme.replace('Bản pair bàn giao ban đầu thiếu 7 cột metadata và sau ghép chỉ có 10 cột. Bản này đã bổ sung đủ:', 'Bản pair trước khi rút gọn đã bổ sung đủ các metadata sau; nay các cột này nằm trong bo_pair_metadata.parquet:')
section='''\n## Bản PAIR: nguồn title và description\n\nFile `bo_pair_chon_deepseek.parquet` có cùng 9 cột với bản cụm. `judge_chon_title` là nguồn model của title cuối (`gemma` hoặc `qwen`); với title sửa tay đây là nguồn bản nháp, chi tiết xem `manual_title_corrections.parquet`.\n\n`judge_chon_description` đếm nguồn **từng tên thuộc tính và từng giá trị không rỗng**, theo thứ tự `gemma`, `qwen`, `quy_tac`, `chua_ro`, chỉ hiện nhóm có số lớn hơn 0. Đây là nguồn nội dung cuối, không phải số lần thắng judge. Đơn vị bị đổi theo quyết định judge được tính cho model sinh bản được áp dụng; đơn vị giữ nguyên dùng nguồn đã chọn trong bảng gốc khi văn bản khớp. Không xác định chắc thì ghi `chua_ro`; đó không phải kết luận bản dịch sai hoặc chưa chấm. Nếu không có đơn vị không rỗng thì ghi `khong_co_don_vi`.\n\n**Đơn vị đếm khác bản CỤM**: pair đếm tên và giá trị riêng; cụm đếm cả cặp tên–giá trị là một mảnh. Không so trực tiếp số đếm giữa hai bản. `bo_pair_description_provenance.parquet` lưu nguồn và bằng chứng cho từng đơn vị, không chứa lại văn bản dịch; `bo_pair_metadata.parquet` giữ các cột gốc để khôi phục schema 17 cột; `pair_provenance_validation.json` ghi đối chiếu. File `bo_pair_danh_dau.parquet` giữ trạng thái judge cũ.\n\n'''
readme=readme.replace('## Trạng thái và phạm vi',section+'## Trạng thái và phạm vi',1)
(O/'README.md').write_text(readme,encoding='utf8')
v=json.loads(get('validation_report.json').read_text());v['columns_by_method']={'cum':9,'pair':9};v['pair_description_provenance']=report;(O/'validation_report.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
manifest=json.loads(get('manifest.json').read_text());manifest['columns_by_method']={'cum':list(new.columns),'pair':list(new.columns)};manifest['pair_description_provenance_source_revision']=head
for f in O.iterdir():
 if f.is_file() and f.name not in ('manifest.json','SHA256SUMS.txt'):manifest['files'][f.name]={'bytes':f.stat().st_size,'sha256':sha(f)}
(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');sums={n:e['sha256'] for n,e in manifest['files'].items()};sums['manifest.json']=sha(O/'manifest.json');(O/'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(sums.items())),encoding='ascii')
files=sorted(O.iterdir()); assert all(f.is_file() for f in files);assert api.dataset_info(repo).sha==head
print('Verified locally; publishing pair update',flush=True)
commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,num_threads=1,operations=[CommitOperationAdd(path_in_repo=prefix+f.name,path_or_fileobj=str(f)) for f in files],commit_message='Expose final pair title and per-unit description provenance')
print('COMMIT',commit.oid,flush=True)
for n,h in sums.items(): assert sha(get(n,commit.oid))==h,n
assert sha(get('SHA256SUMS.txt',commit.oid))==sha(O/'SHA256SUMS.txt');assert pd.read_parquet(get('bo_pair_chon_deepseek.parquet',commit.oid)).equals(new)
(W/'pair_provenance_upload_result.json').write_text(json.dumps({'commit':commit.oid,**report,'verified_remote_files':len(sums)+1},ensure_ascii=False,indent=2),encoding='utf8');print('REMOTE_VERIFIED',len(sums)+1,flush=True)
