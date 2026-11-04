from pathlib import Path
from collections import Counter
import sys,inspect,re,json,hashlib
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download

ROOT=Path(r'D:\download\NCKH\judge_kaggle_handoff');W=Path(r'D:\download\NCKH\judge_kaggle_preflight');OUT=W/'audit_other_value_20261102';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts/judge'));import ghep_bo as G
repo='ntquang0410/zh-vie_ecom';prefix='data/LLM/bilingual_judged/';revision=HfApi().dataset_info(repo).sha
assert revision=='82aa9c98a0b4ec9255b63b1bfc379c66f52a5b25','HF revision changed: verify inputs again'
files={};hashes={}
for method in ('cum','pair'):
 name=f'bo_{method}_chon_deepseek.parquet';path=Path(hf_hub_download(repo,prefix+name,repo_type='dataset',revision=revision));hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();files[method]=pd.read_parquet(path)
CK=W/'title_v1_output/judge_full/data/processed/so_cap';D={}
for method in ('cum','pair'):
 k,e=G.doc_ket_qua_va_tuong_duong('lon_'+method,thu_muc=CK);D[method],_=G.ap_chinh_sach_tuong_duong(k,e,'giu_hien_tai')
rv=pd.read_parquet(W/'hf_judged_backup/126a564f88c72ac7feb2d4cc79e7183348f5aad9/seven_items_assistant_review.parquet')
for r in rv.itertuples():
 if r.assistant_decision in ('gemma','qwen'):D['cum'][r.item_id]=r.assistant_decision
R=[];T=Counter();H=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]');V=re.compile(r'[ĂÂĐÊÔƠƯăâđêôơưÀÁẢÃẠàáảãạÈÉẺẼẸèéẻẽẹÌÍỈĨỊìíỉĩịÒÓỎÕỌòóỏõọÙÚỦŨỤùúủũụỲÝỶỸỴỳýỷỹỵẤẦẨẪẬấầẩẫậẮẰẲẴẶắằẳẵặẾỀỂỄỆếềểễệỐỒỔỖỘốồổỗộỚỜỞỠỢớờởỡợỨỪỬỮỰứừửữự]')
def scan(method,src,pid,part,source,target,item):
 source=str(source);target=str(target);T[(method,src,part,'all')]+=1
 a=set(re.findall(r'\d+(?:[.,]\d+)?',source));b=set(re.findall(r'\d+(?:[.,]\d+)?',target));flags=[]
 if src=='1688' and source.strip() in ('其他','其它') and target.strip().casefold() not in ('khác','loại khác','những loại khác'):
  flags.append('other_value_translation_review')
 if a!=b:flags.append('number_tokens_differ_review')
 if src=='1688' and H.search(target):flags.append('han_remaining_review')
 if src=='tiki' and V.search(target):flags.append('vietnamese_diacritic_review')
 if not target.strip() and source.strip():flags.append('empty_target_review')
 for f in flags:
  T[(method,src,part,f)]+=1;R.append({'method':method,'source':src,'product_id':str(pid),'component':part,'item_id':str(item),'signal':f,'source_text':source,'target_text':target})
G.__dict__['audit_scan']=scan
original_sources={}
def instrument(name,needle,insert):
 source=original_sources.get(name) or inspect.getsource(getattr(G,name));assert source.count(needle)==1;source=source.replace(needle,insert+needle);original_sources[name]=source;exec(compile(source,'audit_instrumented','exec'),G.__dict__)
instrument('ghep_mota_cum','        if dau is not None and',"        audit_scan('cum', src, pid, 'key', p.nguon_ten.iloc[i], ten_moi[-1], ids[i])\n        audit_scan('cum', src, pid, 'value', p.nguon_gt.iloc[i], gt_moi[-1], ids[i])\n")
instrument('ghep_mota_pair','        if u is None:\n            return cu',"        if u is None:\n            audit_scan('pair', src, pid, loai, nguon, cu, '')\n")
instrument('ghep_mota_pair','        return ket_qua',"        audit_scan('pair', src, pid, loai, nguon, ket_qua, iid if u is not None else '')\n")
for method in ('cum','pair'):
 base=pd.read_parquet(G.BASE_CUM if method=='cum' else G.BASE_PAIR);data=G.nap_cap_cum() if method=='cum' else G.nap_pair_unit()
 for src in ('1688','tiki'):
  b=base[base.source.eq(src)];expected=files[method][files[method].source.eq(src)].set_index('product_id')[G.DICH[src][1]]
  if method=='cum':generated,_=G.ghep_mota_cum(b,data[src],D[method],'giu_hien_tai',ngu_canh='lai')
  else:generated,_=G.ghep_mota_pair(b,*data[src],D[method],'giu_hien_tai')
  assert set(generated.index)==set(expected.index)
  assert all(str(v)==str(expected.loc[k]) for k,v in generated.items()),'Replay description mismatch; do not publish audit'
  print(json.dumps({'method':method,'source':src,'products':len(expected),'replay_exact':True}),flush=True)
df=pd.DataFrame(R);df.to_parquet(OUT/'attribute_signals_private.parquet',index=False)
counts=[{'method':a,'source':b,'component':c,'signal':d,'count':int(n)} for (a,b,c,d),n in sorted(T.items())]
report={'revision':revision,'input_sha256':hashes,'description_replay_exact':True,'all_counts':counts,'signal_rows':len(df),'semantic_verified':False,'limits':'Exact number tokens are review signals, not semantic verdicts; names, units, numeral words, case, tokenization and valid brands require review.'}
(OUT/'attribute_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
