from pathlib import Path
import sys,json,re,unicodedata,hashlib
from collections import Counter
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download

W=Path(r'D:\download\NCKH\judge_kaggle_preflight'); ROOT=Path(r'D:\download\NCKH\judge_kaggle_handoff'); OUT=W/'audit_final_finetune';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts/judge'));import ghep_bo as G
repo='ntquang0410/zh-vie_ecom';prefix='data/LLM/bilingual_judged/';head=HfApi().dataset_info(repo).sha
def get(n):return Path(hf_hub_download(repo,prefix+n,repo_type='dataset',revision=head))
manifest=json.loads(get('manifest.json').read_text(encoding='utf8'));dfs={};report={'revision':head,'semantic_verified':False,'methods':{},'limitations':[]};flags=[]
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().casefold()
def nums(s):return set(re.findall(r'\d+(?:[.,]\d+)?',s))
def mark(kind,src,pid,field,flag):flags.append((kind,src,str(pid),field,flag))
han=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]');vi=re.compile(r'[ĂÂĐÊÔƠƯăâđêôơưÀÁẢÃẠàáảãạÈÉẺẼẸèéẻẽẹÌÍỈĨỊìíỉĩịÒÓỎÕỌòóỏõọÙÚỦŨỤùúủũụỲÝỶỸỴỳýỷỹỵẤẦẨẪẬấầẩẫậẮẰẲẴẶắằẳẵặẾỀỂỄỆếềểễệỐỒỔỖỘốồổỗộỚỜỞỠỢớờởỡợỨỪỬỮỰứừửữự]')
for kind in ('cum','pair'):
 name='bo_'+kind+'_chon_deepseek.parquet';p=get(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['files'][name]['sha256'];d=pd.read_parquet(p);dfs[kind]=d
 s={'rows':len(d),'duplicate_composite_ids':int(d.duplicated(['source','product_id']).sum()),'duplicate_product_ids':int(d.product_id.duplicated().sum()),'nulls':d.isna().sum().to_dict(),'source_counts':d.source.value_counts().to_dict(),'signals':{},'duplicate_counts':{}}
 ct=Counter()
 for r in d.itertuples(index=False):
  src=str(r.source);pid=r.product_id
  for field in ('title_zh','title_vi','description_zh','description_vi'):
   val=getattr(r,field);sval='' if pd.isna(val) else str(val);fl=[]
   if not sval.strip():fl.append('empty')
   if '\ufffd' in sval or any(unicodedata.category(c)=='Cc' and c not in '\r\n\t' for c in sval):fl.append('invalid_unicode_control')
   if re.search(r'<\s*/?think\s*>|```|<\|(?:im_start|im_end)\|>',sval,re.I):fl.append('model_format_marker')
   if re.search(r'As an AI|I cannot translate|Tạm thời không thể dịch|作为(?:一个)?(?:AI|人工智能)|无法(?:提供|翻译)',sval,re.I):fl.append('possible_refusal_or_explanation')
   if re.search(r'(.)\1{12,}',sval):fl.append('repeated_character')
   if len(sval)>(512 if field.startswith('title') else 12000):fl.append('long_text_review')
   for f in fl:mark(kind,src,pid,field,f);ct[field+':'+f]+=1
  for field,sc,tc in [('title','title_zh' if src=='1688' else 'title_vi','title_vi' if src=='1688' else 'title_zh'),('description','description_zh' if src=='1688' else 'description_vi','description_vi' if src=='1688' else 'description_zh')]:
   a='' if pd.isna(getattr(r,sc)) else str(getattr(r,sc));b='' if pd.isna(getattr(r,tc)) else str(getattr(r,tc));fl=[]
   if nums(a)!=nums(b):fl.append('numeric_tokens_differ_review')
   if norm(a)==norm(b) and norm(a):fl.append('source_equals_translation_review')
   if src=='1688' and han.search(b):fl.append('han_remaining_review')
   if src=='tiki' and vi.search(b):fl.append('vietnamese_diacritic_review')
   if field=='title' and src=='tiki' and not han.search(b):fl.append('no_han_review')
   for f in fl:mark(kind,src,pid,tc,f);ct[src+':'+field+':'+f]+=1
 for col in ('title_zh','title_vi','description_zh','description_vi'):
  v=d[col].fillna('').map(lambda x:norm(str(x)));s['duplicate_counts'][col]=int((v.ne('')&v.duplicated(keep=False)).sum())
 s['signals']=dict(ct);s['length_quantiles']={f:{str(k):float(v) for k,v in d[f].fillna('').str.len().quantile([.5,.95,.99,1]).items()} for f in ('title_zh','title_vi','description_zh','description_vi')}
 # Source-language fields must match pre-judge base; translations may legitimately differ.
 base=pd.read_parquet(G.BASE_CUM if kind=='cum' else G.BASE_PAIR);joined=d.merge(base,on=['source','product_id'],validate='one_to_one',suffixes=('','_base'))
 for src in ('1688','tiki'):
  z=joined[joined.source.eq(src)];s.setdefault('source_alignment',{})[src]={c:int((z[c].fillna('')!=z[c+'_base'].fillna('')).sum()) for c in (('title_zh','description_zh') if src=='1688' else ('title_vi','description_vi'))}
 report['methods'][kind]=s
print('STRUCTURAL_AND_TEXT_CHECKS_COMPLETE',flush=True)
a,b=dfs['cum'],dfs['pair'];report['cross_method']={'same_id_order':a[['source','product_id']].equals(b[['source','product_id']]),'same_title':a[['title_zh','title_vi']].equals(b[['title_zh','title_vi']]),'different_description_zh':int(a.description_zh.ne(b.description_zh).sum()),'different_description_vi':int(a.description_vi.ne(b.description_vi).sum())}
backup=W/'hf_judged_backup/126a564f88c72ac7feb2d4cc79e7183348f5aad9';report['judge_coverage']={}
for kind in ('cum','pair'):
 marks=pd.read_parquet(backup/f'bo_{kind}_danh_dau.parquet'); report['judge_coverage'][kind]={'products_with_unjudged_description_units':int(marks.mo_ta_so_manh_chua_cham.gt(0).sum()),'unjudged_description_occurrences':int(marks.mo_ta_so_manh_chua_cham.sum()),'equivalent_occurrences':int(marks.mo_ta_so_manh_judge_tuong_duong.sum())}
 for r in marks[marks.mo_ta_so_manh_chua_cham.gt(0)].itertuples():mark(kind,'',r.product_id,'description','judge_unscored_review')
f=pd.DataFrame(flags,columns=['method','source','product_id','field','signal']);f.to_parquet(OUT/'row_signals.parquet',index=False)
report['signal_rows']=len(f);report['limitations']=['Regex flags are screening signals, not confirmed translation errors.','Full semantic correctness requires evaluation beyond this automated pass.','Description numeric sets are document-level and cannot establish attribute-level alignment.','Title and description repeated across products require group-aware train/evaluation splitting.']
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
