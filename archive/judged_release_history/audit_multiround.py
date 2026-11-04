from pathlib import Path
import json,re,hashlib,collections,unicodedata
import pandas as pd
ROOT=Path(r'D:\download\NCKH\judge_kaggle_handoff');CK=Path(r'D:\download\NCKH\judge_kaggle_preflight\title_v1_output\judge_full\data\processed\so_cap');OUT=Path(r'D:\download\NCKH\judge_kaggle_preflight\audit_multiround');OUT.mkdir(exist_ok=True)
report={'scope':'Local original inputs and judge checkpoints; audit preview is not a final release and 7 blocked items remain.','round1_structure':{},'round2_judge_integrity':{},'semantic_verified':False};evidence=[]
def record(src,pid,field,flag,detail=''):evidence.append({'source':src,'product_id':str(pid),'field':field,'flag':flag,'detail':detail})
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s))).strip().casefold()
def numbers(s):return set(re.findall(r'\d+(?:[.,]\d+)?',str(s)))
frames={}
for kind,file in [('cum','ft_buoc4_giu.parquet'),('pair','ft_buoc4_giu_pair.parquet')]:
 d=pd.read_parquet(ROOT/'data/processed/ft_clean'/file);frames[kind]=d;report['round1_structure'][kind]={'rows':len(d),'columns':list(d.columns),'duplicate_composite_ids':int(d.duplicated(['source','product_id']).sum()),'duplicate_product_ids':int(d.product_id.duplicated().sum()),'nulls':d.isna().sum().astype(int).to_dict(),'source_counts':d.source.value_counts().to_dict()}
report['round1_structure']['same_ids_order']=frames['cum'][['source','product_id']].equals(frames['pair'][['source','product_id']]);report['round1_structure']['same_titles']=frames['cum'][['title_zh','title_vi']].equals(frames['pair'][['title_zh','title_vi']])
lasts={}
for kind,index in [('title','muc_title'),('cum','muc_cum_lai'),('pair','muc_pair')]:
 d=pd.read_parquet(ROOT/'data/processed/so_cap'/(index+'.parquet'))
 if kind=='pair':d=d.sort_values(['san_pham_n','item_id'],ascending=[False,True],kind='stable').head(5000)
 rows=[json.loads(l) for l in (CK/f'ket_qua_lon_{kind}.jsonl').read_text(encoding='utf8').splitlines() if l];last={x['item_id']:x for x in rows};lasts[kind]=last;expected=set(d.item_id);good=[x for x in last.values() if x.get('trang_thai')=='ok'];counts=collections.Counter(x.get('ket_qua') for x in good);calls=[l for x in good for l in x.get('luot',[])];eq=sum(any(l.get('tuong_duong') for l in x.get('luot',[])) for x in good);winners=[x for x in good if not any(l.get('tuong_duong') for l in x.get('luot',[]))];pos=[l for x in winners for l in x.get('luot',[]) if l.get('vi_tri_chon') in [1,2]]
 report['round2_judge_integrity'][kind]={'expected':len(expected),'records':len(rows),'unique':len(last),'duplicates':len(rows)-len(last),'missing':len(expected-set(last)),'unexpected':len(set(last)-expected),'ok':len(good),'errors':len(last)-len(good),'judgements':dict(counts),'equivalent':eq,'equivalent_fraction':eq/max(1,len(good)),'choose_position1_fraction':sum(l.get('vi_tri_chon')==1 for l in pos)/max(1,len(pos)),'rubrics':dict(collections.Counter(x.get('phien_ban') for x in good)),'modes':dict(collections.Counter(x.get('che_do') for x in good))}
 for x in last.values():
  if x.get('trang_thai')!='ok':record(x.get('source'),x.get('vi_du_product_id',x['item_id']),'judge_'+kind,'unresolved_judge_error','provider inspection' if 'data_inspection_failed' in str(x.get('ly_do','')) else 'other')
# Technical language scan. All number/length/language signals are REVIEW ONLY.
for kind,d in frames.items():
 stats=collections.Counter()
 for _,r in d.iterrows():
  src=str(r.source);sc,tc=('title_zh','title_vi') if src=='1688' else ('title_vi','title_zh')
  for field in ['title_zh','title_vi','description_zh','description_vi']:
   s=str(r[field]);flags=[]
   if not s.strip():flags.append('empty_text')
   if '\ufffd' in s or any(unicodedata.category(c)=='Cc' and c not in '\r\n\t' for c in s):flags.append('invalid_unicode_or_control')
   if re.search(r'<\s*/?think\s*>|```|<\|(?:im_start|im_end)\|>',s,re.I):flags.append('generation_marker')
   if field.startswith('title') and re.search(r'Tạm thời không thể dịch|As an AI|I cannot translate|^Translation\s*:',s,re.I):flags.append('model_explanation')
   for f in flags:record(src,r.product_id,kind+':'+field,f);stats[f]+=1
  z=str(r[sc]);v=str(r[tc]);flags=[]
  if numbers(z)!=numbers(v):flags.append('title_numeric_tokens_differ_review')
  if norm(z)==norm(v):flags.append('title_identical_review')
  if src=='1688' and re.search('[一-鿿]',v):flags.append('han_remaining_review')
  if src=='tiki' and re.search('[À-ỹĐđ]',v):flags.append('vietnamese_diacritic_review')
  if src=='tiki' and not re.search('[一-鿿]',v):flags.append('no_han_name_or_missing_translation_review')
  for f in flags:record(src,r.product_id,kind+':'+tc,f);stats[f]+=1
 report['round3_'+kind+'_technical_counts']=dict(stats)
 for col in ['title_zh','title_vi']:
  n=d[col].map(norm);report['round3_'+kind+'_'+col+'_duplicate_rows']=int(n.duplicated(keep=False).sum())
report['round4_methodological_findings']=['Seven context judge items remain provider-blocked; completion cannot be declared.','Pair judges top5000 units only; unjudged units keep current translations.','Single-order judgement and first-position bias prevent claiming unbiased ground truth.','Semantic accuracy is not established by structural checks or automatic judge.','Group and duplicate-description exclusions are design choices, not proof of wrong translation.','Source-only grouping IDs were frozen before judge; they must be used for split leakage prevention.','Original description model labels remain provenance of original content, not attribution of mixed post-judge fragments.']
pd.DataFrame(evidence).to_parquet(OUT/'issue_evidence.parquet',index=False);pd.DataFrame(evidence).to_csv(OUT/'issue_evidence.csv',index=False,encoding='utf-8-sig');(OUT/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='round1_structure'},ensure_ascii=False,indent=2))
