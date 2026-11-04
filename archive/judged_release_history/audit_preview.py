from pathlib import Path
import pandas as pd,json,re,collections
root=Path(r'D:\download\NCKH\judge_kaggle_handoff');p=Path(r'D:\download\NCKH\judge_kaggle_preflight\audit_preview_pending_7');out=Path(r'D:\download\NCKH\judge_kaggle_preflight\audit_multiround');out.mkdir(exist_ok=True);report={};samples=[]
for kind,filename in [('cum','ft_buoc4_giu.parquet'),('pair','ft_buoc4_giu_pair.parquet')]:
 base=pd.read_parquet(root/'data/processed/ft_clean'/filename);d=pd.read_parquet(p/f'bo_{kind}_chon_deepseek.parquet');info={'rows':len(d),'columns':len(d.columns),'same_ids_order':d.product_id.astype(str).tolist()==base.product_id.astype(str).tolist(),'source_fields_unchanged':{},'changes':{},'title_signals':{}}
 for source,cols in [('1688',['title_zh','description_zh']),('tiki',['title_vi','description_vi'])]:
  m=d.source.eq(source)
  for c in cols:info['source_fields_unchanged'][source+':'+c]=d.loc[m,c].astype('string').tolist()==base.loc[m,c].astype('string').tolist()
 for col in ['title_zh','title_vi','description_zh','description_vi']:info['changes'][col]=int((d[col].astype('string')!=base[col].astype('string')).sum())
 for source,sc,tc in [('1688','title_zh','title_vi'),('tiki','title_vi','title_zh')]:
  sub=d[d.source.eq(source)]; flags=[]
  for _,r in sub.iterrows():
   f=[]
   if set(re.findall(r'\d+(?:[.,]\d+)?',r[sc]))!=set(re.findall(r'\d+(?:[.,]\d+)?',r[tc])):f.append('numeric_difference')
   if source=='1688' and re.search('[一-鿿]',r[tc]):f.append('han_remaining')
   if source=='tiki' and re.search('[À-ỹĐđ]',r[tc]):f.append('vietnamese_diacritic')
   if source=='tiki' and not re.search('[一-鿿]',r[tc]):f.append('no_han')
   flags.extend(f)
   if f and kind=='cum':samples.append({'product_id':r.product_id,'source':source,'src':r[sc],'target':r[tc],'flags':';'.join(f)})
  info['title_signals'][source]=dict(collections.Counter(flags))
 report[kind]=info
 for field in ['desc_so_manh_da_doi']:
  marks=pd.read_parquet(p/f'bo_{kind}_danh_dau.parquet');info['change_count_consistent']=d[field].astype(int).tolist()==marks.mo_ta_so_manh_da_doi.astype(int).tolist()
(out/'preview_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');pd.DataFrame(samples).to_parquet(out/'postjudge_title_review.parquet',index=False);print(json.dumps(report,ensure_ascii=False,indent=2))
