from pathlib import Path
import sys,json,re,hashlib,shutil
import pandas as pd
R=Path(r'D:\download\NCKH\judge_kaggle_handoff');sys.path.insert(0,str(R/'scripts/ft_clean'));import luat_ft as L
B=Path(r'D:\download\NCKH\judge_kaggle_preflight');src=B/'audit_preview_pending_7';out=B/'corrected_reviewed_release';out.mkdir(exist_ok=True)
a=pd.read_parquet(src/'bo_cum_chon_deepseek.parquet');b=pd.read_parquet(src/'bo_pair_chon_deepseek.parquet');orig=pd.read_parquet(R/'data/processed/ft_clean/ft_buoc4_giu_pair.parquet');before=pd.read_parquet(R/'data/processed/ft_clean/ft_buoc4_giu.parquet');assert a.product_id.tolist()==b.product_id.tolist()==orig.product_id.tolist()==before.product_id.tolist()
# Method-specific metadata, frozen on original pair text rather than judge-rewritten text.
def segments(s):return str(s).splitlines()
def template(source,z,v):
 text=z if source=='1688' else v
 labels=[re.split('[:：]',s,maxsplit=1)[0].strip().casefold() if re.search('[:：]',s) else '' for s in segments(text)]
 return hashlib.md5((source+'#'+'|'.join(labels)).encode()).hexdigest()[:12]
def aligned(z,v):
 zs,vs=segments(z),segments(v)
 return len(zs)==len(vs) and all(L._so(x)==L._so(y) for x,y in zip(zs,vs))
b['huong_goc']=orig.source.map({'1688':'zh->vi','tiki':'vi->zh'})
b['khuon_id']=[template(s,z,v) for s,z,v in zip(orig.source,orig.description_zh,orig.description_vi)]
b['nhom_mo_ta_id']=[L.nhom_mo_ta_id(z,v) for z,v in zip(orig.description_zh,orig.description_vi)]
b['can_hang_chac']=[aligned(z,v) for z,v in zip(orig.description_zh,orig.description_vi)]
b['mem_latin_la_ban_trung']=[L.latin_la(s,z,v) for s,z,v in zip(orig.source,orig.description_zh,orig.description_vi)]
b['mem_viet_sot_title_zh']=[L.viet_sot_title_zh(s,z) for s,z in zip(orig.source,orig.title_zh)]
b['mem_v2_review']=before.mem_v2_review.to_numpy();b=b[list(a.columns)]
patches=[]
for pid,col,oldfrag,newfrag,reason in [('41907586179','title_vi','SNJUE8 Súng','SNJUE8 Súng 8 trong 1','Restore explicit 8-in-1 function from 8合1'),('275246947','title_zh','日本产地垫床','出口日本的地铺床垫','Restore export-to-Japan claim; do not claim Japanese manufacture')]:
 mask=a.product_id.astype(str).eq(pid);assert mask.sum()==1;old=a.loc[mask,col].iloc[0];assert oldfrag in old;new=old.replace(oldfrag,newfrag,1)
 for frame in [a,b]:frame.loc[frame.product_id.astype(str).eq(pid),col]=new
 patches.append({'product_id':pid,'field':col,'before':old,'after':new,'reason':reason,'stage':'manual_semantic_correction','draft_model':a.loc[mask,'title_nguon_dich'].iloc[0]})
pd.DataFrame(patches).to_parquet(out/'manual_title_corrections.parquet',index=False)
# Preserve all seven unresolved records; no fabricated judge choices.
ck=B/'title_v1_output/judge_full/data/processed/so_cap';pending=[]
for line in (ck/'ket_qua_lon_cum.jsonl').read_text(encoding='utf8').splitlines():
 x=json.loads(line)
 if x.get('trang_thai')!='ok':pending.append({'item_id':x['item_id'],'status':'unjudged_provider_blocked','fallback':'keep_current_translation','reason_class':'data_inspection_failed'})
assert len(pending)==7;pd.DataFrame(pending).to_parquet(out/'pending_judge_items.parquet',index=False)
for method,frame in [('cum',a),('pair',b)]:
 base=pd.read_parquet(R/'data/processed/ft_clean'/('ft_buoc4_giu.parquet' if method=='cum' else 'ft_buoc4_giu_pair.parquet'))
 assert len(frame)==25575 and len(frame.columns)==17 and frame.product_id.is_unique and not frame.isna().any().any()
 for source,cols in [('1688',['title_zh','description_zh']),('tiki',['title_vi','description_vi'])]:
  m=frame.source.eq(source)
  for c in cols:assert frame.loc[m,c].tolist()==base.loc[m,c].tolist(),(method,source,c)
 # No descriptions or attributes were changed during repair.
 prev=pd.read_parquet(src/f'bo_{method}_chon_deepseek.parquet')
 for c in ['description_zh','description_vi','category','desc_so_manh_da_doi']:assert frame[c].tolist()==prev[c].tolist(),(method,c)
 frame.to_parquet(out/f'bo_{method}_chon_deepseek.parquet',index=False)
 shutil.copyfile(src/f'bo_{method}_danh_dau.parquet',out/f'bo_{method}_danh_dau.parquet')
assert a[['product_id','source','title_zh','title_vi']].equals(b[['product_id','source','title_zh','title_vi']])
readme='''# Corrected review copy — pending judge items disclosed

Two 25,575-row, 17-column datasets, identical product order and bilingual titles. This is a reviewed local output, not an HF publication and not a claim of complete semantic verification.

Repairs: seven missing pair metadata columns are computed from ORIGINAL pair text before judge. Newline-separated attributes are used for pair templates/alignment; context metadata is not blindly copied. Direction and the original v2 review membership are retained consistently. Metadata is frozen and not recomputed from judge-modified text.

Two source-grounded manual title corrections are documented in manual_title_corrections.parquet. title_nguon_dich identifies the model producing the draft; for those two rows, the final text also includes the documented manual patch. Judge marking files preserve original automated decisions and do not falsely attribute manual corrections to DeepSeek. Use the correction sidecar for complete provenance.

Seven context items were blocked by provider content inspection. Current translations are kept, with unjudged_provider_blocked status in pending_judge_items.parquet; no winner or score was fabricated and no content filter bypass was attempted. Other unjudged pair units also keep their prior text under the original top-5000 coverage policy.

Descriptions, source-language fields, product IDs, categories, row count and ordering are preserved during these repairs. Both context/pair change-count sidecars remain available. Remaining number/language warnings are review signals, not automatically confirmed errors; full semantic accuracy is not certified. Group splitting must keep original template/description groups together and should also consider duplicates across methods.
''';(out/'README.md').write_text(readme,encoding='utf8');report={'rows_each':25575,'columns_each':17,'manual_title_corrections':2,'pair_metadata_fields_added':7,'unjudged_provider_blocked':7,'source_fields_preserved':True,'description_fields_preserved':True,'semantic_verified':False,'published_to_hf':False};(out/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf8');files=list(out.iterdir());manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']};(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');(out/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt'),encoding='ascii');print(json.dumps(report))
