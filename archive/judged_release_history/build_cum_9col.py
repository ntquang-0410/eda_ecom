from pathlib import Path
import sys,json,inspect,collections,hashlib
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download
R=Path(r'D:\download\NCKH\judge_kaggle_handoff');sys.path.insert(0,str(R/'scripts/judge'));import ghep_bo as G
B=Path(r'D:\download\NCKH\judge_kaggle_preflight');OUT=B/'cum_9col_judge_update';OUT.mkdir(exist_ok=True);repo='ntquang0410/zh-vie_ecom';prefix='data/LLM/bilingual_judged/';api=HfApi();head=api.dataset_info(repo).sha
old_path=Path(hf_hub_download(repo,prefix+'bo_cum_chon_deepseek.parquet',repo_type='dataset',revision=head));old=pd.read_parquet(old_path);assert len(old)==25575 and len(old.columns)==17
ck=B/'title_v1_output/judge_full/data/processed/so_cap';kq,eq=G.doc_ket_qua_va_tuong_duong('lon_cum',thu_muc=ck);qd,kq=G.ap_chinh_sach_tuong_duong(kq,eq,'giu_hien_tai')
review_path=Path(hf_hub_download(repo,prefix+'seven_items_assistant_review.parquet',repo_type='dataset',revision=head));review=pd.read_parquet(review_path);direct=set()
for _,r in review.iterrows():
 if r.assistant_decision in ['gemma','qwen']:qd[r.item_id]=r.assistant_decision;direct.add(r.item_id)
trace=[]
source=inspect.getsource(G.ghep_mota_cum)
needle='        if dau is not None and'
insert="""        if c in ('gemma','qwen'):
            candidate=a if c=='gemma' else b
            matches_current=chuan(cu) in (chuan(a),chuan(b))
            accepted=matches_current and (chuan(candidate)==chuan(cu) or an_toan(src,candidate,cu))
            audit_trace.append({'source':src,'product_id':str(pid),'item_id':ids[i],'selected_model':c,'accepted':bool(accepted),'text_changed':bool(da_doi),'review_origin':'current_assistant' if ids[i] in direct_review_ids else 'deepseek','reason':'accepted' if accepted else 'locked_current_not_candidate' if not matches_current else 'safety_guard_rejected'})
"""
assert source.count(needle)==1;source=source.replace(needle,insert+needle);ns=G.__dict__;ns['audit_trace']=trace;ns['direct_review_ids']=direct;exec(compile(source,'instrumented_merge','exec'),ns)
base=pd.read_parquet(R/'data/processed/ft_clean/ft_buoc4_giu.parquet');pairs=G.nap_cap_cum()
for src in ['1688','tiki']:
 generated,_=G.ghep_mota_cum(base[base.source.eq(src)],pairs[src],qd,'giu_hien_tai',ngu_canh='lai');current=old[old.source.eq(src)].set_index('product_id');field='description_vi' if src=='1688' else 'description_zh'
 assert all(str(text)==str(current.loc[pid,field]) for pid,text in generated.items()),'Description replay differs; abort'
t=pd.DataFrame(trace);assert not t.duplicated(['source','product_id','item_id']).any();accepted=t[t.accepted];counts=accepted.groupby(['source','product_id','selected_model']).size().to_dict()
textcols=['product_id','source','category','title_zh','title_vi','description_zh','description_vi'];new=old[textcols].copy();new['judge_chon_title']=old.title_nguon_dich.map({'gemma4':'gemma','qwen3_14b_awq':'qwen'});assert new.judge_chon_title.notna().all();new['judge_chon_description']=[f"gemma: {counts.get((s,str(pid),'gemma'),0)} | qwen: {counts.get((s,str(pid),'qwen'),0)}" for s,pid in zip(old.source,old.product_id)]
assert new[textcols].equals(old[textcols]);assert list(new.columns)==textcols+['judge_chon_title','judge_chon_description']
new.to_parquet(OUT/'bo_cum_chon_deepseek.parquet',index=False);meta=old[['source','product_id']+[c for c in old if c not in textcols]].copy();meta.to_parquet(OUT/'bo_cum_metadata.parquet',index=False);t.to_parquet(OUT/'bo_cum_fragment_selection_trace.parquet',index=False)
mark_path=Path(hf_hub_download(repo,prefix+'bo_cum_danh_dau.parquet',repo_type='dataset',revision=head));marks=pd.read_parquet(mark_path);assert marks.product_id.astype(str).tolist()==new.product_id.astype(str).tolist()
for model in ['gemma','qwen']:marks['mo_ta_so_manh_'+model+'_duoc_ap_dung']=[counts.get((s,str(pid),model),0) for s,pid in zip(new.source,new.product_id)]
marks['mo_ta_so_manh_judge_khong_ap_dung']=[int(((t.source==s)&(t.product_id==str(pid))&~t.accepted).sum()) for s,pid in zip(new.source,new.product_id)] if False else 0
rejected=t[~t.accepted].groupby(['source','product_id']).size().to_dict();marks['mo_ta_so_manh_judge_khong_ap_dung']=[rejected.get((s,str(pid)),0) for s,pid in zip(new.source,new.product_id)];marks.to_parquet(OUT/'bo_cum_danh_dau.parquet',index=False)
for model in ['gemma','qwen']:assert marks['mo_ta_so_manh_'+model+'_duoc_ap_dung'].sum()==int((accepted.selected_model==model).sum())
assert int(accepted.text_changed.sum())==int(old.desc_so_manh_da_doi.sum());report={'source_revision':head,'rows':len(new),'columns':list(new),'translation_content_unchanged':True,'accepted_gemma_fragments':int((accepted.selected_model=='gemma').sum()),'accepted_qwen_fragments':int((accepted.selected_model=='qwen').sum()),'accepted_current_assistant_fragments':int((accepted.review_origin=='current_assistant').sum()),'rejected_recommendation_occurrences':int((~t.accepted).sum()),'zero_model_selection_rows':int(new.judge_chon_description.eq('gemma: 0 | qwen: 0').sum())};(OUT/'schema_change_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False))
