

def main():

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
    import numpy as np
    import pandas as pd
    import title_translation_pipeline as zhqa
    from tiki_translation_pipeline import qa_check_vi_to_zh
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars();root=Path(__file__).resolve().parent;os.chdir(root)
    api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha
    out=root/'outputs/title_review_labels';out.mkdir(parents=True,exist_ok=True);ops=[];summaries=[]
    viet=r'[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]'
    def download(path):return Path(hf_hub_download(repo,path,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']))
    def add(path,frame):
     p=out/path;p.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(p,index=False)
     assert pd.read_parquet(p).equals(frame)
     ops.append(CommitOperationAdd(path_in_repo=path,path_or_fileobj=str(p)))
     return hashlib.sha256(p.read_bytes()).hexdigest()
    for ds in ['1688','tiki']:
     source='title_zh' if ds=='1688' else 'title_vi';target='title_vi' if ds=='1688' else 'title_zh';name=f'{ds}_{"zh_to_vi" if ds=="1688" else "vi_to_zh"}_best_cometkiwi.parquet';remote=f'data/{ds}/title_translations/best_cometkiwi/{name}'
     best=pd.read_parquet(download(remote));unique=best.drop_duplicates(source);scores=pd.to_numeric(unique.selected_cometkiwi_score,errors='coerce').dropna();cutoff=float(scores.quantile(.10))
     labels={};counts=Counter()
     for r in unique.to_dict('records'):
      s,t=r[source],r[target];issues=[]
      q=zhqa.check_translation(s,t) if ds=='1688' else qa_check_vi_to_zh(s,t)
      flags=q.get('hard_flags',[]) if ds=='1688' else q.get('flags',[]);classes={f.split(':',1)[0] for f in flags}
      if (ds=='1688' and re.search(r'[\u3400-\u9fff]',t or '')) or (ds=='tiki' and re.search(viet,t or '')):issues.append('con_ky_tu_nguon_can_xem')
      if classes & {'number_mismatch','currency_amount_mismatch','mat_so_lieu'}:issues.append('nghi_lech_so_lieu')
      if classes & {'model_code_missing'}:issues.append('nghi_thieu_ma_san_pham')
      if classes & {'bi_cat_cut','ao_giac_lap_tu','jin_sai_spam','explanation_or_multiple_lines'}:issues.append('nghi_cat_cut_hoac_lap')
      if ds=='tiki' and 'ti_le_chu_han_thap' in classes:issues.append('ti_le_chu_trung_thap_can_xem')
      if ds=='1688' and re.search(r'(?i)\b([a-zà-ỹ]{3,})\s+\1\b',t or ''):issues.append('nghi_lap_tu')
      score=r.get('selected_cometkiwi_score');selected=r.get('selected_model');other='qwen3_14b_awq' if selected=='gemma4' else 'gemma4' if selected=='qwen3_14b_awq' else None
      # Only compare when the selected text is exactly the scored candidate and the other generation is valid.
      if other and r.get(selected+'_translation')==t and r.get(other+'_generation_ok') and pd.notna(score):
       alt=r.get('cometkiwi_'+other)
       if pd.notna(alt) and float(alt)>float(score)+1e-9:issues.append('ung_vien_kia_diem_cao_hon')
      if pd.notna(score) and float(score)<cutoff and r.get(selected+'_translation')==t:issues.append('diem_comet_thap_tuong_doi')
      labels[s]=';'.join(issues) if issues else 'chua_phat_hien_dau_hieu_ky_thuat'
      counts.update(issues)
     best['title_van_de']=best[source].map(labels).astype('string');best_sha=add(remote,best)
     manifests=[]
     for method in ['independent','context']:
      direction='zh_vi' if ds=='1688' else 'vi_zh';f=f'data/{ds}/full_dataset/{ds}_full_attributes_{method}_{direction}.parquet';old=pd.read_parquet(download(f));old['product_id']=old.product_id.astype('string');comparison=best.copy();comparison['product_id']=comparison.product_id.astype('string');expected=old[['product_id',source,target]].merge(comparison[['product_id',source,target,'title_van_de']],on='product_id',validate='one_to_one',sort=False,suffixes=('_full','_best'))
      assert expected[source+'_full'].astype('string').equals(expected[source+'_best'].astype('string')) and expected[target+'_full'].astype('string').equals(expected[target+'_best'].astype('string'))
      old['title_van_de']=expected.title_van_de.astype('string');sha=add(f,old);manifests.append({'path':f,'sha256':sha,'rows':len(old),'columns':list(old.columns)})
     counts_all=sum(v!='chua_phat_hien_dau_hieu_ky_thuat' for v in labels.values());summary={'dataset':ds,'unique_titles':len(unique),'flagged_unique':counts_all,'label_counts_unique':dict(counts),'relative_low_score_bottom_10_percent_cutoff':cutoff,'source_revision':head,'semantic_verified':False,'best_sha256':best_sha,'full_exports':manifests}
     summaries.append(summary)
     p=out/f'{ds}_summary.json';p.write_text(json.dumps(summary,indent=2),encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=f'data/{ds}/title_translations/best_cometkiwi/title_van_de_summary.json',path_or_fileobj=str(p)))
     for doc in [f'data/{ds}/full_dataset/README.md',f'data/{ds}/title_translations/best_cometkiwi/README.md']:
      s=download(doc).read_text(encoding='utf8');s=s.replace('9 cột','10 cột').replace('đúng thứ tự\n\n','đúng thứ tự\n\n')
      s+=f'''\n\n## Cột `title_van_de`\n\nMột title có thể có nhiều nhãn, phân cách bằng `;`. `con_ky_tu_nguon_can_xem` là dấu hiệu còn chữ Hán ở 1688 hoặc ký tự Việt có dấu ở Tiki; tên thương hiệu/tên riêng có thể hợp lệ. Các nhãn khác gồm `nghi_lech_so_lieu`, `nghi_thieu_ma_san_pham`, `nghi_cat_cut_hoac_lap`, `ti_le_chu_trung_thap_can_xem`, `nghi_lap_tu`, `ung_vien_kia_diem_cao_hon` và `diem_comet_thap_tuong_doi` khi có đủ dữ liệu tương ứng. Điểm thấp tương đối là dưới phân vị 10% của điểm ứng viên được chọn trong bộ này ({cutoff:.6f}), không phải ngưỡng đúng/sai. Ứng viên kia cao điểm chỉ được đánh dấu nếu bản hiện tại đúng nguyên văn bản đã được chấm và ứng viên kia có trạng thái sinh hợp lệ. `chua_phat_hien_dau_hieu_ky_thuat` không bảo đảm đúng ngữ nghĩa. Các nhãn là gợi ý rà soát, không phải kết luận đã xác minh. Chưa tự đánh dấu chính tả, mơ hồ hoặc lệch ngành vì chưa có bộ kiểm chứng đáng tin cậy.\n'''
      p=out/doc;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo=doc,path_or_fileobj=str(p)))
     manifestpath=f'data/{ds}/full_dataset/manifest.json';manifest=json.loads(download(manifestpath).read_text(encoding='utf-8'))
     for e in manifest.get('exports',[]):
      for updated in manifests:
       if e.get('path')==updated['path']:e['sha256']=updated['sha256'];e['columns']=updated['columns']
     manifest['schema']=manifests[0]['columns'];p=out/manifestpath;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(manifest,indent=2),encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=manifestpath,path_or_fileobj=str(p)))
    c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Add transparent title review indicator to combined title and full datasets')
    for s in summaries:
     for e in s['full_exports']:
      p=Path(hf_hub_download(repo,e['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']));assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
    result={'revision':c.oid,'summaries':summaries};(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__ == "__main__":
    main()
