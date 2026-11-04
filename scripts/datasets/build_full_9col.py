

def main():

    # Repository layout bootstrap; translation policy is unchanged.
    import sys as _repo_sys
    from pathlib import Path as _RepoPath
    _repo_root = _RepoPath(__file__).resolve().parents[2]
    if (_repo_root / "project_bootstrap.py").is_file():
        _repo_sys.path.insert(0, str(_repo_root))
        from project_bootstrap import configure_paths as _configure_paths
        _configure_paths()
    import os,json,hashlib
    from pathlib import Path
    import pandas as pd
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars();root=Path(__file__).resolve().parent;os.chdir(root)
    api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha
    out=root/'outputs/full_9col_final';out.mkdir(parents=True,exist_ok=True)
    cols=['product_id','source','category','title_zh','title_vi','description_zh','description_vi','title_ngon_dich','desc_nguon_dich']
    allowed_desc={'gemma4','qwen3_14b_awq','gemma4+qwen3_14b_awq','tu_dien_quy_tac'};allowed_model={'gemma4','qwen3_14b_awq'}
    ops=[];report=[]
    specs=[('1688',16348,6,'zh_vi','zh_to_vi'),('tiki',13200,13,'vi_zh','vi_to_zh')]
    for ds,rows,missing,direction,folder in specs:
     def download(path):return Path(hf_hub_download(repo,path,repo_type='dataset',revision=head,token=os.environ['HF_TOKEN']))
     titlepath=download(f'data/{ds}/title_translations/full_9col_title_input.parquet');title=pd.read_parquet(titlepath)
     bestname='1688_zh_to_vi_best_cometkiwi.parquet' if ds=='1688' else 'tiki_vi_to_zh_best_cometkiwi.parquet'
     best=pd.read_parquet(download(f'data/{ds}/title_translations/best_cometkiwi/{bestname}'))
     cleaned=pd.read_parquet(download('data/1688/cleaned/bilingual_zh_vi_cado.parquet' if ds=='1688' else 'data/tiki/cleaned/tiki_vi_cado.parquet'),columns=['product_id','category'])
     for x in [title,best,cleaned]:x['product_id']=x.product_id.astype('string')
     assert len(title)==len(best)==len(cleaned)==rows and title.product_id.is_unique and best.product_id.is_unique and cleaned.product_id.is_unique
     for col in ['product_id','title_zh','title_vi']:assert title[col].equals(best[col]),f'title_alignment:{ds}:{col}'
     category=cleaned.set_index('product_id').category;assert title.category.tolist()==title.product_id.map(category).tolist()
     valid_cats={'auto','bags','beauty','electronics','fashion','food','home','mother_baby','shoes'} if ds=='1688' else {'auto','beauty','electronics','food','home','mother_baby'}
     assert title.category.notna().all() and set(title.category)<=valid_cats and title.title_ngon_dich.isin(allowed_model).all()
     for method in ['independent','context']:
      descfolder=folder if method=='independent' else folder+'_context'
      descname=f'description_products_{direction}_pair.parquet' if method=='independent' else f'description_products_{direction}.parquet'
      descpath=download(f'data/{ds}/descriptions/{descfolder}/{descname}');desc=pd.read_parquet(descpath)
      desc['product_id']=desc.product_id.astype('string');assert desc.product_id.is_unique and len(desc)==rows-missing
      assert desc[['description_zh','description_vi','desc_nguon_dich']].notna().all().all() and set(desc.desc_nguon_dich)<=allowed_desc
      check=title[['product_id','category']].merge(desc[['product_id','category']],on='product_id',how='inner',validate='one_to_one',suffixes=('_title','_desc'))
      assert len(check)==len(desc) and check.category_title.equals(check.category_desc),f'category_mismatch:{ds}:{method}'
      result=title.merge(desc[['product_id','description_zh','description_vi','desc_nguon_dich']],on='product_id',how='left',sort=False,validate='one_to_one')[cols]
      for col in cols:result[col]=result[col].astype('string')
      assert len(result)==rows and result.product_id.is_unique and result.product_id.equals(title.product_id)
      assert result[['product_id','source','category','title_zh','title_vi','title_ngon_dich']].notna().all().all()
      assert result.description_zh.isna().sum()==result.description_vi.isna().sum()==result.desc_nguon_dich.isna().sum()==missing
      filename=f'{ds}_full_attributes_{method}_{direction}.parquet';path=out/filename;result.to_parquet(path,index=False);pd.testing.assert_frame_equal(result,pd.read_parquet(path))
      remote=f'data/{ds}/full_dataset/{filename}';ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
      report.append({'dataset':ds,'method':method,'path':remote,'rows':rows,'columns':cols,'missing_descriptions':missing,'category_mismatch':0,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'title_input_sha256':hashlib.sha256(titlepath.read_bytes()).hexdigest(),'description_input_sha256':hashlib.sha256(descpath.read_bytes()).hexdigest()})
    for ds,rows,missing,direction,folder in specs:
     r=[x for x in report if x['dataset']==ds]
     note=f'''# {ds}: full title + description, 9 cột\n\nHai file ghép theo `product_id` với phép ghép trái, giữ nguyên thứ tự bảng title: {rows:,} dòng, ID không trùng. Chiều dịch: {'Trung → Việt' if ds=='1688' else 'Việt → Trung'}.\n\n| File | Cách description | SHA256 |\n|---|---|---|\n'''
     for x in r:note+=f"| [{Path(x['path']).name}]({Path(x['path']).name}) | {'Từng pair' if x['method']=='independent' else 'Cả cụm thuộc tính theo ngữ cảnh'} | `{x['sha256']}` |\n"
     note+='''\n## Cột theo đúng thứ tự\n\n`product_id`, `source`, `category`, `title_zh`, `title_vi`, `description_zh`, `description_vi`, `title_ngon_dich`, `desc_nguon_dich`. Tất cả kiểu string; ba cột description có thể null. `source` cố định theo bộ; category lấy từ cleaned, không lấy từ description.\n\n`title_ngon_dich` ghi model tạo ứng viên gốc `gemma4` hoặc `qwen3_14b_awq`. `desc_nguon_dich` do pipeline description cung cấp, gồm `gemma4`, `qwen3_14b_awq`, `gemma4+qwen3_14b_awq` hoặc `tu_dien_quy_tac`; từ điển/quy tắc có thể cùng đóng góp khi cột nêu model.\n\nTitle chọn dựa trên COMETKiwi, sau đó có sửa kỹ thuật; 469 title 1688 dùng `existing_alternative` được chọn bản điểm thấp hơn. Dữ liệu là bản dịch máy; chưa xác minh ngữ nghĩa thủ công toàn bộ (`semantic_verified=false` trong tài liệu thành phần). Không thêm cột điểm hoặc ứng viên vào file full.\n'''
     if ds=='1688':note+='\nCó 9 title đã chỉnh thủ công sau dịch (8 từ Qwen, 1 từ Gemma); `title_ngon_dich` ghi model của bản gốc, không có nghĩa text cuối nguyên văn model.\n'
     note+=f'\nCó {missing} sản phẩm không có trong nguồn description; cả hai description và `desc_nguon_dich` để null, không bỏ dòng. Category vẫn có từ cleaned.\n'
     note+='\nFile theo pair dùng bảng sản phẩm pair; file theo cụm dùng bảng sản phẩm cụm do NA cung cấp. Cụm là bảng thuộc tính cùng ngữ cảnh sản phẩm, không phải toàn bộ văn xuôi/hình ảnh. Không sửa bảng description nguồn.\n'
     readme=out/(ds+'_README.md');readme.write_text(note,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo=f'data/{ds}/full_dataset/README.md',path_or_fileobj=str(readme)))
     manifest=out/(ds+'_manifest.json');manifest.write_text(json.dumps({'schema':cols,'source_revision':head,'exports':r},indent=2),encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo=f'data/{ds}/full_dataset/manifest.json',path_or_fileobj=str(manifest)))
    c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Merge verified nine-column 1688 and Tiki title and description datasets')
    for r in report:
     p=download if False else Path(hf_hub_download(repo,r['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']))
     assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];v=pd.read_parquet(p);assert list(v.columns)==cols and len(v)==r['rows']
    result={'revision':c.oid,'source_revision':head,'exports':report};(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__ == "__main__":
    main()
