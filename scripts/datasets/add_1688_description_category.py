

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
    disable_progress_bars();src=Path('outputs/1688_full_sources');out=Path('outputs/1688_full_category');out.mkdir(parents=True,exist_ok=True);ops=[];results=[]
    for method,source in [('attributes_independent','description_pairs_zh_vi.parquet'),('attributes_context','description_products_zh_vi.parquet')]:
     d=pd.read_parquet(Path((src/(source+'.path')).read_text()));assert (d.groupby('product_id').category.nunique(dropna=False)<=1).all();cat=d[['product_id','category']].drop_duplicates('product_id')
     name='1688_full_'+method+'_zh_vi.parquet';old=pd.read_parquet(Path('outputs/1688_full_minimal')/name);new=old.merge(cat,on='product_id',how='left',sort=False,validate='one_to_one');assert len(new)==len(old)==16348
     for col in old.columns:pd.testing.assert_series_equal(new[col],old[col])
     assert new.category.notna().sum()==16342
     f=out/name;new.to_parquet(f,index=False);pd.testing.assert_frame_equal(new,pd.read_parquet(f));remote='data/1688/full_dataset/'+name;ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(f)));results.append({'path':remote,'rows':len(new),'columns':list(new.columns),'category_missing':int(new.category.isna().sum()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    readme='''# 1688 full: title + description Trung–Việt\n\nMỗi file có 16.348 sản phẩm và 6 cột: `product_id`, `title_zh`, `title_vi`, `description_zh`, `description_vi`, `category`. Category lấy từ nguồn description tương ứng, ghép theo `product_id`. Sáu sản phẩm không có trong nguồn description giữ title; description và category để null.\n\n- `1688_full_attributes_independent_zh_vi.parquet`: ghép các pair thuộc tính được dịch độc lập theo thứ tự pair_id. Nguồn: [dịch theo pair](../descriptions/zh_to_vi/README.md).\n- `1688_full_attributes_context_zh_vi.parquet`: description cấp sản phẩm từ pipeline dịch cả cụm thuộc tính với ngữ cảnh. Nguồn: [dịch theo cụm](../descriptions/zh_to_vi_context/README.md). Cả cụm là bảng thuộc tính, không phải toàn bộ văn xuôi hoặc hình ảnh.\n\nTitle dùng bản chọn bằng COMETKiwi. Không dịch lại khi ghép. Điểm, ứng viên, QA và pair chi tiết nằm ở các bộ thành phần. Dữ liệu dịch máy chưa xác minh ngữ nghĩa thủ công toàn bộ.\n'''
    r=out/'README.md';r.write_text(readme,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo='data/1688/full_dataset/README.md',path_or_fileobj=str(r)))
    m=out/'manifest.json';m.write_text(json.dumps({'exports':results,'category_source':'description dataset; joined by product_id'},indent=2));ops.append(CommitOperationAdd(path_in_repo='data/1688/full_dataset/manifest.json',path_or_fileobj=str(m)))
    api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha;c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Add description category to both 1688 full datasets')
    for item in results:
     f=hf_hub_download(repo,item['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']);assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==item['sha256']
    result={'revision':c.oid,'exports':results};(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__ == "__main__":
    main()
