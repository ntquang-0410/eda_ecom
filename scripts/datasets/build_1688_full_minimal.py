

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
    disable_progress_bars();root=Path.cwd();src=root/'outputs/1688_full_sources';out=root/'outputs/1688_full_minimal';out.mkdir(parents=True,exist_ok=True)
    def load(n):
     p=Path((src/(n+'.path')).read_text());return pd.read_parquet(p),hashlib.sha256(p.read_bytes()).hexdigest()
    a,asha=load('description_pairs_zh_vi.parquet');b,bsha=load('description_pairs_zh_vi_ctx.parquet');p,psha=load('description_products_zh_vi.parquet');t=pd.read_parquet('outputs/comet_best_all_titles/1688_zh_to_vi_best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet')
    assert len(t)==16348 and t.product_id.is_unique and len(p)==16342 and p.product_id.is_unique
    assert set(a.product_id)==set(b.product_id)==set(p.product_id) and set(p.product_id)<=set(t.product_id)
    assert a.pair_id.is_unique
    x=a.sort_values('pair_id',kind='stable').copy()
    for lang in ['zh','vi']:x['line_'+lang]=x['name_'+lang].fillna('').astype(str)+': '+x['gia_tri_'+lang].fillna('').astype(str)
    ind=x.groupby('product_id',sort=False).agg(description_zh=('line_zh','\n'.join),description_vi=('line_vi','\n'.join)).reset_index()
    assert len(ind)==16342
    api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha;ops=[];results=[]
    for name,desc in [('attributes_independent',ind),('attributes_context',p[['product_id','description_zh','description_vi']])]:
     d=t[['product_id','title_zh','title_vi']].merge(desc,on='product_id',how='left',validate='one_to_one',sort=False);d=d[['product_id','title_zh','title_vi','description_zh','description_vi']]
     assert len(d)==16348 and d.product_id.equals(t.product_id) and d.title_zh.equals(t.title_zh) and d.title_vi.equals(t.title_vi)
     assert int((d.description_zh.isna()|d.description_vi.isna()).sum())==6
     f=out/('1688_full_'+name+'_zh_vi.parquet');d.to_parquet(f,index=False);pd.testing.assert_frame_equal(d,pd.read_parquet(f));remote='data/1688/full_dataset/'+f.name;ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(f)));results.append({'path':remote,'rows':len(d),'unique_titles':int(d.title_zh.nunique()),'description_missing':6,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'method':name})
    readme='''# 1688 full: title + description Trung–Việt\n\nMỗi bản gồm 16.348 sản phẩm và 5 cột: `product_id`, `title_zh`, `title_vi`, `description_zh`, `description_vi`. Title sử dụng bản chọn bằng COMETKiwi, giữ bản có điểm cao hơn giữa các ứng viên sinh hợp lệ. 16.342 sản phẩm có description; 6 sản phẩm giữ title với description null.\n\n- `1688_full_attributes_independent_zh_vi.parquet`: description tạo bằng cách nối các cặp thuộc tính được dịch độc lập theo thứ tự pair_id. Nguồn: `../descriptions/zh_to_vi/`.\n- `1688_full_attributes_context_zh_vi.parquet`: description cấp sản phẩm của pipeline dịch cả cụm thuộc tính theo ngữ cảnh; giữ nguyên các chuỗi đã phát hành. Nguồn: `../descriptions/zh_to_vi_context/`. Cả cụm là bảng thuộc tính, không phải toàn bộ văn xuôi hoặc hình ảnh.\n\nKhông dịch lại khi ghép. Điểm, ứng viên, QA và pair chi tiết nằm trong các bộ thành phần. Dữ liệu dịch máy chưa được xác minh ngữ nghĩa thủ công toàn bộ.\n'''
    r=out/'README.md';r.write_text(readme,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo='data/1688/full_dataset/README.md',path_or_fileobj=str(r)))
    m=out/'manifest.json';m.write_text(json.dumps({'exports':results,'independent_pairs_sha256':asha,'context_pairs_sha256':bsha,'context_products_sha256':psha},indent=2));ops.append(CommitOperationAdd(path_in_repo='data/1688/full_dataset/manifest.json',path_or_fileobj=str(m)))
    c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Add 1688 title and description full datasets for both translation methods')
    for z in results:
     f=hf_hub_download(repo,z['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']);assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==z['sha256']
    result={'revision':c.oid,'exports':results};(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__ == "__main__":
    main()
