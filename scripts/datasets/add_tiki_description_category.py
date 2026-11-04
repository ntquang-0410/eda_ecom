

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
    disable_progress_bars();src=Path('outputs/tiki_full_sources');out=Path('outputs/tiki_full_category');out.mkdir(parents=True,exist_ok=True);ops=[];results=[]
    for method,source in [('attributes_independent','description_pairs_vi_zh.parquet'),('attributes_context','description_products_vi_zh.parquet')]:
     d=pd.read_parquet(Path((src/(source+'.path')).read_text()));assert (d.groupby('product_id').category.nunique(dropna=False)<=1).all();cat=d[['product_id','category']].drop_duplicates('product_id')
     name='tiki_full_'+method+'_vi_zh.parquet';old=pd.read_parquet(Path('outputs/tiki_full_minimal')/name);new=old.merge(cat,on='product_id',how='left',sort=False,validate='one_to_one');assert len(new)==len(old)==13200
     for col in old.columns:pd.testing.assert_series_equal(new[col],old[col])
     assert new.category.notna().sum()==13187
     f=out/name;new.to_parquet(f,index=False);pd.testing.assert_frame_equal(new,pd.read_parquet(f));remote='data/tiki/full_dataset/'+name;ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(f)));results.append({'path':remote,'rows':len(new),'columns':list(new.columns),'category_missing':int(new.category.isna().sum()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
     api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom'
     doc=Path(hf_hub_download(repo,'data/tiki/full_dataset/'+name.replace('.parquet','.README.md'),repo_type='dataset',token=os.environ['HF_TOKEN'])).read_text(encoding='utf-8')
     doc=doc.replace('13.200 dòng, 5 cột','13.200 dòng, 6 cột').replace('| `description_zh` | Description thuộc tính tiếng Trung |','| `description_zh` | Description thuộc tính tiếng Trung |\n| `category` | Danh mục từ bộ description, ghép theo ID sản phẩm |').replace('File full chỉ giữ 5 cột trên.','File full chỉ giữ 6 cột trên.').replace('hai cột description để null','hai cột description và category để null')
     r=out/name.replace('.parquet','.README.md');r.write_text(doc,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo='data/tiki/full_dataset/'+r.name,path_or_fileobj=str(r)))
    api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha
    readme=Path(hf_hub_download(repo,'data/tiki/full_dataset/README.md',repo_type='dataset',revision=head,token=os.environ['HF_TOKEN'])).read_text(encoding='utf-8');readme=readme.replace('chỉ 5 cột cốt lõi','6 cột cốt lõi').replace('| `description_zh` | Description thuộc tính tiếng Trung |','| `description_zh` | Description thuộc tính tiếng Trung |\n| `category` | Danh mục từ bộ description |').replace('description null.','description và category null.')
    r=out/'README.md';r.write_text(readme,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo='data/tiki/full_dataset/README.md',path_or_fileobj=str(r)))
    m=out/'manifest.json';m.write_text(json.dumps({'exports':results,'category_source':'description dataset; joined by product_id'},indent=2));ops.append(CommitOperationAdd(path_in_repo='data/tiki/full_dataset/manifest.json',path_or_fileobj=str(m)))
    c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Add description category to both Tiki full datasets')
    for item in results:
     f=hf_hub_download(repo,item['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']);assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==item['sha256']
    result={'revision':c.oid,'exports':results};(out/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__ == "__main__":
    main()
