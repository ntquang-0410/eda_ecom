

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
    disable_progress_bars();repo='ntquang0410/zh-vie_ecom';api=HfApi(token=os.environ['HF_TOKEN']);head=api.repo_info(repo,repo_type='dataset').sha;out=Path('outputs/full_9col_preparation');out.mkdir(parents=True,exist_ok=True);ops=[];report=[]
    for ds in ['1688','tiki']:
     p=out/(ds+'_title_prepared.parquet');d=pd.read_parquet(p)
     if ds=='1688':
      old=pd.read_parquet('outputs/comet_best_all_titles/1688_zh_to_vi_best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet');old.product_id=old.product_id.astype(str);assert d.product_id.equals(old.product_id)
      mask=d.title_ngon_dich.eq('manual_cjk_cleanup');assert int(mask.sum())==9;d.loc[mask,'title_ngon_dich']=old.loc[mask,'selected_model'].values
     assert d.source.eq(ds).all() and d.product_id.is_unique and d.category.notna().all() and d.title_ngon_dich.isin(['gemma4','qwen3_14b_awq']).all()
     expected=['product_id','source','category','title_zh','title_vi','title_ngon_dich'];d=d[expected];assert not d.isna().any().any();assert len(d)==(16348 if ds=='1688' else 13200)
     p=out/(ds+'_title_input_9col_full.parquet');d.to_parquet(p,index=False);pd.testing.assert_frame_equal(d,pd.read_parquet(p))
     remote=f'data/{ds}/title_translations/full_9col_title_input.parquet';ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)));report.append({'dataset':ds,'path':remote,'rows':len(d),'columns':expected,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'models':d.title_ngon_dich.value_counts().to_dict()})
     note=f'''# Bảng title chuẩn bị ghép full — {ds}\n\nFile `full_9col_title_input.parquet` chứa 6 cột phía title: `product_id`, `source`, `category`, `title_zh`, `title_vi`, `title_ngon_dich`. Đây là bảng đầu vào, chưa phải file full 9 cột. Ghép trái theo `product_id` với description cấp sản phẩm để bổ sung `description_zh`, `description_vi`, `desc_nguon_dich` theo thứ tự schema yêu cầu.\n\nCó {len(d):,} dòng, ID không trùng; category lấy từ bảng cleaned nguồn và không null. `title_ngon_dich` là model tạo ứng viên được chọn. {'Có 9 title đã được chỉnh thủ công sau dịch: 8 từ ứng viên Qwen, 1 từ Gemma; cột ghi model gốc, không có nghĩa 9 title này nguyên văn đầu ra model.' if ds=='1688' else 'Các bản title đã được rà và một số được thay bằng ứng viên của model còn lại; cột ghi model của ứng viên được giữ.'}\n\nKhông chứa các cột điểm, hai ứng viên, cờ QA; thông tin chi tiết nằm trong `best_cometkiwi/`.\n'''
     note_path=out/(ds+'_title_input_README.md');note_path.write_text(note,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=f'data/{ds}/title_translations/full_9col_title_input.README.md',path_or_fileobj=str(note_path)))
    c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Add verified title-side inputs for nine-column full datasets')
    for item in report:
     f=hf_hub_download(repo,item['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN']);assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==item['sha256']
    (out/'upload_result.json').write_text(json.dumps({'revision':c.oid,'exports':report},indent=2));print(json.dumps({'revision':c.oid,'exports':report}))


if __name__ == "__main__":
    main()
