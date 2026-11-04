

def main():

    # Repository layout bootstrap; translation policy is unchanged.
    import sys as _repo_sys
    from pathlib import Path as _RepoPath
    _repo_root = _RepoPath(__file__).resolve().parents[2]
    if (_repo_root / "project_bootstrap.py").is_file():
        _repo_sys.path.insert(0, str(_repo_root))
        from project_bootstrap import configure_paths as _configure_paths
        _configure_paths()
    from pathlib import Path
    from collections import Counter
    import pandas as pd,json,re,hashlib
    import title_translation_pipeline as zhqa
    from tiki_translation_pipeline import qa_check_vi_to_zh
    out=Path('outputs/title_bidirectional_audit');out.mkdir(parents=True,exist_ok=True)
    files={'1688':Path('outputs/1688_cjk_repair/publish/data/1688/title_translations/best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet'),'tiki':Path('outputs/tiki_vi_cleanup/data/tiki/title_translations/best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet')}
    viet=r'[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]'
    for name,p in files.items():
     d=pd.read_parquet(p);src,dst=('title_zh','title_vi') if name=='1688' else ('title_vi','title_zh');u=d.drop_duplicates(src);records=[];flags=Counter()
     for s,t in u[[src,dst]].itertuples(index=False,name=None):
      q=zhqa.check_translation(s,t) if name=='1688' else qa_check_vi_to_zh(s,t)
      fs=q.get('hard_flags',[]) if name=='1688' else q.get('flags',[]);passed=q['status']=='pass' if name=='1688' else q['passed']
      remainder=bool(re.search(r'[\u3400-\u9fff]',t or '')) if name=='1688' else bool(re.search(viet,t or ''))
      for f in fs:flags[f.split(':',1)[0]]+=1
      if not passed or remainder:records.append({src:s,dst:t,'technical_pass':passed,'flags_json':json.dumps(fs,ensure_ascii=False),'residual_script':remainder})
     pd.DataFrame(records).to_parquet(out/(name+'_review_queue.parquet'),index=False)
     summary={'rows':len(d),'unique_titles':len(u),'technical_flagged_unique':len(records),'residual_script_unique':sum(r['residual_script'] for r in records),'flag_categories':dict(flags),'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'semantic_verified':False,'residual_detection':'han_characters' if name=='1688' else 'vietnamese_diacritics_only'}
     (out/(name+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps({'dataset':name,**summary}))


if __name__ == "__main__":
    main()
