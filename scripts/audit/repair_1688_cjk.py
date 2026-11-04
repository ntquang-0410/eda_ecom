

def main():

    # Repository layout bootstrap; translation policy is unchanged.
    import sys as _repo_sys
    from pathlib import Path as _RepoPath
    _repo_root = _RepoPath(__file__).resolve().parents[2]
    if (_repo_root / "project_bootstrap.py").is_file():
        _repo_sys.path.insert(0, str(_repo_root))
        from project_bootstrap import configure_paths as _configure_paths
        _configure_paths()
    import os,json,re,time
    from pathlib import Path
    import pandas as pd
    import title_translation_pipeline as qa
    from google import genai
    from google.genai import types
    p=Path('outputs/comet_best_all_titles/1688_zh_to_vi_best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet');u=pd.read_parquet(p).drop_duplicates('title_zh');u=u[u.title_vi.fillna('').str.contains(r'[\u3400-\u9fff]')]
    out=Path('outputs/1688_cjk_repair');out.mkdir(exist_ok=True,parents=True);path=out/'corrections.json';records=json.loads((out/'alternatives.json').read_text(encoding='utf8'));records.update(json.loads(path.read_text(encoding='utf8')) if path.exists() else {});client=genai.Client(api_key=os.environ['GEMINI_API_KEY'],http_options=types.HttpOptions(timeout=15000,retry_options=types.HttpRetryOptions(attempts=1)));stats={'alternate':0,'api':0,'unresolved':0,'api_failures':0}
    for r in u.itertuples():
     if r.title_zh in records:continue
     alt=r.qwen3_14b_awq_translation if r.selected_model=='gemma4' else r.gemma4_translation;valid=r.qwen3_14b_awq_generation_ok if r.selected_model=='gemma4' else r.gemma4_generation_ok
     if isinstance(alt,str) and alt.strip() and not re.search(r'[\u3400-\u9fff]',alt) and valid and qa.check_translation(r.title_zh,alt)['status']=='pass':
      records[r.title_zh]={'text':alt,'method':'existing_alternative','model':'qwen3_14b_awq' if r.selected_model=='gemma4' else 'gemma4'};stats['alternate']+=1;continue
     try:
      prompt='Dịch tiêu đề sản phẩm tiếng Trung sang tiếng Việt tự nhiên, giữ chính xác mọi số, đơn vị và mã model. Phiên âm tên riêng nếu cần, không để lại chữ Hán. Chỉ trả về tiêu đề dịch, không giải thích.\n'+r.title_zh
      response=client.models.generate_content(model='gemma-4-26b-a4b-it',contents=prompt,config=types.GenerateContentConfig(temperature=0,max_output_tokens=256));text=(response.text or '').strip()
      if text and not re.search(r'[\u3400-\u9fff]',text) and qa.check_translation(r.title_zh,text)['status']=='pass':records[r.title_zh]={'text':text,'method':'gemma_cjk_repair','model':'gemma4'};stats['api']+=1
      else:stats['unresolved']+=1
      time.sleep(2)
     except Exception as e:
      stats['api_failures']+=1;print(json.dumps({'status':'paused_api_failure','error_class':type(e).__name__,'completed':len(records)}),flush=True);break
     if len(records)%20==0:path.write_text(json.dumps(records,ensure_ascii=False),encoding='utf8')
    path.write_text(json.dumps(records,ensure_ascii=False),encoding='utf8');print(json.dumps({'total_flagged':len(u),'corrections_saved':len(records),**stats}),flush=True)


if __name__ == "__main__":
    main()
