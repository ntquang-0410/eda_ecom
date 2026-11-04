from pathlib import Path
import json,hashlib,re
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
repo='ntquang0410/zh-vie_ecom';prefix='data/LLM/bilingual_judged/';root=Path(r'D:\download\NCKH\judge_kaggle_preflight\corrected_reviewed_release');api=HfApi();head=api.dataset_info(repo).sha;existing=api.list_repo_files(repo,repo_type='dataset',revision=head);assert not any(p.startswith(prefix) for p in existing),'Target already exists; inspect before overwrite'
keysrc=Path(r'D:\download\NCKH\eda_ecom\api_deepseek_alibaba.txt').read_text(encoding='utf8');keys=re.findall(r'sk-[A-Za-z0-9_.-]{20,}',keysrc);assert len(keys)==1
files=[p for p in sorted(root.iterdir()) if p.is_file()];assert all(p.suffix in ['.parquet','.json','.md','.txt'] for p in files)
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
 h,n=line.split(maxsplit=1);assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
for p in files:assert keys[0].encode() not in p.read_bytes(),'Credential detected; upload forbidden'
a=pd.read_parquet(root/'bo_cum_chon_deepseek.parquet');b=pd.read_parquet(root/'bo_pair_chon_deepseek.parquet');assert len(a)==len(b)==25575 and len(a.columns)==len(b.columns)==17;assert a[['source','product_id','title_zh','title_vi']].equals(b[['source','product_id','title_zh','title_vi']])
ops=[CommitOperationAdd(path_in_repo=prefix+p.name,path_or_fileobj=str(p)) for p in files]
index='data/LLM/README.md'
if index in existing:
 old=Path(hf_hub_download(repo,index,repo_type='dataset',revision=head)).read_text(encoding='utf8');extra='\n## Reviewed DeepSeek selection\n\nSee [bilingual_judged](bilingual_judged/README.md) for two 25,575-row datasets, manual-correction provenance, 7 provider-blocked context items retained unchanged, measured coverage and remaining title-review signals. This does not replace the official dataset or certify semantic correctness.\n';p=root.parent/'LLM_README_judged_index.md';p.write_text(old+extra,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=index,path_or_fileobj=str(p)))
print('Upload starting',len(files),'files',flush=True)
c=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,operations=ops,num_threads=1,commit_message='Add reviewed DeepSeek bilingual selection with explicit unresolved items and quality limitations');print('COMMIT',c.oid,flush=True)
for p in files:
 remote=Path(hf_hub_download(repo,prefix+p.name,repo_type='dataset',revision=c.oid));assert hashlib.sha256(remote.read_bytes()).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest(),p.name
(root.parent/'judged_hf_upload_result.json').write_text(json.dumps({'commit':c.oid,'files_verified':len(files),'target':prefix,'rows_each':25575,'columns_each':17},indent=2),encoding='utf8');print('VERIFIED',len(files),'files',flush=True)
