from pathlib import Path
import hashlib,json
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
repo='ntquang0410/zh-vie_ecom'; prefix='data/LLM/bilingual_judged/'; api=HfApi();head=api.dataset_info(repo).sha
W=Path(r'D:\download\NCKH\judge_kaggle_preflight'); prior=json.loads((W/'pair_provenance_upload_result.json').read_text(encoding='utf8'));assert head==prior['commit']
O=W/'pair_readme_correction';O.mkdir(exist_ok=True)
def get(n,r=None):return Path(hf_hub_download(repo,prefix+n,repo_type='dataset',revision=r or head))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
readme=get('README.md').read_text(encoding='utf8');assert 'CỤM 9 cột/PAIR 17 cột' in readme
readme=readme.replace('CỤM 9 cột/PAIR 17 cột','CỤM 9 cột/PAIR 9 cột')
needle='**Đơn vị đếm khác bản CỤM**:'
extra='Trong 872.034 đơn vị pair không rỗng: 316.057 Gemma, 12.756 Qwen, 237.909 quy tắc và 305.312 chưa rõ. Trong nhóm chưa rõ, 281.020 đơn vị không tìm thấy bản ghi đơn vị tương ứng trong bảng đầu vào theo đúng khóa ghép, 24.292 đơn vị có bản ghi nhưng chưa xác minh được nguồn bản đang giữ. Đây là giới hạn truy vết, không phải số lỗi dịch. Đã đối chiếu 9.054 đơn vị đổi theo judge với bộ đếm thay đổi ở từng sản phẩm; không có sản phẩm lệch.\n\n'
assert needle in readme;readme=readme.replace(needle,extra+needle,1);(O/'README.md').write_text(readme,encoding='utf8')
manifest=json.loads(get('manifest.json').read_text(encoding='utf8'));f=O/'README.md';manifest['files'][f.name]={'bytes':f.stat().st_size,'sha256':sha(f)};(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
sums={n:e['sha256'] for n,e in manifest['files'].items()};sums['manifest.json']=sha(O/'manifest.json');(O/'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(sums.items())),encoding='ascii')
files=list(O.iterdir());assert api.dataset_info(repo).sha==head
commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,num_threads=1,operations=[CommitOperationAdd(path_in_repo=prefix+f.name,path_or_fileobj=str(f)) for f in files],commit_message='Clarify pair provenance coverage and both nine-column schemas')
print('COMMIT',commit.oid,flush=True)
for f in files:assert sha(get(f.name,commit.oid))==sha(f)
assert sha(get('bo_pair_chon_deepseek.parquet',commit.oid))==manifest['files']['bo_pair_chon_deepseek.parquet']['sha256']
(W/'pair_final_result.json').write_text(json.dumps({'commit':commit.oid,'rows':25575,'verified':True,'unknown':305312,'changed_count_mismatches':0},ensure_ascii=False),encoding='utf8');print('README_AND_PAIR_VERIFIED',flush=True)
