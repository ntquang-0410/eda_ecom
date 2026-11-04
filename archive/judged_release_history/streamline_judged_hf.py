from pathlib import Path
import hashlib,json,shutil
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd,CommitOperationDelete

repo='ntquang0410/zh-vie_ecom';prefix='data/LLM/bilingual_judged/';api=HfApi()
info=api.dataset_info(repo,files_metadata=True);head=info.sha
names=sorted(s.rfilename[len(prefix):] for s in info.siblings if s.rfilename.startswith(prefix))
assert all('/' not in n for n in names), 'Inspect unexpected nested files before moving them'
main=['bo_cum_chon_deepseek.parquet','bo_pair_chon_deepseek.parquet','bo_cum_danh_dau.parquet','bo_pair_danh_dau.parquet']
assert set(main+['README.md','manifest.json']).issubset(names)
root=Path(r'D:\download\NCKH\judge_kaggle_preflight');backup=root/'hf_judged_backup'/head;backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(n,rev=None):return Path(hf_hub_download(repo,prefix+n,repo_type='dataset',revision=rev or head))
for n in names:
 p=get(n); target=backup/n
 if not target.exists():shutil.copyfile(p,target)
 assert sha(target)==sha(p),n
manifest0=json.loads((backup/'manifest.json').read_text(encoding='utf8'))
for n,v in manifest0['files'].items():
 assert n in names and sha(backup/n)==v['sha256'],n
(backup/'LOCAL_BACKUP_SHA256.json').write_text(json.dumps({'source_revision':head,'files':{n:sha(backup/n) for n in names}},ensure_ascii=False,indent=2),encoding='utf8')
print('BACKUP_VERIFIED',len(names),str(backup),flush=True)
out=root/'hf_judged_simplified';out.mkdir(exist_ok=True)
for f in out.iterdir():assert f.name in ('README.md','manifest.json'), 'Unexpected staging file'
cum=pd.read_parquet(backup/main[0]);pair=pd.read_parquet(backup/main[1]);expected=['product_id','source','category','title_zh','title_vi','description_zh','description_vi','judge_chon_title','judge_chon_description']
assert list(cum)==expected and list(pair)==expected and len(cum)==len(pair)==25575
assert cum[['source','product_id']].equals(pair[['source','product_id']])
for f in main[2:]:assert len(pd.read_parquet(backup/f))==25575
readme=f'''# Dữ liệu song ngữ sau judge

Thư mục này gồm bốn file Parquet, mỗi file 25.575 sản phẩm. Nguồn 1688 được dịch Trung → Việt; Tiki được dịch Việt → Trung. Hai bản cùng tập sản phẩm và thứ tự.

| File | Nội dung |
|---|---|
| `bo_cum_chon_deepseek.parquet` | Title và description bản dịch theo cụm sau luật, judge và rà soát. |
| `bo_pair_chon_deepseek.parquet` | Title và description bản dịch theo đơn vị pair sau luật, judge và rà soát. |
| `bo_cum_danh_dau.parquet` | Trạng thái và số lượng quyết định judge theo sản phẩm của bản cụm. |
| `bo_pair_danh_dau.parquet` | Trạng thái và số lượng quyết định judge theo sản phẩm của bản pair. |

Hai file dữ liệu có 9 cột: `product_id`, `source`, `category`, `title_zh`, `title_vi`, `description_zh`, `description_vi`, `judge_chon_title`, `judge_chon_description`.

`judge_chon_title` là model tạo bản title cuối được giữ (`gemma` hoặc `qwen`), không nhất thiết là lựa chọn trực tiếp của DeepSeek. Hai title đã được sửa tay; model thể hiện nguồn bản nháp.

`judge_chon_description` đếm nguồn nội dung cuối: `gemma`, `qwen`, `quy_tac`, `chua_ro`; nhóm bằng 0 được ẩn. Đây không phải điểm hay số lần judge chọn. Cụm đếm một cặp tên–giá trị là một mảnh; pair đếm tên và giá trị không rỗng riêng. Không so trực tiếp số đếm giữa hai bản. `chua_ro` biểu thị giới hạn thông tin nguồn, không tự kết luận dịch sai; trong bản cụm còn bao gồm mảnh có tên và giá trị từ hai model khác nhau.

DeepSeek đã chấm 4.439/4.446 mục cụm; bảy mục bị nhà cung cấp chặn đã được trợ lý GPT-6.1-sol rà riêng (hai chọn Gemma, bốn tương đương, một chưa chắc chắn). Bản pair chấm 5.000/34.534 đơn vị tranh chấp; các đơn vị chưa chấm giữ bản trước theo pipeline. Các cờ trong file đánh dấu phản ánh trạng thái judge, không chứng nhận ngữ nghĩa. Bộ dữ liệu chưa được người kiểm chứng toàn bộ và không nên coi là tập gold đã kiểm nghĩa.

Các file truy vết chi tiết, metadata cũ và báo cáo được lưu ngoài bản hiện tại. Có thể xem phiên bản trước khi thu gọn tại [revision {head}](https://huggingface.co/datasets/{repo}/tree/{head}/{prefix.rstrip('/')}); lịch sử repository vẫn chứa các file đó. `manifest.json` ghi số dòng và SHA-256 của bốn file dữ liệu và README.
'''
(out/'README.md').write_text(readme,encoding='utf8')
entries={n:{'bytes':(backup/n).stat().st_size,'sha256':sha(backup/n),'rows':25575} for n in main};entries['README.md']={'bytes':(out/'README.md').stat().st_size,'sha256':sha(out/'README.md')}
manifest={'dataset_status':manifest0.get('dataset_status'),'rows_each':25575,'columns_by_method':{'cum':expected,'pair':expected},'pre_cleanup_revision':head,'files':entries}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
keep=set(main+['README.md','manifest.json']);remove=[n for n in names if n not in keep]
assert api.dataset_info(repo).sha==head
ops=[CommitOperationAdd(path_in_repo=prefix+n,path_or_fileobj=str(out/n)) for n in ('README.md','manifest.json')]+[CommitOperationDelete(path_in_repo=prefix+n) for n in remove]
print('Publishing simplified directory; removed',len(remove),'files',flush=True)
c=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,operations=ops,num_threads=1,commit_message='Simplify judged dataset to four Parquet files and documentation')
print('COMMIT',c.oid,flush=True)
newinfo=api.dataset_info(repo,revision=c.oid);actual={s.rfilename[len(prefix):] for s in newinfo.siblings if s.rfilename.startswith(prefix)};assert actual==keep,(actual,keep)
for n in keep:
 expected_hash=sha(out/n) if n in ('README.md','manifest.json') else sha(backup/n)
 assert sha(get(n,c.oid))==expected_hash,n
(root/'hf_judged_cleanup_result.json').write_text(json.dumps({'commit':c.oid,'backup':str(backup),'backup_revision':head,'files':sorted(keep),'removed_count':len(remove),'verified':True},ensure_ascii=False,indent=2),encoding='utf8')
print('REMOTE_VERIFIED',flush=True)
