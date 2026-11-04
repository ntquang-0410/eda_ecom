"""Atomically reorganize existing dataset files; validate every moved blob."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,posixpath,re,hashlib
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationCopy,CommitOperationAdd,CommitOperationDelete,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
OUT=ROOT/'outputs/hf_reorganized';OUT.mkdir(parents=True,exist_ok=True)
REPO='ntquang0410/zh-vie_ecom';api=HfApi(token=os.environ['HF_TOKEN'])
info=api.repo_info(REPO,repo_type='dataset');base=info.sha
files=api.list_repo_files(REPO,repo_type='dataset',revision=base)

def destination(f):
    if f.startswith('data/translated/title_only/'):
        suffix=f[len('data/translated/title_only/'):]
        if suffix in ('README.md','manifest.json'):return None
        folder,rest=suffix.split('/',1)
        dataset='1688' if folder.startswith('1688_') else 'tiki'
        model='best_cometkiwi' if 'best_cometkiwi' in folder else 'qwen3_14b_awq' if 'qwen3' in folder else 'gemma4'
        return f'data/{dataset}/title_translations/{model}/{rest}'
    for old,new in [
        ('data/cleaned/1688/','data/1688/cleaned/'),('data/cleaned/tiki/','data/tiki/cleaned/'),
        ('data/bronze/tiki_vi/','data/tiki/bronze/tiki_vi/'),('data/bronze/bilingual_zh_vi/','data/1688/bronze/bilingual_zh_vi/'),('data/bronze/mono_zh/','data/1688/bronze/mono_zh/'),
        ('data/description_vi_to_zh/','data/tiki/descriptions/vi_to_zh/'),('data/description_vi_to_zh_cum/','data/tiki/descriptions/vi_to_zh_context/'),('data/description_zh_to_vi/','data/1688/descriptions/zh_to_vi/')]:
        if f.startswith(old):return new+f[len(old):]
    if f.startswith('data/raw/'):
        name=f.rsplit('/',1)[1];dataset='tiki' if name.startswith('tiki_') else '1688'
        return f'data/{dataset}/raw/{name}'
    if f.startswith('data/snapshot/'):
        name=f.rsplit('/',1)[1];dataset='tiki' if name.startswith('tiki') else '1688'
        return f'data/{dataset}/snapshot/{name}'
    if f=='data/cleaned/README.md':return 'data/1688/cleaned/README_legacy_shared.md'
    return f

mapping={f:destination(f) for f in files if f.startswith('data/')}
targets=[v for v in mapping.values() if v];assert len(targets)==len(set(targets)),'destination_collision'
metadata={x.path:x for x in api.get_paths_info(REPO,list(mapping),repo_type='dataset',revision=base)}
def fingerprint(x):return (x.size,x.lfs.sha256 if x.lfs else x.blob_id)
ops=[]
def addtext(remote,text):
    path=OUT/remote;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
for old,new in mapping.items():
    if old==new:continue
    if new:
        # Copy parquet/raw bytes on the Hub, without changing original values.
        ops.append(CommitOperationCopy(src_path_in_repo=old,path_in_repo=new,src_revision=base))
    ops.append(CommitOperationDelete(path_in_repo=old))
if 'data/cleaned/README.md' in files:
    ops.append(CommitOperationCopy(src_path_in_repo='data/cleaned/README.md',path_in_repo='data/tiki/cleaned/README_legacy_shared.md',src_revision=base))

oldreadme=Path(hf_hub_download(REPO,'README.md',repo_type='dataset',revision=base,token=os.environ['HF_TOKEN'])).read_text(encoding='utf-8')
(OUT/'README_before_reorganization.md').write_text(oldreadme,encoding='utf-8')
front=re.match(r'\A---\s*\n.*?\n---\s*\n',oldreadme,re.S)
header=front.group(0) if front else ''
# Old card configs may refer to moved paths; preserve front matter while migrating paths.
for old,new in sorted(mapping.items(),key=lambda kv:len(kv[0]),reverse=True):
    if new:header=header.replace(old,new)

readme='''# Dữ liệu title thương mại điện tử Việt–Trung: 1688 và Tiki

Repo được chia theo hai nguồn dữ liệu: **1688** và **Tiki**. Mỗi bộ có dữ liệu nguồn, các lớp xử lý và kết quả dịch title của Gemma, Qwen, cùng bản kết hợp được chọn bằng COMETKiwi.

## 1. Cấu trúc thư mục

```text
data/
├── 1688/
│   ├── raw/                    # Các đợt thu thập JSONL
│   ├── bronze/                 # Dữ liệu trung gian theo loại/ngành hàng
│   ├── snapshot/               # Snapshot tổng hợp
│   ├── cleaned/                # Dữ liệu nguồn đã xử lý và tài liệu đi kèm
│   ├── descriptions/zh_to_vi/  # Các sản phẩm xử lý description đã có
│   └── title_translations/
│       ├── gemma4/
│       ├── qwen3_14b_awq/
│       └── best_cometkiwi/
└── tiki/
    ├── raw/
    ├── bronze/
    ├── snapshot/
    ├── cleaned/
    ├── descriptions/
    │   ├── vi_to_zh/
    │   └── vi_to_zh_context/
    └── title_translations/
        ├── gemma4/
        ├── qwen3_14b_awq/
        └── best_cometkiwi/
```

Các file nguồn và description được giữ nguyên nội dung khi di chuyển. **Các file trong `title_translations/` chỉ chứa ID, title gốc, title dịch và metadata liên quan đến dịch**, không chứa description hay các thuộc tính sản phẩm như giá, shop, danh mục.

## 2. Tổng quan hai bộ title

| Bộ | Chiều dịch | Dòng sản phẩm | Title gốc duy nhất | Title nguồn | Title đích |
|---|---|---:|---:|---|---|
| 1688 | Trung → Việt | 16.348 | 15.863 | `title_zh` | `title_vi` |
| Tiki | Việt → Trung | 13.200 | 13.102 | `title_vi` | `title_zh` |

Một title có thể xuất hiện ở nhiều sản phẩm. Vì vậy, số dòng sản phẩm lớn hơn số title duy nhất. Việc chấm và đếm số lần model được chọn được thực hiện theo title duy nhất; kết quả sau đó được ánh xạ lại đầy đủ về từng dòng sản phẩm và giữ `product_id` theo thứ tự nguồn.

### 1688

Nguồn đối chiếu của lượt dịch là `data/1688/cleaned/bilingual_zh_vi_cado.parquet`. Hai ứng viên được tạo bởi Gemma (`gemma-4-26b-a4b-it`) và `Qwen/Qwen3-14B-AWQ`. Qwen đã trải qua một lượt sửa kỹ thuật có giới hạn: 804 title sửa được chấp nhận trong nhóm 3.599 title được đánh dấu. Bản gốc được giữ riêng; kết quả sửa có thông tin giai đoạn ứng viên. Một lỗi sinh Qwen còn lại không được ưu tiên hơn ứng viên Gemma hợp lệ khi tạo bản kết hợp.

### Tiki

Nguồn đối chiếu là `data/tiki/cleaned/tiki_vi_cado.parquet`. Chiều dịch là tiếng Việt sang tiếng Trung, dùng cùng hai họ model. Bản Qwen sau lượt sửa kỹ thuật chấp nhận 370 bản sửa. Lượt Gemma có 13.090/13.102 title sinh thành công; 12 title lỗi sinh được giữ trạng thái trong bản riêng. Khi tạo bản kết hợp, ứng viên lỗi sinh không được chọn nếu còn bản hợp lệ của model kia. **Bản kết hợp Tiki vẫn có đầy đủ 13.200 dòng, không có title dịch được chọn bị trống.**

## 3. Nên sử dụng file nào?

| Mục đích | 1688 | Tiki |
|---|---|---|
| Sử dụng bản kết hợp | [1688 best COMETKiwi](data/1688/title_translations/best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet) | [Tiki best COMETKiwi](data/tiki/title_translations/best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet) |
| Xem riêng Gemma | [1688 Gemma](data/1688/title_translations/gemma4/1688_zh_to_vi_gemma4.parquet) | [Tiki Gemma](data/tiki/title_translations/gemma4/tiki_vi_to_zh_gemma4.parquet) |
| Xem riêng Qwen | [1688 Qwen](data/1688/title_translations/qwen3_14b_awq/1688_zh_to_vi_qwen3_14b_awq.parquet) | [Tiki Qwen](data/tiki/title_translations/qwen3_14b_awq/tiki_vi_to_zh_qwen3_14b_awq.parquet) |

## 4. COMETKiwi được dùng như thế nào?

Model chấm: [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da), revision `1ad785194e391eebc6c53e2d0776cada8f83179a`.

COMETKiwi là bộ ước lượng chất lượng dịch **không cần bản dịch tham chiếu của người**. Mỗi ứng viên được chấm với cùng title nguồn: `(src, mt)`. Nguồn là title Trung đối với 1688 và title Việt đối với Tiki. Điểm cao hơn được dùng để xếp hạng hai ứng viên trong cùng title; điểm không phải xác suất bản dịch đúng và không nên hiểu là phần trăm chất lượng.

Quy tắc xuất cuối cùng:

1. Giữ đầy đủ các title và dòng sản phẩm nguồn.
2. Loại ứng viên trống hoặc lỗi sinh khỏi tập lựa chọn khi còn ứng viên sinh hợp lệ.
3. Trong các ứng viên còn dùng được, chọn bản có điểm COMETKiwi cao hơn; nếu điểm bằng nhau, chọn Gemma để kết quả có thể tái lập.
4. **Không dùng ngưỡng 0,72 để bỏ title. Không dùng cờ kiểm tra kỹ thuật để loại ứng viên khỏi so sánh.** Các thông tin kiểm tra chỉ được giữ để tham khảo.
5. Giữ cả hai ứng viên, hai điểm và model được chọn trong file kết hợp. Không chạy dịch lại trong bước chọn.

Lượt chấm hoàn tất 31.726 cặp cho 1688 và 26.113 cặp cho Tiki. Ứng viên trống không được gán điểm giả; điểm có thể để null đối với ứng viên đó. Việc thiếu điểm của một ứng viên không làm mất dòng sản phẩm.

## 5. Kết quả chọn model

### Theo title duy nhất

| Bộ | Gemma được chọn | Qwen được chọn | Tổng | Title dịch được chọn bị trống |
|---|---:|---:|---:|---:|
| 1688 | 9.563 | 6.300 | 15.863 | 0 |
| Tiki | 6.240 | 6.862 | 13.102 | 0 |

### Sau khi ánh xạ về dòng sản phẩm

| Bộ | Dòng chọn Gemma | Dòng chọn Qwen | Tổng |
|---|---:|---:|---:|
| 1688 | 9.889 | 6.459 | 16.348 |
| Tiki | 6.288 | 6.912 | 13.200 |

Gemma được chọn nhiều hơn trong bộ 1688; Qwen được chọn nhiều hơn trong bộ Tiki ở lượt đánh giá này. Đây là kết quả xếp hạng của model chấm trên các ứng viên cụ thể, không phải kết luận model nào tốt hơn trong mọi lĩnh vực.

## 6. Các cột chính của bản kết hợp

| Cột | Ý nghĩa |
|---|---|
| `product_id` | ID để đối chiếu sản phẩm nguồn |
| `title_zh`, `title_vi` | Title gốc và title đích được chọn; chiều dịch phụ thuộc bộ |
| `gemma4_translation` | Ứng viên Gemma |
| `qwen3_14b_awq_translation` | Ứng viên Qwen |
| `cometkiwi_gemma4`, `cometkiwi_qwen3_14b_awq` | Điểm của hai ứng viên trên cùng nguồn |
| `selected_model` | Model của bản dịch cuối cùng |
| `selected_cometkiwi_score` | Điểm của ứng viên được chọn |
| `selection_policy`, `selection_reason` | Quy tắc và lý do lựa chọn |
| `gemma4_generation_ok`, `qwen3_14b_awq_generation_ok` | Trạng thái sinh bản dịch |
| Các cột QA | Thông tin kỹ thuật giữ lại, không dùng làm ngưỡng loại trong bản cuối |
| `semantic_verified` | `false`: chưa xác minh ngữ nghĩa thủ công cho từng title |

Các bản riêng từng model có tên cột metadata khác nhau theo pipeline. `selection_report.json` trong mỗi thư mục `best_cometkiwi/` ghi số lượng, chính sách chọn và SHA256 của file cuối. `manifest.json` ghi các đường dẫn mới và checksum của 6 file title.

## 7. Cách đọc dữ liệu

```python
import pandas as pd

base = 'hf://datasets/ntquang0410/zh-vie_ecom/data'
titles_1688 = pd.read_parquet(
    f'{base}/1688/title_translations/best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet'
)
titles_tiki = pd.read_parquet(
    f'{base}/tiki/title_translations/best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet'
)
```

## 8. Giới hạn và khả năng tái lập

Title thương mại điện tử thường ngắn, nhiều tên riêng, mã, số đo và ký hiệu. COMETKiwi có thể không đánh giá đầy đủ mọi chi tiết này. Bản điểm cao hơn vẫn có thể sai; kết quả không thay thế nhãn chuẩn do người kiểm tra. Hai bộ được đánh giá theo hai chiều dịch khác nhau, vì vậy không dùng điểm trung bình giữa hai bộ để kết luận bộ nào tốt hơn.

Dữ liệu nguồn và các bản ứng viên được giữ lại để có thể đối chiếu. Các file title kết hợp đã được kiểm tra số dòng, thứ tự ID, số title duy nhất, không có title đích được chọn bị trống và checksum local/HF. Model chấm được khóa revision như trên. Không có token hoặc khóa API trong các file dữ liệu và báo cáo được công bố.

Việc chuyển thư mục chỉ thay đổi đường dẫn của dữ liệu đã có. Những tài liệu cũ trong các lớp nguồn mô tả các lượt xử lý trước đó; quy tắc lựa chọn title cuối cùng được mô tả trong README này và các báo cáo lựa chọn mới.
'''
addtext('README.md',header+readme)
for dataset in ['1688','tiki']:
    direction='Trung → Việt' if dataset=='1688' else 'Việt → Trung'
    rows='16.348 dòng / 15.863 title' if dataset=='1688' else '13.200 dòng / 13.102 title'
    addtext(f'data/{dataset}/README.md',f'# {dataset}: {direction}\n\n{rows}.\n\n- `raw/`, `bronze/`, `snapshot/`, `cleaned/`: dữ liệu nguồn và các lớp xử lý.\n- `descriptions/`: kết quả xử lý description đã có.\n- `title_translations/gemma4/`: bản Gemma.\n- `title_translations/qwen3_14b_awq/`: bản Qwen.\n- `title_translations/best_cometkiwi/`: bản chọn theo điểm COMETKiwi, không dùng ngưỡng loại.\n\nXem [README chính](../../README.md) để biết quy tắc lựa chọn, schema và kết quả chi tiết.\n')
latest=json.loads((ROOT/'outputs/comet_best_all_titles/upload_result.json').read_text())
manifest=[]
for old,new in mapping.items():
    if new and new.endswith('.parquet') and '/title_translations/' in new:
        x=metadata[old];manifest.append({'path':new,'bytes':x.size,'sha256':x.lfs.sha256 if x.lfs else None})
addtext('manifest.json',json.dumps({'previous_revision':base,'title_exports':manifest,'layout':'data/{1688,tiki}/','selection_policy':'higher_cometkiwi_no_floor_qa_advisory_only'},indent=2))
(OUT/'move_plan.json').write_text(json.dumps({'base':base,'mapping':mapping},indent=2))
commit=api.create_commit(REPO,repo_type='dataset',parent_commit=base,operations=ops,commit_message='Group repository into 1688 and Tiki and document title translation and COMETKiwi selection')
after={x.path:x for x in api.get_paths_info(REPO,targets,repo_type='dataset',revision=commit.oid)}
for old,new in mapping.items():
    if new:assert fingerprint(metadata[old])==fingerprint(after[new]),f'moved_content_mismatch:{new}'
remaining=api.list_repo_files(REPO,repo_type='dataset',revision=commit.oid)
assert all(not f.startswith('data/') or f.startswith(('data/1688/','data/tiki/')) for f in remaining)
for old,new in mapping.items():assert old==new or old not in remaining
for doc in ['README.md','data/1688/README.md','data/tiki/README.md','manifest.json']:
    f=hf_hub_download(REPO,doc,repo_type='dataset',revision=commit.oid,token=os.environ['HF_TOKEN'])
    assert Path(f).read_bytes()==(OUT/doc).read_bytes()
result={'revision':commit.oid,'moved_files':sum(v is not None and k!=v for k,v in mapping.items()),'verified_unchanged_file_contents':True,'title_exports':manifest}
(OUT/'reorganization_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
