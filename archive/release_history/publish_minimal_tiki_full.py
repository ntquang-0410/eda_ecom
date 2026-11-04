"""Publish five-column full exports; preserve rich component data separately."""

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
disable_progress_bars()
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
OUT=ROOT/'outputs/tiki_full_minimal';OUT.mkdir(parents=True,exist_ok=True)
columns=['product_id','title_vi','title_zh','description_vi','description_zh']
ops=[];exports=[]
def add(remote,path):ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
def text(remote,content):
    p=OUT/remote.rsplit('/',1)[1];p.write_text(content,encoding='utf-8');add(remote,p)
for method,label,pairs in [('attributes_independent','dịch theo pair (từng cặp thuộc tính)',87056),('attributes_context','dịch theo cả cụm thuộc tính có ngữ cảnh',87112)]:
    filename=f'tiki_full_{method}_vi_zh.parquet'
    source=ROOT/'outputs/tiki_full_dataset'/filename
    full=pd.read_parquet(source);slim=full[columns].copy()
    assert len(slim)==13200 and slim.product_id.is_unique
    assert slim.title_vi.notna().all() and slim.title_zh.notna().all()
    assert slim.title_zh.str.strip().ne('').all()
    missing=slim.description_vi.isna()|slim.description_zh.isna()
    assert int(missing.sum())==13
    path=OUT/filename;slim.to_parquet(path,index=False)
    pd.testing.assert_frame_equal(slim,pd.read_parquet(path))
    for c in columns:pd.testing.assert_series_equal(slim[c],full[c])
    remote='data/tiki/full_dataset/'+filename;add(remote,path)
    exports.append({'path':remote,'rows':13200,'columns':columns,'products_with_description':13187,'products_missing_description':13,'description_method':method,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    process=('''Description được tổng hợp từ các pair thuộc tính dịch độc lập: sắp theo `pair_id`, dựng từng dòng `tên: giá trị` rồi nối bằng newline. Không dịch lại toàn bộ đoạn văn trong bước ghép.

Nguồn có 87.056 pair cho 13.187 sản phẩm. Xem [pipeline dịch từng pair](../descriptions/vi_to_zh/README.md).''' if method=='attributes_independent' else '''Model xem title và cả cụm/bảng thuộc tính của một sản phẩm, dùng bản pair làm nháp để rà và chỉnh theo ngữ cảnh. “Cả cụm” không có nghĩa toàn bộ văn xuôi hoặc hình ảnh trên trang sản phẩm.

Bản full giữ nguyên hai chuỗi description trong bảng cấp sản phẩm đã cung cấp; không dịch lại hoặc dựng lại các chuỗi này. Nguồn có 87.112 pair cho 13.187 sản phẩm. Xem [pipeline dịch cả cụm](../descriptions/vi_to_zh_context/README.md).''')
    text('data/tiki/full_dataset/'+filename.replace('.parquet','.README.md'),f'''# Tiki full — {label}

File: [{filename}]({filename}). **13.200 dòng, 5 cột**, ghép theo `product_id` và giữ nguyên thứ tự sản phẩm.

| Cột | Nội dung |
|---|---|
| `product_id` | ID sản phẩm |
| `title_vi` | Title gốc tiếng Việt |
| `title_zh` | Title tiếng Trung được chọn bằng COMETKiwi |
| `description_vi` | Description thuộc tính tiếng Việt |
| `description_zh` | Description thuộc tính tiếng Trung |

## Cách thực hiện description

{process}

## Title và dữ liệu thiếu

Title dùng chung [bản chọn Gemma/Qwen bằng COMETKiwi](../title_translations/best_cometkiwi/README.md): giữ bản có điểm cao hơn giữa các bản không lỗi sinh; không dùng ngưỡng loại. Có description cho **13.187 sản phẩm**; **13 sản phẩm thiếu description nguồn** vẫn giữ đủ ID và title, hai cột description để null. Không bịa nội dung hoặc bỏ dòng.

File full chỉ giữ 5 cột trên. Điểm, bản dịch ứng viên, model, QA, pair chi tiết và provenance nằm trong các bộ thành phần, không lặp lại trong file full. Dữ liệu dịch máy chưa xác minh ngữ nghĩa thủ công toàn bộ.

SHA256: `{exports[-1]['sha256']}`. Xem thêm [README tổng quan](README.md) và [manifest](manifest.json).
''')
text('data/tiki/full_dataset/README.md','''# Tiki full: title + description Việt–Trung

Hai bản dùng cùng title đã chọn bằng COMETKiwi, khác nhau ở cách xử lý description. Mỗi file giữ **13.200 sản phẩm, chỉ 5 cột cốt lõi**:

| Cột | Nội dung |
|---|---|
| `product_id` | ID dùng đối chiếu sản phẩm |
| `title_vi` | Title gốc tiếng Việt |
| `title_zh` | Title dịch tiếng Trung được chọn |
| `description_vi` | Description thuộc tính tiếng Việt |
| `description_zh` | Description thuộc tính tiếng Trung |

## Hai cách description

| Cách | Quy trình | Parquet | Mô tả riêng |
|---|---|---|---|
| Theo pair | Dịch tên/giá trị thuộc tính độc lập; nối các dòng `tên: giá trị` theo `pair_id` | [Tải bản pair](tiki_full_attributes_independent_vi_zh.parquet) | [README pair](tiki_full_attributes_independent_vi_zh.README.md) |
| Cả cụm | Rà/chỉnh bản nháp với title và toàn bộ bảng thuộc tính của sản phẩm làm ngữ cảnh; giữ nguyên description cấp sản phẩm đã cung cấp | [Tải bản cả cụm](tiki_full_attributes_context_vi_zh.parquet) | [README cả cụm](tiki_full_attributes_context_vi_zh.README.md) |

“Cả cụm” là cụm thuộc tính có ngữ cảnh, không phải toàn bộ văn xuôi hay hình ảnh trang sản phẩm. Lượt ghép full không dịch lại và không chấm lại description. Không trộn hai phương pháp trong một file.

Có description cho **13.187/13.200 sản phẩm**; 13 dòng còn lại giữ nguyên title, description null. Hai file có cùng ID và thứ tự sản phẩm. Không có title đích bị trống.

## Chi tiết được lưu ở đâu?

- [Title và hai ứng viên/điểm COMET](../title_translations/best_cometkiwi/README.md).
- [Description theo pair và metadata gốc](../descriptions/vi_to_zh/README.md).
- [Description theo cả cụm và metadata gốc](../descriptions/vi_to_zh_context/README.md).
- [Checksum của hai file full](manifest.json).

Điểm, bản dịch ứng viên, model chọn, QA, danh sách pair và metadata quy trình không nằm trong file full. Dữ liệu dịch máy, chưa xác minh ngữ nghĩa thủ công mọi dòng; chia train/dev/test theo `product_id` để tránh lẫn sản phẩm giữa các tập.
''')
text('data/tiki/full_dataset/manifest.json',json.dumps({'schema':'tiki_full_core_v1','columns':columns,'join_key':'product_id','description_null_policy':'retain_13_products_without_description','exports':exports},indent=2))
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';base=api.repo_info(repo,repo_type='dataset').sha
commit=api.create_commit(repo,repo_type='dataset',parent_commit=base,operations=ops,commit_message='Simplify Tiki full datasets to ID, source and translated titles and descriptions')
for item in exports:
    p=hf_hub_download(repo,item['path'],repo_type='dataset',revision=commit.oid,token=os.environ['HF_TOKEN'])
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==item['sha256']
    assert list(pd.read_parquet(p).columns)==columns
result={'revision':commit.oid,'exports':exports,'component_exports_unchanged':True}
(OUT/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
