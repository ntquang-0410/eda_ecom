
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,hashlib,posixpath,re
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
OUT=ROOT/'outputs/individual_translation_readmes';OUT.mkdir(parents=True,exist_ok=True)
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';base=api.repo_info(repo,repo_type='dataset').sha
existing=set(api.list_repo_files(repo,repo_type='dataset',revision=base));docs={}
common='''## Phạm vi và cách dùng

Mỗi dòng tương ứng một sản phẩm; các sản phẩm có cùng title nguồn có thể dùng chung bản dịch. `product_id` dùng để đối chiếu, không phải văn bản đầu vào cần dịch. Các file title chỉ giữ title và thông tin liên quan tới dịch; không chứa description hay thuộc tính sản phẩm khác.

Đây là bản dịch máy. Cờ kiểm tra kỹ thuật và `semantic_verified=false` được giữ để tham khảo; chúng không có nghĩa mọi bản dịch đã được người xác nhận đúng ngữ nghĩa.
'''
specs=[
 ('1688','gemma4','1688_zh_to_vi_gemma4.parquet','outputs/translation_only/gemma_1688/bilingual_zh_vi_cado_title_gemma4.parquet'),
 ('1688','qwen3_14b_awq','1688_zh_to_vi_qwen3_14b_awq.parquet','outputs/translation_only/qwen_1688_qa_repaired/qwen_1688_qa_repaired.parquet'),
 ('tiki','gemma4','tiki_vi_to_zh_gemma4.parquet','outputs/hf_title_exports/tiki_vi_to_zh_gemma4.parquet'),
 ('tiki','qwen3_14b_awq','tiki_vi_to_zh_qwen3_14b_awq.parquet','outputs/translation_only/qwen_tiki_qa_repaired/qwen_tiki_qa_repaired.parquet'),
]
def column_note(c):
    exact={'product_id':'ID sản phẩm nguồn.','title_zh':'Title tiếng Trung; là nguồn ở 1688 và đích ở Tiki.','title_vi':'Title tiếng Việt; là đích ở 1688 và nguồn ở Tiki.','original_candidate':'Ứng viên trước lượt sửa kỹ thuật.','qa_repair_accepted':'Ứng viên sửa được chấp nhận theo quy tắc kỹ thuật.','generation_status':'Trạng thái sinh bản dịch.','generation_truncated':'Dấu hiệu kết quả sinh bị cắt cụt.','candidate_stage':'Giai đoạn tạo ứng viên: gốc hoặc sửa kỹ thuật.','selected_model':'Model của bản được giữ lại.','selection_reason':'Chọn theo điểm cao hơn hoặc chỉ có một ứng viên sinh hợp lệ.','selection_policy':'Tên chính sách chọn bản dịch.','selected_cometkiwi_score':'Điểm của ứng viên được chọn.','gemma4_translation':'Nội dung ứng viên Gemma.','qwen3_14b_awq_translation':'Nội dung ứng viên Qwen.'}
    if c in exact:return exact[c]
    if 'semantic_verified' in c:return 'Chưa xác minh ngữ nghĩa thủ công từng title (`false`).'
    if 'qa' in c or 'advisory' in c:return 'Trạng thái/cờ kiểm tra kỹ thuật; giữ để đối chiếu.'
    if 'cometkiwi' in c:return 'Điểm COMETKiwi của ứng viên tương ứng; không phải xác suất đúng.'
    if 'generation_ok' in c:return 'Ứng viên có trạng thái sinh hợp lệ, độc lập với cờ kỹ thuật.'
    if 'model_revision' in c:return 'Revision model được ghi trong pipeline.'
    if 'model' in c:return 'Định danh model tạo bản dịch.'
    if 'prompt' in c:return 'Phiên bản prompt được ghi khi tạo bản dịch.'
    if 'provider' in c or 'api' in c:return 'Thông tin nguồn cung cấp hoặc lượt gọi API được pipeline ghi nhận.'
    return 'Metadata phục vụ truy vết lượt dịch.'
def schema(df):return '| Cột | Ý nghĩa |\n|---|---|\n'+'\n'.join(f'| `{c}` | {column_note(c)} |' for c in df)
def read_example(remote):return f"```python\nimport pandas as pd\ndf = pd.read_parquet('hf://datasets/{repo}/{remote}')\n```\n"
for ds,model,filename,local in specs:
    folder=f'data/{ds}/title_translations/{model}';remote=folder+'/'+filename;assert remote in existing
    frame=pd.read_parquet(local);rows=16348 if ds=='1688' else 13200;unique=15863 if ds=='1688' else 13102
    assert len(frame)==rows
    direction='Trung → Việt' if ds=='1688' else 'Việt → Trung'
    modelname='Gemma `gemma-4-26b-a4b-it`' if model=='gemma4' else 'Qwen `Qwen/Qwen3-14B-AWQ`'
    if ds=='1688' and model=='gemma4':
        status='Lượt dịch phủ 15.863 title nguồn. Bản xuất riêng còn 746 title có cờ kỹ thuật được pipeline lưu; không diễn giải đây là số title đã được xác nhận sai bằng người.'
    elif ds=='1688':
        status='Đây là bản sau sửa kỹ thuật. Nhóm sửa gồm 3.599 title; 804 ứng viên sửa được chấp nhận, 2.795 title vẫn giữ cờ. Lượt sửa có giới hạn tối đa hai lần thử mỗi title, không xóa cờ để ép đạt. Kết quả sửa có provenance riêng; không coi prompt sửa là prompt gốc.'
    elif model=='gemma4':
        status='13.090/13.102 title sinh thành công; còn 12 title lỗi sinh. File vẫn giữ đủ 13.200 dòng. Nếu cần bản đủ title dịch không trống, dùng bản kết hợp COMETKiwi bên cạnh.'
    else:
        status='Qwen sinh đủ 13.102 title. Nhóm sửa kỹ thuật gồm 2.853 title; chấp nhận 370 bản sửa và còn 2.483 title giữ cờ. Đây là kết quả sau lượt sửa hữu hạn, không phải bộ đã xác minh ngữ nghĩa hoàn toàn.'
    extra=''
    if model=='qwen3_14b_awq':extra='Model được khóa revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`. Cấu hình sinh: temperature 0,7; top_p 0,8; top_k 20; repetition_penalty 1,05; tối đa 256 token; không thinking. Lượt sửa dùng prompt `qwen-technical-qa-repair-v1`; chỉ nhận bản sửa vượt kiểm tra kỹ thuật, giữ bản trước đó khi sửa thất bại.'
    docs[folder+'/README.md']=f'''# {ds} — {modelname} — title {direction}

## File và quy mô

- File: [{filename}]({filename}).
- {rows:,} dòng sản phẩm, {unique:,} title nguồn duy nhất.
- SHA256: `{hashlib.sha256(Path(local).read_bytes()).hexdigest()}`.
- Chiều dịch: **{direction}**.

## Cách tạo và trạng thái

{status}

{extra}

{common}

## Schema file hiện tại

{schema(frame)}

## Quan hệ với COMETKiwi

File này là bản riêng của model, không phải kết quả chọn giữa hai model. [Bản kết hợp COMETKiwi](../best_cometkiwi/README.md) giữ ứng viên có điểm cao hơn giữa các bản không lỗi sinh, không dùng ngưỡng 0,72 để bỏ title; cờ kỹ thuật chỉ để tham khảo.

## Đọc dữ liệu

{read_example(remote)}
'''

reports=json.loads(Path('outputs/comet_best_all_titles/upload_result.json').read_text())['reports']
for report in reports:
    ds=report['dataset'];name=f'{ds}_{"zh_to_vi" if ds=="1688" else "vi_to_zh"}_best_cometkiwi';local=Path('outputs/comet_best_all_titles')/name/(name+'.parquet');frame=pd.read_parquet(local)
    folder=f'data/{ds}/title_translations/best_cometkiwi';remote=folder+'/'+local.name;assert remote in existing
    scores=report['selection_counts_unique'];counts=report['selection_counts_product_rows']
    docs[folder+'/README.md']=f'''# {ds} — Chọn title Gemma/Qwen bằng COMETKiwi

## Kết quả đang phát hành

- File: [{local.name}]({local.name}).
- **{report['rows']:,} dòng sản phẩm / {report['unique_titles']:,} title nguồn duy nhất**.
- Title dịch được chọn bị trống: **0**; không bỏ dòng sản phẩm.
- Theo title duy nhất: Gemma **{scores['gemma4']:,}**, Qwen **{scores['qwen3_14b_awq']:,}**.
- Theo dòng sản phẩm: Gemma **{counts['gemma4']:,}**, Qwen **{counts['qwen3_14b_awq']:,}**.
- SHA256: `{report['sha256']}`. Báo cáo máy đọc được: [selection_report.json](selection_report.json).

## Cách chấm

Model: `Unbabel/wmt22-cometkiwi-da`, revision `1ad785194e391eebc6c53e2d0776cada8f83179a`. Mỗi ứng viên Gemma và Qwen được chấm với cùng title nguồn theo cặp `(src, mt)`, không có bản dịch tham chiếu do người tạo. Nguồn là tiếng Trung ở 1688 và tiếng Việt ở Tiki. Điểm chỉ dùng để xếp hạng hai ứng viên trong cùng title; không phải phần trăm độ chính xác.

## Quy tắc chọn cuối cùng

1. Giữ đủ các dòng và ID theo thứ tự nguồn.
2. Dùng các ứng viên không trống, không lỗi sinh; nếu chỉ có một bản sinh hợp lệ thì giữ bản đó.
3. Nếu có hai bản dùng được, giữ bản có điểm cao hơn; điểm bằng nhau chọn Gemma để tái lập.
4. **Không dùng ngưỡng 0,72. Không loại ứng viên chỉ vì cờ kiểm tra kỹ thuật.** Cờ được giữ để đối chiếu.
5. Giữ cả hai bản dịch, hai điểm và model được chọn. Bước này chỉ chọn lại từ điểm đã chấm, không dịch lại.

Hai lượt đánh giá 1688 và Tiki là cùng một phương pháp, áp dụng cho hai chiều dịch khác nhau; không nên coi là hai thuật toán COMET khác nhau. Trong trường hợp thiếu ứng viên trống, điểm của ứng viên đó để null, không gán điểm giả. Với hai bộ hiện tại không có dòng nào phải chọn ứng viên lỗi sinh làm phương án cuối.

## Schema

{schema(frame)}

## Giới hạn

Chọn được điểm cao hơn không bảo đảm đúng mọi tên riêng, mã hàng, số đo hoặc nghĩa chuyên ngành. `semantic_verified=false` vẫn được giữ. Bộ này là dữ liệu dịch máy, không phải nhãn chuẩn do người dịch. Không so trực tiếp điểm giữa hai chiều dịch để kết luận bộ nào chính xác hơn.

## Đọc dữ liệu

{read_example(remote)}
'''

full=json.loads(Path('outputs/tiki_full_dataset/upload_result.json').read_text())['exports']
for entry in full:
    context=entry['method']=='attributes_context';label='Cách 2 — dịch cả cụm thuộc tính theo sản phẩm' if context else 'Cách 1 — dịch theo pair (từng cặp thuộc tính)'
    filename=entry['path'].rsplit('/',1)[1]
    process=('''Dùng bản dịch từng pair làm nháp; hai model nhìn title tiếng Việt cùng cả danh sách thuộc tính của một sản phẩm để rà nghĩa và sửa chỗ cần thiết. Các mục cố định/từ điển được giữ; chỉnh sửa không hợp lệ quay về bản nháp. Tài liệu nguồn mô tả thứ tự ưu tiên đầu ra: mục cố định, Gemma hợp lệ, Qwen hợp lệ, rồi bản nháp. `nguon_ban_dich` trong các cặp cho biết nguồn phần giá trị được giữ.

“Cả cụm” nghĩa là **cụm/bảng thuộc tính của một sản phẩm kèm title làm ngữ cảnh**, không phải toàn bộ văn xuôi hay hình ảnh của trang sản phẩm. File full giữ nguyên `description_vi` và `description_zh` trong bảng `description_products_vi_zh.parquet`; không dựng hoặc dịch lại các chuỗi này khi ghép title.''' if context else '''Tách bảng thuộc tính thành các pair có tên và giá trị. Các đơn vị tên/giá trị được dịch riêng bằng Gemma/Qwen theo pipeline nguồn, kết hợp kiểm tra kỹ thuật, ước lượng chất lượng và từ điển/mục cố định. Các trường thương hiệu, mã, số đo có thể được giữ nguyên theo quy tắc của dữ liệu nguồn.

Để có một dòng cho mỗi sản phẩm, sắp các pair theo `pair_id`, ghép mỗi pair thành `name: value`, nối bằng newline để tạo `description_vi` và `description_zh`. Đây là **tổng hợp các thuộc tính đã dịch**, không phải một lượt model dịch toàn bộ đoạn description. Không dùng ngữ cảnh cả cụm trong lượt ghép này.''')
    source='../descriptions/vi_to_zh_context/README.md' if context else '../descriptions/vi_to_zh/README.md'
    docs['data/tiki/full_dataset/'+filename.replace('.parquet','.README.md')]=f'''# Tiki full — {label}

## File

[{filename}]({filename}) — **13.200 sản phẩm**, 13.102 title tiếng Việt duy nhất. Description có cho 13.187 sản phẩm; 13 sản phẩm chưa có description nguồn vẫn được giữ.

SHA256: `{entry['sha256']}`.

## Cách thực hiện description

{process}

Nguồn có **{entry['attribute_pairs']:,} pair thuộc tính**. Danh sách `attribute_pairs` trong file full giữ các pair gốc, ID, độ tin cậy, và nguồn bản dịch nếu có. Xem [README của pipeline description nguồn]({source}) để biết quy tắc chi tiết, provenance và điều kiện sử dụng.

## Ghép với title

Title dùng chung [bản Tiki chọn bằng COMETKiwi](../title_translations/best_cometkiwi/README.md). Ghép `left` theo `product_id`, quan hệ một-một ở bảng sản phẩm; giữ nguyên ID, thứ tự và mọi cột title/điểm/model. Không dùng COMET title để chấm lại hoặc thay đổi description. Không trộn pair của hai cách thành một bản.

## Cột và dữ liệu thiếu

- `product_id`, `title_vi`, `title_zh`: ID, title gốc và title được chọn.
- `description_vi`, `description_zh`: description song ngữ theo cách trên.
- `attribute_pairs`, `attribute_count`: pair chi tiết và số pair của sản phẩm.
- Metadata title: hai ứng viên, hai điểm COMET, model chọn và trạng thái sinh.
- Metadata description: độ tin cậy theo nguồn và cách xử lý; `description_method` là `{entry['method']}`.
- `has_description=false` cho 13 sản phẩm không có trong bản phát hành description; chuỗi description để null, không tự tạo nội dung.

## Kiểm chứng và sử dụng

Đã kiểm tra 13.200 ID không trùng, mọi cột title giữ nguyên, tổng số pair, roundtrip Parquet và checksum local/HF. {'Hai chuỗi description giữ nguyên bảng sản phẩm nguồn.' if context else 'Thứ tự pair theo pair_id được giữ khi dựng description.'} Khi chia train/dev/test, chia theo `product_id` để các pair của một sản phẩm không lọt qua nhiều tập. Dữ liệu do model tạo, chưa được kiểm tra ngữ nghĩa thủ công toàn bộ. Giữ điều kiện sử dụng của các nguồn/model liên quan.

{read_example(entry['path'])}
'''

docs['data/tiki/full_dataset/README.md']='''# Tiki full: title + description song ngữ Việt–Trung

Hai bản dùng **cùng title đã chọn bằng COMETKiwi**, khác nhau ở cách tạo description. Mỗi bản có 13.200 dòng sản phẩm / 13.102 title duy nhất. Có description cho 13.187 sản phẩm; 13 dòng còn lại giữ title, description null và đánh dấu thiếu.

| Tiêu chí | Cách 1: theo pair | Cách 2: cả cụm/ngữ cảnh sản phẩm |
|---|---|---|
| Đơn vị xử lý | Tên/giá trị thuộc tính riêng | Cả bảng thuộc tính kèm title sản phẩm |
| Cơ sở | Dịch độc lập, quy tắc và từ điển | Bản pair làm nháp; model xem cả cụm để rà và sửa |
| Dựng description trong bản full | Nối các dòng `name: value` theo `pair_id` | Giữ nguyên description cấp sản phẩm đã cung cấp |
| Số pair | 87.056 | 87.112 |
| Tài liệu riêng | [README cách 1](tiki_full_attributes_independent_vi_zh.README.md) | [README cách 2](tiki_full_attributes_context_vi_zh.README.md) |
| Parquet | [Bản theo pair](tiki_full_attributes_independent_vi_zh.parquet) | [Bản cả cụm](tiki_full_attributes_context_vi_zh.parquet) |

“Cả cụm” là cụm thuộc tính có ngữ cảnh, **không đồng nghĩa dịch toàn bộ văn xuôi hoặc hình ảnh mô tả trên trang sản phẩm**. Cả hai bản full được ghép từ dữ liệu dịch đã có, không gọi model dịch lại. Bản cả cụm không được mặc định coi là tốt hơn trong mọi dòng.

Các cột chính: `product_id`, `title_vi`, `title_zh`, `description_vi`, `description_zh`, `attribute_pairs`, `attribute_count`, `has_description`, `description_method`. Giữ thêm các ứng viên/điểm/model title và metadata description phục vụ đối chiếu; không thêm giá/shop hoặc thông tin sản phẩm không liên quan.

Quy tắc title: giữ bản có điểm COMET cao hơn giữa các bản không lỗi sinh; không dùng ngưỡng 0,72 và không bỏ title vì cờ kỹ thuật. Xem [README chọn title](../title_translations/best_cometkiwi/README.md). Bước ghép full không áp quy tắc này lên description.

[manifest.json](manifest.json) ghi nguồn, SHA256 và số lượng. Giữ nguyên ID, thứ tự title; join theo `product_id`, không theo chuỗi title để tránh ghép lẫn sản phẩm. Dữ liệu dịch máy chưa xác minh ngữ nghĩa từng dòng; khi huấn luyện nên chia tập theo sản phẩm.
'''
for ds in ['1688','tiki']:
    docs[f'data/{ds}/title_translations/README.md']=f'''# Title {ds}

- [Bản Gemma](gemma4/README.md).
- [Bản Qwen3-14B-AWQ sau sửa kỹ thuật](qwen3_14b_awq/README.md).
- [Bản chọn Gemma/Qwen bằng COMETKiwi](best_cometkiwi/README.md).

Mỗi thư mục có README riêng, Parquet và metadata tương ứng. {'Chiều Trung → Việt, 16.348 dòng / 15.863 title.' if ds=='1688' else 'Chiều Việt → Trung, 13.200 dòng / 13.102 title. Xem thêm [hai bản full title + description](../full_dataset/README.md).'}
'''

# Check repository-relative Markdown links without fetching or printing data rows.
allpaths=existing|set(docs)
for remote,content in docs.items():
    for link in re.findall(r'\]\(([^)]+)\)',content):
        if '://' in link or link.startswith('#'):continue
        target=posixpath.normpath(posixpath.join(posixpath.dirname(remote),link.split('#',1)[0]))
        assert target in allpaths,(remote,target)
ops=[]
for remote,content in docs.items():
    path=OUT/remote;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
commit=api.create_commit(repo,repo_type='dataset',parent_commit=base,operations=ops,commit_message='Document each model title export, both COMET selections and pair versus contextual Tiki full datasets')
for remote in docs:
    p=hf_hub_download(repo,remote,repo_type='dataset',revision=commit.oid,token=os.environ['HF_TOKEN'])
    assert Path(p).read_bytes()==(OUT/remote).read_bytes()
result={'revision':commit.oid,'readmes_written':len(docs),'paths':list(docs),'only_markdown_modified':True}
(OUT/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
