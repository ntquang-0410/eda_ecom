
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
from pathlib import Path
import os,json
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom';head=api.repo_info(repo,repo_type='dataset').sha;ops=[];out=Path('outputs/full_9col_final');report=json.loads((out/'upload_result.json').read_text())['exports']
for ds in ['1688','tiki']:
 p=out/(ds+'_README.md');s=p.read_text(encoding='utf8').replace('469 title 1688 `existing_alternative` đã chọn bản điểm thấp hơn','450 title 1688 `existing_alternative` đã chọn bản điểm thấp hơn');p.write_text(s,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=f'data/{ds}/full_dataset/README.md',path_or_fileobj=str(p)))
for r in report:
 ds=r['dataset'];filename=Path(r['path']).name;name=filename.replace('.parquet','.README.md');old=f'data/{ds}/full_dataset/{name}'
 # Tiki has existing method documentation. Keep it accurate; 1688 gets matching method pages.
 method='pair — dịch từng cặp thuộc tính' if r['method']=='independent' else 'cụm — dịch bảng thuộc tính theo ngữ cảnh sản phẩm'
 text=f'''# {ds} full — {method}\n\nFile [{filename}]({filename}) có {r['rows']:,} dòng và đúng 9 cột theo [README bộ full](README.md). Có {r['missing_descriptions']} sản phẩm thiếu description; hai cột description và `desc_nguon_dich` để null. Category lấy từ cleaned, không null.\n\nNguồn description là bảng cấp sản phẩm do NA cung cấp. Hai cách dùng các bảng riêng, không trộn; dữ liệu title ghép trái theo product_id và giữ nguyên thứ tự. Title là bản chọn COMET và có thể đã sửa sau đó. Dữ liệu dịch máy chưa được xác minh ngữ nghĩa thủ công toàn bộ.\n\nSHA256: `{r['sha256']}`. Xem [manifest](manifest.json) để đối chiếu nguồn và checksum.\n'''
 p=out/(ds+'_'+name);p.write_text(text,encoding='utf8');ops.append(CommitOperationAdd(path_in_repo=old,path_or_fileobj=str(p)))
c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Correct verified lower-score count and document each nine-column dataset');print(json.dumps({'revision':c.oid,'actual_existing_alternative_lower_score_unique':450,'readmes':len(ops)}))
