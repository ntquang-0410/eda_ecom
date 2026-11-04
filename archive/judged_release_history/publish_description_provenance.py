"""Publish only the agreed context provenance display and its documentation."""
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

WORK = Path(r'D:\download\NCKH\judge_kaggle_preflight')
OUT = WORK / 'cum_description_provenance_update'
REPO = 'ntquang0410/zh-vie_ecom'
PREFIX = 'data/LLM/bilingual_judged/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((OUT / 'description_provenance_validation.json').read_text(encoding='utf8'))
    api = HfApi()
    head = api.dataset_info(REPO).sha
    assert head == report['source_revision'], 'Remote changed; stop to inspect rather than overwrite'

    def download(name, revision=head):
        return Path(hf_hub_download(REPO, PREFIX + name, repo_type='dataset', revision=revision))

    old = pd.read_parquet(download('bo_cum_chon_deepseek.parquet'))
    new = pd.read_parquet(OUT / 'bo_cum_chon_deepseek.parquet')
    trace = pd.read_parquet(OUT / 'bo_cum_description_provenance.parquet')
    assert len(new) == 25575 and list(new) == list(old)
    assert new.drop(columns='judge_chon_description').equals(old.drop(columns='judge_chon_description'))
    assert not new.duplicated(['source', 'product_id']).any()
    assert not trace.duplicated(['source', 'product_id', 'pair_id']).any()
    assert set(zip(trace.source, trace.product_id)) == set(zip(new.source, new.product_id))
    counts = trace.groupby(['source', 'product_id', 'final_origin']).size()
    for r in new.itertuples():
        parts = r.judge_chon_description.split(' | ')
        assert all(re.fullmatch(r'(gemma|qwen|quy_tac|chua_ro): [1-9][0-9]*', p) for p in parts)
        displayed = {p.split(': ')[0]: int(p.split(': ')[1]) for p in parts}
        assert len(displayed) == len(parts)
        expected = {label: int(counts.get((r.source, r.product_id, label), 0)) for label in ('gemma', 'qwen', 'quy_tac', 'chua_ro')}
        assert displayed == {k: v for k, v in expected.items() if v > 0}

    readme = download('README.md').read_text(encoding='utf8')
    start = readme.index('### judge_chon_description\n')
    end = readme.index('### File phụ phục vụ truy vết\n', start)
    totals = report['attribute_origins']
    section = f'''### judge_chon_description
Thành phần nguồn của **description cuối đang giữ sau pipeline luật, judge và rà soát**. Không phải điểm số, số lần thắng judge, số từ hoặc số câu. Mỗi mảnh là một cặp thuộc tính `tên: giá trị`, lấy từ bảng cặp gốc theo thứ tự ghép; không tách lại bằng dấu chấm phẩy vì giá trị có thể chứa dấu đó.

Đếm cả các mảnh giữ nguyên khi truy được nguồn. Chỉ hiển thị nhóm có số lớn hơn 0, theo thứ tự `gemma`, `qwen`, `quy_tac`, `chua_ro`:
- `gemma: 5`: năm mảnh có thành phần do Gemma tạo, phần còn lại trong mỗi mảnh chỉ là tên/giá trị cố định theo quy tắc.
- `gemma: 3 | qwen: 2`: ba mảnh Gemma và hai mảnh Qwen.
- `gemma: 3 | quy_tac: 1`: ba mảnh Gemma, một mảnh hoàn toàn từ điển/quy tắc.
- `quy_tac: 4`: cả bốn mảnh là nội dung cố định, sao chép hoặc từ điển theo bằng chứng pipeline.
- `gemma: 3 | chua_ro: 1`: một mảnh chưa gán được một nguồn đơn chắc chắn.

**Nguồn được truy từ dấu vết sinh/sửa và bảng đơn vị đã chọn**, không suy từ tên ứng viên A/B, không gán mặc định Gemma khi hai ứng viên giống nhau. Ứng viên có thể kế thừa nội dung bước trước. Nếu có nhãn chọn model rõ ở bước trước thì được dùng ngay cả khi hai bản giống nhau; chỉ giống chữ mà thiếu dấu vết thì không đủ bằng chứng. Tên/giá trị khóa cứng được ghi quy tắc; phần giữ nguyên được đối chiếu đúng văn bản, loại đơn vị và category. Đề xuất judge bị luật từ chối không được gán cho nguồn đề xuất đó.

Một cặp thuộc tính được **đếm đúng một lần**. Khi tên và giá trị chỉ gồm quy tắc + một model, tính cho model đó. Nếu một phần chưa truy được nguồn, hoặc tên và giá trị thuộc hai model khác nhau, cả mảnh được ghi `chua_ro` để không đếm đôi/ép thành một model. Trường hợp phối hợp hai model được phân biệt bằng `origin_reason=mixed_models_within_one_attribute` và hai cột `name_origin`, `value_origin` trong bảng truy vết; không có nghĩa hai phần đều không biết nguồn. `chua_ro` là hạn chế truy vết, **không phải kết luận bản dịch sai hoặc chưa được judge**.

Tổng hiện tại: **{report['attributes']:,} lượt mảnh**, gồm {totals.get('gemma', 0):,} Gemma, {totals.get('qwen', 0):,} Qwen, {totals.get('quy_tac', 0):,} quy tắc và {totals.get('chua_ro', 0):,} chưa gán được một nguồn đơn. Tổng các số hiển thị khớp số cặp thuộc tính ở từng sản phẩm. Không còn chuỗi `gemma: 0 | qwen: 0`.

Cách đếm cũ chỉ tính đề xuất model rõ được chấp nhận (9.110 Gemma, 642 Qwen, 17.601 sản phẩm có hai số 0). Đó là **thống kê lịch sử về quyết định**, không phải định nghĩa hiện tại của cột. Các số ấy được giữ ở `bo_cum_fragment_selection_trace.parquet` và các cột `*_duoc_ap_dung` của bảng đánh dấu để truy lại, không sửa giả kết quả judge.

'''
    readme = readme[:start] + section + readme[end:]
    readme = readme.replace('### File phụ phục vụ truy vết\n', '### File phụ phục vụ truy vết\n- `bo_cum_description_provenance.parquet`: toàn bộ cặp thuộc tính trong tập sản phẩm cuối, nguồn tên/giá trị, nhãn nguồn cuối, bằng chứng và liên kết item judge. Đây là bảng dùng để tính cột hiển thị hiện tại. Không chứa lại văn bản dịch.\n- `description_provenance_validation.json`: số lượng, băm các đầu vào truy vết và kiểm tra nội dung không đổi; tổng mảnh khớp từng sản phẩm.\n', 1)
    readme = readme.replace('`bo_cum_fragment_selection_trace.parquet`: source, product_id, item_id, model đề xuất, accepted, text_changed, review_origin và lý do chấp nhận/từ chối.', '`bo_cum_fragment_selection_trace.parquet`: bảng quyết định lịch sử, gồm source, product_id, item_id, model đề xuất, accepted, text_changed, review_origin và lý do chấp nhận/từ chối.')
    (OUT / 'README.md').write_text(readme, encoding='utf8')
    validation = json.loads(download('validation_report.json').read_text(encoding='utf8'))
    validation['cum_description_provenance'] = report
    (OUT / 'validation_report.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf8')
    manifest = json.loads(download('manifest.json').read_text(encoding='utf8'))
    manifest['judge_chon_description_definition'] = 'final_attribute_origin_counts_including_retained_fragments_nonzero_groups_only'
    manifest['description_provenance_source_revision'] = head
    for p in OUT.iterdir():
        if p.is_file() and p.name not in ('manifest.json', 'SHA256SUMS.txt'):
            manifest['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    sums = {name: entry['sha256'] for name, entry in manifest['files'].items()}
    sums['manifest.json'] = sha(OUT / 'manifest.json')
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {name}\n' for name, h in sorted(sums.items())), encoding='ascii')
    files = sorted(p for p in OUT.iterdir() if p.is_file())
    # Mutations are limited to this release folder; original inputs and pair stay untouched.
    assert {p.name for p in files} == {'bo_cum_chon_deepseek.parquet', 'bo_cum_description_provenance.parquet', 'description_provenance_validation.json', 'README.md', 'validation_report.json', 'manifest.json', 'SHA256SUMS.txt'}
    print('Local verification passed; publishing 7 context provenance files', flush=True)
    commit = api.create_commit(repo_id=REPO, repo_type='dataset', parent_commit=head, num_threads=1,
        operations=[CommitOperationAdd(path_in_repo=PREFIX + p.name, path_or_fileobj=str(p)) for p in files],
        commit_message='Show final description fragment sources including retained text; hide zero counts')
    print('COMMIT', commit.oid, flush=True)
    for name, expected in sums.items():
        assert sha(download(name, commit.oid)) == expected, f'Remote hash mismatch: {name}'
    assert sha(download('SHA256SUMS.txt', commit.oid)) == sha(OUT / 'SHA256SUMS.txt')
    remote_new = pd.read_parquet(download('bo_cum_chon_deepseek.parquet', commit.oid))
    assert remote_new.equals(new)
    result = {'commit': commit.oid, 'rows': len(new), 'columns': len(new.columns),
              'changed_files': len(files), 'verified_remote_files': len(sums) + 1,
              'only_main_column_changed': 'judge_chon_description', 'pair_unchanged': True,
              'all_attribute_counts_reconciled': True, 'attribute_origins': totals}
    (WORK / 'cum_description_provenance_upload_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
