"""Undo shared layout changes; organize only owned exports and merge Tiki titles/descriptions."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import os,json,hashlib,sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationCopy,CommitOperationDelete,hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
OUT=ROOT/'outputs/tiki_full_dataset';OUT.mkdir(parents=True,exist_ok=True)
api=HfApi(token=os.environ['HF_TOKEN']);repo='ntquang0410/zh-vie_ecom'
plan=json.loads(Path('outputs/hf_reorganized/move_plan.json').read_text());base=plan['base'];mapping=plan['mapping']
head=api.repo_info(repo,repo_type='dataset').sha
current=set(api.list_repo_files(repo,repo_type='dataset',revision=head))
original=set(api.list_repo_files(repo,repo_type='dataset',revision=base))
KEEP_LAYOUT='--keep-layout' in sys.argv
ops=[];deletes=set();moves={};restored=[]
for old,moved in ([] if KEEP_LAYOUT else mapping.items()):
    if old.startswith('data/translated/title_only/'):
        suffix=old[len('data/translated/title_only/'):]
        if suffix in ('README.md','manifest.json'):continue
        folder,filename=suffix.split('/',1)
        dataset='1688' if folder.startswith('1688_') else 'tiki'
        model='best_cometkiwi' if 'best_cometkiwi' in folder else 'qwen3_14b_awq' if 'qwen3' in folder else 'gemma4'
        target=f'data/translated/title_only/{dataset}/{model}/{filename}'
        assert moved in current,'missing_current_export'
        ops.append(CommitOperationCopy(src_path_in_repo=moved,src_revision=head,path_in_repo=target))
        moves[moved]=target
    else:
        assert old in original
        ops.append(CommitOperationCopy(src_path_in_repo=old,src_revision=base,path_in_repo=old));restored.append(old)
    if moved and moved!=old and moved in current:deletes.add(moved)
for extra in ([] if KEEP_LAYOUT else ['data/1688/README.md','data/tiki/README.md','data/tiki/cleaned/README_legacy_shared.md','manifest.json']):
    if extra in current and extra not in original:deletes.add(extra)
# Restore the shared root README byte-for-byte, not another rewritten project card.
if not KEEP_LAYOUT:
    ops.append(CommitOperationCopy(src_path_in_repo='README.md',src_revision=base,path_in_repo='README.md'));restored.append('README.md')

sources=ROOT/'outputs/tiki_full_sources'
def load(name):
    p=Path((sources/(name+'.path')).read_text());return pd.read_parquet(p),hashlib.sha256(p.read_bytes()).hexdigest()
ind,indsha=load('description_pairs_vi_zh.parquet')
ctx,ctxsha=load('description_pairs_vi_zh_ctx.parquet')
products,productsha=load('description_products_vi_zh.parquet')
titlepath=ROOT/'outputs/comet_best_all_titles/tiki_vi_to_zh_best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet'
titles=pd.read_parquet(titlepath)
assert len(titles)==13200 and titles.product_id.is_unique
assert len(products)==13187 and products.product_id.is_unique
assert len(ind)==87056 and len(ctx)==87112
assert set(ind.product_id)==set(ctx.product_id)==set(products.product_id)
assert set(products.product_id)<=set(titles.product_id)

def aggregate_pairs(pairs):
    assert pairs.pair_id.is_unique
    entries=[]
    for pid,group in pairs.sort_values('pair_id',kind='stable').groupby('product_id',sort=False):
        assert group.category.nunique()==1
        records=group.drop(columns=['product_id','category']).to_dict('records')
        entries.append({'product_id':pid,'category':group.category.iloc[0],
            'description_vi':'\n'.join(f'{n}: {v}' for n,v in zip(group.name_vi,group.gia_tri_vi)),
            'description_zh':'\n'.join(f'{n}: {v}' for n,v in zip(group.name_zh,group.gia_tri_zh)),
            'attribute_count':len(group),'attribute_pairs':records,
            'description_confidence_counts':json.dumps(group.do_tin_cay.value_counts().to_dict(),ensure_ascii=False,sort_keys=True)})
    return pd.DataFrame(entries)
indproducts=aggregate_pairs(ind);ctxpairs=aggregate_pairs(ctx)
assert len(indproducts)==len(ctxpairs)==13187
# Preserve the supplied contextual whole-description strings without regenerating them.
ctxproducts=products.rename(columns={'so_cap':'attribute_count','do_tin_cay':'description_confidence','so_cap_khong_cao':'non_high_confidence_attribute_count'}).merge(ctxpairs[['product_id','attribute_pairs','description_confidence_counts']],on='product_id',how='left',validate='one_to_one')
expectedcounts=ctx.groupby('product_id').size()
assert all(n==expectedcounts[pid] for pid,n in zip(ctxproducts.product_id,ctxproducts.attribute_count))
exports=[]
for method,descs,pairsha in [('attributes_independent',indproducts,indsha),('attributes_context',ctxproducts,ctxsha)]:
    frame=titles.merge(descs,on='product_id',how='left',sort=False,validate='one_to_one')
    assert len(frame)==13200 and frame.product_id.equals(titles.product_id)
    for col in titles:assert frame[col].equals(titles[col]),'title_data_changed'
    frame['has_description']=frame.description_vi.notna() & frame.description_zh.notna()
    frame['description_method']=method
    frame['description_missing_reason']=frame.has_description.map({True:None,False:'product_not_in_description_release'})
    assert int(frame.has_description.sum())==13187 and int((~frame.has_description).sum())==13
    assert int(frame.attribute_count.fillna(0).sum())==(87056 if method=='attributes_independent' else 87112)
    path=OUT/f'tiki_full_{method}_vi_zh.parquet'
    table=pa.Table.from_pandas(frame,preserve_index=False);pq.write_table(table,path,compression='snappy')
    assert table.equals(pq.read_table(path)), 'parquet_roundtrip_mismatch'
    if method=='attributes_context':
        check=frame[frame.has_description].set_index('product_id')
        for col in ['description_vi','description_zh']:
            assert check.loc[products.product_id,col].tolist()==products[col].tolist()
    remote=('data/tiki/full_dataset/' if KEEP_LAYOUT else 'data/translated/full_dataset/tiki/')+path.name
    ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path)))
    exports.append({'path':remote,'rows':13200,'unique_products':13200,'unique_titles':int(frame.title_vi.nunique()),'products_with_description':13187,'products_missing_description':13,'attribute_pairs':87056 if method=='attributes_independent' else 87112,'method':method,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'title_source_sha256':hashlib.sha256(titlepath.read_bytes()).hexdigest(),'description_pairs_sha256':pairsha,'description_products_sha256':productsha if method=='attributes_context' else None})

def text(remote,value):
    if KEEP_LAYOUT:
        if remote.startswith('data/translated/title_only/'):return
        remote=remote.replace('data/translated/full_dataset/','data/tiki/full_dataset/')
        value=value.replace('../title_only/','../title_translations/').replace('trong `tiki/`','trong thư mục này')
        value=value.replace('data/description_vi_to_zh/','data/tiki/descriptions/vi_to_zh/').replace('data/description_vi_to_zh_cum/','data/tiki/descriptions/vi_to_zh_context/')
    p=OUT/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8');ops.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)))
text('data/translated/full_dataset/README.md','''# Full datasets: title + description

Thư mục này nằm cạnh `../title_only/` và chỉ chứa các bản ghép, không thay đổi dữ liệu nguồn chung.

## Tiki Việt → Trung

| File trong `tiki/` | Cách description | Dòng sản phẩm | Có description | Cặp thuộc tính |
|---|---|---:|---:|---:|
| `tiki_full_attributes_independent_vi_zh.parquet` | Ghép các cặp thuộc tính dịch độc lập | 13.200 | 13.187 | 87.056 |
| `tiki_full_attributes_context_vi_zh.parquet` | Dịch có ngữ cảnh, giữ nguyên description cấp sản phẩm đã cung cấp | 13.200 | 13.187 | 87.112 |

Title của cả hai file là cùng bản chọn Gemma/Qwen bằng COMETKiwi: chọn điểm cao hơn giữa các ứng viên không lỗi sinh, không dùng ngưỡng 0,72, không loại title vì cờ QA. Giữ nguyên 13.200 ID sản phẩm và thứ tự dòng.

`description_vi`, `description_zh` ở cách độc lập được dựng từ các dòng `name: value`, phân tách bằng newline và sắp theo `pair_id`; không phải dịch lại văn xuôi. Cách ngữ cảnh giữ nguyên hai chuỗi description trong `description_products_vi_zh.parquet`. `attribute_pairs` giữ danh sách các cặp gốc, ID cặp, độ tin cậy và nguồn bản dịch nếu có; không trộn hai cách với nhau.

Cột chính: `product_id`, `title_vi`, `title_zh`, `description_vi`, `description_zh`, `attribute_count`, `attribute_pairs`, `has_description`, `description_method`. Các cột điểm và model title cùng metadata description được giữ để đối chiếu. Không bổ sung giá/shop hoặc các thuộc tính ngoài nội dung dịch.

**13 sản phẩm không xuất hiện trong cả hai bộ description** được giữ nguyên title, description null, `has_description=false`, và có `description_missing_reason`. Không tự bịa nội dung hoặc bỏ các sản phẩm này.

Nguồn description: `data/description_vi_to_zh/description_pairs_vi_zh.parquet` và `data/description_vi_to_zh_cum/{description_pairs_vi_zh_ctx.parquet,description_products_vi_zh.parquet}`. Các tài liệu nguồn giữ nguyên điều kiện sử dụng và provenance. Đây là dữ liệu dịch máy, chưa xác minh ngữ nghĩa thủ công cho từng dòng. `manifest.json` ghi checksum và số lượng kiểm chứng.
''')
text('data/translated/full_dataset/manifest.json',json.dumps({'exports':exports,'join_key':'product_id','join':'left_one_to_one','semantic_verified':False},indent=2))
text('data/translated/title_only/README.md','''# Title-only translation exports

Phạm vi tổ chức chỉ nằm trong thư mục này; dữ liệu nguồn chung của repo giữ cấu trúc cũ.

- `1688/gemma4/`, `1688/qwen3_14b_awq/`, `1688/best_cometkiwi/`: Trung → Việt, 16.348 dòng / 15.863 title.
- `tiki/gemma4/`, `tiki/qwen3_14b_awq/`, `tiki/best_cometkiwi/`: Việt → Trung, 13.200 dòng / 13.102 title.
- Bản kết hợp chọn điểm COMETKiwi cao hơn giữa các bản không lỗi sinh, không dùng ngưỡng loại; giữ đủ title. Model chấm `Unbabel/wmt22-cometkiwi-da`, revision `1ad785194e391eebc6c53e2d0776cada8f83179a`.
- 1688 chọn Gemma 9.563 / Qwen 6.300 title; Tiki chọn Gemma 6.240 / Qwen 6.862 title. Không có title đích được chọn bị trống.
- COMET là ước lượng chất lượng không tham chiếu, không phải xác minh ngữ nghĩa của người. Cờ kỹ thuật chỉ để tham khảo trong bản cuối.
- [Full datasets: title + description](../full_dataset/README.md).
''')
text('data/translated/title_only/manifest.json',json.dumps({'paths':list(moves.values()),'layout':'{1688,tiki}/{gemma4,qwen3_14b_awq,best_cometkiwi}/'},indent=2))
ops.extend(CommitOperationDelete(path_in_repo=f) for f in sorted(deletes))
(OUT/'restore_plan.json').write_text(json.dumps({'base':base,'head':head,'restored':restored,'export_moves':moves,'deleted_generated_paths':sorted(deletes)},indent=2))
c=api.create_commit(repo,repo_type='dataset',parent_commit=head,operations=ops,commit_message='Restore shared dataset layout; scope title folders and add two Tiki title-description full datasets')
def fp(x):return (x.size,x.lfs.sha256 if x.lfs else x.blob_id)
before={x.path:x for x in api.get_paths_info(repo,restored,repo_type='dataset',revision=base)} if restored else {}
after={x.path:x for x in api.get_paths_info(repo,restored,repo_type='dataset',revision=c.oid)} if restored else {}
assert all(fp(before[f])==fp(after[f]) for f in restored),'shared_restore_mismatch'
before={x.path:x for x in api.get_paths_info(repo,list(moves),repo_type='dataset',revision=head)} if moves else {}
after={x.path:x for x in api.get_paths_info(repo,list(moves.values()),repo_type='dataset',revision=c.oid)} if moves else {}
assert all(fp(before[a])==fp(after[b]) for a,b in moves.items()),'title_move_mismatch'
for entry in exports:
    p=hf_hub_download(repo,entry['path'],repo_type='dataset',revision=c.oid,token=os.environ['HF_TOKEN'])
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==entry['sha256']
result={'revision':c.oid,'shared_files_restored_verified':len(restored),'owned_title_files_moved_verified':len(moves),'exports':exports}
(OUT/'upload_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
