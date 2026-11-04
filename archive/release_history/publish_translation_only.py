
# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
root = Path(__file__).resolve().parent
os.chdir(root)
folder = root/'outputs/translation_only'
folder.mkdir(parents=True, exist_ok=True)
repo = 'ntquang0410/zh-vie_ecom'
token = os.environ['HF_TOKEN']
api = HfApi(token=token)
manifest = {'schema': 'translation_only_v1', 'exports': []}
operations = []
for name, source, filename in [
    ('gemma_1688', 'title_translation_cado_free_hybrid_run_v15/bilingual_zh_vi_cado_title_gemma4.parquet', 'bilingual_zh_vi_cado_title_gemma4.parquet'),
    ('qwen_tiki_qa_repaired', 'title_translation_tiki_qwen3_14b_awq_qa_repaired/qwen_tiki_qa_repaired.parquet', 'qwen_tiki_qa_repaired.parquet'),
]:
    d = pd.read_parquet(root/source)
    if name == 'gemma_1688':
        columns = ['product_id', 'title_zh', 'title_vi_candidate_gemma4'] + [c for c in d if c.startswith('title_vi_gemma4_')]
        out = d[columns].rename(columns={'title_vi_candidate_gemma4': 'title_vi'}).copy()
        assert out['title_vi'].equals(d['title_vi_candidate_gemma4'])
    else:
        columns = ['product_id', 'title_vi', 'title_zh', 'original_candidate', 'qa_repair_accepted', 'qa_status', 'qa_flags', 'generation_status', 'generation_truncated', 'semantic_verified']
        out = d[columns].copy()
        out['translation_model'] = 'Qwen/Qwen3-14B-AWQ'
        out['translation_model_revision'] = '31c69efc29464b6bb0aee1398b5a7b50a99340c3'
        out['candidate_stage'] = out['qa_repair_accepted'].map({True: 'qa_repair_accepted', False: 'original'})
        assert out['title_zh'].equals(d['selected_candidate'])
    assert len(out) == len(d) and out['product_id'].equals(d['product_id'])
    assert not any('description' in c or 'desc_' in c for c in out)
    path = folder/name/filename
    path.parent.mkdir(exist_ok=True)
    out.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(out, restored)
    remote = f'data/translated/completed/{name}/{filename}'
    manifest['exports'].append({'name': name, 'path': remote, 'rows': len(out), 'columns': list(out), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
for name, text in [('manifest.json', json.dumps(manifest, indent=2)), ('README.md', '# Title translation exports\n\nEach folder contains original titles, translated titles, product IDs for alignment, and translation metadata only. Descriptions and other product attributes are omitted.\n')]:
    path = folder/name
    path.write_text(text, encoding='utf-8')
    operations.append(CommitOperationAdd(path_in_repo=f'data/translated/completed/{name}', path_or_fileobj=str(path)))
commit = api.create_commit(repo, repo_type='dataset', operations=operations, commit_message='Keep original and translated titles with translation metadata only')
for entry in manifest['exports']:
    p = hf_hub_download(repo, entry['path'], repo_type='dataset', revision=commit.oid, token=token)
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == entry['sha256']
result = {'status': 'uploaded_verified', 'revision': commit.oid, 'exports': manifest['exports']}
(folder/'upload_result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
