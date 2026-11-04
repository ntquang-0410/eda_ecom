"""Publish verified completed exports with explicit QA limitations."""

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
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

def main():
    try:
        from dotenv import dotenv_values
        cfg = dotenv_values(r'D:\download\NCKH\ecom_crawler-main\ecom_crawler-main-feature-1688\.env')
    except (ImportError, OSError):
        cfg = {}
    token = os.environ.get('HF_TOKEN') or cfg.get('HF_TOKEN')
    api = HfApi(token=token)
    repo = 'ntquang0410/zh-vie_ecom'
    api.repo_info(repo, repo_type='dataset')
    exports = [
        ('gemma_1688', ROOT/'title_translation_cado_free_hybrid_run_v15/bilingual_zh_vi_cado_title_gemma4.parquet', '72614a000e0b2c4d0deb627331a408dfbaf2ab679e2af5abeb7d5e69e6f667f9', 16348, 15863, 746),
        ('qwen_tiki_qa_repaired', ROOT/'title_translation_tiki_qwen3_14b_awq_qa_repaired/qwen_tiki_qa_repaired.parquet', 'cd03a92eb667cf73a5393e1a7cbf2fd2d3a7f8d3a7b4aa3ac1c6198f9c1888b2', 13200, 13102, 2483),
    ]
    manifest = {'semantic_verified': False, 'exports': []}
    operations = []
    for name, path, expected, rows, unique, flags in exports:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'local_hash_mismatch'
        remote = f'data/translated/completed/{name}/{path.name}'
        manifest['exports'].append(dict(name=name, path=remote, sha256=expected, rows=rows, unique_titles=unique, technical_qa_flagged_unique=flags, semantic_verified=False))
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
    folder = ROOT/'outputs/hf_completed_upload'
    folder.mkdir(parents=True, exist_ok=True)
    note = folder/'README.md'
    note.write_text('# Completed translation runs\n\nCompleted processing does not mean fully clean translations. Gemma 1688 retains 746 flagged unique titles. Qwen Tiki after bounded QA repair retains 2483 flagged unique titles (370 repairs accepted). QA flags and original candidates are preserved in the exports. Neither export has human semantic verification. Qwen 1688 and COMETKiwi selection are still pending and are not included.\n', encoding='utf-8')
    metadata = folder/'manifest.json'
    metadata.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for path in (note, metadata):
        operations.append(CommitOperationAdd(path_in_repo=f'data/translated/completed/{path.name}', path_or_fileobj=str(path)))
    commit = api.create_commit(repo, repo_type='dataset', operations=operations, commit_message='Add completed Gemma 1688 and QA-repaired Qwen Tiki exports with retained review flags')
    for item in manifest['exports']:
        downloaded = hf_hub_download(repo, item['path'], repo_type='dataset', revision=commit.oid, token=token)
        assert hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() == item['sha256'], 'remote_hash_mismatch'
    result = dict(status='uploaded_verified', repo=repo, revision=commit.oid, commit_url=commit.commit_url, exports=manifest['exports'])
    (folder/'upload_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status='upload_failed', error_class=type(exc).__name__, http_status=getattr(getattr(exc, 'response', None), 'status_code', None))))
        raise SystemExit(1)
