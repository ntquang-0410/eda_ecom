"""Build an isolated Qwen3-14B-AWQ Colab run with non-greedy decoding.

The v6 run keeps the existing v2 Drive checkpoint untouched. It imports only
completed, non-repetitive rows into a new cache and retries the rest with the
official Qwen3 non-thinking sampling recommendations.
"""

from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
from project_bootstrap import source_file, generated_root

import json
import re
import zipfile
from pathlib import Path

import build_title_translation_qwen3_14b_awq_colab as base


ROOT = generated_root()
NOTEBOOK_NAME = "title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload.ipynb"
BUNDLE_NAME = "title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload_bundle.zip"
RUN_DIR_V2 = "title_translation_cado_qwen3_14b_awq_full_v2"
RUN_DIR_V6 = "title_translation_cado_qwen3_14b_awq_full_v3_sampled"
DECODING_POLICY = "qwen3-14b-awq-nonthinking-sampling-v1"


def rewrite_pipeline(source: str) -> str:
    source = base.rewrite_pipeline(source)

    old_kwargs = 'kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1}'
    new_kwargs = (
        'kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": True, '
        '"temperature": 0.7, "top_p": 0.8, "top_k": 20, '
        '"repetition_penalty": 1.05, "num_beams": 1}'
    )
    if old_kwargs not in source:
        raise RuntimeError("Could not locate the AWQ greedy-decoding settings to replace.")
    source = source.replace(old_kwargs, new_kwargs, 1)

    old_lock_setting = '"greedy_decoding": True,'
    new_lock_settings = (
        '"greedy_decoding": False,\n'
        f'            "qwen_decode_policy": "{DECODING_POLICY}",\n'
        '            "qwen_temperature": 0.7,\n'
        '            "qwen_top_p": 0.8,\n'
        '            "qwen_top_k": 20,\n'
        '            "qwen_repetition_penalty": 1.05,'
    )
    if old_lock_setting not in source:
        raise RuntimeError("Could not locate the decoding policy in the environment-lock payload.")
    source = source.replace(old_lock_setting, new_lock_settings, 1)

    old_model_version = '"model_version": f"transformers-local/{_package_version(\'transformers\')}",'
    new_model_version = (
        '"model_version": f"transformers-local/{_package_version(\'transformers\')}; '
        f'{DECODING_POLICY}",'
    )
    if old_model_version not in source:
        raise RuntimeError("Could not add the decoding policy to per-row provenance.")
    source = source.replace(old_model_version, new_model_version, 1)

    compile(source, "title_translation_hybrid.py", "exec")
    return source


def _checkpoint_migration_block() -> str:
    return f'''def _migrate_legacy_qwen_cache(context):
    """Copy only reusable completed rows; never mutate the v2 source checkpoint."""
    import sqlite3
    from datetime import datetime, timezone

    marker = WORKDIR / 'legacy_cache_import.json'
    if marker.exists():
        print('Đã hoàn tất nhập cache v2 trước đó; giữ nguyên cache v3.')
        return

    legacy_workdir = DRIVE_ROOT / '{RUN_DIR_V2}'
    legacy_cache = legacy_workdir / 'cache_qwen3_14b_awq.sqlite3'
    legacy_lock_path = legacy_workdir / 'hybrid_environment_lock.json'
    legacy_status_path = legacy_workdir / 'qwen_live_status.json'
    if legacy_status_path.exists():
        try:
            legacy_status = json.loads(legacy_status_path.read_text(encoding='utf-8'))
            last_state = str(legacy_status.get('state', '')).lower()
            updated = datetime.fromisoformat(str(legacy_status.get('updated_at_utc', '')).replace('Z', '+00:00'))
            is_recent = (datetime.now(timezone.utc) - updated).total_seconds() < 1200
            if is_recent and last_state in {{'running', 'generating', 'loading_model', 'awaiting_code_update'}}:
                raise RuntimeError('Run v2 có trạng thái hoạt động gần đây; hãy dừng hẳn run cũ trước khi nhập cache.')
        except ValueError:
            pass

    if not legacy_cache.exists() or not legacy_lock_path.exists():
        print('Không tìm thấy đủ cache/lock v2; v3 sẽ tiếp tục bằng cache riêng mới.')
        marker.write_text(json.dumps({{'status': 'no_legacy_cache'}}, indent=2), encoding='utf-8')
        return

    legacy_lock = json.loads(legacy_lock_path.read_text(encoding='utf-8'))
    model_key = 'qwen3_14b_awq'
    old_model = legacy_lock.get('models', {{}}).get(model_key, {{}})
    old_revision = old_model.get('revision')
    if old_model.get('model_id') != pipeline.MODELS[model_key]['model_id']:
        raise RuntimeError('Cache v2 thuộc model khác; dừng để tránh trộn kết quả.')

    # Pin v3 to the immutable revision used by the old cache before computing
    # the new config ID. This allows safe reuse while recording the new decoder.
    if old_revision:
        new_lock = context['lock']
        new_models = dict(new_lock['models'])
        new_model = dict(new_models[model_key])
        new_model['revision'] = old_revision
        new_models[model_key] = new_model
        new_config = pipeline._config_payload(
            new_models,
            qwen_batch_size=int(context.get('qwen_batch_size', 1)),
        )
        new_config_id = pipeline.fingerprint(new_config)
        new_lock.update({{'models': new_models, 'config': new_config, 'config_id': new_config_id}})
        context['lock'] = new_lock
        context['state']['config_id'] = new_config_id
        pipeline.atomic_json(WORKDIR / 'hybrid_environment_lock.json', new_lock)
        pipeline.atomic_json(WORKDIR / 'hybrid_run_state.json', context['state'])

    revision = context['lock']['models'][model_key]['revision']
    target_cache = pipeline.Cache(LOCAL_CACHE, model_key, revision, context['lock']['config_id'])
    imported = 0
    skipped = 0
    skipped_smoke = 0
    smoke_titles = set(context.get('samples', {{}}).get('smoke', []))
    with sqlite3.connect(f'file:{{legacy_cache.as_posix()}}?mode=ro', uri=True) as old_db:
        rows = old_db.execute(
            "SELECT source_title,output_text,status,truncated,input_tokens,output_tokens,attempts,"
            "api_requests,elapsed_seconds,error_text,model_version,response_id "
            "FROM translations"
        ).fetchall()
    for row in rows:
        title, output, status, truncated, in_tokens, out_tokens, attempts, api_requests, elapsed, error, version, response_id = row
        if str(title) in smoke_titles:
            # Force the smoke set through the new sampling decoder so the
            # preflight actually tests v6 instead of passing on legacy cache.
            skipped_smoke += 1
            continue
        record = {{
            'output_text': output,
            'status': status,
            'truncated': bool(truncated),
            'input_tokens': in_tokens,
            'output_tokens': out_tokens,
            'attempts': attempts or 0,
            'api_requests': api_requests or attempts or 0,
            'elapsed_seconds': elapsed or 0.0,
            'error_text': error,
            'model_version': (version or 'legacy-qwen3-14b-awq') + '; legacy-greedy',
            'response_id': response_id,
            'generation_prompt_version': 'legacy-qwen3-14b-awq-greedy-v1',
        }}
        if not pipeline._complete(record):
            skipped += 1
            continue
        target_cache.put(str(title), record)
        imported += 1

    persist_qwen_cache(LOCAL_CACHE)
    payload = {{
        'status': 'complete',
        'source_run_dir': '{RUN_DIR_V2}',
        'source_cache_rows': len(rows),
        'imported_completed_nonrepetitive': imported,
        'skipped_incomplete_or_repetitive': skipped,
        'skipped_smoke_for_fresh_sampling_test': skipped_smoke,
        'source_revision': old_revision,
        'target_config_id': context['lock']['config_id'],
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }}
    temp_marker = marker.with_suffix('.json.tmp')
    temp_marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp_marker.replace(marker)
    print('Nhập cache hoàn tất:', json.dumps(payload, ensure_ascii=False))

'''


def rewrite_notebook(notebook: dict) -> dict:
    base.NOTEBOOK_NAME = NOTEBOOK_NAME
    base.BUNDLE_NAME = BUNDLE_NAME
    notebook = base.rewrite_notebook(notebook)

    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace("title_translation_qwen3_14b_awq_colab_v5_auto_reload", "title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload")
        text = text.replace("title_translation_qwen3_14b_awq_v5_auto_reload_source", "title_translation_qwen3_14b_awq_v6_sampled_source")
        text = text.replace(RUN_DIR_V2, RUN_DIR_V6)
        text = text.replace("qwen-full-run-v5", "qwen-full-run-v6")
        text = text.replace("v5 notebook", "v6 notebook")
        text = text.replace("v5 run", "v6 run")
        if "WORKDIR = DRIVE_ROOT /" in text:
            text = text.replace("WORKDIR = DRIVE_ROOT / '" + RUN_DIR_V6 + "'", "WORKDIR = DRIVE_ROOT / '" + RUN_DIR_V6 + "'")
        cell["source"] = text.splitlines(keepends=True)

    # Keep the Colab runtime's preinstalled NumPy ABI stable during pip
    # resolution. Unconstrained AWQ dependency installs can otherwise upgrade
    # NumPy and break optional Transformers imports (the observed numpy.ufunc
    # __module__ error). Record the preserved version in the environment lock.
    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        if "PACKAGE_NAMES = [" not in text:
            continue
        old_names = "PACKAGE_NAMES = ['transformers', 'accelerate', 'gptqmodel', 'huggingface_hub', 'sacrebleu', 'pyarrow', 'pandas']"
        new_names = "PACKAGE_NAMES = ['transformers', 'accelerate', 'gptqmodel', 'huggingface_hub', 'sacrebleu', 'pyarrow', 'pandas', 'numpy']"
        if old_names not in text:
            raise RuntimeError("Could not add NumPy to the Colab runtime package lock.")
        text = text.replace(old_names, new_names, 1)
        old_torch = "versions['torch'] = Version(importlib.metadata.version('torch')).base_version  # ignore +cpu/+cuXXX build tag"
        if old_torch not in text:
            raise RuntimeError("Could not preserve the existing Torch runtime constraint.")
        text = text.replace(
            old_torch,
            old_torch + "\n    versions['numpy'] = importlib.metadata.version('numpy')  # preserve Colab ABI",
            1,
        )
        old_constraint = "torch_constraint.write_text(f\"torch=={importlib.metadata.version('torch')}\\n\", encoding='utf-8')"
        new_constraint = (
            "torch_constraint.write_text("
            "f\"torch=={importlib.metadata.version('torch')}\\n\""
            "f\"numpy=={importlib.metadata.version('numpy')}\\n\", encoding='utf-8')"
        )
        if old_constraint not in text:
            raise RuntimeError("Could not pin the existing NumPy version during dependency installation.")
        text = text.replace(old_constraint, new_constraint)
        cell["source"] = text.splitlines(keepends=True)

    run_cell = notebook["cells"][7]
    run_text = "".join(run_cell.get("source", []))
    hook = "attach_durable_cache(smoke_context, enable_control=True)\nsmoke_result = pipeline.run_stage("
    if hook not in run_text:
        raise RuntimeError("Could not locate the smoke-stage cache setup for the v2-to-v3 migration.")
    run_text = run_text.replace(
        "smoke_context = pipeline.prepare_pipeline(",
        _checkpoint_migration_block() + "smoke_context = pipeline.prepare_pipeline(",
        1,
    )
    run_text = run_text.replace(
        hook,
        "attach_durable_cache(smoke_context, enable_control=True)\n"
        "_migrate_legacy_qwen_cache(smoke_context)\n"
        "smoke_result = pipeline.run_stage(",
        1,
    )
    run_cell["source"] = run_text.splitlines(keepends=True)

    first = notebook["cells"][0]
    first_text = "".join(first.get("source", []))
    first_text = first_text.replace(
        "Notebook này chỉ chạy **`Qwen/Qwen3-14B-AWQ` (4-bit AWQ) local trên GPU Colab**;",
        "Notebook này chạy **`Qwen/Qwen3-14B-AWQ` (4-bit AWQ) trên GPU Colab** với sampling non-thinking chống lặp;",
    )
    first_text += (
        "\n\nBản v6 tạo run folder riêng. Cache v2 không bị sửa; chỉ các bản hoàn chỉnh, "
        "không bị lặp mới được nhập, còn lỗi/lặp sẽ chạy lại theo decoder mới.\n"
    )
    first["source"] = first_text.splitlines(keepends=True)
    notebook["metadata"].setdefault("colab", {})["name"] = NOTEBOOK_NAME
    return notebook


def build() -> tuple[Path, Path, Path]:
    source_path = source_file('title_translation_hybrid.py')
    quality_path = source_file('title_translation_pipeline.py')
    notebook_path = source_file('title_translation_qwen35_colab.ipynb')
    if not all(path.exists() for path in (source_path, quality_path, notebook_path)):
        raise FileNotFoundError("Expected hybrid source, QA source, and base Colab notebook beside this builder.")

    pipeline_source = rewrite_pipeline(source_path.read_text(encoding="utf-8"))
    notebook = rewrite_notebook(json.loads(notebook_path.read_text(encoding="utf-8")))

    notebook_output = ROOT / NOTEBOOK_NAME
    notebook_output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle_output = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(bundle_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("title_translation_hybrid.py", pipeline_source)
        archive.write(quality_path, arcname="title_translation_pipeline.py")

    guide = f'''# Qwen3-14B-AWQ v6: sampled decoding with Drive checkpoint resume

The v6 notebook keeps the existing `{RUN_DIR_V2}` directory untouched and writes to a new `{RUN_DIR_V6}` directory. It uses the same pinned Qwen3-14B-AWQ weights and dataset, with the model-card non-thinking sampling recommendations: temperature 0.7, top-p 0.8, top-k 20. A mild repetition penalty (1.05) is added to reduce loops. Greedy decoding was removed because Qwen's official model guidance warns it may produce endless repetition.

## Resume without overwriting the old checkpoint

- On first run, v6 reads the v2 cache in read-only mode and pins the v2 model revision.
- It imports only completed, non-truncated, non-repetitive outputs into its new cache. Incomplete or repetitive rows are intentionally retried.
- v2 files are not modified. The new cache import is recorded in `legacy_cache_import.json` and can be repeated safely if interrupted.
- Do not run v2 and v6 at the same time against the same source cache.

## Run and observe

1. Open `{NOTEBOOK_NAME}` in Colab, choose a CUDA GPU, and select **Runtime → Run all**. Authorize the Drive mount.
2. Keep the notebook, bundle, `qwen_control.json`, checkpoint and status in `MyDrive/{RUN_DIR_V6}`. The notebook loads the bundle from this Drive folder at startup, so it does not need to be uploaded into Colab's temporary storage each session.
3. The run writes `qwen_live_status.json` and checkpoints in the v6 folder. Status contains progress, throughput, error categories and exception class only; it never writes source titles, generated translations, or raw tracebacks. A heartbeat updates status while the model weights load.
4. Drive control accepts `run`, `pause`, `stop`, and `reload`. To apply a reviewed fix while Colab is running, upload/replace the bundle first, then set `qwen_control.json` to `{{"action":"reload","bundle_sha256":"<SHA-256 of uploaded bundle>"}}`. At a batch boundary, Colab saves its checkpoint, verifies the hash, reloads the approved modules, and resumes. The hash-bound control file is not an arbitrary code executor.
5. Drive cannot start or keep a Colab VM alive. If Colab disconnects or terminates, reconnect and select **Runtime → Run all** to resume. The background Codex monitor can inspect the Drive status and, after tests pass, publish a reviewed bundle and reload command only while the runtime remains alive.
6. The model still requires CUDA. The local laptop's 4-GB GPU cannot host this 14B AWQ model. This version does not call a paid inference API and does not upload output to Hugging Face.

The change addresses the observed repeated-phrase failures at the decoding layer. It still requires the smoke gate before full; technical QA does not verify semantic accuracy, and `semantic_verified` remains false.
'''
    guide_path = ROOT / "TITLE_TRANSLATION_QWEN3_14B_AWQ_COLAB_V6_SAMPLED_AUTORELOAD_GUIDE.md"
    guide_path.write_text(guide, encoding="utf-8")
    return notebook_output, bundle_output, guide_path


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
