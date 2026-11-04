"""Build an isolated Colab notebook/bundle for Qwen3-14B-AWQ title translation."""

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


ROOT = generated_root()
MODEL_KEY = "qwen3_14b_awq"
MODEL_ID = "Qwen/Qwen3-14B-AWQ"
NOTEBOOK_NAME = "title_translation_qwen3_14b_awq_colab_v5_auto_reload.ipynb"
BUNDLE_NAME = "title_translation_qwen3_14b_awq_colab_v5_auto_reload_bundle.zip"


def rewrite_pipeline(source: str) -> str:
    source = source.replace("qwen35", MODEL_KEY)
    source = source.replace("Qwen3.5-9B", "Qwen3-14B-AWQ")
    source = source.replace("Qwen3.5", "Qwen3")

    old_model = '''    "qwen3_14b_awq": {
        "model_id": "Qwen/Qwen3-14B-AWQ",
        "backend": "local_transformers_cuda",
        "provider": "local_cuda_8bit",
        "revision": None,
        "template_version": "qwen3_14b_awq-zh-vi-local-text-no-thinking-v1",
        "candidate_column": "title_vi_candidate_qwen3_14b_awq",
    },'''
    new_model = '''    "qwen3_14b_awq": {
        "model_id": "Qwen/Qwen3-14B-AWQ",
        "backend": "local_transformers_cuda_awq",
        "provider": "local_cuda_awq_4bit",
        "revision": None,
        "template_version": "qwen3-14b-awq-zh-vi-local-no-thinking-v1",
        "candidate_column": "title_vi_candidate_qwen3_14b_awq",
    },'''
    if old_model not in source:
        raise RuntimeError("Could not locate the Qwen model registry entry to specialize for AWQ.")
    source = source.replace(old_model, new_model, 1)

    source = source.replace('"qwen_quantization": "bitsandbytes-8bit"', '"qwen_quantization": "awq-4bit-from-model-config"')
    source = source.replace('"qwen_model_class": "Qwen3_5ForCausalLM"', '"qwen_model_class": "AutoModelForCausalLM"')
    source = source.replace(
        '["torch", "transformers", "accelerate", "bitsandbytes"]',
        '["torch", "transformers", "accelerate", "gptqmodel"]',
    )

    loader = '''def _load_qwen(revision: str, hf_token: str | None, device_index: int = 0) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-14B-AWQ needs a CUDA GPU. Select a GPU runtime before running this task.")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested cuda:{device_index}, but only {torch.cuda.device_count()} GPU is visible.")
    if torch.cuda.get_device_properties(device_index).total_memory < 14 * 1024**3:
        raise RuntimeError(
            "Qwen3-14B-AWQ 4-bit expects at least 14 GiB VRAM. A 16-GB T4 may be close to the limit; "
            "the run stops rather than silently spilling model weights to CPU."
        )

    model_id = MODELS["qwen3_14b_awq"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=hf_token)
    tokenizer.padding_side = "left"
    # The repository's config.json contains the official AWQ quantization config.
    # Let Transformers load that config; do not layer bitsandbytes quantization on top.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=hf_token,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device_index},
    )
    model.eval()
    return model, tokenizer


'''
    source, replacements = re.subn(
        r"def _load_qwen\(revision: str, hf_token: str \| None, device_index: int = 0\).*?\n\ndef _qwen_messages",
        loader + "def _qwen_messages",
        source,
        count=1,
        flags=re.S,
    )
    if replacements != 1:
        raise RuntimeError("Could not replace the Qwen loader with the Transformers AWQ loader.")
    if "Qwen3_5ForCausalLM" in source or "BitsAndBytesConfig" in source:
        raise RuntimeError("The AWQ pipeline still contains Qwen3.5/bitsandbytes loader code.")

    # The Drive file is a small allowlisted control plane, never executable code.
    source = source.replace(
        '    model, tokenizer = _load_qwen(context["lock"]["models"][model_key]["revision"], hf_token)\n',
        '    control_callback = context.get("control_callback")\n'
        '    if control_callback:\n'
        '        initial_progress = {"model_key": model_key, "phase": "loading_model", "processed_this_session": 0, "total_this_session": len(pending), "completed_this_session": 0, "errors_this_session": 0, "completed_before_session": int(context.get("qwen_progress_base_completed", 0))}\n'
        '        initial_action = control_callback(initial_progress)\n'
        '        while initial_action == "pause":\n'
        '            print("[qwen-control] paused before model load; set qwen_control.json action to run or stop", flush=True)\n'
        '            time.sleep(max(1.0, float(context.get("control_poll_seconds", 5))))\n'
        '            initial_action = control_callback(initial_progress)\n'
        '        if initial_action in {"stop", "reload"}:\n'
        '            initial_status = "reload_requested" if initial_action == "reload" else "stopped_by_control"\n'
        '            return {"model_key": model_key, "requested": len(titles), "completed": sum(_complete(value) for value in cached.values()), "failed": sum(not _complete(value) for value in cached.values()), "new_completed": 0, "input_tokens_this_session": 0, "output_tokens_this_session": 0, "provider": MODELS[model_key]["provider"], "batch_size_used": max(1, int(batch_size)), "elapsed_seconds": 0.0, "status": initial_status}\n'
        '    heartbeat_stop = None\n'
        '    heartbeat_thread = None\n'
        '    model_load_started = time.monotonic()\n'
        '    if control_callback:\n'
        '        from threading import Event, Thread\n'
        '        heartbeat_stop = Event()\n'
        '        def _report_model_load_heartbeat():\n'
        '            while not heartbeat_stop.wait(30.0):\n'
        '                loading_progress = dict(initial_progress)\n'
        '                loading_progress["elapsed_seconds"] = max(0.0, time.monotonic() - model_load_started)\n'
        '                control_callback(loading_progress)\n'
        '        heartbeat_thread = Thread(target=_report_model_load_heartbeat, daemon=True)\n'
        '        heartbeat_thread.start()\n'
        '    try:\n'
        '        model, tokenizer = _load_qwen(context["lock"]["models"][model_key]["revision"], hf_token)\n'
        '    except Exception as exc:\n'
        '        if control_callback:\n'
        '            control_callback({"model_key": model_key, "phase": "model_load_error", "processed_this_session": 0, "total_this_session": len(pending), "completed_this_session": 0, "errors_this_session": 0, "completed_before_session": int(context.get("qwen_progress_base_completed", 0)), "last_error": type(exc).__name__})\n'
        '        raise\n'
        '    finally:\n'
        '        if heartbeat_stop is not None:\n'
        '            heartbeat_stop.set()\n'
        '        if heartbeat_thread is not None:\n'
        '            heartbeat_thread.join(timeout=2.0)\n'
        '    if control_callback:\n'
        '        initial_progress = {"model_key": model_key, "phase": "model_ready", "processed_this_session": 0, "total_this_session": len(pending), "completed_this_session": 0, "errors_this_session": 0, "completed_before_session": int(context.get("qwen_progress_base_completed", 0)), "elapsed_seconds": max(0.0, time.monotonic() - model_load_started)}\n'
        '        initial_action = control_callback(initial_progress)\n'
        '        while initial_action == "pause":\n'
        '            time.sleep(max(1.0, float(context.get("control_poll_seconds", 5))))\n'
        '            initial_action = control_callback(initial_progress)\n'
        '        if initial_action in {"stop", "reload"}:\n'
        '            initial_status = "reload_requested" if initial_action == "reload" else "stopped_by_control"\n'
        '            return {"model_key": model_key, "requested": len(titles), "completed": sum(_complete(value) for value in cached.values()), "failed": sum(not _complete(value) for value in cached.values()), "new_completed": 0, "input_tokens_this_session": 0, "output_tokens_this_session": 0, "provider": MODELS[model_key]["provider"], "batch_size_used": max(1, int(batch_size)), "elapsed_seconds": max(0.0, time.monotonic() - model_load_started), "status": initial_status}\n',
        1,
    )
    source = source.replace(
        '    checkpoint_callback = context.get("cache_checkpoint_callback")\n',
        '    checkpoint_callback = context.get("cache_checkpoint_callback")\n'
        '    control_callback = context.get("control_callback")\n'
        '    stopped_by_control = False\n'
        '    reload_requested = False\n',
        1,
    )
    source = source.replace(
        '    cursor = 0\n'
        '    last_checkpoint = 0\n',
        '    cursor = 0\n'
        '    recent_errors = []\n'
        '    last_checkpoint = 0\n',
        1,
    )
    source = source.replace(
        '                    cache.put(title, record)\n'
        '                    input_tokens += int(record.get("input_tokens") or 0)\n',
        '                    cache.put(title, record)\n'
        '                    if incomplete:\n'
        '                        recent_errors.append({"reason": record.get("error_text"), "truncated": bool(record.get("truncated"))})\n'
        '                        recent_errors = recent_errors[-3:]\n'
        '                    input_tokens += int(record.get("input_tokens") or 0)\n',
        1,
    )
    source = source.replace(
        '                    last_checkpoint = cursor\n'
        '            except Exception as exc:\n',
        '                    last_checkpoint = cursor\n'
        '                if control_callback:\n'
        '                    elapsed_now = max(0.0, time.perf_counter() - started)\n'
        '                    progress = {"model_key": model_key, "phase": "generating", "processed_this_session": cursor, "total_this_session": len(pending), "completed_this_session": completed, "errors_this_session": failures, "completed_before_session": int(context.get("qwen_progress_base_completed", 0)), "elapsed_seconds": elapsed_now, "throughput_titles_per_hour": cursor / elapsed_now * 3600 if elapsed_now else 0.0, "recent_errors": recent_errors[-3:]}\n'
        '                    action = control_callback(progress)\n'
        '                    if action == "pause":\n'
        '                        if checkpoint_callback:\n'
        '                            checkpoint_callback(cache.path)\n'
        '                        print("[qwen-control] paused at a batch boundary; waiting for run/stop", flush=True)\n'
        '                        while action == "pause":\n'
        '                            time.sleep(max(1.0, float(context.get("control_poll_seconds", 5))))\n'
        '                            action = control_callback(progress)\n'
        '                    if action in {"stop", "reload"}:\n'
        '                        if checkpoint_callback:\n'
        '                            checkpoint_callback(cache.path)\n'
        '                        stopped_by_control = action == "stop"\n'
        '                        reload_requested = action == "reload"\n'
        '                        print(f"[qwen-control] {action} requested; cache saved after the last completed batch", flush=True)\n'
        '                        break\n'
        '            except Exception as exc:\n'
        '                if control_callback:\n'
        '                    control_callback({"model_key": model_key, "phase": "generation_error", "processed_this_session": cursor, "total_this_session": len(pending), "completed_this_session": completed, "errors_this_session": failures, "completed_before_session": int(context.get("qwen_progress_base_completed", 0)), "last_error": type(exc).__name__})\n',
        1,
    )
    source = source.replace(
        '        "status": "finished_current_batch",\n'
        '    }\n'
        '\n'
        '\ndef _evaluate_model_titles(',
        '        "status": "stopped_by_control" if stopped_by_control else ("reload_requested" if reload_requested else "finished_current_batch"),\n'
        '    }\n'
        '\n'
        '\ndef _evaluate_model_titles(',
        1,
    )
    source = source.replace(
        '    else:\n'
        '        session_titles = stage_titles\n\n'
        '    if model_key == "gemma4":',
        '    else:\n'
        '        session_titles = stage_titles\n'
        '    context["qwen_progress_base_completed"] = len(all_titles) - len(session_titles) if mode == "full" else 0\n\n'
        '    if model_key == "gemma4":',
        1,
    )
    source = source.replace(
        '    if generation.get("status") == "quota_exhausted_checkpoint_saved":\n',
        '    if generation.get("status") in {"stopped_by_control", "reload_requested"}:\n'
        '        if mode == "full":\n'
        '            current_cache = cache.get_many(all_titles)\n'
        '            result["completed_unique_titles"] = sum(_complete(current_cache.get(title)) for title in all_titles)\n'
        '            result["total_unique_titles"] = len(all_titles)\n'
        '            result["remaining_unique_titles"] = len(all_titles) - result["completed_unique_titles"]\n'
        '        control_status = generation.get("status")\n'
        '        result.update({"passed": False, "status": control_status, "checkpoint_preserved": True})\n'
        '        _write_stage_report(context, result)\n'
        '        context["state"]["models"][model_key]["stages"][mode] = result\n'
        '        atomic_json(context["workdir"] / "hybrid_run_state.json", context["state"])\n'
        '        return result\n'
        '    if generation.get("status") == "quota_exhausted_checkpoint_saved":\n',
        1,
    )
    if '"reload_requested" if reload_requested' not in source:
        raise RuntimeError("Could not add Drive control/checkpoint support to the AWQ pipeline.")
    return source


def rewrite_notebook(notebook: dict) -> dict:
    notebook = json.loads(json.dumps(notebook))
    for cell_index, cell in enumerate(notebook["cells"]):
        text = "".join(cell.get("source", []))
        text = text.replace("qwen35", MODEL_KEY)
        text = text.replace("title_translation_cado_qwen3_14b_awq_full_v1", "title_translation_cado_qwen3_14b_awq_full_v2")
        text = text.replace("title_translation_qwen3_14b_awq_colab_bundle.zip", BUNDLE_NAME)
        text = text.replace(
            f"BUNDLE_PATH = DRIVE_ROOT / '{BUNDLE_NAME}'",
            f"BUNDLE_PATH = WORKDIR / '{BUNDLE_NAME}'",
        )
        text = text.replace(
            "import title_translation_hybrid as pipeline",
            "sys.modules.pop('title_translation_hybrid', None)\n"
            "sys.modules.pop('title_translation_pipeline', None)\n"
            "import title_translation_hybrid as pipeline\n"
            "print('Đã nạp lại mã pipeline từ bundle Drive:', BUNDLE_PATH)",
        )
        text = text.replace("/content/title_translation_qwen3_14b_awq_source", "/content/title_translation_qwen3_14b_awq_v5_auto_reload_source")
        text = text.replace("qwen3_14b_awq_colab_package_lock.json", "qwen3_14b_awq_colab_v2_package_lock.json")
        text = text.replace("Qwen3.5-9B", "Qwen3-14B-AWQ (4-bit AWQ)")
        text = text.replace("Qwen3.5", "Qwen3-14B-AWQ")
        text = text.replace("Qwen 8-bit", "Qwen3-14B AWQ 4-bit")
        text = text.replace("8-bit run", "4-bit AWQ run")
        text = text.replace("8-bit", "4-bit AWQ")
        text = text.replace(
            "Runtime bắt đầu với batch 4 và tự giảm xuống 2/1 nếu CUDA OOM.",
            "Runtime bắt đầu với batch 1; không tăng batch trước khi đo tốc độ và VRAM trên runtime thật.",
        )
        text = text.replace(
            "MIN_KERNEL_RESOLVER_TRANSFORMERS",
            "MIN_SUPPORTED_TRANSFORMERS",
        )
        text = text.replace(
            "để sửa resolver FLA nested-kernel.",
            "để dùng phiên bản Transformers đã khóa cho Qwen3 AWQ.",
        )
        text = text.replace("bitsandbytes', 'huggingface_hub'", "huggingface_hub'")
        text = text.replace("'bitsandbytes>=0.45',", "")
        text = text.replace(
            "QWEN_BATCH_SIZE = 4  # tự giảm 4 -> 2 -> 1 nếu CUDA OOM",
            "QWEN_BATCH_SIZE = 1  # AWQ 14B trên T4: bắt đầu batch 1; tăng chỉ sau benchmark riêng",
        )
        text = text.replace("T4 16 GB trở lên được khuyến nghị; pipeline cần tối thiểu 14 GiB VRAM.",
                            "T4 16 GB có thể chạy AWQ 4-bit nhưng khá sát giới hạn; bắt đầu batch 1, không bảo đảm mọi runtime T4 đều đủ VRAM.")
        text = text.replace(
            "- Notebook chạy preflight 20 title để tải model/khóa môi trường, sau đó chạy full ngay. Kết quả QA không chặn giữa chừng; khi full hoàn tất, mỗi dòng Parquet có cờ QA kỹ thuật để rà soát hậu kỳ.",
            "- Notebook chạy preflight 20 title. Nếu còn title chưa hoàn tất hoặc đầu ra lặp/cắt cụt, notebook dừng trước full và vẫn giữ checkpoint; các cờ QA kỹ thuật còn lại có thể được rà soát hậu kỳ theo lựa chọn của bạn.",
        )
        text = text.replace(
            "- Notebook chạy preflight 20 title để tải model/khóa môi trường, sau đó chạy full ngay. Kết quả QA không chặn giữa chừng; khi full hoàn tất, mỗi dòng Parquet có cờ QA kỹ thuật để rà soát hậu kỳ.",
            "- Notebook chạy preflight 20 title. Nếu còn title chưa hoàn tất hoặc đầu ra lặp/cắt cụt, notebook dừng trước full và vẫn giữ checkpoint; các cờ QA kỹ thuật còn lại có thể được rà soát hậu kỳ theo lựa chọn của bạn.",
        )
        text = text.replace("# Cài đúng các thư viện của Qwen; không nâng cấp PyTorch/CUDA có sẵn trong Colab.",
                            "# Cài thư viện Transformers cho AWQ; không nâng cấp PyTorch/CUDA có sẵn trong Colab.")
        text = text.replace(
            "PACKAGE_NAMES = ['transformers', 'accelerate', 'huggingface_hub', 'sacrebleu', 'pyarrow', 'pandas']",
            "PACKAGE_NAMES = ['transformers', 'accelerate', 'gptqmodel', 'huggingface_hub', 'sacrebleu', 'pyarrow', 'pandas']",
        )
        text = text.replace(
            "'accelerate>=1.2', \n    'huggingface_hub>=0.30'",
            "'accelerate>=1.2', 'gptqmodel',\n    'huggingface_hub>=0.30'",
        )
        if "PACKAGE_SPECS = [" in text and "'gptqmodel'" not in text:
            raise RuntimeError("Could not add gptqmodel to the notebook's pinned package install cell.")
        text = text.replace("import importlib.metadata\n", "import importlib.metadata\nimport importlib.util\n")
        text = text.replace(
            "versions['torch'] = importlib.metadata.version('torch')",
            "versions['torch'] = Version(importlib.metadata.version('torch')).base_version  # ignore +cpu/+cuXXX build tag",
        )
        text = text.replace(
            "locked_packages = json.loads(PACKAGE_LOCK.read_text(encoding='utf-8'))\n    locked_transformers =",
            "locked_packages = json.loads(PACKAGE_LOCK.read_text(encoding='utf-8'))\n"
            "    if locked_packages.get('torch'):\n"
            "        locked_packages['torch'] = Version(locked_packages['torch']).base_version\n"
            "    locked_transformers =",
        )
        text = text.replace(
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *exact_specs])",
            "torch_constraint = Path('/tmp/qwen3_14b_awq_torch_constraint.txt')\n"
            "        torch_constraint.write_text(f\"torch=={importlib.metadata.version('torch')}\\n\", encoding='utf-8')\n"
            "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(torch_constraint), *exact_specs])",
        )
        text = text.replace(
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGE_SPECS])",
            "torch_constraint = Path('/tmp/qwen3_14b_awq_torch_constraint.txt')\n"
            "    torch_constraint.write_text(f\"torch=={importlib.metadata.version('torch')}\\n\", encoding='utf-8')\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(torch_constraint), *PACKAGE_SPECS])",
        )
        text = text.replace(
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(torch_constraint), *exact_specs])",
            "_pip_install([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--constraint', str(torch_constraint), *exact_specs])",
        )
        text = text.replace(
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(torch_constraint), *PACKAGE_SPECS])",
            "_pip_install([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--constraint', str(torch_constraint), *PACKAGE_SPECS])",
        )
        if "def _pip_install(command):" not in text:
            text = text.replace(
                "if PACKAGE_LOCK.exists():\n",
                "def _pip_install(command):\n"
                "    from collections import deque\n"
                "    tail = deque(maxlen=300)\n"
                "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)\n"
                "    for line in process.stdout:\n"
                "        print(line, end='', flush=True)\n"
                "        tail.append(line)\n"
                "    return_code = process.wait()\n"
                "    log_text = ''.join(tail)\n"
                "    redactor = globals().get('_qwen_redact')\n"
                "    if callable(redactor):\n"
                "        log_text = redactor(log_text)\n"
                "    (WORKDIR / 'qwen_install_last.log').write_text(log_text, encoding='utf-8')\n"
                "    if return_code:\n"
                "        raise RuntimeError(f'pip install exited with code {return_code}; last output follows:\\n{log_text[-10000:]}')\n\n"
                "if PACKAGE_LOCK.exists():\n",
                1,
            )
        text = text.replace(
            "print('Môi trường khóa trên Drive:', json.dumps(locked_packages, indent=2))",
            "tmp_lock = PACKAGE_LOCK.with_suffix('.json.normalized.tmp')\n"
            "tmp_lock.write_text(json.dumps(locked_packages, indent=2), encoding='utf-8')\n"
            "tmp_lock.replace(PACKAGE_LOCK)\n"
            "from transformers.utils.import_utils import is_gptqmodel_available\n"
            "if not is_gptqmodel_available():\n"
            "    raise RuntimeError('gptqmodel chưa được cài/nhận diện; xem lỗi cài package ở cell này rồi chạy lại.')\n"
            "print('gptqmodel:', importlib.metadata.version('gptqmodel'))\n"
            "print('Môi trường khóa trên Drive:', json.dumps(locked_packages, indent=2))",
        )
        text = text.replace("mode='smoke', model_key='qwen3_14b_awq'", "mode='smoke', model_key='qwen3_14b_awq'")
        if "# Preflight 20 title, sau đó chạy full ngay cả khi QA preflight báo cờ." in text:
            text = text.replace(
                "QWEN_BATCH_SIZE = 1  # AWQ 14B trên T4: bắt đầu batch 1; tăng chỉ sau benchmark riêng",
                "QWEN_BATCH_SIZE = 1  # Giữ batch đã khóa khi smoke/checkpoint được tạo\n"
                "QWEN_FULL_BATCH_SIZE = 4  # Thử tối đa; pipeline tự hạ 4 -> 2 -> 1 nếu OOM lúc generate",
            )
            text = text.replace(
                "full_context,\n        batch_size=QWEN_BATCH_SIZE,\n        full_titles_per_session=",
                "full_context,\n        batch_size=QWEN_FULL_BATCH_SIZE,\n"
                "        allow_qwen_batch_change=True,\n"
                "        full_titles_per_session=",
            )
            text = text.replace(
                "persist_qwen_cache(LOCAL_CACHE)\n"
                "print('Preflight:',",
                "persist_qwen_cache(LOCAL_CACHE)\n"
                "import gc\n"
                "import torch\n"
                "gc.collect()\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.synchronize()\n"
                "    torch.cuda.empty_cache()\n"
                "    free_bytes, total_bytes = torch.cuda.mem_get_info()\n"
                "    print(f'VRAM trống sau smoke: {free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB')\n"
                "print('Preflight:',",
                1,
            )
            text = text.replace(
                "if smoke_result.get('status') == 'quality_gate_failed':\n"
                "    print('QA preflight có cờ; theo lựa chọn của bạn, nó không chặn full. QA toàn bộ sẽ được xuất sau khi dịch xong.')\n",
                "smoke_completion = float(smoke_result.get('completion_rate', 0) or 0)\n"
                "if smoke_completion < 0.99:\n"
                "    raise RuntimeError(\n"
                "        f'Preflight chỉ hoàn tất {smoke_completion:.1%}; đã lưu checkpoint lên Drive và dừng trước full. '\n"
                "        'Không chạy full với đầu ra đang lỗi/lặp. Hãy dùng bundle đã sửa.'\n"
                "    )\n"
                "if smoke_result.get('status') == 'quality_gate_failed':\n"
                "    print('Preflight đủ đầu ra nhưng còn cờ QA kỹ thuật; theo lựa chọn của bạn, sẽ lưu cờ để rà soát sau full.')\n",
            )
            if "QWEN_FULL_BATCH_SIZE = 4" not in text or "allow_qwen_batch_change=True" not in text:
                raise RuntimeError("Could not configure the full stage for OOM-fallback batch-4 speed mode.")
            if "smoke_completion < 0.99" not in text:
                raise RuntimeError("Could not add the preflight completion safety gate before the full run.")
        if cell_index == 4:
            text = text.replace("import hashlib\nimport json\nimport sys\nimport zipfile\n",
                                "import hashlib\nimport importlib\nimport io\nimport json\nimport sys\nimport zipfile\n")
            loader_pattern = (
                r"PROJECT_DIR = Path\([^\n]+\)\n"
                r"PROJECT_DIR\.mkdir\(parents=True, exist_ok=True\)\n"
                r"with zipfile\.ZipFile\(BUNDLE_PATH\) as zf:\n.*?"
                r"print\('Đã nạp lại mã pipeline từ bundle Drive:', BUNDLE_PATH\)\n"
            )
            loader_block = '''PROJECT_DIR = Path('/content/title_translation_qwen3_14b_awq_v5_auto_reload_source')
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
MAX_BUNDLE_BYTES = 10 * 1024 * 1024
EXPECTED_BUNDLE_FILES = {'title_translation_hybrid.py', 'title_translation_pipeline.py'}

def load_pipeline_bundle(expected_sha256=None):
    bundle_bytes = BUNDLE_PATH.read_bytes()
    if len(bundle_bytes) > MAX_BUNDLE_BYTES:
        raise RuntimeError(f'Bundle lớn bất thường: {len(bundle_bytes)} bytes.')
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    if expected_sha256 and bundle_sha256.lower() != str(expected_sha256).lower():
        raise RuntimeError(f'SHA-256 bundle không khớp lệnh reload: {bundle_sha256}')
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = set(zf.namelist())
        if names != EXPECTED_BUNDLE_FILES:
            raise RuntimeError(f'Bundle có tên tệp không được phép: {sorted(names)}')
        for name in EXPECTED_BUNDLE_FILES:
            info = zf.getinfo(name)
            if info.file_size > 2 * 1024 * 1024:
                raise RuntimeError(f'Module trong bundle lớn bất thường: {name}')
            (PROJECT_DIR / name).write_bytes(zf.read(name))
    sys.path.insert(0, str(PROJECT_DIR)) if str(PROJECT_DIR) not in sys.path else None
    importlib.invalidate_caches()
    sys.modules.pop('title_translation_hybrid', None)
    sys.modules.pop('title_translation_pipeline', None)
    module = importlib.import_module('title_translation_hybrid')
    return module, bundle_sha256

pipeline, ACTIVE_BUNDLE_SHA = load_pipeline_bundle()
print('Đã nạp pipeline từ bundle Drive; SHA-256:', ACTIVE_BUNDLE_SHA)
'''
            text, loader_count = re.subn(loader_pattern, loader_block, text, count=1, flags=re.S)
            if loader_count != 1:
                raise RuntimeError('Could not replace the v4 bundle loader with the guarded v5 loader.')
        if cell_index == 3:
            for old_call, new_call in [
                ("subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(torch_constraint), *exact_specs])",
                 "_pip_install([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--constraint', str(torch_constraint), *exact_specs])"),
                ("subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(torch_constraint), *PACKAGE_SPECS])",
                 "_pip_install([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--constraint', str(torch_constraint), *PACKAGE_SPECS])"),
            ]:
                text = text.replace(old_call, new_call)
            install_helper = '''def _pip_install(command):
    from collections import deque
    tail = deque(maxlen=300)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
        tail.append(line)
    return_code = process.wait()
    log_text = ''.join(tail)
    redactor = globals().get('_qwen_redact')
    if callable(redactor):
        log_text = redactor(log_text)
    (WORKDIR / 'qwen_install_last.log').write_text(log_text, encoding='utf-8')
    if return_code:
        raise RuntimeError(f"pip install exited with code {return_code}; last output follows:\\n{log_text[-10000:]}")

'''
            if "def _pip_install(command):" not in text:
                text = text.replace("if PACKAGE_LOCK.exists():\n", install_helper + "if PACKAGE_LOCK.exists():\n", 1)
        cell["source"] = text.splitlines(keepends=True)

    # Add Drive telemetry and an allowlisted run/pause/stop controller.
    run_cell = notebook["cells"][7]
    run_text = "".join(run_cell.get("source", []))
    run_text = "import time\nfrom datetime import datetime, timezone\n" + run_text
    run_text = run_text.replace(
        "# Preflight 20 title, sau đó chạy full ngay cả khi QA preflight báo cờ.\n",
        "# Preflight 20 title, sau đó chạy full ngay cả khi QA preflight báo cờ.\n"
        "_qwen_full_run_cell_marker = 'qwen-full-run-v5'\n",
        1,
    )
    attach_block = '''CHECKPOINT_EVERY_TITLES = 100
CONTROL_FILE = WORKDIR / 'qwen_control.json'
LIVE_STATUS_FILE = WORKDIR / 'qwen_live_status.json'
CONTROL_POLL_SECONDS = 5
if not CONTROL_FILE.exists():
    CONTROL_FILE.write_text(json.dumps({'action': 'run'}, indent=2), encoding='utf-8')

_control_last_status_write = 0.0
_control_last_action = None
_control_expected_bundle_sha256 = None
_SAFE_CONTROL_ERROR_REASONS = {'repeated_phrase_guard', 'truncated_output', 'empty_output'}
_SAFE_EXCEPTION_CLASSES = {'RuntimeError', 'ImportError', 'ModuleNotFoundError', 'CalledProcessError', 'FileNotFoundError', 'TimeoutError', 'ConnectionError', 'HTTPError', 'ValueError', 'TypeError', 'KeyError', 'AttributeError', 'MemoryError', 'OSError'}

def _qwen_safe_recent_errors(items):
    safe_items = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            safe_items.append({'reason': 'other_error'})
            continue
        reason = str(item.get('reason') or '')
        safe_items.append({
            'reason': reason if reason in _SAFE_CONTROL_ERROR_REASONS else 'other_error',
            'truncated': bool(item.get('truncated')),
        })
    return safe_items[-3:]

def _qwen_safe_error_class(value):
    candidate = str(value or '').split(':', 1)[0].strip()
    return candidate if candidate in _SAFE_EXCEPTION_CLASSES else ('unknown_error' if value else None)

def qwen_drive_control(progress):
    global _control_last_status_write, _control_last_action, _control_expected_bundle_sha256
    command_error = None
    try:
        command = json.loads(CONTROL_FILE.read_text(encoding='utf-8'))
        action = str(command.get('action', 'run')).strip().lower()
        _control_expected_bundle_sha256 = command.get('bundle_sha256')
    except Exception as exc:
        action = _control_last_action or 'run'
        command = {'action': action}
        command_error = f'{type(exc).__name__}: {exc}'
    if action not in {'run', 'pause', 'stop', 'reload'}:
        command_error = f"Unsupported action {action!r}; expected run, pause, stop, or reload."
        action = 'run'
    now = time.time()
    processed = int(progress.get('processed_this_session', 0))
    should_write = (
        action != _control_last_action
        or processed % 20 == 0
        or now - _control_last_status_write >= 30
        or bool(progress.get('last_error'))
    )
    if should_write:
        status = {
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_key': progress.get('model_key'),
            'phase': progress.get('phase', 'unknown'),
            'action': action,
            'processed_this_session': processed,
            'total_this_session': int(progress.get('total_this_session', 0)),
            'completed_this_session': int(progress.get('completed_this_session', 0)),
            'errors_this_session': int(progress.get('errors_this_session', 0)),
            'completed_before_session': int(progress.get('completed_before_session', 0)),
            'completed_estimate': int(progress.get('completed_before_session', 0)) + int(progress.get('completed_this_session', 0)),
            'elapsed_seconds': float(progress.get('elapsed_seconds', 0) or 0),
            'throughput_titles_per_hour': float(progress.get('throughput_titles_per_hour', 0) or 0),
            'recent_errors': _qwen_safe_recent_errors(progress.get('recent_errors', [])),
            'last_error_class': _qwen_safe_error_class(progress.get('last_error')),
            'command_error': bool(command_error),
            'checkpoint_file': DRIVE_CACHE.name,
            'runtime': 'Colab; control polled between generation batches',
            'active_bundle_sha256': ACTIVE_BUNDLE_SHA,
            'requested_bundle_sha256': _control_expected_bundle_sha256,
        }
        try:
            temp_status = LIVE_STATUS_FILE.with_suffix('.json.tmp')
            temp_status.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
            temp_status.replace(LIVE_STATUS_FILE)
            _control_last_status_write = now
        except Exception as exc:
            print(f'[qwen-control] could not mirror status to Drive: {type(exc).__name__}: {exc}', flush=True)
    _control_last_action = action
    return action

def attach_durable_cache(context, enable_control=False):
    context['cache_paths'] = {'qwen3_14b_awq': LOCAL_CACHE}
    context['cache_checkpoint_interval'] = CHECKPOINT_EVERY_TITLES
    context['cache_checkpoint_callback'] = persist_qwen_cache
    if enable_control:
        context['control_callback'] = qwen_drive_control
        context['control_poll_seconds'] = CONTROL_POLL_SECONDS
    return context

def _apply_drive_code_update(timeout_seconds=600):
    global pipeline, ACTIVE_BUNDLE_SHA
    old_sha = ACTIVE_BUNDLE_SHA
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            candidate_sha = hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
        except Exception as exc:
            candidate_sha = None
            print(f'[qwen-reload] cannot read Drive bundle: {type(exc).__name__}: {exc}', flush=True)
        expected_sha = _control_expected_bundle_sha256
        if candidate_sha and candidate_sha != old_sha and (not expected_sha or candidate_sha.lower() == str(expected_sha).lower()):
            new_pipeline, new_sha = load_pipeline_bundle(expected_sha256=expected_sha)
            pipeline = new_pipeline
            ACTIVE_BUNDLE_SHA = new_sha
            temp_control = CONTROL_FILE.with_suffix('.json.tmp')
            temp_control.write_text(json.dumps({'action': 'run', 'bundle_sha256': new_sha}, indent=2), encoding='utf-8')
            temp_control.replace(CONTROL_FILE)
            print(f'[qwen-reload] loaded reviewed bundle {new_sha}; resuming from SQLite checkpoint', flush=True)
            return True
        qwen_drive_control({'model_key': 'qwen3_14b_awq', 'phase': 'awaiting_code_update',
                            'processed_this_session': 0, 'total_this_session': 0,
                            'completed_this_session': 0, 'errors_this_session': 0,
                            'completed_before_session': 0})
        time.sleep(CONTROL_POLL_SECONDS)
    print('[qwen-reload] no new matching bundle arrived; checkpoint remains saved. Restart Run all after upload.', flush=True)
    return False
'''
    run_text, attach_count = re.subn(
        r"CHECKPOINT_EVERY_TITLES = 100\n\ndef attach_durable_cache\(context\):\n.*?    return context\n",
        attach_block,
        run_text,
        count=1,
        flags=re.S,
    )
    if attach_count != 1:
        raise RuntimeError("Could not insert the Drive control/status callback in the run cell.")
    run_text = run_text.replace(
        "smoke_result = pipeline.run_stage(\n"
        "    smoke_context, batch_size=QWEN_BATCH_SIZE, allow_qwen_hardware_change=True\n"
        ")\n"
        "persist_qwen_cache(LOCAL_CACHE)\n",
        "smoke_result = pipeline.run_stage(\n"
        "    smoke_context, batch_size=QWEN_BATCH_SIZE, allow_qwen_hardware_change=True\n"
        ")\n"
        "_smoke_reload_count = 0\n"
        "while smoke_result.get('status') == 'reload_requested' and _smoke_reload_count < 5:\n"
        "    _smoke_reload_count += 1\n"
        "    if not _apply_drive_code_update():\n"
        "        break\n"
        "    smoke_context = pipeline.prepare_pipeline(mode='smoke', model_key='qwen3_14b_awq', data_path=SOURCE, workdir=WORKDIR, hf_token=None, qwen_batch_size=QWEN_BATCH_SIZE)\n"
        "    attach_durable_cache(smoke_context, enable_control=True)\n"
        "    smoke_result = pipeline.run_stage(smoke_context, batch_size=QWEN_BATCH_SIZE, allow_qwen_hardware_change=True)\n"
        "persist_qwen_cache(LOCAL_CACHE)\n",
        1,
    )
    run_text = run_text.replace(
        "    if full_result.get('status') != 'checkpoint_saved_continue':\n"
        "        break\n",
        "    if full_result.get('status') == 'reload_requested':\n"
        "        if _apply_drive_code_update():\n"
        "            continue\n"
        "        break\n"
        "    if full_result.get('status') != 'checkpoint_saved_continue':\n"
        "        break\n",
        1,
    )
    run_text = run_text.replace(
        "attach_durable_cache(smoke_context)\n",
        "attach_durable_cache(smoke_context, enable_control=True)\n",
    )
    run_text = run_text.replace(
        "attach_durable_cache(full_context)\n",
        "attach_durable_cache(full_context, enable_control=True)\n",
    )
    if "enable_control=True" not in run_text or "def qwen_drive_control(progress):" not in run_text:
        raise RuntimeError("Drive control/status support did not land in the full-run cell.")
    run_cell["source"] = run_text.splitlines(keepends=True)

    notebook["cells"][6]["source"].extend(
        "\n\n### Theo dõi và can thiệp qua Drive\n\n"
        "Trong thư mục run, `qwen_live_status.json` ghi tiến độ/lỗi gần nhất; `qwen_control.json` nhận `run`, `pause`, `stop`, `reload`. Sau khi upload bundle mới, đặt lệnh `reload` kèm SHA-256; ở ranh giới batch, notebook lưu checkpoint, kiểm tra hash, nạp mã mới rồi tiếp tục. Lệnh chỉ là JSON allowlist, không thực thi Python.\n"
        .splitlines(keepends=True)
    )
    setup_cell = notebook["cells"][2]
    setup_cell["source"].extend(
        "\n\n# Ghi lỗi của mọi cell notebook lên Drive để có thể rà soát mà không cần chụp/paste traceback.\n"
        "import re\n"
        "import traceback\n"
        "from datetime import datetime, timezone\n"
        "from IPython import get_ipython\n"
        "_qwen_status_path = WORKDIR / 'qwen_live_status.json'\n"
        "def _qwen_redact(value):\n"
        "    value = str(value)\n"
        "    value = re.sub(r'hf_[A-Za-z0-9]{16,}', '[REDACTED_HF_TOKEN]', value)\n"
        "    value = re.sub(r'AIza[0-9A-Za-z_-]{20,}', '[REDACTED_GOOGLE_KEY]', value)\n"
        "    value = re.sub(r'AQ\\.[A-Za-z0-9_-]{20,}', '[REDACTED_API_KEY]', value)\n"
        "    return value\n"
        "def _qwen_drive_error_hook(result):\n"
        "    error = getattr(result, 'error_before_exec', None) or getattr(result, 'error_in_exec', None)\n"
        "    if error is None:\n"
        "        raw_cell = getattr(getattr(result, 'info', None), 'raw_cell', '') or ''\n"
        "        if '_qwen_full_run_cell_marker' not in raw_cell:\n"
        "            return\n"
        "        summary = globals().get('full_result')\n"
        "        if not isinstance(summary, dict):\n"
        "            return\n"
        "        try:\n"
        "            payload = {\n"
        "                'state': summary.get('status', 'unknown'),\n"
        "                'phase': 'run_cell_finished',\n"
        "                'updated_at_utc': datetime.now(timezone.utc).isoformat(),\n"
        "                'completed_unique_titles': summary.get('completed_unique_titles'),\n"
        "                'total_unique_titles': summary.get('total_unique_titles'),\n"
        "                'remaining_unique_titles': summary.get('remaining_unique_titles'),\n"
        "                'generation': summary.get('generation'),\n"
        "            }\n"
        "            tmp = _qwen_status_path.with_suffix('.json.tmp')\n"
        "            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')\n"
        "            tmp.replace(_qwen_status_path)\n"
        "        except Exception as log_error:\n"
        "            print(f'[qwen-drive-observer] could not write final status: {type(log_error).__name__}: {log_error}', flush=True)\n"
        "        return\n"
        "    try:\n"
        "        payload = {\n"
        "            'state': 'error',\n"
        "            'phase': 'notebook_cell_error',\n"
        "            'updated_at_utc': datetime.now(timezone.utc).isoformat(),\n"
        "            'cell_execution_count': getattr(result, 'execution_count', None),\n"
        "            'last_error_class': type(error).__name__,\n"
        "        }\n"
        "        tmp = _qwen_status_path.with_suffix('.json.tmp')\n"
        "        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')\n"
        "        tmp.replace(_qwen_status_path)\n"
        "    except Exception as log_error:\n"
        "        print(f'[qwen-drive-observer] could not write cell error: {type(log_error).__name__}: {log_error}', flush=True)\n"
        "_qwen_ip = get_ipython()\n"
        "if _qwen_ip is not None:\n"
        "    _old_hook = getattr(_qwen_ip, '_qwen_drive_error_hook', None)\n"
        "    if _old_hook is not None:\n"
        "        try: _qwen_ip.events.unregister('post_run_cell', _old_hook)\n"
        "        except Exception: pass\n"
        "    _qwen_ip.events.register('post_run_cell', _qwen_drive_error_hook)\n"
        "    _qwen_ip._qwen_drive_error_hook = _qwen_drive_error_hook\n"
        .splitlines(keepends=True)
    )
    setup_text = "".join(notebook["cells"][1].get("source", []))
    setup_text = setup_text.replace(
        f"4. Cell đầu sẽ yêu cầu upload `{BUNDLE_NAME}` một lần. Hãy tải bundle đi kèm notebook lên.",
        f"4. Bundle `{BUNDLE_NAME}` nằm trong thư mục run trên Drive; notebook chỉ hỏi upload nếu file bị thiếu.",
    )
    notebook["cells"][1]["source"] = setup_text.splitlines(keepends=True)

    first_markdown = notebook["cells"][0]
    first_text = "".join(first_markdown["source"])
    first_text = first_text.replace(
        "Notebook này chỉ chạy **Qwen3-14B-AWQ (4-bit AWQ) local trên GPU Colab**;",
        f"Notebook này chỉ chạy **`{MODEL_ID}` (4-bit AWQ) local trên GPU Colab**;",
    )
    first_markdown["source"] = first_text.splitlines(keepends=True)
    notebook["metadata"].setdefault("colab", {})["name"] = NOTEBOOK_NAME
    return notebook


def main() -> None:
    source_path = source_file('title_translation_hybrid.py')
    quality_path = source_file('title_translation_pipeline.py')
    notebook_path = source_file('title_translation_qwen35_colab.ipynb')
    if not all(path.exists() for path in (source_path, quality_path, notebook_path)):
        raise FileNotFoundError("Expected current hybrid source, QA source, and base Colab notebook beside this builder.")

    pipeline_source = rewrite_pipeline(source_path.read_text(encoding="utf-8"))
    compile(pipeline_source, "title_translation_hybrid.py", "exec")
    notebook = rewrite_notebook(json.loads(notebook_path.read_text(encoding="utf-8")))

    notebook_output = ROOT / NOTEBOOK_NAME
    notebook_output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle_output = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(bundle_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("title_translation_hybrid.py", pipeline_source)
        archive.write(quality_path, arcname="title_translation_pipeline.py")

    guide = f'''# Qwen3-14B-AWQ title translation on Google Colab

This is a separate run configuration for `{MODEL_ID}`. It uses the repository's official AWQ 4-bit configuration through Transformers `AutoModelForCausalLM`; it does not use bitsandbytes, Qwen3.5-specific FLA kernels, Gemma, or an API.

## Before switching

- The v5 notebook reuses the existing Qwen3-14B-AWQ folder, cache, run state, source Parquet, and model lock. It does not overwrite the SQLite checkpoint.
- If another full cell is currently running, let it reach a checkpoint or stop it cleanly before starting v5; do not run two workers against the same SQLite file at once.

## Run

1. Put `{NOTEBOOK_NAME}` in Google Drive and open it with Colab. Put `{BUNDLE_NAME}` in `MyDrive/title_translation_cado_qwen3_14b_awq_full_v2`; the notebook reads it there, so it is not uploaded on every session. The control file is created in the same folder on first run.
2. Select a CUDA GPU runtime and run all cells. It installs/checks `gptqmodel`, verifies the pinned source Parquet and resumes the existing cache. Smoke must have at least 99% complete outputs before full.
3. The package lock compares the PyTorch core version while ignoring the `+cpu`/`+cuXXX` build suffix, and pins the currently installed Torch build during dependency installation. This lets an old CPU-tagged lock resume on a CUDA Colab runtime without replacing its GPU-enabled Torch.
4. Smoke keeps the locked batch size 1. Full starts at batch size 4 and automatically backs off to 2 then 1 if generation runs out of VRAM. A T4 16 GB is close to the model's memory limit; after a model-loading OOM, restart the Colab runtime before resuming so the smoke model is fully cleared. Do not change quantization or the checkpoint's locked smoke configuration.
5. Checkpoints are mirrored every 100 processed titles, and immediately on `pause`/`stop`/`reload`. `qwen_live_status.json` records progress/rate, recent repeated/truncated outputs, and errors from any notebook cell (traceback is recorded with common token patterns redacted). The `qwen_control.json` allowlist is `run`, `pause`, `stop`, or `reload`; `pause` waits and `stop` saves and exits.
6. The notebook exports a separate Parquet locally on Drive and does not upload to Hugging Face.

## Observe/edit and automatic safe reload

- The run writes progress, throughput, recent generation errors, and cell tracebacks to Drive. I can inspect those files and upload a reviewed replacement bundle through Drive.
- To apply a reviewed fix while the run is active, upload/replace the bundle first, then set `qwen_control.json` to `{{"action":"reload","bundle_sha256":"<new ZIP SHA-256>"}}`. At the next completed batch the notebook saves its checkpoint, verifies the requested hash, reloads the module, and resumes. It never patches a live generation call; model weights must reload, which costs time.
- The loader accepts only the two expected module filenames and checks bundle size, file names, and SHA-256. The control JSON is an allowlisted command file; it never executes code.
- `run`, `pause`, `stop`, and `reload` are polled between generation batches. If Colab has already terminated or the cell exited, Drive cannot restart its runtime; reconnect and run all to resume. This is Drive-mediated control, not SSH/direct kernel access.
- This AWQ configuration requires CUDA. Running Qwen3-14B-AWQ on CPU is not a practical fallback. For direct agent-triggered notebook runs, connect the Deepnote integration; API access and GPU machine availability depend on its plan and billing.

The 4-bit AWQ model is larger than the previous 9B model and is **not a speed upgrade**. Expected throughput must be measured in the 20-title preflight on the actual Colab GPU before deciding whether to run all 15,863 unique titles. `semantic_verified` remains false until human review.
'''
    guide_path = ROOT / "TITLE_TRANSLATION_QWEN3_14B_AWQ_COLAB_V5_AUTO_RELOAD_GUIDE.md"
    guide_path.write_text(guide, encoding="utf-8")

    print(notebook_output)
    print(bundle_output)
    print(guide_path)


if __name__ == "__main__":
    main()
