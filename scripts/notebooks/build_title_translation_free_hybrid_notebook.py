"""Generate the Colab notebook for the no-HF-inference-credit route."""

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
from pathlib import Path


def clean_multiline(source: str) -> str:
    # The source literals below are laid out as adjacent Python strings for readability.
    source = source.replace('\n"\n        "', '\n')
    return "\n".join(line for line in source.splitlines() if line.strip() != '"') + "\n"


def markdown(source: str) -> dict:
    source = clean_multiline(source)
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    source = clean_multiline(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Pipeline dịch title Cado — Gemma API + Qwen local\n\n"
        "Notebook này dùng Gemma 4-26B-A4B-it qua Google AI Studio API và tải Qwen3.5-9B về GPU Colab để chạy local. "
        "Không dùng HF Inference Providers, không cần HF inference token và không tự upload kết quả.\n\n"
        "**Gemma free tier có quota/RPM/RPD thay đổi theo project; 15.863 title có thể phải chạy nhiều ngày.** "
        "Google nêu dữ liệu free tier có thể được dùng để cải thiện sản phẩm; request chỉ gửi title Trung, "
        "không gửi product ID hay các trường khác. Xem điều khoản/quota trước khi chạy. Qwen tải model weights "
        "về phiên Colab; 8-bit cần GPU khoảng 14 GiB VRAM trở lên.\n\n"
        "Trên mỗi lượt, chọn một model và một stage. Dùng thứ tự smoke → pilot → validate → full; "
        "mỗi model chỉ cần tự qua validate trước khi chạy full. Mỗi lần full xử lý tối đa 500 title duy nhất và "
        "lưu SQLite/state trên Google Drive để resume. Kết quả cuối là hai Parquet riêng; sau đó tải cả "
        "thư mục run về máy và chạy lệnh upload riêng nếu muốn publish HF.\n\n"
        "**Bí mật:** tạo Colab Secret tên `GEMINI_API_KEY`, bật quyền truy cập cho notebook; không dán key vào cell/output. "
        "HF token không bắt buộc cho model Qwen public và không dùng để gọi inference.\n"""
    ),
    code(
        """# Kết nối Google Drive; checkpoint và hai Parquet sẽ nằm trong Drive.\n"
        "from google.colab import drive, files, userdata\n"
        "from pathlib import Path\n"
        "drive.mount('/content/drive')\n"
        "DRIVE_ROOT = Path('/content/drive/MyDrive')\n"
        "BUNDLE_PATH = DRIVE_ROOT / 'title_translation_free_hybrid_bundle.zip'\n"
        "if not BUNDLE_PATH.exists():\n"
        "    print('Chọn file title_translation_free_hybrid_bundle.zip vừa tải từ project.')\n"
        "    uploaded = files.upload()\n"
        "    if 'title_translation_free_hybrid_bundle.zip' not in uploaded:\n"
        "        raise FileNotFoundError('Cần upload đúng file title_translation_free_hybrid_bundle.zip')\n"
        "    BUNDLE_PATH.write_bytes(uploaded['title_translation_free_hybrid_bundle.zip'])\n"
        "print('Bundle:', BUNDLE_PATH)\n"""
    ),
    code(
        """# Cài dependencies; không cài lại PyTorch để giữ CUDA build sẵn có trong Colab.\n"
        "%pip -q install --upgrade 'transformers>=5.10.1,<6' 'accelerate>=1.2' 'bitsandbytes>=0.45' 'google-genai>=1.0' 'huggingface_hub>=0.30' 'sacrebleu' 'pyarrow>=18' 'pandas>=2.2,<3'\n"""
    ),
    code(
        """# Giải nén code + Parquet nguồn và kiểm tra SHA trước mọi request.\n"
        "import hashlib, json, sys, zipfile\n"
        "PROJECT_DIR = Path('/content/title_translation_free_hybrid_bundle')\n"
        "PROJECT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "with zipfile.ZipFile(BUNDLE_PATH) as zf:\n"
        "    zf.extractall(PROJECT_DIR)\n"
        "sys.path.insert(0, str(PROJECT_DIR))\n"
        "import title_translation_hybrid as pipeline\n"
        "SOURCE = PROJECT_DIR / 'bilingual_zh_vi_cado.parquet'\n"
        "WORKDIR = DRIVE_ROOT / 'title_translation_cado_free_hybrid_run_v2'\n"
        "actual_sha = pipeline.quality.sha256_file(SOURCE)\n"
        "print('Rows expected:', pipeline.EXPECTED_ROWS)\n"
        "print('Unique title expected:', pipeline.EXPECTED_UNIQUE_TITLES)\n"
        "print('Source SHA-256:', actual_sha)\n"
        "if actual_sha != pipeline.DATASET_SHA256:\n"
        "    raise ValueError('Nguồn không khớp revision/SHA đã khóa; dừng trước khi gọi model.')\n"
        "print('Source verified. Checkpoint folder:', WORKDIR)\n"""
    ),
    code(
        """# Tuỳ chọn: nhập checkpoint Gemma đã chạy trên máy local để Qwen tiếp tục cùng run.\n"
        "import zipfile\n"
        "CHECKPOINT_ARCHIVE = DRIVE_ROOT / 'title_translation_cado_free_hybrid_checkpoint.zip'\n"
        "if not WORKDIR.exists() and not CHECKPOINT_ARCHIVE.exists():\n"
        "    print('Nếu đã chạy Gemma local, upload title_translation_cado_free_hybrid_checkpoint.zip; nếu chưa thì bỏ qua.')\n"
        "    uploaded = files.upload()\n"
        "    if 'title_translation_cado_free_hybrid_checkpoint.zip' in uploaded:\n"
        "        CHECKPOINT_ARCHIVE.write_bytes(uploaded['title_translation_cado_free_hybrid_checkpoint.zip'])\n"
        "if not WORKDIR.exists() and CHECKPOINT_ARCHIVE.exists():\n"
        "    with zipfile.ZipFile(CHECKPOINT_ARCHIVE) as zf:\n"
        "        zf.extractall(DRIVE_ROOT)\n"
        "    print('Imported local Gemma checkpoint into:', WORKDIR)\n"
        "elif WORKDIR.exists():\n"
        "    print('Existing Drive checkpoint will be resumed:', WORKDIR)\n"
        "else:\n"
        "    print('No local checkpoint imported; this will start a new run.')\n"""
    ),
    markdown(
        """## Cấu hình từng lượt\n\n"
        "Chỉnh hai biến dưới đây rồi chạy cell kế tiếp. Không đổi prompt/model sau khi đã pilot; nếu đổi cần run folder mới và validate lại.\n\n"
        "Các lượt được tự resume từ SQLite. Nếu Gemma báo quota, dừng và chạy lại cùng stage sau khi quota được cấp lại; không tạo run folder mới. "
        "Qwen chạy trên `cuda:0`; với Colab T4 16 GB, notebook chỉ dùng 1 GPU và batch mặc định 1 để giữ ổn định.\n"""
    ),
    code(
        """# Chọn MODEL_KEY = 'gemma4' hoặc 'qwen35'; RUN_MODE = smoke/pilot/validate/full.\n"
        "MODEL_KEY = 'gemma4'\n"
        "RUN_MODE = 'smoke'\n"
        "FULL_TITLES_PER_SESSION = 500\n"
        "QWEN_BATCH_SIZE = 1\n"
        "GEMMA_MIN_INTERVAL_SECONDS = 6.0  # giảm burst; quota thực tế vẫn theo AI Studio\n"
        "\n"
        "model_key = MODEL_KEY.strip().lower()\n"
        "run_mode = RUN_MODE.strip().lower()\n"
        "if model_key not in {'gemma4', 'qwen35'}:\n"
        "    raise ValueError('MODEL_KEY phải là gemma4 hoặc qwen35')\n"
        "if run_mode not in {'smoke', 'pilot', 'validate', 'full'}:\n"
        "    raise ValueError('RUN_MODE phải là smoke, pilot, validate hoặc full')\n"
        "google_key = None\n"
        "if model_key == 'gemma4':\n"
        "    google_key = userdata.get('GEMINI_API_KEY')\n"
        "    if not google_key:\n"
        "        raise RuntimeError('Tạo Colab Secret GEMINI_API_KEY trước; không dán key trực tiếp vào notebook.')\n"
        "if model_key == 'qwen35':\n"
        "    import torch\n"
        "    if not torch.cuda.is_available():\n"
        "        raise RuntimeError('Qwen yêu cầu bật GPU trong Runtime → Change runtime type.')\n"
        "    props = torch.cuda.get_device_properties(0)\n"
        "    print(f'Qwen GPU: {props.name}; VRAM: {props.total_memory / 1024**3:.1f} GiB')\n"
        "    if props.total_memory < 14 * 1024**3:\n"
        "        raise RuntimeError('GPU dưới 14 GiB VRAM; dừng trước khi tải model. Chọn T4/A100.')\n"
        "\n"
        "ctx = pipeline.prepare_pipeline(\n"
        "    mode=run_mode, model_key=model_key, data_path=SOURCE, workdir=WORKDIR, hf_token=None\n"
        ")\n"
        "result = pipeline.run_stage(\n"
        "    ctx, google_api_key=google_key, batch_size=QWEN_BATCH_SIZE,\n"
        "    full_titles_per_session=FULL_TITLES_PER_SESSION,\n"
        "    gemma_min_interval_seconds=GEMMA_MIN_INTERVAL_SECONDS,\n"
        ")\n"
        "print(json.dumps({k: v for k, v in result.items() if k not in {'qa', 'cache'}}, ensure_ascii=False, indent=2, default=str))\n"""
    ),
    code(
        """# Kiểm tra tiến độ và đóng gói checkpoint để có thể tải về máy.\n"
        "import json, shutil\n"
        "state_path = WORKDIR / 'hybrid_run_state.json'\n"
        "if state_path.exists():\n"
        "    state = json.loads(state_path.read_text(encoding='utf-8'))\n"
        "    for key in ('gemma4', 'qwen35'):\n"
        "        stages = state.get('models', {}).get(key, {}).get('stages', {})\n"
        "        print(key, {name: info.get('status') for name, info in stages.items()})\n"
        "    ready = all(state.get('models', {}).get(key, {}).get('stages', {}).get('full', {}).get('status') == 'all_titles_complete' for key in ('gemma4', 'qwen35'))\n"
        "    if ready:\n"
        "        archive = shutil.make_archive(str(DRIVE_ROOT / 'title_translation_cado_free_hybrid_run_v2'), 'zip', root_dir=DRIVE_ROOT, base_dir=WORKDIR.name)\n"
        "        print('Both Parquets complete. Download this ZIP from Google Drive to the local project:', archive)\n"
        "    else:\n"
        "        print('Chưa hoàn tất cả hai model. Checkpoint nằm an toàn trên Drive; chỉnh stage/model rồi chạy lại cell cấu hình.')\n"
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "title_translation_free_hybrid.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path(__file__).with_name("title_translation_free_hybrid.ipynb")
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
