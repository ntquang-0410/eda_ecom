"""Build a Qwen-only Google Colab notebook and its current source-code bundle."""

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
import copy
import zipfile
from pathlib import Path


ROOT = generated_root()


def markdown(source: str) -> dict:
    source = source.strip() + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    source = source.strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Dịch full title Cado bằng Qwen3.5-9B trên Google Colab

Notebook này chỉ chạy **Qwen3.5-9B local trên GPU Colab**; không chạy Gemma, không gọi API dịch, không cần HF token và không tự upload kết quả lên Hugging Face.

- Nguồn được khóa theo revision và SHA-256 trong pipeline: 16.348 dòng, khoảng 15.863 title Trung duy nhất.
- Checkpoint Qwen riêng nằm trong `MyDrive/title_translation_cado_qwen35_full_v1`; không dùng chung checkpoint Gemma.
- Runtime bắt đầu với batch 4 và tự giảm xuống 2/1 nếu CUDA OOM. Cache SQLite được sao lưu lên Drive mỗi 100 title đã xử lý.
- Notebook chạy preflight 20 title để tải model/khóa môi trường, sau đó chạy full ngay. Kết quả QA không chặn giữa chừng; khi full hoàn tất, mỗi dòng Parquet có cờ QA kỹ thuật để rà soát hậu kỳ.
- Nếu Colab ngắt, kết nối lại cùng loại runtime và chọn **Runtime → Run all**. Các bản dịch đã checkpoint sẽ được bỏ qua; model weights có thể cần tải lại.

`semantic_verified` luôn là `False`: QA tự động không thay thế kiểm tra ngữ nghĩa của con người. Notebook không tự publish lên HF.
"""
    ),
    markdown(
        """## Trước khi chạy

1. Mở notebook trong Google Colab.
2. Chọn **Runtime → Change runtime type → GPU**. T4 16 GB trở lên được khuyến nghị; pipeline cần tối thiểu 14 GiB VRAM.
3. Chọn **Runtime → Run all** và cấp quyền mount Google Drive khi Colab hỏi.
4. Cell đầu sẽ yêu cầu upload `title_translation_qwen35_colab_bundle.zip` một lần. Hãy tải bundle đi kèm notebook lên.

Không dán API key hay HF token vào notebook. Dataset và model đều được tải công khai theo revision đã khóa.
"""
    ),
    code(
        """from google.colab import drive, files
from pathlib import Path
import shutil

drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive')
WORKDIR = DRIVE_ROOT / 'title_translation_cado_qwen35_full_v1'
WORKDIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR = Path('/content/title_translation_cado_qwen35_full_v1')
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE = LOCAL_DIR / 'cache_qwen35.sqlite3'
DRIVE_CACHE = WORKDIR / 'cache_qwen35.sqlite3'

if DRIVE_CACHE.exists() and not LOCAL_CACHE.exists():
    shutil.copy2(DRIVE_CACHE, LOCAL_CACHE)
    print('Đã khôi phục SQLite checkpoint từ Drive:', DRIVE_CACHE)
else:
    print('Checkpoint SQLite local:', 'đã có' if LOCAL_CACHE.exists() else 'chưa có; run mới hoặc sẽ khôi phục khi có bản Drive')
print('Run folder riêng của Qwen:', WORKDIR)
"""
    ),
    code(
        """# Cài đúng các thư viện của Qwen; không nâng cấp PyTorch/CUDA có sẵn trong Colab.
import importlib
import importlib.metadata
import json
import subprocess
import sys
from packaging.version import Version

PACKAGE_LOCK = WORKDIR / 'qwen35_colab_package_lock.json'
FAST_TRANSFORMERS_VERSION = '5.20.2'
MIN_KERNEL_RESOLVER_TRANSFORMERS = Version('5.17.0')
PACKAGE_NAMES = ['transformers', 'accelerate', 'bitsandbytes', 'huggingface_hub', 'sacrebleu', 'pyarrow', 'pandas']
PACKAGE_SPECS = [
    f'transformers=={FAST_TRANSFORMERS_VERSION}', 'accelerate>=1.2', 'bitsandbytes>=0.45',
    'huggingface_hub>=0.30', 'sacrebleu', 'pyarrow>=18', 'pandas>=2.2,<3',
]

def installed_package_versions():
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # A persisted lock can outlive a Colab runtime reset; represent absent
            # packages as mismatches so the locked versions are installed below.
            versions[name] = None
    versions['torch'] = importlib.metadata.version('torch')
    return versions

if PACKAGE_LOCK.exists():
    locked_packages = json.loads(PACKAGE_LOCK.read_text(encoding='utf-8'))
    locked_transformers = Version(locked_packages['transformers'])
    if locked_transformers < MIN_KERNEL_RESOLVER_TRANSFORMERS:
        print(
            f'Lock Transformers cũ ({locked_transformers}); nâng lên {FAST_TRANSFORMERS_VERSION} '
            'để sửa resolver FLA nested-kernel.'
        )
        locked_packages['transformers'] = FAST_TRANSFORMERS_VERSION
    current_packages = installed_package_versions()
    if current_packages != locked_packages:
        exact_specs = [f'{name}=={locked_packages[name]}' for name in PACKAGE_NAMES]
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *exact_specs])
        importlib.invalidate_caches()
        current_packages = installed_package_versions()
        if current_packages != locked_packages:
            raise RuntimeError(f'Không khôi phục được môi trường đã khóa. Mong đợi {locked_packages}; hiện có {current_packages}.')
        tmp_lock = PACKAGE_LOCK.with_suffix('.json.tmp')
        tmp_lock.write_text(json.dumps(locked_packages, indent=2), encoding='utf-8')
        tmp_lock.replace(PACKAGE_LOCK)
else:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *PACKAGE_SPECS])
    importlib.invalidate_caches()
    locked_packages = installed_package_versions()
    tmp_lock = PACKAGE_LOCK.with_suffix('.json.tmp')
    tmp_lock.write_text(json.dumps(locked_packages, indent=2), encoding='utf-8')
    tmp_lock.replace(PACKAGE_LOCK)

print('Môi trường khóa trên Drive:', json.dumps(locked_packages, indent=2))
"""
    ),
    code(
        """# Nạp mã pipeline hiện tại từ bundle và tải đúng Parquet nguồn công khai.
import hashlib
import json
import sys
import zipfile
from huggingface_hub import hf_hub_download

BUNDLE_PATH = DRIVE_ROOT / 'title_translation_qwen35_colab_bundle.zip'
if not BUNDLE_PATH.exists():
    print('Upload file title_translation_qwen35_colab_bundle.zip')
    uploaded = files.upload()
    if 'title_translation_qwen35_colab_bundle.zip' not in uploaded:
        raise FileNotFoundError('Cần upload title_translation_qwen35_colab_bundle.zip')
    BUNDLE_PATH.write_bytes(uploaded['title_translation_qwen35_colab_bundle.zip'])

PROJECT_DIR = Path('/content/title_translation_qwen35_source')
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(BUNDLE_PATH) as zf:
    expected_files = {'title_translation_hybrid.py', 'title_translation_pipeline.py'}
    if set(zf.namelist()) != expected_files:
        raise RuntimeError(f'Bundle không đúng nội dung mong đợi: {zf.namelist()}')
    for name in expected_files:
        (PROJECT_DIR / name).write_bytes(zf.read(name))
sys.path.insert(0, str(PROJECT_DIR))
import title_translation_hybrid as pipeline

SOURCE = WORKDIR / Path(pipeline.DATASET_FILE).name
if not SOURCE.exists():
    downloaded = hf_hub_download(
        repo_id=pipeline.DATASET_ID,
        filename=pipeline.DATASET_FILE,
        repo_type='dataset',
        revision=pipeline.DATASET_REVISION,
    )
    shutil.copy2(downloaded, SOURCE)
actual_sha = pipeline.quality.sha256_file(SOURCE)
if actual_sha != pipeline.DATASET_SHA256:
    raise ValueError(f'SHA-256 nguồn không khớp: {actual_sha}; dừng trước khi dịch.')
print(f'Nguồn đã xác minh: {pipeline.EXPECTED_ROWS:,} dòng; {pipeline.EXPECTED_UNIQUE_TITLES:,} title duy nhất; SHA-256={actual_sha}')
"""
    ),
    code(
        """# GPU check và hàm mirror SQLite từ SSD runtime sang Drive.
import json
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError('Chưa bật GPU. Chọn Runtime → Change runtime type → GPU rồi chạy lại.')
gpu = torch.cuda.get_device_properties(0)
gpu_vram_gib = gpu.total_memory / (1024 ** 3)
print(f'GPU: {gpu.name}; VRAM: {gpu_vram_gib:.1f} GiB')
if gpu.total_memory < 14 * 1024 ** 3:
    raise RuntimeError('Qwen 8-bit cần tối thiểu 14 GiB VRAM. Hãy chọn runtime GPU phù hợp.')

def persist_qwen_cache(source_path):
    source_path = Path(source_path)
    if not source_path.exists():
        return
    temporary = DRIVE_CACHE.with_name(DRIVE_CACHE.name + '.uploading')
    shutil.copy2(source_path, temporary)
    temporary.replace(DRIVE_CACHE)

print('Cache sẽ được đồng bộ lên Drive định kỳ; cache local:', LOCAL_CACHE)
"""
    ),
    markdown(
        """## Chạy full Qwen

Chạy cell dưới đây. Preflight chỉ kiểm tra model/runtime và chuẩn bị checkpoint; kết quả QA của 20 title không dùng làm điều kiện chặn full. Toàn bộ title duy nhất sẽ được dịch trong một lượt dài, với SQLite checkpoint định kỳ. Nếu runtime bị ngắt, chạy lại notebook từ đầu để tiếp tục từ bản checkpoint gần nhất trên Drive.

Không đổi model, prompt hoặc bundle giữa chừng. Nếu đổi cấu hình, hãy tạo run folder mới.
"""
    ),
    code(
        """# Preflight 20 title, sau đó chạy full ngay cả khi QA preflight báo cờ.
QWEN_BATCH_SIZE = 4  # tự giảm 4 -> 2 -> 1 nếu CUDA OOM
CHECKPOINT_EVERY_TITLES = 100

def attach_durable_cache(context):
    context['cache_paths'] = {'qwen35': LOCAL_CACHE}
    context['cache_checkpoint_interval'] = CHECKPOINT_EVERY_TITLES
    context['cache_checkpoint_callback'] = persist_qwen_cache
    return context

smoke_context = pipeline.prepare_pipeline(
    mode='smoke', model_key='qwen35', data_path=SOURCE, workdir=WORKDIR,
    hf_token=None, qwen_batch_size=QWEN_BATCH_SIZE
)
attach_durable_cache(smoke_context)
smoke_result = pipeline.run_stage(
    smoke_context, batch_size=QWEN_BATCH_SIZE, allow_qwen_hardware_change=True
)
persist_qwen_cache(LOCAL_CACHE)
print('Preflight:', json.dumps({k: v for k, v in smoke_result.items() if k not in {'qa', 'cache'}}, ensure_ascii=False, indent=2, default=str))
if smoke_result.get('status') == 'quality_gate_failed':
    print('QA preflight có cờ; theo lựa chọn của bạn, nó không chặn full. QA toàn bộ sẽ được xuất sau khi dịch xong.')
smoke_generation = smoke_result.get('generation', {})
smoke_new = int(smoke_generation.get('new_completed', 0))
smoke_seconds = float(smoke_generation.get('elapsed_seconds', 0) or 0)
if smoke_new and smoke_seconds > 0:
    rate = smoke_new / smoke_seconds
    eta_hours = pipeline.EXPECTED_UNIQUE_TITLES / rate / 3600
    print(f'Tốc độ preflight: {rate * 3600:.1f} title/giờ; ETA thô cho full: {eta_hours:.1f} giờ (chưa tính reset/tải model lại).')

full_result = None
for resume_round in range(1, 4):
    full_context = pipeline.prepare_pipeline(
        mode='full', model_key='qwen35', data_path=SOURCE, workdir=WORKDIR,
        hf_token=None, qwen_batch_size=QWEN_BATCH_SIZE
    )
    attach_durable_cache(full_context)
    full_result = pipeline.run_stage(
        full_context,
        batch_size=QWEN_BATCH_SIZE,
        full_titles_per_session=pipeline.EXPECTED_UNIQUE_TITLES,
        allow_unvalidated_full=True,
        defer_full_quality_review=True,
        allow_qwen_hardware_change=True,
    )
    persist_qwen_cache(LOCAL_CACHE)
    if full_result.get('status') != 'checkpoint_saved_continue':
        break
    if resume_round < 3:
        print(f\"Còn title chưa hoàn tất; sẽ thử lại lượt {resume_round + 1}/3 từ checkpoint.\")
    else:
        print('Vẫn còn title chưa hoàn tất sau 3 lượt; checkpoint còn trên Drive. Chạy lại notebook để thử tiếp.')
assert full_result is not None
print('Full run:', json.dumps({k: v for k, v in full_result.items() if k not in {'qa', 'cache'}}, ensure_ascii=False, indent=2, default=str))
if full_result.get('status') != 'all_titles_complete':
    print('Chưa xong toàn bộ; checkpoint đã lưu trên Drive. Sau khi nối runtime lại, chọn Runtime → Run all để resume.')
"""
    ),
    markdown(
        """## Rà soát sau khi dịch xong

Chỉ chạy cell này sau khi full run có trạng thái `all_titles_complete`. Cell tạo CSV toàn bộ title, CSV các title cần xem lại và báo cáo tổng hợp. Cờ QA là kiểm tra kỹ thuật (số, mã, ký tự/ngôn ngữ, đầu ra), không chứng minh bản dịch đúng nghĩa.
"""
    ),
    code(
        """import json
import hashlib
import pandas as pd

state_path = WORKDIR / 'hybrid_run_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
full_status = state.get('models', {}).get('qwen35', {}).get('stages', {}).get('full', {}).get('status')
if full_status != 'all_titles_complete':
    raise RuntimeError(f'Qwen full chưa hoàn tất (status={full_status}); resume full trước khi rà soát.')

OUTPUT = WORKDIR / 'bilingual_zh_vi_cado_title_qwen35.parquet'
if not OUTPUT.exists():
    raise FileNotFoundError(f'Không thấy Parquet hoàn chỉnh: {OUTPUT}')
result_frame = pd.read_parquet(OUTPUT)
if len(result_frame) != pipeline.EXPECTED_ROWS:
    raise AssertionError(f'Số dòng sai: {len(result_frame)}; cần {pipeline.EXPECTED_ROWS}.')
qa_status_col = 'title_vi_qwen35_qa_status'
hard_flags_col = 'title_vi_qwen35_qa_flags'
advisory_col = 'title_vi_qwen35_advisory_flags'
semantic_col = 'title_vi_qwen35_semantic_verified'
required = [qa_status_col, hard_flags_col, advisory_col, semantic_col, 'title_vi_candidate_qwen35']
missing = [name for name in required if name not in result_frame.columns]
if missing:
    raise AssertionError(f'Thiếu cột QA/output: {missing}')

unique_review = result_frame.drop_duplicates('title_zh').copy()
review_cols = [name for name in [
    'product_id', 'title_zh', 'title_vi', 'title_vi_cleaned',
    'title_vi_candidate_qwen35', qa_status_col, hard_flags_col, advisory_col,
] if name in unique_review.columns]
all_review_path = WORKDIR / 'qwen35_postrun_qa_all_unique_titles.csv'
unique_review[review_cols].to_csv(all_review_path, index=False, encoding='utf-8-sig')
flagged = unique_review[
    unique_review[qa_status_col].ne('pass')
    | unique_review[hard_flags_col].fillna('[]').ne('[]')
    | unique_review[advisory_col].fillna('[]').ne('[]')
]
flagged_path = WORKDIR / 'qwen35_postrun_qa_flagged_titles.csv'
flagged[review_cols].to_csv(flagged_path, index=False, encoding='utf-8-sig')

status_counts = unique_review[qa_status_col].fillna('missing').value_counts().to_dict()
hard_counts = {}
for value in unique_review[hard_flags_col].fillna('[]'):
    for flag in json.loads(value):
        hard_counts[flag] = hard_counts.get(flag, 0) + 1
summary = {
    'rows': len(result_frame),
    'unique_titles': int(unique_review['title_zh'].nunique()),
    'qa_status_counts_unique_titles': status_counts,
    'hard_flag_counts_unique_titles': hard_counts,
    'flagged_unique_titles': len(flagged),
    'semantic_verified': bool(result_frame[semantic_col].fillna(True).astype(bool).any()),
    'parquet_sha256': pipeline.quality.sha256_file(OUTPUT),
}
summary_path = WORKDIR / 'qwen35_postrun_qa_summary.json'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
summary_md = [
    '# Qwen3.5-9B title — QA sau full run', '',
    f"- Số dòng: {summary['rows']:,}",
    f"- Title duy nhất: {summary['unique_titles']:,}",
    f"- Trạng thái QA: `{json.dumps(status_counts, ensure_ascii=False)}`",
    f"- Title có cờ cần xem: {summary['flagged_unique_titles']:,}",
    f"- SHA-256 Parquet: `{summary['parquet_sha256']}`",
    '- `semantic_verified=False`: chưa có người xác nhận ngữ nghĩa.',
    '',
    f'- Danh sách đầy đủ: `{all_review_path.name}`',
    f'- Danh sách cần xem: `{flagged_path.name}`',
]
(WORKDIR / 'qwen35_postrun_qa_summary.md').write_text('\\n'.join(summary_md) + '\\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
"""
    ),
    code(
        """# Đóng gói kết quả/checkpoint để tải về máy; không upload HF.
import shutil

archive_base = DRIVE_ROOT / 'title_translation_cado_qwen35_full_v1_results'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=WORKDIR, base_dir='.'))
print('ZIP kết quả:', archive_path)
print('Parquet:', OUTPUT)
print('QA flags:', flagged_path)
files.download(str(archive_path))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "title_translation_qwen35_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_path = source_file('title_translation_qwen35_colab.ipynb')
notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

bundle_path = ROOT / "title_translation_qwen35_colab_bundle.zip"
with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ("title_translation_hybrid.py", "title_translation_pipeline.py"):
        archive.write(source_file(name), arcname=name)

print(notebook_path)
print(bundle_path)

# A second notebook resumes the same durable run but installs the model's optional
# CUDA kernels and lets runtime batch size exceed the original config's safe seed.
fast_cells = copy.deepcopy(cells)
for cell in fast_cells:
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        text = text.replace(
            "# Dịch full title Cado bằng Qwen3.5-9B trên Google Colab",
            "# Dịch full title Cado bằng Qwen3.5-9B — bản tăng tốc CUDA",
        )
        text = text.replace(
            "Runtime bắt đầu với batch 4 và tự giảm xuống 2/1 nếu CUDA OOM. Cache SQLite được sao lưu lên Drive mỗi 100 title đã xử lý.",
            "Cài kernel CUDA FLA + causal-conv1d tương thích, chọn batch theo VRAM (8/16/32) và tự chia đôi nếu CUDA OOM. Cache SQLite tiếp tục dùng cùng thư mục Drive, đồng bộ mỗi 100 title.",
        )
        if "## Trước khi chạy" in text:
            text = """## Trước khi chạy bản tăng tốc

1. Trong notebook cũ, dừng cell đang chạy rồi tạo cell mới chạy `persist_qwen_cache(LOCAL_CACHE)` để đồng bộ checkpoint mới nhất lên Drive.
2. Chọn **Runtime → Restart session** để Transformers được nạp lại sau khi cài kernel.
3. Mở notebook này, chọn GPU mạnh nhất có thể (mã dùng GPU 0; GPU phụ hiện chưa được dùng), rồi chọn **Runtime → Run all** và cấp quyền mount Drive.
4. Khi được hỏi, tải lên `title_translation_qwen35_colab_fast_bundle.zip`. Notebook giữ nguyên thư mục checkpoint cũ và tiếp tục title còn thiếu.

Không cần Gemini key hoặc HF token. Không xóa hoặc đổi tên thư mục run trên Drive.
"""
        text = text.replace(
            "Runtime bắt đầu với batch 4",
            "Runtime chọn batch theo VRAM",
        )
        text = text.replace(
            "# Preflight 20 title, sau đó chạy full ngay cả khi QA preflight báo cờ.",
            "# Preflight 20 title, sau đó chạy full ngay cả khi QA preflight báo cờ.",
        )
        cell["source"] = text.splitlines(keepends=True)
    elif cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "QWEN_BATCH_SIZE = 4  # tự giảm 4 -> 2 -> 1 nếu CUDA OOM" in source:
            source = source.replace(
                "QWEN_BATCH_SIZE = 4  # tự giảm 4 -> 2 -> 1 nếu CUDA OOM\nCHECKPOINT_EVERY_TITLES = 100",
                """# Batch cao hơn khi GPU đủ VRAM; pipeline tự chia đôi khi gặp CUDA OOM.
QWEN_BATCH_SIZE = 64 if gpu_vram_gib >= 72 else 32 if gpu_vram_gib >= 36 else 16 if gpu_vram_gib >= 22 else 8
LOCK_PATH = WORKDIR / 'hybrid_environment_lock.json'
if LOCK_PATH.exists():
    LOCKED_BATCH_SIZE = int(json.loads(LOCK_PATH.read_text(encoding='utf-8'))['config']['generation']['qwen_batch_size'])
else:
    LOCKED_BATCH_SIZE = QWEN_BATCH_SIZE
print(f'Qwen execution batch={QWEN_BATCH_SIZE}; saved config batch={LOCKED_BATCH_SIZE}; OOM sẽ tự giảm một nửa.')
CHECKPOINT_EVERY_TITLES = 100""",
            )
            source = source.replace(
                "hf_token=None, qwen_batch_size=QWEN_BATCH_SIZE",
                "hf_token=None, qwen_batch_size=LOCKED_BATCH_SIZE",
            )
            source = source.replace(
                "smoke_context, batch_size=QWEN_BATCH_SIZE, allow_qwen_hardware_change=True",
                "smoke_context, batch_size=QWEN_BATCH_SIZE, allow_qwen_hardware_change=True,\n    allow_qwen_batch_change=True",
            )
            source = source.replace(
                "        allow_qwen_hardware_change=True,\n",
                "        allow_qwen_hardware_change=True,\n        allow_qwen_batch_change=True,\n",
            )
            cell["source"] = source.splitlines(keepends=True)

fast_cells.insert(
    4,
    code(
        """# Install pinned Qwen3.5 fast kernels without changing the Colab torch/CUDA stack.
import importlib.metadata as metadata
from packaging.version import Version
import subprocess
import sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError('Bản tăng tốc yêu cầu GPU CUDA.')
torch_before = metadata.version('torch')
if Version(torch_before.split('+', 1)[0]) < Version('2.7.0'):
    raise RuntimeError(f'PyTorch {torch_before} quá cũ cho FLA CUDA; giữ checkpoint và dùng runtime mới hơn.')

constraint_file = Path('/tmp/qwen35_fast_kernel_constraints.txt')
constraint_lines = [f'{name}=={version}' for name, version in locked_packages.items()]
try:
    constraint_lines.append(f"triton=={metadata.version('triton')}")
except metadata.PackageNotFoundError:
    pass
constraint_file.write_text('\\n'.join(constraint_lines) + '\\n', encoding='utf-8')

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--constraint', str(constraint_file),
    'flash-linear-attention[cuda]==0.5.2', 'fla-core==0.5.2', 'causal-conv1d==1.7.0',
])
importlib.invalidate_caches()
if metadata.version('torch') != torch_before:
    raise RuntimeError('Cài kernel đã thay đổi PyTorch; dừng để không trộn môi trường/checkpoint.')

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
import fla
import fla.layers  # exposes fla.ops on package-root layouts used by Transformers' resolver
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.utils.import_utils import resolve_internal_import

optional_kernel_checks = {
    'causal_conv1d_fn': resolve_internal_import(importlib.import_module('causal_conv1d'), 'causal_conv1d_fn'),
    'causal_conv1d_update': resolve_internal_import(importlib.import_module('causal_conv1d'), 'causal_conv1d_update'),
    'chunk_gated_delta_rule': resolve_internal_import(fla, 'ops.gated_delta_rule.chunk_gated_delta_rule'),
    'fused_recurrent_gated_delta_rule': resolve_internal_import(fla, 'ops.gated_delta_rule.fused_recurrent_gated_delta_rule'),
}
missing_kernels = [name for name, resolved in optional_kernel_checks.items() if resolved is None]
if missing_kernels:
    raise RuntimeError(
        f'Optional CUDA kernel resolver vẫn chưa nhận: {missing_kernels}. '
        'Dừng trước khi tải model để không chạy PyTorch fallback chậm.'
    )
print('Transformers đã resolve các optional CUDA kernels:', {
    name: f'{fn.__module__}.{fn.__name__}' for name, fn in optional_kernel_checks.items()
})

kernel_versions = {
    name: metadata.version(name)
    for name in ('flash-linear-attention', 'fla-core', 'causal-conv1d')
}
kernel_lock_path = WORKDIR / 'qwen35_fast_kernel_lock.json'
kernel_lock = {'packages': kernel_versions, 'torch': torch_before, 'cuda': torch.version.cuda}
if kernel_lock_path.exists():
    saved_kernel_lock = json.loads(kernel_lock_path.read_text(encoding='utf-8'))
    if saved_kernel_lock != kernel_lock:
        raise RuntimeError(f'Kernel/runtime khác checkpoint lock: {saved_kernel_lock} != {kernel_lock}')
else:
    temp_lock = kernel_lock_path.with_suffix('.json.tmp')
    temp_lock.write_text(json.dumps(kernel_lock, indent=2), encoding='utf-8')
    temp_lock.replace(kernel_lock_path)
print('Fast kernels đã import và khóa phiên bản:', json.dumps(kernel_lock, indent=2))
"""
    ),
)

for cell in fast_cells:
    if cell["cell_type"] == "code" and "gpu = torch.cuda.get_device_properties(0)" in "".join(cell["source"]):
        source = "".join(cell["source"])
        source = source.replace(
            "print(f'GPU: {gpu.name}; VRAM: {gpu_vram_gib:.1f} GiB')",
            "print(f'GPU 0: {gpu.name}; VRAM: {gpu_vram_gib:.1f} GiB; notebook dùng GPU 0.')\n"
            "if torch.cuda.device_count() > 1:\n"
            "    print(f'Có {torch.cuda.device_count()} GPU; các GPU còn lại chưa được dùng trong bản tăng tốc này.')",
        )
        cell["source"] = source.splitlines(keepends=True)
    elif cell["cell_type"] == "code" and "BUNDLE_PATH = DRIVE_ROOT / 'title_translation_qwen35_colab_bundle.zip'" in "".join(cell["source"]):
        source = "".join(cell["source"]).replace(
            "title_translation_qwen35_colab_bundle.zip",
            "title_translation_qwen35_colab_fast_bundle.zip",
        )
        cell["source"] = source.splitlines(keepends=True)
    elif cell["cell_type"] == "code" and "archive_base = DRIVE_ROOT / 'title_translation_cado_qwen35_full_v1_results'" in "".join(cell["source"]):
        source = "".join(cell["source"]).replace(
            "title_translation_cado_qwen35_full_v1_results",
            "title_translation_cado_qwen35_fast_v2_results",
        )
        cell["source"] = source.splitlines(keepends=True)

fast_notebook = {
    **notebook,
    "cells": fast_cells,
    "metadata": {
        **notebook["metadata"],
        "colab": {"name": "title_translation_qwen35_colab_fast.ipynb", "provenance": []},
    },
}
fast_notebook_path = ROOT / "title_translation_qwen35_colab_fast.ipynb"
fast_notebook_path.write_text(json.dumps(fast_notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

fast_bundle_path = ROOT / "title_translation_qwen35_colab_fast_bundle.zip"
with zipfile.ZipFile(fast_bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ("title_translation_hybrid.py", "title_translation_pipeline.py"):
        archive.write(source_file(name), arcname=name)

print(fast_notebook_path)
print(fast_bundle_path)
