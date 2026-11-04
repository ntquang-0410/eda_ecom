"""Build a Kaggle-port of the reviewed Qwen3-14B-AWQ v6 notebook.

This changes only the notebook's platform/storage integration. Model, prompt,
sampling policy, QA gates, source dataset and cache migration logic stay tied to
the v6 Colab bundle.
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

import ast
import json
from pathlib import Path


ROOT = generated_root()
SOURCE_NOTEBOOK = source_file('title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload.ipynb')
NOTEBOOK_NAME = "title_translation_qwen3_14b_awq_kaggle_v6.ipynb"
GUIDE_NAME = "TITLE_TRANSLATION_QWEN3_14B_AWQ_KAGGLE_V6_GUIDE.md"
BUNDLE_NAME = "title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload_bundle.zip"
RUN_DIR_V2 = "title_translation_cado_qwen3_14b_awq_full_v2"
RUN_DIR_V6 = "title_translation_cado_qwen3_14b_awq_full_v3_sampled"
CACHE_NAME = "cache_qwen3_14b_awq.sqlite3"
CHECKPOINT_ZIP_NAME = "qwen3_14b_awq_kaggle_checkpoint.zip"

PIP_INSTALL_RETRY_BLOCK = r'''_DNS_BOOTSTRAP_SOURCE = r"""
import http.client
import ipaddress
import json
import socket
import ssl
import threading
import urllib.parse

_original_getaddrinfo = socket.getaddrinfo
_doh_cache = {}
_doh_lock = threading.Lock()
_doh_disabled = False

def _cloudflare_a_records(host):
    global _doh_disabled
    with _doh_lock:
        if _doh_disabled:
            return ()
        if host in _doh_cache:
            return _doh_cache[host]
        sock = None
        tls = None
        try:
            # Connect to Cloudflare by fixed IP while verifying TLS/SNI by hostname.
            sock = socket.create_connection(('1.1.1.1', 443), timeout=4)
            tls = ssl.create_default_context().wrap_socket(
                sock, server_hostname='cloudflare-dns.com'
            )
            query = urllib.parse.urlencode({'name': host, 'type': 'A'})
            request = (
                f'GET /dns-query?{query} HTTP/1.1\r\n'
                'Host: cloudflare-dns.com\r\n'
                'Accept: application/dns-json\r\n'
                'Connection: close\r\n\r\n'
            )
            tls.sendall(request.encode('ascii'))
            response = http.client.HTTPResponse(tls)
            response.begin()
            if response.status != 200:
                addresses = ()
            else:
                payload = json.loads(response.read())
                addresses = tuple(
                    answer['data'] for answer in payload.get('Answer', [])
                    if answer.get('type') == 1
                    and isinstance(answer.get('data'), str)
                    and not answer['data'].startswith('127.')
                )
        except Exception:
            addresses = ()
            _doh_disabled = True
        finally:
            if tls is not None:
                try:
                    tls.close()
                except Exception:
                    pass
            elif sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        _doh_cache[host] = addresses
        return addresses

def _getaddrinfo_with_doh(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror as original_error:
        if not isinstance(host, str) or family not in (0, socket.AF_UNSPEC, socket.AF_INET):
            raise original_error
        try:
            ipaddress.ip_address(host)
            raise original_error
        except ValueError:
            pass
        addresses = _cloudflare_a_records(host)
        if not addresses:
            raise original_error
        port_value = int(port) if isinstance(port, str) and port.isdigit() else port
        socktype = type or socket.SOCK_STREAM
        protocol = proto or (socket.IPPROTO_TCP if socktype == socket.SOCK_STREAM else 0)
        return [
            (socket.AF_INET, socktype, protocol, '', (address, port_value))
            for address in addresses
        ]

socket.getaddrinfo = _getaddrinfo_with_doh
"""

def _pip_install(command):
    from collections import deque
    import os
    import time

    transient_network_markers = (
        'NameResolutionError',
        'Temporary failure in name resolution',
        'Failed to resolve',
        'Could not resolve host',
        'Connection timed out',
        'ReadTimeoutError',
        'NewConnectionError',
        'Network is unreachable',
    )
    dns_bootstrap_dir = WORKDIR / 'pip_dns_bootstrap'
    dns_bootstrap_dir.mkdir(parents=True, exist_ok=True)
    dns_bootstrap_file = dns_bootstrap_dir / 'sitecustomize.py'
    dns_bootstrap_file.write_text(_DNS_BOOTSTRAP_SOURCE.lstrip(), encoding='utf-8')
    import importlib.util
    dns_spec = importlib.util.spec_from_file_location('_qwen_dns_bootstrap', dns_bootstrap_file)
    dns_module = importlib.util.module_from_spec(dns_spec)
    dns_spec.loader.exec_module(dns_module)
    pip_env = os.environ.copy()
    existing_pythonpath = pip_env.get('PYTHONPATH')
    pip_env['PYTHONPATH'] = os.pathsep.join(
        part for part in (str(dns_bootstrap_dir), existing_pythonpath) if part
    )

    wheelhouse_manifests = list(Path('/kaggle/input').rglob('qwen_offline_wheelhouse.json'))
    if len(wheelhouse_manifests) > 1:
        raise RuntimeError('Tìm thấy nhiều offline wheelhouse; cần đúng một Kaggle Input.')
    if wheelhouse_manifests:
        import hashlib
        import shutil
        manifest_path = wheelhouse_manifests[0]
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        opaque_sdist_manifests = list(Path('/kaggle/input').rglob('qwen_opaque_sdists.json'))
        if len(opaque_sdist_manifests) != 1:
            raise RuntimeError('Cần đúng một Kaggle Input chứa các gói nguồn đã đổi tên.')
        opaque_manifest_path = opaque_sdist_manifests[0]
        opaque_manifest = json.loads(opaque_manifest_path.read_text(encoding='utf-8'))
        opaque_by_name = {
            item['filename']: item for item in opaque_manifest.get('artifacts', [])
        }
        opaque_files = {}
        for opaque_item in opaque_manifest.get('artifacts', []):
            opaque_matches = [
                path for path in opaque_manifest_path.parent.rglob(opaque_item['opaque_filename'])
                if path.is_file()
            ]
            if len(opaque_matches) != 1:
                raise RuntimeError('Không tìm thấy đúng một file gói nguồn đổi tên trong Kaggle Input.')
            opaque_path = opaque_matches[0]
            if opaque_path.stat().st_size != opaque_item['size']:
                raise RuntimeError('Kích thước gói nguồn đổi tên không khớp manifest.')
            opaque_digest = hashlib.sha256(opaque_path.read_bytes()).hexdigest()
            if opaque_digest != opaque_item['sha256']:
                raise RuntimeError('SHA-256 gói nguồn đổi tên không khớp manifest.')
            opaque_files[opaque_item['filename']] = opaque_path

        install_wheelhouse = WORKDIR / 'qwen_offline_wheelhouse'
        install_wheelhouse.mkdir(parents=True, exist_ok=True)
        invalid_artifacts = []
        for artifact in manifest.get('artifacts', []):
            filename = artifact['filename']
            if Path(filename).name != filename:
                invalid_artifacts.append(artifact['filename'])
                continue
            if filename.endswith(('.tar.gz', '.zip')):
                artifact_path = opaque_files.get(filename)
            else:
                artifact_matches = [
                    path for path in manifest_path.parent.rglob(filename)
                    if path.is_file()
                ]
                artifact_path = artifact_matches[0] if len(artifact_matches) == 1 else None
            if artifact_path is None or artifact_path.stat().st_size != artifact['size']:
                invalid_artifacts.append(filename)
                continue
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if digest != artifact['sha256']:
                invalid_artifacts.append(filename)
                continue
            shutil.copyfile(artifact_path, install_wheelhouse / filename)
        if set(opaque_by_name) != {
            artifact['filename'] for artifact in manifest.get('artifacts', [])
            if artifact['filename'].endswith(('.tar.gz', '.zip'))
        }:
            invalid_artifacts.append('opaque_source_manifest')
        if invalid_artifacts:
            raise RuntimeError(f'Offline wheelhouse thiếu/sai hash ở {len(invalid_artifacts)} artifact.')

        pip_command = list(command)
        install_pos = pip_command.index('install')
        pip_command[install_pos + 1:install_pos + 1] = [
            '--no-index', '--find-links', str(install_wheelhouse), '--retries', '0', '--timeout', '20',
        ]
        print(f'Cài dependency offline từ Kaggle Input ({len(manifest.get("artifacts", []))} artifact đã xác minh).', flush=True)
        process = subprocess.Popen(pip_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, env=pip_env)
        tail = deque(maxlen=300)
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
            raise RuntimeError(f'Offline pip install exited with code {return_code}; xem qwen_install_last.log.')
        return

    index_error_markers = transient_network_markers + (
        'No matching distribution found',
        'Could not find a version that satisfies the requirement',
    )
    indexes = (
        ('PyPI', None),
        ('Tsinghua mirror', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'),
        ('Huawei mirror', 'https://mirrors.huaweicloud.com/repository/pypi/simple'),
        ('SUSTech mirror', 'https://mirrors.sustech.edu.cn/pypi/web/simple'),
        ('Aliyun mirror', 'https://mirrors.aliyun.com/pypi/simple/'),
    )
    return_code = 1
    log_text = ''
    for index_name, index_url in indexes:
        pip_command = list(command)
        install_pos = pip_command.index('install')
        pip_options = ['--retries', '1', '--timeout', '25']
        if index_url:
            pip_options.extend(['--index-url', index_url])
        pip_command[install_pos + 1:install_pos + 1] = pip_options

        tail = deque(maxlen=300)
        process = subprocess.Popen(pip_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, env=pip_env)
        for line in process.stdout:
            print(line, end='', flush=True)
            tail.append(line)
        return_code = process.wait()
        log_text = ''.join(tail)
        if not return_code:
            break
        if not any(marker in log_text for marker in index_error_markers):
            break
        if (index_name, index_url) != indexes[-1]:
            print(f'pip không lấy được package từ {index_name}; chuyển sang nguồn dự phòng.', flush=True)

    redactor = globals().get('_qwen_redact')
    if callable(redactor):
        log_text = redactor(log_text)
    (WORKDIR / 'qwen_install_last.log').write_text(log_text, encoding='utf-8')
    if return_code:
        raise RuntimeError(f'pip install exited with code {return_code}; last output follows:\n{log_text[-10000:]}')
'''


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def set_cell_text(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


LOCAL_QWEN_MODEL_SETUP = r'''# Load the exact Qwen snapshot from a read-only Kaggle Input. The Hugging Face
# repository id and revision remain in the run lock; only model file transport
# is redirected to the verified local snapshot.
import hashlib
import os
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_REPO = 'Qwen/Qwen3-14B-AWQ'
QWEN_MODEL_REVISION = '31c69efc29464b6bb0aee1398b5a7b50a99340c3'
QWEN_MODEL_FILES = {
    '.gitattributes': {'size': 1570, 'git_sha1': '52373fe24473b1aa44333d318f578ae6bf04b49b'},
    'LICENSE': {'size': 11544, 'git_sha1': '6f9739bb281a3d2ade8fb5f6dd6dde597a739c80'},
    'README.md': {'size': 16885, 'git_sha1': '6ba98def472ad73c5cc6849d5da9e8389b0f88af'},
    'config.json': {'size': 907, 'git_sha1': 'f05252483c338f12a888a7f59698f1ea405e60e2'},
    'generation_config.json': {'size': 239, 'git_sha1': '20a8a9156fc8c3f25295ca067f61fdf120d517c5'},
    'merges.txt': {'size': 1671853, 'git_sha1': '31349551d90c7606f325fe0f11bbb8bd5fa0d7c7'},
    'model-00001-of-00002.safetensors': {
        'size': 4988339832,
        'sha256': '668eb0f1356638310db286f4819b223c12e3916934123f1a81b2b2c0e148c6a2',
    },
    'model-00002-of-00002.safetensors': {
        'size': 4988350408,
        'sha256': 'c3c1625df80fe01211038bfa520629ebde6adf776556aa80cd49696d986d6657',
    },
    'model.safetensors.index.json': {'size': 82253, 'git_sha1': '2ff701afad2fc062c13976386e06b7ade7d81f12'},
    'tokenizer.json': {
        'size': 11422654,
        'sha256': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    },
    'tokenizer_config.json': {'size': 9732, 'git_sha1': '417d038a63fa3de29cfde265caedae14d1a58d92'},
    'vocab.json': {'size': 2776833, 'git_sha1': '4783fe10ac3adce15ac8f358ef5462739852c569'},
}

def _qwen_verify_model_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected['size']:
        raise RuntimeError('Kaggle Qwen model input is missing a file or has an unexpected size.')
    if 'sha256' in expected:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
                _qwen_health.progress(phase='verify_model', progress_unit='bytes', progress_done=stream.tell(), progress_total=expected['size'])
        actual = digest.hexdigest()
        wanted = expected['sha256']
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {expected['size']}\0".encode('ascii'))
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
                _qwen_health.progress(phase='verify_model', progress_unit='bytes', progress_done=stream.tell(), progress_total=expected['size'])
        actual = digest.hexdigest()
        wanted = expected['git_sha1']
    if actual != wanted:
        raise RuntimeError('Kaggle Qwen model input checksum does not match the pinned official revision.')

qwen_model_roots = set()
for config_path in KAGGLE_INPUT_ROOT.rglob('config.json'):
    candidate = config_path.parent
    if all((candidate / name).is_file() for name in QWEN_MODEL_FILES):
        qwen_model_roots.add(candidate.resolve())
if len(qwen_model_roots) != 1:
    raise FileNotFoundError(
        f'Expected exactly one Kaggle Input for the pinned Qwen model; found {len(qwen_model_roots)}.'
    )
QWEN_LOCAL_MODEL_DIR = next(iter(qwen_model_roots))
_qwen_health.set_phase('verify_model')
for filename, expected in QWEN_MODEL_FILES.items():
    _qwen_verify_model_file(QWEN_LOCAL_MODEL_DIR / filename, expected)

qwen_config = json.loads((QWEN_LOCAL_MODEL_DIR / 'config.json').read_text(encoding='utf-8'))
if qwen_config.get('model_type') != 'qwen3' or (qwen_config.get('quantization_config') or {}).get('quant_method') != 'awq':
    raise RuntimeError('Kaggle model input is not the expected Qwen3 AWQ checkpoint.')
qwen_index = json.loads((QWEN_LOCAL_MODEL_DIR / 'model.safetensors.index.json').read_text(encoding='utf-8'))
if set(qwen_index.get('weight_map', {}).values()) != {
    'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'
}:
    raise RuntimeError('Kaggle model input has an unexpected safetensors shard map.')

def _qwen_from_local_snapshot(original_loader):
    def _wrapped_loader(cls, model_name_or_path, *args, **kwargs):
        if str(model_name_or_path) == QWEN_MODEL_REPO:
            kwargs.pop('revision', None)
            kwargs.pop('token', None)
            kwargs['local_files_only'] = True
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            return original_loader(str(QWEN_LOCAL_MODEL_DIR), *args, **kwargs)
        return original_loader(model_name_or_path, *args, **kwargs)
    return classmethod(_wrapped_loader)

_qwen_original_tokenizer_loader = AutoTokenizer.from_pretrained
_qwen_original_model_loader = AutoModelForCausalLM.from_pretrained
AutoTokenizer.from_pretrained = _qwen_from_local_snapshot(_qwen_original_tokenizer_loader)
AutoModelForCausalLM.from_pretrained = _qwen_from_local_snapshot(_qwen_original_model_loader)

def _qwen_local_revision_lock(hf_token):
    locked = {}
    for key, model in pipeline.MODELS.items():
        if model['backend'] == 'google_gemini_api':
            revision = model['api_model_id']
        elif key == 'qwen3_14b_awq':
            revision = QWEN_MODEL_REVISION
        else:
            raise RuntimeError('Unexpected Hugging Face model in the Kaggle Qwen lock.')
        locked[key] = {
            'model_id': model['model_id'],
            'backend': model['backend'],
            'provider': model['provider'],
            'revision': revision,
            'template_version': model['template_version'],
        }
    return locked

pipeline._model_revision_lock = _qwen_local_revision_lock
print('Verified all 12 files for the pinned Qwen3-14B-AWQ snapshot; model loading will use the local Kaggle Input.', flush=True)
'''


def replace_between(source: str, start: str, end: str, replacement: str) -> str:
    start_at = source.find(start)
    if start_at < 0:
        raise RuntimeError(f"Could not find start anchor: {start!r}")
    end_at = source.find(end, start_at + len(start))
    if end_at < 0:
        raise RuntimeError(f"Could not find end anchor: {end!r}")
    return source[:start_at] + replacement + source[end_at:]


CELL_2 = r'''# Kaggle has read-only inputs and writable outputs; never mount or change Google Drive here.
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from IPython import get_ipython

# Check the actual runtime before dependency installation or model-file hashing.
_qwen_health.set_phase('preflight_gpu', cell=2)
import torch
_gpu_probe = {
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'nvidia_device_present': Path('/dev/nvidia0').exists(),
    'nvidia_smi_available': bool(shutil.which('nvidia-smi')),
}
print('[qwen-boot] ' + json.dumps(_gpu_probe), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('Kaggle chưa cấp CUDA GPU. Vào Edit notebook, Settings / Session options, chọn GPU T4 x2 rồi bắt đầu lại session; không cài dependency trên runtime CPU.')
_gpu_properties = torch.cuda.get_device_properties(0)
if _gpu_properties.total_memory < 14 * 1024**3 or (_gpu_properties.major, _gpu_properties.minor) < (7, 5):
    raise RuntimeError('GPU hiện tại không đủ VRAM hoặc compute capability để chạy Qwen3-14B-AWQ ExLlamaV2.')

KAGGLE_INPUT_ROOT = Path('/kaggle/input')
KAGGLE_WORK_ROOT = Path('/kaggle/working')
WORKDIR = KAGGLE_WORK_ROOT / 'title_translation_cado_qwen3_14b_awq_full_v3_sampled'
WORKDIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR = Path('/kaggle/temp/title_translation_cado_qwen3_14b_awq_full_v3_sampled')
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE = LOCAL_DIR / 'cache_qwen3_14b_awq.sqlite3'
WORKING_CACHE = WORKDIR / 'cache_qwen3_14b_awq.sqlite3'
BUNDLE_NAME = 'title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload_bundle.zip'
LEGACY_RUN_DIR_NAME = 'title_translation_cado_qwen3_14b_awq_full_v2'
CHECKPOINT_ZIP_NAME = 'qwen3_14b_awq_kaggle_checkpoint.zip'

if not KAGGLE_INPUT_ROOT.exists():
    raise RuntimeError('Không thấy /kaggle/input. Hãy mở notebook trong Kaggle và đính kèm Kaggle Dataset đầu vào.')

bundle_candidates = [
    path for path in KAGGLE_INPUT_ROOT.rglob(BUNDLE_NAME)
    if path.is_file()
]
if len(bundle_candidates) == 1:
    BUNDLE_INPUT = bundle_candidates[0]
elif len(bundle_candidates) == 0:
    # Kaggle may recursively unpack both the outer Dataset ZIP and this
    # nested bundle ZIP. In that case the bundle is an extracted directory,
    # not a file with BUNDLE_NAME. Repack only its two known source files in
    # the same flat layout expected by the rest of this setup cell.
    bundle_dir_names = {BUNDLE_NAME, BUNDLE_NAME[:-4]}
    unpacked_bundle_dirs = [
        path for path in KAGGLE_INPUT_ROOT.rglob('*')
        if path.is_dir() and path.name in bundle_dir_names
    ]
    if len(unpacked_bundle_dirs) != 1:
        raise FileNotFoundError(
            'Cần đúng 1 bundle v6 trong Kaggle Input; '
            f'tìm thấy {len(bundle_candidates)} file ZIP và '
            f'{len(unpacked_bundle_dirs)} thư mục bundle đã giải nén. '
            'Hãy đính kèm private dataset theo guide.'
        )
    expected_bundle_files = (
        'title_translation_hybrid.py',
        'title_translation_pipeline.py',
    )
    unpacked_bundle_dir = unpacked_bundle_dirs[0]
    unpacked_files = {}
    for name in expected_bundle_files:
        matches = [path for path in unpacked_bundle_dir.rglob(name) if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f'Thư mục bundle đã bung phải có đúng một {name}; '
                f'tìm thấy {len(matches)}.'
            )
        unpacked_files[name] = matches[0]
    repacked_bundle = LOCAL_DIR / BUNDLE_NAME
    with zipfile.ZipFile(repacked_bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in unpacked_files.items():
            archive.write(path, arcname=name)
    BUNDLE_INPUT = repacked_bundle
    print('Kaggle đã tự bung bundle; đã đóng gói lại hai file nguồn:', unpacked_bundle_dir)
else:
    raise FileNotFoundError(
        f'Cần đúng 1 bundle v6 trong Kaggle Input; hiện tìm thấy {len(bundle_candidates)} file ZIP. '
        'Hãy chỉ đính kèm một private dataset theo guide.'
    )
LEGACY_CACHE_NAME = 'cache_qwen3_14b_awq.sqlite3'
legacy_cache_candidates = [
    path for path in KAGGLE_INPUT_ROOT.rglob(LEGACY_CACHE_NAME)
    if path.parent.name == LEGACY_RUN_DIR_NAME
    and (path.parent / 'hybrid_environment_lock.json').is_file()
]
if len(legacy_cache_candidates) > 1:
    raise RuntimeError('Tìm thấy nhiều checkpoint v2; hãy chỉ đính kèm một bản checkpoint nguồn.')
LEGACY_WORKDIR = legacy_cache_candidates[0].parent if legacy_cache_candidates else None

# A checkpoint ZIP exported by this notebook can restore a v6 run when Kaggle
# Files persistence is unavailable. Only flat, known runtime files are allowed.
allowed_snapshot_files = {
    'cache_qwen3_14b_awq.sqlite3', 'hybrid_environment_lock.json',
    'hybrid_run_state.json', 'legacy_cache_import.json',
    'qwen3_14b_awq_kaggle_package_lock.json', 'qwen_live_status.json', 'qwen_health.json',
}
snapshot_candidates = list(KAGGLE_INPUT_ROOT.rglob(CHECKPOINT_ZIP_NAME))
if len(snapshot_candidates) > 1:
    raise RuntimeError('Tìm thấy nhiều ZIP checkpoint Kaggle; chỉ đính kèm bản mới nhất.')
if snapshot_candidates:
    with zipfile.ZipFile(snapshot_candidates[0]) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or len(path.parts) != 1 or '..' in path.parts:
                raise RuntimeError('ZIP checkpoint có đường dẫn không an toàn; không giải nén.')
            if path.name not in allowed_snapshot_files:
                raise RuntimeError(f'Tệp không được phép trong ZIP checkpoint: {path.name}')
            destination = WORKDIR / path.name
            if not destination.exists() and not member.is_dir():
                destination.write_bytes(archive.read(member))
    print('Đã khôi phục snapshot checkpoint Kaggle vào:', WORKDIR)

BUNDLE_PATH = WORKDIR / BUNDLE_NAME
if BUNDLE_PATH.exists():
    import hashlib
    old_sha = hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
    input_sha = hashlib.sha256(BUNDLE_INPUT.read_bytes()).hexdigest()
    has_prior_run = WORKING_CACHE.exists() or (WORKDIR / 'hybrid_run_state.json').exists()
    if old_sha != input_sha and has_prior_run:
        raise RuntimeError(
            'Bundle Kaggle khác với bundle đã dùng để tạo checkpoint. '
            'Không tự thay code khi resume; hãy dừng và yêu cầu kiểm tra trước.'
        )
    if old_sha != input_sha:
        shutil.copy2(BUNDLE_INPUT, BUNDLE_PATH)
else:
    shutil.copy2(BUNDLE_INPUT, BUNDLE_PATH)

if WORKING_CACHE.exists() and not LOCAL_CACHE.exists():
    shutil.copy2(WORKING_CACHE, LOCAL_CACHE)
    print('Đã khôi phục cache v6 từ /kaggle/working.')

has_v6_checkpoint = WORKING_CACHE.exists() or (WORKDIR / 'legacy_cache_import.json').exists()
if not has_v6_checkpoint and LEGACY_WORKDIR is None:
    raise FileNotFoundError(
        'Thiếu cache nguồn v2. Đính kèm private Kaggle Dataset có '
        'title_translation_cado_qwen3_14b_awq_full_v2/cache_qwen3_14b_awq.sqlite3 '
        'và hybrid_environment_lock.json; xem guide đi kèm.'
    )

if LEGACY_WORKDIR is not None:
    print('Đã tìm thấy checkpoint v2 để nhập chỉ-đọc:', LEGACY_WORKDIR)
else:
    print('Tiếp tục checkpoint v6 đã có; không cần nhập lại cache v2.')
print('Kaggle run folder:', WORKDIR)
print('Cache local runtime:', LOCAL_CACHE)

# Write privacy-safe status locally; there is no Google Drive mount or remote
# code-reload channel in the Kaggle port.
import traceback
_qwen_status_path = WORKDIR / 'qwen_live_status.json'
def _qwen_redact(value):
    value = str(value)
    value = re.sub(r'hf_[A-Za-z0-9]{16,}', '[REDACTED_HF_TOKEN]', value)
    value = re.sub(r'AIza[0-9A-Za-z_-]{20,}', '[REDACTED_GOOGLE_KEY]', value)
    value = re.sub(r'AQ\.[A-Za-z0-9_-]{20,}', '[REDACTED_API_KEY]', value)
    return value
def _qwen_kaggle_error_hook(result):
    error = getattr(result, 'error_before_exec', None) or getattr(result, 'error_in_exec', None)
    if error is None:
        raw_cell = getattr(getattr(result, 'info', None), 'raw_cell', '') or ''
        if '_qwen_full_run_cell_marker' not in raw_cell:
            return
        summary = globals().get('full_result')
        if not isinstance(summary, dict):
            return
        try:
            payload = {
                'state': summary.get('status', 'unknown'),
                'phase': 'run_cell_finished',
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                'completed_unique_titles': summary.get('completed_unique_titles'),
                'total_unique_titles': summary.get('total_unique_titles'),
                'remaining_unique_titles': summary.get('remaining_unique_titles'),
                'generation': summary.get('generation'),
            }
            temp = _qwen_status_path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(_qwen_status_path)
        except Exception as log_error:
            print(f'[qwen-kaggle-observer] status write failed: {type(log_error).__name__}', flush=True)
        return
    try:
        payload = {
            'state': 'error',
            'phase': 'notebook_cell_error',
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'cell_execution_count': getattr(result, 'execution_count', None),
            'last_error_class': type(error).__name__,
        }
        temp = _qwen_status_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(_qwen_status_path)
    except Exception as log_error:
        print(f'[qwen-kaggle-observer] status write failed: {type(log_error).__name__}', flush=True)
_qwen_ip = get_ipython()
if _qwen_ip is not None:
    _old_hook = getattr(_qwen_ip, '_qwen_kaggle_error_hook', None)
    if _old_hook is not None:
        try: _qwen_ip.events.unregister('post_run_cell', _old_hook)
        except Exception: pass
    _qwen_ip.events.register('post_run_cell', _qwen_kaggle_error_hook)
    _qwen_ip._qwen_kaggle_error_hook = _qwen_kaggle_error_hook
'''


CELL_5 = r'''# Fail early unless Kaggle actually attached a compatible CUDA GPU.
import torch
import shutil
from pathlib import Path

if not torch.cuda.is_available():
    raise RuntimeError('Chưa có CUDA GPU. Trong Kaggle Settings, chọn Accelerator = GPU rồi Start Session.')
gpu = torch.cuda.get_device_properties(0)
gpu_vram_gib = gpu.total_memory / (1024 ** 3)
compute_capability = (gpu.major, gpu.minor)
print(f'GPU: {gpu.name}; VRAM: {gpu_vram_gib:.1f} GiB; compute capability: {gpu.major}.{gpu.minor}')
if compute_capability < (7, 5):
    raise RuntimeError(
        'GPU này không đạt Turing (SM 7.5) trở lên cho ExLlamaV2 AWQ của v6. '
        'P100 không tương thích với kernel đã chọn; không bắt đầu dịch.'
    )
if gpu.total_memory < 14 * 1024 ** 3:
    raise RuntimeError('Qwen3-14B AWQ cần tối thiểu 14 GiB VRAM trên GPU đầu tiên; không bắt đầu dịch.')

def persist_qwen_cache(source_path):
    source_path = Path(source_path)
    if not source_path.exists():
        return
    temporary = WORKING_CACHE.with_name(WORKING_CACHE.name + '.uploading')
    shutil.copy2(source_path, temporary)
    temporary.replace(WORKING_CACHE)

print('Cache sẽ được đồng bộ định kỳ vào /kaggle/working; checkpoint:', WORKING_CACHE)
'''


CONTROL_BLOCK = r'''CHECKPOINT_EVERY_TITLES = 100
LIVE_STATUS_FILE = WORKDIR / 'qwen_live_status.json'
CONTROL_POLL_SECONDS = 5
_control_last_status_write = 0.0
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

def qwen_kaggle_progress(progress):
    global _control_last_status_write
    _qwen_health.progress(progress)
    now = time.time()
    processed = int(progress.get('processed_this_session', 0))
    if processed % 20 == 0 or now - _control_last_status_write >= 30 or bool(progress.get('last_error')):
        status = {
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_key': progress.get('model_key'),
            'phase': progress.get('phase', 'unknown'),
            'action': 'run',
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
            'checkpoint_file': WORKING_CACHE.name,
            'runtime': 'Kaggle GPU; files persisted under /kaggle/working',
            'active_bundle_sha256': ACTIVE_BUNDLE_SHA,
        }
        try:
            temp = LIVE_STATUS_FILE.with_suffix('.json.tmp')
            temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(LIVE_STATUS_FILE)
            _control_last_status_write = now
        except Exception as exc:
            print(f'[qwen-kaggle-status] write failed: {type(exc).__name__}', flush=True)
    return 'run'

def attach_durable_cache(context, enable_control=False):
    context['cache_paths'] = {'qwen3_14b_awq': LOCAL_CACHE}
    context['cache_checkpoint_interval'] = CHECKPOINT_EVERY_TITLES
    context['cache_checkpoint_callback'] = persist_qwen_cache
    context['control_callback'] = qwen_kaggle_progress
    context['control_poll_seconds'] = CONTROL_POLL_SECONDS
    return context

'''


def rewrite_notebook(notebook: dict) -> dict:
    cells = notebook.get("cells", [])
    if len(cells) < 11:
        raise RuntimeError("Unexpected v6 notebook layout; expected at least 11 cells.")

    set_cell_text(cells[0], f'''# Dịch title Cado bằng Qwen3-14B-AWQ (4-bit AWQ) trên Kaggle

Bản Kaggle tách biệt từ v6. Giữ nguyên model **`Qwen/Qwen3-14B-AWQ`**, sampling, prompt, bộ QA và quy tắc nhập cache v2; chỉ thay cách tìm đầu vào/lưu checkpoint để phù hợp Kaggle.

- Bản dịch chạy local trên CUDA GPU, không gọi API dịch và không cần HF token.
- Cần đính kèm private Dataset chứa bundle v6, cache v2 và Parquet nguồn; thêm dependency Dataset `nadsa091/qwen3-14b-awq-offline-wheelhouse` và model Dataset `raufrahim/qwen3-14b-awq`. Package, dữ liệu và model đều được nạp từ Kaggle Input; không cần tải chúng qua Internet.
- Cache v2 được đọc chỉ-đọc; bản hoàn thành/không lặp được nhập vào cache v6 riêng. Cache gốc không bị sửa.
- Run/checkpoint nằm trong `/kaggle/working/{RUN_DIR_V6}`; bản mirror được cập nhật mỗi 100 title như v6.
- Notebook từ chối GPU dưới compute capability 7.5 hoặc dưới 14 GiB VRAM. Kaggle có thể cấp P100; nếu vậy, cell GPU dừng trước khi tải model/dịch.
- Kaggle không đồng bộ Google Drive và không nhận hot-reload qua `qwen_control.json`. Dùng Kaggle Files persistence hoặc ZIP checkpoint xuất ra để tiếp tục phiên sau.

`semantic_verified` vẫn là `False`: QA kỹ thuật không thay thế đánh giá đúng nghĩa.
''')

    set_cell_text(cells[1], f'''## Chuẩn bị Kaggle

1. Tải notebook này về máy, vào Kaggle → **Create → New Notebook** → **File → Import Notebook**, rồi chọn file `.ipynb`.
2. Đính kèm private Dataset chứa bundle v6, cache v2 và Parquet nguồn; thêm dependency Datasets `nadsa091/qwen3-14b-awq-offline-wheelhouse`, `nadsa091/qwen3-14b-awq-offline-sdists` và model Dataset `raufrahim/qwen3-14b-awq` làm Input. Notebook xác minh checksum dữ liệu/model rồi nạp cả hai từ Kaggle Input.
3. Trong **Settings / Session options**, chọn **Accelerator → GPU**. Internet có thể tắt; package, Parquet và model đều có sẵn trong Input. Nếu Kaggle cấp P100, cell kiểm tra sẽ dừng; cần phiên có T4/T4x2 hoặc GPU tương thích khác.
4. Nếu giao diện có tùy chọn **Persistence**, chọn **Files only** trước khi chạy để giữ `/kaggle/working` qua lần kết nối lại. Nếu không có, cần xuất/tải ZIP checkpoint trước khi phiên mất.
5. Bấm **Start Session**, rồi chạy **Run All** trong phiên tương tác. Không dùng **Save Version → Save & Run All** để tiếp tục một checkpoint đang chạy: Kaggle mở một phiên sạch riêng cho thao tác đó.

Giữ Dataset ở chế độ **Private**: SQLite checkpoint chứa nội dung bản dịch. Không tải checkpoint lên Dataset công khai.
''')

    set_cell_text(cells[2], CELL_2)

    # Keep the package/runtime pins from v6, but give Kaggle its own lock file.
    package_cell = cell_text(cells[3])
    package_cell = package_cell.replace(
        "# Cài thư viện Transformers cho AWQ; không nâng cấp PyTorch/CUDA có sẵn trong Colab.",
        "# Cài thư viện Transformers cho AWQ; giữ nguyên PyTorch/CUDA Kaggle đang có.",
    )
    package_cell = package_cell.replace(
        "qwen3_14b_awq_colab_v2_package_lock.json",
        "qwen3_14b_awq_kaggle_package_lock.json",
    )
    package_cell = package_cell.replace("Môi trường khóa trên Drive:", "Môi trường khóa Kaggle:")
    package_cell = package_cell.replace("preserve Colab ABI", "preserve Kaggle ABI")
    package_cell = package_cell.replace("A persisted lock can outlive a Colab runtime reset", "A persisted lock can outlive a Kaggle runtime reset")
    # GPTQModel 7.6.0 requires NumPy 2.2.6 on Python 3.13. Preserve Kaggle's
    # Torch/CUDA build, but let the resolver install the NumPy version pinned by
    # GPTQModel from the verified offline wheelhouse.
    package_cell = package_cell.replace(
        "torch_constraint.write_text(f\"torch=={importlib.metadata.version('torch')}\\n\"f\"numpy=={importlib.metadata.version('numpy')}\\n\", encoding='utf-8')",
        "torch_constraint.write_text(f\"torch=={importlib.metadata.version('torch')}\\n\", encoding='utf-8')",
    )
    package_cell = replace_between(
        package_cell,
        "def _pip_install(command):",
        "\nif PACKAGE_LOCK.exists():",
        PIP_INSTALL_RETRY_BLOCK.replace(
            '            tail.append(line)\n',
            '            tail.append(line)\n            _qwen_health.installer_activity()\n',
        ),
    )
    set_cell_text(cells[3], package_cell)

    bundle_start = "BUNDLE_PATH = WORKDIR / 'title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload_bundle.zip'"
    bundle_end = "PROJECT_DIR = Path('/content/title_translation_qwen3_14b_awq_v6_sampled_source')"
    cell4 = cell_text(cells[4])
    replacement = r'''# Bundle đã được copy từ Kaggle Input vào WORKDIR ở cell chuẩn bị.
if not BUNDLE_PATH.is_file():
    raise FileNotFoundError('Thiếu bundle v6. Hãy đính kèm private Kaggle Dataset như mô tả ở đầu notebook.')

'''
    cell4 = replace_between(cell4, bundle_start, bundle_end, replacement)
    cell4 = cell4.replace(
        "print('Đã nạp pipeline từ bundle Drive; SHA-256:', ACTIVE_BUNDLE_SHA)",
        "print('Đã nạp pipeline từ Kaggle Input; SHA-256:', ACTIVE_BUNDLE_SHA)",
    )
    cell4 = cell4.replace("from huggingface_hub import hf_hub_download\n", "")
    source_start = "SOURCE = WORKDIR / Path(pipeline.DATASET_FILE).name"
    source_end = "print(f'Nguồn đã xác minh:"
    local_source_block = r'''source_name = Path(pipeline.DATASET_FILE).name
source_candidates = [
    path for path in KAGGLE_INPUT_ROOT.rglob(source_name)
    if path.is_file()
]
if len(source_candidates) != 1:
    raise FileNotFoundError(
        f'Cần đúng một Parquet nguồn trong Kaggle Input; tìm thấy {len(source_candidates)}.'
    )
SOURCE = source_candidates[0]
actual_sha = pipeline.quality.sha256_file(SOURCE)
if actual_sha.lower() != pipeline.DATASET_SHA256.lower():
    raise ValueError('SHA-256 Parquet nguồn không khớp; dừng trước khi dịch.')
'''
    cell4 = replace_between(cell4, source_start, source_end, local_source_block)
    if "hf_hub_download" in cell4:
        raise RuntimeError("The Kaggle source-data cell still contains an HF download path.")
    set_cell_text(cells[4], cell4)

    set_cell_text(cells[5], CELL_5)
    set_cell_text(cells[6], '''## Smoke test và chạy full trên Kaggle

Chạy cell bên dưới sau khi cell GPU và cài dependency hoàn tất. Smoke test vẫn giữ nguyên gate của v6. Checkpoint được mirror vào `/kaggle/working`; nếu phiên Kaggle kết thúc, kết quả gần nhất chỉ còn nếu Files persistence đang bật hoặc bạn đã xuất/tải checkpoint ZIP.

Không đổi model, prompt, sampling hay bundle khi resume. Kaggle version không theo dõi lệnh Drive và không hot-reload mã.
''')

    cell7 = cell_text(cells[7])
    cell7 = cell7.replace("_qwen_full_run_cell_marker = 'qwen-full-run-v6'", "_qwen_full_run_cell_marker = 'qwen-full-run-kaggle-v6'")
    cell7 = cell7.replace(
        "smoke_context = pipeline.prepare_pipeline(",
        LOCAL_QWEN_MODEL_SETUP + "\nsmoke_context = pipeline.prepare_pipeline(",
    )
    cell7 = replace_between(cell7, "CONTROL_FILE = WORKDIR / 'qwen_control.json'", "def _migrate_legacy_qwen_cache(context):", CONTROL_BLOCK)
    cell7 = cell7.replace(
        "legacy_workdir = DRIVE_ROOT / 'title_translation_cado_qwen3_14b_awq_full_v2'",
        "legacy_workdir = LEGACY_WORKDIR\n    if legacy_workdir is None:\n        print('Không có cache v2 input; bỏ qua nhập cache cũ.')\n        marker.write_text(json.dumps({'status': 'no_legacy_cache'}, indent=2), encoding='utf-8')\n        return",
    )
    smoke_reload_start = cell7.find("_smoke_reload_count = 0")
    smoke_reload_end = cell7.find("persist_qwen_cache(LOCAL_CACHE)", smoke_reload_start)
    if smoke_reload_start < 0 or smoke_reload_end < 0:
        raise RuntimeError("Could not remove the Colab-only smoke hot-reload loop.")
    cell7 = cell7[:smoke_reload_start] + "# Kaggle uses the reviewed bundle attached as a read-only Dataset.\n" + cell7[smoke_reload_end:]
    cell7 = cell7.replace(
        "if full_result.get('status') == 'reload_requested':\n        if _apply_drive_code_update():\n            continue\n        break",
        "if full_result.get('status') == 'reload_requested':\n        raise RuntimeError('Kaggle does not hot-reload bundles. Export the checkpoint, review a new notebook/bundle, then resume.')",
    )
    cell7 = cell7.replace('đã lưu checkpoint lên Drive', 'đã lưu checkpoint trong /kaggle/working')
    cell7 = cell7.replace('checkpoint đã lưu trên Drive', 'checkpoint đã lưu trong /kaggle/working')
    cell7 = cell7.replace('checkpoint còn trên Drive', 'checkpoint còn trong /kaggle/working')
    cell7 = cell7.replace('checkpoint đã lưu trên Drive', 'checkpoint đã lưu trong /kaggle/working')
    cell7 = cell7.replace('Chạy lại notebook để thử tiếp.', 'Chạy lại notebook để thử tiếp nếu Files persistence hoặc snapshot ZIP còn sẵn.')
    set_cell_text(cells[7], cell7)

    set_cell_text(cells[8], '''## Rà soát sau khi dịch xong

Chỉ chạy cell QA sau khi trạng thái full là `all_titles_complete`. Cell tạo CSV toàn bộ title, CSV các title cần xem lại và báo cáo tổng hợp. Cờ QA chỉ là kiểm tra kỹ thuật; không chứng minh bản dịch đúng nghĩa.
''')

    cell9 = cell_text(cells[9])
    set_cell_text(cells[9], cell9)

    set_cell_text(cells[10], f'''# Xuất checkpoint để tải hoặc đưa vào private Kaggle Dataset phiên sau.
import zipfile
from pathlib import Path
from IPython.display import FileLink, display

checkpoint_files = [
    'cache_qwen3_14b_awq.sqlite3',
    'hybrid_environment_lock.json',
    'hybrid_run_state.json',
    'legacy_cache_import.json',
    'qwen3_14b_awq_kaggle_package_lock.json',
    'qwen_live_status.json',
    'qwen_health.json',
]
checkpoint_path = Path('/kaggle/working/{CHECKPOINT_ZIP_NAME}')
with zipfile.ZipFile(checkpoint_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in checkpoint_files:
        path = WORKDIR / name
        if path.is_file():
            archive.write(path, arcname=name)
print('Checkpoint ZIP:', checkpoint_path)
print('Nếu chưa bật Files persistence: tải ZIP này trước khi đóng phiên; để resume, thêm ZIP vào private Kaggle Dataset đầu vào mới.')
display(FileLink(str(checkpoint_path)))
''')

    # Remove Colab-only metadata and old outputs (which can contain private rows).
    notebook.get("metadata", {}).pop("colab", None)
    notebook.get("metadata", {}).pop("widgets", None)
    notebook["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    # Keep the original source text intact: indenting whole cells can alter
    # multiline prompt/string literals. IPython's hook records failed cells.
    health_source = source_file('qwen_kaggle_health.py').read_text(encoding='utf-8')
    health_bootstrap = health_source + f'''
_qwen_health = QwenHealthMonitor(_HealthPath('/kaggle/working/{RUN_DIR_V6}/qwen_health.json'))
def _qwen_health_post_run(result):
    error = getattr(result, 'error_before_exec', None) or getattr(result, 'error_in_exec', None)
    if error is not None:
        _qwen_health.fail(error)
_health_ip = __import__('IPython').get_ipython()
if _health_ip is not None:
    _previous_health_hook = getattr(_health_ip, '_qwen_health_post_run', None)
    if _previous_health_hook is not None:
        try:
            _health_ip.events.unregister('post_run_cell', _previous_health_hook)
        except Exception:
            pass
    _health_ip.events.register('post_run_cell', _qwen_health_post_run)
    _health_ip._qwen_health_post_run = _qwen_health_post_run
'''
    phases = {2: 'prepare_inputs', 3: 'install_dependencies', 4: 'verify_source',
              5: 'check_gpu', 7: 'prepare_smoke', 9: 'review_qa', 10: 'export_checkpoint'}
    for index, phase in phases.items():
        source = cell_text(cells[index])
        prefix = health_bootstrap if index == 2 else ''
        prefix += f"\n_qwen_health.set_phase('{phase}', cell={index})\n"
        suffix = f"\n_qwen_health.cell_finished(final={index == 10!r})\n"
        set_cell_text(cells[index], prefix + source + suffix)

    # Syntax-check every Python code cell before writing the generated notebook.
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell_text(cell)
        ast.parse(source, filename=f"notebook-cell-{index}")
    return isolate_translation_worker(notebook)


def isolate_translation_worker(notebook: dict) -> dict:
    """Never import models in the kernel that pip has just changed NumPy in."""
    cells = notebook['cells']
    worker_source = '\n\n'.join(cell_text(cells[index]) for index in (2, 4, 5, 7, 9, 10))
    ast.parse(worker_source, filename='qwen_fresh_worker.py')
    worker_wrapper = (
        'import json as _worker_json\n'
        'try:\n'
        f'    exec(compile({worker_source!r}, "qwen_fresh_worker.py", "exec"), globals())\n'
        'except BaseException as _worker_error:\n'
        '    if "_qwen_health" in globals():\n'
        '        _qwen_health.fail(_worker_error)\n'
        '    print("[qwen-worker-error] " + _worker_json.dumps({"error_class": type(_worker_error).__name__}), flush=True)\n'
        '    raise SystemExit(1) from None\n'
    )
    launcher = f'''# Pip changed NumPy on disk; execute translation in a fresh interpreter.
import subprocess as _worker_subprocess
import sys as _worker_sys
_qwen_health.close()
_qwen_health.worker.join(timeout=5)
_parent_ip = __import__('IPython').get_ipython()
if _parent_ip is not None:
    for _hook_name in ('_qwen_health_post_run', '_qwen_kaggle_error_hook'):
        _hook = getattr(_parent_ip, _hook_name, None)
        if _hook is not None:
            try:
                _parent_ip.events.unregister('post_run_cell', _hook)
            except Exception:
                pass
_worker_path = WORKDIR / 'qwen_fresh_worker.py'
_worker_path.write_text({worker_wrapper!r}, encoding='utf-8')
print('[qwen-fresh-worker] starting clean interpreter after dependency installation', flush=True)
_worker = _worker_subprocess.Popen([_worker_sys.executable, '-u', str(_worker_path)],
    stdout=_worker_subprocess.PIPE, stderr=_worker_subprocess.STDOUT,
    text=True, encoding='utf-8', errors='replace', bufsize=1)
with (WORKDIR / 'qwen_worker.stdout.log').open('a', encoding='utf-8') as _worker_log:
    for _worker_line in _worker.stdout:
        print(_worker_line, end='', flush=True)
        _worker_log.write(_worker_line)
        _worker_log.flush()
_worker_exit = _worker.wait()
if _worker_exit:
    raise RuntimeError('Qwen worker stopped; inspect qwen_health.json and safe diagnostics.')
print('[qwen-fresh-worker] finished; output and checkpoint are under /kaggle/working', flush=True)
'''
    ast.parse(launcher, filename='fresh-worker-launcher')
    launch_cell = dict(cells[4])
    set_cell_text(launch_cell, launcher)
    notebook['cells'] = cells[:4] + [
        {'cell_type':'markdown','metadata':{},'source':[
            '## Dịch trong tiến trình Python mới\n',
            'Sau khi cài dependency, worker mới thực hiện kiểm tra dữ liệu/GPU, smoke, full, QA và xuất checkpoint. '
            'Cách này tránh trộn NumPy đã nạp trong kernel với phiên bản vừa cài. Theo dõi `[qwen-health]` và `qwen_health.json`.\n']},
        launch_cell,
    ]
    return notebook


def guide_text() -> str:
    return f'''# Qwen3-14B-AWQ Kaggle v6 — hướng dẫn chuyển và chạy

Bản này là notebook riêng cho Kaggle, được tạo từ Colab v6. Model, prompt, sampling, QA và cách nhập cache v2 được giữ nguyên. Thay đổi chỉ ở lưu trữ/runtime. Notebook gốc Colab và dữ liệu Drive không bị sửa.

## A. Tạo private Dataset đầu vào

Notebook cần bundle v6, checkpoint v2 cũ và Parquet nguồn để dùng đầu vào Kaggle cục bộ. Tải các tệp sau:

- [Bundle v6](https://drive.google.com/file/d/10IxY4bGCF1QrRtMMTIGRIGAHr66G53o1/view)
- Parquet nguồn `bilingual_zh_vi_cado.parquet` đã có trong thư mục dự án và cần được đưa vào private Kaggle Dataset.
- Tải checkpoint SQLite [cache](https://drive.google.com/file/d/1EozHEpTyYmjhqZS9VFBK1WyV7T3kcCj5/view) và [environment lock](https://drive.google.com/file/d/15VuDB17N1kNbUdla-TQzFMXGNulbSpI-/view) từ v2. Status không bắt buộc; hãy ngắt phiên Colab cũ trước khi chuyển để tránh thay đổi checkpoint đồng thời.

Trên máy, đặt bundle, Parquet nguồn, cache và environment lock vào cấu trúc sau rồi nén thư mục `qwen3_14b_awq_kaggle_input` thành ZIP:

```text
qwen3_14b_awq_kaggle_input/
├── {BUNDLE_NAME}
├── bilingual_zh_vi_cado.parquet
└── {RUN_DIR_V2}/
    ├── cache_qwen3_14b_awq.sqlite3
    └── hybrid_environment_lock.json
```

Trong Kaggle, vào **Datasets → New Dataset**, tải ZIP lên, đặt quyền **Private**, rồi tạo Dataset. Không công khai Dataset này: SQLite có chứa title và bản dịch. Kaggle tự giải nén ZIP khi gắn làm Input.

Ngoài Dataset dữ liệu, gắn thêm private dependency Dataset đã chuẩn bị sẵn: `nadsa091/qwen3-14b-awq-offline-wheelhouse`. Dataset này chỉ chứa package và manifest SHA-256 để cài dependency offline; không chứa title hay bản dịch.

## B. Import notebook và bật GPU

1. Tải `{NOTEBOOK_NAME}` từ Drive hoặc file đính kèm cùng câu trả lời về máy.
2. Kaggle → **Create → New Notebook** → **File → Import Notebook** → chọn file tải về.
3. Ở **Add Input**, thêm cả private Dataset dữ liệu vừa tạo và `nadsa091/qwen3-14b-awq-offline-wheelhouse`.
4. Trong **Settings / Session options**, chọn **Accelerator = GPU**. Internet có thể tắt vì package, Parquet và model đều nằm trong Kaggle Input. Nếu có **Persistence**, chọn **Files only** trước khi khởi chạy.
5. Bấm **Start Session**, sau đó **Run All**. Cell đầu sẽ xác nhận bundle và cache v2 đã được tìm thấy; cell GPU sẽ dừng ngay nếu không có CUDA, VRAM dưới 14 GiB hoặc compute capability thấp hơn 7.5.

T4/T4x2 có thể phù hợp. Kaggle có thể cấp P100; P100 không chạy được kernel ExLlamaV2 được chọn bởi cấu hình v6, nên notebook sẽ dừng trước khi dịch. GPU/quota/khả dụng phụ thuộc tài khoản và thời điểm.

## C. Chạy tiếp nếu Kaggle ngắt

- Nếu **Files only** persistence đang bật, giữ nguyên notebook/session để `/kaggle/working/{RUN_DIR_V6}` được mang sang lần kết nối lại; chạy lại từ đầu để nạp model và resume cache.
- Nếu persistence không bật, chạy cell cuối để tạo `{CHECKPOINT_ZIP_NAME}` rồi tải ZIP từ Output trước khi đóng phiên. Đưa ZIP này vào một version mới của private Kaggle Dataset; giữ nguyên bundle v6 và thư mục v2 ban đầu. Gắn version Dataset mới vào notebook rồi chạy lại.
- Không bấm **Save Version → Save & Run All** để “resume” khi một phiên đang chạy: thao tác đó chạy notebook từ đầu trong một session riêng. Dùng **Run All** trong session tương tác hiện tại.
- Mỗi 100 title, SQLite cache được mirror vào `/kaggle/working`. Nếu dừng cưỡng bức, có thể mất tối đa phần chưa kịp mirror kể từ checkpoint cuối.

Kaggle không mount Google Drive và bản này không hot-reload `qwen_control.json`. Nếu cần cập nhật code, dừng ở checkpoint, cập nhật notebook/bundle đã review, rồi chạy lại. Mọi output và status Kaggle vẫn ở `/kaggle/working/{RUN_DIR_V6}`; notebook không tự đồng bộ chúng ngược về Drive.

Dependency, Parquet dữ liệu và model đều được nạp từ Kaggle Input; bước chạy không cần tải file qua Hugging Face hoặc PyPI.

## D. Giới hạn hiện tại

- Kaggle Notebook có giới hạn thời lượng phiên/quota GPU; tài liệu hiện tại nêu tối đa 12 giờ cho phiên notebook và khoảng 30 giờ GPU mỗi tuần (có thể thay đổi theo nhu cầu).
- Trước khi rời phiên, xác nhận checkpoint ZIP đã xuất/tải hoặc Files persistence được bật.
- Model được nạp từ Kaggle Input ở runtime mới; Parquet cũng được đọc trực tiếp từ Kaggle Input.
- Python dependencies được cài offline từ wheelhouse; Internet có thể tắt.
- Mọi code cell có trạng thái bước bắt đầu/kết thúc và heartbeat 30 giây trong `qwen_health.json` cùng log `[qwen-health]`. Thời điểm tiến độ cuối tách riêng với heartbeat; quá ngưỡng sẽ báo `suspected_stall`, không tự dừng tiến trình. Bộ đọc local phải báo không quan sát được nếu API Kaggle không cung cấp log; không suy ra tiến độ từ `RUNNING`.

Tài liệu Kaggle: [Notebooks](https://www.kaggle.com/docs/notebooks), [Datasets](https://www.kaggle.com/docs/datasets), [GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).
'''


def build() -> tuple[Path, Path]:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(f"Missing source notebook: {SOURCE_NOTEBOOK}")
    notebook = rewrite_notebook(json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8")))
    notebook_path = ROOT / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    guide_path = ROOT / GUIDE_NAME
    guide_path.write_text(guide_text(), encoding="utf-8")
    return notebook_path, guide_path


if __name__ == "__main__":
    for output in build():
        print(output)
