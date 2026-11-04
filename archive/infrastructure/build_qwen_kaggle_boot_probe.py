"""Create a small private T4 probe before spending time installing AWQ."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import json
from pathlib import Path

ROOT = Path(__file__).parent
STAGE = Path(r'C:\Users\Lenovo\.codex\kaggle-qwen-boot-probe')
STAGE.mkdir(parents=True, exist_ok=True)
health = (ROOT/'qwen_kaggle_health.py').read_text(encoding='utf-8')
probe = r'''
import hashlib
import platform
from datetime import datetime, timezone

work = _HealthPath('/kaggle/working')
work.mkdir(parents=True, exist_ok=True)
monitor = QwenHealthMonitor(work/'qwen_health.json')
report = {'status':'starting', 'started_at_utc':datetime.now(timezone.utc).isoformat(),
          'python':platform.python_version()}
def save():
    (work/'qwen_boot_probe.json').write_text(_health_json.dumps(report), encoding='utf-8')
    print('[qwen-boot] '+_health_json.dumps(report), flush=True)
save()
try:
    with monitor.stage('probe_gpu', cell=0):
        import shutil, subprocess
        smi = shutil.which('nvidia-smi')
        report.update(nvidia_smi_available=bool(smi), nvidia_device_present=_HealthPath('/dev/nvidia0').exists())
        if smi:
            result = subprocess.run([smi, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            report['nvidia_smi_returncode'] = result.returncode
            if result.returncode == 0:
                report['hardware_gpus'] = [dict(name=parts[0].strip(), memory_mib=int(parts[1].strip()), driver=parts[2].strip()) for parts in (line.split(',') for line in result.stdout.splitlines()) if len(parts)==3]
        save()
        import torch
        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        save()
        if not torch.cuda.is_available():
            raise RuntimeError('cuda_gpu_not_assigned')
        gpu = torch.cuda.get_device_properties(0)
        report.update(gpu_name=gpu.name, gpu_vram_gib=round(gpu.total_memory/(1024**3),2),
                      compute_capability=[gpu.major,gpu.minor])
        save()
        if gpu.total_memory < 14*1024**3 or (gpu.major,gpu.minor)<(7,5):
            raise RuntimeError('gpu_not_compatible_with_awq')
    with monitor.stage('probe_source', cell=1, final=True):
        files = list(_HealthPath('/kaggle/input').rglob('bilingual_zh_vi_cado.parquet'))
        report['source_candidates'] = len(files)
        if len(files)!=1:
            raise RuntimeError('source_input_not_attached')
        digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
        report['source_sha256_matches'] = digest=='d5095f50f09d539140822b5bd41d2fba517a199ea5ba4f0204ccf50ac2cd0559'
        if not report['source_sha256_matches']:
            raise RuntimeError('source_checksum_mismatch')
        report.update(status='passed', finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
except BaseException as exc:
    report.update(status='error', error_class=type(exc).__name__)
    save()
    raise SystemExit(1) from None
finally:
    monitor.close()
'''
(STAGE/'qwen_boot_probe.py').write_text(health+probe, encoding='utf-8')
metadata=json.loads(Path(r'C:\Users\Lenovo\.codex\kaggle-qwen-v13-health\kernel-metadata.json').read_text(encoding='utf-8-sig'))
metadata.update(code_file='qwen_boot_probe.py',kernel_type='script',
                dataset_sources=['nadsa091/qwen3-14b-awq-kaggle-input'])
(STAGE/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
compile(health+probe, 'qwen_boot_probe.py', 'exec')
print(STAGE)
