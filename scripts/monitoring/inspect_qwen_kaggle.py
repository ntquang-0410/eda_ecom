"""Read bounded Kaggle live logs and print only safe progress metadata."""
from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import argparse
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def bounded_log_events(api, kernel, seconds):
    from kagglesdk import KaggleEnv
    owner, slug, version = api.parse_kernel_string(kernel)
    with api.build_kaggle_client() as client:
        http = client._http_client
        http._init_session()
        base = http._endpoint if http._env == KaggleEnv.PROD else http._endpoint+'/api'
        headers = dict(http._session.headers)
        headers['Accept'] = 'text/event-stream, */*'
        headers.pop('Content-Type', None)
        params = {'waitForLogsUrlSeconds': 5}
        if version:
            params['versionLabel'] = api._kernel_version_label(version)
        response = http._session.get(base+'/v1/kernels/logs/stream/'+owner+'/'+slug,
            params=params, stream=True, headers=headers, auth=http._session.auth,
            timeout=(10, max(10, seconds)))
        response.raise_for_status()
        content_type = (response.headers.get('Content-Type') or '').lower()
        yield {'observer_transport': {'http_status':response.status_code,
               'content_type':content_type, 'live_stream_open':content_type.startswith('text/event-stream')}}
        try:
            if content_type.startswith('text/event-stream'):
                # A short observation must not wait for requests' 512-byte chunks.
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line or not line.startswith('data:'):
                        continue
                    payload = line[5:].lstrip()
                    if payload == 'END_OF_LOG':
                        return
                    try:
                        yield json.loads(payload)
                    except ValueError:
                        yield {'data':payload}
            else:
                yield from api._iter_blob_lines(response)
        finally:
            response.close()


def main():
    parser = argparse.ArgumentParser()
    active_path = Path(__file__).parent/'qwen_active_run.json'
    active = json.loads(active_path.read_text(encoding='utf-8')) if active_path.is_file() else {}
    parser.add_argument('--kernel')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.kernel and active.get('transport')=='interactive_jupyter':
        status_path=Path(active['controller_status_file'])
        if status_path.is_file():
            from report_tiki_gemma_status import process_alive
            summary=json.loads(status_path.read_text(encoding='utf-8'))
            summary['controller_alive']=process_alive(summary.get('controller_pid',0))
            text=json.dumps(summary,ensure_ascii=True)
            if args.output:
                args.output.write_text(text+'\n',encoding='utf-8')
            print(text,flush=True)
            return
    args.kernel=args.kernel or active.get('kernel','nadsa091/qwen3-14b-awq-kaggle-v6/18')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    status = api.kernels_status(args.kernel)
    summary = {'kernel': args.kernel, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
               'kernel_status': str(getattr(status, 'status', 'unknown')),
               'events_seen': 0, 'stage': 'unknown', 'live_log_state': 'connecting',
               'processed_this_session': None, 'completed_this_session': None,
               'errors_this_session': None, 'total_this_session': None}
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        owner, slug, version = api.parse_kernel_string(args.kernel)
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        if version:
            request.version_label = api._kernel_version_label(version)
        with api.build_kaggle_client() as client:
            metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
        started = metadata.last_run_time
        if started:
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            # Kaggle metadata last_run_time is not the active worker start time.
            summary['metadata_last_run_at_utc'] = started.isoformat()
    except Exception as exc:
        summary['metadata_reader_error_class'] = type(exc).__name__
    lock = threading.Lock()
    finished = threading.Event()
    def consume():
        try:
            for event in bounded_log_events(api, args.kernel, args.seconds):
                if 'observer_transport' in event:
                    with lock:
                        summary['log_transport'] = event['observer_transport']
                    continue
                data = str(event.get('data', ''))
                with lock:
                    for line in data.splitlines():
                        try:
                            progress = json.loads(line.strip())
                            if isinstance(progress, dict) and progress.get('phase') == 'scoring' and 'completed_pairs' in progress:
                                summary['comet_progress'] = {k: progress[k] for k in ('phase','completed_pairs','total_pairs','heartbeat_at_utc') if k in progress}
                                summary['stage'] = 'scoring'
                        except (ValueError, TypeError):
                            pass
                    summary['events_seen'] += 1
                    summary['live_log_state'] = 'receiving'
                    summary['last_log_elapsed_seconds'] = event.get('time')
                    summary['last_event_received_at_utc'] = datetime.now(timezone.utc).isoformat()
                    markers = [('Nguồn đã xác minh', 'source_verified'),
                               ('Verified all 12 files', 'model_snapshot_verified'),
                               ('Loading weights', 'loading_weights'),
                               ('ExLlamaV2 AWQ: compiling', 'compiling_awq_kernel'),
                               ('ExLlamaV2 AWQ: torch.ops JIT extension ready', 'awq_kernel_ready')]
                    for marker, stage in markers:
                        if marker in data:
                            summary['stage'] = stage
                    match = re.search(r'\[qwen3_14b_awq\]\s+uncached\s+(\d+)/(\d+);\s*complete=(\d+);\s*errors=(\d+)', data)
                    if match:
                        summary.update(stage='generation', processed_this_session=int(match[1]),
                                       total_this_session=int(match[2]), completed_this_session=int(match[3]),
                                       errors_this_session=int(match[4]))
                    if '[batch-cache] ' in data:
                        try:
                            cache = json.loads(data.split('[batch-cache] ', 1)[1].strip())
                            allowed = {'cache','completed','cached_records','requested'}
                            filtered = {k:v for k,v in cache.items() if k in allowed}
                            name = filtered.get('cache')
                            by_name = summary.setdefault('cache_progress_by_name', {})
                            previous = by_name.get(name, {})
                            if filtered.get('completed', 0) >= previous.get('completed', 0):
                                by_name[name] = filtered
                            preferred = by_name.get('cache_qwen3_14b_awq.sqlite3')
                            summary['cache_progress'] = preferred or filtered
                        except (ValueError, TypeError):
                            pass
                    if '[batch-health] ' in data:
                        try:
                            batch = json.loads(data.split('[batch-health] ', 1)[1].strip())
                            allowed = {'phase','state','status','completed','total','qa_passed',
                                       'api_attempts','api_errors','quota_status','heartbeat_at_utc',
                                       'last_progress_age_seconds','processed','session_total',
                                       'session_completed','session_errors','processed_this_session',
                                       'completed_this_session','errors_this_session'}
                            summary['batch_health'] = {k:v for k,v in batch.items() if k in allowed}
                        except (ValueError, TypeError):
                            pass
                    if '[qwen-health]' in data:
                        try:
                            health = json.loads(data.split('[qwen-health]', 1)[1].strip())
                            allowed = {'phase','cell','state','heartbeat_at_utc','last_progress_at_utc',
                                       'progress_age_seconds','processed_this_session','completed_this_session',
                                       'errors_this_session','total_this_session','gpu_utilization_percent',
                                       'gpu_memory_used_mib','elapsed_seconds','progress_unit','progress_done','progress_total'}
                            summary['health'] = {k:v for k,v in health.items() if k in allowed}
                            summary['stage'] = health.get('phase', summary['stage'])
                        except (ValueError, TypeError):
                            pass
                    if '[qwen-boot]' in data:
                        try:
                            boot = json.loads(data.split('[qwen-boot]',1)[1].strip())
                            summary['gpu_preflight'] = {k:v for k,v in boot.items() if k in
                                ('torch','cuda_available','nvidia_device_present','nvidia_smi_available')}
                        except (ValueError, TypeError):
                            pass
                    names = re.findall(r'\b(?:LocalEntryNotFoundError|ConnectError|FileNotFoundError|OutOfMemoryError|ImportError|ModuleNotFoundError)\b', data)
                    if names:
                        summary['diagnostic_classes'] = sorted(set(summary.get('diagnostic_classes', []) + names))
            with lock:
                summary['live_log_state'] = 'stream_finished'
        except Exception as exc:
            with lock:
                summary['live_log_state'] = 'unavailable'
                summary['log_reader_error_class'] = type(exc).__name__
        finally:
            finished.set()
    threading.Thread(target=consume, daemon=True).start()
    finished.wait(args.seconds)
    with lock:
        if not summary['events_seen']:
            previous = {}
            if args.output and args.output.is_file():
                try:
                    previous = json.loads(args.output.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    pass
            first = (previous.get('first_unobservable_at_utc')
                     if previous.get('kernel') == args.kernel else None) or summary['checked_at_utc']
            summary['first_unobservable_at_utc'] = first
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(first)).total_seconds()
            summary['telemetry_unavailable_seconds'] = round(max(0, age), 1)
            summary['live_log_state'] = 'no_events_within_observation_window'
            summary['progress_observability'] = 'unavailable; kernel status alone does not prove progress or a hang'
            summary['health_assessment'] = 'unobservable_requires_review' if age > 900 else 'unobservable'
        if summary.get('batch_health'):
            summary['stage'] = summary['batch_health'].get('phase', summary['stage'])
        text = json.dumps(summary, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
