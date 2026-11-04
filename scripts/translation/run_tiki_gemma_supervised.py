"""Resume the existing Tiki Gemma pipeline on a pinned source, with safe telemetry."""
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
from collections import Counter
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from tiki_translation_pipeline import (GemmaAPIClient, SQLiteTranslationCache,
    GEMMA_SYSTEM_PROMPT_VI_ZH, load_tiki_unique_titles, qa_check_vi_to_zh)


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    temp.replace(path)


class Observer:
    def __init__(self, workdir):
        self.path = workdir/'tiki_live_status.json'
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.last_progress = self.started
        self.last_print = 0.0
        self.stop = threading.Event()
        self.state = dict(status='starting', phase='prepare', pid=os.getpid(),
            direction='vi_to_zh', model='gemma-4-26b-a4b-it', api_attempts=0, api_errors=0,
            completed=0, total=0, qa_passed=0, input_tokens=0, output_tokens=0,
            quota_status='not_observed', started_at_utc=utc())
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()

    def update(self, progress=False, emit=False, **values):
        with self.lock:
            self.state.update(values)
            if progress:
                self.last_progress = time.monotonic()
            self.write(emit)

    def write(self, emit=False):
        with self.lock:
            now = time.monotonic()
            snapshot = dict(self.state, heartbeat_at_utc=utc(),
                elapsed_seconds=round(now-self.started, 1),
                last_progress_age_seconds=round(now-self.last_progress, 1))
            snapshot['health'] = ('suspected_stall' if snapshot['status']=='running'
                and now-self.last_progress > 600 else snapshot['status'])
            atomic_json(self.path, snapshot)
            if emit or now-self.last_print >= 30:
                print(json.dumps(snapshot, ensure_ascii=True), flush=True)
                self.last_print = now

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.write(True)


def acquire_lock(workdir):
    handle = (workdir/'worker.lock').open('a+b')
    handle.seek(0)
    if not handle.read(1):
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def aggregate(cache):
    with closing(sqlite3.connect(cache.db_path)) as connection:
        completed, passed = connection.execute(
            "SELECT count(*), coalesce(sum(qa_passed),0) FROM translations WHERE status='ok'"
        ).fetchone()
    return dict(completed=completed, qa_passed=passed)


def error_kind(exc):
    code = getattr(exc, 'code', None)
    message = str(exc).lower()
    if str(code) == '429' or 'resource_exhausted' in message:
        return 'quota_daily' if any(x in message for x in ('perday','per_day','daily','rpd')) else 'rate_limit'
    if str(code) in ('400','401','402','403','404','422'):
        return 'configuration_or_auth'
    return type(exc).__name__


def export(source, cache, workdir, config):
    frame = pd.read_parquet(source)
    column = config['title_column']
    results = [cache.get(str(title)) for title in frame[column]]
    if 'title_zh' in frame and frame['title_zh'].fillna('').astype(str).str.strip().ne('').any():
        frame['title_zh_source'] = frame['title_zh']
    frame['title_zh_gemma4'] = [r['title_zh'] if r and r['status']=='ok' else '' for r in results]
    frame['title_zh_gemma4_status'] = [r['status'] if r else 'pending' for r in results]
    frame['title_zh_gemma4_qa_passed'] = [bool(r and r['status']=='ok' and r['qa_passed']) for r in results]
    frame['title_zh_gemma4_qa_flags'] = [','.join(r['qa_flags']) if r else 'not_translated' for r in results]
    frame['title_zh'] = frame['title_zh_gemma4']
    destination = workdir/'tiki_vi_cado_title_gemma4_vi_zh.parquet'
    temporary = destination.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return dict(output_path=str(destination), output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                output_rows=len(frame))


def run(args):
    args.workdir.mkdir(parents=True, exist_ok=True)
    worker_lock = acquire_lock(args.workdir)
    observer = Observer(args.workdir)
    try:
        source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
        if source_sha != args.source_sha256:
            raise ValueError('source_checksum_mismatch')
        titles = load_tiki_unique_titles(args.source)
        frame = pd.read_parquet(args.source)
        column = 'title_vi_cleaned' if 'title_vi_cleaned' in frame else 'title_vi'
        if frame[column].isna().any() or any(not isinstance(t, str) or not t.strip() for t in titles):
            raise ValueError('missing_or_invalid_source_titles')
        config = dict(source_sha256=source_sha, dataset_revision=args.revision, dataset_file='data/cleaned/tiki/tiki_vi_cado.parquet',
            model='gemma-4-26b-a4b-it', direction='vi_to_zh', title_column=column, rows=len(frame), unique_titles=len(titles),
            prompt_sha256=hashlib.sha256(GEMMA_SYSTEM_PROMPT_VI_ZH.encode()).hexdigest(),
            temperature=0.0, max_output_tokens=256, thinking_level='minimal',
            pipeline_sha256=hashlib.sha256(Path(__import__('tiki_translation_pipeline').__file__).read_bytes()).hexdigest())
        config_path = args.workdir/'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('run_configuration_changed')
        atomic_json(config_path, config)
        cache = SQLiteTranslationCache(args.workdir/'cache_tiki_gemma4.sqlite3', 'gemma4')
        client = GemmaAPIClient(timeout_ms=120000)
        deferred = bool(getattr(args, 'defer_full_quality_review', False))
        observer.update(progress=True, emit=True, status='running', total=len(titles),
                        quality_review_deferred=deferred, **aggregate(cache))
        gates = {}
        consecutive_errors = 0
        next_request = 0.0
        request_interval = args.delay
        successful_requests = 0
        for phase, target in [('smoke',titles[:10]), ('pilot',titles[:50]), ('full',titles)]:
            observer.update(progress=True, emit=True, phase=phase, stage_total=len(target), stage_processed=0)
            for index, title in enumerate(target, 1):
                cached = cache.get(title)
                if cached and cached['status']=='ok':
                    continue
                record = None
                for retry in range(3):
                    time.sleep(max(0, next_request-time.monotonic()))
                    next_request = time.monotonic() + request_interval
                    observer.update(request_in_flight=True, stage_processed=index-1,
                                    api_attempts=observer.state['api_attempts']+1)
                    started = time.monotonic()
                    try:
                        record = client.translate_record(title)
                        consecutive_errors = 0
                        successful_requests += 1
                        if successful_requests >= 20:
                            request_interval = max(args.delay, request_interval * 0.9)
                            successful_requests = 0
                        break
                    except Exception as exc:
                        kind = error_kind(exc)
                        if kind == 'rate_limit':
                            request_interval = min(60.0, max(4.0, request_interval * 2))
                            next_request = time.monotonic() + request_interval
                            successful_requests = 0
                        observer.update(api_errors=observer.state['api_errors']+1, last_error_class=type(exc).__name__,
                                        last_error_kind=kind, request_in_flight=False)
                        consecutive_errors += 1
                        if kind in ('quota_daily', 'configuration_or_auth'):
                            observer.update(emit=True, status='paused_'+kind, quota_status=kind)
                            return
                        if consecutive_errors >= 5:
                            observer.update(emit=True, status='paused_api_errors')
                            return
                        if retry < 2:
                            time.sleep(min(60, 5*(2**retry)))
                if record is None:
                    record = dict(title_zh='', status='error', input_tokens=0, output_tokens=0)
                qa = qa_check_vi_to_zh(title, record['title_zh'])
                if record['status']=='truncated':
                    qa['passed'] = False
                    qa['flags'] = sorted(set(qa['flags'] + ['bi_cat_cut']))
                cache.put(title, record['title_zh'], record['status'], qa, elapsed=time.monotonic()-started)
                observer.update(progress=True, request_in_flight=False, stage_processed=index,
                    input_tokens=observer.state['input_tokens']+record['input_tokens'],
                    output_tokens=observer.state['output_tokens']+record['output_tokens'], **aggregate(cache))
            records = [cache.get(title) for title in target]
            completed = sum(bool(r and r['status']=='ok') for r in records)
            passed = sum(bool(r and r['status']=='ok' and r['qa_passed']) for r in records)
            gate = dict(requested=len(target), completed=completed, qa_passed=passed,
                        completion_rate=completed/max(1,len(target)), technical_pass_rate=passed/max(1,len(target)))
            gate['passed'] = gate['completion_rate']>=0.99 and gate['technical_pass_rate']>=0.90
            gates[phase] = gate
            atomic_json(args.workdir/'stage_reports.json', gates)
            if phase != 'full' and (gate['completion_rate'] < 0.99 or (not gate['passed'] and not deferred)):
                observer.update(emit=True, status='quality_gate_failed', gate=gate)
                return
        flags = Counter()
        with closing(sqlite3.connect(cache.db_path)) as connection:
            for (value,) in connection.execute("SELECT qa_flags FROM translations WHERE qa_passed=0"):
                flags.update(flag.split(':',1)[0] for flag in json.loads(value or '[]'))
        output = export(args.source, cache, args.workdir, config)
        report = dict(config=config, stages=gates, **aggregate(cache), total=len(titles),
                      quality_review_passed=gates['full']['passed'], flag_categories=dict(flags),
                      quality_review_deferred=deferred, semantic_verified=False, finished_at_utc=utc(), **output)
        atomic_json(args.workdir/'tiki_gemma4_report.json', report)
        observer.update(progress=True, emit=True, phase='finished',
            status='completed' if report['completed']==len(titles) else 'incomplete',
            quality_review_passed=report['quality_review_passed'], **output)
    except BaseException as exc:
        observer.update(emit=True, status='error', last_error_class=type(exc).__name__)
        raise SystemExit(1) from None
    finally:
        observer.stop.set()
        observer.worker.join(timeout=2)
        worker_lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--delay', type=float, default=4.0)
    parser.add_argument('--defer-full-quality-review', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
