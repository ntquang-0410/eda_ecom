"""Prepare the same reviewed worker for an editable interactive session."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).parent

def extract_worker(notebook):
    launcher=ast.parse(''.join(notebook['cells'][-1]['source']))
    wrapper=next(node.value for node in ast.walk(launcher) if isinstance(node,ast.Constant)
        and isinstance(node.value,str) and node.value.startswith('import json as _worker_json'))
    return next(node.args[0].value for node in ast.walk(ast.parse(wrapper))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='compile')


def prepare(source):
    tree=ast.parse(source)
    assignments=[n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='smoke_result' for t in n.targets)]
    if len(assignments)!=1:
        raise RuntimeError('Unexpected smoke call layout')
    node=assignments[0]
    original=ast.get_source_segment(source,node)
    reuse='''_live_original_loader = pipeline._load_qwen
_live_model_cache = {}
def _live_cached_loader(revision, hf_token, device_index=0):
    key = (revision, device_index)
    if key not in _live_model_cache:
        _live_model_cache[key] = _live_original_loader(revision, hf_token, device_index)
    return _live_model_cache[key]
pipeline._load_qwen = _live_cached_loader
'''
    retry = reuse + '\nfor _live_smoke_retry in range(1, 6):\n' + textwrap.indent(original,'    ') + '''
    persist_qwen_cache(LOCAL_CACHE)
    if float(smoke_result.get('completion_rate', 0) or 0) >= 0.99:
        break
    print('[qwen-smoke-retry] ' + json.dumps({'attempt':_live_smoke_retry,
        'completed':smoke_result.get('completed'), 'requested':smoke_result.get('requested')}), flush=True)
'''
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=[retry+'\n']
    body=''.join(lines)
    ast.parse(body)
    # Only unfinished smoke rows are retried; successful rows stay cached.
    wrapper=('import json as _live_json\ntry:\n'
        f'    exec(compile({body!r}, "qwen_live_worker.py", "exec"), globals())\n'
        'except BaseException as _live_error:\n'
        '    if "_qwen_health" in globals():\n'
        '        _qwen_health.fail(_live_error)\n'
        '    print("[qwen-worker-error] "+_live_json.dumps({"error_class":type(_live_error).__name__}),flush=True)\n'
        '    raise SystemExit(1) from None\n')
    return wrapper


def build():
    notebook=json.loads((ROOT/'title_translation_qwen3_14b_awq_kaggle_v6.ipynb').read_text(encoding='utf-8'))
    path=ROOT/'qwen_live_worker.py'
    path.write_text(prepare(extract_worker(notebook)),encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
