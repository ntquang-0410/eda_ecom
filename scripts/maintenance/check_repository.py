"""Offline syntax, credential-pattern, notebook-output and inventory verification."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = [
    re.compile(r'hf_[A-Za-z0-9]{20,}'),
    re.compile(r'sk-[A-Za-z0-9_-]{20,}'),
    re.compile(r'AIza[A-Za-z0-9_-]{25,}'),
    re.compile(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
]
EXCLUDED = {'.git', 'outputs', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', 'data'}
TEXT = {'.py', '.md', '.json', '.toml', '.yml', '.yaml', '.ps1', '.txt', '.ipynb'}

def inspect():
    errors = []
    checked = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in EXCLUDED or p.startswith('.venv') for p in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in TEXT:
            continue
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_text(encoding='utf-8-sig')
        checked += 1
        if any(pattern.search(content) for pattern in PATTERNS):
            errors.append({'path': relative, 'error': 'credential_pattern'})
        if path.suffix == '.py':
            try:
                compile(content, relative, 'exec')
            except SyntaxError as exc:
                errors.append({'path': relative, 'error': 'syntax', 'line': exc.lineno})
        if path.suffix == '.ipynb':
            notebook = json.loads(content)
            if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook['cells'] if c.get('cell_type') == 'code'):
                errors.append({'path': relative, 'error': 'saved_notebook_output'})
    inventory = json.loads((ROOT / 'docs/source_inventory.json').read_text(encoding='utf-8'))
    for entry in inventory['files']:
        path = ROOT / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['published_sha256']:
            errors.append({'path': entry['path'], 'error': 'inventory_checksum'})
    return {'checked_text_files': checked, 'migrated_source_files': len(inventory['files']), 'errors': errors}

if __name__ == '__main__':
    result = inspect()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(bool(result['errors']))
