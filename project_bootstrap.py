"""Compatibility paths for the original flat modules after source organization."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
MODULE_DIRS = ('src', 'scripts/translation', 'scripts/comparison', 'scripts/audit',
    'scripts/datasets', 'scripts/notebooks', 'scripts/monitoring', 'scripts/merge',
    'scripts/judge', 'scripts/ft_clean', 'archive/infrastructure')

def configure_paths():
    """Configure imports without reading data, loading models or accessing a network."""
    for relative in reversed(MODULE_DIRS):
        directory = str(ROOT / relative)
        if directory not in sys.path:
            sys.path.insert(0, directory)

def source_file(name):
    """Locate source by basename for builders that embed self-contained workers."""
    if Path(name).name != name:
        raise ValueError('Source lookup accepts a basename only')
    candidates = [ROOT / d / name for d in MODULE_DIRS]
    candidates += [ROOT / 'notebooks/templates' / name]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one source/template for {name}; found {len(found)}')
    return found[0]

def generated_root():
    """Keep generated notebooks/bundles outside versioned source directories."""
    root = Path(os.environ.get('ECOM_GENERATED_DIR') or ROOT / 'outputs/generated')
    root.mkdir(parents=True, exist_ok=True)
    return root
