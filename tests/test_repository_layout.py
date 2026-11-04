from pathlib import Path
import json
import subprocess
import sys
import importlib.util

from project_bootstrap import ROOT, source_file

def test_original_source_names_resolve_to_reorganized_locations():
    assert source_file('title_translation_pipeline.py') == ROOT / 'src/title_translation_pipeline.py'
    assert source_file('compare_1688_cometkiwi.py') == ROOT / 'scripts/comparison/compare_1688_cometkiwi.py'

def test_listing_does_not_require_credentials_or_load_a_model():
    result = subprocess.run([sys.executable, str(ROOT / 'manage.py'), 'list'], capture_output=True, text=True, check=True)
    assert 'audit-release' in result.stdout and 'judge' in result.stdout

def test_subcommand_help_reaches_the_underlying_parser():
    result = subprocess.run([sys.executable, str(ROOT / 'manage.py'), 'audit-release', '--help'], capture_output=True, text=True, check=True)
    assert '--cum' in result.stdout and '--pair' in result.stdout

def test_notebook_templates_have_no_execution_results():
    for path in (ROOT / 'notebooks').rglob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        assert all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells'] if c.get('cell_type') == 'code')

def test_frozen_policies_remain_explicit():
    strict = source_file('compare_1688_cometkiwi.py').read_text(encoding='utf-8')
    original_export = (ROOT / 'archive/release_history/export_comet_best_without_floor.py').read_text(encoding='utf-8')
    assert '.72' in strict or '0.72' in strict
    assert 'higher_cometkiwi_no_floor_qa_advisory_only' in original_export

def test_one_off_dataset_programs_do_not_execute_on_import():
    paths = list((ROOT / 'scripts/datasets').glob('*.py'))
    paths += [ROOT / 'scripts/audit' / n for n in ('label_title_review_flags.py', 'repair_1688_cjk.py', 'audit_bidirectional_titles.py')]
    for i, path in enumerate(paths):
        spec = importlib.util.spec_from_file_location(f'one_off_{i}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
