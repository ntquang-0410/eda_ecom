import pandas as pd
import pytest
from scripts.audit.audit_release import COLUMNS, audit_files, inspect_frame, sha256

def frame():
    return pd.DataFrame([['1', '1688', 'home', '杯子 2', 'Cốc 2', None, None, 'gemma', 'gemma: 0 | qwen: 0']], columns=COLUMNS)

def test_cross_source_ids_are_not_duplicates():
    a = frame()
    b = frame(); b['source'] = 'tiki'; b['title_zh'] = '杯子 2'; b['title_vi'] = 'Cốc 2'
    report, _ = inspect_frame(pd.concat([a, b]))
    assert 'duplicate_source_product_id' not in report['structural_errors']

def test_composite_duplicate_is_structural_error():
    report, _ = inspect_frame(pd.concat([frame(), frame()]))
    assert 'duplicate_source_product_id' in report['structural_errors']

def test_language_signals_do_not_claim_semantic_failure():
    a = frame(); a['title_vi'] = 'Cốc 川 2'
    report, flags = inspect_frame(a)
    assert report['structural_errors'] == [] and report['semantic_verified'] is False
    assert flags[0]['signal'] == 'han_remaining_review'
    assert 'target_text' not in flags[0]

def test_missing_schema_is_reported_without_keyerror():
    report, flags = inspect_frame(pd.DataFrame({'product_id': ['1']}))
    assert report['missing_columns'] and not flags

def test_nullable_descriptions_are_not_fabricated():
    report, _ = inspect_frame(frame())
    assert report['structural_errors'] == [] and report['description_nulls']['description_vi'] == 1

def test_file_audit_preserves_inputs_and_refuses_overwrite(tmp_path):
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    a, b = inputs / 'cum.parquet', inputs / 'pair.parquet'
    frame().to_parquet(a, index=False); frame().to_parquet(b, index=False)
    before = sha256(a), sha256(b)
    report = audit_files(a, b, tmp_path / 'review', 'test-revision')
    assert report['cross_method'] == {'same_product_order': True, 'same_titles': True}
    assert before == (sha256(a), sha256(b))
    with pytest.raises(FileExistsError):
        audit_files(a, b, tmp_path / 'review', 'test-revision')
