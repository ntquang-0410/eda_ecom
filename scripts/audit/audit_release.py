"""Read-only, local audit of the nine-column judged release; flags are not verdicts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

COLUMNS = ['product_id', 'source', 'category', 'title_zh', 'title_vi',
           'description_zh', 'description_vi', 'judge_chon_title', 'judge_chon_description']
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
VI = re.compile(r'[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]', re.I)
NUMBERS = re.compile(r'\d+(?:[.,]\d+)*')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def text(value):
    return '' if pd.isna(value) else str(value)

def inspect_frame(frame):
    """Do not rewrite data or treat language/number differences as confirmed errors."""
    structural = []
    if list(frame.columns) != COLUMNS:
        structural.append('unexpected_nine_column_schema')
    missing = sorted(set(COLUMNS) - set(frame.columns))
    if missing:
        return {'rows': len(frame), 'structural_errors': structural, 'missing_columns': missing}, []
    if frame[['source', 'product_id']].duplicated().any():
        structural.append('duplicate_source_product_id')
    required = ['product_id', 'source', 'category', 'title_zh', 'title_vi', 'judge_chon_title']
    blank = {c: int(frame[c].map(text).str.strip().eq('').sum()) for c in required}
    if any(blank.values()):
        structural.append('blank_required_fields')
    if not frame.source.isin(['1688', 'tiki']).all():
        structural.append('unknown_source')
    if not frame.judge_chon_title.isin(['gemma', 'qwen']).all():
        structural.append('unknown_title_draft_model')
    flags = []
    for row in frame.to_dict('records'):
        src, pid = text(row['source']), text(row['product_id'])
        for field in ['title_zh', 'title_vi', 'description_zh', 'description_vi']:
            value = text(row[field])
            codes = []
            if '\ufffd' in value or any(unicodedata.category(c) == 'Cc' and c not in '\r\n\t' for c in value):
                codes.append('invalid_unicode_review')
            if re.search(r'(.)\1{12,}', value):
                codes.append('repeated_character_review')
            if re.search(r'<\s*/?think\s*>|<\|im_(?:start|end)\|>|```', value, re.I):
                codes.append('model_format_marker_review')
            if len(value) > (512 if field.startswith('title') else 12000):
                codes.append('long_text_review')
            for code in codes:
                flags.append({'source': src, 'product_id': pid, 'field': field, 'signal': code})
        source_field, target_field = ('title_zh', 'title_vi') if src == '1688' else ('title_vi', 'title_zh')
        source, target = text(row[source_field]), text(row[target_field])
        if src == '1688' and HAN.search(target):
            flags.append({'source': src, 'product_id': pid, 'field': target_field, 'signal': 'han_remaining_review'})
        if src == 'tiki' and VI.search(target):
            flags.append({'source': src, 'product_id': pid, 'field': target_field, 'signal': 'vietnamese_remaining_review'})
        if set(NUMBERS.findall(source)) != set(NUMBERS.findall(target)):
            flags.append({'source': src, 'product_id': pid, 'field': target_field, 'signal': 'numeric_tokens_differ_review'})
    return {'rows': len(frame), 'structural_errors': structural, 'blank_required': blank,
            'source_counts': frame.source.astype('string').fillna('<NULL>').value_counts().to_dict(),
            'description_nulls': {c: int(frame[c].isna().sum()) for c in ['description_zh', 'description_vi']},
            'signals': dict(Counter(f['signal'] for f in flags)),
            'semantic_verified': False}, flags

def audit_files(cum, pair, output_dir, revision):
    cum, pair, output_dir = Path(cum).resolve(), Path(pair).resolve(), Path(output_dir).resolve()
    if cum == pair or output_dir in (cum.parent, pair.parent):
        raise ValueError('Use two distinct inputs and a separate audit output directory')
    report_path, flags_path = output_dir / 'audit_report.json', output_dir / 'review_signals.parquet'
    if report_path.exists() or flags_path.exists():
        raise FileExistsError('Audit artifacts already exist; choose a new output directory')
    frames = {'cum': pd.read_parquet(cum), 'pair': pd.read_parquet(pair)}
    paths = {'cum': cum, 'pair': pair}
    report = {'input_revision': revision, 'read_only': True, 'semantic_verified': False,
              'limitations': ['Technical signals only; no exhaustive semantic judgment.',
                              'Number-token differences need unit- and attribute-level review.',
                              'Proper nouns and separators may trigger language signals.'], 'files': {}}
    signals = []
    for method, frame in frames.items():
        summary, flags = inspect_frame(frame)
        summary['sha256'] = sha256(paths[method])
        report['files'][method] = summary
        signals.extend(dict(method=method, **f) for f in flags)
    keys = ['source', 'product_id']
    if all(set(keys + ['title_zh', 'title_vi']) <= set(f.columns) for f in frames.values()):
        report['cross_method'] = {
            'same_product_order': frames['cum'][keys].equals(frames['pair'][keys]),
            'same_titles': frames['cum'][keys + ['title_zh', 'title_vi']].equals(frames['pair'][keys + ['title_zh', 'title_vi']]),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(signals, columns=['method', 'source', 'product_id', 'field', 'signal']).to_parquet(flags_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cum', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--revision', required=True, help='Input HF revision or local snapshot label')
    args = parser.parse_args()
    report = audit_files(args.cum, args.pair, args.out, args.revision)
    print(json.dumps({'status': 'audit_written', 'rows': {m: v['rows'] for m, v in report['files'].items()},
                      'structural_errors': {m: v['structural_errors'] for m, v in report['files'].items()},
                      'semantic_verified': False}))
    if any(f['structural_errors'] for f in report['files'].values()) or not all(report.get('cross_method', {}).values()):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
