"""Reconstruct final context attribute provenance without changing translations."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import sys

import pandas as pd

ROOT = Path(r'D:\download\NCKH\judge_kaggle_handoff')
WORK = Path(r'D:\download\NCKH\judge_kaggle_preflight')
OUT = WORK / 'cum_description_provenance_update'
sys.path.insert(0, str(ROOT / 'scripts/judge'))
import ghep_bo as G

LABELS = ('gemma', 'qwen', 'quy_tac', 'chua_ro')


def norm(value):
    # Preserve case and punctuation: these can change attribute meaning.
    return '' if pd.isna(value) else str(value).strip()


def unit_origin(row, actual):
    if norm(row['ban']) != norm(actual):
        return 'chua_ro', 'upstream_selected_text_does_not_match'
    tag = norm(row['tu'])
    if tag == 'co_dinh':
        return 'quy_tac', 'upstream_explicit_fixed_rule'
    if tag in ('A', 'B'):
        return ('gemma' if tag == 'A' else 'qwen'), 'upstream_explicit_selected_model'
    # Identical candidates alone do not prove which model supplied final text.
    return 'chua_ro', 'upstream_missing_selected_model'


def resolve_unit(units, kind, source, category, actual):
    candidates = units.get((kind, norm(source), norm(category)), [])
    evidence = [unit_origin(u, actual) for u in candidates if norm(u['ban']) == norm(actual)]
    labels = {x[0] for x in evidence}
    if len(labels) == 1:
        label = next(iter(labels))
        return label, evidence[0][1]
    return 'chua_ro', 'conflicting_upstream_records' if labels else 'no_matching_upstream_selected_text'


def component_origin(row, part, suffix, actual, source_text, units):
    if not norm(actual):
        return 'empty', 'empty_component'
    if norm(row.get(f'{part}_cuoi_{suffix}')) != norm(actual):
        return 'chua_ro', 'context_provenance_text_does_not_match'
    tag = norm(row.get(f'{part}_nguon_{suffix}'))
    if tag == 'khoa':
        return 'quy_tac', 'context_explicit_fixed_rule'
    if tag in ('sua_A', 'dich_moi_A'):
        return 'gemma', 'context_explicit_gemma_generation'
    if tag in ('sua_B', 'dich_moi_B'):
        return 'qwen', 'context_explicit_qwen_generation'
    inherited = norm(row.get(f'{part}_nhap'))
    if norm(actual) != inherited:
        return 'chua_ro', 'no_matching_inherited_text_or_explicit_model'
    return resolve_unit(units, 'ten' if part == 'n' else 'gia_tri', source_text, row['category'], actual)


def attribute_origin(name_origin, value_origin):
    origins = {x for x in (name_origin, value_origin) if x != 'empty'}
    if not origins or 'chua_ro' in origins:
        return 'chua_ro'
    models = origins & {'gemma', 'qwen'}
    if len(models) == 1:
        return next(iter(models))
    if len(models) == 2:
        # One attribute is counted once, never duplicated across two models.
        return 'chua_ro'
    return 'quy_tac'


def display(counts):
    return ' | '.join(f'{k}: {int(counts.get(k, 0))}' for k in LABELS if counts.get(k, 0) > 0)


def self_test():
    assert attribute_origin('quy_tac', 'gemma') == 'gemma'
    assert attribute_origin('quy_tac', 'qwen') == 'qwen'
    assert attribute_origin('gemma', 'qwen') == 'chua_ro'
    assert attribute_origin('quy_tac', 'quy_tac') == 'quy_tac'
    assert attribute_origin('gemma', 'chua_ro') == 'chua_ro'
    assert unit_origin({'ban': 'same', 'tu': None}, 'same')[0] == 'chua_ro'
    assert unit_origin({'ban': 'same', 'tu': 'A'}, 'same')[0] == 'gemma'
    assert unit_origin({'ban': 'same', 'tu': 'B'}, 'different')[0] == 'chua_ro'
    assert display({'gemma': 3, 'qwen': 0, 'quy_tac': 1}) == 'gemma: 3 | quy_tac: 1'


def main():
    self_test()
    OUT.mkdir(exist_ok=True)
    old = pd.read_parquet(WORK / 'cum_9col_judge_update/bo_cum_chon_deepseek.parquet')
    old['product_id'] = old.product_id.astype(str)
    accepted_trace = pd.read_parquet(WORK / 'cum_9col_judge_update/bo_cum_fragment_selection_trace.parquet')
    accepted = {(r.source, str(r.product_id), str(r.item_id)): r.selected_model
                for r in accepted_trace.itertuples() if r.accepted}
    checkpoint = WORK / 'title_v1_output/judge_full/data/processed/so_cap'
    results, equivalent = G.doc_ket_qua_va_tuong_duong('lon_cum', thu_muc=checkpoint)
    decisions, _ = G.ap_chinh_sach_tuong_duong(results, equivalent, 'giu_hien_tai')
    for key, model in accepted.items():
        if key[2] not in decisions:
            decisions[key[2]] = model
    base = pd.read_parquet(G.BASE_CUM)
    base['product_id'] = base.product_id.astype(str)
    pairs = G.nap_cap_cum()
    raw = {
        '1688': pd.read_parquet(G.D4 / 'cap_cum_20261026_2123.parquet'),
        'tiki': pd.read_parquet(G.D3 / 'so_sanh_cum_20261024_1616.parquet'),
    }
    unit_paths = {
        '1688': G.D1 / 'don_vi_da_chon_20261022_2126.parquet',
        'tiki': G.D2 / 'don_vi_da_chon_20261023_1656.parquet',
    }
    traces = []
    inputs = [G.BASE_CUM, G.D4 / 'cap_cum_20261026_2123.parquet',
              G.D3 / 'so_sanh_cum_20261024_1616.parquet', G.D3 / 'cap_cum_20261024_1616.parquet',
              checkpoint / 'ket_qua_lon_cum.jsonl', *unit_paths.values()]
    for src in ('1688', 'tiki'):
        units = defaultdict(list)
        u = pd.read_parquet(unit_paths[src])
        textcol = 'van_ban_zh' if src == '1688' else 'van_ban_nguon'
        for r in u.to_dict('records'):
            units[(r['loai'], norm(r[textcol]), norm(r['category']))].append(r)
        full = raw[src]
        assert not full.duplicated(['product_id', 'pair_id']).any()
        full['product_id'] = full.product_id.astype(str)
        lookup = {(r['product_id'], str(r['pair_id'])): r for r in full.to_dict('records')}
        b = base[base.source.eq(src)]
        p = pairs[src].copy()
        p['product_id'] = p.product_id.astype(str)
        p = p[p.product_id.isin(set(b.product_id))].sort_values(['product_id', 'pair_id'])
        for c in ('nguon_ten', 'nguon_gt', 'cuoi_ten', 'cuoi_gt', 'nA', 'vA', 'nB', 'vB'):
            p[c] = p[c].fillna('').astype(str)
        p['catg'] = p.product_id.map(dict(zip(b.product_id, b.category)))
        keys = G.DL.khoa_cum(p.nguon_ten + ': ' + p.nguon_gt, p.nA + ': ' + p.vA, p.nB + ': ' + p.vB, p.catg)
        uses = p.groupby(keys.values).product_id.transform('nunique').values
        item_ids = [G.DL.id_cum(src, k) if count >= 2 else G.DL.id_cum_ctx(src, pid, pair_id)
                    for k, count, pid, pair_id in zip(keys, uses, p.product_id, p.pair_id)]
        generated = defaultdict(list)
        for r, iid in zip(p.to_dict('records'), item_ids):
            pid, pair_id = r['product_id'], str(r['pair_id'])
            provenance = lookup[(pid, pair_id)]
            current = r['cuoi_ten'] + ': ' + r['cuoi_gt']
            model = accepted.get((src, pid, iid))
            c = G.chon(iid, decisions, 'giu_hien_tai')
            name, value = r['cuoi_ten'], r['cuoi_gt']
            suffix = 'AB'
            if c:
                candidate = (r['nA'] + ': ' + r['vA']) if c == 'gemma' else (r['nB'] + ': ' + r['vB'])
                admissible = G.chuan(current) in (G.chuan(r['nA'] + ': ' + r['vA']), G.chuan(r['nB'] + ': ' + r['vB']))
                ok = admissible and (G.chuan(candidate) == G.chuan(current) or G.an_toan(src, candidate, current))
                assert bool(model) == bool(ok), 'Decision trace replay mismatch'
                if ok and G.chuan(candidate) != G.chuan(current):
                    suffix = 'A' if c == 'gemma' else 'B'
                    name, value = r['n' + suffix], r['v' + suffix]
            # When candidate text is unchanged, recover actual retained provenance,
            # rather than relabel it as a new generation by the judge winner.
            n_origin, n_reason = component_origin(provenance, 'n', suffix, name, r['nguon_ten'], units)
            v_origin, v_reason = component_origin(provenance, 'v', suffix, value, r['nguon_gt'], units)
            origin = attribute_origin(n_origin, v_origin)
            reason = 'mixed_models_within_one_attribute' if {n_origin, v_origin} == {'gemma', 'qwen'} else 'component_evidence'
            traces.append({'source': src, 'product_id': pid, 'pair_id': pair_id, 'item_id': iid,
                           'final_origin': origin, 'name_origin': n_origin, 'value_origin': v_origin,
                           'name_evidence': n_reason, 'value_evidence': v_reason, 'origin_reason': reason,
                           'retained_variant': suffix, 'judge_explicit_accepted_model': model or '',
                           'judge_equivalent': iid in equivalent})
            generated[pid].append(name + ': ' + value)
        expected = old[old.source.eq(src)].set_index('product_id')[G.DICH[src][1]]
        assert set(generated) == set(expected.index), 'Missing products in pair replay'
        assert all('; '.join(fragments) == expected.loc[pid] for pid, fragments in generated.items()), 'Description changed'
        print(json.dumps({'source': src, 'products': len(generated), 'attributes': len(p)}, ensure_ascii=False), flush=True)
    trace = pd.DataFrame(traces)
    assert not trace.duplicated(['source', 'product_id', 'pair_id']).any()
    counts = trace.groupby(['source', 'product_id', 'final_origin']).size()
    new = old.copy()
    new['judge_chon_description'] = [display({k: counts.get((s, pid, k), 0) for k in LABELS})
                                     for s, pid in zip(new.source, new.product_id)]
    assert new.drop(columns='judge_chon_description').equals(old.drop(columns='judge_chon_description'))
    assert new.judge_chon_description.str.len().gt(0).all()
    assert new.judge_chon_description.str.contains(': 0', regex=False).sum() == 0
    totals = trace.groupby(['source', 'product_id']).size()
    for r in new.itertuples():
        assert sum(int(part.split(': ')[1]) for part in r.judge_chon_description.split(' | ')) == totals[(r.source, r.product_id)]
    new.to_parquet(OUT / 'bo_cum_chon_deepseek.parquet', index=False)
    trace.to_parquet(OUT / 'bo_cum_description_provenance.parquet', index=False)
    report = {
        'source_revision': json.loads((WORK / 'cum_9col_upload_result.json').read_text())['commit'],
        'rows': len(new), 'columns': list(new), 'content_unchanged': True,
        'attributes': len(trace), 'attribute_origins': trace.final_origin.value_counts().to_dict(),
        'rows_changed_label': int(new.judge_chon_description.ne(old.judge_chon_description).sum()),
        'rows_with_unknown': int(new.judge_chon_description.str.contains('chua_ro:', regex=False).sum()),
        'unknown_reasons': trace[trace.final_origin.eq('chua_ro')].origin_reason.value_counts().to_dict(),
        'all_attribute_counts_reconciled': True,
        'inputs_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'self_tests_passed': 9,
    }
    (OUT / 'description_provenance_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
