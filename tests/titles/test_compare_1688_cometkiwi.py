import math
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import pandas as pd
from compare_1688_cometkiwi import choose,score_key,select,sha,REVISION,open_scores

PASS={'status':'pass'}
FAIL={'status':'fail'}

class ComparisonTests(unittest.TestCase):
    def test_higher_score_wins_when_both_eligible(self):
        self.assertEqual(choose('g','q',.8,.9,PASS,PASS)['model'],'qwen3_14b_awq')

    def test_hard_failure_cannot_win_with_higher_score(self):
        self.assertEqual(choose('g','q',.8,.99,PASS,FAIL)['model'],'gemma4')

    def test_both_below_floor_are_held(self):
        self.assertIsNone(choose('g','q',.71,.7,PASS,PASS)['selected'])

    def test_both_hard_failures_are_held(self):
        self.assertEqual(choose('g','q',.99,.99,FAIL,FAIL)['model'],'hold')

    def test_ties_are_explicit_and_reproducible(self):
        self.assertEqual(choose('g','q',.9,.9,PASS,PASS)['decision'],'equal_score_tie_gemma4')

    def test_missing_or_nonfinite_scores_are_not_replaced_by_zero(self):
        with self.assertRaises(ValueError):choose('g','q',math.nan,.9,PASS,PASS)

    def test_scoring_keys_bind_source_and_candidate(self):
        self.assertNotEqual(score_key('s1','t'),score_key('s2','t'))
        self.assertNotEqual(score_key('s','t1'),score_key('s','t2'))

    def test_export_preserves_duplicate_products_and_holds(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            source=pd.DataFrame({'product_id':['a','b','c'],'title_zh':['甲','乙','甲'],'title_vi':['old']*3})
            source.to_parquet(work/'source_rows.parquet',index=False)
            # Match prepare's actual column order: gemma QA appears before Qwen.
            inputs=pd.DataFrame({'title_zh':['甲','乙'],'gemma4':['g1','g2'],
                                 'gemma4_qa':[json.dumps(PASS)]*2,'qwen3_14b_awq':['q1','q2'],
                                 'qwen3_14b_awq_qa':[json.dumps(PASS)]*2})
            inputs.to_parquet(work/'comparison_inputs.parquet',index=False)
            (work/'comparison_manifest.json').write_text(json.dumps({'expected_rows':3,'expected_unique':2}))
            (work/'scorer_provenance.json').write_text(json.dumps({'revision':REVISION,'comparison_manifest_sha256':sha(work/'comparison_manifest.json')}))
            with contextlib.closing(open_scores(work)) as db:
                db.executemany('insert into scores values(?,?)',[(score_key('甲','g1'),.8),(score_key('甲','q1'),.9),(score_key('乙','g2'),.5),(score_key('乙','q2'),.6)])
                db.commit()
            with contextlib.redirect_stdout(io.StringIO()):select(SimpleNamespace(workdir=work))
            output=pd.read_parquet(work/'bilingual_zh_vi_cado_title_best_cometkiwi.parquet')
            self.assertEqual(output['product_id'].tolist(),['a','b','c'])
            self.assertEqual(output.loc[0,'title_vi'],'q1')
            self.assertTrue(pd.isna(output.loc[1,'title_vi']))
            self.assertEqual(output.loc[2,'title_vi'],'q1')
            self.assertFalse(output['semantic_verified'].any())

if __name__=='__main__':unittest.main()
