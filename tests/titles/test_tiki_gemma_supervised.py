import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import run_tiki_gemma_supervised as runner


class TikiRunnerTests(unittest.TestCase):
    def run_case(self, truncated=False, flagged=False, deferred=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root/'source.parquet'
        pd.DataFrame({'product_id':[1,2,3], 'title_vi':['Đèn 12','Đèn 12','Đèn 24'],
                      'title_zh':['original','','']}).to_parquet(source, index=False)
        args = SimpleNamespace(source=source, workdir=root/'run', revision='pinned-revision',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), delay=0,
            defer_full_quality_review=deferred)
        calls=[]
        class Client:
            def __init__(self, **kwargs): pass
            def translate_record(self, title):
                calls.append(title)
                return {'title_zh':'灯具 '+title.split()[-1], 'status':'truncated' if truncated else 'ok',
                        'input_tokens':5, 'output_tokens':5}
        output=io.StringIO()
        original_qa=runner.qa_check_vi_to_zh
        def qa(source, translated):
            result=original_qa(source, translated)
            if flagged:
                result.update(passed=False, flags=['ti_le_chu_han_thap'])
            return result
        with patch.object(runner,'GemmaAPIClient',Client), patch.object(runner,'qa_check_vi_to_zh',qa), contextlib.redirect_stdout(output):
            runner.run(args)
        return args,calls,output.getvalue()

    def test_resume_cache_and_export_preserve_all_products(self):
        args,calls,logs=self.run_case()
        self.assertEqual(len(calls),2)
        report=json.loads((args.workdir/'tiki_gemma4_report.json').read_text())
        self.assertEqual(report['completed'],2)
        self.assertEqual(report['output_rows'],3)
        frame=pd.read_parquet(report['output_path'])
        self.assertEqual(frame['product_id'].tolist(),[1,2,3])
        self.assertEqual(frame['title_zh_source'].iloc[0],'original')
        self.assertTrue(frame['title_zh_gemma4_qa_passed'].all())
        self.assertNotIn('Đèn',logs)
        self.assertNotIn('灯具',logs)

    def test_truncated_output_blocks_preflight_and_is_not_completed(self):
        args,calls,logs=self.run_case(truncated=True)
        status=json.loads((args.workdir/'tiki_live_status.json').read_text())
        self.assertEqual(status['status'],'quality_gate_failed')
        self.assertEqual(status['completed'],0)
        self.assertFalse((args.workdir/'tiki_gemma4_report.json').exists())

    def test_quota_classification(self):
        class Quota(Exception): code=429
        self.assertEqual(runner.error_kind(Quota('daily quota')),'quota_daily')
        self.assertEqual(runner.error_kind(Quota('requests per minute')),'rate_limit')

    def test_deferred_qa_keeps_failed_review_visible(self):
        args,calls,_=self.run_case(flagged=True, deferred=True)
        report=json.loads((args.workdir/'tiki_gemma4_report.json').read_text())
        self.assertEqual(report['completed'],2)
        self.assertFalse(report['quality_review_passed'])
        self.assertTrue(report['quality_review_deferred'])
        self.assertEqual(report['flag_categories']['ti_le_chu_han_thap'],2)

    def test_deferred_qa_never_bypasses_truncated_completion(self):
        args,_,_=self.run_case(truncated=True, deferred=True)
        status=json.loads((args.workdir/'tiki_live_status.json').read_text())
        self.assertEqual(status['status'],'quality_gate_failed')
        self.assertEqual(status['completed'],0)


if __name__=='__main__':
    unittest.main()
