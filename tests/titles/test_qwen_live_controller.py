import ast
from contextlib import closing
import io
import contextlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import qwen_live_controller as controller
from prepare_live_qwen_worker import prepare


class LiveControllerTests(unittest.TestCase):
    def test_status_probe_compiles_and_only_exports_aggregate(self):
        class Session:
            def execute(self,code,timeout):
                ast.parse(code)
                self.code=code
                return {'status':'ok','values':[{'completed_total':3953}]}
        session=Session()
        self.assertEqual(controller.remote_status(session)['completed_total'],3953)
        self.assertIn("WHERE status='ok' AND truncated=0",session.code)
        self.assertNotIn('SELECT output_text',session.code)

    def test_smoke_retries_keep_successful_completion_requirement(self):
        source='''import json
class Pipeline:
    calls=0
    def _load_qwen(self,*args): return ('model','tokenizer')
    def run_stage(self,*args,**kwargs):
        self.calls+=1
        return {'completion_rate':0.95 if self.calls==1 else 1.0,'completed':19 if self.calls==1 else 20}
pipeline=Pipeline()
smoke_context={}
QWEN_BATCH_SIZE=1
LOCAL_CACHE=None
def persist_qwen_cache(path): pass
smoke_result = pipeline.run_stage(smoke_context,batch_size=QWEN_BATCH_SIZE,allow_qwen_hardware_change=True)
'''
        namespace={}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(prepare(source),namespace)
        self.assertEqual(namespace['pipeline'].calls,2)
        self.assertEqual(namespace['smoke_result']['completion_rate'],1.0)

    def test_checkpoint_regression_does_not_overwrite_local_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); local=root/'local';local.mkdir()
            def database(path,count):
                with closing(sqlite3.connect(path)) as db:
                    db.execute('CREATE TABLE translations(status TEXT,truncated INTEGER)')
                    db.executemany('INSERT INTO translations VALUES (?,?)',[('ok',0)]*count)
                    db.commit()
            database(local/'cache_qwen3_14b_awq.sqlite3',2)
            remote=root/'remote.sqlite3';database(remote,1)
            class Session:
                def get_bytes(self,path):return remote.read_bytes()
            with patch.object(controller,'CHECKPOINT',local),patch.object(controller,'FILES',['cache_qwen3_14b_awq.sqlite3']):
                with self.assertRaisesRegex(RuntimeError,'checkpoint_regression'):
                    controller.backup(Session())
            self.assertEqual(controller.count_ok(local/'cache_qwen3_14b_awq.sqlite3'),2)


if __name__=='__main__':unittest.main()
