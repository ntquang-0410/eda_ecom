import ast
import hashlib
from pathlib import Path
import tempfile
import unittest

from qwen_tiki_live_controller import validate_predecessor,status


class TikiChainTests(unittest.TestCase):
    def test_tiki_requires_completed_verified_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'1688.parquet';output.write_bytes(b'verified-export')
            first={'qwen_1688_completed':True,'output_rows':16348,'output_path':str(output),
                   'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
            self.assertTrue(validate_predecessor(first))
            first['qwen_1688_completed']=False
            self.assertFalse(validate_predecessor(first))
            first['qwen_1688_completed']=True
            output.write_bytes(b'changed-export')
            self.assertFalse(validate_predecessor(first))

    def test_remote_tiki_status_only_selects_aggregate(self):
        class Session:
            def execute(self,code,timeout):
                ast.parse(code)
                self.code=code
                return {'status':'ok','values':[{'completed_total':42}]}
        session=Session()
        self.assertEqual(status(session)['completed_total'],42)
        self.assertNotIn('SELECT title_zh',session.code)
        self.assertIn('count(*)',session.code)


if __name__=='__main__':unittest.main()
