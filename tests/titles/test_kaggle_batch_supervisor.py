import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from kaggle_batch_supervisor import cache_progress


class BatchProgressTests(unittest.TestCase):
    def test_counts_exclude_truncated_and_never_return_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.sqlite3'
            with closing(sqlite3.connect(path)) as db:
                db.execute('create table translations(status text,truncated int,title text)')
                db.executemany('insert into translations values(?,?,?)',[
                    ('ok',0,'private content'),('ok',1,'private content'),('error',0,'private content')])
                db.commit()
            result=cache_progress(path)
            self.assertEqual(result['completed'],1)
            self.assertEqual(result['cached_records'],3)
            self.assertNotIn('private content',json.dumps(result))

    def test_gemma_schema_without_truncation_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.sqlite3'
            with closing(sqlite3.connect(path)) as db:
                db.execute('create table translations(status text)')
                db.executemany('insert into translations values(?)',[('ok',),('error',)])
                db.commit()
            self.assertEqual(cache_progress(path)['completed'],1)


if __name__=='__main__':unittest.main()
