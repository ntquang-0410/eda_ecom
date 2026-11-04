import unittest
import pandas as pd
from audit_qwen_tiki_export import validate_frames


class AuditTests(unittest.TestCase):
    def frames(self):
        source=pd.DataFrame({'product_id':[1,2],'title_vi_cleaned':['a','b'],'title_zh':['old','']})
        output=source.rename(columns={'title_zh':'title_zh_source'}).copy()
        output['title_zh_qwen3_14b_awq']=['甲','乙'];output['title_zh']=output['title_zh_qwen3_14b_awq']
        return source,output

    def test_source_preserved_with_explicit_translation_promotion(self):
        source,output=self.frames()
        self.assertEqual(validate_frames(source,output),'title_vi_cleaned')

    def test_reordered_rows_are_rejected(self):
        source,output=self.frames()
        with self.assertRaisesRegex(ValueError,'source_column_or_order_changed'):
            validate_frames(source,output.iloc[::-1].reset_index(drop=True))

    def test_empty_translation_is_rejected(self):
        source,output=self.frames();output.loc[0,'title_zh_qwen3_14b_awq']=' '
        with self.assertRaisesRegex(ValueError,'blank_translation'):validate_frames(source,output)

if __name__=='__main__':unittest.main()
