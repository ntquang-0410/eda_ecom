import json
import unittest
from qwen_qa_repair_worker import accept,repair_messages

class RepairTests(unittest.TestCase):
    def test_1688_only_accepts_complete_passing_vietnamese(self):
        record={'output_text':'Áo sơ mi 10 chiếc','truncated':False,'repetition_detected':False}
        self.assertTrue(accept('衬衫10件',record,'1688')[0])
        self.assertFalse(accept('衬衫10件',dict(record,truncated=True),'1688')[0])
        self.assertFalse(accept('衬衫10件',dict(record,output_text='衬衫10件'),'1688')[0])

    def test_tiki_does_not_accept_residual_vietnamese_or_missing_numbers(self):
        record={'output_text':'衬衫10件','truncated':False,'repetition_detected':False}
        self.assertTrue(accept('Áo sơ mi 10 chiếc',record,'tiki')[0])
        self.assertFalse(accept('Áo sơ mi 10 chiếc',dict(record,output_text='衬衫'),'tiki')[0])
        self.assertFalse(accept('Áo sơ mi 10 chiếc',dict(record,output_text='Áo sơ mi 10 chiếc'),'tiki')[0])

    def test_repetition_never_accepted(self):
        self.assertFalse(accept('衬衫10件',{'output_text':'Áo sơ mi 10 chiếc','repetition_detected':True},'1688')[0])

    def test_prompt_preserves_source_and_draft_as_separate_data(self):
        messages=repair_messages('source','original draft',['number_mismatch'],'1688')
        payload=json.loads(messages[1]['content'])
        self.assertEqual(payload,{'source':'source','previous_draft':'original draft','technical_flags':['number_mismatch']})
        self.assertIn('Vietnamese',messages[0]['content'])

if __name__=='__main__':unittest.main()
