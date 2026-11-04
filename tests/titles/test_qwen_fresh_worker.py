import ast
import unittest
from build_title_translation_qwen3_14b_awq_kaggle_v6 import isolate_translation_worker, cell_text


class FreshWorkerTests(unittest.TestCase):
    def test_installer_stays_in_parent_and_translation_source_is_preserved(self):
        cells=[{'cell_type':'code','metadata':{},'source':[f'marker_{i} = {i}\n'],
                'outputs':[],'execution_count':None} for i in range(11)]
        cells[3]['source']=['INSTALLER_ONLY = True\n']
        cells[7]['source']=['PROMPT = "line one\\n  line two"\n']
        expected='\n\n'.join(cell_text(cells[i]) for i in (2,4,5,7,9,10))
        result=isolate_translation_worker({'cells':cells})
        launcher=cell_text(result['cells'][-1])
        tree=ast.parse(launcher)
        wrapper=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                     and isinstance(n.value,str) and n.value.startswith('import json as _worker_json'))
        wrapper_tree=ast.parse(wrapper)
        worker_source=next(n.args[0].value for n in ast.walk(wrapper_tree)
                           if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                           and n.func.id=='compile')
        self.assertEqual(worker_source,expected)
        self.assertNotIn('INSTALLER_ONLY',worker_source)
        self.assertIn('INSTALLER_ONLY',cell_text(result['cells'][3]))
        self.assertLess(launcher.index('_qwen_health.close()'),launcher.index('_worker_subprocess.Popen'))
        imported={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
        self.assertNotIn('transformers',imported)
        self.assertNotIn('numpy',imported)


if __name__=='__main__':
    unittest.main()
