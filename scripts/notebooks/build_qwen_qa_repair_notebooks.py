"""Build independent Kaggle QA-repair stages; never overwrite baseline exports."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
from project_bootstrap import source_file, generated_root
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from build_title_translation_qwen3_14b_awq_kaggle_v6 import LOCAL_QWEN_MODEL_SETUP

ROOT=generated_root()

def build():
    base=json.loads(source_file('title_translation_qwen3_14b_awq_kaggle_v6.ipynb').read_text())
    with zipfile.ZipFile(generated_root()/'title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload_bundle.zip') as z:
        matches=[n for n in z.namelist() if Path(n).name=='title_translation_hybrid.py']
        if len(matches)!=1:raise ValueError('generator_bundle_ambiguous')
        generator=z.read(matches[0]).decode('utf-8')
    if '0.7' not in generator or '0.8' not in generator or '1.05' not in generator:
        raise ValueError('sampled_generator_not_found')
    tree=ast.parse(LOCAL_QWEN_MODEL_SETUP)
    manifest=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QWEN_MODEL_FILES' for t in n.targets))
    locations=[]
    for direction,slug,title in [('1688','qwen3-14b-awq-kaggle-v6','Qwen3 14B AWQ Kaggle V6'),('tiki','qwen3-14b-awq-tiki-live','Qwen3 14B AWQ Tiki Live')]:
        code=f"""from pathlib import Path
import json,shutil,sys
root=Path('/kaggle/working/qwen_qa_repair_{direction}');root.mkdir(exist_ok=True)
manifests=list(Path('/kaggle/input').rglob('repair_manifest.json'))
assert len(manifests)==1,'Exactly one QA-repair input Dataset required'
payload=root/'payload';payload.mkdir(exist_ok=True)
for name in ('repair_manifest.json','qa_repair_{direction}.parquet'):shutil.copy2(manifests[0].parent/name,payload/name)
"""
        sources={name:source_file(name).read_text(encoding='utf-8') for name in ('qwen_qa_repair_worker.py','qwen_kaggle_health.py','qwen_tiki_live_worker.py','tiki_translation_pipeline.py','title_translation_pipeline.py','kaggle_batch_supervisor.py')}
        sources['title_translation_hybrid.py']=generator
        for name,text in sources.items():code+=f"(root/{name!r}).write_text({text!r},encoding='utf-8')\n"
        code+=f"(root/'model_manifest.json').write_text({json.dumps(manifest)!r})\n"
        code+=f"sys.path.insert(0,str(root))\nfrom kaggle_batch_supervisor import run as batch_run\nprint(json.dumps(batch_run([sys.executable,'-u',str(root/'qwen_qa_repair_worker.py'),'--direction',{direction!r}],root)))\n"
        compile(code,'qa_repair_notebook','exec')
        notebook={'nbformat':4,'nbformat_minor':5,'metadata':base['metadata'],
                  'cells':[{'cell_type':'markdown','metadata':{},'source':[f'# Qwen {direction}: separate QA repair\nBaseline outputs remain unchanged. Repair prompt qwen-technical-qa-repair-v1; same pinned model and sampled decoding. Only technical-pass corrections are accepted. Unresolved flags remain.']},
                           *[dict(base['cells'][i],outputs=[],execution_count=None) for i in (2,3)],
                           {'cell_type':'code','metadata':{},'source':[code],'outputs':[],'execution_count':None}]}
        out=ROOT/('qwen_qa_repair_'+direction+'_kaggle');out.mkdir(exist_ok=True)
        (out/'qa_repair.ipynb').write_text(json.dumps(notebook))
        metadata={'id':'nadsa091/'+slug,'title':title,'code_file':'qa_repair.ipynb','language':'python','kernel_type':'notebook',
                  'is_private':True,'enable_gpu':True,'enable_internet':False,
                  'dataset_sources':['nadsa091/qwen3-14b-awq-kaggle-input','nadsa091/qwen3-14b-awq-offline-wheelhouse','nadsa091/qwen3-14b-awq-offline-sdists','raufrahim/qwen3-14b-awq','nadsa091/translation-autonomous-resume','nadsa091/qwen-title-qa-repair-inputs'],
                  'competition_sources':[],'kernel_sources':[]}
        (out/'kernel-metadata.json').write_text(json.dumps(metadata));locations.append(str(out))
    return locations

if __name__=='__main__':print(json.dumps(build()))
