"""Build a self-contained Kaggle scorer once comparison inputs are prepared."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
from project_bootstrap import source_file, generated_root
import json
import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
import argparse
from pathlib import Path

ROOT=Path(__file__).parent

def build(hf_token=None):
    destination=generated_root()/'cometkiwi_1688_kaggle';destination.mkdir(exist_ok=True)
    setup=r"""import sys,subprocess,importlib.metadata
from pathlib import Path
from kaggle_secrets import UserSecretsClient
import os
# Credentials remain in memory and are inherited only by the scoring worker.
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
manifests=[p for p in Path('/kaggle/input').rglob('comparison_manifest.json') if (p.parent/'comparison_inputs.parquet').is_file()]
assert len(manifests)==1,'Attach the verified 1688 comparison input Dataset before running.'
import shutil
work=Path('/kaggle/working/cometkiwi_1688');work.mkdir(exist_ok=True)
for name in ('comparison_manifest.json','comparison_inputs.parquet','source_rows.parquet'):
 shutil.copy2(manifests[0].parent/name,work/name)
constraint=Path('/tmp/comet_torch_constraint.txt')
constraint.write_text('torch=='+importlib.metadata.version('torch')+'\n')
subprocess.run([sys.executable,'-m','pip','install','--quiet','--constraint',str(constraint),'unbabel-comet==2.2.7','pyarrow'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','protobuf==5.29.5'],check=True)
"""
    for name in ('compare_1688_cometkiwi.py','title_translation_pipeline.py'):
        setup+=f"(work/{name!r}).write_text({source_file(name).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
    setup+="""# Fresh interpreter avoids stale NumPy imports after dependency installation.
script=work/'compare_1688_cometkiwi.py'
for stage in ('score','select'):
 subprocess.run([sys.executable,'-u',str(script),stage,'--workdir',str(work),'--batch-size','8','--gpus','1'],check=True)
os.environ.pop('HF_TOKEN',None)
"""
    if hf_token:
        raise ValueError('Use Kaggle Secrets HF_TOKEN; embedded tokens are disabled')
    compile(setup,'cometkiwi_notebook','exec')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'markdown','metadata':{},'source':['# 1688 Gemma vs Qwen: COMETKiwi selection\nPinned wmt22-cometkiwi-da; technical QA first; title floor 0.72. HF_TOKEN secret required. Run as Save & Run All. Scores are estimates, not human semantic verification.']},
                       {'cell_type':'code','metadata':{},'source':[setup],'outputs':[],'execution_count':None}]}
    (destination/'compare_1688.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata={'id':'nadsa091/1688-cometkiwi-best-title','title':'1688 CometKiwi Best Title','code_file':'compare_1688.ipynb',
              'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
              'dataset_sources':['nadsa091/1688-cometkiwi-comparison-input'],'competition_sources':[],'kernel_sources':[]}
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata))
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--embed-authorized-token',action='store_true')
    args=parser.parse_args()
    token=os.environ.get('HF_TOKEN') if args.embed_authorized_token else None
    if args.embed_authorized_token and not token:raise SystemExit('HF_TOKEN is required')
    print(build(token))
