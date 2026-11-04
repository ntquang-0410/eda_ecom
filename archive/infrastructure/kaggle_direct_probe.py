"""Execute a controlled GPU diagnostic in the existing authenticated session."""

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from urllib.parse import urlsplit, parse_qs
import requests
import websocket
from kaggle.api.kaggle_api_extended import KaggleApi

api=KaggleApi();api.authenticate()
saved=Path(r'C:\Users\Lenovo\.codex\qwen-interactive-operation.json')
name=json.loads(saved.read_text())['operation_name']
with api.build_kaggle_client() as client:
    operation=client.common.operations_client.get_operation(name=name)
response=operation.response or {}
base=response['jupyterUrl'].rstrip('/')
token=parse_qs(urlsplit(response['tokenizedJupyterUrl']).query)['token'][0]
headers={'Authorization':'token '+token}
session=requests.Session();session.headers.update(headers)
kernels=session.get(base+'/api/kernels',timeout=20).json()
kernel=kernels[0]['id']
sid=uuid.uuid4().hex
msgid=uuid.uuid4().hex
url=base.replace('https://','wss://',1)+'/api/kernels/'+kernel+'/channels?session_id='+sid
result={'session_connected':False,'execution_finished':False}
code=r'''
import json, shutil, pathlib, subprocess
import torch
probe={'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,
       'cuda_runtime':torch.version.cuda,'nvidia_device_present':pathlib.Path('/dev/nvidia0').exists(),
       'nvidia_smi_available':bool(shutil.which('nvidia-smi'))}
if shutil.which('nvidia-smi'):
    check=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    probe['nvidia_smi_returncode']=check.returncode
    if check.returncode==0:
        probe['hardware_gpus']=[{'name':x.split(',')[0].strip(),'memory_mib':int(x.split(',')[1].strip())} for x in check.stdout.splitlines()]
print('[qwen-direct-probe] '+json.dumps(probe),flush=True)
'''
try:
    ws=websocket.create_connection(url,header=headers,origin=urlsplit(base).scheme+'://'+urlsplit(base).netloc,timeout=20)
    result['session_connected']=True
    message={'header':{'msg_id':msgid,'username':'codex','session':sid,
        'date':datetime.now(timezone.utc).isoformat(),'msg_type':'execute_request','version':'5.3'},
        'parent_header':{},'metadata':{},'content':{'code':code,'silent':False,'store_history':False,
        'user_expressions':{},'allow_stdin':False,'stop_on_error':True},'channel':'shell','buffers':[]}
    ws.send(json.dumps(message))
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        event=json.loads(ws.recv())
        if event.get('parent_header',{}).get('msg_id')!=msgid:
            continue
        kind=event.get('msg_type') or event.get('header',{}).get('msg_type')
        content=event.get('content',{})
        if kind=='stream':
            for line in content.get('text','').splitlines():
                if line.startswith('[qwen-direct-probe] '):
                    result['probe']=json.loads(line.split('] ',1)[1])
        if kind=='execute_reply':
            result.update(execution_finished=True,execution_status=content.get('status'))
            if content.get('ename'):
                result['error_class']=content['ename']
            break
    ws.close()
except Exception as exc:
    result['connection_error_class']=type(exc).__name__
Path(r'C:\Users\Lenovo\.codex\qwen-direct-probe-result.json').write_text(json.dumps(result),encoding='utf-8')
print(json.dumps(result))
