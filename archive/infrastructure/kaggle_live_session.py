"""Token-safe access to this task's existing Kaggle Jupyter session."""
from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import requests
import websocket


class KaggleLiveSession:
    def __init__(self, operation_path=None):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.operation_path=Path(operation_path or r'C:\Users\Lenovo\.codex\qwen-interactive-operation.json')
        self.api=KaggleApi(); self.api.authenticate()
        self.http=requests.Session()
        self.base=None
        self.token=None

    def connect(self):
        saved=json.loads(self.operation_path.read_text(encoding='utf-8'))
        with self.api.build_kaggle_client() as client:
            op=client.common.operations_client.get_operation(name=saved['operation_name'])
        if not op.done:
            return False
        if op.error:
            raise RuntimeError('session_creation_failed')
        response=op.response or {}
        if not response.get('jupyterUrl'):
            raise RuntimeError('session_not_available')
        self.base=response['jupyterUrl'].rstrip('/')
        self.token=parse_qs(urlsplit(response['tokenizedJupyterUrl']).query)['token'][0]
        self.http.headers['Authorization']='token '+self.token
        return True

    def execute(self, code, timeout=120, on_event=None):
        kernels=self.http.get(self.base+'/api/kernels',timeout=20)
        kernels.raise_for_status()
        kernel=kernels.json()[0]['id']
        sid=uuid.uuid4().hex; mid=uuid.uuid4().hex
        url=self.base.replace('https://','wss://',1)+'/api/kernels/'+kernel+'/channels?session_id='+sid
        ws=websocket.create_connection(url,header={'Authorization':'token '+self.token},
            origin=urlsplit(self.base).scheme+'://'+urlsplit(self.base).netloc,timeout=20)
        result={'status':'waiting','values':[]}
        message={'header':{'msg_id':mid,'username':'codex','session':sid,'date':datetime.now(timezone.utc).isoformat(),
            'msg_type':'execute_request','version':'5.3'},'parent_header':{},'metadata':{},
            'content':{'code':code,'silent':False,'store_history':False,'user_expressions':{},
                       'allow_stdin':False,'stop_on_error':True},'channel':'shell','buffers':[]}
        text_buffer=''
        try:
            ws.send(json.dumps(message))
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                try:
                    event=json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if event.get('parent_header',{}).get('msg_id')!=mid:
                    continue
                kind=event.get('msg_type') or event.get('header',{}).get('msg_type')
                content=event.get('content',{})
                if kind=='stream':
                    text_buffer+=content.get('text','')
                    while '\n' in text_buffer:
                        line,text_buffer=text_buffer.split('\n',1)
                        for prefix in ('[codex-json] ','[qwen-health] ','[qwen-boot] '):
                            if line.startswith(prefix):
                                try:
                                    value=json.loads(line[len(prefix):])
                                except ValueError:
                                    continue
                                if prefix=='[codex-json] ':
                                    result['values'].append(value)
                                elif on_event:
                                    on_event(prefix.strip(),value)
                if kind=='execute_reply':
                    result['status']=content.get('status')
                    if content.get('ename'):
                        result['error_class']=content['ename']
                    return result
            raise TimeoutError('remote_execution_timeout')
        finally:
            ws.close()

    def contents(self, relative, content=True, format=None):
        params={'content':int(content)}
        if format: params['format']=format
        response=self.http.get(self.base+'/api/contents/'+relative.lstrip('/'),params=params,timeout=30)
        response.raise_for_status()
        return response.json()

    def mkdir(self, relative):
        parts=Path(relative).as_posix().strip('/').split('/')
        for index in range(1,len(parts)+1):
            path='/'.join(parts[:index])
            response=self.http.get(self.base+'/api/contents/'+path,params={'content':0},timeout=20)
            if response.status_code==404:
                response=self.http.put(self.base+'/api/contents/'+path,json={'type':'directory'},timeout=30)
            response.raise_for_status()

    def put_bytes(self, relative, data):
        parent=relative.rsplit('/',1)[0] if '/' in relative else ''
        if parent: self.mkdir(parent)
        response=self.http.put(self.base+'/api/contents/'+relative,
            json={'type':'file','format':'base64','content':base64.b64encode(data).decode('ascii')},timeout=90)
        response.raise_for_status()

    def get_bytes(self, relative):
        model=self.contents(relative,format='base64')
        return base64.b64decode(model['content'])

    def read_json(self, relative):
        model=self.contents(relative,format='text')
        return json.loads(model['content'])
