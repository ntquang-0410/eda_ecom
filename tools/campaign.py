#!/usr/bin/env python3
"""Quản lý campaign jobs (1792 + NLLB subsample + EcomGPT retry)."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[1]; F=R/'campaign/jobs.json'
def load(): return json.loads(F.read_text(encoding='utf-8'))
def save(j): F.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    cmd=sys.argv[1] if len(sys.argv)>1 else 'status'
    jobs=load()
    if cmd=='status':
        c=Counter(j['status'] for j in jobs)
        print('tổng',len(jobs),dict(c),'| giờ còn lại ~',round(sum(j['hours_est'] for j in jobs if j['status']!='done'),1))
        for j in jobs:
            if j['status']!='done': print(f"{j['id']:32s} {j['status']:9s} {j['hours_est']:5.1f}h")
    elif cmd=='mark':
        jid,st=sys.argv[2],sys.argv[3]; assert st in ('pending','running','done','failed')
        for j in jobs:
            if j['id']==jid: j['status']=st
        save(jobs); print('marked',jid,st)
    elif cmd=='next':
        p=[j for j in jobs if j['status']=='pending']
        print(json.dumps(p[0],ensure_ascii=False,indent=2) if p else 'hết job')
if __name__=='__main__': main()
