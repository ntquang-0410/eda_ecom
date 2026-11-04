from pathlib import Path
import re,os,subprocess,sys,json
root=Path(r'D:\download\NCKH\judge_kaggle_handoff');s=Path(r'D:\download\NCKH\eda_ecom\api_deepseek_alibaba.txt').read_text(encoding='utf8');keys=re.findall(r'sk-[A-Za-z0-9_.-]{20,}',s);assert len(keys)==1
env=os.environ.copy();env['DASHSCOPE_API_KEY']=keys[0];env['ALIBABA_BASE_URL']='https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1';env['REQUESTS_CA_BUNDLE']=r'C:\Users\Lenovo\.codex\kaggle-system-ca.pem';env['SSL_CERT_FILE']=env['REQUESTS_CA_BUNDLE'];env['OPENBLAS_NUM_THREADS']='1';py=sys.executable
for kind,cap in [('title',60000),('cum_ctx',120000),('pair',60000)]:
 args=[py,'-I','-X','utf8','scripts/judge/so_cap_chon.py','--loai',kind,'--thu','150','--rubric','v2','--che-do','mot_luot','--gom','15','--tat-suy-luan','--song-song','4','--tran-token',str(cap),'--ten',f'b_{kind}_150','--nha-cung-cap','alibaba']
 r=subprocess.run(args,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace');print('pilot_stage',kind,'exit',r.returncode,'quota_reached',('chạm trần' in r.stdout),flush=True)
 if r.returncode or 'chạm trần' in r.stdout:sys.exit(1)
 p=root/'data/processed/so_cap'/f'tom_tat_b_{kind}_150.json'
 if p.exists():
  j=json.loads(p.read_text(encoding='utf8'));print('summary',kind,json.dumps({x:j.get(x) for x in ['ok_lan_nay','loi_lan_nay','tokens_vao','tokens_ra','dung_do']},ensure_ascii=False),flush=True)
 else:print('missing_summary',kind,flush=True);sys.exit(2)
args=[py,'-I','-X','utf8','scripts/judge/so_sanh_gom.py','--hai-luot','m2_v2_title_150,m2_v2_cum_ctx_150,m2_v2_pair_150','--gom','b_title_150,b_cum_ctx_150,b_pair_150'];r=subprocess.run(args,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace');print('pilot_compare_exit',r.returncode,flush=True);sys.exit(r.returncode)
