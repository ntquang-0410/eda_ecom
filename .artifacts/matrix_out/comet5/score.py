
import json, os
os.environ["USE_TF"]="0"; os.environ["USE_FLAX"]="0"
from pathlib import Path
import numpy as np, pandas as pd, torch, re
from comet import download_model, load_from_checkpoint
import sacrebleu
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
DA = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
QE = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
print("models loaded")

HYP = Path("/kaggle/input/tiki-matrix-hyps")
if not any(HYP.glob("hyp_*.parquet")):
    HYP = sorted(Path("/kaggle/input").rglob("hyp_qwen_a_seed42.parquet"))[0].parent
OUT = Path("/kaggle/working/scores"); OUT.mkdir(exist_ok=True)
metrics = {}
for path in sorted(HYP.glob("hyp_*_seed42.parquet")):
    label = re.sub(r"^hyp_|_seed42\.parquet$", "", path.name)
    df = pd.read_parquet(path).drop_duplicates("eval_id")
    for dataset in ("1688", "tiki"):
        part = df[df["dataset"] == dataset].copy()
        part = part[part["hypothesis"].astype(str).str.strip().ne("")]
        if dataset == "1688":
            part = part[part["reference"].astype(str).str.strip().ne("")]
            data=[{"src":str(s),"mt":str(m),"ref":str(r)} for s,m,r in zip(part.source,part.hypothesis,part.reference)]
            out=DA.predict(data,batch_size=16,gpus=1)
            bleu=sacrebleu.corpus_bleu(part.hypothesis.tolist(),[part.reference.tolist()],tokenize="zh").score
            chrf=sacrebleu.corpus_chrf(part.hypothesis.tolist(),[part.reference.tolist()]).score
            extra={"bleu":round(float(bleu),4),"chrf":round(float(chrf),4)}
        else:
            data=[{"src":str(s),"mt":str(m)} for s,m in zip(part.source,part.hypothesis)]
            out=QE.predict(data,batch_size=16,gpus=1); extra={}
        part["score"]=np.asarray(out.scores,dtype=float)
        part["config"]=label; part["seed"]=42; part["dataset"]=dataset
        part[["eval_id","pair_type","score","config","seed","dataset"]].to_parquet(
            OUT/f"scores_{label}_seed42_{dataset}.parquet",index=False)
        metrics[f"{label}::{dataset}"]={"system_score":round(float(out.system_score),4),"n":int(len(part)),**extra}
        print(label,dataset,metrics[f"{label}::{dataset}"],flush=True)
        (OUT/"comet_scores.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metrics,ensure_ascii=False,indent=2))
