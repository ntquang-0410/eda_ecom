# -*- coding: utf-8 -*-
"""OpenNMT-py (Transformer nho, HUAN LUYEN TU DAU) tren du lieu song ngu tung dong cua minh. Chay trong notebook Kaggle THUONG (khong gan cuoc thi):
GPU T4 + Internet BAT (de cai OpenNMT-py 3.5.1 vao venv rieng, torch 2.2.2). Khong doc test; chi doc Dataset ft-song-ngu (pool + mau eval fold 0).

Cau hinh mo hinh theo Quang (config/opennmt_A.yaml): Transformer d_model 256, 4 lop, 8 dau, ff 1024, dropout 0,1, batch 4096 token x tich luy 2, Noam lr 2,0,
beta2 0,998, label smoothing 0,1, 10 epoch, beam 5, seed 42. Khac: tokenizer SentencePiece unigram 16k moi ngon ngu (Quang chua ghi), khong replace_unk.
Moi (huong, bien_the) la mot mo hinh: huong zh-vi (1688) va vi-zh (Tiki); bien_the "train" (toan bo train fold 1-4) hoac "khop4m_s42" (mau natural 4M token cua LLM).
Dau ra: onmt/<huong>_<bien_the>/sinh.jsonl ({id, nhom, hyp} dung thu tu eval), onmt_report.json.
"""
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

VAO = Path("/kaggle/input")
RA = Path("/kaggle/working")
VENV = Path("/tmp/onmt_env")
PY = str(VENV / "bin" / "python")


def sh(cmd, check=True, tail=2500):
    print(">>", cmd[:300], flush=True)
    r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], text=True, capture_output=True)
    print((r.stdout or "")[-tail:], (r.stderr or "")[-tail:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f"lenh_loi:{r.returncode}:{cmd[:120]}")
    return r


def cai():
    # Kaggle thuong: Ubuntu 24.04, Python 3.13 (venv hong, torch 2.2.2 khong co wheel) -> dung uv de lay Python 3.11
    sh("pip install -q uv")
    sh(f"uv venv --python 3.11 {VENV}")
    sh(f"uv pip install --python {PY} -q 'torch==2.2.2' 'numpy<2' 'OpenNMT-py==3.5.1' sentencepiece")
    sh(f"{PY} -c \"import torch, onmt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'onmt', onmt.__version__)\"")


def tim(ten):
    ds = [p for p in VAO.rglob(ten) if p.is_file()]
    if not ds:
        raise RuntimeError(f"khong_thay:{ten}")
    return ds[0]


def doc(ten, toi_da=None):
    L = tim(ten).read_text(encoding="utf-8").split("\n")
    if L and L[-1] == "":
        L = L[:-1]
    return L[:toi_da] if toi_da else L


def spm_train(files, prefix, vocab=16000):
    mo = f"{prefix}.model"
    if Path(mo).exists():
        return mo
    sh(f"{PY} -c \"import sentencepiece as s; s.SentencePieceTrainer.train(input='{','.join(files)}', model_prefix='{prefix}', vocab_size={vocab}, "
       f"model_type='unigram', character_coverage=0.9995, input_sentence_size=1000000, shuffle_input_sentence=True, unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1, hard_vocab_limit=False)\"")
    return mo


def spm_ma(mo, dong, ra):
    """Ma hoa danh sach dong -> tep cac manh cach nhau bang dau cach; tra danh sach do dai manh."""
    code = ("import sys, sentencepiece as s, json; sp = s.SentencePieceProcessor(model_file=sys.argv[1]); "
            "L = json.load(open(sys.argv[2], encoding='utf-8')); "
            "open(sys.argv[3], 'w', encoding='utf-8').write('\\n'.join(' '.join(sp.encode(x, out_type=str)) for x in L) + '\\n')")
    tmp = Path(ra).with_suffix(".json")
    tmp.write_text(json.dumps(dong, ensure_ascii=False), encoding="utf-8")
    sh(f"{PY} -c \"{code}\" {mo} {tmp} {ra}")
    return [len(x.split()) for x in Path(ra).read_text(encoding="utf-8").split("\n")[:len(dong)]]


def spm_giai(mo, tep_vao, tep_ra):
    code = ("import sys, sentencepiece as s; sp = s.SentencePieceProcessor(model_file=sys.argv[1]); "
            "L = open(sys.argv[2], encoding='utf-8').read().split('\\n'); L = L[:-1] if L and L[-1] == '' else L; "
            "open(sys.argv[3], 'w', encoding='utf-8').write('\\n'.join(sp.decode(x.split()) for x in L) + '\\n')")
    sh(f"{PY} -c \"{code}\" {mo} {tep_vao} {tep_ra}")


def vocab_tu(tep_sp, tep_vocab):
    c = Counter()
    for ln in Path(tep_sp).read_text(encoding="utf-8").split("\n"):
        c.update(ln.split())
    Path(tep_vocab).write_text("\n".join(f"{t}\t{n}" for t, n in c.most_common()) + "\n", encoding="utf-8")


def chay_mot(hg, bien_the, cfg, spm, W):
    src_l, tgt_l = hg.split("-")                      # zh-vi: nguon zh, dich vi
    t0 = time.time()
    k = cfg.get("khoi")
    tr_s, tr_t = doc(f"{hg}.{bien_the}.src", k), doc(f"{hg}.{bien_the}.tgt", k)
    dv_s, dv_t = doc(f"{hg}.dev.src"), doc(f"{hg}.dev.tgt")
    ev_s, ev_ids = doc(f"{hg}.eval.src", k), doc(f"{hg}.eval.id", k)
    ev_nh = doc(f"{hg}.eval.nhom", k)
    d = W / f"{hg}_{bien_the}"
    d.mkdir(parents=True, exist_ok=True)
    ls_tr = spm_ma(spm[src_l], tr_s, d / "train.src")
    lt_tr = spm_ma(spm[tgt_l], tr_t, d / "train.tgt")
    spm_ma(spm[src_l], dv_s, d / "dev.src")
    spm_ma(spm[tgt_l], dv_t, d / "dev.tgt")
    spm_ma(spm[src_l], ev_s, d / "eval.src")
    vocab_tu(d / "train.src", d / "vocab.src")
    vocab_tu(d / "train.tgt", d / "vocab.tgt")
    # bo cap qua dai (OpenNMT cat theo src_seq_length/tgt_seq_length); dem so dong bi cat de ghi bao cao
    n_tok = sum(max(a, b) + 2 for a, b in zip(ls_tr, lt_tr))
    # hieu chinh theo khoi 40.000 dong [DO]: 1 buoc = ~232 cau (batch 4096 token tinh ca padding) = ~3,3k don vi n_tok => buoc/epoch thuc ~ n_tok / 3,3k, gap ~2,2 lan cong thuc ly thuyet
    steps_epoch = max(1, math.ceil(n_tok / (4096 * 2) * 1.12 * 2.2))
    # so_epoch la TRAN; dung som theo valid (ppl, kien nhan 5 epoch); luu moi epoch, trung binh 5 checkpoint cuoi
    steps = int(cfg["so_epoch"]) * steps_epoch if not k else int(cfg.get("khoi_buoc", 200))
    warm = min(2000, max(300, 2 * steps_epoch)) if not k else max(10, steps // 10)
    # Noam: lr_dinh = learning_rate * d_model^-0.5 * warmup^-0.5. Giu dinh lr co dinh (2e-3, gan 2,8e-3 cua Quang) bat ke warmup ngan hay dai
    lr_dinh = float(cfg.get("lr_dinh", 2e-3))
    lr_noam = lr_dinh * math.sqrt(256 * warm)
    # ti le title theo token SentencePiece dich (de doi chieu voi 10,3-10,9% cua cac LLM)
    try:
        ids_tr = doc(f"{hg}.{bien_the}.id", k)
        tt = sum(x for x, i in zip(lt_tr, ids_tr) if "_title_" in i)
        ti_le_title_sp = round(tt / max(1, sum(lt_tr)), 4)
    except Exception:
        ti_le_title_sp = None
    yaml = f"""save_data: {d}/run
src_vocab: {d}/vocab.src
tgt_vocab: {d}/vocab.tgt
overwrite: true
data:
  corpus_1:
    path_src: {d}/train.src
    path_tgt: {d}/train.tgt
  valid:
    path_src: {d}/dev.src
    path_tgt: {d}/dev.tgt
src_seq_length: 256
tgt_seq_length: 256
save_model: {d}/model
save_checkpoint_steps: {150 if k else max(1, min(steps, steps_epoch))}
keep_checkpoint: 5
train_steps: {steps}
valid_steps: {150 if k else max(1, min(steps, steps_epoch))}
early_stopping: {0 if k else 4}
early_stopping_criteria: ppl
warmup_steps: {warm}
report_every: {50 if k else 200}
world_size: 1
gpu_ranks: [0]
batch_type: tokens
batch_size: 4096
valid_batch_size: 2048
accum_count: [2]
accum_steps: [0]
optim: adam
learning_rate: {lr_noam:.4f}
decay_method: noam
adam_beta2: 0.998
label_smoothing: 0.1
param_init: 0.0
param_init_glorot: true
encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: 4
dec_layers: 4
heads: 8
hidden_size: 256
word_vec_size: 256
transformer_ff: 1024
dropout: [0.1]
attention_dropout: [0.1]
seed: {int(cfg['seed'])}
"""
    (d / "train.yaml").write_text(yaml, encoding="utf-8")
    t1 = time.time()
    sh(f"{VENV}/bin/onmt_train -config {d}/train.yaml > {d}/train.log 2>&1", check=False, tail=100)
    r = sh(f"grep -E 'Step [0-9]+/|Validation|Early|Stop' {d}/train.log | grep -v Weighted | awk 'NR%1==0' | tail -n 40", check=False, tail=6000)
    ck = sorted(d.glob("model_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not ck:
        raise RuntimeError(f"khong_co_checkpoint:{hg}:{bien_the}")
    giay_train = time.time() - t1
    mo_hinh_dich, trung_binh = ck[-1], False
    if len(ck) >= 2:
        rr = sh(f"{VENV}/bin/onmt_average_models -models {' '.join(str(p) for p in ck)} -output {d}/avg.pt", check=False)
        if (d / "avg.pt").exists():
            mo_hinh_dich, trung_binh = d / "avg.pt", True
    sh(f"{VENV}/bin/onmt_translate -model {mo_hinh_dich} -src {d}/eval.src -output {d}/eval.hyp.sp -beam_size 5 -batch_size 64 -batch_type sents -gpu 0 -max_length 256 -seed 42", check=False)
    spm_giai(spm[tgt_l], d / "eval.hyp.sp", d / "eval.hyp")
    hyp = (d / "eval.hyp").read_text(encoding="utf-8").split("\n")[:len(ev_s)]
    (d / "sinh.jsonl").write_text("\n".join(json.dumps({"id": i, "nhom": g, "hyp": h}, ensure_ascii=False) for i, g, h in zip(ev_ids, ev_nh, hyp)), encoding="utf-8")
    sh(f"cp {d}/train.log {RA}/onmt/{hg}_{bien_the}_train.log", check=False, tail=100)
    for tep in ("train.src", "train.tgt", "dev.src", "dev.tgt", "eval.src", "eval.hyp.sp", "run*", "model_step_*.pt"):
        for p in d.glob(tep):
            p.unlink()
    return {"huong": hg, "bien_the": bien_the, "dong_train": len(tr_s), "token_sp_train": n_tok, "steps": steps, "steps_epoch": steps_epoch, "warmup": warm,
            "lr_dinh": lr_dinh, "learning_rate_noam": round(lr_noam, 4), "ti_le_title_token_sp_dich": ti_le_title_sp,
            "so_checkpoint": len(ck), "checkpoint_cuoi": ck[-1].name, "trung_binh_checkpoint": trung_binh, "buoc_da_chay": int(ck[-1].stem.split("_")[-1]),
            "giay_train": round(giay_train, 1), "giay_tong": round(time.time() - t0, 1), "n_eval": len(ev_s), "checkpoint": mo_hinh_dich.name,
            "log_cuoi": (r[1] if isinstance(r, tuple) else (r.stdout or ""))[-2500:]}


def dich_cum(cfg):
    """Dich eval ban cum (tung manh) bang mo hinh OpenNMT da huan luyen (avg.pt + spm trong dau ra notebook nlp-onmt, gan qua kernel_sources); khong huan luyen lai."""
    t0 = time.time()
    cai()
    W = RA / "onmt_cum"
    W.mkdir(parents=True, exist_ok=True)
    spm = {l: str(tim(f"spm_{l}.model")) for l in ("zh", "vi")}
    bc = {"cau_hinh": cfg, "viec": []}
    for hg in cfg["huong"]:
        sl, tl = hg.split("-")
        ck = [p for p in VAO.rglob("avg.pt") if p.parent.name == f"{hg}_train"]
        if not ck:
            raise RuntimeError(f"khong_thay_avg.pt:{hg}")
        d = W / hg
        d.mkdir(exist_ok=True)
        src, ids = doc(f"{hg}.evalcum.src"), doc(f"{hg}.evalcum.id")
        spm_ma(spm[sl], src, d / "cum.src")
        sh(f"{VENV}/bin/onmt_translate -model {ck[0]} -src {d}/cum.src -output {d}/cum.hyp.sp -beam_size 5 -batch_size 64 -batch_type sents -gpu 0 -max_length 256 -seed 42", check=False)
        spm_giai(spm[tl], d / "cum.hyp.sp", d / "cum.hyp")
        hyp = (d / "cum.hyp").read_text(encoding="utf-8").split("\n")[:len(src)]
        gom = {}
        for i, h in zip(ids, hyp):
            gom.setdefault(i.rsplit("#", 1)[0], []).append(h)
        nhom = {json.loads(l)["id"]: json.loads(l)["nhom"] for l in tim(f"{hg}.evalcum_items.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}
        (d / "sinh_cum.jsonl").write_text("\n".join(json.dumps({"id": i, "nhom": nhom.get(i, ""), "hyp": "; ".join(v)}, ensure_ascii=False) for i, v in gom.items()), encoding="utf-8")
        bc["viec"].append({"huong": hg, "n_manh": len(src), "n_mau": len(gom), "checkpoint": str(ck[0])})
        for p in (d / "cum.src", d / "cum.hyp.sp"):
            p.unlink()
    bc["giay_tong"] = round(time.time() - t0, 1)
    (RA / "onmt_cum_report.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("XONG dich cum", len(bc["viec"]), flush=True)


def main(cfg):
    if cfg.get("che_do") == "dich_cum":
        return dich_cum(cfg)
    t0 = time.time()
    cai()
    W = RA / "onmt"
    W.mkdir(parents=True, exist_ok=True)
    k = cfg.get("khoi")
    # SentencePiece: mot mo hinh cho moi ngon ngu, hoc tu CA HAI huong cua train day du (ben zh: nguon zh-vi + dich vi-zh; ben vi: dich zh-vi + nguon vi-zh)
    tam = W / "spm_data"
    tam.mkdir(exist_ok=True)
    (tam / "zh.txt").write_text("\n".join(doc("zh-vi.train.src", k) + doc("vi-zh.train.tgt", k)) + "\n", encoding="utf-8")
    (tam / "vi.txt").write_text("\n".join(doc("zh-vi.train.tgt", k) + doc("vi-zh.train.src", k)) + "\n", encoding="utf-8")
    spm = {"zh": spm_train([str(tam / "zh.txt")], str(W / "spm_zh")), "vi": spm_train([str(tam / "vi.txt")], str(W / "spm_vi"))}
    bc = {"cau_hinh": cfg, "viec": []}
    for hg in cfg["huong"]:
        for bt in cfg["bien_the"]:
            try:
                bc["viec"].append(chay_mot(hg, bt, cfg, spm, W))
            except Exception as e:
                import traceback
                bc["viec"].append({"huong": hg, "bien_the": bt, "loi": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2500:]})
                print("LOI", hg, bt, e, flush=True)
            (RA / "onmt_report.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1), encoding="utf-8")
    bc["giay_tong"] = round(time.time() - t0, 1)
    (RA / "onmt_report.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("XONG", len(bc["viec"]), flush=True)


if __name__ == "__main__":
    raise SystemExit("Nhung vao notebook, goi main(CAU_HINH)")
