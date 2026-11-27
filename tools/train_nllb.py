#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLLB-600M LoRA trainer (bản chạy được trên Colab/Kaggle, có resume + checkpoint).

Giữ đúng recipe của `kaggle/matrix_nllb`: LoRA r16/all-linear, lr 3e-5, warmup 20 +
cosine 500, batch 2, max_seq 768, greedy decode theo max_new của eval.

Chạy:
  python tools/train_nllb.py --units ... --sample ... --eval ... --model facebook/nllb-200-distilled-600M \
      --out runs/nllb_b_sub42 --seed 42 --save-every 25
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max-seq", type=int, default=768)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=0, help=">0: giới hạn max_new khi infer (chẩn đoán)")
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--upload-script", default="", help="bash script nhận (ckpt_dir, step)")
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(args.units).reset_index(drop=True)
    s = pd.read_parquet(args.sample)
    ev = pd.read_parquet(args.eval)
    log(f"units {len(u)} | sample draws {len(s)} steps {s.step.nunique()} | eval {len(ev)}")

    tok = AutoTokenizer.from_pretrained(args.model, src_lang="vie_Latn", tgt_lang="zho_Hans")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tgt_id = tok.convert_tokens_to_ids("zho_Hans")

    base = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        log(f"warm-start {args.init_adapter}")
    else:
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                                task_type="SEQ_2_SEQ_LM", target_modules="all-linear"))
    model.to("cuda").train(); model.config.use_cache = False

    def encode(src, tgt):
        a = tok(str(src or "").strip(), add_special_tokens=True, truncation=True,
                max_length=args.max_seq)["input_ids"]
        b = tok(str(tgt or "").strip(), add_special_tokens=True, truncation=True,
                max_length=args.max_seq)["input_ids"]
        if len(a) < 2 or len(b) < 2:
            return None
        return a, b

    def collate(items):
        mi = max(len(x[0]) for x in items); mo = max(len(x[1]) for x in items)
        inp, att, lab = [], [], []
        for a, b in items:
            inp.append(a + [tok.pad_token_id] * (mi - len(a)))
            att.append([1] * len(a) + [0] * (mi - len(a)))
            lab.append(b + [-100] * (mo - len(b)))
        return {"input_ids": torch.tensor(inp), "attention_mask": torch.tensor(att),
                "labels": torch.tensor(lab)}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / 20) * max(0.0, 0.5 * (1 + math.cos(
            math.pi * min(1.0, step / args.steps)))))
    for _ in range(args.start_step):
        sched.step()

    def save(step: int, final: bool) -> None:
        name = f"adapter_{'final' if final else 'ckpt'}"
        dst = out / name
        model.save_pretrained(dst)
        tok.save_pretrained(dst)
        (dst / "STEP.txt").write_text(str(step), encoding="utf-8")
        log(f"save {dst.name} step {step}")
        if args.upload_script and not final:
            import subprocess
            subprocess.run(["bash", args.upload_script, str(dst), str(step)], check=False)

    steps = sorted(s.step.unique().tolist())
    started = time.time()
    for si, step in enumerate(steps):
        if step < args.start_step:
            continue
        idxs = s[s.step == step].row_index.astype(int).tolist()
        items = [x for x in (encode(u.source_text.iloc[i], u.target_text.iloc[i]) for i in idxs) if x]
        if not items:
            continue
        opt.zero_grad(set_to_none=True)
        step_tokens = sum(len(x[1]) for x in items)
        for j in range(0, len(items), args.batch):
            batch = {k: v.to("cuda") for k, v in collate(items[j:j + args.batch]).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss * (int((batch["labels"] != -100).sum()) / max(1.0, step_tokens))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 25 == 0:
            log(f"step {step}/{steps[-1]} min={(time.time()-started)/60:.1f} lr={sched.get_last_lr()[0]:.2e}")
        if args.save_every and (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            save(step + 1, final=False)
    save(args.steps, final=True)
    if args.skip_infer:
        return

    model.eval(); model.config.use_cache = True
    recs = []
    for max_new, part in ev.groupby("max_new"):
        part = part.reset_index(drop=True)
        mn = int(max_new) if not args.max_new else min(int(max_new), args.max_new)
        bs = 1 if mn > 300 else 8
        log(f"infer max_new={max_new} rows={len(part)} bs={bs}")
        for k in range(0, len(part), bs):
            chunk = part.iloc[k:k + bs]
            enc = tok([str(x) for x in chunk.source], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_seq, add_special_tokens=True).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=mn,
                                     do_sample=False, num_beams=1, eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.pad_token_id)
            for eid, text in zip(chunk.eval_id, tok.batch_decode(gen, skip_special_tokens=True)):
                recs.append({"eval_id": eid, "hypothesis": text.strip()})

    hyp = ev[["eval_id", "dataset", "pair_type", "source", "reference"]].copy()
    h = pd.DataFrame(recs).set_index("eval_id")
    hyp["hypothesis"] = [str(h.hypothesis.get(e, "")) for e in hyp.eval_id]
    hyp["config"] = "nllb_b_sub4842"
    hyp["seed"] = args.seed
    dst = out / f"hyp_seed{args.seed}.parquet"
    hyp.to_parquet(dst, index=False)
    info = {"rows": len(hyp), "empty": int(hyp.hypothesis.str.strip().eq("").sum()),
            "minutes": round((time.time() - started) / 60, 1), "steps": args.steps,
            "start_step": args.start_step, "lr": args.lr, "max_seq": args.max_seq,
            "sample": str(args.sample), "units": str(args.units)}
    (out / f"run_seed{args.seed}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"WROTE {dst} {info}")


if __name__ == "__main__":
    main()
