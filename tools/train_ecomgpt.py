#!/usr/bin/env python3
"""EcomGPT-7B LoRA trainer cho ma trận Tiki (recipe của kaggle/tiki-matrix-ecomgpt-a2).

Khác trainer Qwen: dùng instruction template của EcomGPT, tokenize thô source/target,
quota lấy từ canonical units (Kaggle dataset). Hỗ trợ resume:
  --init-adapter DIR --start-step N   (pre-advance scheduler tới N)

Sau khi train xong (hoặc ngay nếu start_step >= steps) chạy inference và ghi
  <out>/hyp_<config>_seed<seed>.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

INSTRUCTION = ("Translate the following Vietnamese e-commerce text into Simplified Chinese. "
               "Return only the translation.")
TEMPLATE = ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")


def chunked_ce(logits, labels, chunk: int = 256):
    """Cross-entropy theo từng khúc sequence: cùng giá trị toán học với CE đầy đủ
    nhưng tránh cấp phát logits.float() toàn bộ (vocab BLOOM 250k)."""
    import torch.nn.functional as F
    n = int((labels != -100).sum().item())
    if n == 0:
        return logits.sum() * 0.0
    total = None
    for s in range(0, logits.shape[1], chunk):
        lg = logits[:, s:s + chunk].float()
        lb = labels[:, s:s + chunk]
        part = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb.reshape(-1),
                               ignore_index=-100, reduction="sum")
        total = part if total is None else total + part
    return total / n

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_plan(rows: pd.DataFrame, tokenizer, steps: int, tokens_per_step: float,
               maxlen: int) -> list[list[tuple[list[int], list[int]]]]:
    def encode(row):
        p = TEMPLATE.format(instruction=INSTRUCTION, input=str(row["source_text"]))
        pid = tokenizer(p, add_special_tokens=False).input_ids
        if len(pid) >= maxlen - 8:
            return None
        full = tokenizer(p + str(row["target_text"]) + tokenizer.eos_token,
                         add_special_tokens=False, truncation=True, max_length=maxlen).input_ids
        if len(full) <= len(pid) + 1:
            return None
        return full, [-100] * len(pid) + full[len(pid):]

    plan, cur, cur_tok = [], [], 0.0
    for _, row in rows.iterrows():
        item = encode(row)
        if item is None:
            continue
        cur.append(item)
        cur_tok += len(item[0])
        if cur_tok >= tokens_per_step and len(plan) < steps - 1:
            plan.append(cur)
            cur, cur_tok = [], 0.0
    if cur:
        plan.append(cur)
    while len(plan) < steps:
        last = plan.pop()
        if len(last) <= 1:
            plan.extend([last, last])
            continue
        mid = len(last) // 2
        plan.extend([last[:mid], last[mid:]])
    return plan[:steps]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens", type=int, default=4_000_000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--configs", default="natural")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=960)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--upload-script", default=None)
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    units = pd.read_parquet(data_root / f"processed_{args.dataset}" / "units.parquet")
    meta = json.loads((data_root / f"meta_{args.dataset}.json").read_text(encoding="utf-8"))
    eval_frame = pd.read_parquet(data_root / f"eval_{args.dataset}.parquet")

    sys.path.insert(0, "/content/work/eda_ecom/tools")
    from quota_sampling import quota_indices
    cfg = meta["configs"].get(args.configs)
    idx, actual = quota_indices(units, cfg, args.seed, args.tokens)
    sample = units.iloc[idx].reset_index(drop=True)
    log(f"sample rows={len(sample)} tokens={sample.eff_tokens.sum()/1e6:.3f}M config={args.configs} actual={actual}")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:
        log(f"load adapter {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                bias="none", task_type="CAUSAL_LM",
                                                target_modules="all-linear"))
    model.to("cuda")
    log(f"param dtype={next(model.parameters()).dtype}")
    model.train()
    model.config.use_cache = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    start = int(args.start_step)
    if start:
        log(f"pre-advance scheduler {start} steps")
        for _ in range(start):
            sched.step()

    plan = build_plan(sample, tokenizer, args.steps, 8000.0, args.max_seq)
    log(f"plan steps={len(plan)} avg_rows={sum(len(s) for s in plan)/len(plan):.2f} start={start}")

    def save(step: int, final: bool) -> None:
        name = f"adapter_{args.configs}_seed{args.seed}" + ("" if final else "_ckpt")
        dst = out / name
        model.save_pretrained(dst)
        (dst / "STEP.txt").write_text(str(step), encoding="utf-8")
        log(f"checkpoint {dst.name} step {step}")
        if args.upload_script and not final:
            subprocess.run(["bash", args.upload_script, str(dst), str(step)], check=False)

    started = time.time()
    last_loss = float("nan")
    for step in range(start, args.steps):
        items = plan[step]
        opt.zero_grad(set_to_none=True)
        step_tokens = sum(len(x[0]) for x in items)
        for s0 in range(0, len(items), args.micro_batch):
            chunk = items[s0:s0 + args.micro_batch]
            L = max(len(x[0]) for x in chunk)
            ids = torch.tensor([x[0] + [tokenizer.pad_token_id] * (L - len(x[0])) for x in chunk]).cuda()
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in chunk]).cuda()
            mask = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in chunk]).cuda()
            o = model(input_ids=ids, attention_mask=mask)
            n_tok = int((lab != -100).sum().item())
            loss = chunked_ce(o.logits, lab) * (n_tok / max(1.0, step_tokens))
            loss.backward()
            last_loss = float(loss.detach()) * (step_tokens / max(1.0, n_tok))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(1, args.log_every) == 0 or step == start:
            log(f"{args.configs} step {step+1}/{args.steps} loss={last_loss:.4f} "
                f"min={(time.time()-started)/60:.1f} lr={sched.get_last_lr()[0]:.2e}")
        if args.save_every and (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            save(step + 1, final=False)

    save(args.steps, final=True)

    if args.skip_infer:
        return 0

    # ---- inference (recipe cell 5 + 6) ----
    import re
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    def build(row):
        return TEMPLATE.format(instruction=INSTRUCTION, input=str(row["source"]))

    pred: dict[str, str] = {}
    for max_new, part in eval_frame.groupby("max_new"):
        log(f"generate max_new={max_new} rows={len(part)}")
        for s0 in range(0, len(part), 8):
            batch = part.iloc[s0:s0 + 8]
            enc = tokenizer([build(r) for _, r in batch.iterrows()], return_tensors="pt",
                            padding=True, truncation=True, max_length=1024).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=int(max_new), do_sample=False,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for eid, text in zip(batch.eval_id, texts):
                pred[eid] = text.strip().split("###")[0].strip()

    def detok(text: str) -> str:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
        t = re.sub(r"\s+(?=[，。！？；：、）】》])", "", t)
        t = re.sub(r"(?<=[（【《])\s+", "", t)
        return t

    label = f"ecomgpt_{args.dataset.lower()}" + ("" if args.configs == "natural" else f"_{args.configs}")
    hyp = eval_frame[["eval_id", "dataset", "pair_type", "source", "reference"]].copy()
    hyp["hypothesis"] = [detok(pred.get(eid, "")) for eid in eval_frame.eval_id]
    hyp["config"] = label
    hyp["seed"] = args.seed
    dst = out / f"hyp_{args.configs}_seed{args.seed}.parquet"
    hyp.to_parquet(dst, index=False)
    info = {"rows": int(len(hyp)), "empty": int(hyp.hypothesis.str.strip().eq("").sum()),
            "minutes": round((time.time() - started) / 60, 1), "steps": args.steps,
            "start_step": start, "config": label}
    (out / f"run_{args.configs}_seed{args.seed}.json").write_text(json.dumps(info, indent=2))
    log(f"WROTE {dst} {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
