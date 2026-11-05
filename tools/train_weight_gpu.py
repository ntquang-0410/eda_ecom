#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train weight experiment trên GPU đơn (RTX Pro 6000 / A100 / T4).

Khác `train_weight_t4.py`: dùng đúng scheduler công bằng của bản TPU:
  - 5M token
  - 625 optimizer step
  - mỗi step ~8.000 token
  - chỉ khác phân bố group

Ví dụ:
  python code/train_weight_gpu.py \
    --model Qwen/Qwen3-4B \
    --train-view data/weight_eval/train_view_4b.parquet \
    --eval-dir data/weight_eval \
    --configs natural balanced no_desc tiki_heavy title_heavy \
    --seeds 42 123 --out /runs/weight_4b
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tpu_weight_train import (  # noqa: E402
    BUCKETS, CONFIGS, SYSTEM, build_plan, collate, encode_row, generate_rows,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def train_one(args, config_name: str, seed: int, train: pd.DataFrame,
              tokenizer, eval_1688: pd.DataFrame, eval_tiki: pd.DataFrame,
              out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    plan, actual = build_plan(train, CONFIGS[config_name], seed,
                              args.tokens, args.steps, 1)
    log(f"{config_name} seed={seed}: {len(plan)} steps, {actual/1e6:.2f}M tokens")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(
        base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         task_type="CAUSAL_LM", target_modules="all-linear"))
    model.to("cuda")
    model.train()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = 20
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup) *
        max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    use_fp16 = args.dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16

    started = time.time()
    total_tokens = 0.0
    for step, step_indices in enumerate(plan):
        optimizer.zero_grad(set_to_none=True)
        step_tokens = sum(float(train["eff_tokens"].iloc[i]) for i in step_indices)
        for start in range(0, len(step_indices), args.micro_batch):
            micro = step_indices[start:start + args.micro_batch]
            items = []
            for idx in micro:
                item = encode_row(tokenizer, train.iloc[idx], args.max_seq)
                if item is not None:
                    items.append(item)
            if not items:
                continue
            batch = {k: v.to("cuda") for k, v in
                     collate(items, tokenizer.pad_token_id).items()}
            with torch.autocast("cuda", dtype=autocast_dtype):
                out = model(**batch)
            n_tok = int((batch["labels"] != -100).sum().item())
            loss = out.loss * (n_tok / max(1.0, step_tokens))
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if use_fp16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        total_tokens += step_tokens
        if step % 25 == 0:
            log(f"{config_name} seed={seed} step {step}/{args.steps} "
                f"tokens={total_tokens/1e6:.2f}M min={(time.time()-started)/60:.1f}")

    adapter_dir = out_dir / f"adapter_{config_name}_seed{seed}"
    model.save_pretrained(adapter_dir)

    frames = []
    for dataset_name, frame in (("1688", eval_1688), ("tiki", eval_tiki)):
        records = generate_rows(model, tokenizer, frame, 0, 1, "cuda",
                                args.max_seq)
        part = frame[["eval_id", "dataset", "pair_type", "product_id",
                      "source", "reference"]].merge(
            pd.DataFrame(records), on="eval_id", how="left")
        part["hypothesis"] = part["hypothesis"].fillna("")
        part["config"] = config_name
        part["seed"] = seed
        part["dataset"] = dataset_name
        frames.append(part)
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / f"hyp_{config_name}_seed{seed}.parquet", index=False)

    info = {"config": config_name, "seed": seed, "steps": len(plan),
            "tokens": round(actual), "minutes": round((time.time()-started)/60, 1)}
    (out_dir / f"run_{config_name}_seed{seed}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done {info}")
    del model, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--train-view", default="data/weight_eval/train_view_4b.parquet")
    ap.add_argument("--eval-dir", default="data/weight_eval")
    ap.add_argument("--out", default="/runs/weight_4b")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--tokens", type=int, default=5_000_000)
    ap.add_argument("--steps", type=int, default=625)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.train_view)
    train = train[train["eff_tokens"] <= 960].reset_index(drop=True)
    eval_1688 = pd.read_parquet(Path(args.eval_dir) / "eval_1688.parquet")
    eval_tiki = pd.read_parquet(Path(args.eval_dir) / "eval_tiki.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    results = []
    for seed in args.seeds:
        for config_name in args.configs:
            results.append(train_one(args, config_name, seed, train, tokenizer,
                                     eval_1688, eval_tiki, out))
            (out / "summary.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
