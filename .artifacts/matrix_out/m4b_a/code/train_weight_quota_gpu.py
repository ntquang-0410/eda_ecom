#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train weight survey A/B bằng quota-based sampling.

Mỗi config:
  - quota theo group: T_g = s_g * 4M
  - sample đều trong group đến đủ T_g
  - 500 optimizer step, ~8k token/step
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
from quota_sampling import balance_plan  # noqa: E402
from tpu_weight_train import collate, encode_row, generate_rows  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_one(args, config_name: str, config: dict | None, seed: int,
              train: pd.DataFrame, tokenizer, eval_frame: pd.DataFrame,
              out_dir: Path) -> dict:
    plan, actual = balance_plan(train, config, seed, args.tokens, args.steps)
    log(f"{config_name}: {len(plan)} steps, actual share "
        f"{ {k: round(v,3) for k,v in actual.items()} }")

    # Lưu eff_tokens thực tế đã sample cho từng cell để kiểm tra chéo
    sample_rows = []
    for step, idxs in enumerate(plan):
        for i in idxs:
            sample_rows.append({
                "step": step,
                "row_index": int(i),
                "pair_id": str(train["pair_id"].iloc[i]) if "pair_id" in train.columns else None,
                "group": str(train["group"].iloc[i]),
                "eff_tokens": float(train["eff_tokens"].iloc[i]),
            })
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_parquet(out_dir / f"sample_{config_name}_seed{seed}.parquet", index=False)
    sampled_group = {str(g): round(float(v), 3)
                     for g, v in sample_df.groupby("group")["eff_tokens"].sum().items()}

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True,
            local_files_only=True, low_cpu_mem_usage=True)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True,
            local_files_only=True, low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(
        base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         task_type="CAUSAL_LM", target_modules="all-linear"))
    model.to("cuda").train()
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
    autocast = torch.float16 if use_fp16 else torch.bfloat16
    started = time.time()
    for step, idxs in enumerate(plan):
        optimizer.zero_grad(set_to_none=True)
        step_tokens = sum(float(train["eff_tokens"].iloc[i]) for i in idxs)
        for start in range(0, len(idxs), args.micro_batch):
            micro = idxs[start:start + args.micro_batch]
            items = []
            for idx in micro:
                item = encode_row(tokenizer, train.iloc[idx], args.max_seq)
                if item is not None:
                    items.append(item)
            if not items:
                continue
            batch = {k: v.to("cuda") for k, v in collate(items, tokenizer.pad_token_id).items()}
            with torch.autocast("cuda", dtype=autocast):
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
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        if step % 25 == 0:
            log(f"{config_name} step {step}/{args.steps} min={(time.time()-started)/60:.1f}")

    adapter = out_dir / f"adapter_{config_name}_seed{seed}"
    model.save_pretrained(adapter)

    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache = True
    records = generate_rows(model, tokenizer, eval_frame, 0, 1, "cuda", args.max_seq)
    part = eval_frame[["eval_id", "dataset", "pair_type", "source", "reference"]].merge(
        pd.DataFrame(records), on="eval_id", how="left")
    part["hypothesis"] = part["hypothesis"].fillna("")
    part["config"] = config_name
    part["seed"] = seed
    part.to_parquet(out_dir / f"hyp_{config_name}_seed{seed}.parquet", index=False)
    info = {"config": config_name, "seed": seed, "steps": len(plan),
            "tokens": round(sum(float(train["eff_tokens"].iloc[i]) for s in plan for i in s)),
            "sampled_eff_tokens_total": round(float(sample_df["eff_tokens"].sum()), 3),
            "sampled_eff_tokens_group": sampled_group,
            "sampled_unique_rows": int(sample_df["row_index"].nunique()),
            "train_rows": int(len(train)),
            "train_tokens_million": round(float(train["eff_tokens"].sum()) / 1e6, 3),
            "epochs_seen": round(float(sample_df["eff_tokens"].sum())
                                 / max(1.0, float(train["eff_tokens"].sum())), 4),
            "minutes": round((time.time()-started)/60, 1),
            "actual_share": actual}
    (out_dir / f"run_{config_name}_seed{seed}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done {info}")
    del model, base, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/kaggle/input/models/qwen-lm/qwen-3/transformers/4b/1")
    ap.add_argument("--dataset", choices=("A", "B"), required=True)
    ap.add_argument("--data-root", default="/kaggle/input/tiki-weight-ab-processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="", help="danh sách config, phân tách bằng dấu phẩy")
    ap.add_argument("--tokens", type=int, default=4_000_000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max-seq", type=int, default=960)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = ap.parse_args()

    data = Path(args.data_root)
    train = pd.read_parquet(data / f"processed_{args.dataset}/units.parquet")
    eval_frame = pd.read_parquet(data / f"eval_{args.dataset}.parquet")
    meta = json.loads((data / f"meta_{args.dataset}.json").read_text(encoding="utf-8"))
    configs = meta["configs"]
    if args.configs:
        want = [c.strip() for c in args.configs.split(",") if c.strip()]
        configs = {k: v for k, v in configs.items() if k in want}
        missing = set(want) - set(configs)
        if missing:
            raise SystemExit(f"config không có trong meta: {sorted(missing)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, cfg in configs.items():
        results.append(train_one(args, name, cfg, args.seed, train, tokenizer, eval_frame, out))
        (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
