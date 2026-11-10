#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Qwen3 LoRA theo 5 cấu hình weight trên GPU T4 (hoặc GPU đơn).

Dùng cho pilot / fallback khi TPU queue lâu. Mỗi run:
  - sample đúng token budget theo weight
  - train LoRA
  - sinh hypothesis trên eval_1688 + eval_tiki cố định
  - lưu adapter + hyp_<config>_seed<seed>.parquet

Ví dụ:
  python tools/train_weight_t4.py \
    --model /content/models/Qwen3-1.7B \
    --train-view data/cleaned/weight_eval/train_view_1.7b.parquet \
    --configs natural balanced no_desc tiki_heavy title_heavy \
    --seeds 42 --tokens 1000000 --out /content/runs/t4_pilot
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

CONFIGS = {
    "natural": None,
    "balanced": {"title_1688": 0.15, "attr_1688": 0.35,
                 "desc_1688": 0.25, "tiki": 0.25},
    "no_desc": {"title_1688": 0.25, "attr_1688": 0.45,
                "desc_1688": 0.00, "tiki": 0.30},
    "tiki_heavy": {"title_1688": 0.10, "attr_1688": 0.25,
                   "desc_1688": 0.20, "tiki": 0.45},
    "title_heavy": {"title_1688": 0.30, "attr_1688": 0.30,
                    "desc_1688": 0.15, "tiki": 0.25},
}
SYSTEM = ("You are a professional Vietnamese-to-Simplified Chinese e-commerce translator. "
          "Translate the user text into Simplified Chinese. Preserve numbers, units and brand names. "
          "Return only the translation, no explanation.")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sample_indices(train: pd.DataFrame, config: dict | None, seed: int,
                   budget: int) -> tuple[list[int], float]:
    rng = np.random.default_rng(seed)
    n = len(train)
    eff = train["eff_tokens"].to_numpy(dtype=float)
    groups = train["group"].to_numpy()
    mass = pd.Series(eff).groupby(groups).sum()
    if config is None:
        weights = {g: 1.0 for g in mass.index}
    else:
        weights = {g: float(config.get(g, 0.0)) / max(1.0, float(mass[g]))
                   for g in mass.index}
    raw = np.array([weights.get(g, 0.0) for g in groups]) * eff
    prob = raw / raw.sum()
    chosen, total = [], 0.0
    while total < budget:
        draw = rng.choice(n, size=4096, p=prob)
        stop = False
        for idx in draw:
            value = float(eff[idx])
            if total + value > budget:
                stop = True
                break
            chosen.append(int(idx))
            total += value
        if stop:
            break
    return chosen, total


def user_text(row) -> str:
    return str(row["source_text"] or "").strip()


def build_prompt(tokenizer, source: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": source}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def encode_row(tokenizer, row, max_seq: int):
    prompt = build_prompt(tokenizer, user_text(row))
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(prompt_ids) >= max_seq - 8:
        return None
    target = str(row["target_text"] or "").strip()
    if not target:
        return None
    full = tokenizer(prompt + target + tokenizer.eos_token,
                     add_special_tokens=False).input_ids[:max_seq]
    if len(full) <= len(prompt_ids) + 1:
        return None
    labels = [-100] * len(prompt_ids) + full[len(prompt_ids):]
    return full, labels


class TokenDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_seq: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.cache: dict[int, tuple | None] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if index not in self.cache:
            self.cache[index] = encode_row(self.tokenizer, self.rows[index],
                                           self.max_seq)
        item = self.cache[index]
        if item is None:
            # fallback: dùng một dòng ngắn khác
            for offset in range(1, len(self.rows)):
                alt = (index + offset) % len(self.rows)
                candidate = encode_row(self.tokenizer, self.rows[alt],
                                       self.max_seq)
                if candidate is not None:
                    return candidate
            raise RuntimeError("không encode được dòng nào")
        return item


def collate(batch, pad_id: int):
    length = max(len(x[0]) for x in batch)
    ids, labs, mask = [], [], []
    for seq, lab in batch:
        pad = length - len(seq)
        ids.append(seq + [pad_id] * pad)
        labs.append(lab + [-100] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return {"input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(mask)}


def generate_rows(model, tokenizer, frame: pd.DataFrame, batch_size: int,
                  max_seq: int) -> list[str]:
    model.eval()
    out_map: dict[str, str] = {}
    original = frame.copy()
    original["prompt"] = [build_prompt(tokenizer, str(s)) for s in original["source"]]
    original["prompt_tokens"] = original["prompt"].map(
        lambda x: len(tokenizer(x, add_special_tokens=False).input_ids))
    frame = original[original["prompt_tokens"] < max_seq - 8]
    for max_new, part in frame.groupby("max_new"):
        part = part.sort_values("prompt_tokens")
        records = part.to_dict("records")
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            prompts = [r["prompt"] for r in chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq).to("cuda")
            with torch.inference_mode():
                gen = model.generate(
                    **enc, max_new_tokens=int(max_new), do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(
                gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, text in zip(chunk, texts):
                out_map[row["eval_id"]] = text.strip()
            if start % (batch_size * 20) == 0:
                log(f"  generate max_new={max_new}: {start + len(chunk)}/{len(records)}")
    return [out_map.get(x, "") for x in original["eval_id"]]


def train_one(args, config_name: str, seed: int, train: pd.DataFrame,
              tokenizer, eval_1688: pd.DataFrame, eval_tiki: pd.DataFrame,
              out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = CONFIGS[config_name]
    chosen, tokens = sample_indices(train, config, seed, args.tokens)
    log(f"run {config_name} seed={seed}: {len(chosen)} examples, {tokens:.0f} tokens")

    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=model_dtype, trust_remote_code=True,
            low_cpu_mem_usage=True)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=model_dtype, trust_remote_code=True,
            low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(
        base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         task_type="CAUSAL_LM", target_modules="all-linear"))
    model.to("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    use_fp16 = args.dtype == "fp16"
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    rows = train.iloc[chosen].to_dict("records")
    dataset = TokenDataset(rows, tokenizer, args.max_seq)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                        num_workers=0)

    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = steps_per_epoch
    warmup = max(1, int(total_steps * 0.03))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup) *
        max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    step = 0
    started = time.time()
    for batch in loader:
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=autocast_dtype):
            loss = model(**batch).loss / args.accum
        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        micro += 1
        if micro >= args.accum:
            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if step % 25 == 0:
                elapsed = (time.time() - started) / 60
                log(f"  step {step} | loss {float(loss.item() * args.accum):.4f} "
                    f"| {elapsed:.1f} min | {tokens * step / max(1, total_steps) / 1e6:.2f}M tok")
            if args.max_steps and step >= args.max_steps:
                break
    train_min = (time.time() - started) / 60

    adapter_dir = out_dir / f"adapter_{config_name}_seed{seed}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    hyp_path = out_dir / f"hyp_{config_name}_seed{seed}.parquet"
    if not args.no_generate:
        frames = []
        for dataset_name, frame in (("1688", eval_1688), ("tiki", eval_tiki)):
            hyps = generate_rows(model, tokenizer, frame, args.gen_batch,
                                 args.max_seq)
            part = frame.copy()
            part["hypothesis"] = hyps
            part["dataset"] = dataset_name
            part["config"] = config_name
            part["seed"] = seed
            frames.append(part)
        pd.concat(frames, ignore_index=True).to_parquet(hyp_path, index=False)

    info = {"config": config_name, "seed": seed, "examples": len(chosen),
            "tokens": round(tokens), "steps": step, "train_min": round(train_min, 2),
            "hyp": str(hyp_path) if hyp_path.exists() else None}
    (out_dir / f"run_{config_name}_seed{seed}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done {config_name} seed={seed}: {info}")

    del model, base, optimizer, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/content/models/Qwen3-1.7B")
    ap.add_argument("--train-view", default="data/cleaned/weight_eval/train_view_1.7b.parquet")
    ap.add_argument("--eval-dir", default="data/cleaned/weight_eval")
    ap.add_argument("--out", default="/content/runs/t4_pilot")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.train_view)
    eval_1688 = pd.read_parquet(Path(args.eval_dir) / "eval_1688.parquet")
    eval_tiki = pd.read_parquet(Path(args.eval_dir) / "eval_tiki.parquet")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              local_files_only=Path(args.model).exists())
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
