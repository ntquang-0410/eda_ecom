#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Qwen3-4B LoRA trên TPU v5e-8 cho thực nghiệm weight.

Thiết kế công bằng:
  - cùng eval set, cùng id, cùng prompt
  - cùng token budget (5M) và cùng số optimizer step (625)
  - mỗi step đúng ~8000 token, chia đều cho 8 chip
  - chỉ khác phân bố token theo group (natural/balanced/...)

Chạy trong Kaggle TPU kernel:
  python tpu_weight_train.py --seed 42 --out /kaggle/working/weight_runs
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
BUCKETS = (64, 128, 256, 512, 1024)
MODEL_PATH = ""
TRAIN = None
EVAL_1688 = None
EVAL_TIKI = None
OUT_DIR = Path("/kaggle/working/weight_runs")
SEED = 42
STEPS = 625
BUDGET = 5_000_000
MAX_SEQ = 1024
MICRO_BATCH = 2
WORLD = 8
SKIP_GENERATION = False


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
        draw = rng.choice(n, size=8192, p=prob)
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


def build_plan(train: pd.DataFrame, config: dict | None, seed: int,
               budget: int, steps: int, world: int) -> tuple[list[list[int]], float]:
    stream, total = sample_indices(train, config, seed, budget)
    target = budget / steps
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0.0
    cursor = 0
    while len(groups) < steps:
        if cursor >= len(stream):
            cursor = 0
        idx = stream[cursor]
        cursor += 1
        current.append(idx)
        current_tokens += float(train["eff_tokens"].iloc[idx])
        if current_tokens >= target and len(current) >= world:
            groups.append(current)
            current, current_tokens = [], 0.0
    if current and len(groups) == steps:
        groups[-1].extend(current)
    actual = sum(float(train["eff_tokens"].iloc[idx]) for group in groups for idx in group)
    return groups, actual


def assign_ranks(step: list[int], train: pd.DataFrame, world: int) -> list[list[int]]:
    eff = train["eff_tokens"].to_numpy(dtype=float)
    bins: list[list[int]] = [[] for _ in range(world)]
    loads = [0.0] * world
    counts = [0] * world
    for idx in sorted(step, key=lambda i: float(eff[i]), reverse=True):
        rank = min(range(world), key=lambda r: (loads[r], counts[r], r))
        bins[rank].append(int(idx))
        loads[rank] += float(eff[idx])
        counts[rank] += 1
    return bins


def build_prompt(tokenizer, source: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": str(source or "").strip()}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def encode_row(tokenizer, row, max_seq: int):
    prompt = build_prompt(tokenizer, row["source_text"])
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


def collate(items, pad_id: int):
    length = max(len(x[0]) for x in items)
    length = min([b for b in BUCKETS if b >= length] or [BUCKETS[-1]])
    ids, labs, mask = [], [], []
    for seq, lab in items:
        pad = length - len(seq)
        ids.append(seq + [pad_id] * pad)
        labs.append(lab + [-100] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long)}


def generate_rows(model, tokenizer, frame: pd.DataFrame, rank: int, world: int,
                  device, max_seq: int):
    part = frame.iloc[rank::world].copy().reset_index(drop=True)
    part["prompt"] = [build_prompt(tokenizer, s) for s in part["source"]]
    part["prompt_tokens"] = part["prompt"].map(
        lambda x: len(tokenizer(x, add_special_tokens=False).input_ids))
    part = part[part["prompt_tokens"] < max_seq - 8]
    out_map: dict[str, str] = {}
    for max_new, group in part.groupby("max_new"):
        records = group.sort_values("prompt_tokens").to_dict("records")
        batch_size = 1 if int(max_new) > 300 else 4
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            enc = tokenizer([r["prompt"] for r in chunk], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=max_seq).to(device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc, max_new_tokens=int(max_new), do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(
                gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, text in zip(chunk, texts):
                out_map[row["eval_id"]] = text.strip()
    return [{"eval_id": eid, "hypothesis": out_map.get(eid, "")}
            for eid in part["eval_id"]]


def _mp_fn(index: int) -> None:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world = xr.global_ordinal(), xr.world_size()
    device = xm.xla_device()
    log(f"rank {rank}/{world} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = None

    plan = json.loads((OUT_DIR / f"plan_seed{SEED}.json").read_text(encoding="utf-8"))
    runs = list(plan.keys())

    for run_idx, run_name in enumerate(runs):
        config_name, seed_text = run_name.split("::")
        seed = int(seed_text)
        steps_data = plan[run_name]
        # mọi rank phải khởi tạo LoRA giống nhau
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                              task_type="CAUSAL_LM", target_modules="all-linear")
        adapter_name = f"a{run_idx}"
        if model is None:
            model = get_peft_model(base, lora_cfg, adapter_name=adapter_name)
        else:
            model.add_adapter(adapter_name, lora_cfg)
        model.set_adapter(adapter_name)
        model.to(device)
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        warmup = 20
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: min(1.0, (s + 1) / warmup) *
            max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / STEPS)))))

        started = time.time()
        total_tokens = 0.0
        for step, step_indices in enumerate(steps_data):
            optimizer.zero_grad(set_to_none=True)
            local = assign_ranks(step_indices, TRAIN, world)[rank]
            step_tokens = sum(float(TRAIN["eff_tokens"].iloc[idx]) for idx in step_indices)
            local_loss = 0.0
            local_tokens = 0
            for start in range(0, len(local), MICRO_BATCH):
                micro = local[start:start + MICRO_BATCH]
                items = []
                for idx in micro:
                    item = encode_row(tokenizer, TRAIN.iloc[idx], MAX_SEQ)
                    if item is not None:
                        items.append(item)
                if not items:
                    continue
                batch = {k: v.to(device) for k, v in collate(items, tokenizer.pad_token_id).items()}
                out = model(**batch)
                n_tok = int((batch["labels"] != -100).sum().item())
                loss = out.loss * (n_tok / max(1.0, step_tokens))
                loss.backward()
                xm.mark_step()
                local_loss += float(out.loss.item()) * n_tok
                local_tokens += n_tok
            xm.optimizer_step(optimizer, barrier=True)
            scheduler.step()
            xm.mark_step()
            total_tokens += step_tokens
            if step % 25 == 0 and rank == 0:
                log(f"{config_name} seed={seed} step {step}/{STEPS} "
                    f"loss={local_loss / max(1, local_tokens):.4f} "
                    f"tokens={total_tokens/1e6:.2f}M "
                    f"min={(time.time()-started)/60:.1f}")

        if rank == 0:
            adapter_dir = OUT_DIR / f"adapter_{config_name}_seed{seed}"
            model.save_pretrained(adapter_dir)
            log(f"saved {adapter_dir}")
        xm.rendezvous(f"save_{run_idx}")

        if SKIP_GENERATION:
            if rank == 0:
                (OUT_DIR / f"run_{config_name}_seed{seed}.json").write_text(
                    json.dumps({"config": config_name, "seed": seed,
                                "steps": len(steps_data), "tokens": round(total_tokens),
                                "minutes": round((time.time()-started)/60, 1),
                                "generation": "skipped"}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            xm.rendezvous(f"done_{run_idx}")
            model.delete_adapter(adapter_name)
            gc.collect()
            continue

        # generation
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        shard_dir = OUT_DIR / "hyp_shards"
        shard_dir.mkdir(exist_ok=True)
        for dataset_name, frame in (("1688", EVAL_1688), ("tiki", EVAL_TIKI)):
            records = generate_rows(model, tokenizer, frame, rank, world,
                                    device, MAX_SEQ)
            for record in records:
                record["dataset"] = dataset_name
                record["config"] = config_name
                record["seed"] = seed
            pd.DataFrame(records).to_parquet(
                shard_dir / f"{config_name}_{seed}_{dataset}_r{rank}.parquet",
                index=False)
        xm.rendezvous(f"gen_{run_idx}")

        if rank == 0:
            for dataset_name in ("1688", "tiki"):
                parts = []
                for r in range(world):
                    path = shard_dir / f"{config_name}_{seed}_{dataset_name}_r{r}.parquet"
                    if path.exists():
                        parts.append(pd.read_parquet(path))
                merged = pd.concat(parts, ignore_index=True).sort_values("eval_id")
                frame = (EVAL_1688 if dataset_name == "1688" else EVAL_TIKI)
                merged = frame[["eval_id", "dataset", "pair_type", "product_id", "source", "reference"]].merge(
                    merged[["eval_id", "hypothesis", "config", "seed"]], on="eval_id", how="left")
                merged["hypothesis"] = merged["hypothesis"].fillna("")
                merged["dataset"] = dataset_name
                merged.to_parquet(
                    OUT_DIR / f"hyp_{config_name}_seed{seed}_{dataset_name}.parquet",
                    index=False)
            log(f"merged {config_name} seed={seed}")

        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if rank == 0:
            (OUT_DIR / f"run_{config_name}_seed{seed}.json").write_text(
                json.dumps({"config": config_name, "seed": seed,
                            "steps": len(steps_data), "tokens": round(total_tokens),
                            "minutes": round((time.time()-started)/60, 1)},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        xm.rendezvous(f"done_{run_idx}")

        # giải phóng adapter cũ, giữ base
        model.delete_adapter(adapter_name)
        gc.collect()

    log(f"rank {rank} finished")


def find_model() -> str:
    try:
        import kagglehub
        return kagglehub.model_download("qwen-lm/qwen-3/Transformers/4b/1")
    except Exception as error:
        log(f"kagglehub failed: {error}")
    from huggingface_hub import snapshot_download
    return snapshot_download("Qwen/Qwen3-4B")


def main() -> int:
    global MODEL_PATH, TRAIN, EVAL_1688, EVAL_TIKI, OUT_DIR, SEED, STEPS, BUDGET, SKIP_GENERATION
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/kaggle/working/weight_eval")
    ap.add_argument("--out", default="/kaggle/working/weight_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--steps", type=int, default=625)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.configs = ["natural"]
        args.steps = 2
        args.budget = 16_000
        args.skip_generation = True
        args.out = "/kaggle/working/smoke_runs"
    SKIP_GENERATION = bool(args.skip_generation)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED, STEPS, BUDGET = args.seed, args.steps, args.budget

    TRAIN = pd.read_parquet(Path(args.eval_dir) / "train_view_4b.parquet")
    TRAIN = TRAIN[TRAIN["eff_tokens"] <= 960].reset_index(drop=True)
    EVAL_1688 = pd.read_parquet(Path(args.eval_dir) / "eval_1688.parquet")
    EVAL_TIKI = pd.read_parquet(Path(args.eval_dir) / "eval_tiki.parquet")

    plan = {}
    for config_name in args.configs:
        groups, actual = build_plan(TRAIN, CONFIGS[config_name], SEED, BUDGET,
                                    STEPS, WORLD)
        plan[f"{config_name}::{SEED}"] = groups
        log(f"plan {config_name}: {len(groups)} steps, {actual/1e6:.2f}M tokens")
    (OUT_DIR / f"plan_seed{SEED}.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        return 0

    MODEL_PATH = find_model()
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=(), nprocs=WORLD, start_method="fork")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
