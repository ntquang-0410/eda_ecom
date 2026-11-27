#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh lại hypothesis cho các dòng eval bị guard max_seq loại (audit 858).

Chạy trên GPU:
  python tools/reinfer_long_rows.py --model Qwen/Qwen3-4B \
      --adapter /content/work/ckpt_a42 --out /content/work/extra42.parquet \
      --eval-max-seq 1280 --dataset tiki

Chỉ sinh cho các dòng có prompt_tokens >= --threshold (mặc định 952) — tức đúng
những dòng bị `generate_rows` bỏ trước đây. Không đụng các dòng khác.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from tpu_weight_train import build_prompt, generate_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=str(ROOT / "data/cleaned/eval_paired_858.parquet"))
    ap.add_argument("--dataset", default="tiki")
    ap.add_argument("--eval-max-seq", type=int, default=1280)
    ap.add_argument("--threshold", type=int, default=952)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    idx = pd.read_parquet(args.index)
    idx["eval_id"] = idx.eval_id.astype(str)
    idx = idx[idx.dataset == args.dataset].copy()
    idx["prompt_tokens"] = [len(tok(build_prompt(tok, s), add_special_tokens=False).input_ids)
                            for s in idx.source]
    todo = idx[idx.prompt_tokens >= args.threshold].copy()
    print(f"rows cần sinh: {len(todo)} / {len(idx)} (max prompt {int(todo.prompt_tokens.max()) if len(todo) else 0})",
          flush=True)
    if not len(todo):
        print("không có dòng nào", flush=True)
        return

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.to("cuda").eval()
    model.config.use_cache = True

    frame = todo[["eval_id", "source", "max_new"]].reset_index(drop=True)
    recs = generate_rows(model, tok, frame, 0, 1, "cuda", args.eval_max_seq, truncate=True)
    out = pd.DataFrame(recs)
    out["prompt_tokens"] = todo.prompt_tokens.to_numpy()
    out.to_parquet(args.out, index=False)
    print(out[["eval_id", "prompt_tokens"]].assign(
        hyp_len=out.hypothesis.astype(str).str.len(),
        empty=out.hypothesis.astype(str).str.strip().eq("")).to_string(index=False), flush=True)
    print(json.dumps({"rows": len(out), "empty": int(out.hypothesis.astype(str).str.strip().eq("").sum()),
                      "eval_max_seq": args.eval_max_seq, "adapter": args.adapter}), flush=True)


if __name__ == "__main__":
    main()
