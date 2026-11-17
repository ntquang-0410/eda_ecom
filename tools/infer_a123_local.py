#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-only Qwen3-4B A seed 123 trên A100 (Colab), dùng adapter đã train.

Mirror đúng generate_rows của tools/tpu_weight_train.py:
  - build_prompt (chat template, enable_thinking=False)
  - bucket 64/128/256/512/1024, padding max_length=bucket
  - batch 1 nếu max_new > 300, ngược lại 4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/content/eda_ecom")
sys.path.insert(0, str(ROOT / "tools"))
from tpu_weight_train import BUCKETS, build_prompt  # noqa: E402

EVAL = Path("/content/dl/a123/matrix_data/eval_A.parquet")
ADAPTER = Path("/content/dl/a123/out_A_qwen4b_natural/adapter_natural_seed123")
OUT = ROOT / ".artifacts/hyp/hyp_qwen4b_a_seed123.parquet"
MAX_SEQ = 960


def main() -> int:
    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float16,
                                                low_cpu_mem_usage=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).to("cuda").eval()
    print("loaded adapter", flush=True)

    ev = pd.read_parquet(EVAL)
    t0 = time.time()
    part = ev.copy()
    part["prompt"] = [build_prompt(tok, s) for s in part["source"]]
    part["prompt_tokens"] = part["prompt"].map(
        lambda x: len(tok(x, add_special_tokens=False).input_ids))
    part = part[part["prompt_tokens"] < MAX_SEQ - 8]
    part["bucket"] = part["prompt_tokens"].map(
        lambda n: min([b for b in BUCKETS if b >= max(16, int(n))] or [BUCKETS[-1]]))
    print("to generate", len(part), "of", len(ev), flush=True)

    out_map: dict[str, str] = {}
    for (max_new, bucket), group in part.groupby(["max_new", "bucket"]):
        recs = group.sort_values("prompt_tokens").to_dict("records")
        bs = 1 if int(max_new) > 300 else 4
        for k in range(0, len(recs), bs):
            chunk = recs[k:k + bs]
            enc = tok([r["prompt"] for r in chunk], return_tensors="pt",
                      padding="max_length", truncation=True,
                      max_length=int(bucket)).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=int(max_new), do_sample=False,
                                     eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            for row, text in zip(chunk, texts):
                out_map[row["eval_id"]] = text.strip()
            if k % 40 == 0:
                print("max_new", max_new, "bucket", bucket, k, "/", len(recs),
                      "min", round((time.time() - t0) / 60, 1), flush=True)

    hyp = ev[["eval_id", "dataset", "pair_type", "source", "reference"]].copy()
    hyp["hypothesis"] = hyp.eval_id.map(out_map).fillna("")
    hyp["config"] = "natural"
    hyp["seed"] = 123
    hyp.to_parquet(OUT, index=False)
    info = {"rows": int(len(hyp)), "generated": int(len(out_map)),
            "empty": int(hyp.hypothesis.str.strip().eq("").sum()),
            "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / ".artifacts/hyp/run_a_s123_infer.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
