#!/usr/bin/env python3
"""So sánh right vs left padding khi generate (ảnh hưởng tỷ lệ rỗng)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/content/eda_ecom")
sys.path.insert(0, str(ROOT / "tools"))
from tpu_weight_train import BUCKETS, build_prompt  # noqa: E402

ADAPTER = "/content/dl/adapter"
EVAL = ROOT / ".artifacts/matrix_out/m4b_a/matrix_data/eval_A.parquet"


def run(tok, model, part, side, bs, max_new):
    tok.padding_side = side
    recs = part.to_dict("records")
    empty = 0
    outs = []
    for k in range(0, len(recs), bs):
        chunk = recs[k:k + bs]
        max_len = max(int(x["bucket"]) for x in chunk)
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding="max_length",
                  truncation=True, max_length=max_len).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=int(max_new), do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for t in texts:
            outs.append(t.strip())
            empty += int(not t.strip())
    return outs, empty


def main() -> int:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float16,
                                                low_cpu_mem_usage=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER).to("cuda").eval()

    ev = pd.read_parquet(EVAL)
    ev["prompt"] = [build_prompt(tok, s) for s in ev["source"]]
    ev["prompt_tokens"] = ev["prompt"].map(lambda x: len(tok(x, add_special_tokens=False).input_ids))
    ev = ev[ev["prompt_tokens"] < 952].copy()
    ev["bucket"] = ev["prompt_tokens"].map(
        lambda n: min([b for b in BUCKETS if b >= max(16, int(n))] or [BUCKETS[-1]]))

    for label, sel, bs, mx in [
        ("SHORT title", ev[(ev.dataset == "1688") & (ev.pair_type == "title")].head(40), 4, 64),
        ("LONG desc", ev[(ev.dataset == "tiki") & (ev.pair_type == "description")].head(8), 4, 768),
    ]:
        print(f"\n=== {label} (batch {bs}) ===", flush=True)
        for side in ("right", "left"):
            outs, empty = run(tok, model, sel, side, bs, mx)
            print(f"  {side:5s}: empty {empty}/{len(outs)} | ví dụ: {outs[0][:60]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
