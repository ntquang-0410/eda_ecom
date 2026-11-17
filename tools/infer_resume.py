#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference Qwen3-4B từ adapter, có checkpoint từng batch (resume được).

Khác bản `infer_a123_local.py`:
  - ghi JSONL sau mỗi batch -> ngắt GPU giữa chừng không mất kết quả
  - resume: bỏ qua các eval_id đã có trong JSONL
  - batch lớn hơn cho prompt ngắn; description dùng batch 4 với left padding
    (greedy nên kết quả tương đương batch 1, chỉ nhanh hơn)

Ví dụ:
  python tools/infer_resume.py \
    --adapter /content/dl/a123/out_A_qwen4b_natural/adapter_natural_seed123 \
    --eval /content/dl/a123/matrix_data/eval_A.parquet \
    --out .artifacts/hyp/hyp_qwen4b_a_seed123.parquet \
    --state .artifacts/hyp/a123_rows.jsonl --seed 123 --config natural
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from tpu_weight_train import BUCKETS, build_prompt  # noqa: E402


def load_state(path: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[row["eval_id"]] = row["hypothesis"]
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--config", default="natural")
    ap.add_argument("--max-seq", type=int, default=960)
    ap.add_argument("--batch-long", type=int, default=4)
    ap.add_argument("--batch-short", type=int, default=16)
    ap.add_argument("--padding-side", default="right",
                    choices=["right", "left"],
                    help="right = giống protocol train (padding='max_length'); left = nhanh/chuẩn cho decoder-only")
    args = ap.parse_args()

    state = Path(args.state)
    state.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_state(state)
    print("resume: đã có", len(done), "dòng", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = args.padding_side
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                low_cpu_mem_usage=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda").eval()
    print("model ready", flush=True)

    ev = pd.read_parquet(args.eval)
    part = ev[~ev.eval_id.isin(done)].copy()
    part["prompt"] = [build_prompt(tok, s) for s in part["source"]]
    part["prompt_tokens"] = part["prompt"].map(
        lambda x: len(tok(x, add_special_tokens=False).input_ids))
    part = part[part["prompt_tokens"] < args.max_seq - 8]
    part["bucket"] = part["prompt_tokens"].map(
        lambda n: min([b for b in BUCKETS if b >= max(16, int(n))] or [BUCKETS[-1]]))
    print("cần sinh", len(part), "dòng", flush=True)

    t0 = time.time()
    with state.open("a", encoding="utf-8") as fh:
        for (max_new, bucket), group in part.groupby(["max_new", "bucket"]):
            recs = group.sort_values("prompt_tokens").to_dict("records")
            bs = args.batch_short if int(max_new) <= 300 else args.batch_long
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
                    rec = {"eval_id": row["eval_id"], "hypothesis": text.strip()}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if k % (bs * 10) == 0:
                    print("max_new", max_new, "bucket", bucket, k, "/", len(recs),
                          "min", round((time.time() - t0) / 60, 1), flush=True)

    done = load_state(state)
    hyp = ev[["eval_id", "dataset", "pair_type", "source", "reference"]].copy()
    hyp["hypothesis"] = hyp.eval_id.map(done).fillna("")
    hyp["config"] = args.config
    hyp["seed"] = args.seed
    hyp.to_parquet(out, index=False)
    info = {"rows": int(len(hyp)), "generated": int(hyp.hypothesis.str.strip().ne("").sum()),
            "empty": int(hyp.hypothesis.str.strip().eq("").sum()),
            "state_rows": len(done), "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(out).replace(".parquet", "_run.json")).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
