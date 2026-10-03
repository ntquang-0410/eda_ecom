# -*- coding: utf-8 -*-
"""Execute every code cell of the patched notebook locally, with stubs for what the
local venv lacks (torch/transformers/matplotlib): the model is bge-m3 served by the
local Ollama (verified to match the notebook's transformers scores to ~1e-3)."""
import hashlib, json, re, sys, time, urllib.request
from collections import Counter
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NB, N_ROWS = sys.argv[1], int(sys.argv[2])

class _Any:
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __getitem__(self, k): return _Any()
    def __iter__(self): return iter([_Any(), _Any()])

def display(x):
    if hasattr(x, "head"):
        with pd.option_context("display.max_columns", 20, "display.width", 200, "display.max_colwidth", 50):
            print(x.head(12).to_string())
    else:
        print(x)

class _T:
    def __init__(self, a): self.a = a
    def numpy(self): return self.a

def _post(batch):
    body = json.dumps({"model": "bge-m3", "input": batch}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["embeddings"]

bad_inputs = []
def _embed_batch(batch):
    try:
        return _post(batch)
    except urllib.error.HTTPError as e:
        if len(batch) > 1:                         # split to isolate the string Ollama rejects
            mid = len(batch) // 2
            return _embed_batch(batch[:mid]) + _embed_batch(batch[mid:])
        msg = e.read().decode("utf-8", "replace")[:200]
        bad_inputs.append((batch[0], e.code, msg))
        print(f"[harness] Ollama {e.code} on {batch[0]!r}: {msg}", flush=True)
        return _post(["."])                        # neutral placeholder so the run can go on

n_embedded = [0]
def encode(texts, max_length=512, batch_size=64):
    # Ollama rejects over-long inputs (HTTP 500) where transformers truncates at max_length=512;
    # cut long strings here so the harness mimics that truncation.
    texts = [" " if not t else str(t)[:480] for t in ([texts] if isinstance(texts, str) else texts)]
    out = []
    for i in range(0, len(texts), 64):
        out.extend(_embed_batch(texts[i:i + 64]))
    n_embedded[0] += len(texts)
    e = np.asarray(out, dtype=np.float32)
    return _T(e / np.linalg.norm(e, axis=1, keepdims=True))

local_parquet = str(ROOT / "data" / "snapshot" / "bilingual_zh_vi.parquet")
g = {"np": np, "pd": pd, "re": re, "hashlib": hashlib, "Counter": Counter, "plt": _Any(),
     "display": display, "encode": encode, "PARQUET_PATH": local_parquet, "__name__": "__main__"}

nb = json.load(open(NB, encoding="utf-8"))
t0 = time.time()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith(("# Cài thư viện", "import re")):
        continue                                   # %pip / torch+transformers imports
    if "AutoModel.from_pretrained" in src:
        src = "def encode_np" + src.split("def encode_np", 1)[1]   # keep only the helper
    print(f"\n######## CELL {i}: {src.strip().splitlines()[0][:70]}", flush=True)
    exec(compile(src, f"cell{i}", "exec"), g)
    if "df = load_parquet" in src:
        g["df"] = g["df"].sample(N_ROWS, random_state=1).reset_index(drop=True)
        print(f"[harness] df subsampled to {len(g['df'])} rows", flush=True)
print(f"\n[harness] ALL CELLS OK in {time.time() - t0:.0f}s, {n_embedded[0]} texts embedded, {len(bad_inputs)} inputs rejected by Ollama")
for s_, code, msg in bad_inputs[:50]:
    print(f"   rejected {code}: {s_!r} | {msg}")
