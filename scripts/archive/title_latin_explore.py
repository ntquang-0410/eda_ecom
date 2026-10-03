# -*- coding: utf-8 -*-
"""Explore Latin-script (non-Vietnamese) tokens in title_vi."""
import re, sys, os
from collections import Counter, defaultdict
os.chdir(r"D:\nam4_ky1\e-commerce - crawl\ecom_crawler")
sys.path.insert(0, ".")
import pandas as pd
from huggingface_hub import hf_hub_download
from config import load_settings

s = load_settings()
df = pd.read_parquet(hf_hub_download(s.hf_repo_id, "data/snapshot/bilingual_zh_vi.parquet",
                                     repo_type=s.hf_repo_type, token=s.hf_token)).reset_index(drop=True)
df.to_parquet(r"C:\Users\ADMIN\AppData\Local\Temp\claude\d--nam4-ky1-AI-4-IOT-D4\dee1c3c9-eb48-4d7f-8a4a-9dcbb11cc25f\scratchpad\bi.parquet")

ONSET = r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
RHYME = r"(?:a|ai|ao|au|ay|an|am|ang|anh|e|eo|en|em|eng|i|ia|iu|in|im|inh|o|oi|on|om|ong|oong|oa|oai|oay|oan|oang|oanh|oe|oeo|oen|u|ua|ui|uy|un|um|ung|uya|uyu|uynh|y)"
VN_ASCII = re.compile(rf"^{ONSET}{RHYME}$")
TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[.\-+'][0-9A-Za-zÀ-ỹĐđ]+)*")

def is_foreign(tok):
    letters = re.sub(r"[^A-Za-zÀ-ỹĐđ]", "", tok)
    if not letters:
        return False
    if not letters.isascii():
        return False                      # has Vietnamese diacritics
    if re.search(r"\d", tok):
        return True                       # mixed code/unit, classified later
    return not VN_ASCII.match(tok.lower())

cnt, examples = Counter(), {}
for i, t in enumerate(df["title_vi"]):
    seen = set()
    for tok in TOKEN.findall(t or ""):
        if is_foreign(tok):
            k = tok.lower()
            if k not in seen:
                seen.add(k); cnt[k] += 1
                examples.setdefault(k, i)
print("distinct foreign tokens:", len(cnt), "| rows with any:", sum(1 for t in df["title_vi"] if any(is_foreign(x) for x in TOKEN.findall(t or ""))))
pure = [(k, c) for k, c in cnt.most_common() if not re.search(r"\d", k)]
print("pure-letter distinct:", len(pure))
for k, c in pure[:300]:
    i = examples[k]
    inzh = k in (df.at[i, "title_zh"] or "").lower()
    print(f"{c:5d} {k:18s} zh={int(inzh)} | {df.at[i,'title_vi'][:90]}")
