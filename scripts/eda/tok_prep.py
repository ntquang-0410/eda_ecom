# -*- coding: utf-8 -*-
"""Build review material for title_latin_tokens.csv:
  - pinyin tokens: the Chinese characters they transliterate (matched in title_zh via pypinyin)
  - every token: short vi context, share of occurrences glued to another English word
Writes review_<group>.tsv to data/interim/review/."""
import os, re, sys
from collections import Counter
import pandas as pd
from pypinyin import lazy_pinyin

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SP))
REVIEW = os.path.join(ROOT, "data", "interim", "review")
os.makedirs(REVIEW, exist_ok=True)
tok = pd.read_csv(os.path.join(ROOT, "data", "eda", "title_audit", "title_latin_tokens.csv"), encoding="utf-8-sig")
df = pd.read_parquet(os.path.join(ROOT, "data", "snapshot", "bilingual_zh_vi.parquet")).reset_index(drop=True)
df["title_vi"] = df["title_vi"].fillna(""); df["title_zh"] = df["title_zh"].fillna("")
pid2i = {str(p): i for i, p in enumerate(df["product_id"].astype(str))}
grp = dict(zip(tok["token"], tok["nhom"]))
ENG = {"tieng_anh_thua", "tieng_anh_can_xem"}
TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[.\-+'’][0-9A-Za-zÀ-ỹĐđ]+)*")
HAN = re.compile(r"[\u4e00-\u9fff]+")

def han_match(key, zh):
    """Chinese substring of zh whose toneless pinyin == key."""
    key = key.replace("'", "").replace("’", "")
    out = []
    for run in HAN.findall(zh):
        py = lazy_pinyin(run)
        for a in range(len(run)):
            s = ""
            for b in range(a, min(len(run), a + 6)):
                s += py[b]
                if s == key:
                    out.append(run[a:b + 1])
                if len(s) >= len(key):
                    break
    return out

recs = []
for _, r in tok.iterrows():
    k = r["token"]
    if grp[k] not in ENG | {"phien_am_pinyin"}:
        continue
    rows = [pid2i[p] for p in str(r["product_ids"]).split(";") if p in pid2i]
    han, glued, n_occ, ctx = Counter(), 0, 0, []
    for i in rows:
        vi = df.at[i, "title_vi"]
        toks = list(TOKEN.finditer(vi))
        for j, m in enumerate(toks):
            if m.group().lower() != k:
                continue
            n_occ += 1
            nb = [toks[x].group().lower() for x in (j - 1, j + 1) if 0 <= x < len(toks)]
            # glued = next to another English word AND not separated by punctuation
            for x in (j - 1, j + 1):
                if 0 <= x < len(toks) and grp.get(toks[x].group().lower()) in ENG:
                    between = vi[min(m.end(), toks[x].end()):max(m.start(), toks[x].start())]
                    if not re.search(r"[,;|()\[\]【】/]", between):
                        glued += 1; break
            if len(ctx) < 2:
                s = max(0, m.start() - 30); e = min(len(vi), m.end() + 30)
                ctx.append(vi[s:e].replace("\t", " "))
        if grp[k] == "phien_am_pinyin":
            han.update(set(han_match(k, df.at[i, "title_zh"])))
    recs.append({
        "token": k, "nhom": grp[k], "n": r["so_dong"], "var": r["bien_the"],
        "han": " ".join(f"{h}:{c}" for h, c in han.most_common(3)),
        "glued": round(glued / max(1, n_occ), 2), "tiki": r["so_sp_tiki_dung"],
        "ctx": " || ".join(ctx), "zh": df.at[rows[0], "title_zh"][:40] if rows else "",
    })
rv = pd.DataFrame(recs)
for g in ["tieng_anh_thua", "tieng_anh_can_xem", "phien_am_pinyin"]:
    sub = rv[rv["nhom"] == g].sort_values("n", ascending=False)
    sub.to_csv(os.path.join(REVIEW, f"review_{g}.tsv"), sep="\t", index=False, encoding="utf-8")
    print(g, len(sub))
