# -*- coding: utf-8 -*-
"""title_latin_tokens.csv + hand decisions -> title_latin_tokens_editted.csv
Only the token table is written; title_vi in the dataset is NOT modified."""
import os, re, sys
from collections import Counter
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import dec_pinyin, dec_en

ROOT = os.path.dirname(os.path.dirname(SP))
AUD = os.path.join(ROOT, "data", "eda", "title_audit")
tok = pd.read_csv(os.path.join(AUD, "title_latin_tokens.csv"), encoding="utf-8-sig")
df = pd.read_parquet(os.path.join(ROOT, "data", "snapshot", "bilingual_zh_vi.parquet")).reset_index(drop=True)
df["title_vi"] = df["title_vi"].fillna(""); df["title_zh"] = df["title_zh"].fillna("")

def norm(v):
    return v if len(v) == 4 else (v[0], v[1], "", v[2])

MERGED = {k: norm(v) for k, v in dec_pinyin.DEC.items()}
MERGED.update({k: norm(v) for k, v in dec_en.DEC.items()})
MERGED.update({k: norm(v) for k, v in dec_en.OVERRIDE.items()})

known = set(tok["token"])
missing = sorted(k for k in MERGED if k not in known)
if missing:
    print("WARNING decisions for tokens not in the file:", missing)

DEFAULT = {
    "phien_am_pinyin": "Tên hãng viết pinyin -> giữ (theo chính sách anh chọn)",
    "tieng_anh_can_xem": "Tên hãng / model / tên sản phẩm -> giữ",
    "tieng_anh_thua": "Giữ",
    "tu_muon_pho_bien": "Từ mượn phổ biến ở VN -> giữ (anh đã chốt)",
    "ten_rieng": "Tên riêng / thương hiệu -> giữ",
    "thuat_ngu_ky_thuat": "Thuật ngữ kỹ thuật -> giữ",
    "co_trong_ban_goc": "Bản gốc tiếng Trung cũng viết Latin -> giữ",
    "ma_model": "Mã / model -> giữ",
    "ky_hieu": "Ký hiệu có trong bản gốc -> giữ",
    "don_vi_so": "Số + đơn vị -> giữ",
}

def decide(r):
    k = r["token"]
    if k in MERGED:
        return MERGED[k]
    note = dec_pinyin.NOTE_GIU.get(k) or DEFAULT[r["nhom"]]
    return ("GIU", "", "", note)

W = r"0-9A-Za-zÀ-ỹĐđ"
def pat(s):
    return re.compile(rf"(?<![{W}]){re.escape(s)}(?![{W}])", re.I)

def keep_case(m, rep):
    # capitalise a lower-case replacement only where a sentence starts (title start, after [ 【 ( . ! ?)
    before = m.string[:m.start()].rstrip()
    starts = before == "" or before[-1] in "[【(.!?:|"
    if rep and starts and m.group()[:1].isupper() and rep[:1].islower():
        return rep[0].upper() + rep[1:]
    return rep

def rules_of(s):
    out = []
    for part in [p for p in s.split(" ; ") if p.strip()]:
        a, _, b = part.partition(" => ")
        out.append((a.strip(), b.strip()))
    return out

def apply(title, zh, k, action, rep, rules):
    s, done = title, []
    for a, b in rules:
        def hold(m, b=b):
            done.append(keep_case(m, b))           # protect phrase results from the single-token pass
            return f"\x00{len(done) - 1}\x00"
        s = pat(a).sub(hold, s)
    s = _single(s, zh, k, action, rep)
    return re.sub("\x00(\\d+)\x00", lambda m: done[int(m.group(1))], s)

ONSET = r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
RHYME = (r"(?:a|ai|ao|au|ay|an|am|ang|anh|e|eo|en|em|eng|i|ia|iu|in|im|inh|o|oi|on|om|ong|oong|"
         r"oa|oai|oay|oan|oang|oanh|oe|oeo|oen|u|ua|ui|uy|un|um|ung|uya|uyu|uynh|y)")
VN_ASCII = re.compile(rf"^{ONSET}{RHYME}$")

def is_latin_word(w):
    return bool(w) and w.isascii() and w.isalpha() and not VN_ASCII.match(w.lower())

def _single(s, zh, k, action, rep):
    if action == "THEO_CUM" and rep:
        # only where the token stands among Vietnamese words; next to another Latin word it is
        # usually part of a name (Baigou New City, Black King Kong) -> leave it for manual review
        def sub(m):
            prev = re.findall(rf"[{W}]+", s[:m.start()])[-1:] or [""]
            nxt = re.findall(rf"[{W}]+", s[m.end():])[:1] or [""]
            if is_latin_word(prev[0]) and not re.search(r"[,;|()\[\]【】/]\s*$", s[:m.start()]):
                return m.group()
            if is_latin_word(nxt[0]) and not re.match(r"\s*[,;|()\[\]【】/]", s[m.end():]):
                return m.group()
            return keep_case(m, rep)
        return pat(k).sub(sub, s)
    return _single_plain(s, zh, k, action, rep)

def _single_plain(s, zh, k, action, rep):
    if action in ("THAY", "HANVIET", "THEO_CUM") and rep:
        if " | " in rep and "→" in rep:          # per-Chinese-form mapping (yiwu)
            choice = ""
            for part in rep.split(" | "):
                han, _, vn = part.partition("→")
                if han in zh:
                    choice = vn
            if not choice:
                return s
            rep = choice
        s = pat(k).sub(lambda m: keep_case(m, rep), s)
    return s

# full row membership (the csv only lists up to 50 product ids per token)
TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[.\-+'’][0-9A-Za-zÀ-ỹĐđ]+)*")
rows_of = {}
for i, t in enumerate(df["title_vi"]):
    for w in {x.lower() for x in TOKEN.findall(t)}:
        rows_of.setdefault(w, []).append(i)

recs = []
for _, r in tok.iterrows():
    k = r["token"]
    action, rep, rules, note = decide(r)
    rl = rules_of(rules)
    idx = [i for i in rows_of.get(k, [])]
    ex = idx[0] if idx else None
    # prefer an example where the decision actually changes something
    if action != "GIU":
        for i in idx[:400]:
            if apply(df.at[i, "title_vi"], df.at[i, "title_zh"], k, action, rep, rl) != df.at[i, "title_vi"]:
                ex = i
                break
    vi = df.at[ex, "title_vi"] if ex is not None else r["vi_du_title_vi"]
    zh = df.at[ex, "title_zh"] if ex is not None else r["vi_du_title_zh"]
    after = apply(vi, zh, k, action, rep, rl) if action != "GIU" else ""
    recs.append({
        "token": k, "nhom": r["nhom"], "hanh_dong": action, "thay_bang": rep, "quy_tac_cum": rules,
        "ghi_chu": note, "so_dong": r["so_dong"], "bien_the": r["bien_the"],
        "so_dong_zh_cung_co": r["so_dong_zh_cung_co"], "so_sp_tiki_dung": r["so_sp_tiki_dung"],
        "category_chinh": r["category_chinh"],
        "vi_du_title_vi": vi, "vi_du_sau_sua": after if after != vi else ("(ví dụ này không đổi)" if action in ("THAY", "HANVIET", "THEO_CUM") else ""),
        "vi_du_title_zh": zh,
        "anh_duyet": "", "anh_ghi_chu": "",
        "product_ids": r["product_ids"],
    })
out = pd.DataFrame(recs)
ORDER = {"THAY": 0, "HANVIET": 1, "THEO_CUM": 2, "XEM_TAY": 3, "GIU": 4}
out["_o"] = out["hanh_dong"].map(ORDER)
out = out.sort_values(["_o", "so_dong", "token"], ascending=[True, False, True]).drop(columns="_o")
path = os.path.join(AUD, "title_latin_tokens_editted.csv")
out.to_csv(path, index=False, encoding="utf-8-sig")

print("written:", path, len(out), "tokens")
agg = out.groupby("hanh_dong").agg(so_token=("token", "size"), luot_dong=("so_dong", "sum")).reindex(list(ORDER))
print(agg.to_string())
for a in ["THAY", "HANVIET", "THEO_CUM", "XEM_TAY"]:
    ks = out.loc[out["hanh_dong"] == a, "token"]
    rows = set().union(*[set(rows_of.get(k, [])) for k in ks]) if len(ks) else set()
    print(f"rows touched by {a}: {len(rows)}")
act = out.loc[out["hanh_dong"] != "GIU", "token"]
print("rows with >=1 token to act on:", len(set().union(*[set(rows_of.get(k, [])) for k in act])))
print("changed by group:\n", pd.crosstab(out["nhom"], out["hanh_dong"]).to_string())
print("\nexamples with no visible change:", int((out["vi_du_sau_sua"] == "(ví dụ này không đổi)").sum()))
print(out.loc[out["vi_du_sau_sua"] == "(ví dụ này không đổi)", ["token", "hanh_dong", "thay_bang", "quy_tac_cum"]].head(40).to_string())
