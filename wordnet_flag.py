# -*- coding: utf-8 -*-
"""
So khớp từ tiếng Anh trong danh sách sàng lọc với cây phân cấp
"Physical Object" của Princeton WordNet, kiểm tra từ tiếng Trung
tương ứng chính xác (qua Open Multilingual WordNet - OMW, mã cmn),
rồi gắn cờ.

Đầu vào : english_words_filtered.csv
Đầu ra : english_words_wordnet_flagged.csv
"""
import csv
import os
from collections import deque

from nltk.corpus import wordnet as wn

def get_csv_path(filename: str) -> str:
    csv_path = os.path.join("csv", filename)
    if os.path.exists(csv_path):
        return csv_path
    return filename

# ---------- 1. Xây cây con "Physical Object" (object.n.01) ----------
ROOT = wn.synset("object.n.01")
subtree = {ROOT}
queue = deque([ROOT])
while queue:
    s = queue.popleft()
    for h in s.hyponyms():
        if h not in subtree:
            subtree.add(h)
            queue.append(h)
print("Kích thước cây con Physical Object:", len(subtree))


def zh_lemmas(syn):
    """Trả về danh sách từ tiếng Trung (cmn) của một synset."""
    try:
        return syn.lemma_names("cmn")
    except Exception:
        return []


# ---------- 2. Đọc danh sách từ ----------
rows = []
with open(get_csv_path("english_words_filtered.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# ---------- 3. So khớp & gắn cờ ----------
results = []
for r in rows:
    word = r["word"]
    key = word.lower()

    syns = wn.synsets(key, pos=wn.NOUN)
    phys = [s for s in syns if s in subtree]

    zh = []
    for s in phys:
        for lemma in zh_lemmas(s):
            if lemma not in zh:
                zh.append(lemma)

    in_phys = len(phys) > 0
    has_zh = len(zh) > 0

    if in_phys and has_zh:
        flag = "physical_object_zh_match"
    elif in_phys:
        flag = "physical_object_no_zh"
    else:
        flag = "not_physical_object"

    results.append({
        "word": word,
        "count": r.get("count", ""),
        "columns": r.get("columns", ""),
        "flag": flag,
        "zh_translations": "|".join(zh),
        "synsets": "|".join(s.name() for s in phys),
    })

# ---------- 4. Ghi kết quả ----------
fieldnames = ["word", "count", "columns", "flag", "zh_translations", "synsets"]
out_path = os.path.join("csv", "english_words_wordnet_flagged.csv") if os.path.exists("csv") else "english_words_wordnet_flagged.csv"
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

# ---------- 5. Thống kê ----------
n_phys = sum(1 for r in results if r["flag"] != "not_physical_object")
n_zh = sum(1 for r in results if r["flag"] == "physical_object_zh_match")
n_no_zh = sum(1 for r in results if r["flag"] == "physical_object_no_zh")
print("Tổng từ:", len(results))
print("  - Physical object + có từ tiếng Trung:", n_zh)
print("  - Physical object nhưng chưa có tiếng Trung:", n_no_zh)
print("  - Không phải physical object:", len(results) - n_phys)
print("Đã ghi english_words_wordnet_flagged.csv")
