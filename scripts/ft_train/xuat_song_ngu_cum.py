# -*- coding: utf-8 -*-
"""Them vao song_ngu/<huong>/ cac tep EVAL BAN CUM (1.200 mau eval cum cua LLM) o dang tung manh de OpenNMT/Moses dich roi ghep lai bang "; ". Chi doc pool, khong test.

  evalcum.src/.tgt/.id   moi dong = mot manh: title (1 dong) hoac mot thuoc tinh cua mo ta (tach theo "; "). id = <id mau>#<chi so manh>
  evalcum_items.jsonl    moi dong = mot mau eval cum: {id, nhom, loai, nguon, dich, n_manh}; id giong id cua LLM (pid_loai_dong_j)
Mo hinh co dien huan luyen tren dong cua ban pair (ban cum khac pair ~1% manh), danh gia tren ca eval pair va eval cum.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\nam4_ky1\e-commerce - EDA")
POOL = ROOT / "data" / "processed" / "ft_run" / "mau" / "pool_cum.jsonl"
EV = ROOT / "data" / "interim" / "quota_phase0" / "eval_cum_sample.csv"
OUT = ROOT / "data" / "processed" / "ft_run" / "song_ngu"
HUONG = {"1688": "zh-vi", "tiki": "vi-zh"}


def main():
    pool = {}
    for l in open(POOL, encoding="utf-8"):
        j = json.loads(l)
        pool[(str(j["product_id"]), j["loai"])] = j
    ev = pd.read_csv(EV, encoding="utf-8-sig")
    rows = {"zh-vi": [], "vi-zh": []}
    items = {"zh-vi": [], "vi-zh": []}
    thieu = 0
    for j, r in ev.iterrows():
        pid, loai_ref, nhom = str(r["product_id"]), r["loai"], r["group"]
        hg = HUONG[nhom.split("_")[0]]
        rec = pool.get((pid, "title" if loai_ref == "title" else "mo_ta"))
        if rec is None:
            thieu += 1
            continue
        id_ = f"{pid}_{loai_ref}_{r['dong']}_{int(j)}"
        nguon, dich = str(rec["nguon"]), str(rec["dich"])
        if loai_ref == "title":
            ps, pd_ = [nguon.strip()], [dich.strip()]
        else:
            ps = [p.strip() for p in nguon.split("; ") if p.strip()]
            pd_ = [p.strip() for p in dich.split("; ") if p.strip()]
        for k, s in enumerate(ps):
            rows[hg].append((f"{id_}#{k}", s.replace("\t", " ").replace("\n", " "), (pd_[k] if k < len(pd_) else "").replace("\t", " ").replace("\n", " ")))
        items[hg].append({"id": id_, "nhom": nhom, "loai": "title" if loai_ref == "title" else "mo_ta", "nguon": nguon, "dich": dich, "n_manh": len(ps)})
    for hg in rows:
        d = OUT / hg
        d.mkdir(parents=True, exist_ok=True)
        (d / "evalcum.src").write_text("\n".join(x[1] for x in rows[hg]) + "\n", encoding="utf-8")
        (d / "evalcum.tgt").write_text("\n".join(x[2] for x in rows[hg]) + "\n", encoding="utf-8")
        (d / "evalcum.id").write_text("\n".join(x[0] for x in rows[hg]) + "\n", encoding="utf-8")
        (d / "evalcum_items.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items[hg]) + "\n", encoding="utf-8")
        print(hg, "mau", len(items[hg]), "manh", len(rows[hg]), "title", sum(1 for x in items[hg] if x["loai"] == "title"))
    print("thieu trong pool:", thieu)


if __name__ == "__main__":
    main()
