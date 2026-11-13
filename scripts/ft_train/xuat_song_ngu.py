# -*- coding: utf-8 -*-
"""Xuat du lieu song ngu TUNG DONG (title + dong thuoc tinh cua ban pair) cho OpenNMT va Moses. Khong GPU, khong doc test (chi doc pool).

Ra (data/processed/ft_run/song_ngu/<huong>/, huong = zh-vi (1688) va vi-zh (Tiki)):
  train.src/.tgt/.id          toan bo train fold 1-4 (cach huan luyen co dien)
  khop4m_s<seed>.src/.tgt/.id mau theo "natural" 4M token cua LLM (doc tu chon_pair_natural_s<seed>.json, neu co; co lap lai do boc co hoan lai)
  dev.src/.tgt/.id            dung de chinh trong so Moses (MERT) va valid OpenNMT: 200 title + 800 dong thuoc tinh cua fold 0, KHONG nam trong mau eval
  eval.src/.tgt/.id/.nhom     dung 1.200 mau eval cua cac LLM (cung product_id, cung dong); nhom = 1688_title, 1688_attribute, tiki_title, tiki_attribute
Moi dong la mot cap (khong co title ngu canh: OpenNMT va Moses dich tung dong rieng le).

Chay: python xuat_song_ngu.py [--chon_dir DUONG_DAN_THU_MUC_CHON_JSON]
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\nam4_ky1\e-commerce - EDA")
INTERIM = ROOT / "data" / "interim" / "quota_phase0"
POOL = ROOT / "data" / "processed" / "ft_run" / "mau" / "pool_pair.jsonl"
OUT = ROOT / "data" / "processed" / "ft_run" / "song_ngu"
HUONG = {"1688": "zh-vi", "tiki": "vi-zh"}


def doc_pool():
    by = {}
    with open(POOL, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            by[(str(j["product_id"]), j["loai"])] = j
    return by


def dong_pair(pool_by, pid, loai_ref, dong, tach):
    """Tra (nguon, dich) cua mot dong; None neu khong can duoc (giong kaggle_quota.build_mau)."""
    if loai_ref == "title":
        r = pool_by.get((pid, "title"))
        return (r["nguon"], r["dich"]) if r else None
    r = pool_by.get((pid, "mo_ta"))
    if r is None:
        return None
    if pid not in tach:
        tach[pid] = (str(r["nguon"]).split("\n"), str(r["dich"]).split("\n"))
    ls, ld = tach[pid]
    dong = int(dong)
    if dong < 0 or dong >= len(ls) or dong >= len(ld):
        return None
    return ls[dong], ld[dong]


def hang(pool_by, ref, j, tach):
    r = ref.iloc[int(j)]
    pid, nhom, loai_ref = str(r["product_id"]), r["group"], r["loai"]
    kq = dong_pair(pool_by, pid, loai_ref, r["dong"], tach)
    if kq is None:
        return None
    return {"id": f"{pid}_{loai_ref}_{r['dong']}_{int(j)}", "huong": HUONG[nhom.split('_')[0]], "nhom": nhom,
            "src": kq[0].replace("\t", " ").replace("\n", " ").strip(), "tgt": kq[1].replace("\t", " ").replace("\n", " ").strip()}


def ghi(thu_muc, ten, rows, co_nhom=False):
    thu_muc.mkdir(parents=True, exist_ok=True)
    (thu_muc / f"{ten}.src").write_text("\n".join(r["src"] for r in rows) + "\n", encoding="utf-8")
    (thu_muc / f"{ten}.tgt").write_text("\n".join(r["tgt"] for r in rows) + "\n", encoding="utf-8")
    (thu_muc / f"{ten}.id").write_text("\n".join(r["id"] for r in rows) + "\n", encoding="utf-8")
    if co_nhom:
        (thu_muc / f"{ten}.nhom").write_text("\n".join(r["nhom"] for r in rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chon_dir", default=None, help="thu muc chua chon_pair_natural_s<seed>.json (dau ra notebook dry)")
    a = ap.parse_args()
    pool_by = doc_pool()
    tach = {}
    ref = pd.read_csv(INTERIM / "train_ref_pair_flt0.csv", encoding="utf-8-sig")
    ev = pd.read_csv(INTERIM / "eval_pair_sample.csv", encoding="utf-8-sig")
    mf = {"nguon": "pool_pair.jsonl (chi pool, khong test)", "tap": {}}

    # 1) train day du
    tr = {"zh-vi": [], "vi-zh": []}
    bo = 0
    for j in range(len(ref)):
        h = hang(pool_by, ref, j, tach)
        if h is None:
            bo += 1
            continue
        tr[h["huong"]].append(h)
    for hg, rows in tr.items():
        ghi(OUT / hg, "train", rows)
        mf["tap"][f"{hg}/train"] = len(rows)
    mf["train_bo"] = bo

    # 2) khop ngan sach 4M token (neu da co danh sach chon)
    if a.chon_dir:
        for seed in (42, 123):
            p = Path(a.chon_dir) / f"chon_pair_natural_s{seed}.json"
            if not p.exists():
                continue
            ids = json.loads(p.read_text(encoding="utf-8"))
            # id = pid_loai_dong_j : j la chi so dong cua train_ref
            rows = {"zh-vi": [], "vi-zh": []}
            for s in ids:
                j = int(s.rsplit("_", 1)[1])
                h = hang(pool_by, ref, j, tach)
                if h is not None:
                    rows[h["huong"]].append(h)
            for hg, rr in rows.items():
                ghi(OUT / hg, f"khop4m_s{seed}", rr)
                mf["tap"][f"{hg}/khop4m_s{seed}"] = len(rr)

    # 3) eval: dung 1.200 mau cua LLM
    evr = {"zh-vi": [], "vi-zh": []}
    for j in range(len(ev)):
        h = hang(pool_by, ev, j, tach)
        if h is not None:
            evr[h["huong"]].append(h)
    for hg, rows in evr.items():
        ghi(OUT / hg, "eval", rows, co_nhom=True)
        mf["tap"][f"{hg}/eval"] = len(rows)
    sp_eval = {str(x) for x in ev["product_id"]}

    # 4) dev: fold 0, khong thuoc mau eval, 200 title + 800 dong thuoc tinh moi huong
    rng = random.Random(20261112)
    for hg, nguon in (("zh-vi", "1688"), ("vi-zh", "tiki")):
        title, attr = [], []
        for (pid, loai), r in pool_by.items():
            if int(r["fold5"]) != 0 or pid in sp_eval or r["source"] != nguon:
                continue
            if loai == "title":
                title.append((pid, r))
            else:
                ls, ld = str(r["nguon"]).split("\n"), str(r["dich"]).split("\n")
                for d in range(min(len(ls), len(ld))):
                    attr.append((pid, d, ls[d], ld[d]))
        rng.shuffle(title)
        rng.shuffle(attr)
        rows = [{"id": f"{pid}_title_-1_dev", "nhom": f"{nguon}_title", "src": r["nguon"].replace("\t", " ").replace("\n", " ").strip(),
                 "tgt": r["dich"].replace("\t", " ").replace("\n", " ").strip()} for pid, r in title[:200]]
        rows += [{"id": f"{pid}_attribute_{d}_dev", "nhom": f"{nguon}_attribute", "src": s.replace("\t", " ").strip(), "tgt": t.replace("\t", " ").strip()}
                 for pid, d, s, t in attr[:800]]
        rng.shuffle(rows)
        ghi(OUT / hg, "dev", rows)
        mf["tap"][f"{hg}/dev"] = len(rows)

    # 5) kiem: eval va dev khong trung san pham voi train (train la fold 1-4, dev/eval la fold 0)
    sp_train = {str(x) for x in ref["product_id"]}
    assert not (sp_eval & sp_train), "eval trung san pham voi train"
    mf["sha256"] = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob("*.src"))}
    (OUT / "MANIFEST.json").write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(mf["tap"], indent=1), "bo:", bo)


if __name__ == "__main__":
    main()
