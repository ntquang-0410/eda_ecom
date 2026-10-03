# -*- coding: utf-8 -*-
"""Đo trước: bge-m3 cosine (Ollama local) có tách được cặp giá trị description DUNG/SAI không.

Dùng các cặp Claude đã gắn nhãn (chỉ đọc):
  - tầng 1: duyet/P10_tang1_v7_editted.csv + _bo_sung.csv (DUNG/SAI, bỏ KHONG_CHAC)
  - tầng 2: duyet/P10_qwen_hieu_chinh.csv (300 cặp hiếm ngẫu nhiên)
Hai cách ghép chuỗi: chỉ giá trị, và "tên: giá trị" (tên Việt lấy theo cách dịch chuẩn trong bản công bố).
Số đo ở đây chạy trên CPU/Ollama, cắt 480 ký tự: chỉ để quyết định có đáng dùng hay không (CLAUDE.md).

    python scripts/eda/desc_bge_do_truoc.py
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DUYET = ROOT / "data" / "processed" / "desc" / "duyet"
OUT = ROOT / "data" / "eda" / "description" / "bge_do_truoc"
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/embed"
SEED = 42


def embed(texts, bs=64):
    out, t0 = [], time.time()
    for i in range(0, len(texts), bs):
        body = json.dumps({"model": "bge-m3", "input": [t[:480] for t in texts[i:i + bs]]}).encode("utf-8")
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out.extend(json.loads(r.read())["embeddings"])
        if (i // bs) % 20 == 0:
            print(f"  embed {len(out)}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    e = np.asarray(out, dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def auc(sai, dung):
    """P(cặp SAI có cosine thấp hơn cặp DUNG). 0,5 = đoán mò, 1 = tách hoàn hảo."""
    v = np.concatenate([sai, dung])
    r = pd.Series(v).rank().to_numpy()
    n1, n0 = len(sai), len(dung)
    return 1 - (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bang_nguong(d, col):
    rows = []
    for t in np.round(np.arange(0.30, 0.91, 0.05), 2):
        co = d[col] < t                      # gắn cờ SAI khi cosine < t
        tp = int((co & (d.nhan == "SAI")).sum())
        giu = d[~co]
        rows.append({"nguong": t, "co": int(co.sum()), "bat_dung_SAI": tp,
                     "precision": round(tp / max(co.sum(), 1), 3),
                     "recall": round(tp / max((d.nhan == "SAI").sum(), 1), 3),
                     "giu": len(giu), "ti_le_sai_trong_phan_giu": round((giu.nhan == "SAI").mean(), 3) if len(giu) else None})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = pd.concat([pd.read_csv(DUYET / "P10_tang1_v7_editted.csv", dtype=str),
                    pd.read_csv(DUYET / "P10_tang1_v7_editted_bo_sung.csv", dtype=str)]).fillna("")
    t1 = t1[t1.claude_nhan.isin(["DUNG", "SAI"])].assign(tap="tang1", nhan=lambda d: d.claude_nhan)
    t2 = pd.read_csv(DUYET / "P10_qwen_hieu_chinh.csv", dtype=str).fillna("")
    t2 = t2[t2.claude_nhan.isin(["DUNG", "SAI"])].assign(tap="tang2", nhan=lambda d: d.claude_nhan)
    D = pd.concat([t1, t2], ignore_index=True)[["tap", "ten_thuoc_tinh", "gia_tri_zh", "gia_tri_vi", "nhan"]]

    # tên tiếng Việt chuẩn theo bản công bố (cách dịch nhiều nhất của mỗi name_zh)
    cap = pd.read_parquet(ROOT / "data" / "release" / "desc_v1" / "bilingual_zh_vi_desc_pairs(nhatanh).parquet",
                          columns=["name_zh", "name_vi"])
    ten_vi = cap.groupby("name_zh").name_vi.agg(lambda s: s.value_counts().index[0]).to_dict()
    D["ten_vi"] = D.ten_thuoc_tinh.map(ten_vi).fillna(D.ten_thuoc_tinh)
    D["zh_ten"] = D.ten_thuoc_tinh + ": " + D.gia_tri_zh
    D["vi_ten"] = D.ten_vi + ": " + D.gia_tri_vi
    print(D.groupby(["tap", "nhan"]).size().to_string(), flush=True)

    texts = pd.unique(pd.concat([D.gia_tri_zh, D.gia_tri_vi, D.zh_ten, D.vi_ten])).tolist()
    print(f"embed {len(texts)} chuỗi khác nhau", flush=True)
    E = dict(zip(texts, embed(texts)))
    cos = lambda a, b: np.array([float(E[x] @ E[y]) for x, y in zip(a, b)])
    D["cos_gia_tri"] = cos(D.gia_tri_zh, D.gia_tri_vi)
    D["cos_ten_gia_tri"] = cos(D.zh_ten, D.vi_ten)
    # mốc so sánh: ghép bản Việt của một cặp KHÁC cùng thuộc tính (chắc chắn sai)
    rng = np.random.default_rng(SEED)
    lech = D.groupby("ten_thuoc_tinh").gia_tri_vi.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    D["cos_lech_cung_ten"] = cos(D.gia_tri_zh, lech)

    tom = []
    for tap in ["tang1", "tang2", "tat_ca"]:
        d = D if tap == "tat_ca" else D[D.tap == tap]
        for col in ["cos_gia_tri", "cos_ten_gia_tri"]:
            s, g = d[d.nhan == "SAI"][col].to_numpy(), d[d.nhan == "DUNG"][col].to_numpy()
            tom.append({"tap": tap, "cach": col, "n_dung": len(g), "n_sai": len(s),
                        "tb_dung": round(g.mean(), 3), "tb_sai": round(s.mean(), 3), "auc": round(auc(s, g), 3)})
    tom = pd.DataFrame(tom)
    print("\nAUC (0,5 = đoán mò):\n" + tom.to_string(index=False))
    print(f"\nMốc cặp lệch cùng thuộc tính: trung vị cos {D.cos_lech_cung_ten.median():.3f}, p95 {D.cos_lech_cung_ten.quantile(.95):.3f}")

    for tap in ["tang1", "tang2"]:
        d = D[D.tap == tap]
        best = max(["cos_gia_tri", "cos_ten_gia_tri"], key=lambda c: tom[(tom.tap == tap) & (tom.cach == c)].auc.iloc[0])
        b = bang_nguong(d, best)
        print(f"\nBảng ngưỡng {tap} ({best}):\n" + b.to_string(index=False))
        b.to_csv(OUT / f"bang_nguong_{tap}.csv", index=False, encoding="utf-8-sig")

    D.to_csv(OUT / "cap_da_gan_nhan_cosine.csv", index=False, encoding="utf-8-sig")
    tom.to_csv(OUT / "tom_tat_auc.csv", index=False, encoding="utf-8-sig")
    # ví dụ: SAI mà cosine cao (bge-m3 bỏ lọt) và DUNG mà cosine thấp (cờ nhầm)
    for tap in ["tang1", "tang2"]:
        d = D[D.tap == tap]
        print(f"\n[{tap}] SAI nhưng cosine cao nhất:")
        print(d[d.nhan == "SAI"].nlargest(8, "cos_gia_tri")[["gia_tri_zh", "gia_tri_vi", "cos_gia_tri"]].to_string(index=False))
        print(f"[{tap}] DUNG nhưng cosine thấp nhất:")
        print(d[d.nhan == "DUNG"].nsmallest(8, "cos_gia_tri")[["gia_tri_zh", "gia_tri_vi", "cos_gia_tri"]].to_string(index=False))


if __name__ == "__main__":
    main()
