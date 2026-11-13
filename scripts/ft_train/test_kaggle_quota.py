# -*- coding: utf-8 -*-
"""Test bo dung mau quota tren may (khong can GPU, khong doc test)."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_quota as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BO = ROOT / "data" / "processed" / "bo_chon_hf"
INTERIM = ROOT / "data" / "interim" / "quota_phase0"


def pool_tu_parquet(tap, pids):
    f = "bo_cum_chon_deepseek.parquet" if tap == "cum" else "bo_pair_chon_deepseek.parquet"
    d = pd.read_parquet(BO / f)
    d = d[d["product_id"].astype(str).isin(set(map(str, pids)))]
    out = {}
    for _, r in d.iterrows():
        pid = str(r["product_id"])
        zh = r["source"] == "1688"
        out[(pid, "title")] = {
            "product_id": pid, "loai": "title", "huong": "zh->vi" if zh else "vi->zh",
            "category": r["category"],
            "nguon": r["title_zh"] if zh else r["title_vi"],
            "dich": r["title_vi"] if zh else r["title_zh"],
        }
        out[(pid, "mo_ta")] = {
            "product_id": pid, "loai": "mo_ta", "huong": "zh->vi" if zh else "vi->zh",
            "category": r["category"],
            "nguon": r["description_zh"] if zh else r["description_vi"],
            "dich": r["description_vi"] if zh else r["description_zh"],
        }
    return out


def test_tap(tap):
    ref = pd.read_csv(INTERIM / f"train_ref_{tap}_flt0.csv", encoding="utf-8-sig")
    idx = Q.np.load(INTERIM / f"idx_q_{tap}_balanced_cap960_flt0.npy")[:8]
    pids = ref.iloc[idx]["product_id"].astype(str).tolist()
    pool = pool_tu_parquet(tap, pids)
    mau = Q.build_mau(ref, idx, pool, tap)
    assert len(mau) == len(idx), (tap, len(mau), len(idx))
    assert all(m["fold5"] != 0 for m in mau), "lo mau dev vao train"
    for m in mau:
        assert m["messages"][0]["role"] == "user" and m["messages"][1]["role"] == "assistant"
        assert m["messages"][1]["content"] == m["dich"]
        assert m["huong"] in ("zh->vi", "vi->zh")
        assert m["loai"] in ("title", "mo_ta")
        assert m["nguon"].strip() and m["dich"].strip()
    print(f"OK {tap}: {len(mau)} mau; vi du id={mau[0]['id']} group={mau[0]['group']} loai={mau[0]['loai']}")
    print("  user:", mau[0]["messages"][0]["content"][:160].replace("\n", " | "))
    print("  assistant:", mau[0]["messages"][1]["content"][:100])


if __name__ == "__main__":
    test_tap("cum")
    test_tap("pair")
    print("TAT CA DAT")
