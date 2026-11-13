# -*- coding: utf-8 -*-
"""Cham diem OpenNMT / Moses tren DUNG cac mau eval pair cua LLM (cung id, cung tham chieu), bang cham_diem.cham (cung chrF, cung kiem so/latin/lap).

  python -X utf8 cham_co_dien.py --ten "OpenNMT" --sinh zh-vi=<sinh.jsonl> vi-zh=<sinh.jsonl> [--ra bao_cao.json]

Tham chieu lay tu data/processed/ft_run/song_ngu/<huong>/eval.{id,src,tgt,nhom} (do xuat_song_ngu.py dung ra, dung 598 + 600 id eval pair cua LLM).
Chi so: zh->vi title, mo_ta, chinh (0,10 title + 0,90 mo_ta, W_TITLE_ZHVI cua kaggle_quota), vi->zh title/mo_ta, macro (trung binh deu hai huong).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cham_diem as D  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
SONG_NGU = GOC / "data" / "processed" / "ft_run" / "song_ngu"
W_TITLE_ZHVI = 0.10
HUONG = {"zh-vi": "zh->vi", "vi-zh": "vi->zh"}


def doc_dong(p):
    L = Path(p).read_text(encoding="utf-8").split("\n")
    return L[:-1] if L and L[-1] == "" else L


def tham_chieu(hg, nhanh="pair"):
    if nhanh == "cum":      # eval ban cum: moi mau = ca title hoac ca mo ta (cac manh noi bang "; ")
        it = [json.loads(l) for l in (SONG_NGU / hg / "evalcum_items.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
        return pd.DataFrame({"id": [x["id"] for x in it], "huong": HUONG[hg], "loai": [x["loai"] for x in it],
                             "nguon": [x["nguon"] for x in it], "dich": [x["dich"] for x in it], "category": "", "nhanh": "cum"})
    ids, src, tgt, nh = (doc_dong(SONG_NGU / hg / f"eval.{x}") for x in ("id", "src", "tgt", "nhom"))
    assert len(ids) == len(src) == len(tgt) == len(nh), (hg, len(ids), len(src), len(tgt), len(nh))
    return pd.DataFrame({"id": ids, "huong": HUONG[hg], "loai": ["title" if g.endswith("title") else "mo_ta" for g in nh],
                         "nguon": src, "dich": tgt, "category": "", "nhanh": "pair"})


def cham(sinh_theo_huong, nhanh="pair"):
    """sinh_theo_huong: {'zh-vi': duong_dan, 'vi-zh': duong_dan}. Tra (tong_hop_dict, bang_tom_tat_dict)."""
    refs, hyps = [], []
    for hg, p in sinh_theo_huong.items():
        refs.append(tham_chieu(hg, nhanh))
        for l in Path(p).read_text(encoding="utf-8").split("\n"):
            if l.strip():
                r = json.loads(l)
                hyps.append({"id": r["id"], "hyp": r["hyp"]})
    _, th = D.cham(pd.concat(refs, ignore_index=True), pd.DataFrame(hyps), nhanh=nhanh)
    nhom = th["theo_nhom"]
    g = lambda k: (nhom.get(k) or {}).get("chrf_corpus")
    t, m = g("zh->vi/title"), g("zh->vi/mo_ta")
    tom = {"zh->vi/title": t, "zh->vi/mo_ta": m, "chinh_w0.10": None if t is None or m is None else W_TITLE_ZHVI * t + (1 - W_TITLE_ZHVI) * m,
           "vi->zh/title": g("vi->zh/title"), "vi->zh/mo_ta": g("vi->zh/mo_ta"), "macro": th["diem_chinh"]["chrf_macro_huong"],
           "ti_le_sinh_lap": {k: (v or {}).get("ti_le_sinh_lap") for k, v in nhom.items()},
           "ti_le_giu_so_dat": {k: (v or {}).get("ti_le_giu_so_dat") for k, v in nhom.items()},
           "thieu_hyp": {k: (v or {}).get("thieu_hyp") for k, v in nhom.items()}}
    return th, tom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", required=True)
    ap.add_argument("--sinh", nargs="+", required=True, help="huong=duong_dan, vi du zh-vi=.../sinh.jsonl")
    ap.add_argument("--ra", default=None)
    a = ap.parse_args()
    sinh = dict(x.split("=", 1) for x in a.sinh)
    th, tom = cham(sinh)
    print(a.ten, json.dumps(tom, ensure_ascii=False, indent=1))
    if a.ra:
        Path(a.ra).write_text(json.dumps({"ten": a.ten, "tom_tat": tom, "tong_hop": D._sach(th)}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
