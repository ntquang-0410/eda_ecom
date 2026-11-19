# -*- coding: utf-8 -*-
"""Ghi sha256 cac tep dau vao da doc cho P1. Chi doc tep du lieu, khong doc test.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p1_sha256.py
"""
import hashlib
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import tong_hop_5mh as T  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest()


def thu_thap():
    tep = []

    def them(p):
        p = Path(p)
        if p.is_file():
            tep.append(p)
        else:
            thieu.append(str(p))

    thieu = []
    for nhanh in ("pair", "cum"):
        them(R / "mau" / f"pool_{nhanh}.jsonl")
        for hg in ("zh-vi", "vi-zh"):
            if nhanh == "cum":
                them(R / "song_ngu" / hg / "evalcum_items.jsonl")
            else:
                for x in ("id", "src", "tgt", "nhom"):
                    them(R / "song_ngu" / hg / f"eval.{x}")
        T.NHANH = nhanh
        for m in T.cac_mo_hinh():
            for ds in m["lan_chay"].values():
                for p in ds:
                    them(p)
        them(R / ("tong_hop_5mh" if nhanh == "pair" else "tong_hop_5mh_cum") / "ket_qua.json")
        them(GOC / "data" / "processed" / "bo_chon_hf" / f"bo_{nhanh}_chon_deepseek.parquet")
    for p in (
        R / "kaggle_cheo_that" / "cheo" / "sinh_zero_shot_pair.jsonl",
        R / "kaggle_cheo_that" / "cheo" / "sinh_zero_shot_cum.jsonl",
        R / "kaggle_dong6" / "dong6" / "sinh_pair_f0_p11_pair.jsonl",
        R / "kaggle_dong6" / "dong6" / "sinh_cum_f0_p11_cum.jsonl",
        R / "kaggle_onmt_that" / "onmt_report.json",
        R / "kaggle_onmt_that" / "onmt" / "zh-vi_train" / "train.log",
        R / "kaggle_onmt_that" / "onmt" / "vi-zh_train" / "train.log",
        R / "kaggle_quota_step1" / "quota_report.json",
        R / "kaggle_quota_nlp-8b" / "quota_report.json",
        R / "kaggle_quota_nlp-ecom" / "quota_report.json",
        R / "kaggle_moses_v2_zhvi" / "moses_report.json",
        R / "kaggle_moses_v2_vizh" / "moses_report.json",
    ):
        them(p)
    return tep, thieu


def main():
    tep, thieu = thu_thap()
    ra = {}
    for p in tep:
        ra[str(p.relative_to(GOC)).replace("\\", "/")] = {"byte": p.stat().st_size, "sha256": sha256(p)}
    obj = {"so_tep": len(ra), "tep": ra, "thieu": thieu,
           "ghi_chu": "Chỉ gồm đầu vào của P1. Không có test_khoa, split.parquet, ds_test.txt."}
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "sha256_dau_vao.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("so tep:", len(ra), "thieu:", len(thieu))
    for p in thieu:
        print("  THIEU", p)


if __name__ == "__main__":
    main()
