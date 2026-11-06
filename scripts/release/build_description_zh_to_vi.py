# -*- coding: utf-8 -*-
"""Dựng thư mục công bố bộ song ngữ 1 (description, dịch lại bằng LLM) để đẩy lên HF: data/description_zh_to_vi/.

Chép nguyên byte file công bố 9 cột, làm sạch thẻ nguồn gốc (bỏ tên secret theo đuôi khoá, đường dẫn Kaggle, khối HY-MT không
dùng, câu gọi thử), ghi SHA256SUMS. README.md (thẻ dữ liệu tiếng Anh) viết tay, nằm sẵn trong thư mục đích.

    .venv\\Scripts\\python.exe scripts/release/build_description_zh_to_vi.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NHAN = "20261022_2126"     # bản 2059 cộng luật "cân" chặt hơn (bắt 半斤 "nửa cân", "một cân", "(cân/43 tệ)"...): bớt 29 cặp
SRC = ROOT / "data" / "processed" / "song_ngu_1"
DST = ROOT / "data" / "release" / "description_zh_to_vi"
PARQUET = "description_pairs_zh_vi.parquet"
PROV = "provenance.json"
NGUON_HF = "data/cleaned/1688/bilingual_zh_vi_desc_pairs.parquet"   # cùng SHA256 với file cặp đầu vào (48f11ff3...)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lam_sach(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "theo_khoa":          # tên secret đặt theo 4 ký tự cuối khoá API: chỉ giữ số khoá
                out["so_khoa"] = len(v)
            elif k in ("goi_thu", "dau_ra_cu"):
                continue
            elif k == "hy_mt15_7b":
                out[k] = "removed: HY-MT1.5-7B was an earlier model-B candidate; none of its output is in this release"
            else:
                out[k] = lam_sach(v)
        return out
    if isinstance(x, list):
        return [lam_sach(v) for v in x]
    return x


def main():
    DST.mkdir(parents=True, exist_ok=True)
    f_cb = SRC / f"bo_song_ngu_1_cong_bo_{NHAN}.parquet"
    d = pd.read_parquet(f_cb)
    assert list(d.columns) == ["pair_id", "product_id", "category", "nhom_trung", "name_zh", "gia_tri_zh", "name_vi",
                               "gia_tri_vi", "do_tin_cay"], list(d.columns)
    assert d.pair_id.is_unique and not d.isna().any().any() and len(d) == 394662
    shutil.copyfile(f_cb, DST / PARQUET)
    assert sha(DST / PARQUET) == sha(f_cb)

    the = json.loads((SRC / f"the_nguon_goc_{NHAN}.json").read_text(encoding="utf-8"))
    assert the["chuan_bi"]["sha256_file_cap"].lower() == "48f11ff3c7661bf17791ae286001c82c8c7b8420518d6fd5cca841cf49a0d447"
    the = lam_sach(the)
    the["chuan_bi"]["file_cap"] = NGUON_HF
    the["file"] = {"published": PARQUET, "build_tag": NHAN}
    the = {"_note": "Provenance card of the build that produced description_pairs_zh_vi.parquet. Keys are Vietnamese; "
                    "see README.md, section 'Provenance file'. API key names and Kaggle paths were removed before publishing.",
           **the}
    txt = json.dumps(the, ensure_ascii=False, indent=1)
    for cam in ("GEMINI_API_KEY_", "/kaggle/", "trinhnhatanh", "hy-mt-translate"):
        assert cam not in txt, cam
    (DST / PROV).write_text(txt + "\n", encoding="utf-8")

    tong = "".join(f"{sha(DST / f)}  {f}\n" for f in (PARQUET, PROV))
    (DST / "SHA256SUMS.txt").write_text(tong, encoding="ascii", newline="\n")
    sys.stdout.write(tong)
    print({f.name: f.stat().st_size for f in DST.iterdir()})


if __name__ == "__main__":
    main()
