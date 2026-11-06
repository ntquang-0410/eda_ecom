# -*- coding: utf-8 -*-
"""Dựng thư mục công bố bộ song ngữ 2 (Tiki Việt -> Trung, thuộc tính description, dịch bằng LLM) để đẩy lên HF: data/description_vi_to_zh/.

Chép nguyên byte file công bố 8 cột (anh bỏ cột split của nguồn, 23/10/2026), làm sạch thẻ nguồn gốc (bỏ tên secret theo đuôi khoá, câu gọi
thử, chia tập của nguồn), ghi SHA256SUMS. README.md (thẻ dữ liệu tiếng Anh) viết tay, nằm sẵn trong thư mục đích.

    .venv\\Scripts\\python.exe scripts/release/build_description_vi_to_zh.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NHAN = "20261023_1656"     # bản 1557 cộng lọc tên shop tự xưng (shopquyetnam, PET SHOP UYTINPRO, Head Phát Tiến: bớt 92 cặp) và bỏ _x000D_ (4 cặp)
SRC = ROOT / "data" / "processed" / "song_ngu_2"
DST = ROOT / "data" / "release" / "description_vi_to_zh"
PARQUET = "description_pairs_vi_zh.parquet"
PROV = "provenance.json"
NGUON_HF = "data/cleaned/tiki/tiki_eda.parquet"   # cùng SHA256 với file nguồn đã dùng (7af2a262..., đã đối chiếu trên HF 23/10/2026)
SHA_NGUON = "7af2a2620b8ca2def3b80762cb8f99196bfe7c64baeaeaa8b99d974df570d3bd"
COT = ["pair_id", "product_id", "category", "name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh", "do_tin_cay"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lam_sach(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "theo_khoa":          # tên secret đặt theo 4 ký tự cuối khoá API: chỉ giữ số khoá
                out["so_khoa"] = len(v)
            elif k in ("goi_thu", "dau_ra_cu", "split_nguon_san_pham"):
                continue
            else:
                out[k] = lam_sach(v)
        return out
    if isinstance(x, list):
        return [lam_sach(v) for v in x]
    return x


def main():
    DST.mkdir(parents=True, exist_ok=True)
    f_cb = SRC / f"bo_song_ngu_2_cong_bo_{NHAN}.parquet"
    d = pd.read_parquet(f_cb)
    assert list(d.columns) == COT, list(d.columns)
    assert d.pair_id.is_unique and not d.isna().any().any() and len(d) == 87056 and d.product_id.nunique() == 13187
    shutil.copyfile(f_cb, DST / PARQUET)
    assert sha(DST / PARQUET) == sha(f_cb)

    the = json.loads((SRC / f"the_nguon_goc_{NHAN}.json").read_text(encoding="utf-8"))
    assert the["chuan_bi"]["sha256_file_nguon"].lower() == SHA_NGUON
    the = lam_sach(the)
    the["chuan_bi"]["file_nguon"] = NGUON_HF
    the["file"] = {"published": PARQUET, "build_tag": NHAN}
    the = {"_note": "Provenance card of the build that produced description_pairs_vi_zh.parquet. Keys are Vietnamese; "
                    "see README.md, section 'Provenance file'. API key names were removed before publishing.",
           **the}
    txt = json.dumps(the, ensure_ascii=False, indent=1)
    for cam in ("GEMINI_API_KEY_", "/kaggle/", "trinhnhatanh", "split_nguon", "goi_thu", "theo_khoa"):
        assert cam not in txt, cam
    (DST / PROV).write_text(txt + "\n", encoding="utf-8")

    tong = "".join(f"{sha(DST / f)}  {f}\n" for f in (PARQUET, PROV))
    (DST / "SHA256SUMS.txt").write_text(tong, encoding="ascii", newline="\n")
    sys.stdout.write(tong)
    print({f.name: f.stat().st_size for f in DST.iterdir()})


if __name__ == "__main__":
    main()
