"""Xuất BẢN PAIR của đúng 25.575 sản phẩm còn lại sau bước làm sạch (cùng tập sản phẩm với bản cụm) để chấm judge và fine-tune
so sánh hai bộ. Không sửa nội dung; chỉ lọc theo product_id.

  python -I -X utf8 scripts/ft_clean/xuat_ban_pair.py"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parents[2]
PAIR_V2 = GOC / "data" / "snapshot" / "bilingual_official_v2" / "bilingual_official_pair.parquet"
SHA_PAIR_V2 = "5feec0cf2c2799845b7b3d7aaa4e8f337529264d3e2061009acb65f49ff66936"     # manifest.json của commit 4a23574
GIU_CUM = GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu.parquet"
RA = GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu_pair.parquet"
COT_GOC = ["product_id", "source", "category", "title_zh", "title_vi", "description_zh", "description_vi",
           "title_nguon_dich", "desc_nguon_dich"]
# Hai bản phải giống hệt ở các cột này. desc_nguon_dich KHÁC nhau giữa hai bản (nguồn dịch mô tả khác: bản pair là dịch từng cặp).
COT_CHUNG = ["source", "category", "title_zh", "title_vi", "title_nguon_dich"]


def xuat(pair_path=PAIR_V2, giu_path=GIU_CUM, ra=RA, kiem_sha=True):
    if kiem_sha:
        h = hashlib.sha256(open(pair_path, "rb").read()).hexdigest()
        assert h == SHA_PAIR_V2, f"SHA256 bản pair không khớp manifest: {h}"
    p = pd.read_parquet(pair_path)
    c = pd.read_parquet(giu_path, columns=["product_id"] + COT_CHUNG)
    assert list(p.columns) == COT_GOC and p.product_id.is_unique
    g = p[p.product_id.isin(set(c.product_id))].reset_index(drop=True)
    assert len(g) == len(c) and g.product_id.tolist() == c.product_id.tolist(), "khác tập hoặc khác thứ tự so với bản cụm"
    for col in COT_CHUNG:
        assert (g[col].fillna("") == c[col].fillna("")).all(), f"cột {col} khác giữa hai bản"
    Path(ra).parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(ra, index=False)
    return g


if __name__ == "__main__":
    g = xuat()
    print("đã ghi", RA, len(g), "sản phẩm;", g.source.value_counts().to_dict())
    sys.exit(0)
