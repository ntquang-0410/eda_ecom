# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT / "data" / "snapshot" / "bilingual_zh_vi.parquet").reset_index(drop=True)

def split_desc(text):
    segs = [x for x in str(text or "").split("; ") if x.strip()]
    out = []
    for sgm in segs:
        name, sep, val = sgm.partition(": ")
        out.append((name.strip(), val.strip()) if sep else ("", sgm.strip()))
    return out

zh = df["description_zh"].map(split_desc)
vi = df["description_vi"].map(split_desc)
nz, nv = zh.map(len), vi.map(len)
same = nz == nv
print("rows:", len(df))
print("rows where #segments zh == vi:", int(same.sum()), f"({100*same.mean():.2f}%)")
if "n_attributes_aligned" in df.columns:
    eq_aligned = same & (nz == df["n_attributes_aligned"])
    print("... and == n_attributes_aligned:", int(eq_aligned.sum()), f"({100*eq_aligned.mean():.2f}%)")
bad = df[~same]
print("mismatch examples:")
for _, r in bad.head(3).iterrows():
    print("  zh segs", len(split_desc(r["description_zh"])), "vi segs", len(split_desc(r["description_vi"])))
    print("   ZH:", r["description_zh"][:220])
    print("   VI:", r["description_vi"][:220])

rows = []
for i in df.index[same]:
    for (nzh, vzh), (nvi, vvi) in zip(zh[i], vi[i]):
        rows.append((i, nzh, vzh, nvi, vvi))
pairs = pd.DataFrame(rows, columns=["row", "name_zh", "val_zh", "name_vi", "val_vi"])
print("\nattribute pairs:", len(pairs))
print("empty name (no ': '):", int((pairs["name_zh"] == "").sum()), int((pairs["name_vi"] == "").sum()))
print("unique name_zh:", pairs["name_zh"].nunique(), "| unique name_vi:", pairs["name_vi"].nunique(),
      "| unique (name_zh,name_vi):", len(pairs[["name_zh", "name_vi"]].drop_duplicates()))
print("unique val_zh:", pairs["val_zh"].nunique(), "| unique val_vi:", pairs["val_vi"].nunique(),
      "| unique (val_zh,val_vi):", len(pairs[["val_zh", "val_vi"]].drop_duplicates()))
num = pairs["val_vi"].str.contains(r"(?:Không|Có|Là|Khác)\.?\d+\s*$", regex=True)
print("value pairs with numbering artifact (vi ends with Không/Có/Là/Khác+digit):", int(num.sum()), f"({100*num.mean():.2f}%)")
cjk = pairs["val_vi"].str.contains("[\u4e00-\u9fff]") | pairs["name_vi"].str.contains("[\u4e00-\u9fff]")
print("attribute pairs with Chinese chars left in vi:", int(cjk.sum()), f"({100*cjk.mean():.2f}%)")
print("top vi names containing CJK:", pairs.loc[pairs['name_vi'].str.contains('[\u4e00-\u9fff]'), 'name_vi'].value_counts().head(5).to_dict())
print("sample vi values with CJK:", pairs.loc[pairs['val_vi'].str.contains('[\u4e00-\u9fff]'), ['name_zh','val_zh','val_vi']].head(5).to_dict('records'))
OUT = ROOT / "data" / "interim"
OUT.mkdir(parents=True, exist_ok=True)
pairs.to_parquet(OUT / "attr_pairs.parquet")

