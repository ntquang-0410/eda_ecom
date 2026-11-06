# -*- coding: utf-8 -*-
"""Vá README và SHA256SUMS của 4 thư mục description cho cột `desc_nguon_dich` (Prompt 523/524); chạy SAU them_cot_desc_nguon_dich.py.

Đọc README gốc ở data/release/description_*/ (bản đang khớp HF), thay đúng chuỗi (mỗi chuỗi phải xuất hiện đúng 1 lần, nếu không thì dừng),
ghi README.md và SHA256SUMS.txt mới vào data/release/desc_nguon_dich_20261028/<đường dẫn trên HF>/. Không sửa data/release/description_*.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "data" / "release"
OUT = REL / "desc_nguon_dich_20261028"
NGAY = "28 Oct 2026"

# thư mục -> (nguồn, loại, thư mục release gốc, đường dẫn HF, file sản phẩm, file cặp, chiều)
BO = {
    "1688_cum": dict(nguon="1688", loai="cum", goc=REL / "description_zh_to_vi_cum", hf="data/1688/descriptions/zh_to_vi_context", sp="description_products_zh_vi.parquet",
                     cap="description_pairs_zh_vi_ctx.parquet", ban_nhap="`data/1688/descriptions/zh_to_vi/`"),
    "tiki_cum": dict(nguon="tiki", loai="cum", goc=REL / "description_vi_to_zh_cum", hf="data/tiki/descriptions/vi_to_zh_context", sp="description_products_vi_zh.parquet",
                     cap="description_pairs_vi_zh_ctx.parquet", ban_nhap="`data/tiki/descriptions/vi_to_zh/`"),
    "1688_pair": dict(nguon="1688", loai="pair", goc=REL / "description_zh_to_vi", hf="data/1688/descriptions/zh_to_vi", sp="description_products_zh_vi_pair.parquet",
                      cap="description_pairs_zh_vi.parquet", ngon=("zh", "vi")),
    "tiki_pair": dict(nguon="tiki", loai="pair", goc=REL / "description_vi_to_zh", hf="data/tiki/descriptions/vi_to_zh", sp="description_products_vi_zh_pair.parquet",
                      cap="description_pairs_vi_zh.parquet", ngon=("vi", "zh")),
}
NHAN = ["gemma4", "gemma4+qwen3_14b_awq", "qwen3_14b_awq", "tu_dien_quy_tac"]
Y_NGHIA = {"gemma4": "at least one pair was translated or corrected by Gemma, none by Qwen",
           "qwen3_14b_awq": "at least one pair was translated or corrected by Qwen, none by Gemma",
           "gemma4+qwen3_14b_awq": "both models worked on at least one pair",
           "tu_dien_quy_tac": "no pair was produced by a model: dictionary and rules only"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def thay(txt, cu, moi):
    assert txt.count(cu) == 1, f"chuỗi phải xuất hiện đúng 1 lần ({txt.count(cu)}): {cu[:90]!r}"
    return txt.replace(cu, moi)


def bang_nhan(phan_bo, tong):
    d = ["| Value | Meaning | Products |", "|---|---|---:|"]
    for k in NHAN:
        n = phan_bo.get(k, 0)
        d.append(f"| `{k}` | {Y_NGHIA[k]} | {n:,} ({n / tong:.1%}) |")
    return "\n".join(d)


def doan_nhan_cum(c, tk, bd):
    ph = tk[c["nguon"]]["cum"]
    kt = ph["cap_ke_thua_khong_tra_duoc"]
    khong_tra = (f" {kt} inherited pairs could not be traced to the draft and count for neither model." if kt else "")
    return f"""### `desc_nguon_dich` (translation source of a product)

Added on {NGAY} as the **last column** of the products table; the other columns did not change. It names the model(s) that translated or corrected at
least one pair of the product. The reviewed dictionaries and fixed rules contribute to every product, so the column only says which model, if any,
also worked on it.

{bang_nhan(ph["phan_bo"], ph["san_pham"])}

A pair counts for **Gemma** (Gemma 4 26B-A4B) when its `nguon_ban_dich` is `gemma_sua` or `dich_moi`, or when it is `giu_ban_dich_don_vi` and the draft in
{bd} had chosen Gemma's output for it. It counts for **Qwen** (Qwen3-14B-AWQ) when its `nguon_ban_dich` is `qwen_sua`, or the draft had chosen Qwen's output
alone. A draft pair where both models produced the same text is credited to Gemma, which the pipeline prefers, so the share of Qwen is a lower bound.
The draft's per-pair choice is not published; the label was computed from the build files.{khong_tra}
"""


def doan_nhan_pair(c, tk):
    ph = tk[c["nguon"]]["pair"]
    zh, vi = c["ngon"]
    return f"""
### `{c['sp']}` (added {NGAY})

One row per product, built from the pair table of this folder. The columns are the same as the products table of the context-aware companion (`{c['hf']}_context/`), in the same order, so the two can be stacked.

| Column | Type | Meaning |
|---|---|---|
| `product_id`, `category` | string | As above. |
| `so_cap` | int64 | Number of pairs of the product in this release. |
| `do_tin_cay` | string | Lowest `do_tin_cay` among the pairs of the product. |
| `so_cap_khong_cao` | int64 | Number of pairs of the product whose label is not `cao`. |
| `description_{zh}`, `description_{vi}` | string | The attribute list as one text, one `name: value` per line (joined with `\\n`), in `pair_id` order. It is the description published in `data/{c['nguon']}/full_dataset/{c['nguon']}_full_attributes_independent_{zh}_{vi}.parquet`. |
| `so_cap_thap` | int64 | Number of pairs of the product whose label is `thap`. |
| `desc_nguon_dich` | string | Model(s) that translated at least one pair of the product: see below. |

`desc_nguon_dich` names the model(s) that produced at least one pair of the product. The reviewed dictionaries and fixed rules contribute to every product,
so it only says which model, if any, also worked on it. A pair counts for **Gemma** (Gemma 4 26B-A4B, "model A") when the unit was kept from Gemma's output
(including units where both models gave the same text, which the pipeline credits to Gemma), and for **Qwen** (Qwen3-14B-AWQ, "model B") when Qwen's output
alone was kept. The per-pair choice (`cach_chon`) is not published in the pair table; the label was computed from the build file.

{bang_nhan(ph["phan_bo"], ph["san_pham"])}
"""


def va_cum(c, tk, ten):
    txt = (c["goc"] / "README.md").read_text(encoding="utf-8")
    p_sp = OUT / c["hf"] / c["sp"]
    n_dong = tk[c["nguon"]]["cum"]["san_pham"]
    cu_sha = sha(c["goc"] / c["sp"])
    cu_dong = next(ln for ln in txt.splitlines() if ln.startswith(f"| `{c['sp']}` | {n_dong:,} | 8 |"))
    moi_dong = f"| `{c['sp']}` | {n_dong:,} | 9 | {p_sp.stat().st_size:,} | `{sha(p_sp)}` |"
    assert f"`{cu_sha}`" in cu_dong and f"{(c['goc'] / c['sp']).stat().st_size:,}" in cu_dong, "dòng README không khớp file gốc"
    txt = thay(txt, cu_dong, moi_dong)
    # hàng mới trong bảng cột của bảng sản phẩm: ngay sau dòng so_cap_thap
    dong_thap = next(ln for ln in txt.splitlines() if ln.startswith("| `so_cap_thap` | int64 |"))
    txt = thay(txt, dong_thap, dong_thap + f"\n| `desc_nguon_dich` | string | Model(s) that translated or corrected at least one pair of the product: `gemma4`, `qwen3_14b_awq`, `gemma4+qwen3_14b_awq` or `tu_dien_quy_tac`. Added on {NGAY}; see below. |")
    bd = c["ban_nhap"]
    moc = "### Examples (pairs)"
    txt = thay(txt, moc, doan_nhan_cum(c, tk, bd) + "\n" + moc)
    return txt


def va_pair(c, tk, ten):
    txt = (c["goc"] / "README.md").read_text(encoding="utf-8")
    p_sp = OUT / c["hf"] / c["sp"]
    n = tk[c["nguon"]]["pair"]["san_pham"]
    dong_cap = next(ln for ln in txt.splitlines() if ln.startswith(f"| `{c['cap']}` |"))
    dong_moi = f"| `{c['sp']}` | {n:,} | 9 | {p_sp.stat().st_size:,} | `{sha(p_sp)}` |"
    # SHA256SUMS mới (3 dòng) -> kích thước dòng 'SHA256SUMS.txt' trong README
    sums = "".join(f"{sha(f)}  {f.name}\n" for f in (c["goc"] / c["cap"], p_sp, c["goc"] / "provenance.json"))
    old_sums_row = next(ln for ln in txt.splitlines() if ln.startswith("| `SHA256SUMS.txt` |"))
    new_sums_row = f"| `SHA256SUMS.txt` | – | – | {len(sums.encode('ascii')):,} | checksums of the three files above |"
    txt = thay(txt, old_sums_row, new_sums_row)
    txt = thay(txt, dong_cap, dong_cap + "\n" + dong_moi)
    # đoạn mô tả bảng sản phẩm đặt ngay trước "### Examples"
    moc = "### Examples"
    txt = thay(txt, moc, doan_nhan_pair(c, tk).lstrip("\n") + "\n" + moc)
    ghi_chu = (f"**Update {NGAY}:** `{c['sp']}` (one row per product, with the new column `desc_nguon_dich`) was added to this folder. "
               "The pair table and `provenance.json` did not change.\n\n")
    txt = thay(txt, "## Files\n", ghi_chu + "## Files\n")
    return txt, sums


def main():
    tk = {n: json.loads((OUT / f"thong_ke_{n}.json").read_text(encoding="utf-8")) for n in ("1688", "tiki")}
    for ten, c in BO.items():
        d = OUT / c["hf"]
        if c["loai"] == "cum":
            txt = va_cum(c, tk, ten)
            sums = "".join(f"{sha(f)}  {f.name}\n" for f in (c["goc"] / c["cap"], d / c["sp"], c["goc"] / "provenance.json"))
            sizes = {"SHA256SUMS.txt": len(sums.encode("ascii"))}
            old = (c["goc"] / "SHA256SUMS.txt").stat().st_size
            assert sizes["SHA256SUMS.txt"] == old, (sizes, old)          # README ghi 285 byte cho SHA256SUMS.txt; hash đổi nhưng độ dài giữ nguyên
        else:
            txt, sums = va_pair(c, tk, ten)
        (d / "README.md").write_text(txt, encoding="utf-8", newline="\n")
        (d / "SHA256SUMS.txt").write_text(sums, encoding="ascii", newline="\n")
        print(f"OK {c['hf']}: README.md ({(d / 'README.md').stat().st_size:,} byte), SHA256SUMS.txt ({len(sums.splitlines())} dòng)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
