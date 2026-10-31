"""So TẦNG 2 (có ngữ cảnh, chạy lại các mục 'không nhất quán') với TẦNG 1 (không ngữ cảnh) trên CÙNG các mục.

  python -I -X utf8 so_sanh_tang.py --loai title --tang1 title_100 --tang2 title_100_t2 [--in-mau 8]
  python -I -X utf8 so_sanh_tang.py --loai pair  --tang1 pair_100  --tang2 pair_100_t2

Chỉ đọc file kết quả (không gọi API) và ghi so_sanh_tang_<tang1>_<tang2>.json (kèm các mục để đọc). Trả lời: tầng 2 gỡ được bao nhiêu mục
'không nhất quán' (và theo hướng nào), còn bao nhiêu vẫn chưa quyết, tốn bao nhiêu token. Không có 'chân lý' ở đây."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so_cap_du_lieu as DL  # noqa: E402

KQ = ("gemma", "qwen", "hoa_that", "khong_nhat_quan")


def doc(duong_ds):
    cuoi = {}
    for duong in duong_ds:
        for dong in open(duong, encoding="utf-8"):
            d = json.loads(dong)
            if d.get("trang_thai") == "ok":
                cuoi[d["item_id"]] = d
    return cuoi


def so_sanh(t1, t2):
    """t1, t2: {item_id: bản ghi}. So các mục có ở cả hai (tầng 2 chỉ chạy các mục không nhất quán của tầng 1)."""
    chung = [i for i in t2 if i in t1]
    r = {"so_muc_tang1": len(t1), "so_muc_tang1_khong_nhat_quan": sum(d["ket_qua"] == "khong_nhat_quan" for d in t1.values()),
         "so_muc_tang2": len(t2), "so_muc_so_duoc": len(chung)}
    if not chung:
        return r
    r["ma_tran_tang1_sang_tang2"] = dict(Counter(f"{t1[i]['ket_qua']}->{t2[i]['ket_qua']}" for i in chung))
    quyet = ("gemma", "qwen")
    moi_quyet = [i for i in chung if t1[i]["ket_qua"] not in quyet and t2[i]["ket_qua"] in quyet]
    r["go_duoc_nho_tang2"] = {"so_muc": len(moi_quyet), "ti_le_trong_so_duoc_chay_lai": round(len(moi_quyet) / len(chung), 3),
                               "gemma": sum(t2[i]["ket_qua"] == "gemma" for i in moi_quyet),
                               "qwen": sum(t2[i]["ket_qua"] == "qwen" for i in moi_quyet)}
    r["thanh_hoa_that"] = sum(t2[i]["ket_qua"] == "hoa_that" for i in chung)
    r["van_khong_nhat_quan"] = sum(t2[i]["ket_qua"] == "khong_nhat_quan" for i in chung)
    r["tren_toan_bo_tang1_sau_tang2"] = {
        "co_nguoi_thang_truoc": sum(d["ket_qua"] in quyet for d in t1.values()),
        "co_nguoi_thang_sau": sum((t2.get(i, d)["ket_qua"] if i in t2 else d["ket_qua"]) in quyet for i, d in t1.items()),
        "tong": len(t1)}
    tv1 = sum(t1[i]["tokens_vao"] for i in chung) / len(chung)
    tv2 = sum(t2[i]["tokens_vao"] for i in chung) / len(chung)
    r["tokens_vao_moi_muc"] = {"tang1": round(tv1), "tang2": round(tv2), "ti_so": round(tv2 / tv1, 2) if tv1 else None}
    r["tokens_ra_moi_muc_tang2"] = round(sum(t2[i]["tokens_ra"] for i in chung) / len(chung))
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loai", choices=["title", "pair"], required=True)
    p.add_argument("--tang1", required=True, help="tên các đợt tầng 1, cách nhau bằng dấu phẩy")
    p.add_argument("--tang2", required=True)
    p.add_argument("--in-mau", type=int, default=0)
    p.add_argument("--ra", default=None)
    a = p.parse_args(argv)
    ra = Path(a.ra) if a.ra else DL.RA
    t1 = doc([ra / f"ket_qua_{t}.jsonl" for t in a.tang1.split(",") if t])
    t2 = doc([ra / f"ket_qua_{t}.jsonl" for t in a.tang2.split(",") if t])
    r = so_sanh(t1, t2)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    ca = []
    if r.get("so_muc_so_duoc"):
        muc = pd.read_parquet(DL.RA / f"muc_{a.loai}_ctx.parquet", columns=["item_id", "nguon", "cand_gemma", "cand_qwen"]).set_index("item_id")
        for i, d in t2.items():
            if i in t1 and i in muc.index:
                m = muc.loc[i]
                ca.append({"item_id": i, "tang1": t1[i]["ket_qua"], "tang2": d["ket_qua"], "nguon": m.nguon, "gemma": m.cand_gemma,
                           "qwen": m.cand_qwen, "ly_do_tang1": [l["ly_do"] for l in t1[i]["luot"]],
                           "ly_do_tang2": [l["ly_do"] for l in d["luot"]]})
    ten = f"so_sanh_tang_{a.tang1.replace(',', '+')}_{a.tang2.replace(',', '+')}.json"
    (ra / ten).write_text(json.dumps({"loai": a.loai, "so_sanh": r, "cac_muc": ca}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n(đã ghi {ra / ten}; {len(ca)} mục để đọc)")
    for c in [c for c in ca if c["tang2"] in ("gemma", "qwen")][:a.in_mau]:
        print(f"\n== {c['item_id']}: tầng 1 = {c['tang1']} | tầng 2 = {c['tang2']}")
        print("  nguồn:", c["nguon"], "\n  Gemma:", c["gemma"], "\n  Qwen: ", c["qwen"])
        for nhan, ds in (("T1", c["ly_do_tang1"]), ("T2", c["ly_do_tang2"])):
            for k, ld in enumerate(ds):
                print(f"  [{nhan} lượt {k + 1}] {ld}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
